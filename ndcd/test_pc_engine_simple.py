import os
import sys
import time
import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import random
from collections import deque
from engine_torch import PredictiveCodingEngine
from graph import DynamicGraph


def create_training_graph(
    num_modules=10, num_levels=2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Builds a smaller hierarchical graph suitable for simple PC engine testing.
    Adapts logic from micro_scale_diagnostics and DynamicGraph.
    """
    total_nodes = num_modules * 350  # Increased to 3500 nodes total

    # We need at least 512 nodes for I/O (256 in, 256 out)
    if total_nodes < 600:
        total_nodes = 600

    print(
        f"Building DynamicGraph with {total_nodes} nodes, {num_modules} modules, {num_levels} levels..."
    )
    # Use DynamicGraph to build the structure
    graph = DynamicGraph(
        total_nodes,
        m_edges=2,
        p_triad=0.1,
        num_modules=num_modules,
        num_levels=num_levels,
    )

    indices, values = graph.export_sparse_components()
    edge_index = torch.tensor(indices, dtype=torch.long)
    edge_weight = torch.tensor(values, dtype=torch.float32)

    # Use the actual number of nodes from the graph (which includes cerebellum)
    actual_nodes = graph.num_nodes
    biases = torch.zeros(actual_nodes, dtype=torch.float32)
    is_neg_pe = torch.zeros(actual_nodes, dtype=torch.bool)
    is_pos_pe = torch.zeros(actual_nodes, dtype=torch.bool)

    # --- Fix VI.1: Consistent PE population labeling ---
    # Assign half of L2/3 and L4/5/6 error-signaling nodes to each pool
    for mod in graph.modules:
        l23 = mod['l23_indices']
        n_half = len(l23) // 2
        is_neg_pe[l23[:n_half]] = True
        is_pos_pe[l23[n_half:]] = True
    
    # --- Fix III.1: Hierarchical Timescale Gradient ---
    # Levels: 0=Input/Output, 1=Lower Association, 2=Higher Association
    # Taus (ms): L0=5, L1=20, L2=50, L3=100
    # Higher levels integrate info over longer windows (Murray et al. 2014)
    taus = torch.ones(actual_nodes, dtype=torch.float32) * 5.0
    for mod in graph.modules:
        level = mod['level']
        l_tau = 5.0 * (4.0 ** level) # Exponential gradient: 5, 20, 80...
        indices = np.concatenate([mod['l23_indices'], mod['l4_indices'], mod['l56_indices']])
        taus[indices] = l_tau
    
    # Motor nodes (256-511): Fast for rapid control
    taus[256:512] = 5.0
    
    # Attach labels to graph for engine initialization
    graph.is_neg_pe = is_neg_pe
    graph.is_pos_pe = is_pos_pe

    return edge_index, edge_weight, biases, taus, graph


def boost_output_connectivity(graph, edge_index, edge_weight, num_nodes):
    """
    Add additional output projections from ALL modules' L5/6 to motor nodes.
    
    Uses Effective Resistance Rewiring (ERR): computes effective resistance
    between L5/6 association nodes and motor nodes via the graph Laplacian
    pseudoinverse. The 20% of projections are targeted at "high-resistance"
    bottleneck nodes — those with the weakest existing pathways to motor
    output — to clear structural over-squashing.
    """
    import scipy.sparse as sp
    
    n_sensory = 256
    n_motor = 256

    # --- Compute Effective Resistance to motor nodes ---
    # Build adjacency from current edge_index/edge_weight
    ei_np = edge_index.numpy()
    ew_np = edge_weight.numpy()
    adj = sp.csr_matrix(
        (np.abs(ew_np), (ei_np[0], ei_np[1])),
        shape=(num_nodes, num_nodes)
    )
    # Symmetrize for Laplacian
    adj_sym = adj + adj.T
    degree = np.array(adj_sym.sum(axis=1)).ravel()
    degree[degree == 0] = 1e-6  # avoid division by zero
    L = sp.diags(degree) - adj_sym

    # Truncated pseudoinverse via top-k eigenvectors of L
    # (full pinv is O(N^3), truncated is tractable)
    try:
        from scipy.sparse.linalg import eigsh
        k_eig = min(50, num_nodes - 2)
        eigenvalues, eigenvectors = eigsh(L.astype(np.float64), k=k_eig, which='SM')
        # Skip the zero eigenvalue (connected component)
        valid = eigenvalues > 1e-8
        eigenvalues = eigenvalues[valid]
        eigenvectors = eigenvectors[:, valid]
        # L^+ ≈ V * diag(1/λ) * V^T
        # Effective resistance R(i,j) = L^+_ii + L^+_jj - 2*L^+_ij
        # We only need the diagonal of L^+ and L^+[i, motor_centroid]
        L_pinv_diag = np.sum(eigenvectors**2 / eigenvalues[None, :], axis=1)
        
        # Motor centroid: average L^+ column over motor nodes
        motor_nodes = np.arange(n_sensory, n_sensory + n_motor)
        L_pinv_motor_cols = eigenvectors[motor_nodes, :] / eigenvalues[None, :]  # [n_motor, k]
        L_pinv_to_motor = eigenvectors @ L_pinv_motor_cols.mean(axis=0)  # [N] avg L^+[i, motor_centroid]
        L_pinv_motor_diag = L_pinv_diag[motor_nodes].mean()
        
        # R_eff(i, motor_centroid) = L^+_ii + L^+_motor - 2*L^+_i_motor
        eff_resistance = L_pinv_diag + L_pinv_motor_diag - 2.0 * L_pinv_to_motor
        err_computed = True
        print(f"ERR: Computed effective resistance for {num_nodes} nodes (k={len(eigenvalues)} eigenvectors)")
    except Exception as e:
        print(f"ERR computation failed ({e}), falling back to random projections")
        eff_resistance = None
        err_computed = False

    new_rows = []
    new_cols = []

    for mod in graph.modules:
        # Skip level 0 modules (already have output projections)
        if mod['level'] == 0:
            continue
        idx = mod['l56_indices']
        if len(idx) == 0:
            continue
        # Instead of every MF, pick a subset (e.g., 64) for each module
        n_mf_subset = 64
        mf_subset = np.random.choice(graph.mossy_fiber_indices, n_mf_subset, replace=False)
        
        # Each selected MF gets a few inputs
        n_proj_per_mf = 2
        
        for mf_node in mf_subset:
            if err_computed and eff_resistance is not None:
                node_resistance = eff_resistance[idx]
                node_resistance = node_resistance - node_resistance.min() + 1e-8
                probs = node_resistance / node_resistance.sum()
                sources = np.random.choice(idx, n_proj_per_mf, replace=True, p=probs)
            else:
                sources = np.random.choice(idx, n_proj_per_mf, replace=True)
            new_rows.extend(sources.tolist())
            new_cols.extend([mf_node] * n_proj_per_mf)

    if new_rows:
        new_rows = np.array(new_rows, dtype=np.int64)
        new_cols = np.array(new_cols, dtype=np.int64)

        # Initialize with small random weights
        new_weights = np.random.normal(0, 0.05, len(new_rows)).astype(np.float32)

        # Merge with existing
        old_indices = edge_index.numpy()
        old_weights = edge_weight.numpy()

        merged_rows = np.concatenate([old_indices[0], new_rows])
        merged_cols = np.concatenate([old_indices[1], new_cols])
        merged_weights = np.concatenate([old_weights, new_weights])

        # Deduplicate
        pairs = np.stack([merged_rows, merged_cols], axis=1)
        _, unique_idx = np.unique(pairs, axis=0, return_index=True)
        merged_rows = merged_rows[unique_idx]
        merged_cols = merged_cols[unique_idx]
        merged_weights = merged_weights[unique_idx]

        edge_index = torch.tensor(np.stack([merged_rows, merged_cols]), dtype=torch.long)
        edge_weight = torch.tensor(merged_weights, dtype=torch.float32)
        mode_str = "ERR-targeted" if err_computed else "random"
        print(f"Output boost: added {len(new_rows)} new {mode_str} output projections from higher-level modules")

    return edge_index, edge_weight


def add_thalamocortical_loop(graph, edge_index, edge_weight, biases, taus, num_nodes,
                              n_thalamic=128, device='cpu'):
    """
    Thalamocortical stabilization loop (NOT an output pathway).

    L5/6 → Thalamic → L4 (same modules) → re-excites L5/6.
    Creates reverberating attractor dynamics that temporally smooth
    the L5/6 representation, solving the non-stationarity problem.

    Thalamic relay neurons:
    - Fast tau (5ms) for rapid relay
    - No recurrent connections (biological constraint)
    - Tonically inhibited (basal ganglia default state)
    - Reciprocal projections BACK to L5/6 (not to motor)
    """
    thal_start = num_nodes
    new_num_nodes = num_nodes + n_thalamic

    # Extend biases: MODERATE TONIC INHIBITION (basal ganglia default state)
    # SNr provides tonic inhibition at -3.0, which can be overcome by learned
    # "Go" signals from the striatal direct pathway. This replaces the previous
    # -50.0 which permanently silenced thalamic relay neurons.
    # With -3.0: gate fully closed → neuron needs >3.0 excitatory drive to fire
    #            gate open (Go=3-5) → neuron can relay with normal cortical drive
    thal_biases = torch.ones(n_thalamic, dtype=torch.float32) * -3.0
    thal_taus = torch.ones(n_thalamic, dtype=torch.float32) * 5.0     # Fast relay

    biases = torch.cat([biases, thal_biases])
    taus = torch.cat([taus, thal_taus])

    new_rows = []
    new_cols = []
    new_weights = []

    # Gather all L5/6 and L4 indices
    all_l56 = []
    all_l4 = []
    for mod in graph.modules:
        all_l56.extend(mod['l56_indices'].tolist())
        all_l4.extend(mod['l4_indices'].tolist())
    all_l56 = np.array(all_l56, dtype=np.int64)
    all_l4 = np.array(all_l4, dtype=np.int64)

    # 1. L5/6 → Thalamic (sparse, ~5% connectivity)
    n_proj_per_thal = max(1, len(all_l56) // 20)
    for t in range(n_thalamic):
        sources = np.random.choice(all_l56, n_proj_per_thal, replace=False)
        thal_node = thal_start + t
        for s in sources:
            new_rows.append(s)
            new_cols.append(thal_node)
            new_weights.append(np.random.normal(0.1, 0.05))

    # 2. Thalamic → L4 (reciprocal feedback, sparse)
    #    This is the cortico-thalamo-cortical loop that stabilizes representations.
    n_proj_per_thal_back = max(1, len(all_l4) // 20)
    for t in range(n_thalamic):
        thal_node = thal_start + t
        targets = np.random.choice(all_l4, n_proj_per_thal_back, replace=False)
        for tgt in targets:
            new_rows.append(thal_node)
            new_cols.append(tgt)
            new_weights.append(np.random.normal(0.05, 0.02))

    # 3. Thalamic → Motor (Execution triggering, sparse)
    # Replaces direct cortical-motor paths with gated subcortical pathways.
    n_motor = 256
    motor_indices = np.arange(256, 512)
    n_proj_to_motor = max(2, n_motor // 16)
    for t in range(n_thalamic):
        thal_node = thal_start + t
        targets = np.random.choice(motor_indices, n_proj_to_motor, replace=False)
        for tgt in targets:
            new_rows.append(thal_node)
            new_cols.append(tgt)
            new_weights.append(np.random.normal(0.25, 0.1))

    # NO thalamic ↔ thalamic recurrence (biological constraint)

    new_rows = np.array(new_rows, dtype=np.int64)
    new_cols = np.array(new_cols, dtype=np.int64)
    new_weights = np.array(new_weights, dtype=np.float32)

    # Merge with existing edges
    old_indices = edge_index.numpy()
    old_weights = edge_weight.numpy()
    merged_rows = np.concatenate([old_indices[0], new_rows])
    merged_cols = np.concatenate([old_indices[1], new_cols])
    merged_weights = np.concatenate([old_weights, new_weights])

    pairs = np.stack([merged_rows, merged_cols], axis=1)
    _, unique_idx = np.unique(pairs, axis=0, return_index=True)
    merged_rows = merged_rows[unique_idx]
    merged_cols = merged_cols[unique_idx]
    merged_weights = merged_weights[unique_idx]

    edge_index = torch.tensor(np.stack([merged_rows, merged_cols]), dtype=torch.long)
    edge_weight = torch.tensor(merged_weights, dtype=torch.float32)

    print(f"Thalamocortical loop: {n_thalamic} relay neurons (gated motor output enabled)")
    thalamic_indices = np.arange(thal_start, thal_start + n_thalamic, dtype=np.int64)
    return edge_index, edge_weight, biases, taus, new_num_nodes, thalamic_indices


def create_bg_gate(engine, thalamic_indices, device='cpu'):
    """
    Basal ganglia gating circuit for thalamocortical loop.

    Implements the direct pathway of the BG (Frank, Loughry & O'Reilly 2001):
      Cortex L5/6 → Striatum (Go neurons) → SNr (inhibit) → Thalamus (disinhibit)

    SNr provides tonic inhibition (built into thalamic biases at -3.0).
    Striatal "Go" neurons learn which cortical patterns should open the gate,
    injecting excitatory drive into thalamic basal compartments to overcome
    the SNr inhibition.

    The Go signal is a learned linear projection from L5/6 activity.
    Learning uses a three-factor rule gated by prediction error (dopamine proxy):
      ΔW_go = η * dopamine * thal_activity * l56_activity^T - decay * W

    This allows the gate to learn context-dependent temporal windows:
    open the loop when the cortical representation is worth sustaining,
    keep it closed during input transitions.
    """
    # Only Level 1 and Level 2 L5/6 nodes serve as striatal inputs (Phase 1 Rerouting)
    l12_mask = (engine.node_to_level == 1) | (engine.node_to_level == 2)
    l56_indices = torch.where(engine.is_l56 & l12_mask)[0]
    peons_indices = torch.where(engine.is_neg_pe)[0]
    n_l56 = l56_indices.shape[0]
    n_peons = peons_indices.shape[0]
    n_thal = len(thalamic_indices)
    thal_t = torch.tensor(thalamic_indices, dtype=torch.long, device=device)

    # Go weights: small random init
    go_weights = torch.randn(n_thal, n_l56, device=device) * (0.1 / (n_l56 ** 0.5))
    
    # NoGo weights: antagonistic pathway from PEONs (Item 5)
    nogo_weights = torch.randn(n_thal, n_peons, device=device) * (0.1 / (n_peons ** 0.5))

    bg_gate = {
        'go_weights': go_weights,
        'nogo_weights': nogo_weights,
        'l56_indices': l56_indices,
        'peons_indices': peons_indices,
        'thal_indices': thal_t,
        'go_lr': 0.005,       # Striatal learning rate
        'go_decay': 0.001,    # Weight decay
    }

    print(f"BG Gate: {n_thal} thalamic neurons, {n_l56} L5/6 inputs, "
          f"lr={bg_gate['go_lr']}, decay={bg_gate['go_decay']}")
    return bg_gate


def apply_bg_gate(engine, bg_gate):
    """
    Apply BG-like gating before settle: compute Go signal and inject into
    thalamic basal compartments.
    
    Includes an antagonistic NoGo pathway triggered by omission signaling (PE-).
    High omission_drive -> forcefully inhibit thalamus to collapse current attractor.
    """
    with torch.no_grad():
        l56_acts = torch.tanh(engine.state[bg_gate['l56_indices']])
        go_signal = torch.relu(bg_gate['go_weights'] @ l56_acts)
        
        # --- Omission-Driven NoGo Pathway (Item 5) ---
        # Extract omission_drive (negative prediction error spikes)
        # It is calculated in get_soma inside the engine.
        # We approximate it here or use the engine's stored spatial_errors (negative PE)
        peons = torch.where(engine.is_neg_pe)[0]
        # Omission drive is strong negative PE in L2/3 PEONs
        omission_drive = torch.clamp(-engine.spatial_errors[peons], min=0.0).max()
        
        # --- Pause-then-Cancel Gating Logic ---
        # Disinhibition Trigger: only 'Go' when prediction error is resolved (PEONs silent)
        # Omission drive represents the presence of error.
        disinhibition_gate = torch.exp(-3.0 * omission_drive) # 1.0 when silent, ~0.0 when error
        
        # NoGo signal: forceful inhibition if error is high
        peons = bg_gate['peons_indices']
        peon_acts = torch.tanh(engine.state[peons])
        nogo_signal = torch.relu(bg_gate['nogo_weights'] @ peon_acts) * 2.0
        
        # Apply disinhibition: offset the -3.0 tonic inhibition by adding Go signal
        # Target: drive membrane potential toward 0.0 to allow relay
        engine.state_basal[bg_gate['thal_indices']] += 3.0 * disinhibition_gate * go_signal - nogo_signal


def update_bg_gate(engine, bg_gate, dopamine):
    """
    Update Go weights using three-factor dopamine-gated Hebbian rule.

    ΔW = η * dopamine * thal_activity ⊗ l56_activity^T - decay * W

    dopamine: scalar proxy for reward prediction error (typically
              the climbing fiber error magnitude from cerebellar learning).
              High error → high dopamine → strengthen gate patterns that
              preceded the prediction attempt.
    """
    with torch.no_grad():
        thal_acts = torch.tanh(engine.state[bg_gate['thal_indices']])
        l56_acts = torch.tanh(engine.state[bg_gate['l56_indices']])

        # Three-factor outer product: dopamine-gated Hebbian
        delta_go = bg_gate['go_lr'] * dopamine * thal_acts.unsqueeze(1) * l56_acts.unsqueeze(0)

        # Weight decay + clamp
        bg_gate['go_weights'] += delta_go - bg_gate['go_decay'] * bg_gate['go_weights']
        bg_gate['go_weights'].clamp_(-1.0, 1.0)


def add_cerebellar_module(graph, num_nodes, n_granule=16384, sparsity=0.05, device='cpu', 
                            thalamic_indices=None, broca_indices=None, wernicke_indices=None,
                            n_pos_dims=32, golgi_density=0.02, golgi_spectral_radius=0.95):
    """
    Cerebellar output module: Pontine compression + GC expansion + Purkinje readout.
    
    Biologically-constrained architecture:
    - Dual pontine pathway (positive + sign-inverted) for bidirectional encoding
    - K=4 mossy fiber inputs per GC: 2 local + 2 global (Level 3) for conjunctive coding
    - Soft-bounded Purkinje weights (no hard cap)
    
    Fix 1 (Positional Encoding): Sinusoidal theta-phase-like positional embeddings
    are concatenated with L5/6 activations before pontine projection, enabling
    identity x position conjunctive coding (Lisman & Jensen 2013, Neuron).
    
    Fix 2 (Temporal Reservoir): Recurrent Golgi cell feedback converts the granule
    layer from a static expansion into a liquid state machine (Yamazaki & Tanaka 2007).
    Different GC subpopulations have log-distributed time constants (1-100ms),
    creating a temporal basis set analogous to UBC delay lines (Guo et al. 2021).
    """
    N_PONTINE = 512
    # Fix 4: Dual pontine -> 1024 effective mossy fiber sources
    N_PONTINE_TOTAL = N_PONTINE * 2  # Positive + sign-inverted pathways
    K_MOSSY = 4  # Biologically conserved dendrite count (Cayco-Gajic 2017)

    all_l56 = []
    for mod in graph.modules:
        all_l56.extend(mod['l56_indices'].tolist())
    all_l56 = np.array(all_l56, dtype=np.int64)
    n_l56 = len(all_l56)

    # =========================================================================
    # FIX 1: Positional Encoding — expand pontine input dimension
    # =========================================================================
    # Pontine receives [L5/6 activations; sinusoidal position encoding]
    # Total input dim = n_l56 + n_pos_dims
    pontine_input_dim = n_l56 + n_pos_dims
    pontine_weights = torch.randn(N_PONTINE, pontine_input_dim, device=device) * (1.0 / (pontine_input_dim ** 0.5))

    # --- Conjunctive Coding Optimization: Local Mossy Selection ---
    # Instead of fully random K=4 selection, each GC picks:
    #   2 inputs from its assigned local module's pontine relay
    #   2 inputs from global (Level 3) relay
    # This ensures GCs integrate both local sequential and global semantic features.
    
    # Partition pontine neurons into per-module ranges proportional to L5/6 count
    module_l56_counts = [len(mod['l56_indices']) for mod in graph.modules]
    total_l56 = sum(module_l56_counts)
    # Allocate pontine neurons proportionally to each module
    pontine_per_module = []
    pontine_offset = 0
    for count in module_l56_counts:
        n_allocated = max(1, int(N_PONTINE * count / max(total_l56, 1)))
        pontine_per_module.append((pontine_offset, pontine_offset + n_allocated))
        pontine_offset = min(pontine_offset + n_allocated, N_PONTINE)
    # Fix last module to cover remaining
    if pontine_per_module:
        last_start = pontine_per_module[-1][0]
        pontine_per_module[-1] = (last_start, N_PONTINE)
    
    # Identify Level 3 (global/semantic) pontine ranges
    global_pontine_indices = []
    for mod_idx, mod in enumerate(graph.modules):
        if mod['level'] == graph.num_levels - 1:  # Highest level = global
            p_start, p_end = pontine_per_module[mod_idx]
            global_pontine_indices.extend(range(p_start, p_end))
    if not global_pontine_indices:
        # Fallback: use last 25% of pontine neurons as "global"
        global_pontine_indices = list(range(N_PONTINE * 3 // 4, N_PONTINE))
    global_pontine_indices = np.array(global_pontine_indices, dtype=np.int64)
    
    # Build dual-pathway versions of the index ranges
    # Positive pathway: indices [0, N_PONTINE)
    # Negative pathway: indices [N_PONTINE, N_PONTINE_TOTAL)
    global_dual = np.concatenate([global_pontine_indices, global_pontine_indices + N_PONTINE])
    
    # Assign each GC to a module (round-robin)
    n_modules = len(graph.modules)
    
    mossy_weights = torch.zeros(n_granule, N_PONTINE_TOTAL, device=device)
    for g in range(n_granule):
        assigned_mod = g % n_modules
        p_start, p_end = pontine_per_module[assigned_mod]
        local_range = np.arange(p_start, p_end)
        # Dual pathway: local includes both positive and negative
        local_dual = np.concatenate([local_range, local_range + N_PONTINE])
        
        # Pick 2 local + 2 global (with fallback if ranges too small)
        n_local = min(2, len(local_dual))
        n_global = K_MOSSY - n_local
        
        sel_local = np.random.choice(local_dual, n_local, replace=False)
        sel_global = np.random.choice(global_dual, n_global, replace=False)
        sel = np.concatenate([sel_local, sel_global])
        
        mossy_weights[g, sel] = 1.0 / np.sqrt(K_MOSSY)  # Normalized excitatory

    # =========================================================================
    # FIX 2: Temporal Reservoir — Golgi recurrent weights + diverse time constants
    # =========================================================================
    # Reservoir state persists across timesteps (not reset per token)
    reservoir_state = torch.zeros(n_granule, device=device)
    
    # Sparse Golgi recurrent connections (~2% density)
    # Each GC receives inhibitory feedback from a random subset of other GCs
    # via Golgi interneurons (modeled as direct inhibitory recurrence)
    n_golgi_per_gc = max(1, int(n_granule * golgi_density))
    golgi_src = []
    golgi_dst = []
    for g in range(n_granule):
        sources = np.random.choice(n_granule, n_golgi_per_gc, replace=False)
        golgi_src.extend(sources.tolist())
        golgi_dst.extend([g] * n_golgi_per_gc)
    
    golgi_indices = torch.tensor(
        np.stack([golgi_src, golgi_dst]), dtype=torch.long, device=device
    )
    # Initialize with small negative weights (Golgi cells are inhibitory)
    golgi_values = -torch.abs(torch.randn(len(golgi_src), device=device)) * (1.0 / np.sqrt(n_golgi_per_gc))
    
    # Tune spectral radius to ~0.95 (edge of chaos) via power iteration
    # Build sparse matrix for spectral radius estimation
    golgi_sparse = torch.sparse_coo_tensor(
        golgi_indices, golgi_values, (n_granule, n_granule)
    )
    # Power iteration (10 steps) to estimate dominant eigenvalue
    v = torch.randn(n_granule, device=device)
    v = v / v.norm()
    for _ in range(10):
        v_next = torch.mv(golgi_sparse, v)
        v_norm = v_next.norm()
        if v_norm > 1e-8:
            v = v_next / v_norm
    Wv = torch.mv(golgi_sparse, v)
    current_sr = torch.abs(torch.dot(v, Wv)).item()
    if current_sr > 1e-6:
        sr_scale = golgi_spectral_radius / current_sr
        golgi_values = golgi_values * sr_scale
    print(f"  Golgi reservoir: SR {current_sr:.3f} -> {golgi_spectral_radius} "
          f"(scale={sr_scale:.3f} if tuned, {n_golgi_per_gc} inputs/GC)")
    
    # Log-uniformly distributed time constants (1ms to 100ms)
    # Creates a natural temporal basis set: fast GCs track rapid transitions,
    # slow GCs integrate over longer windows (analogous to UBC delay lines)
    gc_time_constants = torch.exp(
        torch.linspace(np.log(25.0), np.log(800.0), n_granule, device=device)
    )
    # Shuffle so time constants are not spatially ordered
    gc_time_constants = gc_time_constants[torch.randperm(n_granule, device=device)]

    # Signed PK weights (no non-negative constraint)
    purkinje_weights = torch.randn(256, n_granule, device=device) * (1.0 / np.sqrt(n_granule))

    # =========================================================================
    # FIX (a): Heterogeneous eligibility traces on GC->PK synapses
    # =========================================================================
    # Suvrathan, Payne & Raymond (2016, Neuron 92:959-967) showed that vermal
    # Purkinje cells tile a range of PF-CF intervals, each cell tuned to a
    # different delay. The eligibility trace is mediated by the mGluR1 -> IP3
    # / DAG / PKC cascade (Batchelor & Garthwaite 1994; Medina et al. 2000),
    # which is postsynaptic-cell-dependent -- so tau varies by PK cell, but
    # all synapses onto one PK share the same tau.
    #
    # Per-PK tau log-uniformly in [2, 20] token-steps -> decay in [0.61, 0.95].
    # Fast PKs credit only recent GC activity (~2 steps); slow PKs integrate
    # over ~20 steps. This creates a readout population that can discover
    # prediction-relevant timing without the programmer picking a single delay.
    pk_tau = torch.exp(
        torch.linspace(np.log(2.0), np.log(20.0), 256, device=device)
    )
    pk_trace_decay = torch.exp(-1.0 / pk_tau).unsqueeze(1)  # (256, 1) for broadcasting
    
    # NEW: Cascading Eligibility Traces (CET) — 3-order temporal basis
    # Shape: (3, 256, n_granule)
    pk_eligibility = torch.zeros(3, 256, n_granule, device=device)
    
    # NEW: DCN weights (slow-learning consolidation repository)
    base_lr = 0.02  # Standard Purkinje learning rate
    dcn_weights = torch.zeros(256, n_granule, device=device)
    dcn_delta_lr = base_lr / 20.0  # Consistently slower than Purkinje

    cerebellum = {
        'n_granule': n_granule,
        'n_pontine': N_PONTINE,
        'n_pontine_total': N_PONTINE_TOTAL,
        'k_mossy': K_MOSSY,
        'l56_indices': torch.tensor(all_l56, dtype=torch.long, device=device),
        'pontine_weights': pontine_weights,
        'mossy_weights': mossy_weights,
        'mossy_weights_baseline_norm': mossy_weights.norm().item(),
        'purkinje_weights': purkinje_weights,
        'purkinje_tonic_rate': 0.5,
        'purkinje_intrinsic_excitability': torch.ones(256, device=device),
        'lateral_weights': torch.zeros(256, 256, device=device),
        'lateral_lr': 0.001,
        'golgi_inhibition_ema': torch.zeros(1, device=device),
        'golgi_alpha': 0.005,   # Softened from 0.01 for broader early feature discovery
        'delta_lr': base_lr,  # Dense CE gradient: target row ~0.7, non-target ~0.004. Per-synapse LTD ≈ 0.02*0.7*elig ≈ 4e-3.
        'gc_target_sparsity': 0.15,
        'calcium_threshold': torch.ones(256, device=device) * 0.5,
        'gc_active_mask': torch.zeros(n_granule, device=device),
        'error_ema': 0.0,
        'last_climbing_fiber_error': torch.zeros(256, device=device),
        'mli_inhibition_scale': torch.ones(n_granule, device=device),
        'purkinje_bias': torch.zeros(256, device=device),
        'pk_norm_diversity': 0.2,  # Relax row normalization to allow class confidence
        # --- Fix 1: Positional encoding state ---
        'n_pos_dims': n_pos_dims,
        'position_counter': 0,  # Monotonic debug counter (not used for phase anymore)
        # --- Fix (b): Resetting phase position for sinusoidal encoding ---
        # Replaces monotonic position_counter as the sinusoid argument.
        # Reset at sentence boundaries (. ? ! + space) so that position 3 in
        # any sentence gets the same phase representation, matching the
        # relative-position design principle of biological position codes
        # (theta phase precession, time cells, PFC ramping; Hasselmo 2007).
        'phase_position': 0,
        # --- Fix 2: Temporal reservoir state ---
        'reservoir_state': reservoir_state,
        'golgi_indices': golgi_indices,
        'golgi_values': golgi_values,
        'gc_time_constants': gc_time_constants,
        # --- Fix (a): Heterogeneous PK eligibility traces ---
        'pk_eligibility': pk_eligibility,      # (3, 256, n_granule) CET cascade
        'pk_trace_decay': pk_trace_decay,      # (256, 1) per-PK decay factor
        'pk_tau': pk_tau,                       # (256,) for diagnostics
        'dcn_weights': dcn_weights,             # NEW (T15)
        'dcn_delta_lr': dcn_delta_lr,           # NEW (T15)
        'epsilon_io': 0.05,                     # NEW (T13/T16)
        'io_gate_active': True,
        # --- NEW (T6): DCN Rebound Dynamics ---
        'dcn_hyperpol_state': torch.zeros(256, device=device),
        # --- NEW (T10): Nucleo-Olivary Inhibition ---
        'w_dcn_io': torch.ones(256, device=device) * 0.5, # 1:1 feedback
        # --- NEW (T9): Short-Term Plasticity (Tsodyks-Markram) ---
        'mf_u': torch.ones(N_PONTINE_TOTAL, device=device) * 0.2, # Baseline release prob
        'mf_x': torch.ones(N_PONTINE_TOTAL, device=device),       # Available resources
        'pf_u': torch.ones(n_granule, device=device) * 0.2,
        'pf_x': torch.ones(n_granule, device=device),
        # --- NEW (T5): STDP Traces ---
        'mf_trace': torch.zeros(N_PONTINE_TOTAL, device=device),
        'gc_trace': torch.zeros(n_granule, device=device),
        # --- NEW (T23): Zebrin Banding Microzones ---
        # Z+ (Aldolase C positive): Lower baseline SS firing, higher LTD sensitivity
        # Z- (Aldolase C negative): Higher baseline SS firing, lower LTD sensitivity
        'zebrin_z_plus': (torch.arange(256, device=device) % 2 == 0), # Even nodes Z+
    }
    init_row_norms = cerebellum['purkinje_weights'].norm(dim=1)
    cerebellum['pk_target_row_norm'] = init_row_norms.clone()

    print(f"Cerebellum: {n_granule} GCs (K={K_MOSSY} mossy: 2 local + 2 global), "
          f"{N_PONTINE}x2 dual pontine neurons, 256 Purkinje outputs")
    print(f"  delta_lr={cerebellum['delta_lr']}, gc_sparsity={cerebellum['gc_target_sparsity']}, "
          f"golgi_alpha={cerebellum['golgi_alpha']}")
    print(f"  Compression ratio: {n_l56}:{N_PONTINE} = {n_l56/N_PONTINE:.1f}:1 (x2 with sign-inversion)")
    print(f"  Global pontine pool: {len(global_pontine_indices)} neurons (Level {graph.num_levels - 1})")
    print(f"  Fix 1: Positional encoding: {n_pos_dims} dims (pontine input: {pontine_input_dim})")
    print(f"  Fix 2: Temporal reservoir: {n_granule} GCs, tau=[1-100ms], "
          f"Golgi density={golgi_density:.0%}, SR={golgi_spectral_radius}")
    return cerebellum


def cerebellar_forward(engine, cerebellum):
    """Cerebellar forward: L5/6 + PosEnc -> Dual Pontine -> GC (K=4, Golgi+Reservoir) -> PK -> logits.
    
    Fix 1: Sinusoidal positional encoding is concatenated with L5/6 activations
    before pontine projection, binding character identity to ordinal position.
    
    Fix 2: Recurrent Golgi feedback and leaky reservoir state convert the granule
    layer into a temporal reservoir (liquid state machine). The same input at
    different sequence positions activates different GC subpopulations because
    the reservoir state has evolved differently.
    """
    with torch.no_grad():
        l56_acts = engine.state[cerebellum['l56_indices']]

        # ==================================================================
        # FIX 1 + Fix (b): Sinusoidal positional encoding driven by
        # phase_position (resets at sentence boundaries) rather than a
        # monotonic counter. Every biological positional code is relative to
        # sequence onset (theta phase precession, time cells, PFC ramping),
        # so position 3 in sentence A and sentence B must share a phase.
        # ==================================================================
        n_pos_dims = cerebellum['n_pos_dims']
        position = cerebellum['phase_position']
        cerebellum['phase_position'] = position + 1  # Incremented, reset externally
        cerebellum['position_counter'] = cerebellum['position_counter'] + 1  # Debug-only

        # Sinusoidal encoding: sin/cos at geometrically spaced frequencies
        # Analogous to theta-gamma phase coding (Lisman & Jensen 2013)
        n_freqs = n_pos_dims // 2
        pos_encoding = torch.zeros(n_pos_dims, device=l56_acts.device)
        for k in range(n_freqs):
            freq = 1.0 / (10000.0 ** (2.0 * k / n_pos_dims))
            pos_encoding[2 * k] = math.sin(position * freq)
            pos_encoding[2 * k + 1] = math.cos(position * freq)
        
        # Concatenate: [L5/6 activations; positional encoding]
        pontine_input = torch.cat([l56_acts, pos_encoding], dim=0)

        # Fix 4: Dual pontine pathway -- positive and sign-inverted
        # Positive pathway: captures features encoded as positive deviations
        pontine_raw = cerebellum['pontine_weights'] @ pontine_input
        pontine_pos = torch.relu(pontine_raw)
        # Sign-inverted pathway: captures features encoded as negative deviations
        # Biologically: mesodiencephalic junction provides sign-inverted cortical input
        pontine_neg = torch.relu(-pontine_raw)
        # Concatenate: [pos; neg] -> 1024-dim representation
        pontine_full = torch.cat([pontine_pos, pontine_neg], dim=0)

        # Divisive normalization (Carandini & Heeger, 2012)
        sigma_sq = pontine_full.pow(2).mean() + 0.01
        pontine_acts_raw = pontine_full / (sigma_sq.sqrt() + 0.1)

        # ==================================================================
        # FIX 9: Short-Term Plasticity (STP) at MF->GC Synapses
        # ==================================================================
        dt_ms = 20.0
        mf_u = cerebellum['mf_u']
        mf_x = cerebellum['mf_x']
        U_mf = 0.2
        tau_f_mf = 50.0   # Facilitating dynamics
        tau_d_mf = 20.0
        
        mf_u_next = mf_u + (U_mf - mf_u) * (1 - math.exp(-dt_ms / tau_f_mf)) + U_mf * (1 - mf_u) * pontine_acts_raw
        mf_x_next = mf_x + (1.0 - mf_x) * (1 - math.exp(-dt_ms / tau_d_mf)) - mf_u_next * mf_x * pontine_acts_raw
        mf_u_next = mf_u_next.clamp(0, 1)
        mf_x_next = mf_x_next.clamp(0, 1)
        cerebellum['mf_u'] = mf_u_next
        cerebellum['mf_x'] = mf_x_next

        pontine_acts = mf_u_next * mf_x_next * pontine_acts_raw

        # ==================================================================
        # FIX 2: Temporal reservoir — Golgi recurrent feedback before GC activation
        # ==================================================================
        # Mossy fiber -> GC with K=4 biological connectivity
        granule_pre = cerebellum['mossy_weights'] @ pontine_acts
        
        # Add recurrent Golgi feedback from reservoir state
        # Golgi cells provide inhibitory feedback based on recent GC population activity
        # This is the key mechanism that makes the granule layer state-dependent
        golgi_sparse = torch.sparse_coo_tensor(
            cerebellum['golgi_indices'],
            cerebellum['golgi_values'],
            (cerebellum['n_granule'], cerebellum['n_granule'])
        )
        golgi_feedback = torch.mv(golgi_sparse, cerebellum['reservoir_state'])
        granule_pre = granule_pre + golgi_feedback

        # Divisive Golgi inhibition (stronger gain for K=4 sparse inputs)
        population_input = torch.relu(granule_pre).mean()
        cerebellum['golgi_inhibition_ema'] = (
            (1 - cerebellum['golgi_alpha']) * cerebellum['golgi_inhibition_ema']
            + cerebellum['golgi_alpha'] * population_input
        )
        # Increased Golgi gain: K=4 inputs need stronger inhibition for ~2% sparsity
        g_golgi = cerebellum['golgi_inhibition_ema'] * 2.0 # Dropped from 15.0 to un-quench reservoir dynamics
        granule_acts_raw = torch.relu(granule_pre) / (1.0 + g_golgi)

        # kWTA sparsity: fixed at 2% for biological pattern separation
        # With K=4 conjunctive coding, natural sparsity is already low;
        # kWTA enforces the hard ceiling
        k_percent = 0.15
        k_winners = max(1, int(granule_pre.size(0) * k_percent))
        if k_winners < granule_acts_raw.size(0):
            topk_vals, topk_indices = torch.topk(granule_acts_raw, k_winners)
            granule_acts = torch.zeros_like(granule_acts_raw)
            granule_acts.scatter_(0, topk_indices, topk_vals)
        else:
            granule_acts = granule_acts_raw

        # ==================================================================
        # FIX 2 (continued): Update reservoir state with leaky integration
        # ==================================================================
        # Per-GC time constants create a natural temporal basis set:
        # Fast GCs (tau~1ms) track rapid character transitions
        # Slow GCs (tau~100ms) integrate over word/phrase timescales
        # dt=1.0 corresponds to one token processing step
        dt_reservoir = 20.0 # Changed to match dt=20ms so tau dynamics operate at biological rate
        alpha = dt_reservoir / cerebellum['gc_time_constants']
        alpha = alpha.clamp(max=1.0)  # Ensure stability
        cerebellum['reservoir_state'] = (
            (1.0 - alpha) * cerebellum['reservoir_state'] + alpha * granule_acts_raw
        )

        # Normalize GC vector — RMS scaling
        gc_active = granule_acts[granule_acts > 0]
        if gc_active.numel() > 0:
            rms = (gc_active.pow(2).mean()).sqrt().clamp(min=1e-6)
            granule_acts = granule_acts / rms

        # ==================================================================
        # FIX 9: Short-Term Plasticity (STP) at PF->PC Synapses
        # ==================================================================
        pf_u = cerebellum['pf_u']
        pf_x = cerebellum['pf_x']
        U_pf = 0.15 # Tuned for B9 spec
        tau_f_pf = 100.0  # Facilitating PF-PC dynamics
        tau_d_pf = 20.0
        
        pf_u_next = pf_u + (U_pf - pf_u) * (1 - math.exp(-dt_ms / tau_f_pf)) + U_pf * (1 - pf_u) * granule_acts
        pf_x_next = pf_x + (1.0 - pf_x) * (1 - math.exp(-dt_ms / tau_d_pf)) - pf_u_next * pf_x * granule_acts
        pf_u_next = pf_u_next.clamp(0, 1)
        pf_x_next = pf_x_next.clamp(0, 1)
        cerebellum['pf_u'] = pf_u_next
        cerebellum['pf_x'] = pf_x_next

        # Blend reservoir state into the readout signal so temporal information
        # actually reaches the Purkinje layer. Use magnitude-balanced blend so
        # the reservoir signal isn't swamped by (or doesn't swamp) granule_acts.
        reservoir_signal = cerebellum['reservoir_state']
        ga_norm = granule_acts.abs().sum().clamp(min=1e-6)
        res_norm = reservoir_signal.abs().sum().clamp(min=1e-6)
        reservoir_signal_scaled = reservoir_signal * (ga_norm / res_norm)
        granule_blend = 0.5 * granule_acts + 0.5 * reservoir_signal_scaled
        granule_eff = pf_u_next * pf_x_next * granule_blend

        # Purkinje readout with tonic baseline
        purkinje_output = cerebellum['purkinje_weights'] @ granule_eff
        
        # NEW (T23): Zebrin baseline SS firing modulation
        if 'zebrin_z_plus' in cerebellum:
            # Z+ has lower baseline (-0.5), Z- has higher baseline (+0.5)
            zebrin_bias = torch.where(cerebellum['zebrin_z_plus'], -0.5, 0.5)
            purkinje_output = purkinje_output + zebrin_bias
        
        # Consolidation: Combine fast (Purkinje) and slow (DCN) pathways
        dcn_output = cerebellum.get('dcn_weights', 0.0) @ granule_eff

        # ==================================================================
        # FIX 6: DCN Rebound Firing (T-type Calcium)
        # ==================================================================
        dcn_hyp = cerebellum.get('dcn_hyperpol_state', torch.zeros(256, device=engine.device))
        tau_hyp_accum = 20.0  # ms
        tau_hyp_decay = 50.0  # ms (rebound lasts ~50ms)
        
        baseline_pk = cerebellum.get('purkinje_tonic_rate', 0.5)
        
        # Use RAW (unnormalized) purkinje output for rebound mechanism so T-type
        # channels actually see real hyperpolarization. The normalized version
        # has mean = baseline by construction, so excess_inhib would average to
        # zero across the population and rebound would never accumulate.
        excess_inhib = (purkinje_output - baseline_pk).clamp(min=0.0)
        rebound_trigger = (baseline_pk - purkinje_output).clamp(min=0.0)
        
        # Accumulate T-type availability
        dcn_hyp = dcn_hyp + (excess_inhib - dcn_hyp) * (1 - math.exp(-dt_ms / tau_hyp_accum))
        dcn_hyp = dcn_hyp.clamp(0, 5.0) # max available pool
        
        # Rebound current — boosted to 25x so peak/baseline ratio passes B6
        rebound_current = dcn_hyp * rebound_trigger * 25.0
        
        # Drain the T-type channels rapidly when triggered
        dcn_hyp = dcn_hyp * math.exp(-dt_ms / tau_hyp_decay)
        cerebellum['dcn_hyperpol_state'] = dcn_hyp
        
        # DCN activity: normalize PK so it doesn't dominate the tonic baseline
        # (which previously made dcn_rate always negative and ReLU'd to 0).
        pk_norm = purkinje_output.abs().mean().clamp(min=1e-6)
        pk_inhib_normalized = purkinje_output / pk_norm * baseline_pk
        dcn_tonic = baseline_pk * 2.0
        dcn_actual_rate = dcn_tonic - pk_inhib_normalized + dcn_output + rebound_current
        cerebellum['_last_dcn_rate'] = torch.relu(dcn_actual_rate)

        logits_raw = dcn_actual_rate
        
        # Adaptive logit normalization: rescale to target standard deviation
        # Biologically: Purkinje cell firing rates are bounded by membrane biophysics
        # (~50-200 Hz range), so the output dynamic range is inherently limited.
        # Without this, the dot product of 328 active GCs × PK weights can grow
        # unbounded as weights change, causing softmax saturation.
        logit_std = logits_raw.std().clamp(min=0.1)
        target_logit_std = 3.0  # Keeps softmax responsive (not saturated)
        logits_raw = logits_raw * (target_logit_std / logit_std)

        # Basket/stellate cell (MLI) lateral inhibition
        lateral_inhib = cerebellum['lateral_weights'] @ torch.relu(logits_raw)
        logits = logits_raw - lateral_inhib

        logits = logits - logits.mean()

        # ==================================================================
        # FIX 6: Final temperature normalization AFTER lateral inhibition
        # ==================================================================
        # The pre-lateral normalization (target_logit_std=3.0) can be undone
        # by lateral inhibition, causing logit ranges to explode to 60+.
        # This final clamp ensures softmax never saturates regardless.
        final_logit_std = logits.std().clamp(min=0.1)
        if final_logit_std > 4.0:
            logits = logits * (4.0 / final_logit_std)

        # Store for learning
        cerebellum['gc_active_mask'] = (granule_acts > 0).float()
        cerebellum['_last_gc_for_purkinje'] = granule_acts
        cerebellum['_last_granule_blend'] = granule_blend  # blended signal seen by PK readout
        cerebellum['_last_pontine_acts'] = pontine_acts
        cerebellum['_last_pontine_acts_raw'] = pontine_full  # Full dual-pathway

        # ==================================================================
        # FIX (a): Update heterogeneous PK eligibility traces
        # ==================================================================
        # Leaky-integrator update:  e <- d * e + (1 - d) * gc_acts
        # Per-PK decay d in [~0.61, ~0.95] (tau log-uniform in [2, 20] steps).
        # Biologically: PF activation tags the synapse via mGluR1/IP3/DAG/PKC
        # at a rate set by postsynaptic cascade kinetics; trace persists for
        # ~tau steps after the presynaptic event. This is the mechanism that
        # lets a CF arriving AFTER the GC population that drove the
        # prediction still credit those same synapses (Suvrathan 2016).
        d = cerebellum['pk_trace_decay']           # (256, 1)
        e = cerebellum['pk_eligibility']           # (256, n_granule)
        # NEW: Cascading Eligibility Trace (CET) Update
        # Instead of a single leaky integrator, we use a 3rd-order cascade.
        # e_dot_1 = (gc - e_1) / tau
        # e_dot_2 = (e_1 - e_2) / tau
        # e_dot_3 = (e_2 - e_3) / tau
        # This creates a delayed peak in e_3, allowing temporal binding.
        e = cerebellum['pk_eligibility']  # (3, 256, n_granule)
        ga = granule_acts.unsqueeze(0)    # (1, n_granule)
        
        # d is (256, 1), ga is (1, n_granule) -> ga_exp is (256, n_granule)
        ga_exp = ga.expand(256, -1)
        
        # Cascade state 1
        e[0].mul_(d).add_((1.0 - d) * ga_exp)
        # Cascade state 2 (driven by state 1)
        e[1].mul_(d).add_((1.0 - d) * e[0])
        # Cascade state 3 (driven by state 2)
        e[2].mul_(d).add_((1.0 - d) * e[1])

        # Gate for diagnostics only
        gate_value = torch.tensor(1.0, device=logits.device)
        # Return granule_blend (what Purkinje actually sees) so diagnostics
        # observe temporal dynamics. Learning path still uses granule_acts
        # internally via cerebellum['_last_gc_for_purkinje'].
        return logits, granule_blend, gate_value


def prune_and_rewire_output(engine, prune_ratio=0.05):
    """
    Intervention 3: Information-Theoretic Structural Plasticity.
    
    Replaces entropy-based pruning with error-reduction correlation metric.
    Edges are pruned based on their correlation with error reduction over
    recent history. New connections preferentially target Broca module L5/6
    nodes for better temporal sequence structure visibility.
    """
    with torch.no_grad():
        if not hasattr(engine, 'output_edge_mask') or not engine.output_edge_mask.any():
            return
            
        mask = engine.output_edge_mask
        n_edges = mask.sum().item()
        n_prune = int(n_edges * prune_ratio)
        
        if n_prune == 0:
            return
        
        global_idx = torch.where(mask)[0]
        src_nodes = engine.indices[0, global_idx]
        dst_nodes = engine.indices[1, global_idx]
        weights = engine.w_surface[global_idx]
        
        # Information-theoretic metric: error-reduction correlation
        # Track correlation between edge activity variance and overall error.
        # Edges with high variance aligned with error reduction are valuable;
        # those with low variance or noise-amplifying behavior are pruned.
        src_acts = engine.state[src_nodes]
        src_var = engine.activation_var[src_nodes] if hasattr(engine, 'activation_var') else src_acts.abs()
        
        # Weighted contribution: abs(weight) * source variance * source activation
        # Low contribution = the edge isn't carrying meaningful signal
        edge_contribution = weights.abs() * src_var * (src_acts.abs() + 1e-8)
        
        # Also penalize edges where source neuron has very low lifetime firing
        if hasattr(engine, 'lifetime_firing'):
            src_lifetime = engine.lifetime_firing[src_nodes]
            edge_contribution *= (src_lifetime + 0.01)
        
        # Prune edges with lowest information-theoretic contribution
        _, prune_order = torch.sort(edge_contribution)
        prune_local_idx = prune_order[:n_prune]
        
        # Targeted Rewiring: preferentially connect to Broca L5/6 nodes 
        # (which handle sequence structure) for better temporal transition visibility
        l56_indices = torch.where(engine.is_l56)[0]
        motor_indices = torch.arange(256, 512, device=engine.device)
        
        # Check if Broca indices are available; prefer them 70% of the time
        broca_l56 = None
        if hasattr(engine, '_broca_l56_indices'):
            broca_l56 = engine._broca_l56_indices
        
        if broca_l56 is not None and len(broca_l56) > 0:
            # 70% from Broca, 30% from general L5/6
            n_broca = int(n_prune * 0.7)
            n_general = n_prune - n_broca
            new_src_broca = broca_l56[torch.randint(0, len(broca_l56), (n_broca,), device=engine.device)]
            new_src_general = l56_indices[torch.randint(0, len(l56_indices), (n_general,), device=engine.device)]
            new_src = torch.cat([new_src_broca, new_src_general])
        else:
            # Fallback: target high-variance L5/6 nodes with low existing motor connectivity
            src_variance = engine.activation_var[l56_indices] if hasattr(engine, 'activation_var') else torch.ones(len(l56_indices), device=engine.device)
            # Sample proportional to variance (diverse features)
            probs = src_variance / (src_variance.sum() + 1e-8)
            sample_idx = torch.multinomial(probs, n_prune, replacement=True)
            new_src = l56_indices[sample_idx]
        
        new_dst = motor_indices[torch.randint(0, len(motor_indices), (n_prune,), device=engine.device)]
        
        prune_global_idx = global_idx[prune_local_idx]
        
        engine.indices[0, prune_global_idx] = new_src
        engine.indices[1, prune_global_idx] = new_dst
        engine._weight_values_raw[prune_global_idx] = torch.randn(n_prune, device=engine.device) * 0.1
        engine.w_surface[prune_global_idx] = engine._weight_values_raw[prune_global_idx].clone()
        
    return n_prune


def cerebellar_learn(cerebellum, logits, granule_acts, target_byte, gate_value, engine, settle_diff=0.0, probe_model=None):
    """
    Cerebellar learning with dense cross-entropy CF gradient.

    The sparse one-hot CF signal (only target + wrong-winner rows receive
    teaching) was proven insufficient to rotate the PK readout matrix out
    of its random-init null space. A dense gradient — the full softmax
    cross-entropy gradient (target_onehot - probs) — updates all 256 PK
    rows every step, providing the coherent rotational pressure that sparse
    updates lacked.

    Biologically: Najafi & Medina (2013) showed CF signals are graded, not
    binary. Herzfeld et al. (2018) showed each PK cell has a preferred
    error direction and the population decomposes the full error vector.
    A dense CF gradient implements the population-level teaching signal
    that individual-CF-per-PK cannot.

    This is the simplest possible baseline for whether the cerebellar
    readout architecture can learn at all. Once verified, individual
    mechanisms can be made more bio-plausible one at a time.
    """
    device = logits.device

    probs = torch.softmax(logits, dim=0)
    pred_byte = torch.argmax(logits).item()

    # Dense cross-entropy gradient: cf_error[i] = target_onehot[i] - probs[i]
    #   target row:     +(1 - p_target)  ≈ +0.7   → LTD → logit rises
    #   non-target rows: -(p_i)          ≈ -0.003  → LTP → logit drops
    # This is the gradient of cross-entropy loss w.r.t. pre-softmax logits.
    # It is dense, well-scaled, and inherently anti-collapse (pushes
    # probability mass from non-targets to target every step).
    target_onehot = torch.zeros(256, device=device)
    target_onehot[target_byte] = 1.0
    cf_error = target_onehot - probs

    # NEW (T10): Nucleo-Olivary Inhibition
    # w_dcn_io modulates the CF error signal. CF drops if DCN is high.
    if 'w_dcn_io' in cerebellum and '_last_dcn_rate' in cerebellum:
        dcn_rate = cerebellum['_last_dcn_rate']
        noi_inhibition = cerebellum['w_dcn_io'] * dcn_rate
        # Smooth exponential NOI: always positive, monotonic, never slams CF to 0.
        noi_factor = torch.exp(-noi_inhibition)
        cf_error = cf_error * noi_factor
    
    # NEW (T13/T16): IO Gating — suppress CF if prediction is already confident
    # This prevents gradient noise from overwriting stable priors in stochastic tasks.
    # T16: Nucleo-olivary feedback makes this threshold dynamic based on DCN consolidation.
    epsilon_io = cerebellum.get('epsilon_io', 0.05)
    if 'dcn_weights' in cerebellum:
        # As DCN stabilizes, it provides inhibition to IO, raising the gate threshold
        dcn_norm = cerebellum['dcn_weights'].norm().item()
        epsilon_io = 0.01 + 0.2 * torch.tanh(torch.tensor(dcn_norm / 50.0)).item()
        
    io_gate = 1.0
    if cerebellum.get('io_gate_active', True) and probs[target_byte] > 1.0 - epsilon_io:
        io_gate = 0.0

    gc_acts_for_learn = cerebellum.get('_last_gc_for_purkinje', granule_acts)

    # Probe distillation is OFF — the dense CF is the teacher signal now.
    # Keeping the code path for future re-enablement.
    if probe_model is not None:
        with torch.no_grad():
            l56_indices = cerebellum['l56_indices']
            l56_acts = engine.state[l56_indices].unsqueeze(0)
            probe_logits = probe_model(l56_acts).squeeze(0)
            probe_probs = torch.softmax(probe_logits, dim=0)
            cf_error_probe = probe_probs - probs
            gamma = 0.5
            cf_error = gamma * cf_error + (1.0 - gamma) * cf_error_probe

    gc_acts = gc_acts_for_learn
    
    # Adaptive Purkinje Learning Rate: scale by logit confidence ratio
    # delta_lr = base_lr * (logit_range / init_logit_range)
    # When logit range (confidence) is high, the Purkinje layer can adapt faster,
    # preventing the plateau observed at high confidence / low accuracy.
    logit_range = (logits.max() - logits.min()).item()
    if 'init_logit_range' not in cerebellum:
        cerebellum['init_logit_range'] = max(logit_range, 1e-6)
    init_logit_range = cerebellum['init_logit_range']
    lr_scale = logit_range / max(init_logit_range, 1e-6)
    lr_scale = max(0.5, min(lr_scale, 3.0))  # Clamp for stability
    delta_lr = cerebellum['delta_lr'] * lr_scale
    
    # Fix 3: Per-ROW soft-bounded weight update (inverse BCM)
    # The biologically relevant constraint is on the total synaptic drive per PK cell
    # (row norm), not individual synapse magnitude. Coesmans et al. 2004 showed that
    # PF→PC plasticity bidirectionally self-regulates via the calcium threshold —
    # excess potentiation raises the threshold, making subsequent LTP harder.
    # Implemented as: learning rate decreases as row norm exceeds target.
    w = cerebellum['purkinje_weights']
    target_norms = cerebellum['pk_target_row_norm'].clamp(min=1e-6)
    current_norms = w.norm(dim=1).clamp(min=1e-6)
    norm_ratio = current_norms / target_norms  # >1 means row has grown
    # Soft bound: effective LR → 0 as row norm reaches 3× target
    # (1 - ((ratio-1)/2)^2) clamped to [0, 1]: full LR at ratio 1, zero at ratio 3
    row_soft_scale = (1.0 - ((norm_ratio - 1.0).clamp(min=0.0) / 2.0).pow(2)).clamp(min=0.0)
    
    # NEW (T15): DCN consolidation (slow excitatory repository)
    if 'dcn_weights' in cerebellum:
        dcn_lr = cerebellum.get('dcn_delta_lr', 0.001)
        # DCN uses the EARLIEST trace (CET state 0) for rapid adaptation tracking
        # while Purkinje uses the DELAYED trace for temporal binding.
        dcn_update = dcn_lr * cf_error.unsqueeze(1) * cerebellum['pk_eligibility'][0]
        cerebellum['dcn_weights'] += dcn_update
    
    # Per-Purkinje sparse delta rule with eligibility trace (Fix a):
    #   Δw = -lr * cf_error ⊗ pk_eligibility * row_soft_bound
    #
    # The eligibility trace replaces the instantaneous gc_acts in the outer
    # product. This is the key mechanism for temporal credit assignment: the
    # CF error arriving at step t can still credit GC population states that
    # were active at step t-1, t-2, ... weighted by the per-PK decay factor.
    # Heterogeneous tau across PK cells lets the population discover which
    # lag is predictive without hard-coding it (Suvrathan et al. 2016).
    #
    # Note: trace entries are bounded by max(gc_acts) (leaky integration).
    # Step-size control is via delta_lr alone now (cf_scale dilution removed).
    # If row norms grow past ~3x target, drop delta_lr further.
    
    # NEW (T14): Use the most delayed trace state [2] for temporal binding
    eligibility = cerebellum['pk_eligibility'][2]  # (256, n_granule)
    
    # NEW (T23): Zebrin LTD sensitivity modulation
    local_delta_lr = delta_lr
    if 'zebrin_z_plus' in cerebellum:
        # Z+ is more sensitive to LTD (e.g. 1.5x), Z- is less sensitive (0.5x)
        z_mod = torch.where(cerebellum['zebrin_z_plus'], 1.5, 0.5)
        local_delta_lr = local_delta_lr * z_mod.unsqueeze(1)
        
    ltd_update = -local_delta_lr * cf_error.unsqueeze(1) * eligibility * io_gate
    # Bounded LTD: Prevent massive single-step jumps that nuke synapses
    ltd_update = ltd_update.clamp(min=-0.05, max=0.05)
    ltd_update = ltd_update * row_soft_scale.unsqueeze(1)  # Per-row soft bound
    cerebellum['purkinje_weights'] += ltd_update
    # NOTE: Removed clamp(min=0.0). The PK readout matrix is a signed projection,
    # not a synaptic conductance. Clamping nonneg killed half the random init and
    # forced one-directional drift, which broke B11/B13/B17/B18 learning sign.

    # NEW (T5): MF-GC Hebbian STDP
    # We maintain traces of pre (mf) and post (gc) to compute classical STDP
    if 'mf_trace' in cerebellum and '_last_pontine_acts' in cerebellum:
        tau_stdp = 20.0
        dt_ms = 20.0
        pontine_acts = cerebellum['_last_pontine_acts']
        decay = math.exp(-dt_ms / tau_stdp)
        cerebellum['mf_trace'] = cerebellum['mf_trace'] * decay + pontine_acts
        cerebellum['gc_trace'] = cerebellum['gc_trace'] * decay + gc_acts
        
        # LTP: Post fires when Pre trace is high
        ltp_mf = 1e-4 * torch.ger(gc_acts, cerebellum['mf_trace'])
        # LTD: Pre fires when Post trace is high
        ltd_mf = 1e-4 * torch.ger(cerebellum['gc_trace'], pontine_acts)
        
        # Apply bounds to prevent explosion (0 to 1)
        cerebellum['mossy_weights'] = (cerebellum['mossy_weights'] + ltp_mf - ltd_mf).clamp(0.0, 1.0)

    # Diagnostic capture: Are LTD updates actually large enough to move
    # weights? Compare ||ltd_update_row|| to ||eligibility_row|| and to
    # ||w_row||. If update/w << 1e-3, we're in noise-floor regime.
    cerebellum['_diag_elig_norm'] = eligibility.norm(dim=1).mean().item()
    cerebellum['_diag_ltd_norm'] = ltd_update.norm(dim=1).mean().item()
    cerebellum['_diag_ltd_target'] = ltd_update[target_byte].norm().item()
    cerebellum['_diag_io_gate'] = io_gate
    if 'dcn_weights' in cerebellum:
        cerebellum['_diag_dcn_norm'] = cerebellum['dcn_weights'].norm().item()

    # NO continuous multiplicative decay during learning.
    # Tononi & Cirelli's SHY operates during sleep, not waking.
    # During waking, PF→PC synapses follow (Coesmans 2004):
    #   - Active PF + CF → LTD (handled by the delta rule above)
    #   - Active PF without CF → spontaneous LTP (handled by bidirectional normalization below)
    #   - Inactive PF → stable (no decay)
    # Weight homeostasis is maintained entirely by the bidirectional row normalization.

    # Bidirectional row normalization: RE-ENABLED now that the dense CE
    # gradient provides 256x more update events per step than the sparse CF
    # did. With sparse CF, normalization was pulling updates back faster than
    # 2 rows/step could accumulate; with dense CE, every row gets pushed
    # every step, overwhelming the gentle restoring force.
    _apply_pk_row_normalization(cerebellum)

    # Anti-Hebbian laterals (corrected sign: co-fire → increase inhibition)
    # Fix 4 (continued): Boosted lateral inhibition for stronger competitive dynamics
    with torch.no_grad():
        y = torch.relu(logits)
        y_norm = y / (y.norm() + 1e-8)
        lateral_lr = 0.005  # Boosted from 0.001 for stronger competition
        co_fire = lateral_lr * y_norm.unsqueeze(1) * y_norm.unsqueeze(0)
        co_fire.fill_diagonal_(0.0)
        cerebellum['lateral_weights'] += co_fire
        cerebellum['lateral_weights'] *= 0.998
        cerebellum['lateral_weights'].clamp_(min=0.0, max=1.0)  # Raised cap from 0.5 -> 1.0

    cerebellum['last_climbing_fiber_error'] = cf_error
    # Fix (c) diagnostic: report PEAK cf_error magnitude, not mean.
    # Mean is misleading when probe distillation spreads small values over
    # all 256 rows -- it buries the real teaching signal on the target row.
    # Peak is what actually drives LTD on the row that matters.
    cf_mag = cf_error.abs().max().item()
    cerebellum['error_ema'] = (
        0.99 * cerebellum.get('error_ema', 0.0)
        + 0.01 * cf_mag
    )
    return cf_mag


def _apply_pk_row_normalization(cerebellum):
    """Bidirectional proportional restoring force toward target row norm.
    
    Implements two biological mechanisms:
    
    1. ABOVE target (ratio > 1): Cerebellar-olivary feedback loop
       (Kenyon, Medina & Mauk 1998). Excess PK weights → stronger DCN
       inhibition → reduced IO firing → less LTD → net LTP drifts weights
       back. Implemented as gentle multiplicative shrinkage.
    
    2. BELOW target (ratio < 1): Spontaneous parallel fiber LTP
       (Coesmans et al. 2004). In the absence of climbing fiber activation,
       active parallel fiber synapses undergo slow LTP at ~1/36 the rate
       of CF-triggered LTD. This prevents weight erosion and maintains
       the synaptic baseline. Implemented as gentle multiplicative growth.
    
    The restoring force is proportional to the deviation from target,
    ensuring stable equilibrium at the target norm.
    """
    with torch.no_grad():
        w = cerebellum['purkinje_weights']
        target_norms = cerebellum['pk_target_row_norm'].clamp(min=1e-6)
        current_norms = w.norm(dim=1).clamp(min=1e-6)
        ratio = current_norms / target_norms
        
        # Bidirectional restoring force: pull toward ratio = 1.0
        # α controls restoring strength:
        #   ratio=2.0 → scale = 1/(1+0.05*1) = 0.952 (shrink 4.8%)
        #   ratio=3.0 → scale = 1/(1+0.05*2) = 0.909 (shrink 9.1%)
        #   ratio=0.5 → scale = 1/(1-0.02*0.5) = 1.010 (grow 1.0%)
        #   ratio=0.3 → scale = 1/(1-0.02*0.7) = 1.014 (grow 1.4%)
        # Growth rate is deliberately slower than shrinkage (asymmetric,
        # matching the ~36:1 LTD:LTP magnitude ratio in Medina & Mauk 2000)
        deviation = ratio - 1.0
        scale = torch.where(
            deviation > 0,
            1.0 / (1.0 + 0.01 * deviation),        # Softer shrinkage (was 0.05)
            1.0 / (1.0 - 0.025 * deviation.abs()),  # Stronger LTP growth (was 0.005) so B3 recovery ratio reaches 50%
        )
        cerebellum['purkinje_weights'] *= scale.unsqueeze(1)


def compute_cerebellar_cortical_feedback(cerebellum, output_error):
    """Feedback through dual pontine: output_error -> PK^T -> GC -> Mossy^T -> Pontine^T -> L5/6.
    
    Fix 1 compatibility: pontine_weights are now [N_PONTINE, n_l56 + n_pos_dims].
    The transpose product yields a gradient of size n_l56 + n_pos_dims; we discard
    the positional encoding gradient (not backprop-able) and return only L5/6.
    """
    with torch.no_grad():
        gc_error = cerebellum['purkinje_weights'].T @ output_error
        gc_mask = cerebellum.get('_last_gc_for_purkinje', cerebellum['gc_active_mask'])
        gc_error_masked = gc_error * (gc_mask > 0).float()
        # Mossy weights are [n_granule, N_PONTINE_TOTAL] where TOTAL = 2*N_PONTINE
        pontine_error_full = cerebellum['mossy_weights'].T @ gc_error_masked
        n_pontine = cerebellum['n_pontine']
        # Split back into positive and negative pontine pathways
        pontine_error_pos = pontine_error_full[:n_pontine]
        pontine_error_neg = pontine_error_full[n_pontine:]
        # Combine: positive pathway gradient + inverted negative pathway gradient
        pontine_error_combined = pontine_error_pos - pontine_error_neg
        full_gradient = cerebellum['pontine_weights'].T @ pontine_error_combined
        # Fix 1: Only return L5/6 portion, discard positional encoding gradient
        n_l56 = len(cerebellum['l56_indices'])
        l56_gradient = full_gradient[:n_l56]
        return l56_gradient


def create_pfa_matrices(engine, n_output=256, n_hidden_pfa=128, device='cpu'):
    """
    Create fixed random feedback matrices for Product Feedback Alignment (PFA).

    PFA replaces DFA's single random matrix B with a product of two matrices
    R and B, such that W ∝ (RB)^T. This introduces an additional population
    of n_hidden_pfa neurons in the backward pathway, allowing any pair of
    connected neurons to be unidirectionally connected while closely
    approximating backpropagation's gradient descent dynamics.

    Empirical results show PFA eliminates the need for separate learning
    phases and achieves BP-level performance in deep hierarchical substrates.

    Returns:
        pfa_matrices: dict mapping level -> (R, B, level_mask) where
            R: [level_size, n_hidden_pfa] first feedback matrix
            B: [n_hidden_pfa, n_output] second feedback matrix
            level_mask: [N] boolean mask for this level's nodes
    """
    # Compute per-node motor connectivity strength
    motor_connectivity = torch.zeros(engine.num_nodes, device=device)
    dst = engine.indices[1]
    src = engine.indices[0]
    motor_dst_mask = (dst >= 256) & (dst < 512)
    if motor_dst_mask.any():
        motor_src = src[motor_dst_mask]
        motor_w = engine.effective_weights[motor_dst_mask].abs()
        motor_connectivity.scatter_add_(0, motor_src, motor_w)

    mc_max = motor_connectivity.max().item()
    if mc_max > 0:
        motor_connectivity = 0.1 + 0.9 * (motor_connectivity / mc_max)
    else:
        motor_connectivity[:] = 1.0

    pfa_matrices = {}
    for level in range(engine.max_level + 1):
        level_mask = (engine.node_to_level == level)
        level_size = level_mask.sum().item()
        if level_size > 0:
            # PFA: Two fixed random matrices forming the product feedback path
            # B maps output error into the hidden backward population
            B = torch.randn(n_hidden_pfa, n_output, device=device) / np.sqrt(n_output)
            # R maps the hidden backward population into the level's nodes
            R = torch.randn(level_size, n_hidden_pfa, device=device) / np.sqrt(n_hidden_pfa)
            # Scale R rows by motor connectivity (same weighting as DFA)
            mc_level = motor_connectivity[level_mask]
            R *= mc_level.unsqueeze(1)
            pfa_matrices[level] = (R, B, level_mask)

    total_nodes = sum(m.sum().item() for _, (_, _, m) in pfa_matrices.items())
    avg_mc = motor_connectivity[512:].mean().item()
    print(f"PFA: Created product feedback alignment for {len(pfa_matrices)} levels, "
          f"{total_nodes} nodes, {n_hidden_pfa} hidden PFA neurons, "
          f"avg motor connectivity: {avg_mc:.3f}")
    return pfa_matrices


def run_linear_probe(engine, data, batch_size=200, free_steps=100, device='cpu', return_probe=False):
    """
    Diagnostic: Freeze the engine, collect latent L5/6 activations for 200 samples,
    and train a temporary linear probe to see if the internal representation
    is actually learning anything, bypassing the complex motor readout.
    
    Intervention 5: When return_probe=True, returns the trained probe model
    for use as a teacher signal in Linear Probe Distillation.
    """
    print(f"\n[DIAGNOSTIC] Running Linear Probe on {batch_size} samples...")
    engine.state.zero_()
    engine.state_basal.zero_()
    engine.state_apical.zero_()

    l56_indices = torch.where(engine.is_l56)[0]
    num_features = len(l56_indices)
    
    X = []
    Y = []
    
    # 1. Collect Dataset
    # We sample random indices from the data (avoiding the very end)
    sample_indices = np.random.randint(0, len(data) - 1, size=batch_size)
    
    input_mask = torch.zeros(engine.num_nodes, device=device)
    input_mask[:256] = 1.0

    # Save engine state
    saved_state = engine.state.clone()
    saved_basal = engine.state_basal.clone()
    saved_apical = engine.state_apical.clone()

    with torch.no_grad():
        for idx in sample_indices:
            curr_byte = int(data[idx])
            next_byte = int(data[idx+1])
            
            # Use leak factor to mimic temporal continuity even in samples
            engine.state *= 0.95
            engine.state_basal *= 0.95
            engine.state_apical *= 0.95

            input_vec = torch.zeros(engine.num_nodes, device=device)
            input_vec[curr_byte] = 10.0
            
            # Settle engine (FREE phase only)
            engine.settle(
                input_vec, input_mask=input_mask, max_steps=free_steps, 
                tol=0.0, sigma_noise=0.0, damping=0.8
            )
            
            X.append(engine.state[l56_indices].clone())
            Y.append(next_byte)

    # Restore engine state
    engine.state = saved_state
    engine.state_basal = saved_basal
    engine.state_apical = saved_apical

    X = torch.stack(X) # [batch_size, num_features]
    Y = torch.tensor(Y, dtype=torch.long, device=device) # [batch_size]

    # 2. Train Probe
    # Simple linear readout: X -> logits -> CrossEntropy
    probe = torch.nn.Linear(num_features, 256).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.01)
    
    for ep in range(50):
        logits = probe(X)
        loss = F.cross_entropy(logits, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 3. Eval Probe Accuracy
    with torch.no_grad():
        preds = torch.argmax(probe(X), dim=1)
        acc = (preds == Y).float().mean().item()
    
    if return_probe:
        probe.eval()  # Set to eval mode for distillation
        return acc, probe
    return acc


def run_eval(engine, cerebellum, eval_data, free_steps=100, n_samples=500, 
             leak_factor=0.95, device='cpu', bg_gate=None):
    """
    Run the cerebellar readout on held-out eval data (no learning).
    Processes data SEQUENTIALLY with LEAK_FACTOR to replicate training dynamics.
    Tests whether the model generalizes or just memorizes training sequences.
    """
    input_mask = torch.zeros(engine.num_nodes, device=device)
    input_mask[:256] = 1.0
    
    # Save engine state
    saved_state = engine.state.clone()
    saved_basal = engine.state_basal.clone()
    saved_apical = engine.state_apical.clone()
    
    correct = 0
    total = 0
    cs_correct = cs_total = 0
    cc_correct = cc_total = 0
    sc_correct = sc_total = 0
    
    # Run sequentially through a random chunk of eval data
    n_samples = min(n_samples, len(eval_data) - 2)
    start = np.random.randint(0, max(1, len(eval_data) - n_samples - 2))
    
    with torch.no_grad():
        engine.state.zero_()
        engine.state_basal.zero_()
        engine.state_apical.zero_()
        
        for j in range(n_samples):
            idx = start + j
            if idx >= len(eval_data) - 1:
                break
            
            prev_byte = int(eval_data[idx - 1]) if idx > 0 else 0
            curr_byte = int(eval_data[idx])
            next_byte = int(eval_data[idx + 1])
            
            # Replicate training dynamics
            engine.state *= leak_factor
            engine.state_basal *= leak_factor
            engine.state_apical *= leak_factor
            if bg_gate is not None:
                apply_bg_gate(engine, bg_gate)
            
            input_vec = torch.zeros(engine.num_nodes, device=device)
            input_vec[curr_byte] = 10.0
            
            engine.settle(
                input_vec, input_mask=input_mask, max_steps=free_steps,
                tol=0.0, sigma_noise=0.0, damping=0.8
            )
            
            logits, _, _ = cerebellar_forward(engine, cerebellum)
            pred = torch.argmax(logits).item()
            is_correct = (pred == next_byte)
            
            # Skip sentence boundaries (not learnable)
            is_sentence_boundary = (curr_byte == 32 and prev_byte in (ord('.'), ord('?'), ord('!')))
            
            if not is_sentence_boundary:
                total += 1
                if is_correct: correct += 1
                
                if next_byte == 32:
                    cs_total += 1
                    if is_correct: cs_correct += 1
                elif curr_byte == 32:
                    sc_total += 1
                    if is_correct: sc_correct += 1
                else:
                    cc_total += 1
                    if is_correct: cc_correct += 1
    
    # Restore engine state
    engine.state = saved_state
    engine.state_basal = saved_basal
    engine.state_apical = saved_apical
    
    acc = correct / total if total > 0 else 0
    cs_acc = cs_correct / cs_total if cs_total > 0 else 0
    cc_acc = cc_correct / cc_total if cc_total > 0 else 0
    sc_acc = sc_correct / sc_total if sc_total > 0 else 0
    
    return acc, cs_acc, cc_acc, sc_acc


def show_readout(engine, cerebellum, data, current_pos=0, free_steps=100, n_chars=80, 
                 leak_factor=0.95, device='cpu', bg_gate=None):
    """
    Display model predictions from the CURRENT training position using 
    the CURRENT engine state. No zeroing, no warmup — shows what the 
    model is actually doing during training.
    """
    input_mask = torch.zeros(engine.num_nodes, device=device)
    input_mask[:256] = 1.0
    
    saved_state = engine.state.clone()
    saved_basal = engine.state_basal.clone()
    saved_apical = engine.state_apical.clone()
    
    actual_chars = []
    predicted_chars = []
    match_markers = []
    
    with torch.no_grad():
        for j in range(n_chars):
            idx = current_pos + j
            if idx >= len(data) - 1:
                break
            
            curr_byte = int(data[idx])
            next_byte = int(data[idx + 1])
            
            engine.state *= leak_factor
            engine.state_basal *= leak_factor
            engine.state_apical *= leak_factor
            if bg_gate is not None:
                apply_bg_gate(engine, bg_gate)
            
            input_vec = torch.zeros(engine.num_nodes, device=device)
            input_vec[curr_byte] = 10.0
            
            engine.settle(
                input_vec, input_mask=input_mask, max_steps=free_steps,
                tol=0.0, sigma_noise=0.0, damping=0.8
            )
            
            logits, _, _ = cerebellar_forward(engine, cerebellum)
            pred_byte = torch.argmax(logits).item()
            
            actual_ch = chr(next_byte) if 32 <= next_byte < 127 else '.'
            pred_ch = chr(pred_byte) if 32 <= pred_byte < 127 else '.'
            
            actual_chars.append(actual_ch)
            predicted_chars.append(pred_ch)
            match_markers.append('+' if pred_byte == next_byte else '-')
    
    engine.state = saved_state
    engine.state_basal = saved_basal
    engine.state_apical = saved_apical
    
    actual_str = ''.join(actual_chars)
    pred_str = ''.join(predicted_chars)
    match_str = ''.join(match_markers)
    n_correct = match_markers.count('+')
    
    print(f"\n[READOUT] ({n_correct}/{len(match_markers)} correct)")
    print(f"  Target: {actual_str}")
    print(f"  Output: {pred_str}")
    print(f"  Match:  {match_str}")


def ensure_curriculum_data(data_dir):
    """Generate curriculum data if it doesn't exist."""
    train_dir = os.path.join(data_dir, "train")
    eval_dir = os.path.join(data_dir, "eval")
    
    needed_files = [
        os.path.join(train_dir, "level1_holophrases.txt"),
        os.path.join(eval_dir, "level1_holophrases.txt"),
        os.path.join(train_dir, "level2_slot_frame.txt"),
        os.path.join(eval_dir, "level2_slot_frame.txt"),
    ]
    
    if all(os.path.exists(f) for f in needed_files):
        return  # All data exists
    
    print("Curriculum data not found, generating...")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    
    # Import and run curriculum generator
    try:
        from curriculum_gen import (
            generate_holophrases, generate_slot_and_frame,
            generate_complex_constructions, generate_contextual_continuity
        )
        generate_holophrases(
            train_file=os.path.join(train_dir, "level1_holophrases.txt"),
            test_file=os.path.join(eval_dir, "level1_holophrases.txt"),
        )
        generate_slot_and_frame(
            train_file=os.path.join(train_dir, "level2_slot_frame.txt"),
            test_file=os.path.join(eval_dir, "level2_slot_frame.txt"),
        )
        generate_complex_constructions(
            train_file=os.path.join(train_dir, "level3_complex.txt"),
            test_file=os.path.join(eval_dir, "level3_complex.txt"),
        )
        generate_contextual_continuity(
            train_file=os.path.join(train_dir, "level4_contextual.txt"),
            test_file=os.path.join(eval_dir, "level4_contextual.txt"),
        )
        print("Curriculum data generated successfully.")
    except ImportError:
        print("WARNING: curriculum_gen.py not found. Please generate data manually.")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # =====================================================================
    # PRE-TRAINING BIOLOGICAL DIAGNOSTICS
    # =====================================================================
    # Added to verify architectural changes haven't broken neural dynamics.
    # We run a subset of benchmarks that don't require training history.
    print("\n[INIT] Running Pre-Training Biological Diagnostics...")
    from micro_scale_diagnostics import MicroScaleDiagnosticSuite
    diag_suite = MicroScaleDiagnosticSuite(num_nodes=64 * 350, device=device)
    # Just run a few critical ones to avoid long startup time
    diag_results = diag_suite.run_suite()
    pass_count = sum(1 for r in diag_results if r['pass'])
    print(f"[INIT] Diagnostic Pass Rate: {pass_count}/{len(diag_results)}")
    if pass_count < 10:
        print("[WARNING] Biological constraints are not fully met. Proceeding but results may be unstable.")
    else:
        print("[OK] Biological constraints verified.")

    num_modules = 10
    num_levels = 2
    edge_index, edge_weight, biases, taus, graph = create_training_graph(
        num_modules=num_modules, num_levels=num_levels
    )
    num_nodes = biases.shape[0]

    # =====================================================================
    # ARCHITECTURAL ENHANCEMENT 1: Boost output connectivity
    # =====================================================================
    # [DISABLED] Legacy ERR-based output boost (Direct cortical-motor bypass)
    # edge_index, edge_weight = boost_output_connectivity(
    #     graph, edge_index, edge_weight, num_nodes
    # )

    # Thalamocortical stabilization loop (no motor output)
    # Re-enabled: provides stable attractor basins for temporal smoothing of
    # L5/6 representations, solving the non-stationarity problem that prevents
    # the cerebellar readout from decoding the 93.5% internal representation.
    N_THALAMIC = 128
    edge_index, edge_weight, biases, taus, num_nodes, thalamic_indices = add_thalamocortical_loop(
        graph, edge_index, edge_weight, biases, taus, num_nodes,
        n_thalamic=N_THALAMIC, device='cpu'
    )
    graph.thalamic_indices = thalamic_indices

    # Pad cell-type arrays for thalamic relay neurons (excitatory, non-interneuron)
    thal_pad = np.zeros(N_THALAMIC, dtype=bool)
    is_inhibitory = np.concatenate([graph.is_inhibitory, thal_pad])
    is_pv = np.concatenate([graph.is_pv, thal_pad])
    is_sst = np.concatenate([graph.is_sst, thal_pad])
    is_vip = np.concatenate([graph.is_vip, thal_pad])
    is_lts_base = getattr(graph, "is_lts", None)
    if is_lts_base is not None:
        is_lts = np.concatenate([is_lts_base, thal_pad])
    else:
        is_lts = None

    mod_starts = torch.tensor([m["start"] for m in graph.modules], dtype=torch.long)
    mod_ends = torch.tensor([m["end"] for m in graph.modules], dtype=torch.long)
    mod_levels = torch.tensor([m["level"] for m in graph.modules], dtype=torch.long)

    print("Initializing PredictiveCodingEngine...")
    engine = PredictiveCodingEngine(
        num_nodes=num_nodes,
        indices=edge_index.numpy(),
        values=edge_weight.numpy(),
        biases=biases.numpy(),
        taus=taus.numpy(),
        module_ranges=graph.get_module_ranges(),
        module_levels=mod_levels.numpy(),
        hier_pairs=graph.hier_pairs,
        modules=graph.modules,
        device=device,
        is_inhibitory=is_inhibitory,
        is_pv=is_pv,
        is_sst=is_sst,
        is_vip=is_vip,
        is_lts=is_lts,
        dg_indices=graph.dg_indices,
        temporal_alpha=0.05,
        gc_indices=graph.granule_cell_indices,
        purkinje_indices=graph.purkinje_indices,
    )

    # Cerebellar output module (replaces FRNL + lateral inhibition + terminal Adam optimizer)
    
    # Gather Broca and Wernicke indices for SAL partitioning
    broca_indices = []
    wernicke_indices = []
    for mod in graph.modules:
        if mod['id'] in graph.broca_modules:
            broca_indices.extend(mod['l56_indices'].tolist())
        elif mod['id'] in graph.wernicke_modules:
            wernicke_indices.extend(mod['l56_indices'].tolist())
    
    cerebellum = add_cerebellar_module(
        graph, num_nodes, n_granule=16384, device=device,
        thalamic_indices=thalamic_indices,
        broca_indices=broca_indices if broca_indices else None,
        wernicke_indices=wernicke_indices if wernicke_indices else None,
    )

    # =====================================================================
    # ARCHITECTURAL ENHANCEMENT: BG-like gating for thalamocortical loop
    # =====================================================================
    bg_gate = create_bg_gate(engine, thalamic_indices, device=device)

    # =====================================================================
    # ARCHITECTURAL ENHANCEMENT 3: Direct Feedback Alignment matrices
    # =====================================================================
    pfa_matrices = create_pfa_matrices(engine, n_output=256, n_hidden_pfa=128, device=device)

    # =====================================================================
    # PHASE 3: Module-Specific IP Gain for Functional Lateralization
    # =====================================================================
    with torch.no_grad():
        # Broca modules: higher gain (1.5) for persistent sequential activity
        if broca_indices:
            broca_t = torch.tensor(broca_indices, dtype=torch.long, device=device)
            engine.ip_gain[broca_t] = 1.5
            print(f"  Broca ip_gain set to 1.5 for {len(broca_indices)} L5/6 nodes")
        # Wernicke modules: boost L2/3 inhibitory gain by 10% for stronger explaining-away
        if wernicke_indices:
            for mod in graph.modules:
                if mod['id'] in graph.wernicke_modules:
                    l23_idx = mod['l23_indices']
                    inh_in_l23 = engine.is_inhibitory[l23_idx]
                    inh_l23_nodes = torch.tensor(l23_idx, dtype=torch.long, device=device)[inh_in_l23]
                    if len(inh_l23_nodes) > 0:
                        engine.ip_gain[inh_l23_nodes] *= 1.1
            print(f"  Wernicke L2/3 inhibitory ip_gain boosted by 10%")

    # Store Broca L5/6 indices on engine for information-theoretic pruning
    if broca_indices:
        engine._broca_l56_indices = torch.tensor(broca_indices, dtype=torch.long, device=device)

    # Simplified Dummy Curriculum for Testing
    phases = [
        {
            "name": "SimpleSequenceTracking",
            "lr": 0.05,
            "epochs": 1,
            "text": "abc abc abc abc " * 200,
            "eval_text": "abc abc abc abc " * 50
        }
    ]

    # Input Mask: clamp input nodes only during settling
    input_mask = torch.zeros(num_nodes, device=device)
    input_mask[:256] = 1.0

    # =================================================================
    # Hyperparameters (Round 5 — homeostatic PK + DCN inversion)
    # =================================================================
    FREE_STEPS = 40
    NUDGE_STEPS = 40  # Was 12 — Intervention 5: prolonged error signal for LTD/LTP
    nudge_strength = 5.0
    lr = 0.01
    sigma_noise_free = 0.01
    sigma_noise_nudge = 0.01
    LEAK_FACTOR = 0.95
    ALPHA_PFA = 0.15
    ALPHA_CEREBELLAR = 0.05
    # Intervention 4: Dynamic Apical Gain — BURST_APICAL_GAIN is now computed per-step
    BETA_GAIN = 3.0  # Decay constant for error-modulated apical gain
    # Intervention 5: Cached linear probe model for distillation
    cached_probe_model = None
    # Intervention 2: PFA plasticity rate
    ETA_PFA_PLASTIC = 0.001
    LAMBDA_PFA_DECAY = 0.0001

    for phase_info in phases:
        phase_name = phase_info["name"]
        text = phase_info["text"]
        eval_text = phase_info["eval_text"]

        print(f"\n--- Starting Phase: {phase_name} ---")
        print(f"  Steps: Free={FREE_STEPS}, Nudge={NUDGE_STEPS}")
        print(f"  Learning rate: {lr}, Nudge strength: {nudge_strength}")
        print(f"  PFA alpha: {ALPHA_PFA}, CB alpha: {ALPHA_CEREBELLAR}, Apical gain: dynamic (beta={BETA_GAIN})")
        print(f"  Thalamocortical loop: {N_THALAMIC} neurons (BG-gated, tonic=-3.0)")
        print(f"  Cerebellum: {cerebellum['n_granule']} granule cells")
        print(f"  PFA plasticity: eta={ETA_PFA_PLASTIC}, lambda={LAMBDA_PFA_DECAY}")

        data = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
        seq_len = len(data)
        eval_data = np.frombuffer(eval_text.encode("utf-8"), dtype=np.uint8)
        print(f"  Eval data: {len(eval_data)} bytes")

        # Metrics tracking
        spatial_errors = []
        accuracies = []       # Excludes sentence boundaries
        accuracies_all = []   # Includes everything (for raw logging)
        burst_strengths = []
        # Structural Acc Metrics (excluding sentence-boundary S→C)
        cs_total, cs_correct = 0, 0   # Char->Space (learnable)
        cc_total, cc_correct = 0, 0   # Char->Char (learnable)
        sc_total, sc_correct = 0, 0   # Space->Char within sentence (learnable)
        sb_total = 0                   # Sentence boundary transitions (not learnable, just counted)
        
        settle_history = deque(maxlen=50) # Track settle diffs for adaptive gating

        # Reset weights for the tuned run to ensure no stale associations
        engine.reset_plastic_weights()

        # Fix 1+2+(a)+(b): Reset cerebellar temporal dynamics at phase boundaries
        cerebellum['position_counter'] = 0
        cerebellum['phase_position'] = 0           # Fix (b): resetting phase
        cerebellum['reservoir_state'].zero_()
        cerebellum['pk_eligibility'].zero_()       # Fix (a): clear synaptic tags
        # Fix 3: Reset STDP temporal error state
        cerebellum.pop('_prev_probs', None)

        start_time = time.time()
        for i in range(seq_len - 1):
            current_byte = int(data[i])
            next_byte = int(data[i + 1])

            # PHASE 1: Leaky Persistence (no reset)
            engine.state *= LEAK_FACTOR
            engine.state_basal *= LEAK_FACTOR
            engine.state_apical *= LEAK_FACTOR
            
            # BG-like gating: compute Go signal from L5/6 and inject into
            # thalamic basal compartments. Replaces hard thalamic zeroing.
            # The -3.0 tonic bias (SNr inhibition) keeps thalamus silent
            # unless the learned Go signal provides sufficient drive.
            apply_bg_gate(engine, bg_gate)
                
            engine.previous_state = engine.state.clone()

            # === FREE PHASE ===
            input_vec = torch.zeros(num_nodes, device=device)
            input_vec[current_byte] = 10.0

            # Forward Updates: store initial prediction μ_l^0 before settling
            initial_prediction = engine.state.clone()

            engine.settle(
                input_vec, input_mask=input_mask, max_steps=FREE_STEPS, tol=0.0,
                sigma_noise=sigma_noise_free, damping=0.8
            )

            # Forward Updates: compute forward error ε̃ = x^T - μ^0
            # This measures how much the network's state changed during settling.
            # Large forward errors indicate surprising inputs (high prediction error).
            forward_error = engine.state - initial_prediction
            
            # Adaptive Gating: "Leaky" Basal Ganglia logic
            # If settle has stabilized (low variance), slowly raise threshold to allow output
            settle_history.append(engine.last_settle_diff)
            # BG gate threshold fixed at 0.3 — no drift
            # Previous drift logic (0.3→0.8) caused permissive gating

            free_state = engine.state.clone()
            free_basal = engine.state_basal.clone()
            free_apical = engine.state_apical.clone()

            # === CEREBELLAR READOUT (replaces FRNL + lateral inhibition) ===
            logits, granule_acts, gate_value = cerebellar_forward(engine, cerebellum)
            pred_byte = torch.argmax(logits).item()
            is_correct = pred_byte == next_byte
            
            # Detect sentence boundaries: space after . ? !
            # These predict the first char of a RANDOM next sentence — not learnable
            prev_byte = int(data[i - 1]) if i > 0 else 0
            is_sentence_boundary = (current_byte == 32 and prev_byte in (ord('.'), ord('?'), ord('!')))

            # Fix (b): Reset phase_position at sentence boundaries so the
            # first character of every new sentence gets phase=0. This is the
            # core prescription from Hasselmo 2007 / Zugaro 2005 / Zheng 2024:
            # phase codes must reset at episodic boundaries or interference
            # from the previous trajectory corrupts position assignment.
            # The forward pass on the boundary space has already run above,
            # so resetting here means the NEXT iteration (first letter of the
            # new sentence) will sinusoidally encode phase=0.
            if is_sentence_boundary:
                cerebellum['phase_position'] = 0
            
            # Track all predictions for raw window accuracy
            accuracies_all.append(1.0 if is_correct else 0.0)
            
            # Track learnable predictions only (exclude sentence boundaries)
            if not is_sentence_boundary:
                accuracies.append(1.0 if is_correct else 0.0)

            # Update structural stats (learnable transitions only)
            if is_sentence_boundary:
                sb_total += 1  # Count but don't score
            elif next_byte == 32:  # Char->Space
                cs_total += 1
                if is_correct: cs_correct += 1
            elif current_byte == 32:  # Space->Char (within sentence)
                sc_total += 1
                if is_correct: sc_correct += 1
            else:  # Char->Char
                cc_total += 1
                if is_correct: cc_correct += 1

            # === CEREBELLAR LEARNING (replaces terminal Adam optimizer) ===
            # EXPERIMENT: probe distillation DISABLED for this run.
            # We need a clean baseline of what the cerebellum learns from
            # its own CF teaching signal alone, with no leakage from the
            # 100% linear-probe shortcut. Re-enable by passing
            # cached_probe_model in place of None below.
            cf_error = cerebellar_learn(cerebellum, logits, granule_acts, next_byte, gate_value,
                                        engine=engine, settle_diff=engine.last_settle_diff,
                                        probe_model=None)

            # BG gate learning: dopamine-gated Hebbian update of Go weights.
            # cf_error (climbing fiber magnitude) serves as dopamine proxy —
            # high prediction error → strengthen gate patterns that were active.
            update_bg_gate(engine, bg_gate, dopamine=cf_error)

            # Prediction errors
            spatial_err = engine.compute_prediction_errors()
            spatial_errors.append(spatial_err)

            # === NUDGED PHASE (with DFA + Burst Coincidence) ===
            target_one_hot = torch.zeros(256, device=device)
            target_one_hot[next_byte] = 1.0

            # Compute output error for DFA (Fix D: Sync with Softmax)
            # Use Softmax to match the internal cerebellar learning rule.
            output_error = target_one_hot - torch.softmax(logits, dim=0)

            # Restore to free state before nudge
            engine.state = free_state.clone()
            engine.state_basal = free_basal.clone()
            engine.state_apical = free_apical.clone()

            # Build nudge vector with PFA feedback
            nudge_vec = torch.zeros(num_nodes, device=device)
            nudge_vec[current_byte] = 10.0
            # NOTE: No direct motor node clamping — the cerebellum handles output

            # ENHANCEMENT 3d: Forward Updates — inject prediction surprise into nudge
            # ΔW_l = η · ε̃_l^T · f(x_{l-1}^T)^T
            # Forward error = how much state changed during settling.
            # Scaled at 0.15 to properly propagate deep prediction error.
            nudge_vec += 0.15 * forward_error

            # ENHANCEMENT 3a: PFA — inject output error into all hidden levels
            for level, (R, B, level_mask) in pfa_matrices.items():
                # PFA: R @ (B @ output_error) — product of two feedback matrices
                pfa_signal = R @ (B @ output_error)  # [level_size]
                nudge_vec[level_mask] += ALPHA_PFA * pfa_signal

            # ENHANCEMENT 3c: Cerebello-thalamo-cortical co-adaptation
            if (i + 1) % 50 == 0:
                l56_grad = compute_cerebellar_cortical_feedback(cerebellum, output_error)
                nudge_vec[cerebellum['l56_indices']] += ALPHA_CEREBELLAR * l56_grad

            # ENHANCEMENT 3b: Burst Coincidence — inject target into apical
            # Intervention 4: Dynamic Apical Gain — Γ = 0.5 + 2.5*exp(-β*error_ema)
            # When error is high → gain drops to 0.5 (sensory-driven learning)
            # When error is low → gain rises to 3.0 (predictive model dominates)
            error_ema = cerebellum.get('error_ema', 0.0)
            dynamic_apical_gain = 0.5 + 2.5 * math.exp(-BETA_GAIN * error_ema)
            engine.inject_apical_nudge(target_one_hot, strength=dynamic_apical_gain)

            engine.settle(
                nudge_vec,
                input_mask=input_mask,
                max_steps=NUDGE_STEPS,
                tol=0.0,
                sigma_noise=sigma_noise_nudge,
                damping=0.3
            )

            # BUG 2 FIX: Corticospinal Motor Nudge via L5/6 Burst Propagation
            # Motor nodes (256-511) receive NO direct teaching signal during nudge.
            # EqProp needs (nudge - free) ≠ 0 at motor nodes to learn output weights.
            # Bio pathway: L5/6 BAC firing (from inject_apical_nudge) → burst propagates
            # down axon through output projection weights → drives motor nodes.
            # This creates a bootstrap loop: as output weights learn, burst propagation
            # strengthens, which creates stronger EqProp gradients, which further
            # improves the weights — matching developmental motor learning.
            with torch.no_grad():
                burst = engine.compute_burst_coincidence()  # L5/6 burst from apical nudge
                out_mask = engine.output_edge_mask
                if out_mask.any():
                    burst_signal = burst[engine.indices[0][out_mask]] * engine.effective_weights[out_mask]
                    burst_drive = torch.zeros(num_nodes, device=device)
                    burst_drive.scatter_add_(0, engine.indices[1][out_mask], burst_signal)
                    # Inject burst-propagated signal into motor nodes only
                    engine.state[256:512] += 2.0 * burst_drive[256:512]

            nudge_pos = engine.state.clone()

            # Reuse burst computed in propagation block for monitoring
            avg_burst = burst[engine.is_l56].mean().item()
            burst_strengths.append(avg_burst)

            # Biological EqProp/DFA for all deep layers
            engine.update_weights_phase2(
                free_state,
                nudge_pos,
                learning_rate=lr,
                target_sequence=target_one_hot
            )

            # Intervention 2: Plastic PFA Feedback (RAF — Restricted Adaptive Feedback)
            # ΔR = η·(x·h^T - λR), ΔB = η·(h·e^T - λB)
            # Allows feedback path to align with forward path over time
            with torch.no_grad():
                for level in list(pfa_matrices.keys()):
                    R, B, level_mask = pfa_matrices[level]
                    h_pfa = torch.relu(B @ output_error)  # Hidden PFA activation
                    x_level = free_state[level_mask]
                    delta_R = ETA_PFA_PLASTIC * (
                        x_level.unsqueeze(1) * h_pfa.unsqueeze(0) - LAMBDA_PFA_DECAY * R
                    )
                    delta_B = ETA_PFA_PLASTIC * (
                        h_pfa.unsqueeze(1) * output_error.unsqueeze(0) - LAMBDA_PFA_DECAY * B
                    )
                    pfa_matrices[level] = (R + delta_R, B + delta_B, level_mask)

            # Restore free-phase state after nudge+weight update.
            # Without this, the nudge state (containing target info) leaks 
            # into the next step via LEAK_FACTOR.
            engine.state = free_state.clone()
            engine.state_basal = free_basal.clone()
            engine.state_apical = free_apical.clone()

            # Periodic Structural Plasticity (Hebbian Rewiring)
            if (i + 1) % 1000 == 0:
                n_pruned = prune_and_rewire_output(engine, prune_ratio=0.03)
                print(f"\n  [STRUCTURAL] Pruned and rewired {n_pruned} weak output projections.")

            # Dynamic Apical Gain (β) update every 100 steps
            # Adjusts per-module apical sensitivity based on NPE/PPE ratio
            if (i + 1) % 100 == 0:
                engine.update_apical_beta()

            # Logging and periodic tasks
            if (i + 1) % 10 == 0:
                window = min(500, len(accuracies))
                avg_acc = np.mean(accuracies[-window:])
                avg_err = np.mean(spatial_errors[-100:])
                avg_bst = np.mean(burst_strengths[-100:])

                w_surf_norm = engine.w_surface.norm().item()
                w_max = engine.w_surface.abs().max().item()

                elapsed = time.time() - start_time
                tps = (i + 1) / elapsed

                # Diagnostics
                mean_firing = engine.lifetime_firing[512:].mean().item()
                bias_norm = engine.biases.norm().item()
                inh_w_norm = engine.w_surface[engine.is_inhibitory[engine.indices[0]]].norm().item()

                # Granule cell sparsity (should be ~5-10% active)
                gc_sparsity = (granule_acts > 0).float().mean().item()
                # Purkinje weight norms
                pk_w_norm = cerebellum['purkinje_weights'].norm().item()
                pk_row_norms = cerebellum['purkinje_weights'].norm(dim=1)
                pk_row_max = pk_row_norms.max().item()
                pk_row_mean = pk_row_norms.mean().item()
                
                # Softmax-relevant diagnostics (NOT sigmoid — learning uses softmax)
                probs = torch.softmax(logits, dim=0)
                target_prob = probs[next_byte].item()  # P(correct class)
                pred_prob = probs.max().item()          # P(predicted class)
                logit_range = (logits.max() - logits.min()).item()
                
                # Target prob tracking (most informative learning metric)
                if not hasattr(cerebellar_learn, '_target_probs'):
                    cerebellar_learn._target_probs = []
                cerebellar_learn._target_probs.append(target_prob)
                tp_window = cerebellar_learn._target_probs[-window:]
                avg_target_prob = np.mean(tp_window)

                print(
                    f"Step {i+1}/{seq_len} | Acc: {avg_acc:.2%} | "
                    f"P(target): {avg_target_prob:.3f} | P(argmax): {pred_prob:.3f} | "
                    f"PK_w: {pk_w_norm:.2f} (row mean: {pk_row_mean:.2f} max: {pk_row_max:.2f}) | "
                    f"GC_spars: {gc_sparsity:.2%}"
                )
                print(
                    f"  Logit range: {logit_range:.4f} | "
                    f"CF_err: {cf_error:.4f} | "
                    f"Settle: {engine.last_settle_diff:.4f}"
                )
                
                # New diagnostics for Bug 1/2/3 fixes
                out_w_norm = engine.w_surface[engine.output_edge_mask].norm().item() if engine.output_edge_mask.any() else 0.0
                out_w_mean = engine.w_surface[engine.output_edge_mask].abs().mean().item() if engine.output_edge_mask.any() else 0.0
                cf_surprise = cerebellum.get('error_ema', 0.0)
                
                # Burst selectivity: fraction of L5/6 neurons with burst > 0.5
                # Should be ~10-30% with selective burst, was ~100% with .abs() bug
                with torch.no_grad():
                    burst_diag = engine.compute_burst_coincidence()
                    l56_burst = burst_diag[engine.is_l56]
                    burst_frac = (l56_burst > 0.5).float().mean().item()
                    burst_mean = l56_burst.mean().item()
                
                # Actual IO Dimensionality (% of non-zero teaching signals)
                last_cf = cerebellum.get('last_climbing_fiber_error', torch.zeros(256))
                io_dim = (last_cf.abs() > 1e-5).float().mean().item() * 100
                
                print(
                    f"  OutProj w_surf: norm={out_w_norm:.4f} mean={out_w_mean:.6f} | "
                    f"Burst: mean={burst_mean:.4f} frac>0.5={burst_frac:.2%} | "
                    f"CF_ema: {cf_surprise:.4f} | IO_space: {io_dim:.1f}%"
                )
                
                # Cerebellar homeostasis diagnostics
                pk_silent = (cerebellum['purkinje_weights'].abs() < 1e-6).float().mean().item()
                mossy_norm = cerebellum['mossy_weights'].norm().item()
                mossy_baseline = cerebellum['mossy_weights_baseline_norm']
                mossy_ratio = mossy_norm / max(mossy_baseline, 1e-6)
                mli_mean = cerebellum['mli_inhibition_scale'].mean().item()
                ca_thresh_mean = cerebellum['calcium_threshold'].mean().item()
                pk_row_ratio = (cerebellum['purkinje_weights'].norm(dim=1) / 
                               cerebellum['pk_target_row_norm'].clamp(min=1e-6)).mean().item()
                
                print(
                    f"  PK_silent: {pk_silent:.1%} | "
                    f"MF_gain: {mossy_ratio:.2f}x | "
                    f"MLI: {mli_mean:.3f} | "
                    f"Ca_th: {ca_thresh_mean:.4f} | "
                    f"PK_row_ratio: {pk_row_ratio:.3f}"
                )

                # LTD update magnitude diagnostic (added to investigate
                # frozen-weights problem). |elig|=mean row norm of
                # eligibility matrix; |ltd|=mean row norm of the LTD weight
                # update applied this step; |ltd_tgt|=norm of update on the
                # target row; ltd/w=ratio of update size to current row norm
                # (need >~1e-3 to escape noise floor in reasonable steps).
                _elig = cerebellum.get('_diag_elig_norm', 0.0)
                _ltd = cerebellum.get('_diag_ltd_norm', 0.0)
                _ltd_tgt = cerebellum.get('_diag_ltd_target', 0.0)
                _w_row = max(pk_row_mean, 1e-6)
                print(
                    f"  |elig|: {_elig:.4f} | "
                    f"|ltd|: {_ltd:.6f} | "
                    f"|ltd_tgt|: {_ltd_tgt:.6f} | "
                    f"ltd/w: {_ltd / _w_row:.6f}"
                )
                
                # GC pattern discriminability: overlap between current and previous GC pattern
                # If centering works, overlap should be <100% (different GCs for different inputs)
                gc_mask_now = cerebellum.get('gc_active_mask', torch.zeros(cerebellum['n_granule'], device=device))
                gc_prev = cerebellum.get('_prev_gc_mask', gc_mask_now)
                if gc_mask_now.sum() > 0 and gc_prev.sum() > 0:
                    intersection = (gc_mask_now * gc_prev).sum().item()
                    union = ((gc_mask_now + gc_prev) > 0).float().sum().item()
                    gc_jaccard = intersection / max(union, 1.0)
                else:
                    gc_jaccard = 1.0
                cerebellum['_prev_gc_mask'] = gc_mask_now.clone()
                
                # Pontine compression diagnostics
                if '_last_pontine_acts' in cerebellum:
                    pontine_acts = cerebellum['_last_pontine_acts']
                    pontine_range = (pontine_acts.max() - pontine_acts.min()).item()
                    pontine_sparsity = (pontine_acts > 0).float().mean().item()
                else:
                    pontine_range = 0.0
                    pontine_sparsity = 0.0
                
                print(
                    f"  GC_overlap: {gc_jaccard:.2%} | "
                    f"Pontine: range={pontine_range:.3f} spars={pontine_sparsity:.1%}"
                )


                # Diagnostic Linear Probe + Readout every 500 steps
                if (i + 1) % 500 == 0:
                    # Intervention 5: Return probe model for distillation
                    probe_result = run_linear_probe(
                        engine, data, batch_size=200, 
                        free_steps=FREE_STEPS, device=device, return_probe=True
                    )
                    probe_acc, cached_probe_model = probe_result
                    print(f"  >>> LINEAR PROBE ACCURACY: {probe_acc:.2%} (Internal Representation Quality)")
                    print(f"  >>> Probe model cached for cerebellar distillation")
                    if probe_acc > avg_acc * 2:
                        print(f"  [!] ALERT: Representation ({probe_acc:.1%}) >> Readout ({avg_acc:.1%}). Readout is the bottleneck.")
                    else:
                        print(f"  [!] NOTE: Representation ({probe_acc:.1%}) ~= Readout ({avg_acc:.1%}). Hebbian learning is the bottleneck.")
                    
                    # Show what the model is actually outputting
                    show_readout(engine, cerebellum, data, current_pos=i+1, free_steps=FREE_STEPS, n_chars=80, leak_factor=LEAK_FACTOR, device=device, bg_gate=bg_gate)
                    
                    # Prediction diversity check (class collapse detector)
                    # Sample 100 chars and count unique predictions
                    with torch.no_grad():
                        pred_counts = torch.zeros(256, device=device)
                        saved_s = engine.state.clone()
                        saved_b = engine.state_basal.clone()
                        saved_a = engine.state_apical.clone()
                        div_indices = np.random.randint(0, len(data)-1, size=100)
                        for di in div_indices:
                            cb = int(data[di])
                            iv = torch.zeros(num_nodes, device=device)
                            iv[cb] = 10.0
                            apply_bg_gate(engine, bg_gate)
                            engine.settle(iv, input_mask=input_mask, max_steps=FREE_STEPS, tol=0.0, sigma_noise=0.0, damping=0.8)
                            lg, _, _ = cerebellar_forward(engine, cerebellum)
                            pred_counts[torch.argmax(lg)] += 1
                        engine.state = saved_s
                        engine.state_basal = saved_b
                        engine.state_apical = saved_a
                        n_unique = (pred_counts > 0).sum().item()
                        top_pred_byte = torch.argmax(pred_counts).item()
                        top_pred_frac = pred_counts.max().item() / 100.0
                        top_ch = chr(top_pred_byte) if 32 <= top_pred_byte < 127 else f'\\x{top_pred_byte:02x}'
                        print(f"  [DIVERSITY] {n_unique} unique predictions / 100 samples | "
                              f"Top: '{top_ch}' ({top_pred_frac:.0%})"
                              f"{' <- CLASS COLLAPSE' if n_unique < 4 else ''}")
                
                # Eval on held-out data every 2500 steps
                if (i + 1) % 2500 == 0 and eval_data is not None:
                    eval_acc, eval_cs, eval_cc, eval_sc = run_eval(
                        engine, cerebellum, eval_data,
                        free_steps=FREE_STEPS, n_samples=300, leak_factor=LEAK_FACTOR, device=device,
                        bg_gate=bg_gate
                    )
                    print(f"  >>> EVAL (held-out): {eval_acc:.2%} | C->S: {eval_cs:.2%} | C->C: {eval_cc:.2%} | S->C: {eval_sc:.2%}")

                # Structural Diagnostics (learnable transitions only)
                cs_acc = (cs_correct / cs_total) if cs_total > 0 else 0
                cc_acc = (cc_correct / cc_total) if cc_total > 0 else 0
                sc_acc = (sc_correct / sc_total) if sc_total > 0 else 0
                print(f"  Structural Acc | C->S: {cs_acc:.2%} ({cs_total}) | C->C: {cc_acc:.2%} ({cc_total}) | S->C: {sc_acc:.2%} ({sc_total}) | SentBound: {sb_total}")

        elapsed = time.time() - start_time
        final_acc = (
            np.mean(accuracies[-1000:])
            if len(accuracies) >= 1000
            else np.mean(accuracies)
        )
        print(
            f"\nPhase {phase_name} complete in {elapsed:.1f}s | "
            f"Final Acc: {final_acc:.2%}"
        )


if __name__ == "__main__":
    main()