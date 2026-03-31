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

    # Default biases and taus
    biases = torch.zeros(total_nodes, dtype=torch.float32)
    taus = torch.ones(total_nodes, dtype=torch.float32) * 5.0  # default tau

    return edge_index, edge_weight, biases, taus, graph


def boost_output_connectivity(graph, edge_index, edge_weight, num_nodes):
    """
    Add additional output projections from ALL modules' L5/6 to motor nodes.
    The original graph only projects from Level 0 L5/6 → motor. This adds
    projections from ALL levels, giving the output nodes broader visibility.
    """
    n_sensory = 256
    n_motor = 256

    new_rows = []
    new_cols = []

    for mod in graph.modules:
        # Skip level 0 modules (already have output projections)
        if mod['level'] == 0:
            continue
        idx = mod['l56_indices']
        if len(idx) == 0:
            continue
        # Sparse projection: 20% of L5/6 nodes per output (was 10%)
        # Increased for broader visibility of abstract features
        n_proj = max(1, len(idx) // 5)
        for m in range(n_motor):
            sources = np.random.choice(idx, n_proj, replace=True)
            motor_node = n_sensory + m
            new_rows.extend(sources.tolist())
            new_cols.extend([motor_node] * n_proj)

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
        print(f"Output boost: added {len(new_rows)} new output projections from higher-level modules")

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

    # Extend biases: TONIC INHIBITION (basal ganglia default state)
    # Relay neurons are silent by default, only fire when gate opens.
    # We bypass the loop for sequence generation by setting deep tonic inhibition (-50.0).
    thal_biases = torch.ones(n_thalamic, dtype=torch.float32) * -50.0  # Deep tonic inhibition
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

    # NO thalamic → motor projections (that's the cerebellum's job now)
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

    print(f"Thalamocortical loop: {n_thalamic} relay neurons (stabilization only, no motor output)")
    thalamic_indices = np.arange(thal_start, thal_start + n_thalamic, dtype=np.int64)
    return edge_index, edge_weight, biases, taus, new_num_nodes, thalamic_indices


def add_cerebellar_module(graph, num_nodes, n_granule=16384, sparsity=0.05, device='cpu', 
                            thalamic_indices=None, broca_indices=None, wernicke_indices=None):
    """
    Cerebellar output module: Pontine compression + GC expansion + Purkinje readout.
    
    Biologically-constrained architecture:
    - Dual pontine pathway (positive + sign-inverted) for bidirectional encoding
    - K=4 mossy fiber inputs per GC (Billings 2014, Litwin-Kumar 2017)
    - Soft-bounded Purkinje weights (no hard cap)
    """
    N_PONTINE = 512
    # Fix 4: Dual pontine → 1024 effective mossy fiber sources
    N_PONTINE_TOTAL = N_PONTINE * 2  # Positive + sign-inverted pathways
    K_MOSSY = 4  # Biologically conserved dendrite count (Cayco-Gajic 2017)

    all_l56 = []
    for mod in graph.modules:
        all_l56.extend(mod['l56_indices'].tolist())
    all_l56 = np.array(all_l56, dtype=np.int64)
    n_l56 = len(all_l56)

    # Pontine: random projection (same weights used for both pos and neg pathways)
    pontine_weights = torch.randn(N_PONTINE, n_l56, device=device) * (1.0 / (n_l56 ** 0.5))

    # Fix 2: K=4 mossy fiber inputs per GC (biologically conserved)
    # Each GC randomly selects exactly K_MOSSY inputs from 1024 pontine neurons
    # This enables conjunctive coding: GC fires only when all 4 inputs are active
    mossy_weights = torch.zeros(n_granule, N_PONTINE_TOTAL, device=device)
    for g in range(n_granule):
        sel = np.random.choice(N_PONTINE_TOTAL, K_MOSSY, replace=False)
        mossy_weights[g, sel] = 1.0 / np.sqrt(K_MOSSY)  # Normalized excitatory

    # Signed PK weights (no non-negative constraint)
    purkinje_weights = torch.randn(256, n_granule, device=device) * (1.0 / np.sqrt(n_granule))

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
        'golgi_alpha': 0.01,
        'delta_lr': 0.005,  # Lower LR for sparse CF (was 0.02 — caused logit explosion)
        'gc_target_sparsity': 0.02,
        'calcium_threshold': torch.ones(256, device=device) * 0.5,
        'gc_active_mask': torch.zeros(n_granule, device=device),
        'error_ema': 0.0,
        'last_climbing_fiber_error': torch.zeros(256, device=device),
        'mli_inhibition_scale': torch.ones(n_granule, device=device),
        'purkinje_bias': torch.zeros(256, device=device),
    }
    init_row_norms = cerebellum['purkinje_weights'].norm(dim=1)
    cerebellum['pk_target_row_norm'] = init_row_norms.clone()

    print(f"Cerebellum: {n_granule} GCs (K={K_MOSSY} mossy inputs each, biological), "
          f"{N_PONTINE}×2 dual pontine neurons, 256 Purkinje outputs")
    print(f"  delta_lr={cerebellum['delta_lr']}, gc_sparsity={cerebellum['gc_target_sparsity']}")
    print(f"  Compression ratio: {n_l56}:{N_PONTINE} = {n_l56/N_PONTINE:.1f}:1 (×2 with sign-inversion)")
    return cerebellum


def cerebellar_forward(engine, cerebellum):
    """Cerebellar forward: L5/6 → Dual Pontine → GC (K=4, Golgi kWTA) → PK → logits."""
    with torch.no_grad():
        l56_acts = engine.state[cerebellum['l56_indices']]

        # Fix 4: Dual pontine pathway — positive and sign-inverted
        # Positive pathway: captures features encoded as positive deviations
        pontine_raw = cerebellum['pontine_weights'] @ l56_acts
        pontine_pos = torch.relu(pontine_raw)
        # Sign-inverted pathway: captures features encoded as negative deviations
        # Biologically: mesodiencephalic junction provides sign-inverted cortical input
        pontine_neg = torch.relu(-pontine_raw)
        # Concatenate: [pos; neg] → 1024-dim representation
        pontine_full = torch.cat([pontine_pos, pontine_neg], dim=0)

        # Divisive normalization (Carandini & Heeger, 2012)
        sigma_sq = pontine_full.pow(2).mean() + 0.01
        pontine_acts = pontine_full / (sigma_sq.sqrt() + 0.1)

        # Fix 2: Mossy fiber → GC with K=4 biological connectivity
        granule_pre = cerebellum['mossy_weights'] @ pontine_acts

        # Divisive Golgi inhibition (stronger gain for K=4 sparse inputs)
        population_input = torch.relu(granule_pre).mean()
        cerebellum['golgi_inhibition_ema'] = (
            (1 - cerebellum['golgi_alpha']) * cerebellum['golgi_inhibition_ema']
            + cerebellum['golgi_alpha'] * population_input
        )
        # Increased Golgi gain: K=4 inputs need stronger inhibition for ~2% sparsity
        g_golgi = cerebellum['golgi_inhibition_ema'] * 15.0
        granule_acts_raw = torch.relu(granule_pre) / (1.0 + g_golgi)

        # kWTA sparsity: fixed at 2% for biological pattern separation
        # With K=4 conjunctive coding, natural sparsity is already low;
        # kWTA enforces the hard ceiling
        k_percent = 0.02
        k_winners = max(1, int(granule_pre.size(0) * k_percent))
        if k_winners < granule_acts_raw.size(0):
            topk_vals, topk_indices = torch.topk(granule_acts_raw, k_winners)
            granule_acts = torch.zeros_like(granule_acts_raw)
            granule_acts.scatter_(0, topk_indices, topk_vals)
        else:
            granule_acts = granule_acts_raw

        # Normalize GC vector — RMS scaling
        gc_active = granule_acts[granule_acts > 0]
        if gc_active.numel() > 0:
            rms = (gc_active.pow(2).mean()).sqrt().clamp(min=1e-6)
            granule_acts = granule_acts / rms

        # Purkinje readout with tonic baseline
        purkinje_output = cerebellum['purkinje_weights'] @ granule_acts
        logits_raw = cerebellum['purkinje_tonic_rate'] - purkinje_output
        
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

        # Store for learning
        cerebellum['gc_active_mask'] = (granule_acts > 0).float()
        cerebellum['_last_gc_for_purkinje'] = granule_acts
        cerebellum['_last_pontine_acts'] = pontine_acts
        cerebellum['_last_pontine_acts_raw'] = pontine_full  # Full dual-pathway

        # Gate for diagnostics only
        gate_value = torch.tensor(1.0, device=logits.device)
        return logits, granule_acts, gate_value


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
    Biologically-constrained cerebellar learning with:
    - Fix 1: Per-Purkinje scalar climbing fiber error (sparse, strong)
    - Fix 3: Soft-bounded inverse BCM plasticity (no hard cap)
    - Optional Linear Probe Distillation (Intervention 5)
    """
    device = logits.device
    
    # Fix 1: Per-Purkinje scalar CF error (sparse, graded)
    # Each PK cell receives ONE climbing fiber carrying a scalar error (Najafi & Medina 2013).
    # Only the target class and wrong-winner receive teaching signals.
    #
    # CRITICAL: The error magnitude must be scaled by 1/sqrt(n_active_GC) to normalize
    # the effective row-level weight update. Without this, the outer product
    # cf_error * gc_acts produces a row update whose L2 norm scales with sqrt(n_active),
    # causing logit explosion when n_active >> 1 (328 GCs at 2% sparsity).
    cf_error = torch.zeros(256, device=device)
    
    pred_byte = torch.argmax(logits).item()
    
    # Use softmax probabilities for graded error (not saturating sigmoid)
    # This keeps error magnitude in [0, 1) and provides useful gradient even with large logits
    probs = torch.softmax(logits, dim=0)
    
    # Target PK cell: CF → LTD at co-active PF synapses
    # LTD decreases PK weights → PK fires less → DCN disinhibited → logit RISES
    # Positive cf_error → negative weight update (via -delta_lr * cf_error * gc)
    cf_error[target_byte] = (1.0 - probs[target_byte])
    
    # Wrong-winner PK cell: rebound LTP → increase PK weights → more DCN inhibition → logit DROPS
    # Negative cf_error → positive weight update
    if pred_byte != target_byte:
        cf_error[pred_byte] = -probs[pred_byte]
    
    # Scale by 1/sqrt(n_active_GC) to normalize row-level gradient energy
    gc_acts_for_learn = cerebellum.get('_last_gc_for_purkinje', granule_acts)
    n_active_gc = (gc_acts_for_learn > 0).sum().item()
    cf_scale = 1.0 / max(n_active_gc ** 0.5, 1.0)
    cf_error = cf_error * cf_scale
    
    # Intervention 5: Linear Probe Distillation (blend sparse CF with probe signal)
    if probe_model is not None:
        with torch.no_grad():
            l56_indices = cerebellum['l56_indices']
            l56_acts = engine.state[l56_indices].unsqueeze(0)
            probe_logits = probe_model(l56_acts).squeeze(0)
            probe_probs = torch.softmax(probe_logits, dim=0)
            cf_error_probe = probe_probs - probs
            # Blend: keep sparse CF dominant (0.7) with probe smoothing (0.3)
            gamma = 0.7
            cf_error = gamma * cf_error + (1.0 - gamma) * cf_error_probe

    gc_acts = gc_acts_for_learn
    
    delta_lr = cerebellum['delta_lr']
    
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
    
    # Per-Purkinje sparse delta rule: Δw = -lr * cf_error ⊗ gc_acts * row_soft_bound
    ltd_update = -delta_lr * cf_error.unsqueeze(1) * gc_acts.unsqueeze(0)
    ltd_update = ltd_update * row_soft_scale.unsqueeze(1)  # Per-row soft bound
    cerebellum['purkinje_weights'] += ltd_update

    # Multiplicative homeostatic decay (Tononi & Cirelli SHY)
    # Slightly stronger decay (0.9998) to counteract residual weight growth
    cerebellum['purkinje_weights'] *= 0.9998

    # Soft normalization: gentle pull toward target norm (not hard clamp)
    _apply_pk_row_normalization(cerebellum)

    # Anti-Hebbian laterals (corrected sign: co-fire → increase inhibition)
    with torch.no_grad():
        y = torch.relu(logits)
        y_norm = y / (y.norm() + 1e-8)
        co_fire = cerebellum['lateral_lr'] * y_norm.unsqueeze(1) * y_norm.unsqueeze(0)
        co_fire.fill_diagonal_(0.0)
        cerebellum['lateral_weights'] += co_fire
        cerebellum['lateral_weights'] *= 0.998
        cerebellum['lateral_weights'].clamp_(min=0.0, max=0.5)

    cerebellum['last_climbing_fiber_error'] = cf_error
    # Track EMA of the non-zero CF signals only (sparse signal → larger per-cell values)
    active_cf = cf_error[cf_error.abs() > 1e-6]
    cf_mag = active_cf.abs().mean().item() if active_cf.numel() > 0 else 0.0
    cerebellum['error_ema'] = (
        0.99 * cerebellum.get('error_ema', 0.0)
        + 0.01 * cf_mag
    )
    return cf_mag


def _apply_pk_row_normalization(cerebellum):
    """Fix 3: Proportional restoring force toward target row norm.
    
    Replaces the weak 1%/step spring with a proportional restoring force:
    scale = 1 / (1 + α*(ratio - 1)) for ratio > 1
    
    This is biologically motivated by the cerebellar-olivary feedback loop
    (Kenyon, Medina & Mauk 1998): the IO self-regulates to maintain an
    equilibrium where expected net weight change is zero. Excess PK weights
    increase DCN inhibition → decrease IO firing → less LTD → net LTP shifts
    weights back down. The restoring force is proportional to the excess.
    """
    with torch.no_grad():
        w = cerebellum['purkinje_weights']
        target_norms = cerebellum['pk_target_row_norm']
        current_norms = w.norm(dim=1).clamp(min=1e-6)
        ratio = current_norms / target_norms.clamp(min=1e-6)
        # Proportional restoring: stronger pull the further from target
        # α=0.05: at ratio=2.0, scale=1/(1+0.05)=0.952; at ratio=3.0, scale=0.909
        excess = (ratio - 1.0).clamp(min=0.0)
        scale = 1.0 / (1.0 + 0.05 * excess)
        cerebellum['purkinje_weights'] *= scale.unsqueeze(1)


def compute_cerebellar_cortical_feedback(cerebellum, output_error):
    """Feedback through dual pontine: output_error → PK^T → GC → Mossy^T → Pontine^T → L5/6."""
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
        l56_gradient = cerebellum['pontine_weights'].T @ pontine_error_combined
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

    with torch.no_grad():
        for idx in sample_indices:
            curr_byte = int(data[idx])
            next_byte = int(data[idx+1])
            
            input_vec = torch.zeros(engine.num_nodes, device=device)
            input_vec[curr_byte] = 10.0
            
            # Settle engine (FREE phase only)
            engine.settle(
                input_vec, input_mask=input_mask, max_steps=free_steps, 
                tol=0.0, sigma_noise=0.0, damping=0.8
            )
            
            X.append(engine.state[l56_indices].clone())
            Y.append(next_byte)

    X = torch.stack(X) # [batch_size, num_features]
    Y = torch.tensor(Y, device=device) # [batch_size]

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
             leak_factor=0.95, device='cpu'):
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
                 leak_factor=0.95, device='cpu'):
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

    num_modules = 10
    num_levels = 2
    edge_index, edge_weight, biases, taus, graph = create_training_graph(
        num_modules=num_modules, num_levels=num_levels
    )
    num_nodes = biases.shape[0]

    # =====================================================================
    # ARCHITECTURAL ENHANCEMENT 1: Boost output connectivity
    # =====================================================================
    edge_index, edge_weight = boost_output_connectivity(
        graph, edge_index, edge_weight, num_nodes
    )

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
        ca3_indices=graph.ca3_indices,
        temporal_alpha=0.05,
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
        print(f"  Thalamocortical loop: {N_THALAMIC} neurons")
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

        start_time = time.time()
        for i in range(seq_len - 1):
            current_byte = int(data[i])
            next_byte = int(data[i + 1])

            # PHASE 1: Leaky Persistence (no reset)
            engine.state *= LEAK_FACTOR
            engine.state_basal *= LEAK_FACTOR
            engine.state_apical *= LEAK_FACTOR
            
            # Target 5: Thalamocortical Gating - rapidly reset thalamic state at each sequence token
            if getattr(graph, 'thalamic_indices', None) is not None:
                engine.state[graph.thalamic_indices] = 0.0
                engine.state_basal[graph.thalamic_indices] = 0.0
                engine.state_apical[graph.thalamic_indices] = 0.0
                
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
            # Intervention 5: Pass cached probe model for distillation
            cf_error = cerebellar_learn(cerebellum, logits, granule_acts, next_byte, gate_value,
                                        engine=engine, settle_diff=engine.last_settle_diff,
                                        probe_model=cached_probe_model)

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
                    f"P(target): {avg_target_prob:.3f} | "
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
                    f"Ca_θ: {ca_thresh_mean:.4f} | "
                    f"PK_row_ratio: {pk_row_ratio:.3f}"
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
                    show_readout(engine, cerebellum, data, current_pos=i+1, free_steps=FREE_STEPS, n_chars=80, leak_factor=LEAK_FACTOR, device=device)
                    
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
                              f"{' ← CLASS COLLAPSE' if n_unique <= 3 else ''}")
                
                # Eval on held-out data every 2500 steps
                if (i + 1) % 2500 == 0 and eval_data is not None:
                    eval_acc, eval_cs, eval_cc, eval_sc = run_eval(
                        engine, cerebellum, eval_data,
                        free_steps=FREE_STEPS, n_samples=300, leak_factor=LEAK_FACTOR, device=device
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