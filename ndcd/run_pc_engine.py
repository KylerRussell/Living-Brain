import os
import time
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
    Builds a smaller hierarchical graph suitable for initial PC engine training.
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
        # Sparse projection: 10% of L5/6 nodes per output
        n_proj = max(1, len(idx) // 10)
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
    # Relay neurons are silent by default, only fire when gate opens
    thal_biases = torch.ones(n_thalamic, dtype=torch.float32) * -0.5  # Reduced tonic inhibition
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
    return edge_index, edge_weight, biases, taus, new_num_nodes


def add_cerebellar_module(graph, num_nodes, n_granule=2048, sparsity=0.05, device='cpu'):
    """
    Cerebellar output module: Granule Cell expansion + Purkinje readout.

    Architecture:
      L5/6 ──[fixed random sparse]──→ Granule Cells (2048)
      Granule Cells ──[plastic, climbing fiber LTD]──→ Motor/Purkinje (256)

    The granule cell expansion acts as a random kernel that makes the
    L5/6 representation linearly separable (Cover's theorem). The Purkinje
    readout is a single plastic layer trained by the inferior olive error.

    Key biological constraints:
    - Mossy fiber → Granule projections are FIXED (non-plastic)
    - Granule cells have NO recurrent connections
    - Each granule cell receives from ~4 mossy fibers (extreme convergence)
    - Granule → Purkinje weights are the ONLY plastic pathway
    - Climbing fiber (inferior olive) carries target - output error

    Args:
        n_granule: Number of granule cells (2048 recommended, ~8x output dim)
        sparsity: Fraction of L5/6 nodes each granule cell samples from
    Returns:
        cerebellum: dict with all cerebellar state
    """
    # Gather all L5/6 indices
    all_l56 = []
    for mod in graph.modules:
        all_l56.extend(mod['l56_indices'].tolist())
    all_l56 = np.array(all_l56, dtype=np.int64)
    n_l56 = len(all_l56)

    # --- Mossy Fiber Projection Matrix (FIXED, non-plastic) ---
    # Each granule cell samples from ~4-5% of L5/6 nodes (biological: ~4 mossy fibers)
    # Stored as a dense [n_granule, n_l56] matrix for simplicity.
    n_inputs_per_granule = max(4, int(n_l56 * sparsity))

    # Build sparse binary connectivity mask
    mossy_mask = torch.zeros(n_granule, n_l56, dtype=torch.bool)
    for g in range(n_granule):
        selected = np.random.choice(n_l56, n_inputs_per_granule, replace=False)
        mossy_mask[g, selected] = True

    # Random weights, masked and normalized
    mossy_weights = torch.randn(n_granule, n_l56) * mossy_mask.float()
    # Normalize each granule cell's input weights to unit variance
    row_norms = mossy_weights.norm(dim=1, keepdim=True).clamp(min=1e-6)
    mossy_weights = mossy_weights / row_norms

    # --- Parallel Fiber → Purkinje Weights (PLASTIC via climbing fiber LTD) ---
    # [256, n_granule] — each Purkinje cell (motor node) reads all granule cells
    purkinje_weights = torch.randn(256, n_granule) / np.sqrt(n_granule)

    # --- Granule Cell State ---
    granule_state = torch.zeros(n_granule)

    # --- Basal Ganglia Gate State ---
    bg_gate_open = False
    bg_confidence_threshold = 0.3   # settle_diff/sqrt(N) must be below this to open gate
    bg_gate_sharpness = 20.0        # sigmoid sharpness for soft gating

    cerebellum = {
        'n_granule': n_granule,
        'l56_indices': torch.tensor(all_l56, dtype=torch.long),
        'mossy_weights': mossy_weights.to(device),     # [n_granule, n_l56] FIXED
        'purkinje_weights': purkinje_weights.to(device), # [256, n_granule] PLASTIC
        'granule_state': granule_state.to(device),       # [n_granule]
        'bg_confidence_threshold': bg_confidence_threshold,
        'bg_gate_sharpness': bg_gate_sharpness,
        'climbing_fiber_lr': 0.01,                        # Climbing fiber learning rate
        'purkinje_eligibility': torch.zeros(256, n_granule, device=device),  # Eligibility trace
        'tau_eligibility': 5.0,                           # Eligibility trace time constant
    }

    print(f"Cerebellum: {n_granule} granule cells, {n_inputs_per_granule} mossy fibers each, "
          f"256 Purkinje outputs")
    return cerebellum


def cerebellar_forward(engine, cerebellum):
    """
    Cerebellar forward pass: compute output logits from L5/6 state.

    1. Basal ganglia gate: check if settle has converged
    2. Extract L5/6 activations
    3. Mossy fiber → Granule cell activation (fixed weights, ReLU)
    4. Parallel fiber → Purkinje cell activation (plastic weights)
    5. Return raw logits (256-dim)

    Returns:
        logits: [256] tensor of output predictions
        granule_acts: [n_granule] tensor (needed for learning rule)
        gate_value: scalar (0-1) indicating gate openness
    """
    with torch.no_grad():
        # 1. Basal ganglia gate based on settle convergence
        gate_value = engine.get_bg_gate_confidence(
            threshold=cerebellum['bg_confidence_threshold'],
            sharpness=cerebellum['bg_gate_sharpness']
        )

        # 2. Extract L5/6 activations
        l56_idx = cerebellum['l56_indices']
        l56_acts = torch.tanh(engine.state[l56_idx])  # [n_l56]

        # 3. Mossy fiber → Granule cells (fixed projection + ReLU)
        granule_pre = cerebellum['mossy_weights'] @ l56_acts  # [n_granule]
        granule_acts = torch.relu(granule_pre)  # Sparse ReLU activation

        # Apply gate: scale granule activity by basal ganglia confidence
        granule_acts = granule_acts * gate_value

        # Store for diagnostics
        cerebellum['granule_state'] = granule_acts

        # 4. Parallel fiber → Purkinje (plastic weights, linear readout)
        logits = cerebellum['purkinje_weights'] @ granule_acts  # [256]

        return logits, granule_acts, gate_value


def cerebellar_learn(cerebellum, logits, granule_acts, target_byte, gate_value):
    """
    Climbing fiber learning rule (inferior olive → Purkinje LTD/LTP).

    The inferior olive computes: error = target - purkinje_output
    The climbing fiber delivers this error to each Purkinje cell.
    Parallel fiber → Purkinje synapses undergo:
      - LTD when climbing fiber fires AND parallel fiber is active
        (wrong prediction while granule cell was active → weaken)
      - LTP when climbing fiber is silent AND parallel fiber is active
        (correct prediction while granule cell was active → strengthen)

    Learning rule:
      Δw_ij = -η * error_j * granule_i * gate

    The eligibility trace adds a temporal buffer so that the error signal
    from the next timestep can update weights based on the current
    granule cell activity.
    """
    with torch.no_grad():
        lr = cerebellum['climbing_fiber_lr']
        tau_e = cerebellum['tau_eligibility']

        # Target one-hot
        target = torch.zeros(256, device=logits.device)
        target[target_byte] = 1.0

        # Inferior olive error signal (climbing fiber)
        purkinje_output = torch.softmax(logits, dim=0)
        climbing_fiber_error = target - purkinje_output  # [256]

        # Update eligibility trace (low-pass filter of granule activity)
        cerebellum['purkinje_eligibility'] *= (1.0 - 1.0 / tau_e)
        cerebellum['purkinje_eligibility'] += (1.0 / tau_e) * granule_acts.unsqueeze(0)

        # Climbing fiber modulated update:
        # Δw_ij = η * error_i * eligibility_ij * gate
        delta_w = lr * climbing_fiber_error.unsqueeze(1) * cerebellum['purkinje_eligibility'] * gate_value

        # Apply update
        cerebellum['purkinje_weights'] += delta_w

        # Gentle weight decay
        cerebellum['purkinje_weights'] *= 0.9999

        return climbing_fiber_error.abs().mean().item()


def create_dfa_matrices(engine, n_output=256, device='cpu'):
    """
    Create fixed random feedback matrices for Direct Feedback Alignment,
    weighted by each node's existing connectivity to motor output.

    DFA uses random, fixed (non-learned) projections of the output error
    back to each hidden layer. This version scales each node's feedback
    by its existing forward connectivity to motor nodes — nodes that
    already project to motor get stronger DFA signal, focusing learning
    on output-relevant pathways.

    Returns:
        dfa_matrices: dict mapping level -> (B, level_mask) where
            B: [level_size, n_output] feedback matrix
            level_mask: [N] boolean mask for this level's nodes
    """
    # Compute per-node motor connectivity strength
    # Sum of absolute weights projecting to motor nodes (256-511)
    motor_connectivity = torch.zeros(engine.num_nodes, device=device)
    dst = engine.indices[1]
    src = engine.indices[0]
    motor_dst_mask = (dst >= 256) & (dst < 512)
    if motor_dst_mask.any():
        motor_src = src[motor_dst_mask]
        motor_w = engine.effective_weights[motor_dst_mask].abs()
        motor_connectivity.scatter_add_(0, motor_src, motor_w)

    # Normalize to [0.1, 1.0] range — floor at 0.1 so unconnected nodes
    # still get some signal (they may develop motor projections later)
    mc_max = motor_connectivity.max().item()
    if mc_max > 0:
        motor_connectivity = 0.1 + 0.9 * (motor_connectivity / mc_max)
    else:
        motor_connectivity[:] = 1.0

    dfa_matrices = {}
    for level in range(engine.max_level + 1):
        level_mask = (engine.node_to_level == level)
        level_size = level_mask.sum().item()
        if level_size > 0:
            # Fixed random matrix, normalized by sqrt(fan_in)
            B = torch.randn(level_size, n_output, device=device) / np.sqrt(n_output)
            # Scale each row by that node's motor connectivity
            mc_level = motor_connectivity[level_mask]  # [level_size]
            B *= mc_level.unsqueeze(1)  # broadcast: [level_size, 1]
            dfa_matrices[level] = (B, level_mask)

    # Stats
    total_nodes = sum(m.sum().item() for _, (_, m) in dfa_matrices.items())
    avg_mc = motor_connectivity[512:].mean().item()
    print(f"DFA: Created connectivity-weighted feedback for {len(dfa_matrices)} levels, "
          f"{total_nodes} nodes, avg motor connectivity: {avg_mc:.3f}")
    return dfa_matrices


def run_linear_probe(engine, data, batch_size=200, free_steps=100, device='cpu'):
    """
    Diagnostic: Freeze the engine, collect latent L5/6 activations for 200 samples,
    and train a temporary linear probe to see if the internal representation
    is actually learning anything, bypassing the complex motor readout.
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
    
    return acc


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
    N_THALAMIC = 128
    edge_index, edge_weight, biases, taus, num_nodes = add_thalamocortical_loop(
        graph, edge_index, edge_weight, biases, taus, num_nodes,
        n_thalamic=N_THALAMIC, device=device
    )

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
    cerebellum = add_cerebellar_module(graph, num_nodes, n_granule=2048, device=device)

    # =====================================================================
    # ARCHITECTURAL ENHANCEMENT 3: Direct Feedback Alignment matrices
    # =====================================================================
    dfa_matrices = create_dfa_matrices(engine, n_output=256, device=device)

    # Curriculum setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")

    phases = [
        {
            "name": "Holophrases",
            "file": "train/level1_holophrases.txt",
            "lr": 0.05,
            "epochs": 5,
        },
    ]

    # Input Mask: clamp input nodes only during settling
    input_mask = torch.zeros(num_nodes, device=device)
    input_mask[:256] = 1.0

    # =================================================================
    # Hyperparameters (Round 3 — with DFA + Thalamic + Burst)
    # =================================================================
    FREE_STEPS = 100
    NUDGE_STEPS = 30
    nudge_strength = 5.0
    lr = 0.01
    sigma_noise_free = 0.01
    sigma_noise_nudge = 0.01
    LEAK_FACTOR = 0.95
    ALPHA_DFA = 0.15         # DFA feedback strength
    BURST_APICAL_GAIN = 1.5  # Apical injection strength for burst coincidence

    for phase_info in phases:
        phase_name = phase_info["name"]
        file_path = os.path.join(data_dir, phase_info["file"])

        print(f"\n--- Starting Phase: {phase_name} ---")
        print(f"  Steps: Free={FREE_STEPS}, Nudge={NUDGE_STEPS}")
        print(f"  Learning rate: {lr}, Nudge strength: {nudge_strength}")
        print(f"  DFA alpha: {ALPHA_DFA}, Burst apical gain: {BURST_APICAL_GAIN}")
        print(f"  Thalamocortical loop: {N_THALAMIC} neurons")
        print(f"  Cerebellum: {cerebellum['n_granule']} granule cells")

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        data = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
        seq_len = len(data)

        # Metrics tracking
        spatial_errors = []
        accuracies = []
        burst_strengths = []
        bigram_total = {}
        bigram_correct = {}

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
            engine.previous_state = engine.state.clone()

            # === FREE PHASE ===
            input_vec = torch.zeros(num_nodes, device=device)
            input_vec[current_byte] = 10.0

            engine.settle(
                input_vec, input_mask=input_mask, max_steps=FREE_STEPS, tol=0.0,
                sigma_noise=sigma_noise_free, damping=0.8
            )
            free_state = engine.state.clone()
            free_basal = engine.state_basal.clone()
            free_apical = engine.state_apical.clone()

            # === CEREBELLAR READOUT (replaces FRNL + lateral inhibition) ===
            logits, granule_acts, gate_value = cerebellar_forward(engine, cerebellum)
            pred_byte = torch.argmax(logits).item()
            is_correct = pred_byte == next_byte
            accuracies.append(1.0 if is_correct else 0.0)

            # Update bigram stats
            key = (current_byte, next_byte)
            bigram_total[key] = bigram_total.get(key, 0) + 1
            if is_correct:
                bigram_correct[key] = bigram_correct.get(key, 0) + 1

            # === CEREBELLAR LEARNING (replaces terminal Adam optimizer) ===
            cf_error = cerebellar_learn(cerebellum, logits, granule_acts, next_byte, gate_value)

            # Prediction errors
            spatial_err = engine.compute_prediction_errors()
            spatial_errors.append(spatial_err)

            # === NUDGED PHASE (with DFA + Burst Coincidence) ===
            target_one_hot = torch.zeros(256, device=device)
            target_one_hot[next_byte] = 1.0

            # Compute output error for DFA (injecting cerebellar precision error)
            output_error = target_one_hot - torch.softmax(logits, dim=0)

            # Restore to free state before nudge
            engine.state = free_state.clone()
            engine.state_basal = free_basal.clone()
            engine.state_apical = free_apical.clone()

            # Build nudge vector with DFA feedback
            nudge_vec = torch.zeros(num_nodes, device=device)
            nudge_vec[current_byte] = 10.0
            # NOTE: No direct motor node clamping — the cerebellum handles output

            # ENHANCEMENT 3a: DFA — inject output error into all hidden levels
            for level, (B, level_mask) in dfa_matrices.items():
                # B @ output_error gives a [level_size] feedback signal
                dfa_signal = B @ output_error  # [level_size]
                nudge_vec[level_mask] += ALPHA_DFA * dfa_signal

            # ENHANCEMENT 3b: Burst Coincidence — inject target into apical
            engine.inject_apical_nudge(target_one_hot, strength=BURST_APICAL_GAIN)

            engine.settle(
                nudge_vec,
                input_mask=input_mask,
                max_steps=NUDGE_STEPS,
                tol=0.0,
                sigma_noise=sigma_noise_nudge,
                damping=0.3
            )
            nudge_pos = engine.state.clone()

            # Compute burst metric for monitoring
            burst = engine.compute_burst_coincidence()
            avg_burst = burst[engine.is_l56].mean().item()
            burst_strengths.append(avg_burst)

            # Biological EqProp/DFA for all deep layers
            engine.update_weights_phase2(
                free_state,
                nudge_pos,
                learning_rate=lr,
            )

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
                # Purkinje weight norm
                pk_w_norm = cerebellum['purkinje_weights'].norm().item()

                print(
                    f"Step {i+1}/{seq_len} | Acc: {avg_acc:.2%} | "
                    f"Gate: {gate_value:.2f} | GC_spars: {gc_sparsity:.2%} | "
                    f"CF_err: {cf_error:.4f} | PK_w: {pk_w_norm:.2f} | "
                    f"Settle: {engine.last_settle_diff:.4f}"
                )

                # Diagnostic Linear Probe every 500 steps
                if (i + 1) % 500 == 0:
                    probe_acc = run_linear_probe(
                        engine, data, batch_size=200, 
                        free_steps=FREE_STEPS, device=device
                    )
                    print(f"  >>> LINEAR PROBE ACCURACY: {probe_acc:.2%} (Internal Representation Quality)")
                    if probe_acc > avg_acc * 2:
                        print(f"  [!] ALERT: Representation ({probe_acc:.1%}) >> Readout ({avg_acc:.1%}). Readout is the bottleneck.")
                    else:
                        print(f"  [!] NOTE: Representation ({probe_acc:.1%}) ≈ Readout ({avg_acc:.1%}). Hebbian learning is the bottleneck.")

                # Bigram stats
                top_bigrams = sorted(
                    [(k, bigram_correct.get(k, 0) / bigram_total[k], bigram_total[k]) for k in bigram_total if bigram_total[k] >= 5],
                    key=lambda x: x[1], reverse=True
                )[:3]
                if top_bigrams:
                    bigram_strs = [f"'{chr(k[0])}{chr(k[1])}': {acc:.0%} ({tot})" for k, acc, tot in top_bigrams]
                    print(f"  Top Bigrams (min 5): {', '.join(bigram_strs)}")

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
