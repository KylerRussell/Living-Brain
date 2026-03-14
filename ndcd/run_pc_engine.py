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


def add_thalamic_relay(graph, edge_index, edge_weight, biases, taus, num_nodes,
                       n_thalamic=64, device='cpu'):
    """
    Add a thalamic relay bottleneck between association L5/6 and motor nodes.

    The thalamus in the brain acts as a low-rank hub that:
    - Lacks local excitatory recurrence (prevents runaway)
    - Compresses high-dimensional cortical manifolds into readable format
    - Has fast time constants (~5ms relay cells)
    - Projects densely to motor cortex

    This forces the network to route information through a narrow bottleneck,
    making the motor readout far more effective.

    Args:
        n_thalamic: Number of thalamic relay neurons (default 64)
    Returns:
        Updated edge_index, edge_weight, biases, taus, new_num_nodes
    """
    thal_start = num_nodes
    new_num_nodes = num_nodes + n_thalamic

    # Extend biases and taus
    thal_biases = torch.zeros(n_thalamic, dtype=torch.float32)
    thal_taus = torch.ones(n_thalamic, dtype=torch.float32) * 5.0  # Fast relay

    biases = torch.cat([biases, thal_biases])
    taus = torch.cat([taus, thal_taus])

    new_rows = []
    new_cols = []
    new_weights = []

    # 1. L5/6 → Thalamic (sparse, ~5% connectivity per thalamic neuron)
    all_l56 = []
    for mod in graph.modules:
        all_l56.extend(mod['l56_indices'].tolist())
    all_l56 = np.array(all_l56, dtype=np.int64)

    n_proj_per_thal = max(1, len(all_l56) // 20)  # ~5% of all L5/6
    for t in range(n_thalamic):
        sources = np.random.choice(all_l56, n_proj_per_thal, replace=False)
        thal_node = thal_start + t
        for s in sources:
            new_rows.append(s)
            new_cols.append(thal_node)
            new_weights.append(np.random.normal(0.1, 0.05))

    # 2. Thalamic → Motor (dense, full connectivity)
    for t in range(n_thalamic):
        thal_node = thal_start + t
        for m in range(256):
            motor_node = 256 + m
            new_rows.append(thal_node)
            new_cols.append(motor_node)
            new_weights.append(np.random.normal(0.05, 0.02))

    # NO thalamic ↔ thalamic recurrent connections (biological constraint)

    new_rows = np.array(new_rows, dtype=np.int64)
    new_cols = np.array(new_cols, dtype=np.int64)
    new_weights = np.array(new_weights, dtype=np.float32)

    # Merge
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

    n_l56_to_thal = n_thalamic * n_proj_per_thal
    n_thal_to_motor = n_thalamic * 256
    print(f"Thalamic relay: {n_thalamic} neurons, "
          f"{n_l56_to_thal} L5/6→thal edges, "
          f"{n_thal_to_motor} thal→motor edges")

    return edge_index, edge_weight, biases, taus, new_num_nodes


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

    # =====================================================================
    # ARCHITECTURAL ENHANCEMENT 2: Thalamic relay bottleneck
    # =====================================================================
    N_THALAMIC = 64
    edge_index, edge_weight, biases, taus, num_nodes = add_thalamic_relay(
        graph, edge_index, edge_weight, biases, taus, num_nodes,
        n_thalamic=N_THALAMIC, device=device
    )

    mod_starts = torch.tensor([m["start"] for m in graph.modules], dtype=torch.long)
    mod_ends = torch.tensor([m["end"] for m in graph.modules], dtype=torch.long)
    mod_levels = torch.tensor([m["level"] for m in graph.modules], dtype=torch.long)


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
    ALPHA_DFA = 0.15         # DFA feedback strength (reduced: 0.5 caused error saturation)
    BURST_APICAL_GAIN = 1.5  # Apical injection strength for burst coincidence

    for phase_info in phases:
        phase_name = phase_info["name"]
        file_path = os.path.join(data_dir, phase_info["file"])

        print(f"\n--- Starting Phase: {phase_name} ---")
        print(f"  Steps: Free={FREE_STEPS}, Nudge={NUDGE_STEPS}")
        print(f"  Learning rate: {lr}, Nudge strength: {nudge_strength}")
        print(f"  DFA alpha: {ALPHA_DFA}, Burst apical gain: {BURST_APICAL_GAIN}")
        print(f"  Thalamic relay: {N_THALAMIC} neurons")

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

            # Read output prediction from raw output nodes
            logits = free_state[256:512]
            pred_byte = torch.argmax(logits).item()
            is_correct = pred_byte == next_byte
            accuracies.append(1.0 if is_correct else 0.0)

            key = (current_byte, next_byte)
            bigram_total[key] = bigram_total.get(key, 0) + 1
            if is_correct:
                bigram_correct[key] = bigram_correct.get(key, 0) + 1

            # Prediction errors
            spatial_err = engine.compute_prediction_errors()
            spatial_errors.append(spatial_err)

            # === NUDGED PHASE (with DFA + Burst Coincidence) ===
            target_one_hot = torch.zeros(256, device=device)
            target_one_hot[next_byte] = 1.0

            # Compute output error for DFA
            output_activations = torch.tanh(free_state[256:512])
            output_error = target_one_hot - output_activations  # [256]

            # Restore to free state before nudge
            engine.state = free_state.clone()
            engine.state_basal = free_basal.clone()
            engine.state_apical = free_apical.clone()

            # Build nudge vector with DFA feedback
            nudge_vec = torch.zeros(num_nodes, device=device)
            nudge_vec[current_byte] = 10.0
            nudge_vec[256:512] = nudge_strength * target_one_hot

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

            # Phase 2 Hebbian update (Biological Stabilization)
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

                print(
                    f"Step {i + 1}/{seq_len} | Acc: {avg_acc:.2%} | Err: {avg_err:.4f} | "
                    f"Bst: {avg_bst:.3f} | "
                    f"W_surf: {w_surf_norm:.4f} | Wmx: {w_max:.3f} | "
                    f"Fr: {mean_firing:.3f} | B: {bias_norm:.2f} | Inh: {inh_w_norm:.4f} | "
                    f"{tps:.0f} tok/s"
                )

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
