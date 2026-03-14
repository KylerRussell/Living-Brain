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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_modules = 10
    num_levels = 2
    edge_index, edge_weight, biases, taus, graph = create_training_graph(
        num_modules=num_modules, num_levels=num_levels
    )
    num_nodes = biases.shape[0]

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
        is_inhibitory=graph.is_inhibitory,
        is_pv=graph.is_pv,
        is_sst=graph.is_sst,
        is_vip=graph.is_vip,
        is_lts=getattr(graph, "is_lts", None),
        dg_indices=graph.dg_indices,
        ca3_indices=graph.ca3_indices,
        temporal_alpha=0.05,
    )

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
    # KEY DESIGN DECISION: Streaming dynamics vs. equilibrium settling
    # =================================================================
    # For sequential prediction, the network must retain temporal context
    # from previous tokens. Settling to equilibrium (150 steps) erases
    # this context -- slow-tau nodes fully equilibrate to the current
    # input, losing history.
    #
    # Instead, we use FEW integration steps per token. This means:
    # - Fast nodes (tau~20, level 0) partially respond to new input
    # - Slow nodes (tau~175, level 1) retain most state from prior tokens
    # - Output nodes reflect both current input AND temporal context
    # - STSP facilitation/depression traces carry working memory
    #
    # This is biologically correct: cortex never "settles to equilibrium"
    # on each phoneme -- it integrates input with ongoing dynamics.
    # =================================================================
    # Optimized for CPU Throughput
    FREE_STEPS = 100
    NUDGE_STEPS = 10
    nudge_strength = 2.0
    lr = 0.05
    sigma_noise_free = 0.01
    sigma_noise_nudge = 0.01
    LEAK_FACTOR = 0.85

    for phase_info in phases:
        phase_name = phase_info["name"]
        file_path = os.path.join(data_dir, phase_info["file"])

        print(f"\n--- Starting Phase: {phase_name} (Phase 0 Recovery Baseline) ---")
        print(f"  Steps: Free={FREE_STEPS}, Nudge={NUDGE_STEPS}")
        print(f"  Learning rate: {lr}, Nudge strength: {nudge_strength}")

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

            # Read output prediction
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

            # === SINGLE-SIDED NUDGED PHASE ===
            target_one_hot = torch.zeros(256, device=device)
            target_one_hot[next_byte] = 1.0

            # Restore to free state before each nudge
            engine.state = free_state.clone()
            engine.state_basal = free_basal.clone()
            engine.state_apical = free_apical.clone()

            nudge_vec_pos = torch.zeros(num_nodes, device=device)
            nudge_vec_pos[current_byte] = 10.0
            nudge_vec_pos[256:512] = nudge_strength * target_one_hot

            engine.settle(
                nudge_vec_pos,
                input_mask=input_mask,
                max_steps=NUDGE_STEPS,
                tol=0.0,
                sigma_noise=sigma_noise_nudge,
                damping=0.8
            )
            nudge_pos = engine.state.clone()

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

                out_range = f"[{free_state[256:512].min().item():.3f}, {free_state[256:512].max().item():.3f}]"
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
            f"\nPhase {phase_name} complete in {elapsed:.1f}s | Final Acc: {final_acc:.2%}"
        )


if __name__ == "__main__":
    main()
