import os
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
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
    )

    # Curriculum setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")

    phases = [
        {
            "name": "Holophrases",
            "file": "level1_holophrases.txt",
            "lr": 0.05,
            "epochs": 1,
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
    FREE_STEPS = 32  # Increased to 32 for Phase 2.3 convergence search
    NUDGE_STEPS = 5  # Steps for nudge perturbation to propagate
    nudge_strength = 0.05  # Reduced from 0.1 to reduce oscillation in EqProp

    for phase_info in phases:
        phase_name = phase_info["name"]
        file_path = os.path.join(data_dir, phase_info["file"])
        lr = phase_info["lr"]

        print(f"\n--- Starting Phase: {phase_name} ---")
        print(f"  Free steps: {FREE_STEPS}, Nudge steps: {NUDGE_STEPS}")
        print(f"  Learning rate: {lr}, Nudge strength: {nudge_strength}")

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        data = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)

        # Zero transient states for new phase, but NOT between tokens
        print("Zeroing engine transient states for new phase...")
        engine.zero_states()

        seq_len = len(data)

        # Metrics tracking
        spatial_errors = []
        accuracies = []
        topdown_vars = []

        start_time = time.time()
        for i in range(seq_len - 1):
            current_byte = int(data[i])
            next_byte = int(data[i + 1])
            char = (
                chr(current_byte)
                if 32 <= current_byte <= 126
                else f"\\x{current_byte:02x}"
            )

            # Level Gating
            active_level_max = 1
            if current_byte == 46:  # Period
                active_level_max = num_levels - 1

            engine.previous_state = engine.state.clone()

            # === FREE PHASE ===
            input_vec = torch.zeros(num_nodes, device=device)
            input_vec[current_byte] = 1.0

            engine.settle(
                input_vec, input_mask=input_mask, max_steps=FREE_STEPS, tol=0.0
            )
            free_state = engine.state.clone()

            # Read output prediction
            logits = free_state[256:512]
            pred_byte = torch.argmax(logits).item()
            is_correct = pred_byte == next_byte
            accuracies.append(1.0 if is_correct else 0.0)

            # Prediction errors
            spatial_err = engine.compute_prediction_errors()
            spatial_errors.append(spatial_err)
            if hasattr(engine, "topdown_pred_var"):
                topdown_vars.append(engine.topdown_pred_var)

            # === SYMMETRIC NUDGED PHASE ===
            # Phase 1: Positive Nudge (+beta)
            target_one_hot = torch.zeros(256, device=device)
            target_one_hot[next_byte] = 1.0

            # Attenuation factor alpha
            alpha = torch.exp(
                -torch.tensor(engine.last_settle_diff / 5.0, device=device)
            )
            beta = nudge_strength * alpha

            # Restore to free state before each nudge
            engine.state = free_state.clone()
            nudge_vec_pos = torch.zeros(num_nodes, device=device)
            nudge_vec_pos[current_byte] = 1.0
            nudge_vec_pos[256:512] = beta * (target_one_hot - free_state[256:512])

            engine.settle(
                nudge_vec_pos,
                input_mask=input_mask,
                max_steps=NUDGE_STEPS,
                tol=0.0,
                sigma_noise=0.15,
            )
            nudge_pos = engine.state.clone()

            # Phase 2: Negative Nudge (-beta)
            engine.state = free_state.clone()
            nudge_vec_neg = torch.zeros(num_nodes, device=device)
            nudge_vec_neg[current_byte] = 1.0
            nudge_vec_neg[256:512] = -beta * (target_one_hot - free_state[256:512])

            engine.settle(
                nudge_vec_neg,
                input_mask=input_mask,
                max_steps=NUDGE_STEPS,
                tol=0.0,
                sigma_noise=0.15,
            )
            nudge_neg = engine.state.clone()

            # Symmetric Contrastive Hebbian update
            engine.update_weights_predictive(
                free_state,
                nudge_pos,
                nudge_neg=nudge_neg,
                learning_rate=lr,
                active_level_max=active_level_max,
            )

            # Restore to FREE state for next token
            engine.state = free_state.clone()

            # Logging and periodic tasks
            if (i + 1) % 100 == 0:
                window = min(500, len(accuracies))
                avg_acc = np.mean(accuracies[-window:])
                avg_err = np.mean(spatial_errors[-100:])
                avg_var = (
                    np.mean(topdown_vars[-100:])
                    if len(topdown_vars) >= 100
                    else (np.mean(topdown_vars) if topdown_vars else 0.0)
                )

                out_range = f"[{free_state[256:512].min().item():.3f}, {free_state[256:512].max().item():.3f}]"
                assoc_mean = free_state[512:].abs().mean().item()
                w_surf_norm = engine.w_surface.norm().item()
                nudge_delta = (nudge_pos - nudge_neg).norm().item()
                elapsed = time.time() - start_time
                tps = (i + 1) / elapsed

                print(
                    f"Step {i + 1}/{seq_len} | Acc: {avg_acc:.2%} | Err: {avg_err:.4f} | "
                    f"TD Var: {avg_var:.6f} | Out: {out_range} | Assoc: {assoc_mean:.4f} | "
                    f"wSurf: {w_surf_norm:.4f} | dNudge: {nudge_delta:.4f} | {tps:.0f} tok/s"
                )

            if (i + 1) % 2000 == 0:
                engine.cascade_transfer(include_deep=False)
            if (i + 1) % 10000 == 0:
                engine.offline_renormalization(num_replay_cycles=5)

        # Phase boundaries
        engine.consolidate_importance()
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
