
import torch
import numpy as np
import os
import time
import argparse
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from ndcd.graph import DynamicGraph
from ndcd.engine_torch import PredictiveCodingEngine
from ndcd.curriculum_gen import generate_chars, generate_toddler_words, generate_quotes

def ensure_data(data_path, phase_name):
    if not os.path.exists(data_path):
        print(f"Data {data_path} not found. Generating for {phase_name}...")
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        if "level1" in data_path: generate_chars(data_path)
        elif "level2" in data_path: generate_toddler_words(data_path)
        elif "level3" in data_path: generate_quotes(data_path)
        elif "sherlock" in data_path:
             import urllib.request
             url = "https://www.gutenberg.org/files/1661/1661-0.txt"
             try:
                 urllib.request.urlretrieve(url, data_path)
             except Exception as e:
                 print(f"Failed to download Sherlock: {e}")

class SequentialTrainer:
    def __init__(self, num_nodes=50000, device='cpu', num_modules=50):
        self.device = device
        self.num_nodes = num_nodes
        if num_nodes < 512:
            raise ValueError(f"num_nodes ({num_nodes}) must be >= 512 to support 256 input + 256 output nodes.")

        # 1. Initialize Hierarchical Modular Graph
        print("Initializing Hierarchical Modular Graph...")
        self.graph = DynamicGraph(
            num_nodes=num_nodes,
            m_edges=20,  # Legacy param, unused in modular topology
            p_triad=0.1,
            seed=42,
            num_modules=num_modules,
            num_levels=4,
        )
        indices, values = self.graph.export_sparse_components()
        biases = self.graph.biases
        taus = self.graph.taus

        self.indices = indices
        self.initial_values = values

        # Input scale factor (fixed to 1.0)
        self.input_scale_factor = 1.0

        # I/O projection scaling and spectral tuning are handled in
        # graph.py (single pass). No second tuning needed here.

        # 2. Initialize Predictive Coding Engine
        # dt=0.5: IMEX is stable for large dt; converges in 10-20 steps
        module_ranges = self.graph.get_module_ranges()
        module_levels = self.graph.module_levels
        hier_pairs = self.graph.hier_pairs

        self.engine = PredictiveCodingEngine(
            num_nodes,
            indices,
            self.initial_values,
            biases,
            taus,
            module_ranges=module_ranges,
            module_levels=module_levels,
            hier_pairs=hier_pairs,
            positions=self.graph.pos,
            dt=0.5,
            device=device,
            temporal_alpha=0.5,
        )

        # 3. Define I/O Masks
        self.input_indices = list(range(0, 256))
        self.output_indices = list(range(256, 512))

        # Pre-compute One-Hot Identity Matrices
        self.eye = torch.eye(256, device=device)

        # --- Linear readout from level-0 module activations ---
        # The recurrent network produces representations; a separate readout
        # maps them to predictions. This is standard reservoir computing /
        # echo state network practice and decouples representation from
        # prediction, giving an immediate credit assignment path.
        self.level0_indices = []
        module_ranges = self.graph.get_module_ranges()
        for mod_id in self.graph.level_modules[0]:
            s, e = module_ranges[mod_id]
            self.level0_indices.extend(range(s, e))
        self.level0_indices = torch.tensor(self.level0_indices, dtype=torch.long, device=device)
        n_reservoir = len(self.level0_indices)
        n_readout = n_reservoir  # reservoir state only
        self.readout_W = torch.randn(256, n_readout, device=device) * (1.0 / np.sqrt(n_readout))
        self.readout_b = torch.zeros(256, device=device)
        print(f"Linear readout: {n_reservoir} level-0 = {n_readout} features → 256 classes")

        # --- Complementary Learning Systems setup ---
        # Store module classification and connectivity on engine for access during training
        self.hippocampal_modules = self.graph.hippocampal_modules
        self.neocortical_modules = self.graph.neocortical_modules
        self.node_to_module = torch.tensor(
            self.graph.node_to_module, dtype=torch.long, device=device)

        # Build per-edge module type mask for differentiated learning rates
        src_modules = self.graph.node_to_module[indices[0]]
        self.hippo_edge_mask = torch.tensor(
            np.isin(src_modules, list(self.hippocampal_modules)),
            dtype=torch.bool, device=device)

        # Set differentiated cascade rates for hippocampal modules
        # Hippocampal: faster deep cascade (τ_deep = 1000 not 10000)
        self.engine.tau_deep_hippo = 1000.0
        self.engine.tau_deep_neo = 10000.0

        # Synaptic intelligence consolidation interval
        self.si_consolidation_interval = 10000

        # Sleep phase interval
        self.sleep_interval = 5000

    def tune_spectral_radius(self, target_radius=0.95):
        """Tunes the spectral radius of the weight matrix to a target value.

        Only meaningful at initialization (random weights). After training,
        cascade weights accumulate to spectral radii of 50-100+, and rescaling
        by 0.006x lobotomizes the network. Use phase_boundary_reset() instead.
        """
        print(f"Tuning Spectral Radius to {target_radius:.2f}...")

        if hasattr(self, 'engine'):
            # Use effective weights (cascade sum) for spectral analysis
            w_tensor = self.engine.effective_weights.cpu().numpy()
            indices = self.engine.indices.cpu().numpy()
        else:
            w_tensor = self.initial_values
            indices = self.indices

        row = indices[0]
        col = indices[1]
        w_sparse = sp.csr_matrix((w_tensor, (row, col)), shape=(self.num_nodes, self.num_nodes))

        try:
            eigvals = eigs(w_sparse, k=1, which='LM', return_eigenvectors=False)
            max_eig = np.abs(eigvals[0])
            print(f"Current Spectral Radius: {max_eig:.4f}")

            scale_factor = target_radius / (max_eig + 1e-8)
            print(f"Scaled weights by {scale_factor:.4f}")

            self.input_scale_factor = 1.0

            if hasattr(self, 'engine'):
                # Scale all cascade levels proportionally
                self.engine.w_deep *= scale_factor
                self.engine.w_mid *= scale_factor
                self.engine.w_surface *= scale_factor
            else:
                self.initial_values = w_tensor * scale_factor

        except Exception as e:
            print(f"Warning: Spectral tuning failed ({e}). Using default.")

    def phase_boundary_reset(self, phase_from, phase_to):
        """Reset fast-timescale weights at phase boundaries; preserve w_deep.

        Unlike tune_spectral_radius(), this does NOT rescale w_deep.
        After 50K chars steps, w_deep reaches spectral radius ~124 —
        rescaling to 0.80 multiplies all weights by 0.006x, destroying
        everything the network learned. Instead we:
        - Keep w_deep intact (long-term memory / inductive bias)
        - Zero w_surface and w_mid (transient learning from prior phase)
        - Zero network state (avoid prior-phase attractor lock-in)
        - Zero SI accumulators (omega was double-counting protection)
        - Re-initialize readout from scratch (old readout is calibrated
          to the prior phase's distribution and misleads early learning)
        """
        print(f"\n--- Phase boundary: {phase_from} → {phase_to} ---")

        with torch.no_grad():
            # Preserve w_deep (long-term memory), reset transients
            deep_mag = self.engine.w_deep.abs().mean().item()
            self.engine.w_surface.zero_()
            self.engine.w_mid.zero_()

            # Reset network state to avoid prior-phase attractors
            self.engine.state.zero_()
            self.engine.previous_state.zero_()

            # Reset synaptic intelligence — omega double-counts with cascade
            self.engine.omega.zero_()
            self.engine.running_contribution.zero_()

            # Re-initialize readout from scratch for the new task
            n_readout = len(self.level0_indices)
            self.readout_W = torch.randn(256, n_readout, device=self.device) * (1.0 / np.sqrt(n_readout))
            self.readout_b.zero_()

        print(f"  w_deep preserved (|w_deep|={deep_mag:.4f}), "
              f"w_surface/w_mid zeroed, state reset, readout re-initialized")

    def train_phase(self, phase_name, data_path, iterations, steps_per_iter, lr=0.01,
                    settle_steps=20, input_gain=0.5, warmup_steps=0):
        """
        Predictive Coding training loop.

        For each token:
        1. Clamp input byte at level-0 input nodes
        2. Store previous state for temporal prediction
        3. Settle dynamics (single phase, IMEX steps)
        4. Compute prediction errors (spatial + temporal)
        5. Update weights using LOCAL Hebbian rule
        6. Update short-term plasticity (from Phase 1)
        7. Log metrics

        No nudging, no beta, no three-phase settling.

        warmup_steps: Number of initial steps where only the readout is
            trained and recurrent weight updates are frozen. This lets the
            readout adapt to the new task's distribution before the reservoir
            starts shifting, preventing the reservoir from being pulled in
            random directions by a misaligned readout gradient.
        """
        print(f"\n=== Starting Phase: {phase_name} ===")
        print(f"Run started at: {time.ctime()}")
        if warmup_steps > 0:
            print(f"Readout-only warmup for first {warmup_steps} steps (recurrent weights frozen)")
        ensure_data(data_path, phase_name)

        if not os.path.exists(data_path):
            print(f"Skipping {phase_name} (Data missing)")
            return

        with open(data_path, 'rb') as f:
            data = f.read()

        data_len = len(data)
        curr_idx = 0

        start_time = time.time()

        total_steps = iterations * steps_per_iter

        loss_accum = 0.0
        energy_accum = 0.0
        acc_window = []
        acc_top3_window = []

        for step in range(total_steps):
            # 1. Get Data Stream
            if curr_idx >= data_len - 1:
                curr_idx = 0

            input_byte = data[curr_idx]
            target_byte = data[curr_idx + 1]
            curr_idx += 1

            # 2. Input Setup — full one-hot across all 256 input nodes.
            # A single node at 5.0 with 255 zeroes wastes input projection
            # bandwidth and gets drowned by ~500 recurrent neighbors.
            input_mask = torch.zeros(self.num_nodes, device=self.device)
            input_mask[self.input_indices] = 1.0

            input_vec = torch.zeros(self.num_nodes, device=self.device)
            input_vec[0:256] = self.eye[input_byte] * input_gain

            # 3. Store previous state for temporal prediction
            self.engine.store_previous_state()

            # 4. Reset fast nodes only at word boundaries (space/newline/tab).
            # Previously reset every step, destroying intra-word context that
            # word-level and sentence-level tasks depend on. For chars (same
            # repeating sequence) this didn't matter, but for words/quotes the
            # accumulated context within a word is critical for prediction.
            if input_byte in (32, 10, 13, 9):  # space, LF, CR, tab
                fast_mask = self.engine.taus < 0.5
                self.engine.state[fast_mask] *= 0.1

            # 5. Single-phase settle (IMEX) -> FREE PHASE
            # Dynamic tolerance: relax to 1e-2 during the first 2000 steps
            # of each phase, preventing the solver from fruitlessly exhausting
            # max_steps during transient reorganization after phase boundaries.
            settle_tol = 1e-2 if step < 2000 else 5e-3
            self.engine.settle(
                input_vec,
                input_mask=input_mask,
                max_steps=settle_steps,
                tol=settle_tol,
            )
            free_state = self.engine.state.clone()
            free_diff = self.engine.last_settle_diff

            # 6. Compute prediction errors (top-down only; I/O zeroed) based on free state
            energy = self.engine.compute_prediction_errors()

            # 6b. EqProp Nudge Phase
            beta = 0.5
            output_target = self.eye[target_byte]                          # one-hot [256]
            nudge_vec = input_vec.clone()
            nudge_vec[256:512] = output_target * beta
            
            self.engine.settle(
                nudge_vec,
                input_mask=input_mask,
                max_steps=settle_steps,
                tol=settle_tol,
            )
            nudge_state = self.engine.state.clone()
            nudge_diff = self.engine.last_settle_diff

            output_actual = torch.tanh(free_state[256:512])         # evaluate on free_state
            output_error = output_target - output_actual                   # observation error
            
            energy += 0.5 * torch.sum(output_error ** 2).item()
            energy_accum += energy

            if step < 5:
                # Debug output nodes 
                rho_free = torch.tanh(free_state)
                rho_nudge = torch.tanh(nudge_state)
                target_diff = rho_nudge[256:512] - rho_free[256:512]
                print(f"DEBUG Step {step}: target_byte={target_byte}")
                print(f"  Target node diff= {target_diff[target_byte].item():.5f}, Non-target avg diff= {target_diff.mean().item():.5f}")
                print(f"  Max free state output= {rho_free[256:512].max().item():.5f}, Min= {rho_free[256:512].min().item():.5f}")

            # 7. Measure Prediction via linear readout (not output nodes).
            # Readout sees reservoir state + raw input (standard ESN practice).
            # This gives a direct linear path from input to prediction even if
            # reservoir representations are degenerate.
            with torch.no_grad():
                level0_acts = torch.tanh(free_state[self.level0_indices])
                features = level0_acts
                logits = self.readout_W @ features + self.readout_b
                probs = torch.softmax(logits, dim=0)
                pred_idx = torch.argmax(probs).item()

            state_norm = torch.norm(free_state) / np.sqrt(self.num_nodes)

            is_correct = (pred_idx == target_byte)
            acc_window.append(1.0 if is_correct else 0.0)

            _, top3_indices = torch.topk(probs, 3)
            if target_byte in top3_indices.tolist():
                acc_top3_window.append(1.0)
            else:
                acc_top3_window.append(0.0)

            loss = -torch.log(probs[target_byte] + 1e-8).item()
            loss_accum += loss

            # 8. Cosine LR schedule (used by readout, recurrent update, CLS)
            lr_mult = 0.5 * (1.0 + np.cos(np.pi * step / total_steps))
            effective_lr = lr * max(lr_mult, 0.1)

            # 8a. Train readout with cross-entropy gradient descent.
            with torch.no_grad():
                target_one_hot = self.eye[target_byte]
                readout_grad = probs - target_one_hot  # softmax CE gradient
                self.readout_W -= effective_lr * torch.outer(readout_grad, features)
                self.readout_b -= effective_lr * readout_grad

            # 8b-8e: Skip recurrent weight updates during warmup period.
            # During warmup, only the readout adapts to the new task distribution.
            # This prevents the reservoir from being pulled in random directions
            # before the readout has calibrated to the new data statistics.
            if step >= warmup_steps:
                # 8b. Update recurrent weights using EqProp rule
                self.engine.update_weights_predictive(
                    free_state,
                    nudge_state,
                    beta=beta,
                    learning_rate=effective_lr,
                    hippo_edge_mask=self.hippo_edge_mask)
                
                if step < 5:
                    # Inspect the motor weights
                    t_mask = self.engine.topdown_edge_mask
                    b_mask = self.engine.bottomup_edge_mask
                    td_motor = (self.engine.indices[1][t_mask] >= 256) & (self.engine.indices[1][t_mask] < 512)
                    bu_motor = (self.engine.indices[0][b_mask] >= 256) & (self.engine.indices[0][b_mask] < 512)
                    print(f"  Motor TD grads mean=|{self.engine.w_surface[t_mask][td_motor].abs().mean().item():.6f}|")
                    print(f"  Motor BU grads mean=|{self.engine.w_surface[b_mask][bu_motor].abs().mean().item():.6f}|")

                # 8d. Cascade transfer — step-count + magnitude gated.
                # Surface must integrate gradients autonomously for at least
                # 500 steps before any magnitude bleeds into mid-layer.
                # Magnitude gating (>0.02 threshold) is handled inside
                # cascade_transfer() to ensure w_surface accumulates a
                # substantial structural representation before transfer.
                if step < 500:
                    pass  # No transfer — let surface accumulate first
                elif step < 5000:
                    # Surface→mid only (magnitude-gated inside)
                    self.engine.cascade_transfer(include_deep=False)
                else:
                    # Full cascade (magnitude-gated inside)
                    self.engine.cascade_transfer()

                # 8e. Synaptic intelligence tracking
                self.engine.update_synaptic_intelligence(current_loss=energy)

            # 8f. Periodic synaptic intelligence consolidation
            if step > 0 and step % self.si_consolidation_interval == 0:
                self.engine.consolidate_importance()

            # 8g. Sleep replay phase for memory consolidation
            if step > 0 and step % self.sleep_interval == 0:
                self.sleep_phase(num_replay_cycles=200, replay_lr_mult=0.1)

            # 9. VICReg Regularization logic removed.
            # The previous logic incorrectly subtracted a scalar positive loss 
            # value directly from the network biases, driving all biases to -1.0 
            # after ~8000 steps and destroying network activations.
            # Representational decorrelation is already properly handled by
            # the anti-Hebbian Oja rule and k-WTA inside engine_torch.py.

            # 10. Logging
            if step % 100 == 0:
                if len(acc_window) > 1000: acc_window = acc_window[-1000:]
                if len(acc_top3_window) > 1000: acc_top3_window = acc_top3_window[-1000:]

                elapsed = time.time() - start_time

                if step % 1000 == 0:
                    acc_1k = sum(acc_window) / len(acc_window) if acc_window else 0.0
                    acc3_1k = sum(acc_top3_window) / len(acc_top3_window) if acc_top3_window else 0.0
                    avg_loss = loss_accum / max(step, 1)
                    avg_energy = energy_accum / max(step, 1)

                    # Per-level error breakdown
                    level_errors = self.engine.get_prediction_error_by_level()
                    err_str = " | ".join(
                        f"L{l}: s={d['spatial']:.4f} t={d['temporal']:.4f}"
                        for l, d in sorted(level_errors.items())
                    )

                    print(f"Step {step}/{total_steps} | Time: {elapsed:.0f}s | "
                          f"AvgLoss: {avg_loss:.4f} | Energy: {avg_energy:.4f} | "
                          f"Acc@1k: {acc_1k:.2%} | Top3@1k: {acc3_1k:.2%} | "
                          f"||s||/√N: {state_norm:.4f}")
                    print(f"  PredErr: {err_str}")

                    # Cascade distribution diagnostic
                    cstats = self.engine.cascade_stats()
                    print(f"  Cascade: surface={cstats['surface']:.4f} "
                          f"mid={cstats['mid']:.4f} deep={cstats['deep']:.4f}")

                    # Diagnostic: top-down weight stats per level pair.
                    # If L1 spatial error is stuck high (~5.4), check whether
                    # top-down weights from L1→L0 are actually updating.
                    td_stats = self.engine.get_topdown_weight_stats()
                    td_parts = [f"L{s}→L{d}: μ={v['mean']:.4f} σ={v['std']:.4f}"
                                for (s, d), v in sorted(td_stats.items())]
                    if td_parts:
                        print(f"  TopDown: {' | '.join(td_parts)}")

                    # Convergence and top-down prediction diagnostics
                    print(f"  Settle: last_diff={self.engine.last_settle_diff:.6f} | "
                          f"TD pred var: {self.engine.topdown_pred_var:.6f}")

                    # Per-level state variance after settling
                    level_vars = []
                    for level in range(self.engine.max_level + 1):
                        lmask = self.engine.node_to_level == level
                        lvar = self.engine.state[lmask].var().item()
                        level_vars.append(f"L{level}={lvar:.6f}")
                    print(f"  State var: {' | '.join(level_vars)}")
                else:
                    recent_acc = sum(acc_window[-100:]) / min(len(acc_window), 100)
                    recent_acc3 = sum(acc_top3_window[-100:]) / min(len(acc_top3_window), 100)
                    print(f"  step {step}/{total_steps} | Loss: {loss:.4f} | "
                          f"Energy: {energy:.4f} | Acc: {recent_acc:.2%} | "
                          f"Top3: {recent_acc3:.2%} | ||s||/√N: {state_norm:.4f}        ", end='\r')

        final_acc = sum(acc_window)/len(acc_window) if len(acc_window) > 0 else 0.0
        final_acc3 = sum(acc_top3_window)/len(acc_top3_window) if len(acc_top3_window) > 0 else 0.0
        print(f"\nPhase Complete. Avg Loss: {loss_accum/total_steps:.4f} | "
              f"Avg Energy: {energy_accum/total_steps:.4f} | Final Acc: {final_acc:.2%}")

    def compute_network_energy(self):
        """Compute total network energy (sum of squared states + weight interactions)."""
        rho = torch.tanh(self.engine.state)
        cascade_weights = self.engine.effective_weights
        weights = torch.sparse_coo_tensor(
            self.engine.indices, cascade_weights,
            (self.num_nodes, self.num_nodes))
        interaction = -0.5 * torch.dot(rho, torch.mv(weights, rho))
        field = -torch.dot(self.engine.biases, rho)
        return (interaction + field).item()

    def sleep_phase(self, num_replay_cycles=200, replay_lr_mult=0.1):
        """
        Offline consolidation: replay learned patterns by settling
        from noise without external input, then strengthen attractors.

        The energy-based architecture is inherently generative — disconnecting
        input and settling into energy minima naturally replays learned patterns.
        """
        print("\n--- Sleep Phase: Consolidation ---")

        # Save current state
        awake_state = self.engine.state.clone()

        # Identify Level 3 nodes for targeted perturbation
        level3_mask = self.engine.node_to_level == 3

        for cycle in range(num_replay_cycles):
            # 1. Initialize with low-amplitude noise biased toward recent activity
            noise = torch.randn(self.num_nodes, device=self.device) * 0.1
            
            # Add targeted high-variance perturbation to Level 3 nodes to break mode collapse
            noise[level3_mask] = torch.randn(level3_mask.sum(), device=self.device) * 0.5
            
            replay_init = awake_state * 0.05 + noise
            self.engine.state = replay_init

            # 2. Settle with NO external input, NO input clamping
            #    The network falls into a learned attractor — a "memory"
            input_vec = torch.zeros(self.num_nodes, device=self.device)
            self.engine.settle(input_vec, max_steps=50)

            # 3. Strengthen this attractor with Hebbian update
            rho = torch.tanh(self.engine.state)
            idx_i = self.engine.indices[0]
            idx_j = self.engine.indices[1]
            ri = rho[idx_i]
            rj = rho[idx_j]

            # Only update the mid level during sleep (surface is for online learning)
            # Use Oja's rule to prevent unbounded exponential weight explosion.
            # CRITICAL FIX: The decay term MUST use effective_weights! If we only
            # decay based on w_mid, then w_mid will grow large enough to balance
            # the Hebbian term on its own, ignoring the already-large w_deep!
            w_eff = self.engine.effective_weights
            hebbian_update = replay_lr_mult * 0.001 * (ri * rj - w_eff * rj.pow(2))
            self.engine.w_mid += hebbian_update

            # 4. CLS: hippocampal-to-neocortical transfer during replay
            #    Hippocampal modules replay their rapidly-learned patterns,
            #    and the replay signal trains the neocortical modules.
            for hippo_mod_id in self.hippocampal_modules:
                hippo_mod = self.graph.modules[hippo_mod_id]
                hippo_start, hippo_end = hippo_mod['start'], hippo_mod['end']
                hippo_activity = rho[hippo_start:hippo_end]

                for neo_mod_id in self.graph.get_connected_neocortical(hippo_mod_id):
                    neo_mod = self.graph.modules[neo_mod_id]
                    neo_start, neo_end = neo_mod['start'], neo_mod['end']
                    neo_activity = rho[neo_start:neo_end]

                    # Find edges between hippo and neo modules and strengthen them
                    # Use a stronger Hebbian signal for hippo→neo transfer
                    hippo_mean = hippo_activity.mean()
                    neo_update = replay_lr_mult * 0.005 * hippo_mean * neo_activity
                    self.engine.biases[neo_start:neo_end] += neo_update.clamp(-0.01, 0.01)

            # Prevent unbounded bias accumulation during 200 sleep cycles
            self.engine.biases.clamp_(-1.0, 1.0)

            # Log every 50 cycles
            if cycle % 50 == 0:
                energy = self.compute_network_energy()
                num_active = (rho.abs() > 0.3).sum().item()
                print(f"  Replay {cycle}/{num_replay_cycles} | "
                      f"Energy: {energy:.4f} | Active nodes: {num_active}/{self.num_nodes}")

        # Restore awake state (partial — allow some sleep influence)
        self.engine.state = awake_state * 0.5

        # Consolidate synaptic intelligence after sleep
        self.engine.consolidate_importance()

        print("--- Sleep Phase Complete ---")

    def evaluate_retention(self, data_path, num_samples=5000):
        """
        Test accuracy on earlier curriculum data without updating weights.
        Used to measure backward transfer (retention of earlier learning).
        """
        ensure_data(data_path, "retention_eval")
        if not os.path.exists(data_path):
            print(f"Cannot evaluate retention: {data_path} not found")
            return 0.0

        with open(data_path, 'rb') as f:
            data = f.read()

        data_len = len(data)
        num_samples = min(num_samples, data_len - 1)

        correct = 0
        correct_top3 = 0

        # Save state to restore after evaluation
        saved_state = self.engine.state.clone()

        for i in range(num_samples):
            idx = i % (data_len - 1)
            input_byte = data[idx]
            target_byte = data[idx + 1]

            input_vec = torch.zeros(self.num_nodes, device=self.device)
            input_vec[0:256] = self.eye[input_byte] * 2.0

            input_mask = torch.zeros(self.num_nodes, device=self.device)
            input_mask[self.input_indices] = 1.0

            self.engine.settle(input_vec, input_mask=input_mask, max_steps=10)

            level0_acts = torch.tanh(self.engine.state[self.level0_indices])
            features = level0_acts
            logits = self.readout_W @ features + self.readout_b
            probs = torch.softmax(logits, dim=0)
            pred_idx = torch.argmax(probs).item()

            if pred_idx == target_byte:
                correct += 1
            _, top3 = torch.topk(probs, 3)
            if target_byte in top3.tolist():
                correct_top3 += 1

        # Restore state
        self.engine.state = saved_state

        acc = correct / num_samples
        acc3 = correct_top3 / num_samples
        print(f"Retention eval on {data_path}: Acc={acc:.2%} Top3={acc3:.2%} ({num_samples} samples)")
        return acc

    def generate(self, start_text="The", length=100):
        print(f"\n--- Generating: {start_text} ... ---")
        curr_text = start_text

        # Prime
        last_byte = ord(start_text[-1]) if start_text else 0
        for char in start_text:
            val = ord(char)
            if val > 255: val = 0
            last_byte = val
            input_vec = torch.zeros(self.num_nodes, device=self.device)
            input_vec[0:256] = self.eye[val] * 2.0
            self.engine.settle(input_vec, max_steps=10)

        for _ in range(length):
            level0_acts = torch.tanh(self.engine.state[self.level0_indices])
            features = level0_acts
            logits = self.readout_W @ features + self.readout_b
            probs = torch.softmax(logits, dim=0)

            next_byte = torch.multinomial(probs, 1).item()
            last_byte = next_byte
            char = chr(next_byte) if 0 <= next_byte < 128 else '?'
            curr_text += char

            input_vec = torch.zeros(self.num_nodes, device=self.device)
            input_vec[0:256] = self.eye[next_byte] * 2.0
            self.engine.settle(input_vec, max_steps=10)

        print(curr_text)
        print("--------------------------------------")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--nodes", type=int, default=2000)
    parser.add_argument("--modules", type=int, default=20)
    args = parser.parse_args()

    device = args.device
    if torch.backends.mps.is_available() and device == 'cpu':
        device = 'mps'
    if torch.cuda.is_available() and device == 'cpu':
        device = 'cuda'

    print(f"Using device: {device}")

    trainer = SequentialTrainer(num_nodes=args.nodes, device=device, num_modules=args.modules)

    # Reset state before curriculum begins
    trainer.engine.state.zero_()

    # Phase 1: Chars
    # lr=0.05: compensates for tiny activation products
    # settle_steps=30: L1 (tau=0.5) and L2 (tau=2.0) need time to respond.
    #   With dt=0.5 and 8 steps, only 4 time units elapsed — L2 barely moved.
    #   30 steps = 15 time units, enough for L1/L2 to participate.
    # input_gain=5.0: stronger input to overcome recurrent attractor
    trainer.train_phase("Chars", "ndcd/data/level1_chars.txt",
                        iterations=500, steps_per_iter=100, lr=0.05,
                        settle_steps=30, input_gain=0.5)
    trainer.generate(start_text="A")

    # --- Phase boundary: Chars → Words ---
    # Chars w_deep encodes deterministic cycle positions (a→b→c...) which
    # don't transfer to stochastic word tasks. Keep w_deep intact as
    # inductive bias (letter representations) but reset transients.
    # DO NOT call tune_spectral_radius() here — after 50K steps the spectral
    # radius is ~124, and rescaling by 0.006x destroys all learned weights.
    trainer.phase_boundary_reset("Chars", "Words")

    # Phase 2: Words — evaluate retention on Chars after training
    # settle_steps=30: L1 (tau=0.75) and L2 (tau=5.0) need more time
    # to participate. With dt=0.5 and 8 steps, only 4 effective time units
    # elapse — L2 barely moves. 30 steps = 15 time units, enough for L1/L2.
    trainer.train_phase("Words", "ndcd/data/level2_words.txt",
                        iterations=200, steps_per_iter=100, lr=0.05,
                        settle_steps=30, input_gain=0.5, warmup_steps=5000)
    trainer.evaluate_retention("ndcd/data/level1_chars.txt")
    trainer.generate()

    # --- Phase boundary: Words → Quotes ---
    trainer.phase_boundary_reset("Words", "Quotes")

    # Phase 3: Quotes — evaluate retention on Chars and Words
    # settle_steps=50: Quotes need L2 (tau=5.0) and L3 (tau=25.0) to
    # participate. 50 steps = 25 time units, enough for L2 to fully settle
    # and L3 to begin contributing sentence-level context.
    trainer.train_phase("Quotes", "ndcd/data/level3_quotes.txt",
                        iterations=200, steps_per_iter=200, lr=0.05,
                        settle_steps=50, input_gain=0.5, warmup_steps=5000)
    trainer.evaluate_retention("ndcd/data/level1_chars.txt")
    trainer.evaluate_retention("ndcd/data/level2_words.txt")
    trainer.generate()

    # --- Phase boundary: Quotes → Literature ---
    trainer.phase_boundary_reset("Quotes", "Literature")

    # Phase 4: Literature — evaluate retention on all prior phases
    # settle_steps=50: Same as quotes — full hierarchy participation needed.
    trainer.train_phase("Literature", "ndcd/data/sherlock.txt",
                        iterations=500, steps_per_iter=500, lr=0.05,
                        settle_steps=50, input_gain=0.5, warmup_steps=5000)
    trainer.evaluate_retention("ndcd/data/level1_chars.txt")
    trainer.evaluate_retention("ndcd/data/level2_words.txt")
    trainer.evaluate_retention("ndcd/data/level3_quotes.txt")
    trainer.generate(start_text="Sherlock", length=200)

if __name__ == "__main__":
    main()
