import torch
import numpy as np
import os
os.environ["MIOPEN_FIND_MODE"] = "2"
os.environ["MIOPEN_DEBUG_DISABLE_CONV_ALGO_SEARCH"] = "1"
import time
import argparse
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from ndcd.graph import DynamicGraph
from ndcd.engine_torch import PredictiveCodingEngine
from ndcd.curriculum_gen import (
    generate_holophrases,
    generate_slot_and_frame,
    generate_complex_constructions,
    generate_contextual_continuity
)

def ensure_data(data_path, phase_name):
    if not os.path.exists(data_path):
        print(f"Data {data_path} not found. Generating for {phase_name}...")
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        if "level1" in data_path: generate_holophrases(data_path)
        elif "level2" in data_path: generate_slot_and_frame(data_path)
        elif "level3" in data_path: generate_complex_constructions(data_path)
        elif "level4" in data_path: generate_contextual_continuity(data_path)
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
            modules=self.graph.modules,
            positions=self.graph.pos,
            dt=0.5,
            device=device,
            temporal_alpha=0.5,
            is_inhibitory=self.graph.is_inhibitory,
            is_pv=self.graph.is_pv,
            is_sst=self.graph.is_sst,
            is_vip=self.graph.is_vip,
            dg_indices=self.graph.dg_indices,
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
        # Hippocampal: faster deep cascade
        self.engine.tau_deep_hippo = 1000.0
        self.engine.tau_deep_neo = 1000.0
        
        # Apply the edge-wise mask to configure mid_to_deep tau heterogeneous parameters
        self.engine.tau_mid_to_deep = torch.where(
            self.hippo_edge_mask,
            torch.full_like(self.engine.tau_mid_to_deep, self.engine.tau_deep_hippo),
            torch.full_like(self.engine.tau_mid_to_deep, self.engine.tau_deep_neo)
        )

        # Synaptic intelligence consolidation interval
        self.si_consolidation_interval = 10000

        # Sleep phase interval
        self.sleep_interval = 5000

        # Dynamic beta starting value
        self.dynamic_beta = 0.5

    def tune_spectral_radius(self, target_radius=0.95):
        """Tunes the spectral radius of the weight matrix to a target value.

        Only meaningful at initialization (random weights). After training,
        cascade weights accumulate to spectral radii of 50-100+, and rescaling
        by 0.006x lobotomizes the network. Use phase_boundary_reset() instead.
        """
        # Feature 5: Local Homeostatic Scaling (Variance Control) replaces global spectral radius tuning.
        pass

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
        - Apply a decay factor to w_surface and w_mid to preserve residual trace
        - Zero network state (avoid prior-phase attractor lock-in)
        - Zero SI accumulators (omega was double-counting protection)
        - Preserve linear readout so it can track evolving hierarchical structure
        """
        print(f"\n--- Phase boundary: {phase_from} → {phase_to} ---")

        with torch.no_grad():
            # Preserve w_deep (long-term memory), apply decay to transients
            deep_mag = self.engine.w_deep.abs().mean().item()
            self.engine.w_surface *= 0.1
            self.engine.w_mid *= 0.1

            # Flush immediate context with exponential decay instead of hard zeroing
            self.engine.state *= 0.5
            self.engine.previous_state *= 0.5

            # Reset synaptic intelligence — omega double-counts with cascade
            self.engine.omega.zero_()
            self.engine.running_contribution.zero_()
            
            # Reset stabilization states
            self.engine.threshold_adaptation.fill_(1.0)
            self.engine.inh_depression_soma.fill_(1.0)
            self.engine.inh_depression_dend.fill_(1.0)
            self.engine.apical_bp1.zero_()
            self.engine.apical_bp2.zero_()

            # Readout is preserved intact — dedicated SGD realignment
            # (realign_readout) is called after phase_boundary_reset to
            # properly recalibrate against the new task distribution.

        print(f"  w_deep preserved (|w_deep|={deep_mag:.4f}), "
              f"w_surface/w_mid decayed by 0.1, state decayed to 0.5, readout preserved for realignment")

    def realign_readout(self, data_path, num_iterations=2000, lr=0.01):
        """Dedicated high-iteration SGD realignment of the linear readout.

        Freezes all recurrent weights and runs SGD on just the readout layer
        to align it with the current reservoir representations. Called at
        phase boundaries and after significant topological shifts in the
        core recurrent reservoir.
        """
        ensure_data(data_path, "readout_realignment")
        if not os.path.exists(data_path):
            print(f"  Cannot realign readout: {data_path} not found")
            return

        with open(data_path, 'rb') as f:
            data = f.read()
        data_len = len(data)
        if data_len < 2:
            return

        print(f"  Realigning readout ({num_iterations} SGD steps on {data_path})...")

        saved_state = self.engine.state.clone()
        self.engine.zero_states()

        acc_count = 0
        
        accum_grad_W = torch.zeros_like(self.readout_W)
        accum_grad_b = torch.zeros_like(self.readout_b)
        for i in range(num_iterations):
            idx = i % (data_len - 1)
            input_byte = data[idx]
            target_byte = data[idx + 1]

            input_vec = torch.zeros(self.num_nodes, device=self.device)
            input_vec[0:256] = self.eye[input_byte] * 0.5
            input_mask = torch.zeros(self.num_nodes, device=self.device)
            input_mask[self.input_indices] = 1.0

            # Word boundary EMA reset (consistent with training loop)
            if input_byte in (32, 10, 13, 9):
                fast_mask = self.engine.taus < 0.5
                self.engine.state[fast_mask] *= 0.5

            self.engine.store_previous_state()
            self.engine.settle(input_vec, input_mask=input_mask, max_steps=200, tol=1e-5)

            with torch.no_grad():
                level0_acts = torch.tanh(self.engine.state[self.level0_indices])
                features = torch.nn.functional.layer_norm(level0_acts, level0_acts.size())
                logits = self.readout_W @ features + self.readout_b
                probs = torch.softmax(logits, dim=0)

                target_one_hot = self.eye[target_byte]
                grad = probs - target_one_hot
                
                accum_grad_W += torch.outer(grad, features)
                accum_grad_b += grad
                
                if input_byte in (32, 10, 13, 9):
                    self.readout_W -= lr * accum_grad_W
                    self.readout_b -= lr * accum_grad_b
                    accum_grad_W.zero_()
                    accum_grad_b.zero_()

                if torch.argmax(probs).item() == target_byte:
                    acc_count += 1

        self.engine.state = saved_state
        acc = acc_count / num_iterations
        print(f"  Readout realignment complete. Acc={acc:.2%}")

    def train_phase(self, phase_name, data_path, iterations, steps_per_iter, lr=0.01,
                    settle_steps=20, input_gain=0.5):
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
        """
        print(f"\n=== Starting Phase: {phase_name} ===")
        print(f"Run started at: {time.ctime()}")
        ensure_data(data_path, phase_name)

        if not os.path.exists(data_path):
            print(f"Skipping {phase_name} (Data missing)")
            return

        with open(data_path, 'rb') as f:
            data = f.read()
            
        # Try to load corresponding evaluation data
        eval_data_path = data_path.replace('train/', 'eval/')
        has_eval = os.path.exists(eval_data_path)
        if has_eval:
            with open(eval_data_path, 'rb') as f:
                eval_data = f.read()
        else:
            eval_data = None

        data_len = len(data)
        eval_len = len(eval_data) if eval_data else 0

        start_time = time.time()

        total_steps = iterations * steps_per_iter
        steps_per_epoch = data_len - 1
        
        # Calculate how many full epochs this represents
        epochs = max(1, total_steps // steps_per_epoch) if data_len > 1 else 1

        loss_accum = 0.0
        energy_accum = 0.0
        acc_window = []
        acc_top3_window = []

        overall_step = 0

        # Readout gradient buffer for word-boundary-aligned updates (Strategy 1).
        # Gradients accumulate across within-word characters and are flushed
        # as a batch-average at each word boundary, aligning the readout's
        # credit signal with the word-level consolidation phase.
        accumulated_readout_grad_W = torch.zeros_like(self.readout_W)
        accumulated_readout_grad_b = torch.zeros_like(self.readout_b)
        readout_acc_count = 0
        readout_lr = lr * 0.5
        
        # Neuromodulator running states
        running_dopamine = 1.0       # DA: Reward Prediction Error (RPE) surrogate
        running_acetylcholine = 1.0  # ACh: Expected Uncertainty surrogate
        loss_ema = 0.0

        for epoch in range(epochs):
            print(f"--- Epoch {epoch+1}/{epochs} ---")
            
            for curr_idx in range(steps_per_epoch):
                step = overall_step
                if step >= total_steps:
                    break

                # 1. Get Data Stream
                input_byte = data[curr_idx]
                target_byte = data[curr_idx + 1]

                # Boundary classification for level-gated updates (Strategy 2).
                # Sentence boundaries are a strict superset of word boundaries.
                is_sentence_boundary = input_byte in (46, 63)  # . ?
                is_word_boundary = (input_byte in (32, 10, 13, 9)) or is_sentence_boundary

                # 2. Input Setup — full one-hot across all 256 input nodes.
                # A single node at 5.0 with 255 zeroes wastes input projection
                # bandwidth and gets drowned by ~500 recurrent neighbors.
                input_mask = torch.zeros(self.num_nodes, device=self.device)
                input_mask[self.input_indices] = 1.0

                input_vec = torch.zeros(self.num_nodes, device=self.device)
                input_vec[0:256] = self.eye[input_byte] * input_gain

                # 3. Store previous state for temporal prediction
                self.engine.store_previous_state()

                # 4. Update context EMA before boundary reset, then apply
                # gradual EMA decay instead of harsh 90% reset.
                self.engine.update_context_ema()

                if is_word_boundary:  # space, LF, CR, tab, or sentence-ending punctuation
                    fast_mask = self.engine.taus < 0.5
                    # EMA: retain 50% of fast node state instead of 10%
                    # Preserves cross-clause contextual linking
                    self.engine.state[fast_mask] *= 0.5
                    # Slower nodes (L1+) get gentler decay for long-range context
                    mid_mask = (self.engine.taus >= 0.5) & (self.engine.taus < 2.0)
                    self.engine.state[mid_mask] *= 0.8
                    # Blend context EMA back for continuity
                    self.engine.blend_context(blend_factor=0.2)

                # 5. Single-phase settle (IMEX) -> FREE PHASE
                # Dynamic tolerance: relax to 1e-3 during the first 2000 steps
                # of each phase, minimizing early computational burden while still
                # demanding strict 1e-5 convergence later.
                settle_tol = 1e-3 if step < 2000 else 1e-5
                self.engine.settle(
                    input_vec,
                    input_mask=input_mask,
                    max_steps=settle_steps,
                    tol=settle_tol,
                    implicit_damping=2.0,  # was 3.0; needed to handle SR > 1 but without complete lockdown
                )
                free_state = self.engine.state.clone()
                free_diff = self.engine.last_settle_diff

                # 6. Compute prediction errors (top-down only; I/O zeroed) based on free state
                energy = self.engine.compute_prediction_errors()

                # 6b. Prospective Configuration Phase (Clamped Target)
                output_actual = torch.tanh(free_state[256:512])         # evaluate on free_state
                output_target = self.eye[target_byte]                   # one-hot [256]
                output_error = output_target - output_actual            # observation error
                
                # Clamping mask: target nodes + input nodes are clamped
                pc_mask = input_mask.clone()
                pc_mask[256:512] = 1.0  # Clamp the motor/output nodes
                
                # Vector containing targets
                pc_vec = input_vec.clone()
                pc_vec[256:512] = output_target
                
                self.engine.settle(
                    pc_vec,
                    input_mask=pc_mask,
                    max_steps=settle_steps,
                    tol=settle_tol,
                    implicit_damping=2.0,
                )
                nudge_state = self.engine.state.clone()
                nudge_diff = self.engine.last_settle_diff
                
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
                    features = torch.nn.functional.layer_norm(level0_acts, level0_acts.size())
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

                # 8a. Buffer readout gradient for word-boundary-aligned updates (Strategy 1).
                # Accumulating per-character gradients and flushing their average at each
                # word boundary aligns the readout's credit signal with the word-level
                # consolidation phase rather than injecting per-character noise.
                with torch.no_grad():
                    target_one_hot = self.eye[target_byte]
                    readout_grad = probs - target_one_hot  # softmax CE gradient
                    accumulated_readout_grad_W += torch.outer(readout_grad, features)
                    accumulated_readout_grad_b += readout_grad
                    readout_acc_count += 1

                # 8b. Compute Neuromodulators
                # Dopamine (RPE): If loss is shockingly *lower* than EMA, high DA (positive surprise).
                # Acetylcholine (Uncertainty): If EMA loss is high overall, high ACh (needs faster learning).
                if step == 0:
                    loss_ema = loss
                else:
                    loss_ema = 0.99 * loss_ema + 0.01 * loss
                
                # ACh: High when baseline uncertainty (loss EMA) is high
                ach_target = max(0.1, min(2.0, loss_ema / 2.0))
                running_acetylcholine = 0.9 * running_acetylcholine + 0.1 * ach_target
                
                # DA: High when sudden unexpected success occurs (loss << loss_ema)
                da_target = max(0.1, min(3.0, (loss_ema + 1e-5) / (loss + 1e-5))) 
                running_dopamine = 0.9 * running_dopamine + 0.1 * da_target

                # 8c. Update L0 recurrent weights per-character (Strategy 2).
                # L1-L3 edges are frozen at character timescale; they receive updates
                # only at word and sentence boundaries via consolidated states, ensuring
                # each level's credit assignment matches its natural temporal scale.
                self.engine.update_weights_predictive(
                    free_state,
                    nudge_state,
                    learning_rate=effective_lr,
                    hippo_edge_mask=self.hippo_edge_mask,
                    active_level_max=1,
                    dopamine=running_dopamine,
                    acetylcholine=running_acetylcholine)

                if step < 5:
                    # Inspect the motor weights
                    t_mask = self.engine.topdown_edge_mask
                    b_mask = self.engine.bottomup_edge_mask
                    td_motor = (self.engine.indices[1][t_mask] >= 256) & (self.engine.indices[1][t_mask] < 512)
                    bu_motor = (self.engine.indices[0][b_mask] >= 256) & (self.engine.indices[0][b_mask] < 512)
                    print(f"  Motor TD grads mean=|{self.engine.w_surface[t_mask][td_motor].abs().mean().item():.6f}|")
                    print(f"  Motor BU grads mean=|{self.engine.w_surface[b_mask][bu_motor].abs().mean().item():.6f}|")

                # 8d. Level-gated boundary updates (Strategies 1, 2, 3 composed).
                #
                # Word boundary: extended autonomous settling (Strategy 3) lets L2/L3
                # form stable attractors representing the completed word, then L1+L2
                # weights update from that consolidated state (Strategy 2), and the
                # buffered readout gradient is flushed (Strategy 1).
                if is_word_boundary:
                    _zero = torch.zeros(self.num_nodes, device=self.device)
                    self.engine.settle(_zero, input_mask=None, max_steps=200,
                                       tol=settle_tol, implicit_damping=2.0)
                    word_free_state = self.engine.state.clone()
                    # Clamp target at word boundary: first char of next word (= target_byte)
                    pc_vec = torch.zeros(self.num_nodes, device=self.device)
                    pc_vec[256:512] = self.eye[target_byte]
                    
                    pc_mask = torch.zeros(self.num_nodes, device=self.device)
                    pc_mask[256:512] = 1.0
                    
                    self.engine.settle(pc_vec, input_mask=pc_mask,
                                       max_steps=settle_steps, tol=settle_tol,
                                       implicit_damping=2.0)
                    word_nudge_state = self.engine.state.clone()
                    self.engine.update_weights_predictive(
                        word_free_state, word_nudge_state,
                        learning_rate=effective_lr, hippo_edge_mask=self.hippo_edge_mask,
                        active_level_max=2,
                        dopamine=running_dopamine,
                        acetylcholine=running_acetylcholine)
                    # Flush buffered readout gradient (Strategy 1)
                    if readout_acc_count > 0:
                        self.readout_W -= readout_lr * accumulated_readout_grad_W / readout_acc_count
                        self.readout_b -= readout_lr * accumulated_readout_grad_b / readout_acc_count
                        accumulated_readout_grad_W.zero_()
                        accumulated_readout_grad_b.zero_()
                        readout_acc_count = 0

                # Sentence boundary: longest settling + L3 update + cascade transfer + SI.
                # Cascade transfer is now boundary-triggered rather than per-step so that
                # surface weights have time to accumulate meaningful signal before draining.
                if is_sentence_boundary:
                    _zero = torch.zeros(self.num_nodes, device=self.device)
                    self.engine.settle(_zero, input_mask=None, max_steps=300,
                                       tol=settle_tol, implicit_damping=2.0)
                    sent_free_state = self.engine.state.clone()
                    pc_vec = torch.zeros(self.num_nodes, device=self.device)
                    pc_vec[256:512] = self.eye[target_byte]
                    
                    pc_mask = torch.zeros(self.num_nodes, device=self.device)
                    pc_mask[256:512] = 1.0
                    
                    self.engine.settle(pc_vec, input_mask=pc_mask,
                                       max_steps=settle_steps, tol=settle_tol,
                                       implicit_damping=2.0)
                    sent_nudge_state = self.engine.state.clone()
                    self.engine.update_weights_predictive(
                        sent_free_state, sent_nudge_state,
                        learning_rate=effective_lr, hippo_edge_mask=self.hippo_edge_mask,
                        active_level_max=self.engine.max_level,
                        dopamine=running_dopamine,
                        acetylcholine=running_acetylcholine)
                    # Cascade transfer at sentence boundary (not per-step)
                    surface_mag = self.engine.w_surface.abs().mean().item()
                    if step >= 5000 or surface_mag >= 0.02:
                        self.engine.cascade_transfer(include_deep=(step >= 5000 or surface_mag >= 0.02))
                    # SI consolidation at sentence boundary
                    self.engine.consolidate_importance()

                # 8e. Synaptic intelligence tracking
                self.engine.update_synaptic_intelligence(current_loss=energy)

                # Periodic Jacobian-based spectral radius enforcement.
                if step % 200 == 0:
                    self.engine.enforce_jacobian_spectral_radius(target_max=0.95)

                # 8f. Periodic synaptic intelligence consolidation
                if step > 0 and step % self.si_consolidation_interval == 0:
                    self.engine.consolidate_importance()

                # 8g. Offline Renormalization phase for memory consolidation
                if step > 0 and step % self.sleep_interval == 0:
                    self.offline_renormalization(num_replay_cycles=200, replay_lr_mult=0.1)


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
                        sr_val = getattr(self.engine, 'last_sr', 0.0)
                        print(f"  Cascade: surface={cstats['surface']:.4f} "
                              f"mid={cstats['mid']:.4f} deep={cstats['deep']:.4f} | "
                              f"SR: {sr_val:.4f}")

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

                    # 11. Periodic Evaluation on Holdout Data
                    if step > 0 and step % 5000 == 0 and has_eval and eval_len > 1:
                        print("\n--- Running Evaluation on Holdout Data ---")
                        eval_acc, eval_acc3 = self.evaluate_generalization(eval_data)
                        # Use recent training accuracy to compute Generalization Gap
                        train_acc = sum(acc_window[-5000:]) / max(1, len(acc_window[-5000:]))
                        gap = train_acc - eval_acc
                        ratio = eval_acc / max(1e-8, train_acc)
                        print(f"Generalization Check: Train Acc = {train_acc:.2%}, Eval Acc = {eval_acc:.2%} | Gap: {gap:+.2%} | Ratio: {ratio:.2f}")
                        if ratio < 0.5:
                            print("  WARNING: Ratio < 0.5 indicates pathological memorization!")
                        elif gap > 0.1:
                            print("  NOTE: Significant generalization gap detected.")
                        else:
                            print("  PASS: Strong generalization performance.")

                # 12. Topological shift detection — trigger readout realignment
                # when Jacobian SR shifts by >30%, indicating the reservoir's
                # representational structure has changed significantly.
                if step > 0 and step % 5000 == 0:
                    current_sr = getattr(self.engine, 'last_jacobian_sr', 0.0)
                    prev_sr = getattr(self, '_prev_checkpoint_sr', current_sr)
                    sr_shift = abs(current_sr - prev_sr) / max(prev_sr, 1e-6)
                    if sr_shift > 0.3:
                        print(f"\n  Topological shift detected (SR: {prev_sr:.3f} -> {current_sr:.3f})")
                        self.realign_readout(data_path, num_iterations=500, lr=0.005)
                    self._prev_checkpoint_sr = current_sr

                # FIX 2: overall_step must increment EVERY iteration, not just
                # when step % 100 == 0. Previously it was trapped inside the
                # logging conditional, causing the step counter to freeze at 1
                # after the first increment (since 1 % 100 != 0).
                overall_step += 1
            
            if overall_step >= total_steps:
                break

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

    def offline_renormalization(self, num_replay_cycles=200, replay_lr_mult=0.1):
        """
        Offline consolidation: replay learned patterns by settling
        from noise without external input, then strengthen attractors.
        Also performs global synaptic renormalization (SHY cycle).
        """
        print("\n--- Offline Renormalization: Consolidation & SHY ---")
        
        # 1. Global Synaptic Renormalization (SHY Cycle)
        # Downscales net strength while preserving high-information backbone
        self.engine.offline_renormalization()


        # 2. Pattern Replay and Attractor Reinforcement

        # Save current state
        awake_state = self.engine.state.clone()

        # Identify Level 3 nodes for targeted perturbation
        level3_mask = self.engine.node_to_level == 3

        for cycle in range(num_replay_cycles):
            # 1. Temporal vs Generative Replay Split
            is_rem = (cycle % 4 == 0) # 25% REM, 75% NREM
            
            if is_rem:
                # Generative REM Replay ("Dreaming")
                # Randomize top-level prior (Level 3) with high variance to synthesize
                # pseudo-data, breaking strict anchoring to the last awake frame.
                noise = torch.randn(self.num_nodes, device=self.device) * 0.1
                noise[level3_mask] = torch.randn(level3_mask.sum(), device=self.device) * 1.0
                replay_init = awake_state * 0.05 + noise
                self.engine.state = replay_init
            else:
                # Structured NREM Replay (Temporal & Reverse)
                # Traces the sequences forwards or backwards using the temporal transition matrix
                with torch.no_grad():
                    for mod_idx, (start, end) in enumerate(self.engine.module_ranges):
                        a_diag = self.engine.temporal_A[mod_idx]
                        if cycle < num_replay_cycles // 2:
                            # Forward Temporal Replay
                            self.engine.state[start:end] = (a_diag * self.engine.state[start:end]).clamp(-1.5, 1.5)
                        else:
                            # Reverse Temporal Replay (bounded inverse)
                            inv_a = a_diag.sign() / a_diag.abs().clamp(min=0.1, max=1.0)
                            self.engine.state[start:end] = (inv_a * self.engine.state[start:end]).clamp(-1.5, 1.5)

            # 2. Settle with NO external input, NO input clamping
            #    The network falls into a learned attractor — a "memory"
            #    Max steps increased to 200 to allow top-down cascade to fully settle.
            input_vec = torch.zeros(self.num_nodes, device=self.device)
            self.engine.settle(input_vec, max_steps=200, tol=1e-5)

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
            
            # 5. Structural Plasticity / Memory Purging
            if cycle == num_replay_cycles - 1:
                n_pruned = self.engine.remodel_structure(prune_ratio=0.01)
                print(f"  Structural Plasticity: Pruned and reinitialized {n_pruned} low-utility synapses.")

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

            self.engine.settle(input_vec, input_mask=input_mask, max_steps=100, tol=1e-5)

            level0_acts = torch.tanh(self.engine.state[self.level0_indices])
            features = torch.nn.functional.layer_norm(level0_acts, level0_acts.size())
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

    def evaluate_generalization(self, eval_data, num_samples=5000, max_steps=100):
        """
        Evaluate model using a fully decoupled inference engine to calculate the formal Generalization Gap.
        All learning mechanisms are strictly disabled. Local temporal buffer and state are zeroed.
        """
        import copy
        
        eval_len = len(eval_data)
        num_samples = min(num_samples, eval_len - 1)
        if num_samples <= 0:
            return 0.0, 0.0

        # Create decoupled evaluation instance of engine
        eval_engine = PredictiveCodingEngine(
            self.num_nodes,
            self.indices,
            self.initial_values.copy(), # Only used for shape initially
            self.engine.biases.detach().cpu().numpy(),
            self.engine.taus.detach().cpu().numpy(),
            module_ranges=self.graph.get_module_ranges(),
            module_levels=self.graph.module_levels,
            hier_pairs=self.graph.hier_pairs,
            positions=self.graph.pos,
            dt=self.engine.dt,
            device=self.device,
            temporal_alpha=self.engine.temporal_alpha,
            is_inhibitory=self.graph.is_inhibitory,
            is_pv=self.graph.is_pv,
            is_sst=self.graph.is_sst,
            is_vip=self.graph.is_vip,
        )
        
        # Deep copy current weights
        with torch.no_grad():
            eval_engine.weight_values.copy_(self.engine.weight_values)
            eval_engine.w_deep.copy_(self.engine.w_deep)
            eval_engine.w_mid.copy_(self.engine.w_mid)
            eval_engine.w_surface.copy_(self.engine.w_surface)
            eval_engine.biases.copy_(self.engine.biases)
            eval_engine.temporal_A = copy.deepcopy(self.engine.temporal_A)
            
            # Ensure states are fully zeroed out for clean evaluate
            eval_engine.zero_states()
            eval_engine.previous_state.zero_()
            eval_engine.facilitation.fill_(0.2)
            eval_engine.depression.fill_(1.0)
            
        correct = 0
        correct_top3 = 0

        for i in range(num_samples):
            input_byte = eval_data[i]
            target_byte = eval_data[i + 1]

            input_vec = torch.zeros(self.num_nodes, device=self.device)
            input_vec[0:256] = self.eye[input_byte] * 2.0

            input_mask = torch.zeros(self.num_nodes, device=self.device)
            input_mask[self.input_indices] = 1.0

            eval_engine.store_previous_state()
            
            # Same reset logic
            if input_byte in (32, 10, 13, 9):  # space, LF, CR, tab
                fast_mask = eval_engine.taus < 0.5
                eval_engine.state[fast_mask] *= 0.1

            eval_engine.settle(input_vec, input_mask=input_mask, max_steps=max_steps, tol=1e-5)

            level0_acts = torch.tanh(eval_engine.state[self.level0_indices])
            features = torch.nn.functional.layer_norm(level0_acts, level0_acts.size())
            
            with torch.no_grad():
                logits = self.readout_W @ features + self.readout_b
                probs = torch.softmax(logits, dim=0)
                pred_idx = torch.argmax(probs).item()

            if pred_idx == target_byte:
                correct += 1
            _, top3 = torch.topk(probs, 3)
            if target_byte in top3.tolist():
                correct_top3 += 1

        acc = correct / num_samples
        acc3 = correct_top3 / num_samples
        return acc, acc3

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
            self.engine.settle(input_vec, max_steps=100, tol=1e-5)

        for _ in range(length):
            level0_acts = torch.tanh(self.engine.state[self.level0_indices])
            features = torch.nn.functional.layer_norm(level0_acts, level0_acts.size())
            logits = self.readout_W @ features + self.readout_b
            probs = torch.softmax(logits, dim=0)

            next_byte = torch.multinomial(probs, 1).item()
            last_byte = next_byte
            char = chr(next_byte) if 0 <= next_byte < 128 else '?'
            curr_text += char

            input_vec = torch.zeros(self.num_nodes, device=self.device)
            input_vec[0:256] = self.eye[next_byte] * 2.0
            self.engine.settle(input_vec, max_steps=100, tol=1e-5)

        print(curr_text)
        print("--------------------------------------")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--nodes", type=int, default=25000)
    parser.add_argument("--modules", type=int, default=50)
    args = parser.parse_args()

    device = args.device
    if torch.backends.mps.is_available() and device == 'cpu':
        device = 'mps'
    if torch.cuda.is_available() and device == 'cpu':
        device = 'cuda'

    torch.backends.cuda.matmul.allow_tf32 = True

    print(f"Using device: {device}")

    trainer = SequentialTrainer(num_nodes=args.nodes, device=device, num_modules=args.modules)

    # Reset state before curriculum begins
    trainer.engine.zero_states()

    # Phase 1: Holophrases
    trainer.train_phase("Holophrases", "ndcd/data/level1_holophrases.txt",
                        iterations=500, steps_per_iter=100, lr=0.05,
                        settle_steps=200, input_gain=0.5)
    trainer.generate(start_text="L")

    # --- Phase boundary: Holophrases → Slot-and-Frame ---
    trainer.phase_boundary_reset("Holophrases", "Slot-and-Frame")
    trainer.realign_readout("ndcd/data/train/level2_slot_frame.txt", num_iterations=2000, lr=0.01)

    # Phase 2: Slot-and-Frame
    trainer.train_phase("Slot-and-Frame", "ndcd/data/level2_slot_frame.txt",
                        iterations=500, steps_per_iter=100, lr=0.05,
                        settle_steps=200, input_gain=0.5)
    trainer.evaluate_retention("ndcd/data/level1_holophrases.txt")
    trainer.generate(start_text="W")

    # --- Phase boundary: Slot-and-Frame → Complex Constructions ---
    trainer.phase_boundary_reset("Slot-and-Frame", "Complex Constructions")
    trainer.realign_readout("ndcd/data/train/level3_complex.txt", num_iterations=2000, lr=0.01)

    # Phase 3: Complex Constructions
    trainer.train_phase("Complex Constructions", "ndcd/data/level3_complex.txt",
                        iterations=500, steps_per_iter=100, lr=0.05,
                        settle_steps=200, input_gain=0.5)
    trainer.evaluate_retention("ndcd/data/level1_holophrases.txt")
    trainer.evaluate_retention("ndcd/data/level2_slot_frame.txt")
    trainer.generate(start_text="I")

    # --- Phase boundary: Complex Constructions → Contextual Continuity ---
    trainer.phase_boundary_reset("Complex Constructions", "Contextual Continuity")
    trainer.realign_readout("ndcd/data/train/level4_contextual.txt", num_iterations=2000, lr=0.01)

    # Phase 4: Contextual Continuity
    trainer.train_phase("Contextual Continuity", "ndcd/data/level4_contextual.txt",
                        iterations=500, steps_per_iter=100, lr=0.05,
                        settle_steps=200, input_gain=0.5)
    trainer.evaluate_retention("ndcd/data/level1_holophrases.txt")
    trainer.evaluate_retention("ndcd/data/level2_slot_frame.txt")
    trainer.evaluate_retention("ndcd/data/level3_complex.txt")
    trainer.generate(start_text="I", length=200)

if __name__ == "__main__":
    main()