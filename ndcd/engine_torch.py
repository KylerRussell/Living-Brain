import numpy as np
import torch
from typing import Optional, Tuple, List


@torch.jit.script
def jit_solve_dynamics_imex(
    initial_state: torch.Tensor,
    indices: torch.Tensor,
    weight_values: torch.Tensor,
    biases: torch.Tensor,
    taus: torch.Tensor,
    input_vector: torch.Tensor,
    dt: float,
    max_steps: int,
    tol: float,
    input_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, float]:
    """
    Semi-implicit (IMEX) dynamics solver for predictive coding.

    Replaces RK4 with an implicit-explicit scheme:
    - Implicit: linear decay term (-x / tau)
    - Explicit: nonlinear interaction (W*tanh(x) + bias + input) / tau

    Update rule:
        x_{n+1} = (x_n + dt * nonlinear_term / tau) / (1 + dt / tau)

    This is unconditionally stable on the linear part, allowing dt=0.5-1.0
    and convergence in 10-20 steps instead of 100 RK4 steps.

    Returns (settled_state, final_diff) where final_diff indicates convergence.
    Includes adaptive dt: halves timestep when diff increases (diverging).
    """
    num_nodes = initial_state.size(0)
    current_s = initial_state.clone()

    weights = torch.sparse_coo_tensor(indices, weight_values, (num_nodes, num_nodes))

    # Pre-compute IMEX denominator: (1 + dt / tau) per node
    current_dt = dt
    imex_denom = 1.0 + current_dt / taus
    min_dt: float = 0.01

    step_count = 0
    diff = tol + 1.0  # Ensure at least one step
    prev_diff: float = 1e6  # Large initial for adaptive dt

    while step_count < max_steps and diff > tol:
        # Hard clamp input nodes
        if input_mask is not None:
            current_s = current_s * (1.0 - input_mask) + input_vector * input_mask

        old_state = current_s.clone()

        # Compute nonlinear term
        rho = torch.tanh(current_s)
        synaptic = torch.mv(weights, rho)
        nonlinear = synaptic + biases + input_vector

        # Semi-implicit update: treats -x/tau implicitly, rest explicitly
        current_s = (current_s + current_dt * nonlinear / taus) / imex_denom

        # State clamping: ±1.5 keeps tanh ≈ 0.91 (18% gradient headroom)
        # ±3.0 gave tanh ≈ 0.995 (<1% headroom → total saturation)
        current_s = current_s.clamp(-1.5, 1.5)

        # Hard clamp input nodes after update
        if input_mask is not None:
            current_s = current_s * (1.0 - input_mask) + input_vector * input_mask

        # Convergence check
        diff = torch.norm(current_s - old_state).item()

        # Adaptive dt: halve timestep if diverging
        if diff > prev_diff and current_dt > min_dt:
            current_dt = current_dt * 0.5
            imex_denom = 1.0 + current_dt / taus

        prev_diff = diff
        step_count += 1

    return current_s, diff


class PredictiveCodingEngine:
    """
    Hierarchical Predictive Coding engine replacing flat EqProp.

    Key changes from DragonEngineTorch:
    1. IMEX integration instead of RK4 (10-20 steps vs 100)
    2. Predictive coding errors instead of energy-based EqProp
    3. Per-module temporal prediction matrices
    4. Local Hebbian weight updates (no nudge/beta phases)
    5. Single settle phase per token (no free/pos/neg phases)
    """

    def __init__(self, num_nodes, indices, values, biases, taus,
                 module_ranges, module_levels, hier_pairs,
                 positions: Optional[np.ndarray] = None,
                 dt=0.5, device='cuda' if torch.cuda.is_available() else 'cpu',
                 temporal_alpha=0.5):
        """
        Args:
            num_nodes: Total number of nodes.
            indices: [2, E] edge index array.
            values: [E] edge weight array.
            biases: [N] bias vector.
            taus: [N] time constants.
            module_ranges: List of (start, end) for each module.
            module_levels: Array of level per module.
            hier_pairs: List of (upper_mod_id, lower_mod_id) pairs.
            positions: Optional [N, 3] spatial positions.
            dt: Integration timestep (0.5-1.0 for IMEX).
            device: Torch device.
            temporal_alpha: Balance between spatial and temporal prediction (0.5).
        """
        self.dt = dt
        self.device = device
        self.num_nodes = num_nodes
        self.temporal_alpha = temporal_alpha

        # Spatial positions
        if positions is not None:
            self.positions = torch.tensor(positions, dtype=torch.float32, device=device)
        else:
            self.positions = torch.zeros((num_nodes, 3), dtype=torch.float32, device=device)

        # Sparse weights (COO)
        self.indices = torch.tensor(indices, dtype=torch.long, device=device)
        self.weight_values = torch.tensor(values, dtype=torch.float32, device=device)

        # Parameters
        self.taus = torch.tensor(taus, dtype=torch.float32, device=device)
        self.biases = torch.tensor(biases, dtype=torch.float32, device=device)

        # State
        self.state = torch.zeros(num_nodes, dtype=torch.float32, device=device)

        # Module metadata
        self.module_ranges = module_ranges  # [(start, end), ...]
        self.module_levels = module_levels  # np array of levels
        self.hier_pairs = hier_pairs        # [(upper_id, lower_id), ...]
        self.num_modules = len(module_ranges)

        # Build module-level lookup tensors for fast access
        self._build_module_tensors()

        # --- Top-down edge mask for proper predictive coding ---
        # In hierarchical predictive coding, the spatial prediction error at
        # level ℓ is: ε_ℓ = x_ℓ − f(W_topdown * x_{ℓ+1})
        # We must isolate top-down edges (higher→lower level) from the full
        # weight matrix. Using all synaptic input (intra-module, lateral,
        # bottom-up) gives the dynamics residual, not a prediction error.
        self.node_to_level = torch.zeros(num_nodes, dtype=torch.long, device=device)
        self.node_to_level[:512] = 0  # I/O nodes at level 0
        for mod_idx, (start, end) in enumerate(self.module_ranges):
            self.node_to_level[start:end] = self.module_levels[mod_idx]
        self.max_level = int(self.node_to_level.max().item())

        # Per-edge level lookups for top-down, bottom-up, and lateral masks
        src_levels = self.node_to_level[self.indices[0]]
        dst_levels = self.node_to_level[self.indices[1]]

        # Top-down: source at strictly higher level than destination
        self.topdown_edge_mask = src_levels > dst_levels
        self.topdown_indices = self.indices[:, self.topdown_edge_mask]

        # Bottom-up: source at strictly lower level than destination
        self.bottomup_edge_mask = src_levels < dst_levels

        # Lateral: same level, excluding I/O nodes (which are at level 0
        # but aren't association nodes)
        self.lateral_edge_mask = (
            (src_levels == dst_levels) &
            (self.indices[0] >= 512) &
            (self.indices[1] >= 512)
        )

        # --- Temporal prediction state ---
        # Previous state buffer per module (for temporal prediction errors)
        self.previous_state = torch.zeros(num_nodes, dtype=torch.float32, device=device)

        # Temporal transition matrices A_ℓ per module
        # Stored as diagonal approximation for efficiency:
        # A_ℓ is a vector of size module_size (diagonal of the full matrix)
        # Full matrix would be module_size x module_size but too expensive
        self.temporal_A = []
        for start, end in self.module_ranges:
            size = end - start
            # Initialize near identity (predict persistence)
            a = torch.ones(size, dtype=torch.float32, device=device) * 0.9
            a += torch.randn(size, dtype=torch.float32, device=device) * 0.05
            self.temporal_A.append(a)

        # --- Metaplastic cascade: 3 timescales per synapse ---
        # Surface: fast, updated every step
        # Mid: medium, τ ≈ 100 steps
        # Deep: slow, τ ≈ 10000 steps
        self.w_deep = self.weight_values.clone()   # Initialize with current learned weights
        self.w_surface = torch.zeros_like(self.weight_values)
        self.w_mid = torch.zeros_like(self.weight_values)

        # Cascade transfer rates — reduced from 100/10000.
        # At tau_mid_to_deep=10000, w_mid≈0.02 transfers only 2e-6 per step.
        # Over 40K steps that accumulates ~0.08 in w_deep — far too slow for
        # meaningful consolidation. At 1000, the same w_mid transfers 2e-5/step
        # → ~0.8 over 40K steps, giving w_deep real content to preserve.
        self.tau_surface_to_mid = 50.0    # was 100
        self.tau_mid_to_deep = 1000.0     # was 10000

        # Metaplastic scaling: how much accumulated deep weight
        # reduces surface learning rate. Reduced from 1.0 to 0.1 because
        # w_deep ≈ 0.83 after chars was causing 1.83x LR reduction (with
        # omega adding another ~10x). At 0.1, the cascade inertia of
        # w_deep already protects important weights without also killing
        # the effective learning rate.
        self.meta_scale = 0.1  # was 1.0

        # --- Synaptic intelligence (Zenke et al., 2017) ---
        self.omega = torch.zeros_like(self.weight_values)       # accumulated importance
        self.prev_weights = self.effective_weights.clone()       # for computing Δw
        self.running_contribution = torch.zeros_like(self.weight_values)  # path integral
        self.si_damping = 0.1  # prevents omega from growing unboundedly

        # Short-term plasticity (Mongillo et al., 2008)
        self.facilitation = torch.ones_like(self.weight_values) * 0.2
        self.depression = torch.ones_like(self.weight_values)
        self.tau_facil = 150.0
        self.tau_depress = 20.0

        # Store latest prediction errors for weight updates
        self.spatial_errors = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        self.temporal_errors = torch.zeros(num_nodes, dtype=torch.float32, device=device)

        # Convergence tracking for training gate (Fix 4)
        self.last_settle_diff = 0.0
        self.topdown_pred_var = 0.0

    def _build_module_tensors(self):
        """Pre-build tensors for module start/end ranges for fast slicing."""
        self.mod_starts = torch.tensor(
            [r[0] for r in self.module_ranges], dtype=torch.long, device=self.device)
        self.mod_ends = torch.tensor(
            [r[1] for r in self.module_ranges], dtype=torch.long, device=self.device)
        self.mod_level_tensor = torch.tensor(
            self.module_levels, dtype=torch.long, device=self.device)

    def activation_function(self, s):
        return torch.tanh(s)

    @property
    def effective_weights(self):
        """Effective weight is always the sum of all three cascade levels."""
        return self.w_surface + self.w_mid + self.w_deep

    def cascade_transfer(self):
        """Call once per training step after weight update.

        Transfers weight magnitude downward through the cascade:
        surface → mid (fast) and mid → deep (slow).
        """
        # Surface → Mid (fast transfer)
        transfer_sm = self.w_surface / self.tau_surface_to_mid
        self.w_mid += transfer_sm
        self.w_surface -= transfer_sm

        # Mid → Deep (slow transfer)
        transfer_md = self.w_mid / self.tau_mid_to_deep
        self.w_deep += transfer_md
        self.w_mid -= transfer_md

    def update_synaptic_intelligence(self, current_loss):
        """Call after each weight update with the current prediction error.

        Tracks how much each synapse contributes to reducing prediction error,
        giving a second importance signal alongside metaplastic depth.
        """
        # Compute weight change since last call
        delta_w = self.effective_weights - self.prev_weights

        # Approximate gradient contribution: -loss * delta_w
        self.running_contribution += -current_loss * delta_w

        self.prev_weights = self.effective_weights.clone()

    def consolidate_importance(self):
        """Call at phase boundaries or every ~10K steps.

        Transfers running contribution to permanent importance (omega).
        """
        delta_w_total = self.effective_weights - self.prev_weights
        # Normalize by total weight change to get per-unit importance
        self.omega += torch.relu(self.running_contribution) / (delta_w_total.pow(2) + 1e-6)

        # Decay old importance slowly to allow forgetting truly obsolete knowledge
        self.omega *= (1.0 - self.si_damping)

        # Reset accumulator
        self.running_contribution.zero_()

    def settle(self, input_vector, max_steps=20, tol=1e-3,
               input_mask=None):
        """
        Single-phase settling using IMEX integration.

        No nudge/beta needed — predictive coding computes errors
        as part of the dynamics, and learning is purely local.
        """
        if not isinstance(input_vector, torch.Tensor):
            input_vector = torch.tensor(input_vector, dtype=torch.float32, device=self.device)

        input_m = None
        if input_mask is not None:
            if not isinstance(input_mask, torch.Tensor):
                input_m = torch.tensor(input_mask, dtype=torch.float32, device=self.device)
            else:
                input_m = input_mask

        # Compute effective weights: cascade sum modulated by short-term plasticity
        cascade_weights = self.effective_weights
        effective_weights = cascade_weights * self.facilitation * self.depression

        # Single-phase IMEX settling
        self.state, self.last_settle_diff = jit_solve_dynamics_imex(
            self.state,
            self.indices,
            effective_weights,
            self.biases,
            self.taus,
            input_vector,
            self.dt,
            max_steps,
            tol,
            input_m,
        )

        # Update short-term plasticity after settling
        self._update_short_term_plasticity()

        return self.activation_function(self.state)

    def compute_prediction_errors(self):
        """
        Compute hierarchical prediction errors.

        Spatial prediction error at level ℓ:
            ε_ℓ = x_ℓ − W_topdown * tanh(x_{ℓ+1})

        Uses ONLY top-down edges (higher→lower level) for the prediction,
        not the full weight matrix. The full synaptic input includes
        intra-module, lateral, and bottom-up contributions which are part
        of the dynamics, not the hierarchical prediction.

        I/O nodes (0-511) get zero spatial error here — the output target
        is injected separately by the training loop.
        Top-level nodes also get zero (no parent to predict them).

        Returns total prediction error energy F = 0.5 * sum(||ε||²)
        """
        rho = torch.tanh(self.state)

        # Build sparse matrix with ONLY top-down edges
        cascade_weights = self.effective_weights
        td_vals = (cascade_weights[self.topdown_edge_mask] *
                   self.facilitation[self.topdown_edge_mask] *
                   self.depression[self.topdown_edge_mask])
        td_sparse = torch.sparse_coo_tensor(
            self.topdown_indices, td_vals,
            (self.num_nodes, self.num_nodes))

        # Top-down prediction: what higher levels predict for lower levels
        topdown_pred = torch.mv(td_sparse, rho)

        # Spatial error = actual state - top-down prediction
        self.spatial_errors = self.state - topdown_pred

        # Store top-down prediction variance for diagnostics
        self.topdown_pred_var = topdown_pred[512:].var().item()

        # Zero errors for nodes without meaningful top-down prediction:
        # - I/O nodes: input is clamped; output target injected by trainer
        self.spatial_errors[:512] = 0
        # - Top-level nodes: no parent level predicts them (they are the prior)
        top_mask = self.node_to_level == self.max_level
        self.spatial_errors[top_mask] = 0

        spatial_energy = 0.5 * torch.sum(self.spatial_errors ** 2).item()

        # --- Temporal prediction errors ---
        # ε_temporal_ℓ = x_ℓ(t) - A_ℓ * x_ℓ(t-1)
        self.temporal_errors.zero_()
        temporal_energy = 0.0

        for mod_idx, (start, end) in enumerate(self.module_ranges):
            current = self.state[start:end]
            prev = self.previous_state[start:end]
            a_diag = self.temporal_A[mod_idx]

            # Temporal prediction: A * previous_state
            temporal_pred = a_diag * prev
            t_error = current - temporal_pred
            self.temporal_errors[start:end] = t_error

            temporal_energy += 0.5 * torch.sum(t_error ** 2).item()

        # Total free energy
        total_energy = spatial_energy + self.temporal_alpha * temporal_energy

        return total_energy

    def update_weights_predictive(self, learning_rate=0.01):
        """
        Local Hebbian weight update based on prediction errors.

        Spatial weight update:
            ΔW_ij ∝ meta_lr * ε_{lower_j} * tanh(x_upper_i)^T

        Updates go to w_surface only. The metaplastic scaling and synaptic
        intelligence reduce per-synapse learning rate for consolidated and
        important synapses, preventing catastrophic forgetting.

        Temporal transition update:
            ΔA_ℓ ∝ lr * ε_temporal * x_prev^T
        """
        with torch.no_grad():
            rho = torch.tanh(self.state)

            # --- Spatial weight update (top-down + bottom-up + lateral) ---
            idx_i = self.indices[0]
            idx_j = self.indices[1]

            grad = torch.zeros_like(self.weight_values)

            # Top-down edges: learn to predict lower-level states
            # ΔW_td ∝ ε_lower × tanh(x_upper)
            td_error_j = self.spatial_errors[idx_j[self.topdown_edge_mask]]
            td_rho_i = rho[idx_i[self.topdown_edge_mask]]
            grad[self.topdown_edge_mask] = td_error_j * td_rho_i

            # Bottom-up edges: learn to propagate prediction errors upward.
            # Without this, BU weights stay at random init and higher levels
            # never receive structured input from lower levels.
            # ΔW_bu ∝ ε_higher × tanh(x_lower)
            bu_error_j = self.spatial_errors[idx_j[self.bottomup_edge_mask]]
            bu_rho_i = rho[idx_i[self.bottomup_edge_mask]]
            grad[self.bottomup_edge_mask] = bu_error_j * bu_rho_i

            # Lateral edges: Oja's rule (Hebbian with self-normalizing decay).
            # Pure Hebbian (rho_i * rho_j) always strengthens co-active
            # connections. With L1 saturation (all nodes at +1.5), every
            # lateral pair has rho_i*rho_j ≈ 0.81 every step, driving all
            # lateral weights uniformly positive → single attractor (Test
            # 4B: sim=0.994). Oja's rule adds -w * rho_j^2 decay that
            # self-normalizes weights and prevents uniform growth.
            lat_rho_i = rho[idx_i[self.lateral_edge_mask]]
            lat_rho_j = rho[idx_j[self.lateral_edge_mask]]
            lat_w = self.effective_weights[self.lateral_edge_mask]
            grad[self.lateral_edge_mask] = 0.1 * (lat_rho_i * lat_rho_j - lat_w * lat_rho_j.pow(2))

            # Gradient clipping
            grad = grad.clamp(-1.0, 1.0)

            # Combined importance-aware learning rate (metaplastic + SI)
            consolidation = torch.abs(self.w_deep)
            importance = self.omega
            meta_lr = learning_rate / (1.0 + self.meta_scale * consolidation + 0.1 * importance)

            # Apply update only to surface level
            self.w_surface += meta_lr * grad
            # Keep total effective weight in bounds.
            # ±3.0 gives 3.0 - 0.93 = 2.07 headroom for surface weights.
            # The old ±1.0 clamp left only 0.07 headroom (w_deep=0.83 +
            # w_mid=0.10 = 0.93), causing every surface update to be
            # immediately clipped — the reservoir was physically unable to
            # change. The ±1.5 state clamp in IMEX prevents activation
            # saturation regardless of weight magnitude.
            effective = self.effective_weights
            effective.clamp_(-3.0, 3.0)
            # Redistribute clamped values back
            self.w_surface = effective - self.w_mid - self.w_deep

            # Frobenius norm soft control on transient weights only.
            # The previous version scaled all cascade levels (including w_deep),
            # which destroyed long-term memory: w_deep went from 0.053 to 0.0001
            # over 50K steps because the norm control fired every step and
            # repeatedly eroded deep weights. w_deep is long-term memory and
            # must NEVER be scaled by runtime norm control.
            transient_frob = (self.w_surface + self.w_mid).norm().item()
            target_frob = 0.95 * np.sqrt(self.num_nodes)
            if transient_frob > target_frob * 2.0:  # More lenient threshold
                scale = target_frob / transient_frob
                self.w_surface *= scale
                self.w_mid *= scale
                # w_deep is NEVER scaled — it is long-term memory

            # --- Top-down weight diversity regularization ---
            # Prevents mode collapse where all outgoing top-down weights from
            # a source node converge to the same value (σ→0). After spectral
            # retuning nukes everything, surface weights re-grow fast and
            # collapse to a low-diversity attractor. This nudges each source
            # node's top-down weights toward zero mean (1% per step).
            td_idx = torch.where(self.topdown_edge_mask)[0]
            td_src = self.topdown_indices[0]
            td_vals = self.w_deep[td_idx]

            # Vectorized per-source-node mean via scatter
            unique_src, inverse = torch.unique(td_src, return_inverse=True)
            src_sums = torch.zeros(len(unique_src), device=self.device)
            src_counts = torch.zeros(len(unique_src), device=self.device)
            src_sums.scatter_add_(0, inverse, td_vals)
            src_counts.scatter_add_(0, inverse, torch.ones_like(td_vals))
            src_means = src_sums / src_counts.clamp(min=1)

            # Subtract 1% of each source node's mean from its outgoing weights
            correction = src_means[inverse] * 0.01
            self.w_deep[td_idx] -= correction

            # --- Top-down weight norm constraint ---
            # Prevent TD weight divergence when lower levels are saturated.
            # Without this, L1→L0 σ grows 0.076 → 1.49 in 20K steps because
            # saturated L1 gives constant tanh ≈ ±0.9 and errors never decrease.
            td_effective = self.effective_weights[self.topdown_edge_mask]
            td_dst = self.topdown_indices[1]

            # Per-destination-node: compute RMS of incoming TD weights
            unique_dst_td, inverse_td = torch.unique(td_dst, return_inverse=True)
            dst_sq_sums = torch.zeros(len(unique_dst_td), device=self.device)
            dst_counts_td = torch.zeros(len(unique_dst_td), device=self.device)
            dst_sq_sums.scatter_add_(0, inverse_td, td_effective.pow(2))
            dst_counts_td.scatter_add_(0, inverse_td, torch.ones_like(td_effective))
            dst_rms = torch.sqrt(dst_sq_sums / dst_counts_td.clamp(min=1))

            # Soft clip: scale down weights for nodes where RMS > 1.0
            max_rms = 1.0
            scale_per_dst = torch.where(dst_rms > max_rms, max_rms / dst_rms, torch.ones_like(dst_rms))
            weight_scale = scale_per_dst[inverse_td]

            # Apply to all cascade levels proportionally
            self.w_surface[self.topdown_edge_mask] *= weight_scale
            self.w_mid[self.topdown_edge_mask] *= weight_scale
            self.w_deep[self.topdown_edge_mask] *= weight_scale

            # --- Bias update from prediction errors ---
            # Biases absorb mean prediction errors
            bias_grad = self.spatial_errors + self.temporal_alpha * self.temporal_errors
            self.biases += learning_rate * 0.1 * bias_grad
            self.biases.clamp_(-1.0, 1.0)

            # --- Temporal transition matrix update ---
            # ΔA_ℓ ∝ lr * ε_temporal * x_prev (diagonal approximation)
            for mod_idx, (start, end) in enumerate(self.module_ranges):
                t_error = self.temporal_errors[start:end]
                prev = self.previous_state[start:end]

                delta_a = learning_rate * t_error * prev
                delta_a = delta_a.clamp(-0.1, 0.1)
                self.temporal_A[mod_idx] += delta_a
                # Keep A values reasonable
                self.temporal_A[mod_idx].clamp_(-1.5, 1.5)

    def store_previous_state(self):
        """Store current state as previous state for temporal prediction."""
        self.previous_state = self.state.clone()

    def _update_short_term_plasticity(self):
        """
        Update facilitation and depression variables based on presynaptic activity.
        """
        with torch.no_grad():
            rho = torch.tanh(self.state)
            pre_activity = rho[self.indices[0]]

            self.facilitation += (
                (-self.facilitation + 0.2) / self.tau_facil
                + 0.1 * pre_activity * (1 - self.facilitation)
            )

            self.depression += (
                (1.0 - self.depression) / self.tau_depress
                - 0.05 * pre_activity * self.facilitation * self.depression
            )

            self.facilitation.clamp_(0.0, 1.0)
            self.depression.clamp_(0.0, 1.0)

    def damp_weights(self, factor=0.9):
        """Damps recurrent weights by a factor (applied to deep level)."""
        self.w_deep *= factor

    def cascade_stats(self):
        """Returns mean absolute magnitude at each cascade level for diagnostics."""
        return {
            'surface': self.w_surface.abs().mean().item(),
            'mid': self.w_mid.abs().mean().item(),
            'deep': self.w_deep.abs().mean().item(),
        }

    def get_prediction_error_by_level(self):
        """
        Returns dict of average prediction error magnitude per level.
        Useful for monitoring whether hierarchy is learning properly.
        """
        errors_by_level = {}
        for mod_idx, (start, end) in enumerate(self.module_ranges):
            level = self.module_levels[mod_idx]
            spatial_err = torch.mean(torch.abs(self.spatial_errors[start:end])).item()
            temporal_err = torch.mean(torch.abs(self.temporal_errors[start:end])).item()

            if level not in errors_by_level:
                errors_by_level[level] = {'spatial': [], 'temporal': [], 'count': 0}
            errors_by_level[level]['spatial'].append(spatial_err)
            errors_by_level[level]['temporal'].append(temporal_err)
            errors_by_level[level]['count'] += 1

        result = {}
        for level, data in errors_by_level.items():
            result[level] = {
                'spatial': np.mean(data['spatial']),
                'temporal': np.mean(data['temporal']),
            }
        return result

    def get_topdown_weight_stats(self):
        """
        Returns mean and std of top-down weights grouped by (src_level, dst_level).

        If L1 spatial error is stuck high, these stats reveal whether
        top-down weights are actually updating or frozen.
        """
        with torch.no_grad():
            td_weights = self.effective_weights[self.topdown_edge_mask]
            src_levels = self.node_to_level[self.topdown_indices[0]]
            dst_levels = self.node_to_level[self.topdown_indices[1]]

            stats = {}
            # Group by (src_level, dst_level) pair
            for src_l in range(self.max_level + 1):
                for dst_l in range(src_l):
                    mask = (src_levels == src_l) & (dst_levels == dst_l)
                    if mask.any():
                        vals = td_weights[mask]
                        stats[(src_l, dst_l)] = {
                            'mean': vals.mean().item(),
                            'std': vals.std().item(),
                        }
            return stats
