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
    # FIX 3: Sparsity parameters for soft WTA during settling
    module_starts: torch.Tensor,
    module_ends: torch.Tensor,
    sparsity_alpha: float,  # 0.0 = no sparsity, 0.3 = moderate
    damping: float = 0.15,
    implicit_damping: float = 1.2,
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

    # Pre-compute IMEX denominator: (1 + implicit_damping * dt / tau) per node
    # The implicit_damping coefficient (>1.0) strengthens the implicit decay
    # term, widening the basin of attraction against explicit overshoots
    # from high-energy input projections.
    current_dt = dt
    imex_denom = 1.0 + implicit_damping * current_dt / taus
    min_dt: float = 0.05

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

        # Homeostatic sparsity (L1 penalty) — reduced from 0.01 to 0.001.
        # The old 0.01 coefficient was excessively punitive, driving 33% of
        # nodes into the linear regime of tanh and suppressing distinct
        # attractors. At 0.001, nodes can utilize the full dynamic range
        # of tanh, pushing activations into the saturated extremes required
        # to distinctly separate representations for different inputs.
        current_s -= 0.001 * current_s.sign() * current_dt

        # State clamping: ±1.5 keeps tanh ≈ 0.91 (18% gradient headroom)
        current_s = current_s.clamp(-1.5, 1.5)

        # Hard clamp input nodes after update
        if input_mask is not None:
            current_s = current_s * (1.0 - input_mask) + input_vector * input_mask

        # k-Winner-Take-All (k-WTA) lateral inhibition every 5 steps.
        # Replaces the old soft mean-field approach that was insufficient
        # to break representational singularity (cosine similarity ~0.98).
        # Strict k-WTA forces modules to explicitly select disparate
        # sub-populations of active nodes for differing sensory inputs,
        # generating the orthogonal state vectors needed to separate
        # representations. Top 20% by magnitude win; losers are suppressed
        # to 5% residual (not hard zero, to avoid settling oscillation).
        if sparsity_alpha > 0.0 and step_count % 5 == 4:
            for m in range(module_starts.size(0)):
                ms = module_starts[m].item()
                me = module_ends[m].item()
                mod_s = current_s[ms:me]
                mod_size = me - ms
                k = max(1, mod_size // 5)  # 20% winners
                if mod_size > k:
                    topk_vals = torch.topk(mod_s.abs(), k).values
                    threshold = topk_vals[-1]
                    below = mod_s.abs() < threshold
                    current_s[ms:me] = torch.where(below, mod_s * 0.05, mod_s)
                    current_s[ms:me] = current_s[ms:me].clamp(-1.5, 1.5)

        # Convergence check
        diff = torch.norm(current_s - old_state).item()

        # Relax tolerance slightly during internal steps to avoid infinite halving deadlocks
        if diff <= tol * 1.5:
            break

        # Adaptive dt: smaller steps during volatile phases, larger as it nears steady-state
        if diff > prev_diff and current_dt > min_dt:
            current_dt = max(current_dt * 0.5, min_dt)
            imex_denom = 1.0 + implicit_damping * current_dt / taus
        elif diff < prev_diff * 0.8 and current_dt < dt:
            current_dt = min(current_dt * 1.2, dt)
            imex_denom = 1.0 + implicit_damping * current_dt / taus

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
        self.node_to_level[:512] = -1  # I/O nodes conceptually at level -1
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
        self.w_deep = torch.zeros_like(self.weight_values)
        self.w_surface = torch.zeros_like(self.weight_values)
        self.w_mid = torch.zeros_like(self.weight_values)

        # Cascade transfer rates — significantly increased to allow transient
        # syntactic rules to meaningfully accumulate in surface weights.
        self.tau_surface_to_mid = 10000.0   # was 1000.0
        self.tau_mid_to_deep = 50000.0     # was 2000.0

        # Metaplastic scaling: how much accumulated deep weight
        # reduces surface learning rate. Reduced from 1.0 to 0.1 because
        # w_deep ≈ 0.83 after chars was causing 1.83x LR reduction (with
        # omega adding another ~10x). At 0.1, the cascade inertia of
        # w_deep already protects important weights without also killing
        # the effective learning rate.
        self.meta_scale = 0.1  # was 1.0

        # FIX 1: Track the initial w_deep Frobenius norm as a target.
        # This is the SR-tuned initialization; effective weights should
        # never exceed ~2x this norm during training.
        self._initial_deep_frob = self.weight_values.norm().item()

        # --- Synaptic intelligence (Zenke et al., 2017) ---
        self.omega = torch.zeros_like(self.weight_values)       # accumulated importance
        self.prev_weights = self.effective_weights.clone()       # for computing Δw per step
        self.si_baseline_weights = self.effective_weights.clone() # for computing total Δw over epoch
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

        # FIX 3: Sparsity parameter (controllable from trainer)
        self.sparsity_alpha = 0.05

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
        """Effective weight is base topology + learned cascade deltas."""
        return self.weight_values + self.w_surface + self.w_mid + self.w_deep

    def cascade_transfer(self, include_deep=True):
        """Call once per training step after weight update.

        Transfers weight magnitude downward through the cascade:
        surface → mid (fast) and optionally mid → deep (slow).

        Magnitude-gated: transfers only occur when the source level has
        accumulated sufficient structural magnitude (> 0.02), ensuring
        the surface layer integrates gradients autonomously before any
        content bleeds into deeper timescales.

        Args:
            include_deep: If False, only surface→mid transfer occurs
                (used during early training steps 500-5000).
        """
        # Surface → Mid: Continuous leaky integration
        transfer_sm = self.w_surface / self.tau_surface_to_mid
        self.w_mid += transfer_sm
        self.w_surface -= transfer_sm

        # Mid → Deep: Continuous leaky integration if enabled
        if include_deep:
            transfer_md = self.w_mid / self.tau_mid_to_deep
            self.w_deep += transfer_md
            self.w_mid -= transfer_md

        # Gentle w_deep norm control — prevent SR explosion.
        # Soft ceiling: w_deep can grow up to 2x its initial norm
        # (to encode learned structure) but no further. Proportional
        # scaling preserves relative weight patterns (the actual
        # "memory") while preventing magnitude blow-up.
        deep_frob = self.w_deep.norm().item()
        max_deep_frob = self._initial_deep_frob * 2.0
        if deep_frob > max_deep_frob:
            self.w_deep *= max_deep_frob / deep_frob

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
        # delta_w_total MUST be relative to the start of the consolidation epoch,
        # otherwise we are dividing 10K steps of accumulated contribution by
        # the tiny delta_w of a single step, which explodes omega near infinity.
        delta_w_total = self.effective_weights - self.si_baseline_weights
        
        # Normalize by total weight change to get per-unit importance
        self.omega += torch.relu(self.running_contribution) / (delta_w_total.pow(2) + 1e-6)

        # Decay old importance slowly to allow forgetting truly obsolete knowledge
        self.omega *= (1.0 - self.si_damping)

        # Reset accumulator and baseline
        self.running_contribution.zero_()
        self.si_baseline_weights = self.effective_weights.clone()

    def settle(self, input_vector, max_steps=20, tol=5e-3,
               input_mask=None, damping=0.15, implicit_damping=1.2):
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
        # FIX 3: Pass module ranges and sparsity parameter to JIT solver
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
            self.mod_starts,
            self.mod_ends,
            self.sparsity_alpha,
            damping,
            implicit_damping,
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
            self.topdown_indices.flip(0), td_vals,
            (self.num_nodes, self.num_nodes))

        # Top-down prediction: what higher levels predict for lower levels
        topdown_pred = torch.mv(td_sparse, rho)

        if getattr(self, 'temporal_variance_ema', None) is None:
            self.temporal_variance_ema = torch.ones(self.num_nodes, device=self.device)
        else:
            # Bound variation so EMA isn't wildly inflating temporal precision
            var_change = (rho - rho.mean()).pow(2).clamp(0, 5.0)
            self.temporal_variance_ema = 0.99 * self.temporal_variance_ema + 0.01 * var_change
            
        # FIX: Drop precision weighting. The EMA was inflating prediction variance
        # and suppressing top-down gradients to near zero.
        precision = 1.0
        
        # Spatial error = actual state - precision-weighted top-down prediction
        self.spatial_errors = self.state - precision * topdown_pred

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

    def update_weights_predictive(self, free_state, nudge_state, beta=0.5, learning_rate=0.01, hippo_edge_mask=None):
        """
        Local Hebbian weight update based on True Equilibrium Propagation.

        Top-Down Spatial weight update (EqProp):
            ΔW_ij ∝ (rho_nudge_i * rho_nudge_j - rho_free_i * rho_free_j) / beta

        Updates go to w_surface only. The metaplastic scaling and synaptic
        intelligence reduce per-synapse learning rate for consolidated and
        important synapses, preventing catastrophic forgetting.

        Temporal transition update:
            ΔA_ℓ ∝ lr * ε_temporal * x_prev^T
        """
        with torch.no_grad():
            rho_free = torch.tanh(free_state)
            rho_nudge = torch.tanh(nudge_state)
            
            # Use free phase as baseline for associative rules
            rho = rho_free

            # --- Spatial weight update (top-down + bottom-up + lateral) ---
            idx_i = self.indices[0]
            idx_j = self.indices[1]

            grad = torch.zeros_like(self.weight_values)

            # Top-down edges: EqProp gradient
            # ΔW_ij ∝ (rho_nudge_i * rho_nudge_j - rho_free_i * rho_free_j) / beta
            td_rho_nudge_i = rho_nudge[idx_i[self.topdown_edge_mask]]
            td_rho_nudge_j = rho_nudge[idx_j[self.topdown_edge_mask]]
            td_rho_free_i = rho_free[idx_i[self.topdown_edge_mask]]
            td_rho_free_j = rho_free[idx_j[self.topdown_edge_mask]]
            
            grad[self.topdown_edge_mask] = (td_rho_nudge_i * td_rho_nudge_j - td_rho_free_i * td_rho_free_j) / beta

            # Pre-synaptic state t-1 for temporal prediction
            prev_rho = torch.tanh(self.previous_state)

            # Bottom-up edges: pure associative Oja rule (no spatial error).
            # The spatial prediction error must be gated strictly through
            # topdown_edge_mask. Bottom-up edges build structural feature
            # representations via co-occurrence, not generative modeling.
            # Using spatial error here caused destructive interference
            # between generative (TD) and associative (BU) pathways.
            bu_prev_rho_i = prev_rho[idx_i[self.bottomup_edge_mask]]
            bu_rho_j = rho[idx_j[self.bottomup_edge_mask]]
            bu_w = self.effective_weights[self.bottomup_edge_mask]
            # Pure associative Oja rule: co-occurrence with self-normalization
            grad[self.bottomup_edge_mask] = 0.1 * (bu_prev_rho_i * bu_rho_j - bu_w * bu_rho_j.pow(2))

            # FIX 4: Lateral edges — anti-Hebbian inhibitory + Oja excitatory.
            #
            # The old pure Oja (0.1 * (rho_i*rho_j - w*rho_j^2)) still drove
            # all lateral weights uniformly positive because with L1 saturation,
            # rho_i*rho_j ≈ constant for all pairs. The self-normalizing -w*rho_j^2
            # term balances each weight individually but doesn't create
            # competition between nodes.
            #
            # Fix: Add an inhibitory component that depends on MODULE-LEVEL
            # mean activity. When the module mean is high (all nodes active),
            # inhibition dominates and lateral weights decrease. When the
            # module mean is moderate (sparse code), Oja excitation dominates
            # for co-active pairs. This creates genuine competition.
            lat_i = idx_i[self.lateral_edge_mask]
            lat_j = idx_j[self.lateral_edge_mask]
            lat_prev_rho_i = prev_rho[lat_i]
            lat_rho_j = rho[lat_j]
            lat_w = self.effective_weights[self.lateral_edge_mask]

            # Per-module mean activity for inhibition scaling
            module_mean_act = torch.zeros(self.num_modules, device=self.device)
            for mod_idx, (start, end) in enumerate(self.module_ranges):
                module_mean_act[mod_idx] = rho[start:end].abs().mean()

            # Retrieve mapped module levels from index mapping directly via node_to_module
            # rather than nested loops, fixing gradient leak.
            lat_src_mod = torch.zeros(lat_i.size(0), dtype=torch.long, device=self.device)
            if hasattr(self, 'node_to_module'):
                lat_src_mod = self.node_to_module[lat_i]
            else:
                for mod_idx, (start, end) in enumerate(self.module_ranges):
                    mask = (lat_i >= start) & (lat_i < end)
                    lat_src_mod[mask] = mod_idx

            mod_act = module_mean_act[lat_src_mod]

            # Inhibition strength increases with module mean activity.
            # At mod_act=0.2 (sparse): inhibition ≈ 0, Oja dominates
            # At mod_act=0.7 (dense): inhibition ≈ 0.5, suppresses co-activation
            inhibition_strength = torch.clamp(mod_act - 0.3, min=0.0)

            oja_excitatory = lat_prev_rho_i * lat_rho_j - lat_w * lat_rho_j.pow(2)
            inhibitory = -inhibition_strength * lat_prev_rho_i.abs() * lat_rho_j.abs()
            grad[self.lateral_edge_mask] = 0.1 * (oja_excitatory + inhibitory)

            # Gradient clipping
            grad = grad.clamp(-1.0, 1.0)

            # Combined importance-aware learning rate (metaplastic + SI)
            consolidation = torch.abs(self.w_deep)
            importance = self.omega
            meta_lr = learning_rate / (1.0 + self.meta_scale * consolidation + 0.1 * importance)

            # CLS: hippocampal synapses get 10x learning rate for rapid
            # acquisition of episodic patterns, restoring the intended
            # biological asymmetry of the Complementary Learning System.
            if hippo_edge_mask is not None:
                meta_lr = meta_lr * (1.0 + hippo_edge_mask.float() * 9.0)

            # Apply update only to surface level
            self.w_surface += meta_lr * grad

            # Apply RMS normalization to the learned deltas (cascade) universally
            # across all edges. Grouping by destination node ensures no neuron
            # becomes excessively overwhelmed by incoming synaptic changes.
            cascade_all = self.w_surface + self.w_mid + self.w_deep
            
            # Prevent extreme runaway of cascade weights before RMS
            cascade_all = cascade_all.clamp(-3.0, 3.0)
            self.w_surface = cascade_all - self.w_mid - self.w_deep

            dst_nodes = self.indices[1]
            unique_dst, inverse_dst = torch.unique(dst_nodes, return_inverse=True)
            n_unique = len(unique_dst)

            dst_sq_sums = torch.zeros(n_unique, device=self.device)
            dst_counts = torch.zeros(n_unique, device=self.device)
            dst_sq_sums.scatter_add_(0, inverse_dst, cascade_all.pow(2))
            dst_counts.scatter_add_(0, inverse_dst, torch.ones_like(cascade_all))
            dst_rms = torch.sqrt(dst_sq_sums / dst_counts.clamp(min=1))

            # Target RMS per destination node.
            # With avg fan-in ~150, RMS=0.15 gives total input magnitude
            # ≈ sqrt(150) * 0.15 ≈ 1.84 — well within the ±1.5 clamp range
            # when multiplied by tanh activations (which are ≤1).
            max_rms = 0.15
            scale_per_dst = torch.where(
                dst_rms > max_rms,
                max_rms / dst_rms,
                torch.ones_like(dst_rms)
            )
            weight_scale = scale_per_dst[inverse_dst]

            # Apply to transient cascade levels only; w_deep is shielded
            self.w_surface *= weight_scale
            self.w_mid *= weight_scale

            # --- Top-down weight diversity regularization ---
            # (Kept from original — prevents mode collapse)
            td_idx = torch.where(self.topdown_edge_mask)[0]
            td_src = self.topdown_indices[0]
            td_vals = self.w_surface[td_idx]

            unique_src, inverse = torch.unique(td_src, return_inverse=True)
            src_sums = torch.zeros(len(unique_src), device=self.device)
            src_counts = torch.zeros(len(unique_src), device=self.device)
            src_sums.scatter_add_(0, inverse, td_vals)
            src_counts.scatter_add_(0, inverse, torch.ones_like(td_vals))
            src_means = src_sums / src_counts.clamp(min=1)

            correction = src_means[inverse] * 0.01
            self.w_surface[td_idx] -= correction

            # --- Bias update from prediction errors ---
            # EqProp bias update: Δb_i ∝ (rho_nudge_i - rho_free_i) / beta
            # + temporal errors
            bias_grad = (rho_nudge - rho_free) / beta + self.temporal_alpha * self.temporal_errors
            self.biases += learning_rate * 0.1 * bias_grad
            self.biases.clamp_(-1.0, 1.0)

            # --- Temporal transition matrix update ---
            # ΔA_ℓ ∝ lr * ε_temporal * x_prev (diagonal approximation)
            for mod_idx, (start, end) in enumerate(self.module_ranges):
                t_error = self.temporal_errors[start:end]
                prev = self.previous_state[start:end]  # Match linear state used in prediction

                # ΔA_ℓ ∝ lr * ε_temporal * x_prev
                # FIX: Boost temporal learning rate. The old 0.1x multiplier 
                # combined with tiny t_error meant Temporal A never deviated
                # from its initialization of 0.9.
                a_mod = learning_rate * 25.0 * t_error * prev
                # Bound update delta
                self.temporal_A[mod_idx] += a_mod.clamp(-0.05, 0.05)
                # Bound A diagonal to stop exponential explosion
                self.temporal_A[mod_idx].clamp_(-1.5, 1.5)

    def store_previous_state(self):
        """Store current state as previous state for temporal prediction."""
        self.previous_state = self.state.clone()

    def enforce_spectral_radius(self, target_max=0.95):
        """
        Estimate dominant eigenvalue of effective weights using power iteration,
        and dampen w_surface if the spectral radius exceeds target_max.
        """
        with torch.no_grad():
            if getattr(self, '_power_iter_v', None) is None:
                self._power_iter_v = torch.randn(self.num_nodes, device=self.device)
                norm = torch.norm(self._power_iter_v)
                if norm > 0:
                    self._power_iter_v /= norm

            # Build sparse effective weight matrix
            W_eff = torch.sparse_coo_tensor(
                self.indices, self.effective_weights, 
                (self.num_nodes, self.num_nodes)
            )

            # Power iteration
            v = self._power_iter_v
            for _ in range(5):
                v_next = torch.mv(W_eff, v)
                norm = torch.norm(v_next)
                if norm > 1e-8:
                    v = v_next / norm

            self._power_iter_v = v
            
            # Rayleigh quotient to estimate dominant eigenvalue
            Wv = torch.mv(W_eff, v)
            eigenvalue = torch.dot(v, Wv)
            sr = torch.abs(eigenvalue).item()

            dampening_factor = 1.0
            if sr > target_max:
                dampening_factor = target_max / sr
                self.w_surface *= dampening_factor
            
            self.last_sr = sr
            return sr, dampening_factor

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
