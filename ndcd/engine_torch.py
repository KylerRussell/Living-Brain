import numpy as np
import torch
from typing import Optional, Tuple, List


@torch.jit.script
def get_soma(b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    # Apical amplifies basal if aligned, ignores if weak
    # Burst coincidence detection: if both b and a are highly active and aligned
    burst = torch.relu(b) * torch.relu(a)
    # Using a bounded modulation: soma = basal * (1 + tanh(apical)) + burst
    # Ensure it remains within [-1.5, 1.5]
    return (b * (1.0 + torch.tanh(a)) + burst).clamp(-1.5, 1.5)

@torch.jit.script
def jit_solve_dynamics_imex(
    initial_basal: torch.Tensor,
    initial_apical: torch.Tensor,
    indices_basal: torch.Tensor,
    weights_basal: torch.Tensor,
    indices_apical: torch.Tensor,
    weights_apical: torch.Tensor,
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, int, float]:
    """
    Semi-implicit (IMEX) dynamics solver for Multi-Compartment Predictive Coding.

    Neurons now have segregated compartments:
    - Basal: integrates feedforward (bottom-up + lateral) signals
    - Apical: integrates feedback (top-down) signals
    - Soma (output): Non-linear combination (e.g., basal * (1 + apical))

    Returns (basal_state, apical_state, somatic_state, final_diff)
    """
    num_nodes = initial_basal.size(0)
    current_b = initial_basal.clone()
    current_a = initial_apical.clone()

    w_basal_sparse = torch.sparse_coo_tensor(indices_basal, weights_basal, (num_nodes, num_nodes))
    w_apical_sparse = torch.sparse_coo_tensor(indices_apical, weights_apical, (num_nodes, num_nodes))

    current_dt = dt
    imex_denom = 1.0 + implicit_damping * current_dt / taus
    min_dt: float = 0.05

    step_count = 0
    diff = tol + 1.0
    prev_diff: float = 1e6

    current_s = get_soma(current_b, current_a)

    while step_count < max_steps and diff > tol:
        # Hard clamp input nodes
        if input_mask is not None:
            current_b = current_b * (1.0 - input_mask) + input_vector * input_mask
            current_s = get_soma(current_b, current_a)

        old_soma = current_s.clone()

        # Layer Normalization per module before activation
        normed_s = current_s.clone()
        for m in range(module_starts.size(0)):
            ms = module_starts[m].item()
            me = module_ends[m].item()
            mod_s = current_s[ms:me]
            mean = mod_s.mean()
            var = mod_s.var(unbiased=False)
            normed_s[ms:me] = (mod_s - mean) / torch.sqrt(var + 1e-5)
            
        rho = torch.tanh(normed_s)
        
        # Basal processing (feedforward / lateral)
        synaptic_basal = torch.mv(w_basal_sparse, rho)
        # Apical processing (feedback / top-down)
        synaptic_apical = torch.mv(w_apical_sparse, rho)

        # Jacobian gain control applied separately to compartments to avoid explosion
        drho = 1.0 - rho * rho
        state_norm = rho.norm().clamp(min=1e-6)
        
        # Basal SR
        jac_basal = torch.mv(w_basal_sparse, rho * drho)
        eff_sr_basal = jac_basal.norm() / state_norm
        target_max_sr: float = 0.95
        if eff_sr_basal > target_max_sr:
            synaptic_basal = synaptic_basal * (target_max_sr / eff_sr_basal)
            
        # Apical SR
        jac_apical = torch.mv(w_apical_sparse, rho * drho)
        eff_sr_apical = jac_apical.norm() / state_norm
        if eff_sr_apical > target_max_sr:
            synaptic_apical = synaptic_apical * (target_max_sr / eff_sr_apical)

        nonlinear_basal = synaptic_basal + biases + input_vector
        nonlinear_apical = synaptic_apical  # pure prediction context

        # Semi-implicit compartment updates
        current_b = (current_b + current_dt * nonlinear_basal / taus) / imex_denom
        # Apical dendrites often have slower time constants (calcium spikes vs sodium)
        current_a = (current_a + current_dt * nonlinear_apical / (taus * 1.5)) / imex_denom

        # L1 sparsity on basal (the driving feature)
        current_b -= 0.0001 * current_b.sign() * current_dt

        # Clamp compartments
        current_b = current_b.clamp(-1.5, 1.5)
        current_a = current_a.clamp(-1.5, 1.5)
        
        # Compute somatic state
        current_s = get_soma(current_b, current_a)

        # Hard clamp input nodes after update
        if input_mask is not None:
            current_b = current_b * (1.0 - input_mask) + input_vector * input_mask
            current_a = current_a * (1.0 - input_mask) # No top down expectation forces input directly
            current_s = get_soma(current_b, current_a)

        # k-Winner-Take-All (k-WTA) lateral inhibition on the SOMATIC output
        # Lateral inhibition sharpens the actual firing rate output
        if sparsity_alpha > 0.0 and step_count % 5 == 4:
            MexicanHatSoma = current_s.clone()
            for m in range(module_starts.size(0)):
                ms = module_starts[m].item()
                me = module_ends[m].item()
                mod_s = MexicanHatSoma[ms:me]
                mod_size = me - ms
                k = max(1, int(mod_size * 0.05))  # 5% global inhibition sparsity
                if mod_size > k:
                    topk_vals = torch.topk(mod_s.abs(), k).values
                    threshold = topk_vals[-1]
                    below = mod_s.abs() < threshold
                    MexicanHatSoma[ms:me] = torch.where(below, mod_s * 0.05, mod_s)
                    
            # Propagate the somatic suppression back into the basal drive
            suppressing_ratio = (MexicanHatSoma.abs() + 1e-5) / (current_s.abs() + 1e-5)
            current_b = current_b * suppressing_ratio
            current_b = current_b.clamp(-1.5, 1.5)
            current_s = get_soma(current_b, current_a)

        diff = torch.norm(current_s - old_soma).item()

        if diff <= tol * 1.2:
            break

        if diff > prev_diff and current_dt > min_dt:
            current_dt = max(current_dt * 0.5, min_dt)
            imex_denom = 1.0 + implicit_damping * current_dt / taus
        elif diff < prev_diff * 0.8 and current_dt < dt:
            current_dt = min(current_dt * 1.2, dt)
            imex_denom = 1.0 + implicit_damping * current_dt / taus

        prev_diff = diff
        step_count += 1

    return current_b, current_a, current_s, diff, step_count, current_dt


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
                 modules=None,
                 positions: Optional[np.ndarray] = None,
                 dt=0.5, device='cuda' if torch.cuda.is_available() else 'cpu',
                 temporal_alpha=0.5,
                 is_inhibitory: Optional[np.ndarray] = None):
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
        self.state_basal = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        self.state_apical = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        self.state = torch.zeros(num_nodes, dtype=torch.float32, device=device)  # somatic state

        # Context EMA buffer for cross-boundary context preservation
        self.context_ema = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        self.context_ema_alpha = 0.1  # Blend rate: 10% new, 90% old

        # Module metadata
        self.module_ranges = module_ranges  # [(start, end), ...]
        self.module_levels = module_levels  # np array of levels
        self.hier_pairs = hier_pairs        # [(upper_id, lower_id), ...]
        self.num_modules = len(module_ranges)

        # Laminar Segregation Masks
        self.is_l4 = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        self.is_l23 = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        self.is_l56 = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        if modules is not None:
            for mod in modules:
                self.is_l4[mod['l4_indices']] = True
                self.is_l23[mod['l23_indices']] = True
                self.is_l56[mod['l56_indices']] = True

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

        # Fix 4 removed: Uncoupled Product Feedback Alignment (PFA)
        # Weights are left completely asymmetric.


        # --- Free-edge mask for spectral radius enforcement ---
        # I/O projection edges (source or dest < 512) are external forcing,
        # not autonomous recurrence. They are 6.5x boosted and constitute
        # ~39% of edges. Including them in spectral radius estimation
        # inflates the measured SR from ~0.57 (free edges) to ~29 (full
        # matrix), causing enforce_spectral_radius to multiply all learned
        # weights by 0.95/29 ≈ 0.033 every 100 steps — zeroing them out.
        # This mask matches graph.py's initialization, which tunes SR on
        # free edges only.
        self.free_edge_mask = (self.indices[0] >= 512) & (self.indices[1] >= 512)
        self.free_edge_indices = self.indices[:, self.free_edge_mask]

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
        self.tau_surface_to_mid = 2000.0   # was 1000.0; slower drain lets surface accumulate
        
        # We now track separate mid-to-deep transfer rates per edge depending on 
        # whether the source node belongs to a hippocampal or neocortical module.
        self.tau_mid_to_deep = torch.ones_like(self.weight_values) * 1000.0

        # Metaplastic scaling: how much accumulated deep weight
        # reduces surface learning rate. Reduced from 1.0 to 0.1 because
        # w_deep ≈ 0.83 after chars was causing 1.83x LR reduction (with
        # omega adding another ~10x). At 0.1, the cascade inertia of
        # w_deep already protects important weights without also killing
        # the effective learning rate.
        self.meta_scale = 0.1  # was 0.25; reduce inertia so surface LR isn't crushed by w_deep

        # FIX 1: Track the initial w_deep Frobenius norm as a target.
        # This is the SR-tuned initialization; effective weights should
        # never exceed ~2x this norm during training.
        # CHANGED: Compute on free edges only, matching graph.py's tuning.
        self._initial_deep_frob = self.weight_values[self.free_edge_mask].norm().item()

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
        self.sparsity_alpha = 0.8
        
        # Thalamic Gate tracking
        self.surprise_ema = 0.0

        # Dale's law assignment
        if is_inhibitory is not None:
            self.is_inhibitory = torch.tensor(is_inhibitory, dtype=torch.bool, device=device)
        else:
            self.is_inhibitory = torch.zeros(num_nodes, dtype=torch.bool, device=device)

        # Homeostatic Synaptic Scaling (HSS)
        self.calcium_traces = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        self.calcium_target = 0.1
        self.tau_calcium = 1000.0
        self.hss_rho = 0.001

        # Three-Factor Learning (Eligibility Traces)
        self.eligibility_traces = torch.zeros_like(self.weight_values)
        self.tau_eligibility = 2.0  # Short-term memory of local coincidence

    def _build_module_tensors(self):
        """Pre-build tensors for module start/end ranges for fast slicing."""
        self.mod_starts = torch.tensor(
            [r[0] for r in self.module_ranges], dtype=torch.long, device=self.device)
        self.mod_ends = torch.tensor(
            [r[1] for r in self.module_ranges], dtype=torch.long, device=self.device)
        self.mod_level_tensor = torch.tensor(
            self.module_levels, dtype=torch.long, device=self.device)

    def activation_function(self, s):
        return torch.tanh(s) + 0.01 * s
        
    @property
    def effective_weights(self):
        """Effective weight is base topology + learned cascade deltas."""
        return self.weight_values + self.w_surface + self.w_mid + self.w_deep

    def cascade_transfer(self, include_deep=True, surface_floor=0.01):
        """Call periodically (every ~500 steps) after weight update.

        Transfers weight magnitude downward through the cascade:
        surface → mid (above floor) and optionally mid → deep (gated).

        The surface floor ensures that fast transient weights always retain
        a minimum magnitude for hierarchical signal propagation. Without
        this, continuous proportional drain drives TD/BU surface weights
        to zero, making higher levels input-invariant.

        Args:
            include_deep: If False, only surface→mid transfer occurs.
            surface_floor: Minimum surface magnitude to retain (default 0.003).
        """
        # Surface → Mid: Transfer only excess above floor
        surface_abs = self.w_surface.abs()
        excess_mask = surface_abs > surface_floor
        if excess_mask.any():
            excess = (surface_abs - surface_floor) * self.w_surface.sign()
            # Transfer 20% of excess per call (called every ~500 steps)
            transfer_sm = torch.zeros_like(self.w_surface)
            transfer_sm[excess_mask] = excess[excess_mask] * 0.2
            self.w_mid += transfer_sm
            self.w_surface -= transfer_sm

        # Mid → Deep: Continuous leaky integration if enabled
        if include_deep:
            gate_mask = self.w_mid.abs() > 0.005  # lowered from 0.02
            transfer_md = torch.zeros_like(self.w_mid)
            transfer_md[gate_mask] = self.w_mid[gate_mask] / self.tau_mid_to_deep[gate_mask]
            self.w_deep += transfer_md
            self.w_mid -= transfer_md

        # Gentle w_deep norm control (unchanged)
        deep_frob = self.w_deep[self.free_edge_mask].norm().item()
        max_deep_frob = self._initial_deep_frob
        if deep_frob > max_deep_frob:
            self.w_deep[self.free_edge_mask] *= max_deep_frob / deep_frob

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

    def remodel_structure(self, prune_ratio=0.01):
        """
        Selective Structural Plasticity.
        Prunes the `prune_ratio` lowest-utility synapses (based on omega) 
        and reinitializes them. This mimics biological synaptogenesis.
        Only applies to free edges (not I/O).
        """
        if prune_ratio <= 0.0:
            return 0
            
        with torch.no_grad():
            free_omega = self.omega[self.free_edge_mask]
            n_prune = int(free_omega.size(0) * prune_ratio)
            
            if n_prune == 0:
                return 0
                
            threshold = torch.kthvalue(free_omega, n_prune).values.item()
            prune_mask_free = free_omega <= threshold
            
            prune_mask = torch.zeros_like(self.omega, dtype=torch.bool)
            prune_mask[self.free_edge_mask] = prune_mask_free
            
            # Clear cascade and tracking stats for pruned edges
            self.w_surface[prune_mask] = 0.0
            self.w_mid[prune_mask] = 0.0
            self.w_deep[prune_mask] = 0.0
            self.eligibility_traces[prune_mask] = 0.0
            self.omega[prune_mask] = 0.0
            self.running_contribution[prune_mask] = 0.0
            
            n_reset = prune_mask.sum().item()
            src_inh = self.is_inhibitory[self.indices[0, prune_mask]]
            
            avg_magnitude = self.weight_values[self.free_edge_mask].abs().mean().item()
            new_weights = torch.rand(n_reset, device=self.device) * (avg_magnitude * 2.0)
            new_weights = torch.where(src_inh, -new_weights, new_weights)
            
            self.weight_values[prune_mask] = new_weights
            
        
            
            return n_reset

    def settle(self, input_vector, max_steps=20, tol=5e-3,
               input_mask=None, damping=0.15, implicit_damping=2.0):
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
        
        # Split into Basal and Apical topologies
        basal_mask = self.bottomup_edge_mask | self.lateral_edge_mask
        apical_mask = self.topdown_edge_mask
        
        indices_basal = self.indices[:, basal_mask]
        weights_basal = effective_weights[basal_mask]
        
        indices_apical = self.indices[:, apical_mask]
        weights_apical = effective_weights[apical_mask]

        # Single-phase IMEX settling
        # FIX 3: Pass module ranges and sparsity parameter to JIT solver
        self.state_basal, self.state_apical, self.state, self.last_settle_diff, self.last_settle_steps, self.last_settle_dt = jit_solve_dynamics_imex(
            self.state_basal,
            self.state_apical,
            indices_basal,
            weights_basal,
            indices_apical,
            weights_apical,
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
            
        # FIX: Rebalancing Generative Predictive Coding & Laminar Segregation
        # PRC (Inhibitory) units regulate "volume" of error signals via divisive normalization
        # Average inhibitory activity is used as a proxy for confidence (precision)
        prc_activity = torch.relu(self.state[self.is_inhibitory]).mean()
        precision = 5.0 * (1.0 + prc_activity)
        
        # Spatial error = actual state - precision-weighted prediction
        raw_error = self.state - precision * topdown_pred
        
        # Divisive normalization of the error signal by PRC units
        divisive_factor = 1.0 + torch.relu(self.state[self.is_inhibitory]).mean()
        normalized_error = raw_error / divisive_factor
        
        # ERR units (L2/3) predominantly ascend errors. Suppress error on EXP units (L5/6).
        self.spatial_errors = normalized_error.clamp(-1.0, 1.0)
        self.spatial_errors[self.is_l56] *= 0.1  # EXP units don't broadcast upward prediction errors
        self.spatial_errors[self.is_l4] *= 0.5   # L4 are input recipients, moderate error

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

        # Total free energy with Metabolic Cost (minimizing surprisal and restricting activity bounds)
        metabolic_cost = 0.001 * torch.sum(self.state.abs()).item()
        total_energy = spatial_energy + self.temporal_alpha * temporal_energy + metabolic_cost

        return total_energy

    def update_weights_predictive(self, free_state, nudge_state, learning_rate=0.01, 
                                  hippo_edge_mask=None, active_level_max=None,
                                  dopamine: float = 1.0, acetylcholine: float = 1.0):
        """
        Local Hebbian weight update based on True Equilibrium Propagation.

        Top-Down Spatial weight update (Prospective Configuration):
            ΔW_ij ∝ (rho_nudge_i - rho_free_i) * rho_nudge_j

        Neuromodulatory Gating:
        - Dopamine (DA): RPE surrogate. Scales plasticity based on surprise/success.
        - Acetylcholine (ACh): Uncertainty/Attention surrogate. High ACh = rapid memory encoding.
        """
        with torch.no_grad():
            rho_free = torch.tanh(free_state)
            rho_nudge = torch.tanh(nudge_state)
            
            # --- Intrinsic Plasticity (IP) Update ---
            # Track the temporal mean of each node's absolute activation
            if getattr(self, 'activation_ema', None) is None:
                self.activation_ema = torch.zeros(self.num_nodes, device=self.device)
            self.activation_ema = 0.99 * self.activation_ema + 0.01 * rho_free.abs()
            
            # Adjust bias to explicitly pull temporal mean toward target sparsity
            ip_gradient = self.sparsity_alpha - self.activation_ema
            
            # Use free phase as baseline for associative rules
            rho = rho_free
            
            # --- Update Calcium Traces for HSS ---
            self.calcium_traces += (rho.abs() - self.calcium_traces) / self.tau_calcium
            
            # Calculate somatic bursts from current basal/apical compartments
            bursts = torch.relu(self.state_basal) * torch.relu(self.state_apical)

            # --- Spatial weight update (top-down + bottom-up + lateral) ---
            idx_i = self.indices[0]
            idx_j = self.indices[1]

            grad = torch.zeros_like(self.weight_values)

            # Top-down edges: Prospective Configuration update
            # ΔW_ij ∝ (rho_nudge_i - rho_free_i) * rho_nudge_j
            td_rho_nudge_i = rho_nudge[idx_i[self.topdown_edge_mask]]
            td_rho_nudge_j = rho_nudge[idx_j[self.topdown_edge_mask]]
            td_rho_free_i = rho_free[idx_i[self.topdown_edge_mask]]
            
            grad[self.topdown_edge_mask] = 100.0 * (td_rho_nudge_i - td_rho_free_i) * td_rho_nudge_j

            # Pre-synaptic state t-1 for temporal prediction
            prev_rho = torch.tanh(self.previous_state)

            # Bottom-up edges: pure associative Oja rule (no spatial error).
            bu_prev_rho_i = prev_rho[idx_i[self.bottomup_edge_mask]]
            bu_rho_j = rho[idx_j[self.bottomup_edge_mask]]
            bu_w_plastic = (self.w_surface[self.bottomup_edge_mask] + 
                            self.w_mid[self.bottomup_edge_mask] + 
                            self.w_deep[self.bottomup_edge_mask])
            bu_burst_j = bursts[idx_j[self.bottomup_edge_mask]]
            # Pure associative Oja rule: co-occurrence with self-normalization, gated by apical bursts
            grad[self.bottomup_edge_mask] = 0.1 * (1.0 + bu_burst_j) * (bu_prev_rho_i * bu_rho_j - bu_w_plastic * bu_rho_j.pow(2))

        

            # FIX 4: Lateral edges — anti-Hebbian inhibitory + Oja excitatory.
            lat_i = idx_i[self.lateral_edge_mask]
            lat_j = idx_j[self.lateral_edge_mask]
            lat_prev_rho_i = prev_rho[lat_i]
            lat_rho_j = rho[lat_j]
            lat_w = self.effective_weights[self.lateral_edge_mask]

            # Per-module mean activity for inhibition scaling
            module_mean_act = torch.zeros(self.num_modules, device=self.device)
            for mod_idx, (start, end) in enumerate(self.module_ranges):
                module_mean_act[mod_idx] = rho[start:end].abs().mean()

            lat_src_mod = torch.zeros(lat_i.size(0), dtype=torch.long, device=self.device)
            if hasattr(self, 'node_to_module'):
                lat_src_mod = self.node_to_module[lat_i]
            else:
                for mod_idx, (start, end) in enumerate(self.module_ranges):
                    mask = (lat_i >= start) & (lat_i < end)
                    lat_src_mod[mask] = mod_idx

            mod_act = module_mean_act[lat_src_mod]

            inhibition_strength = torch.clamp(mod_act - 0.3, min=0.0)

            oja_excitatory = lat_prev_rho_i * lat_rho_j - lat_w * lat_rho_j.pow(2)
            inhibitory = -inhibition_strength * lat_prev_rho_i.abs() * lat_rho_j.abs()
            grad[self.lateral_edge_mask] = 0.1 * (oja_excitatory + inhibitory)

            # --- Orthogonalization penalty on lateral weights ---
            with torch.enable_grad():
                lat_w_var = lat_w.detach().clone().requires_grad_(True)
                total_penalty = 0.0
                
                for mod_idx, (start, end) in enumerate(self.module_ranges):
                    mod_size = end - start
                    if mod_size <= 1: continue
                    
                    mask = (lat_i >= start) & (lat_i < end) & (lat_j >= start) & (lat_j < end)
                    if not mask.any(): continue
                    
                    mod_i_idx = lat_i[mask] - start
                    mod_j_idx = lat_j[mask] - start
                    mod_w_var = lat_w_var[mask]
                    
                    W_dense = torch.sparse_coo_tensor(
                        torch.stack([mod_i_idx, mod_j_idx]), mod_w_var, (mod_size, mod_size)
                    ).to_dense()
                    
                    W_norm = torch.nn.functional.normalize(W_dense, p=2, dim=1, eps=1e-6)
                    sim_matrix = torch.mm(W_norm, W_norm.t())
                    
                    I = torch.eye(mod_size, device=self.device)
                    penalty = torch.sum((sim_matrix - I) ** 2)
                    total_penalty = total_penalty + penalty
                    
                if isinstance(total_penalty, torch.Tensor) and total_penalty.requires_grad:
                    ortho_grad = torch.autograd.grad(total_penalty, lat_w_var)[0]
                    grad[self.lateral_edge_mask] -= 0.05 * ortho_grad

            # Gradient clipping: Explicit max_norm=1.0 limit
            grad_norm = grad.norm()
            if grad_norm > 1.0:
                grad = grad * (1.0 / grad_norm)
            
            # --- Metabolic Cost (Weight Penalty) ---
            grad -= 0.0001 * self.effective_weights.sign()

            grad = grad.clamp(-1.0, 1.0)

            # --- Dale's ANNs (DANNs) Fisher Information Scaling ---
            if not hasattr(self, 'fisher_info'):
                self.fisher_info = torch.ones(self.num_nodes, device=self.device)
            var_inst = (rho_free - getattr(self, 'activation_ema', torch.zeros_like(rho_free))).pow(2)
            self.fisher_info = 0.99 * self.fisher_info + 0.01 * var_inst
            
            src_inh = self.is_inhibitory[idx_i]
            fisher_scale = 1.0 / (self.fisher_info[idx_i] + 1e-4)
            fisher_scale = torch.clamp(fisher_scale, 0.1, 5.0)
            grad[src_inh] *= fisher_scale[src_inh] * 0.2  # Dampen and normalize inhibitory updates

            # --- Local Homeostatic Scaling (Variance Control) ---
            target_var = 0.1
            tau_update = 0.05 * (var_inst - target_var)
            self.taus = torch.clamp(self.taus + tau_update, 0.5, 100.0)

            # --- Calculate Thalamic Saliency Gate ---
            # Use total network energy as a proxy for 'surprise'.
            # If current_energy is much higher than the EMA, it's a novel/important signal -> scale up learning.
            # If it's matching or lower, it's predictable noise (like spaces) -> scale down learning.
            current_energy = 0.5 * torch.sum(self.spatial_errors ** 2).item() + self.temporal_alpha * torch.sum(self.temporal_errors ** 2).item()
            if self.surprise_ema == 0.0:
                self.surprise_ema = current_energy
            else:
                self.surprise_ema = 0.99 * self.surprise_ema + 0.01 * current_energy
            
            saliency_gate = torch.clamp(torch.tensor(current_energy / max(self.surprise_ema, 1e-6)), min=0.1, max=3.0).item()

            # Combined importance-aware learning rate (metaplastic + SI)
            consolidation = torch.abs(self.w_deep)
            importance = self.omega
            
            # Base learning rate explicitly gated by Neuromodulators
            # Dopamine scales overall magnitude
            # Acetylcholine scales rate of new structural learning (surface weights)
            global_neuromodulation = max(0.01, dopamine * acetylcholine)
            
            meta_lr = (learning_rate * global_neuromodulation) / (1.0 + self.meta_scale * consolidation + 0.1 * importance)
            
            # Apply Thalamic Gating
            meta_lr = meta_lr * saliency_gate

            # CLS: hippocampal synapses get highly boosted learning under high ACh
            if hippo_edge_mask is not None:
                meta_lr = meta_lr * (1.0 + hippo_edge_mask.float() * (9.0 * acetylcholine))

            # Level gate (Strategy 2): restrict updates to edges whose max endpoint
            # level is <= active_level_max.  L0-only per-character, L1+L2 at word
            # boundaries, L3 at sentence boundaries.
            if active_level_max is not None:
                src_lvls = self.node_to_level[self.indices[0]]
                dst_lvls = self.node_to_level[self.indices[1]]
                edge_max_lvl = torch.maximum(src_lvls, dst_lvls)
                level_gate = (edge_max_lvl <= active_level_max).float()
                grad = grad * level_gate

            # --- Three-Factor Learning ---
            # 1. Accumulate local Hebbian correlation into the eligibility trace
            self.eligibility_traces = self.eligibility_traces * (1.0 - 1.0/self.tau_eligibility) + grad * (1.0/self.tau_eligibility)

            # 2. Global Neuromodulatory Gating (M(t))
            # meta_lr contains global_neuromodulation (DA * ACh) and Thalamic Saliency
            self.w_surface += meta_lr * self.eligibility_traces

            # --- Homeostatic Synaptic Scaling (HSS) ---
            # dw = -rho_hss * w * (C - epsilon)
            # Reverses sign for inhibitory synapses to increase inhibition when overly active
            calcium_dev = self.calcium_traces[self.indices[1]] - self.calcium_target
            hss_mod = -self.hss_rho * calcium_dev
            src_inh = self.is_inhibitory[self.indices[0]]
            hss_mod[src_inh] *= -1.0  
            w_plastic = self.w_surface + self.w_mid + self.w_deep
            self.w_surface += hss_mod * w_plastic

            # Apply RMS normalization to the learned deltas (cascade) universally
            # across all edges. Grouping by destination node ensures no neuron
            # becomes excessively overwhelmed by incoming synaptic changes.
            cascade_all = self.w_surface + self.w_mid + self.w_deep

            # Phase 1: Active Spectral Regularization (free edges only)
            # Must match enforce_spectral_radius by excluding I/O projection
            # edges from the norm computation. I/O weights are fixed external
            # forcing — including them makes the ceiling effectively ~0.3x
            # the actual free-edge norm, crushing all learned structure.
            free_cascade = cascade_all[self.free_edge_mask]
            current_frob = free_cascade.norm().item()
            target_max = 2.0 * self._initial_deep_frob
            if current_frob > target_max:
                scale_penalty = target_max / current_frob
                # Only scale the free edges of w_surface, preserve I/O projections
                self.w_surface[self.free_edge_mask] *= scale_penalty
                self.w_mid[self.free_edge_mask] *= scale_penalty
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

            # --- Dale's Law: Enforce E/I constraints ---
            # Excitatory neurons can only have positive outgoing weights.
            # Inhibitory neurons can only have negative outgoing weights.
            src_inh = self.is_inhibitory[self.indices[0]]
            cascade_all = self.w_surface + self.w_mid + self.w_deep + self.weight_values
            
            # Constraint: total effective_weight >= 0 for Exc, <= 0 for Inh
            cascade_all_clamped = torch.where(src_inh, cascade_all.clamp(max=0.0), cascade_all.clamp(min=0.0))
            
            # Reconstruct w_surface from the clamped effective weight
            self.w_surface = cascade_all_clamped - self.w_mid - self.w_deep - self.weight_values

            # --- Bias update from prediction errors + IP ---
            bias_grad = (rho_nudge - rho_free) + self.temporal_alpha * self.temporal_errors + ip_gradient
            if active_level_max is not None:
                node_gate = (self.node_to_level <= active_level_max).float()
                bias_grad = bias_grad * node_gate
            self.biases += learning_rate * 0.1 * bias_grad
            self.biases.clamp_(-1.0, 1.0)

            # --- Temporal transition matrix update ---
            eta = learning_rate * 25.0  # Dedicated temporal learning rate (η)
            for mod_idx, (start, end) in enumerate(self.module_ranges):
                # Level gate: skip modules above the active update threshold
                if active_level_max is not None and self.module_levels[mod_idx] > active_level_max:
                    continue
                t_error = self.temporal_errors[start:end]
                prev = self.previous_state[start:end]

                a_mod = eta * t_error * prev

                a_mod = a_mod.clamp(-0.05, 0.05) - 0.005 * self.temporal_A[mod_idx]
                self.temporal_A[mod_idx] += a_mod

                self.temporal_A[mod_idx].clamp_(-1.0, 1.0)

    def store_previous_state(self):
        """Store current state as previous state for temporal prediction."""
        self.previous_state = self.state.clone()

    def enforce_spectral_radius(self, target_max=0.95):
        """
        Estimate dominant eigenvalue of the CASCADE-ONLY weights (learned
        deltas: w_surface + w_mid + w_deep) on FREE edges, and dampen
        w_surface if the cascade spectral radius exceeds target_max.

        Decoupled from base weights: The base weight_values were already
        SR-tuned at initialization (graph.py). Enforcing on the total
        effective weight (base + cascade) was crushing learned structure
        because the base SR (~0.90) consumed most of the budget, leaving
        almost no room for the cascade to add meaningful structure.

        By enforcing on cascade-only, the learned weights are constrained
        independently, and the base initialization is preserved.

        Only dampens w_surface — w_mid and w_deep are protected long-term
        memory that must never be rescaled by a transient SR measurement.
        """
        with torch.no_grad():
            # Power iteration vector (persistent across calls for convergence)
            if getattr(self, '_power_iter_v', None) is None:
                self._power_iter_v = torch.randn(self.num_nodes, device=self.device)
                norm = torch.norm(self._power_iter_v)
                if norm > 0:
                    self._power_iter_v /= norm

            # Build sparse matrix from FREE edges of CASCADE ONLY (not base weights)
            cascade_weights = (self.w_surface + self.w_mid + self.w_deep)[self.free_edge_mask]

            # Skip if cascade is negligible
            cascade_norm = cascade_weights.norm().item()
            if cascade_norm < 1e-6:
                self.last_sr = 0.0
                return

            W_cascade = torch.sparse_coo_tensor(
                self.free_edge_indices, cascade_weights,
                (self.num_nodes, self.num_nodes)
            )

            # Power iteration (5 steps)
            v = self._power_iter_v
            for _ in range(5):
                v_next = torch.mv(W_cascade, v)
                norm = torch.norm(v_next)
                if norm > 1e-8:
                    v = v_next / norm

            self._power_iter_v = v

            # Rayleigh quotient estimate
            Wv = torch.mv(W_cascade, v)
            eigenvalue = torch.dot(v, Wv)
            sr = torch.abs(eigenvalue).item()
            self.last_sr = sr

            # Only dampen w_surface (fast transient weights)
            if sr > target_max:
                dampening_factor = target_max / sr
                self.w_surface[self.free_edge_mask] *= dampening_factor

    def enforce_jacobian_spectral_radius(self, target_max=0.95):
        """
        Enforce spectral radius on the actual Jacobian J = (1/τ)(-I + W*diag(sech²(x))),
        not just weight norms. This bounds the true dynamical instability.

        Uses power iteration on the full Jacobian (with I/O rows/cols zeroed)
        and dampens w_surface on free edges when SR exceeds target.
        """
        with torch.no_grad():
            J = self.get_jacobian()  # Already zeros I/O rows/cols

            # Power iteration on J (5 steps, persistent vector)
            if getattr(self, '_jac_power_v', None) is None:
                self._jac_power_v = torch.randn(self.num_nodes, device=self.device)
                norm = self._jac_power_v.norm()
                if norm > 0:
                    self._jac_power_v /= norm

            v = self._jac_power_v
            for _ in range(5):
                v_next = J @ v
                norm = v_next.norm()
                if norm > 1e-8:
                    v = v_next / norm

            self._jac_power_v = v

            # Rayleigh quotient estimate of spectral radius
            Jv = J @ v
            sr = torch.abs(torch.dot(v, Jv)).item()
            self.last_jacobian_sr = sr

            # Only dampen w_surface (fast transient weights) on free edges
            if sr > target_max:
                dampening = target_max / sr
                self.w_surface[self.free_edge_mask] *= dampening

    def get_jacobian(self) -> torch.Tensor:
        """
        Computes the Jacobian of the temporal transition dynamics at the current state.
        This enables gradient alignment tracking (Diagnostic Test 2).
        
        Returns:
            [N, N] Dense Jacobian matrix tensor.
        """
        with torch.no_grad():
            N = self.num_nodes
            
            rho = torch.tanh(self.state)
            drho = 1.0 - rho.pow(2)
            
            W_eff = torch.sparse_coo_tensor(
                self.indices, self.effective_weights, 
                (N, N)
            ).to_dense()
            
            W_drho = W_eff * drho.unsqueeze(0)
            
            taus_inv = 1.0 / self.taus.unsqueeze(1)
            I = torch.eye(N, dtype=torch.float32, device=self.device)
            
            J = taus_inv * (-I + W_drho)

            # Zero out I/O node rows/columns (nodes 0-511).
            # I/O nodes have tau=0.1 which amplifies their Jacobian rows
            # by 10x, inflating the measured spectral radius. Since I/O
            # nodes are clamped during settling, their Jacobian
            # contribution is meaningless but dominates the eigenvalue.
            J[:512, :] = 0
            J[:, :512] = 0

            return J

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

    def update_context_ema(self):
        """Update running EMA of state for cross-boundary context preservation."""
        self.context_ema = (1 - self.context_ema_alpha) * self.context_ema + self.context_ema_alpha * self.state

    def blend_context(self, blend_factor=0.2):
        """Blend stored context EMA back into state after boundary reset."""
        self.state = (1 - blend_factor) * self.state + blend_factor * self.context_ema

    def damp_weights(self, factor=0.9):
        """Damps recurrent weights by a factor (applied to deep level)."""
        self.w_deep *= factor

    def cascade_transfer(self, include_deep=True):
        """
        Transfers learned weights down the memory cascade.
        Surface -> Mid -> Deep.
        """
        with torch.no_grad():
            # 1. Surface to Mid
            transfer_s2m = self.w_surface / self.tau_surface_to_mid
            self.w_mid += transfer_s2m
            self.w_surface -= transfer_s2m

            # 2. Mid to Deep
            if include_deep:
                transfer_m2d = self.w_mid / self.tau_mid_to_deep
                
                # FIX 3: Magnitude-gated hippocampal transfer
                # Neocortical synapses (τ_deep=2000) drip-feed continuously.
                # Hippocampal synapses (τ_deep=50) wait until a coherent pattern
                # forms in w_mid (magnitude > 0.1) before flushing rapidly.
                # This prevents noisy transient patterns from cluttering deep CLS memory.
                is_hippo = self.tau_mid_to_deep < 100.0
                has_magnitude = self.w_mid.abs() > 0.1
                
                # Zero out transfer for hippo edges lacking sufficient structural magnitude
                transfer_m2d[is_hippo & ~has_magnitude] = 0.0

                self.w_deep += transfer_m2d
                self.w_mid -= transfer_m2d


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
        """
        with torch.no_grad():
            td_weights = self.effective_weights[self.topdown_edge_mask]
            src_levels = self.node_to_level[self.topdown_indices[0]]
            dst_levels = self.node_to_level[self.topdown_indices[1]]

            stats = {}
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