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
) -> torch.Tensor:
    """
    Semi-implicit (IMEX) dynamics solver for predictive coding.

    Replaces RK4 with an implicit-explicit scheme:
    - Implicit: linear decay term (-x / tau)
    - Explicit: nonlinear interaction (W*tanh(x) + bias + input) / tau

    Update rule:
        x_{n+1} = (x_n + dt * nonlinear_term / tau) / (1 + dt / tau)

    This is unconditionally stable on the linear part, allowing dt=0.5-1.0
    and convergence in 10-20 steps instead of 100 RK4 steps.
    """
    num_nodes = initial_state.size(0)
    current_s = initial_state.clone()

    weights = torch.sparse_coo_tensor(indices, weight_values, (num_nodes, num_nodes))

    # Pre-compute IMEX denominator: (1 + dt / tau) per node
    imex_denom = 1.0 + dt / taus

    step_count = 0
    diff = tol + 1.0  # Ensure at least one step

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
        current_s = (current_s + dt * nonlinear / taus) / imex_denom

        # State clamping to prevent saturation
        current_s = current_s.clamp(-3.0, 3.0)

        # Hard clamp input nodes after update
        if input_mask is not None:
            current_s = current_s * (1.0 - input_mask) + input_vector * input_mask

        # Convergence check
        diff = torch.norm(current_s - old_state).item()
        step_count += 1

    return current_s


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

        # Short-term plasticity (Mongillo et al., 2008)
        self.facilitation = torch.ones_like(self.weight_values) * 0.2
        self.depression = torch.ones_like(self.weight_values)
        self.tau_facil = 150.0
        self.tau_depress = 20.0

        # Store latest prediction errors for weight updates
        self.spatial_errors = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        self.temporal_errors = torch.zeros(num_nodes, dtype=torch.float32, device=device)

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

        # Compute effective weights with short-term plasticity
        effective_weights = self.weight_values * self.facilitation * self.depression

        # Single-phase IMEX settling
        self.state = jit_solve_dynamics_imex(
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
        Compute hierarchical prediction errors for all module pairs.

        Spatial prediction error at level ℓ:
            ε_ℓ = x_ℓ − f(W_topdown * x_{ℓ+1})

        Since we use a flat weight matrix, the top-down prediction is
        already implicit in the settled state. We compute per-module errors
        by comparing each module's state against the top-down prediction
        from its parent modules.

        Returns total prediction error energy F = 0.5 * sum(||ε||²)
        """
        rho = torch.tanh(self.state)

        # Build the full weight matrix for prediction extraction
        weights = torch.sparse_coo_tensor(
            self.indices, self.weight_values * self.facilitation * self.depression,
            (self.num_nodes, self.num_nodes))

        # Compute what the network predicts for each node (W * rho)
        predicted = torch.mv(weights, rho)

        # --- Spatial prediction errors ---
        # For each hierarchical pair (upper predicts lower):
        # ε_lower = x_lower - predicted_lower (from upper's top-down weights)
        self.spatial_errors.zero_()
        spatial_energy = 0.0

        for upper_id, lower_id in self.hier_pairs:
            lower_start, lower_end = self.module_ranges[lower_id]

            # The prediction for lower module nodes comes from the
            # global synaptic input (which includes top-down from upper)
            actual = self.state[lower_start:lower_end]
            pred = predicted[lower_start:lower_end]

            # Prediction error: actual - predicted
            error = actual - pred
            self.spatial_errors[lower_start:lower_end] += error

            spatial_energy += 0.5 * torch.sum(error ** 2).item()

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
            ΔW_ij ∝ lr * ε_{lower_j} * tanh(x_upper_i)^T

        This is the predictive coding learning rule: weights change
        to reduce prediction errors at lower levels using activations
        from higher levels.

        Temporal transition update:
            ΔA_ℓ ∝ lr * ε_temporal * x_prev^T
        """
        with torch.no_grad():
            rho = torch.tanh(self.state)

            # --- Spatial weight update ---
            # For each edge (i->j), compute: lr * spatial_error[j] * rho[i]
            idx_i = self.indices[0]
            idx_j = self.indices[1]

            # Prediction error at target node * activation at source node
            error_j = self.spatial_errors[idx_j]
            rho_i = rho[idx_i]

            # Weight gradient: reduce prediction errors
            grad = error_j * rho_i

            # Gradient clipping
            grad = grad.clamp(-1.0, 1.0)

            self.weight_values += learning_rate * grad
            self.weight_values.clamp_(-1.0, 1.0)

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
        """Damps recurrent weights by a factor."""
        self.weight_values *= factor

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
