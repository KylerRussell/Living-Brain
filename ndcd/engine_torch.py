import numpy as np
import torch
from typing import Optional, Tuple

@torch.jit.script
def jit_solve_dynamics(
    initial_state: torch.Tensor,
    initial_traces: torch.Tensor,
    indices: torch.Tensor,
    weight_values: torch.Tensor,
    biases: torch.Tensor,
    taus: torch.Tensor,
    input_vector: torch.Tensor,
    dt: float,
    max_steps: int,
    tol: float,
    nudge_target: Optional[torch.Tensor],
    nudge_mask: Optional[torch.Tensor],
    beta: float,
    input_mask: Optional[torch.Tensor],
    inhibition_mask: Optional[torch.Tensor],
    inhibition_beta: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    JIT-compiled static function for the physics loop.
    Solves dx/dt = (-x + W*tanh(x) + b + I) / tau
    AND dz/dt = (-z + rho(xi)rho(xj)) / tau_z
    """
    num_nodes = initial_state.size(0)
    current_s = initial_state
    
    # Eligibility Trace State (Flattened values matching 'weight_values')
    current_z = initial_traces
    tau_z = 5.0 # Slow decay for traces
    
    # Construct sparse tensor view for matmul on the fly
    weights = torch.sparse_coo_tensor(indices, weight_values, (num_nodes, num_nodes))

    step_count = 0
    diff = 1.0
    
    # Pre-calculate inhibition constant part if needed
    # But inhibition depends on current_s state, so it's dynamic.
    
    while step_count < max_steps and diff > tol:
        old_state = current_s
        
        # 1. Nudge Logic / Input
        current_input = input_vector
        if nudge_target is not None and beta > 0.0:
            rho_s = torch.tanh(current_s)
            diff_nudge = (nudge_target - rho_s)
            if nudge_mask is not None:
                diff_nudge = diff_nudge * nudge_mask
            nudge_force = beta * diff_nudge
            current_input = current_input + nudge_force
            
            
        # 1.5 Lateral Inhibition (Softmax-like competition)
        # I_inhib = -beta_inhib * (Sum(rho * mask) - (rho * mask))
        # Effectively: Everyone inhibited by the Total Activity of the group, except themselves.
        if inhibition_mask is not None and inhibition_beta > 0.0:
            rho_s = torch.tanh(current_s)
            # Masked activity
            masked_activity = rho_s * inhibition_mask
            total_activity = torch.sum(masked_activity)
            # Inhibition signal: Total - Self
            # We want to inhibit 's' by this amount.
            # I_inhib_vec = -inhibition_beta * (total_activity - masked_activity)
            # But we only apply this inhibition TO the masked nodes.
            
            inhibition_signal = (total_activity - masked_activity) * inhibition_mask
            current_input = current_input - (inhibition_beta * inhibition_signal)

        # 2. RK4 Step
        # Unrolled for JIT compatibility
        
        # k1
        rho_s = torch.tanh(current_s)
        if current_s.dim() == 1:
            synaptic_input = torch.mv(weights, rho_s)
        else:
            synaptic_input = torch.matmul(weights, rho_s.t()).t()
            
        total_input = synaptic_input + biases + current_input
        k1 = (-current_s + total_input) / taus
        
        # k2
        s2 = current_s + 0.5 * dt * k1
        rho_s = torch.tanh(s2)
        if s2.dim() == 1:
            synaptic_input = torch.mv(weights, rho_s)
        else:
            synaptic_input = torch.matmul(weights, rho_s.t()).t()
        total_input = synaptic_input + biases + current_input
        k2 = (-s2 + total_input) / taus
        
        # k3
        s3 = current_s + 0.5 * dt * k2
        rho_s = torch.tanh(s3)
        if s3.dim() == 1:
            synaptic_input = torch.mv(weights, rho_s)
        else:
            synaptic_input = torch.matmul(weights, rho_s.t()).t()
        total_input = synaptic_input + biases + current_input
        k3 = (-s3 + total_input) / taus
        
        # k4
        s4 = current_s + dt * k3
        rho_s = torch.tanh(s4)
        if s4.dim() == 1:
            synaptic_input = torch.mv(weights, rho_s)
        else:
            synaptic_input = torch.matmul(weights, rho_s.t()).t()
        total_input = synaptic_input + biases + current_input
        k4 = (-s4 + total_input) / taus
        
        current_s = current_s + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # --- Hard Clamping ---
        if input_mask is not None:
             current_s = current_s * (1 - input_mask) + input_vector * input_mask

        # --- Update Eligibility Traces ---
        # dz/dt = -z + rho(i) * rho(j)
        
        idx_i = indices[0]
        idx_j = indices[1]
        
        final_rho = torch.tanh(current_s)
        rho_i = final_rho[idx_i]
        rho_j = final_rho[idx_j]
        
        coincidence = rho_i * rho_j
        
        decay_factor = 1.0 - (dt / tau_z)
        current_z = current_z * decay_factor + (dt * coincidence)
        
        # Check Convergence
        diff = torch.norm(current_s - old_state)
        step_count += 1

    return current_s, current_z

class DragonEngineTorch:
    def __init__(self, num_nodes, indices, values, biases, taus, dt=0.01, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.dt = dt
        self.device = device
        self.num_nodes = num_nodes
        
        # Sparse Weights (COO) components
        self.indices = torch.tensor(indices, dtype=torch.long, device=device)
        self.weight_values = torch.tensor(values, dtype=torch.float32, device=device)
        
        # Parameters
        self.taus = torch.tensor(taus, dtype=torch.float32, device=device)
        self.biases = torch.tensor(biases, dtype=torch.float32, device=device)
        
        # State
        self.state = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        self.trace_values = torch.zeros_like(self.weight_values) # Store traces as sparse values
        
    def activation_function(self, s):
        return torch.tanh(s)

    def settle(self, input_vector, max_steps=5000, tol=1e-4, nudge_target=None, beta=0.0, nudge_mask=None, input_mask=None, inhibition_mask=None, inhibition_beta=0.0):
        """
        Runs the settling loop using JIT compiled function.
        """
        if not isinstance(input_vector, torch.Tensor):
            input_vector = torch.tensor(input_vector, dtype=torch.float32, device=self.device)
            
        nudge_t = None
        if nudge_target is not None:
             if not isinstance(nudge_target, torch.Tensor):
                 nudge_t = torch.tensor(nudge_target, dtype=torch.float32, device=self.device)
             else:
                 nudge_t = nudge_target

        nudge_m = None
        if nudge_mask is not None:
             if not isinstance(nudge_mask, torch.Tensor):
                 nudge_m = torch.tensor(nudge_mask, dtype=torch.float32, device=self.device)
             else:
                 nudge_m = nudge_mask
        
        input_m = None
        if input_mask is not None:
             if not isinstance(input_mask, torch.Tensor):
                 input_m = torch.tensor(input_mask, dtype=torch.float32, device=self.device)
             else:
                 input_m = input_mask
                  
        inhib_m = None
        if inhibition_mask is not None:
             if not isinstance(inhibition_mask, torch.Tensor):
                 inhib_m = torch.tensor(inhibition_mask, dtype=torch.float32, device=self.device)
             else:
                 inhib_m = inhibition_mask
        
        # Call JIT function
        self.state, self.trace_values = jit_solve_dynamics(
            self.state,
            self.trace_values,
            self.indices,
            self.weight_values,
            self.biases,
            self.taus,
            input_vector,
            self.dt,
            max_steps,
            tol,
            nudge_t,
            nudge_m,
            beta,
            input_m,
            inhib_m,
            inhibition_beta
        )
        return self.activation_function(self.state)

    def update_weights_eq_prop(self, state_free, state_nudged, beta, learning_rate):
        """
        EqProp Update: dW ~ (rho_cov_nudged - rho_cov_free) / beta
        """
        rho_free = self.activation_function(state_free)
        rho_nudged = self.activation_function(state_nudged)
        
        # Sparse Update: dW_ij ~ (rho_n[i]*rho_n[j] - rho_f[i]*rho_f[j]) / beta
        # We only update existing edges (defined by self.indices)
        
        idx_i = self.indices[0]
        idx_j = self.indices[1]
        
        # Vectorized gather of activations for edge endpoints
        rf_i = rho_free[idx_i]
        rf_j = rho_free[idx_j]
        rn_i = rho_nudged[idx_i]
        rn_j = rho_nudged[idx_j]
        
        # Compute gradient for each edge value
        grad_values = ((rn_i * rn_j) - (rf_i * rf_j)) / beta
        
        # Apply update
        self.weight_values += learning_rate * grad_values
        
    def enforce_symmetry(self):
        """
        Enforces W_ij = W_ji by averaging values for symmetric pairs.
        Note: This is expensive if edges are not ordered.
        Assumption: The graph construction ensures if (i,j) exists, (j,i) exists at the reciprocal index.
        Optimization: We can't easily find reciprocal index in COO without sorting.
        Fast Approximate Enforce:
        Actually, if we update W_ij and W_ji identically, symmetry is preserved.
        jit_solve_dynamics updates Z_ij using rho_i * rho_j.
        rho_i * rho_j is symmetric.
        So Z_ij calculation is symmetric IF Z_ij started symmetric.
        
        Same for weight updates below.
        So we just need to ensure initialization is symmetric (Graph does this).
        But let's add a check/fix method just in case drifts happen (e.g. numerical error).
        """
        # For now, rely on updates being symmetric by definition (product of scalars).
        pass

    def update_weights_dopamine(self, reward_signal, learning_rate):
        """
        Dopamine-modulated plasticity.
        Delta W_ij = eta * D(t) * z_ij
        """
        # Simple update
        dw = learning_rate * reward_signal * self.trace_values
        self.weight_values += dw
        
    def update_weights_hebbian(self, learning_rate, decay=0.0001):
        """
        Pure Hebbian Learning (Section 5.1).
        Delta W_ij = eta * (rho_i * rho_j - alpha * W_ij)
        """
        # Re-compute coincidence
        # Note: jit_solve ALREADY computed coincidence into traces.
        # But Hebbian is instantaneous rho*rho, or filtered?
        # PDF says "Heabbian Learning" for "Babbling".
        # Eq: Delta W ~ rho_i * rho_j.
        
        idx_i = self.indices[0]
        idx_j = self.indices[1]
        
        rho = self.activation_function(self.state)
        ri = rho[idx_i]
        rj = rho[idx_j]
        
        coincidence = ri * rj
        
        # Update with decay
        delta = learning_rate * (coincidence - decay * self.weight_values)
        self.weight_values += delta
