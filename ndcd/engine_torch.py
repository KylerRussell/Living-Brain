import numpy as np
import torch
from typing import Optional

@torch.jit.script
def jit_solve_dynamics(
    initial_state: torch.Tensor,
    indices: torch.Tensor,
    weight_values: torch.Tensor,
    biases: torch.Tensor,
    taus: torch.Tensor,
    input_vector: torch.Tensor,
    dt: float,
    duration_steps: int,
    nudge_target: Optional[torch.Tensor],
    beta: float
) -> torch.Tensor:
    """
    JIT-compiled static function for the physics loop.
    Solves dx/dt = (-x + W*tanh(x) + b + I) / tau
    """
    num_nodes = initial_state.size(0)
    current_s = initial_state
    
    # Construct sparse tensor view for matmul on the fly
    # Note: In JIT, this construction is efficient if indices/values are tensors
    weights = torch.sparse_coo_tensor(indices, weight_values, (num_nodes, num_nodes))

    for _ in range(duration_steps):
        # 1. Nudge Logic / Input
        current_input = input_vector
        if nudge_target is not None and beta > 0.0:
            rho_s = torch.tanh(current_s)
            nudge_force = beta * (nudge_target - rho_s)
            current_input = current_input + nudge_force
            
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
        
    return current_s

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
        
    def activation_function(self, s):
        return torch.tanh(s)

    def settle(self, input_vector, duration_steps, nudge_target=None, beta=0.0):
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
        
        # Call JIT function
        self.state = jit_solve_dynamics(
            self.state,
            self.indices,
            self.weight_values,
            self.biases,
            self.taus,
            input_vector,
            self.dt,
            duration_steps,
            nudge_t,
            beta
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
        
    def update_weights_hebbian(self, state, learning_rate, decay=0.0):
        # Hebbian implementation for sparse is non-trivial if decay depends on W
        # Leaving as placeholder or sparse implementation if needed
        pass
