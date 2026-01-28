
import numpy as np
import torch

class DragonEngineTorch:
    def __init__(self, graph, dt=0.01, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.dt = dt
        self.device = device
        
        # Convert graph components to PyTorch tensors
        self.num_nodes = graph.num_nodes
        
        # Weights (sparse or dense depending on size, using dense for now for speed with <10k nodes)
        # For huge graphs, torch.sparse is better, but dense matmul is faster for sub-10k.
        self.weights = torch.tensor(graph.weights, dtype=torch.float32, device=device)
        
        # Time constants
        self.taus = torch.tensor(graph.taus, dtype=torch.float32, device=device)
        
        # Biases
        self.biases = torch.tensor(graph.biases, dtype=torch.float32, device=device)
        
        # State (hidden state s) - Batch size 1 by default, but can be expanded
        self.state = torch.tensor(graph.states, dtype=torch.float32, device=device)
        
        # Eligibility Traces (for RL)
        self.traces = torch.zeros((self.num_nodes, self.num_nodes), dtype=torch.float32, device=device)
        
    def activation_function(self, s):
        return torch.tanh(s)

    def compute_derivative(self, state, clamped_input):
        """
        Computes ds/dt = (-s + W*rho(s) + b + I) / tau
        """
        rho_s = self.activation_function(state)
        # Vectorized synaptic input: W @ rho
        # If state is [Batch, Nodes], we need W @ rho.T or similar.
        # Here we assume state is [Nodes] (1D) for simplicity, or handle batching.
        
        if state.dim() == 1:
            synaptic_input = torch.mv(self.weights, rho_s)
        else:
            # Batch mode: state is [Batch, Nodes], W is [Nodes, Nodes]
            # Output should be [Batch, Nodes] -> (rho @ W.T)
            synaptic_input = torch.matmul(rho_s, self.weights.t())
            
        total_input = synaptic_input + self.biases + clamped_input
        
        d_s = (-state + total_input) / self.taus
        return d_s

    def rk4_step(self, current_state, clamped_input):
        k1 = self.compute_derivative(current_state, clamped_input)
        k2 = self.compute_derivative(current_state + 0.5 * self.dt * k1, clamped_input)
        k3 = self.compute_derivative(current_state + 0.5 * self.dt * k2, clamped_input)
        k4 = self.compute_derivative(current_state + self.dt * k3, clamped_input)
        
        new_state = current_state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return new_state

    def settle(self, input_vector, duration_steps, nudge_target=None, beta=0.0):
        """
        Runs the settling loop.
        Input vector should be a Tensor on the correct device.
        """
        if not isinstance(input_vector, torch.Tensor):
            input_vector = torch.tensor(input_vector, dtype=torch.float32, device=self.device)
            
        current_s = self.state.clone()
        
        # Nudging logic for EqProp (Static input modification)
        effective_input = input_vector.clone()
        
        # Dynamic nudging loop
        for _ in range(duration_steps):
            current_input = effective_input
            if nudge_target is not None and beta > 0:
                 if not isinstance(nudge_target, torch.Tensor):
                     nudge_target = torch.tensor(nudge_target, dtype=torch.float32, device=self.device)
                     
                 # Force ~ beta * (Target - rho(s))
                 rho_s = self.activation_function(current_s)
                 nudge_force = beta * (nudge_target - rho_s)
                 current_input = current_input + nudge_force

            current_s = self.rk4_step(current_s, current_input)
            
        self.state = current_s
        return self.activation_function(current_s)

    def update_weights_eq_prop(self, state_free, state_nudged, beta, learning_rate):
        """
        EqProp Update: dW ~ (rho_cov_nudged - rho_cov_free) / beta
        """
        rho_free = self.activation_function(state_free)
        rho_nudged = self.activation_function(state_nudged)
        
        # Outer products
        # If batch, we average over batch
        if rho_free.dim() == 1:
            co_free = torch.outer(rho_free, rho_free)
            co_nudged = torch.outer(rho_nudged, rho_nudged)
        else:
            # Batch mode: [B, N] -> [B, N, N] -> mean -> [N, N]
            # einsum 'bi,bj->bij'
            co_free = torch.einsum('bi,bj->bij', rho_free, rho_free).mean(dim=0)
            co_nudged = torch.einsum('bi,bj->bij', rho_nudged, rho_nudged).mean(dim=0)
            
        gradient = (co_nudged - co_free) / beta
        
        self.weights += learning_rate * gradient
        
        # Symmetrize
        self.weights = (self.weights + self.weights.t()) / 2

    def update_weights_hebbian(self, state, learning_rate, decay=0.0):
        rho = self.activation_function(state)
        if rho.dim() == 1:
            hebbian = torch.outer(rho, rho)
        else:
            hebbian = torch.einsum('bi,bj->bij', rho, rho).mean(dim=0)
            
        decay_term = decay * self.weights
        self.weights += learning_rate * (hebbian - decay_term)
        self.weights = (self.weights + self.weights.t()) / 2
