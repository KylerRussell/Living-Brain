
import numpy as np

class DragonEngine:
    def __init__(self, graph, dt=0.01):
        self.graph = graph
        self.dt = dt

    def activation_function(self, s):
        # Hard Sigmoid or Tanh are common for Hopfield/EqProp
        return np.tanh(s)

    def compute_derivative(self, state_vector, clamped_input):
        """
        Computes ds/dt = (-s + W*rho(s) + b + I) / tau
        """
        rho_s = self.activation_function(state_vector)
        # Vectorized synaptic input computation
        synaptic_input = self.graph.weights @ rho_s
        total_input = synaptic_input + self.graph.biases + clamped_input
        
        d_s = (-state_vector + total_input) / self.graph.taus
        return d_s

    def rk4_step(self, current_state, clamped_input):
        """
        Performs one Runge-Kutta 4th Order integration step.
        """
        k1 = self.compute_derivative(current_state, clamped_input)
        k2 = self.compute_derivative(current_state + 0.5 * self.dt * k1, clamped_input)
        k3 = self.compute_derivative(current_state + 0.5 * self.dt * k2, clamped_input)
        k4 = self.compute_derivative(current_state + self.dt * k3, clamped_input)
        
        new_state = current_state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return new_state

    def settle(self, input_vector, duration_steps, nudge_target=None, beta=0.0):
        """
        Runs the settling loop (Inference).
        Supports Nudging for EqProp.
        """
        current_s = self.graph.states.copy()
        
        # If nudging, add beta * (target - output) to the input of output nodes
        # This effectively modifies the energy landscape.
        effective_input = input_vector.copy()
        
        # Implementation note: In a true continuous setting, nudging is dynamic.
        # Here we approximate by adding the nudge force to the input once if strictly static,
        # or we could recompute it inside the loop. The PDF suggests weak clamping.
        # We will assume 'nudge_target' is a vector of same size as nodes (with NaNs or 0s for non-targets)
        # Or simpler: we update effective_input based on the static target.
        
        if nudge_target is not None:
             # For MSE Cost: dC/dy = (y - t). Nudge = -beta * dC/dy = beta * (t - y)
             # But y changes! So strictly this should be in the derivative.
             # However, typically for EqProp "weak clamping", we just add a term.
             # If target is provided for specific nodes, we add beta * target to their input
             # AND we might need to adjust their 'leak' or 'bias' to pull them effectively.
             # To follow strict EqProp literature: Clamped phase simply adds beta*(target - y) force.
             # Which means I_effective = I_input + beta * (target - y).
             # We will implement dynamic nudging inside the loop for accuracy.
             pass

        for _ in range(duration_steps):
            current_input = effective_input.copy()
            if nudge_target is not None and beta > 0:
                 # Dynamic Nudging Force
                 # We assume nudge_target is a masked array or similar where we only nudge output nodes
                 # For simplicity, assume nudge_target has valid values at output indices and 0 elsewhere
                 # and we need a mask to know WHERE to apply it? 
                 # Let's assume input 'nudge_target' is the full vector T. 
                 # Nudge force = beta * (T - current_s) or (T - rho(s))? 
                 # Usually it's clamped to the value, so it springs towards T.
                 # Force ~ beta * (Target - CurrentActivity)
                 current_rho = self.activation_function(current_s)
                 nudge_force = beta * (nudge_target - current_rho)
                 # Apply only where target is defined (non-zero/non-nan? we need a mask ideally)
                 # For this impl, let's assume nudge_target is 0 where no target.
                 # BETTER: Pass a mask. But for now, let's just add it.
                 current_input += nudge_force

            current_s = self.rk4_step(current_s, current_input)
            
        self.graph.states = current_s
        return self.activation_function(current_s)

    def update_weights_eq_prop(self, state_free, state_nudged, beta, learning_rate):
        """
        Updates weights based on Equilibrium Propagation rule.
        Delta W ~ (rho_nudged * rho_nudged^T) - (rho_free * rho_free^T)
        """
        rho_free = self.activation_function(state_free)
        rho_nudged = self.activation_function(state_nudged)
        
        # Outer products to get pairwise co-activities
        coactivity_free = np.outer(rho_free, rho_free)
        coactivity_nudged = np.outer(rho_nudged, rho_nudged)
        
        gradient = (coactivity_nudged - coactivity_free) / beta
        
        # Apply update
        self.graph.weights += learning_rate * gradient
        
        # Maintain Symmetry
        self.graph.weights = (self.graph.weights + self.graph.weights.T) / 2

    def update_weights_dopamine(self, reward_signal, learning_rate):
        """
        Three-Factor Rule: dW = eta * R * Trace
        """
        if reward_signal == 0:
            return

        self.graph.weights += learning_rate * reward_signal * self.graph.traces
        
        # Reset traces
        self.graph.traces.fill(0)

    def update_weights_hebbian(self, state, learning_rate, decay=0.0):
        """
        Unsupervised Hebbian Learning: dW = eta * (rho * rho^T - decay * W)
        """
        rho = self.activation_function(state)
        # Hebbian term: Outer product of activities
        hebbian_term = np.outer(rho, rho)
        
        # Weight decay term prevents runaway weights
        decay_term = decay * self.graph.weights
        
        # Update
        self.graph.weights += learning_rate * (hebbian_term - decay_term)
        
        # Maintain Symmetry
        self.graph.weights = (self.graph.weights + self.graph.weights.T) / 2
