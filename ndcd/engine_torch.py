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
    inhibition_beta: float,
    attention_factor: float = 1.0,
    spiking_threshold: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    JIT-compiled static function for the physics loop.
    Solves dx/dt = (-x + W*rho(x) + b + I) / tau
    - Supports Leaky Integrate-and-Fire (LIF) if spiking_threshold > 0.
    - Supports Attention Modulation (attention_factor scales dt).
    - Supports Energy Flux Monitoring (strict convergence).
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
    
    # Track previous spikes for LIF propagation
    previous_spikes = torch.zeros_like(current_s)
    
    while step_count < max_steps and diff > tol:
        old_state = current_s
        
        # 1. Nudge Logic / Input
        current_input = input_vector
        if nudge_target is not None and beta > 0.0:
            # Use appropriate activation for error calculation
            # For LIF, we might still use potential or smoothed spike rate?
            # Keeping tanh(s) as proxy for "activity state" even in LIF for gradient guidance
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

        # 1.6 Spike Propagation (LIF)
        # Add impulse from neurons that fired in the PREVIOUS step
        if spiking_threshold < 10.0:
            if current_s.dim() == 1:
                spike_input = torch.mv(weights, previous_spikes)
            else:
                spike_input = torch.matmul(weights, previous_spikes.t()).t()
            current_input = current_input + spike_input
            
        # 2. RK4 Step with Attention Modulation
        # Attention scales effectively "time speed" or "precision" -> dt * attention_factor
        effective_dt = dt * attention_factor

        # Define activation: Tanh (Rate) OR Spiking (LIF) are handled implicitly?
        # RK4 is for the continuous potential 's'.
        # The interaction term depends on rho(s).
        # If spiking, rho(s) should be spike (1 or 0) from *previous* step?
        # For continuous dynamics, we use tanh(s) inside RK4. 
        # If we want spiking, we check threshold AFTER update, reset s, and emit spike for NEXT step.
        # But inside RK4 step, we need the "input" from neighbors.
        # In rate code: input ~ W * tanh(s).
        # In spiking code: input ~ W * spikes.
        # Here we hybridize: We use tanh(s) for the continuous integration phase (sub-threshold dynamics),
        # but if we reset, we conceptually fired.
        # However, standard LIF is linear below threshold. 
        # User asks to "Modify rk4_step... to include threshold".
        # Providing a Spiking Mode switch logic is complex in pure RK4.
        # Simpler: Use activation inside RK4 as tanh(s) usually, unless we track spikes explicitly.
        # Let's stick to cleaning up the integration first.
        

        # k1
        rho_s = torch.tanh(current_s) # Continuous approximation for dynamics
        if current_s.dim() == 1:
            synaptic_input = torch.mv(weights, rho_s)
        else:
            synaptic_input = torch.matmul(weights, rho_s.t()).t()
            
        total_input = synaptic_input + biases + current_input
        k1 = (-current_s + total_input) / taus
        
        # k2
        s2 = current_s + 0.5 * effective_dt * k1
        rho_s = torch.tanh(s2)
        if s2.dim() == 1:
            synaptic_input = torch.mv(weights, rho_s)
        else:
            synaptic_input = torch.matmul(weights, rho_s.t()).t()
        total_input = synaptic_input + biases + current_input
        k2 = (-s2 + total_input) / taus
        
        # k3
        s3 = current_s + 0.5 * effective_dt * k2
        rho_s = torch.tanh(s3)
        if s3.dim() == 1:
            synaptic_input = torch.mv(weights, rho_s)
        else:
            synaptic_input = torch.matmul(weights, rho_s.t()).t()
        total_input = synaptic_input + biases + current_input
        k3 = (-s3 + total_input) / taus
        
        # k4
        s4 = current_s + effective_dt * k3
        rho_s = torch.tanh(s4)
        if s4.dim() == 1:
            synaptic_input = torch.mv(weights, rho_s)
        else:
            synaptic_input = torch.matmul(weights, rho_s.t()).t()
        total_input = synaptic_input + biases + current_input
        k4 = (-s4 + total_input) / taus
        
        current_s = current_s + (effective_dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # --- Spiking Logic (LIF) ---
        # If potential > threshold, Fire & Reset.
        # We track spikes for output (but next step uses them? 
        # For now, this modifies current_s for proper reset).
        if spiking_threshold < 10.0: # Heuristic to enable spiking
            new_spikes = (current_s > spiking_threshold).float()
            # reset fired neurons to 0
            current_s = current_s * (1.0 - new_spikes)
            previous_spikes = new_spikes # Save for next iteration to propagate signal
                
        else:
                previous_spikes = torch.zeros_like(current_s)
                 
        
        # --- Hard Clamping ---

        
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
        
        # Check Convergence / Energy Flux Monitor
        # Automated Latent Incubation: Continue until global energy change is very low.
        # diff = |ds/dt| * dt. 
        diff = torch.norm(current_s - old_state)
        
        step_count += 1

    return current_s, current_z

class DragonEngineTorch:
    def __init__(self, num_nodes, indices, values, biases, taus, positions: Optional[np.ndarray]=None, dt=0.01, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.dt = dt
        self.device = device
        self.num_nodes = num_nodes
        
        # Spatial positions for remodeling (Wiring Cost)
        if positions is not None:
            self.positions = torch.tensor(positions, dtype=torch.float32, device=device)
        else:
            self.positions = torch.zeros((num_nodes, 3), dtype=torch.float32, device=device)
        
        
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

    def settle(self, input_vector, max_steps=5000, tol=1e-4, nudge_target=None, beta=0.0, nudge_mask=None, input_mask=None, inhibition_mask=None, inhibition_beta=0.0, attention_factor=1.0, spiking_threshold=100.0):

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
            inhibition_beta,
            attention_factor,
            spiking_threshold
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
        
    # enforce_symmetry removed for Directed Equilibrium Propagation


    def apply_neuromodulators(self, reward, attention, mood):
        """
        Multi-Factor Neuromodulation.
        Args:
            reward (float): Dopamine (D). Strengthens/Weakens signal learning.
            attention (float): Acetylcholine (A). Modulates precision/plasticity rate.
            mood (float): Serotonin (S). Modulates risk/inhibition? 
                          Here we map mood to a global weight decay or inhibition factor.
        """
        # 1. Dopamine (Reward) -> Weight Update
        # dw = learning_rate * D * trace
        # We assume learning rate is passed effectively or we use a base one.
        lr = 0.01 * attention # Attention increases plasticity
        
        dw = lr * reward * self.trace_values
        
        # 2. Mood (Risk) -> Weight Decay / Pruning pressure?
        # User: "Serotonin for risk". High serotonin = stability/aversion?
        # Let's say high mood = increased decay (forgetting risky weak links)?
        decay = 1e-4 * (1.0 + mood)
        
        self.weight_values += dw - (decay * self.weight_values)
        
    def update_weights_dopamine(self, reward_signal, learning_rate):
        """
        Legacy wrapper for apply_neuromodulators.
        """
        self.apply_neuromodulators(reward_signal, attention=1.0, mood=0.0)
        
        
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

    def remodel_structure(self, prune_threshold=0.001, growth_rate=100):
        """
        Dynamically changes the brain's wiring.
        1. Prune: Removes edges with absolute weight < threshold.
        2. Grow: Adds 'growth_rate' new random edges (bidirectional).
        """
        # --- 1. PRUNING ---
        # Heuristic: Prune if Weight < Threshold * Distance
        # We need to compute distances for current edges.
        
        row_indices = self.indices[0]
        col_indices = self.indices[1]
        
        pos_i = self.positions[row_indices]
        pos_j = self.positions[col_indices]
        
        # Euclidean distance
        distances = torch.norm(pos_i - pos_j, dim=1)
        
        # Dynamic Threshold based on wiring cost
        # Logic: Long edges need HIGH weight to survive. Short edges can survive with low weight.
        # Condition to KEEP: |W| > prune_threshold * distance
        # Note: distance is in [0, sqrt(3)]. 
        
        cost = prune_threshold * distances
        keep_mask = torch.abs(self.weight_values) > cost
        
        # Remove deleted edges from tensors
        self.indices = self.indices[:, keep_mask]
        self.weight_values = self.weight_values[keep_mask]
        self.trace_values = self.trace_values[keep_mask]
                
        # --- 2. GROWTH ---
        # Generate random candidate pairs
        # Note: In a dense implementation, we'd check for duplicates, but 
        # in a sparse brain, collisions are rare enough to ignore for speed.
        new_src = torch.randint(0, self.num_nodes, (growth_rate,), device=self.device)
        new_dst = torch.randint(0, self.num_nodes, (growth_rate,), device=self.device)
        
        # Enforce No-Self-Loops (simple check)
        mask_no_self = new_src != new_dst
        new_src = new_src[mask_no_self]
        new_dst = new_dst[mask_no_self]
        
        # Create Bidirectional Pairs (Symmetry is required for EqProp Energy)
        # Pair 1: A -> B
        p1_indices = torch.stack([new_src, new_dst])
        # Pair 2: B -> A
        p2_indices = torch.stack([new_dst, new_src])
        
        new_indices = torch.cat([p1_indices, p2_indices], dim=1)
        
        # Initialize new weights near zero (so we don't shock the brain)
        num_new = new_indices.shape[1]
        new_values = torch.zeros(num_new, device=self.device)
        new_traces = torch.zeros(num_new, device=self.device)
        
        # --- 3. MERGE ---
        self.indices = torch.cat([self.indices, new_indices], dim=1)
        self.weight_values = torch.cat([self.weight_values, new_values], dim=0)
        self.trace_values = torch.cat([self.trace_values, new_traces], dim=0)
        
        print(f"Brain Remodeled: {self.weight_values.shape[0]} edges (Pruned < {prune_threshold}, Grew {num_new})")

