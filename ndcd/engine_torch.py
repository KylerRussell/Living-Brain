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
        # 1. Input Clamping (Moved to start)
        if input_mask is not None:
             current_s = current_s * (1 - input_mask) + input_vector * input_mask

        old_state = current_s
        
        # 1. Nudge Logic / Input
        current_input = input_vector
        if nudge_target is not None and beta != 0.0:
            # Use appropriate activation for error calculation
            # For LIF, we might still use potential or smoothed spike rate?
            # Keeping tanh(s) as proxy for "activity state" even in LIF for gradient guidance
            rho_s = torch.tanh(current_s)
            # Fix: Apply error directly to potential space: diff_nudge = nudge_target - rho_s
            # Removed derivative d_rho to fix gradient scaling
            diff_nudge = (nudge_target - rho_s)
            
            if nudge_mask is not None:
                diff_nudge = diff_nudge * nudge_mask
            nudge_force = beta * diff_nudge
            current_input = current_input + nudge_force
            
            
        if input_mask is not None:
            # Re-clamp inputs after nudge (if they overlap) to ensure strict clamping
             current_s = current_s * (1 - input_mask) + input_vector * input_mask

        # 1.6 Lateral Inhibition (Softmax-like competition)
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

        # --- State Clamping ---
        # Prevent saturation: keep neurons in responsive regime of tanh
        # At |s|=3, tanh'(s) ~ 0.01 which is small but usable;
        # at |s|=6, tanh'(s) ~ 0.00001 which kills all gradients.
        current_s = current_s.clamp(-3.0, 3.0)

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

        
        # --- Hard Clamping (Removed - moved to start) ---

        # 1.5 Nudge Logic / Input

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

    def update_weights_eq_prop(self, state_pos, state_neg, beta, learning_rate, decay=0.0):
        """
        Symmetric EqProp Update: dW ~ (rho_pos*rho_pos - rho_neg*rho_neg) / (2 * beta)
        Args:
            state_pos: Equilibrium state with +beta nudge
            state_neg: Equilibrium state with -beta nudge
            beta: Nudge strength
        """
        rho_pos = self.activation_function(state_pos)
        rho_neg = self.activation_function(state_neg)
        
        # Sparse Update: dW_ij ~ (rho_pos[i]*rho_pos[j] - rho_neg[i]*rho_neg[j]) / (2 * beta)
        # We only update existing edges (defined by self.indices)
        
        idx_i = self.indices[0]
        idx_j = self.indices[1]
        
        # Vectorized gather of activations for edge endpoints
        rp_i = rho_pos[idx_i]
        rp_j = rho_pos[idx_j]
        rn_i = rho_neg[idx_i]
        rn_j = rho_neg[idx_j]
        
        # Compute gradient for each edge value
        # Symmetry Note: Both directions produce the same term since rho_i * rho_j = rho_j * rho_i
        
        grad_values = ((rp_i * rp_j) - (rn_i * rn_j)) / (2.0 * beta)
        
        # Apply update
        delta = learning_rate * grad_values
        if decay > 0.0:
            delta -= (decay * self.weight_values)
            
        self.weight_values += delta
        
        # --- Fix: Enable Bias Learning ---
        # Gradient for bias is (rho_pos - rho_neg) / (2 * beta)
        bias_grad = (rho_pos - rho_neg) / (2.0 * beta)
        self.biases += learning_rate * bias_grad

        # Fix: Strict Weight Clipping and Removal of Symmetry Enforcement
        # Directed Equilibrium Propagation requires asymmetric weights (no forced symmetry).
        self.weight_values.clamp_(-1.0, 1.0)
        
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
        
        
    def update_weights_hebbian(self, learning_rate=0.01, oja_alpha=0.001):
        """
        Hebbian update with proper Oja normalization
        """
        with torch.no_grad():
            current_activation = torch.tanh(self.state)
            
            # Get pre and post activations
            ri = current_activation[self.indices[0]]  # Presynaptic (as per user map)
            rj = current_activation[self.indices[1]]  # Postsynaptic (as per user map)
            
            # Standard Hebbian term
            hebbian_update = learning_rate * ri * rj
        
        # Oja normalization (use postsynaptic variance as requested)
        oja_decay = oja_alpha * (rj * rj) * self.weight_values
        
        # Combined update
        self.weight_values += hebbian_update - oja_decay
        
        # CRITICAL FIX 2: Clip weights after each update
        self.weight_values.clamp_(-1.0, 1.0)

    def damp_weights(self, factor=0.9):
        """
        Damps the recurrent weights by a factor.
        Used for reactive control when spectral radius explodes.
        """
        self.weight_values *= factor

    def remodel_structure(self, turnover_rate=0.05, protected_nodes=None):
        """
        Dynamically changes the brain's wiring to maintain constant density.
        1. Prune: Removes bottom 'turnover_rate' (e.g. 5%) of edges based on Value/Cost.
           Metric: |Weight| / Distance.
           *Protected Nodes*: Edges attached to these nodes are never pruned.
        2. Grow: Adds exactly the number of edges pruned (maintaining total count).
        """
        # --- 1. METRIC CALCULATION ---
        current_count = self.weight_values.shape[0]
        num_to_prune = int(current_count * turnover_rate)
        
        row_indices = self.indices[0]
        col_indices = self.indices[1]
        
        pos_i = self.positions[row_indices]
        pos_j = self.positions[col_indices]
        
        # Euclidean distance (add epsilon to avoid div-by-zero)
        distances = torch.norm(pos_i - pos_j, dim=1) + 1e-6
        
        # Score: ROI (Return on Investment). High weight at long distance is harder to keep.
        scores = torch.abs(self.weight_values) / distances
        
        # --- PROTECT NODES ---
        if protected_nodes is not None:
             if not isinstance(protected_nodes, torch.Tensor):
                 protected_nodes = torch.tensor(protected_nodes, device=self.device)
             
             # Create mask of protected nodes
             # We can't use isin efficiently in older torch, so let's use a bool mask map
             node_mask = torch.zeros(self.num_nodes, dtype=torch.bool, device=self.device)
             node_mask[protected_nodes] = True
             
             # Check if src OR dst is protected
             is_protected = node_mask[row_indices] | node_mask[col_indices]
             
             # Set score to infinity so they are never in the bottom percentile
             scores[is_protected] = float('inf')
        
        # --- 2. PRUNING (Percentile) ---
        # We need to find the threshold score that separates the bottom 5%.
        # topk(largest=False) gives smallest.
        
        if num_to_prune > 0:
            # Find the indices of the smallest scores
            # torch.topk with largest=False returns smallest elements
            _, prune_indices = torch.topk(scores, num_to_prune, largest=False)
            
            # Create a boolean mask of edges to KEEP
            # It's faster to create a ones mask and set prune indices to 0
            keep_mask = torch.ones(current_count, dtype=torch.bool, device=self.device)
            keep_mask[prune_indices] = False
            
            # Apply Mask
            self.indices = self.indices[:, keep_mask]
            self.weight_values = self.weight_values[keep_mask]
            self.trace_values = self.trace_values[keep_mask]
            
        
        # --- 3. GROWTH (Restoration) ---
        # We want to get back to 'current_count'
        num_kept = self.weight_values.shape[0]
        num_to_add = current_count - num_kept
        
        # Generate random candidate pairs
        new_src = torch.randint(0, self.num_nodes, (num_to_add,), device=self.device)
        new_dst = torch.randint(0, self.num_nodes, (num_to_add,), device=self.device)
        
        # Enforce No-Self-Loops (simple check)
        mask_no_self = new_src != new_dst
        new_src = new_src[mask_no_self]
        new_dst = new_dst[mask_no_self]
        
        # Note: If self-loops removed, we might add slightly fewer than intended.
        # That's fine, it prevents infinite growth if we accidentally added more.
        # If we want exact count, we'd need a while loop, but approximate homeostasis is fine.
        
        # Create Bidirectional Pairs? 
        # Previous code created bidirectional:
        # p1_indices = torch.stack([new_src, new_dst])
        # p2_indices = torch.stack([new_dst, new_src])
        # This doubled the growth rate.
        # Here we are counting EDGES. 
        # If the graph is directed (indices has shape [2, M]), then each column is an edge.
        # If we prune M edges and add M edges, we are good.
        # BUT, if we want symmetry, we should add M/2 pairs.
        # The previous code: grew 'growth_rate' pairs -> 2 * growth_rate edges.
        # Our `current_count` is total directed edges.
        # If we just add random directed edges, we might lose symmetry.
        # EqProp usually assumes symmetry (W_ij = W_ji).
        # Let's enforce symmetry in growth.
        # We have 'num_to_add' slots.
        # We should generate num_to_add // 2 PAIRS.
        
        pairs_to_add = num_to_add // 2
        if pairs_to_add > 0:
            p_src = torch.randint(0, self.num_nodes, (pairs_to_add,), device=self.device)
            p_dst = torch.randint(0, self.num_nodes, (pairs_to_add,), device=self.device)
            
            # No self loops
            mask = p_src != p_dst
            p_src = p_src[mask]
            p_dst = p_dst[mask]
            
            # Pair 1: A -> B
            p1 = torch.stack([p_src, p_dst])
            # Pair 2: B -> A
            p2 = torch.stack([p_dst, p_src])
            
            new_indices = torch.cat([p1, p2], dim=1)
            
            # Initialize new weights
            # Small random or zero? Start at 0 to not shock dynamics.
            num_new = new_indices.shape[1]
            new_values = torch.zeros(num_new, device=self.device)
            new_traces = torch.zeros(num_new, device=self.device) # Traces 0
            
            self.indices = torch.cat([self.indices, new_indices], dim=1)
            self.weight_values = torch.cat([self.weight_values, new_values], dim=0)
            self.trace_values = torch.cat([self.trace_values, new_traces], dim=0)
            
        final_count = self.weight_values.shape[0]
        print(f"Brain Remodeled: {final_count} edges (Pruned {num_to_prune}, Added {final_count - num_kept})")

