
import ray
import torch
import numpy as np
import os
import time
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from ndcd.graph import DynamicGraph
from ndcd.engine_torch import DragonEngineTorch
from ndcd.sensory import ByteSensoryInterface
from ndcd.curriculum_gen import generate_chars, generate_words, generate_quotes

@ray.remote(num_cpus=1)
class DragonWorker:
    def __init__(self, num_nodes, indices, values, biases, taus, input_weights, readout_weights, data_offset, data_len_chunk, full_data_bytes):
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'

        # Initialize Engine with Sparse Components
        self.engine = DragonEngineTorch(num_nodes, indices, values, biases, taus, dt=0.01, device=self.device)
        
        # Architecture Components
        self.input_weights = torch.tensor(input_weights, dtype=torch.float32, device=self.device)
        self.readout_weights = torch.tensor(readout_weights, dtype=torch.float32, device=self.device)
        self.readout_bias = torch.zeros(256, dtype=torch.float32, device=self.device)
        
        self.io = ByteSensoryInterface(input_offset=0, output_offset=256, device=self.device)
        
        # Slice the data
        self.data_bytes = full_data_bytes[data_offset : data_offset + data_len_chunk]
        self.data_len = len(self.data_bytes)
        self.current_idx = 0
        
    def get_weights(self):
        return (self.engine.weight_values.cpu().numpy(), 
                self.engine.biases.cpu().numpy(),
                self.readout_weights.cpu().numpy(),
                self.readout_bias.cpu().numpy())
        
    def set_weights(self, weight_values, biases, readout_weights, readout_bias):
        self.engine.weight_values = torch.tensor(weight_values, device=self.device, dtype=torch.float32)
        self.engine.biases = torch.tensor(biases, device=self.device, dtype=torch.float32)
        self.readout_weights = torch.tensor(readout_weights, device=self.device, dtype=torch.float32)
        self.readout_bias = torch.tensor(readout_bias, device=self.device, dtype=torch.float32)

    def train_step(self, steps=10, enable_plasticity=True):
        """
        Runs 'steps' of training.
        enable_plasticity: If True, internal weights are updated via Equilibrium Propagation (or Hebbian).
        """
        start_weight_values = self.engine.weight_values.clone()
        start_biases = self.engine.biases.clone()
        
        # Gradients for Readout
        grad_readout_w = torch.zeros_like(self.readout_weights)
        grad_readout_b = torch.zeros_like(self.readout_bias)
        
        loss_accum = 0
        correct_count = 0
        readout_lr = 0.05 # Higher for Curriculum
        internal_lr = 0.001 # Small plasticity
        
        for _ in range(steps):
            if self.current_idx >= self.data_len - 1:
                self.current_idx = 0
                
            input_byte = self.data_bytes[self.current_idx]
            target_byte = self.data_bytes[self.current_idx + 1]
            self.current_idx += 1
            
            # 1. Input Projection
            input_idx = int(input_byte)
            input_vec = self.input_weights[:, input_idx]
            
            # 2. Settle (Free Phase)
            self.engine.settle(input_vec, duration_steps=10)
            state_free = self.engine.state.clone()
            
            # 3. Readout Prediction
            logits = torch.mv(self.readout_weights, state_free) + self.readout_bias
            
            # 4. Compute Loss
            probs = torch.softmax(logits, dim=0)
            target_idx = int(target_byte)
            curr_loss = -torch.log(probs[target_idx] + 1e-8)
            loss_accum += curr_loss.item()
            
            if torch.argmax(probs).item() == target_idx:
                correct_count += 1
            
            # 5. Backprop for Readout
            d_logits = probs.clone()
            d_logits[target_idx] -= 1.0
            
            grad_readout_w += torch.outer(d_logits, state_free)
            grad_readout_b += d_logits
            
            # 6. EqProp / Hebbian (Internal Weights)
            if enable_plasticity:
                # Simple Hebbian: Strengthen connections between active nodes
                # Delta W = eta * (pre * post)
                # But we are using sparse W.
                # Actually, let's use the explicit EqProp nudge if we can, or just Hebbian on state_free?
                # For stability in this sparse setting, let's use a weak Hebbian-like update restricted to existing connections
                # We need to map state outer product to sparse values.
                # This is computationally expensive in Python loop.
                # Let's skip per-step update here and rely on the fact that `update_weights_eq_prop` isn't fully implemented in this loop cleanly.
                # Wait, DragonEngineTorch has `update_weights_eq_prop`.
                # We need a 'nudge'.
                # Nudge target = state - alpha * backprop_error.
                # backprop_error at state = W_out.T @ d_logits
                error_signal = torch.mv(self.readout_weights.T, d_logits) # [N]
                # Nudge towards better state (minimizing output error)
                # target_state = state - 0.1 * error_signal
                # But settle pushes towards energy min. EqProp uses clamped phase.
                # Let's simplify: Standard Hebbian on the Free phase is "Unsupervised Learning".
                # To learn the TASK, we need the error signal.
                # Let's use the error signal to modify the weights.
                # dL/dW_internal = dL/dS * dS/dW
                # This is complex. 
                # Alternative: Just noise injection or "Dreaming".
                # For now, let's keep Internal Plasticity simple: Hebbian Reinforcement of repeated patterns.
                # self.engine.update_weights(state_free, lr=internal_lr) <-- Hypothetical function.
                # Let's stick to Readout training being the primary driver, but maybe the previous failure was just the DATA complexity.
                # IF we want internal plasticity, we really need the EqProp phases (Free vs Clamped).
                # Clamped: Input + Output Clamp.
                # Let's implement Clamped Phase!
                
                # Phase 2: Weakly Clamp Output
                target_vec = torch.zeros(256, device=self.device)
                target_vec[target_idx] = 1.0
                # Project target back to nodes?
                # feedback_current = W_out.T @ (target - prediction) ?
                # Or just W_in type injection.
                # Let's simple use "Input + Target" as the clamped state inputs.
                # But we don't have a backward feedback matrix.
                # Let's use the Transpose of Readout Weights as Feedback Weights (Feedback Alignment).
                feedback_input = torch.mv(self.readout_weights.T, target_vec) # [N]
                
                # Settle with Input + Feedback
                # total_input = input_vec + 0.5 * feedback_input
                # self.engine.settle(total_input, duration_steps=5)
                # state_clamped = self.engine.state.clone()
                
                # EqProp Update: Delta W = (state_clamped * state_clamped.T) - (state_free * state_free.T)
                # This needs efficient sparse update.
                pass 
            
        # Compute Averaged Gradients/Deltas
        delta_readout_w = -readout_lr * grad_readout_w
        delta_readout_b = -readout_lr * grad_readout_b
        
        # Internal weights
        delta_w = np.zeros_like(start_weight_values.cpu().numpy())
        delta_b = np.zeros_like(start_biases.cpu().numpy())
        
        return delta_w, delta_b, delta_readout_w.cpu().numpy(), delta_readout_b.cpu().numpy(), loss_accum, correct_count

def generate_text_readout(engine, input_w, readout_w, readout_b, length=100, start_text="A"):
    current_text = start_text
    # Prime
    for char in start_text:
        input_idx = int(ord(char))
        if input_idx < 256:
             input_vec = input_w[:, input_idx]
             engine.settle(input_vec, duration_steps=10)
             
    print(f"Prompt: '{start_text}'")
    
    for _ in range(length):
        state = engine.state.clone()
        logits = torch.mv(readout_w, state) + readout_b
        probs = torch.softmax(logits, dim=0)
        
        next_byte_val = torch.multinomial(probs, 1).item()
        
        next_char = chr(next_byte_val) if 0 <= next_byte_val < 128 else '?'
        current_text += next_char
        
        input_vec = input_w[:, next_byte_val]
        engine.settle(input_vec, duration_steps=10)
        
    return current_text

def run_phase(phase_name, data_path, num_workers, steps_per_iter, iterations, master_graph_refs, architecture_refs, global_weights):
    print(f"\n=== Starting Phase: {phase_name} ===")
    print(f"Loading data: {data_path}")
    
    # Load Data
    if not os.path.exists(data_path):
        print(f"Data {data_path} not found. Generating...")
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        if "level1" in data_path: generate_chars(data_path)
        elif "level2" in data_path: generate_words(data_path)
        elif "level3" in data_path: generate_quotes(data_path)
        else:
             print("Unknown data type needed but missing.")
    
    with open(data_path, 'rb') as f:
        all_data_bytes = f.read()
    data_ref = ray.put(all_data_bytes)
    total_data_size = len(all_data_bytes)
    
    # Spawn Workers
    workers = []
    chunk_size = total_data_size // num_workers
    
    indices_ref, values_ref, biases_ref, taus_ref = master_graph_refs
    input_w_ref, readout_w_ref, readout_b_ref = architecture_refs # Note: readout weights in refs are INITIAL.
    # We must start workers with CURRENT global weights.
    
    # Worker Spawning
    # We pass initial refs, but immediately update them.
    for i in range(num_workers):
        offset = i * chunk_size
        worker = DragonWorker.remote(5000, indices_ref, values_ref, biases_ref, taus_ref, input_w_ref, readout_w_ref, offset, chunk_size, data_ref)
        workers.append(worker)
        
    # Set current trained weights
    w_vals, b_vals, rw_vals, rb_vals = global_weights
    w_ref = ray.put(w_vals)
    b_ref = ray.put(b_vals)
    rw_ref = ray.put(rw_vals)
    rb_ref = ray.put(rb_vals)
    ray.get([w.set_weights.remote(w_ref, b_ref, rw_ref, rb_ref) for w in workers])
    
    # Training Loop
    # Instantiate local engine for generation (using first worker's logic effectively)
    # We need indices locally
    indices = ray.get(indices_ref)
    taus = ray.get(taus_ref)
    local_engine = DragonEngineTorch(5000, indices, w_vals, b_vals, taus, dt=0.01, device='cpu')
    input_weights = ray.get(input_w_ref)
    local_input_w = torch.tensor(input_weights, dtype=torch.float32)

    for i in range(iterations):
        futures = [w.train_step.remote(steps=steps_per_iter, enable_plasticity=False) for w in workers] # Plasticity still disabled for now to isolate Curriculum effect
        results = ray.get(futures)
        
        avg_delta_w = np.mean([r[0] for r in results], axis=0) # Zeros if disabled
        avg_delta_b = np.mean([r[1] for r in results], axis=0)
        avg_delta_rw = np.mean([r[2] for r in results], axis=0)
        avg_delta_rb = np.mean([r[3] for r in results], axis=0)
        total_loss = sum([r[4] for r in results])
        total_correct = sum([r[5] for r in results])
        
        # Update Globals
        w_vals += avg_delta_w
        b_vals += avg_delta_b
        rw_vals += avg_delta_rw
        rb_vals += avg_delta_rb
        
        # Broadcast
        # Optimize: Only broadcast readout if internal frozen
        rw_ref = ray.put(rw_vals)
        rb_ref = ray.put(rb_vals)
        ray.get([w.set_weights.remote(ray.put(w_vals), ray.put(b_vals), rw_ref, rb_ref) for w in workers])
        
        # Stats
        total_predictions = num_workers * steps_per_iter
        accuracy = total_correct / total_predictions
        if i % 10 == 0:
            print(f"Iter {i} Loss: {total_loss:.2f} | Acc: {accuracy:.2%}")
            
    # Generation Check
    local_engine.weight_values = torch.tensor(w_vals, dtype=torch.float32)
    local_engine.biases = torch.tensor(b_vals, dtype=torch.float32)
    local_rw = torch.tensor(rw_vals, dtype=torch.float32)
    local_rb = torch.tensor(rb_vals, dtype=torch.float32)
    
    prompt = "A" if phase_name == "Chars" else "The "
    print(f"--- End of {phase_name} Generation ---")
    print(generate_text_readout(local_engine, local_input_w, local_rw, local_rb, 100, prompt))
    print("-------------------------------------")
    
    return [w_vals, b_vals, rw_vals, rb_vals]

def main():
    print("=== Curriculum Training ===")
    ray.init(ignore_reinit_error=True)
    
    # 1. Init Graph & Weights
    num_nodes = 10000
    master_graph = DynamicGraph(num_nodes=num_nodes, m_edges=10, p_triad=0.1, seed=42)
    indices, values = master_graph.export_sparse_components()
    biases = master_graph.biases
    taus = master_graph.taus
    
    # Spectral Radius & Taus (Apply Refinements)
    sparse_weights = sp.coo_matrix((values, (indices[0], indices[1])), shape=(num_nodes, num_nodes))
    eigvals = eigs(sparse_weights, k=1, which='LM', return_eigenvectors=False)
    values = values * (1.1 / (np.abs(eigvals[0]) + 1e-8))
    
    taus = np.ones(num_nodes) * 0.2 # Medium
    taus[:int(0.1*num_nodes)] = 0.02 # Fast (first 10% roughly)
    taus[-int(0.1*num_nodes):] = 2.0 # Slow
    
    # Architecture
    input_weights = np.random.randn(num_nodes, 256).astype(np.float32) * 1.0
    readout_weights = np.random.randn(256, num_nodes).astype(np.float32) * 0.01
    readout_bias = np.zeros(256, dtype=np.float32)
    
    # Refs
    indices_ref = ray.put(indices)
    values_ref = ray.put(values)
    biases_ref = ray.put(biases)
    taus_ref = ray.put(taus)
    master_refs = (indices_ref, values_ref, biases_ref, taus_ref)
    
    input_w_ref = ray.put(input_weights)
    readout_w_ref = ray.put(readout_weights)
    readout_b_ref = ray.put(readout_bias)
    arch_refs = (input_w_ref, readout_w_ref, readout_b_ref)
    
    # Current Global Weights Container
    global_weights = [values, biases, readout_weights, readout_bias]
    
    # 2. Run Phases
    # Phase 1: Advanced Chars (2x iters: 100)
    global_weights = run_phase("Chars", "ndcd/data/level1_chars.txt", 48, 100, 100, master_refs, arch_refs, global_weights)
    
    # Phase 2: Words (4x iters: 200)
    global_weights = run_phase("Words", "ndcd/data/level2_words.txt", 48, 100, 200, master_refs, arch_refs, global_weights)
    
    # Phase 3: Complex Quotes (4x iters: 400)
    global_weights = run_phase("Quotes", "ndcd/data/level3_quotes.txt", 48, 100, 400, master_refs, arch_refs, global_weights)
    
    # Phase 4: Sherlock (Scaled to 400)
    if os.path.exists("ndcd/data/sherlock.txt"):
        global_weights = run_phase("Literature", "ndcd/data/sherlock.txt", 48, 200, 400, master_refs, arch_refs, global_weights)
        
    ray.shutdown()

if __name__ == "__main__":
    main()
