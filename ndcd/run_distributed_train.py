
import ray
import torch
import numpy as np
import os
import os
import time
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from ndcd.graph import DynamicGraph
from ndcd.engine_torch import DragonEngineTorch
from ndcd.sensory import ByteSensoryInterface

@ray.remote(num_cpus=1)
class DragonWorker:
    def __init__(self, num_nodes, indices, values, biases, taus, input_weights, readout_weights, data_offset, data_len_chunk, full_data_bytes):
        """
        Worker actor that holds a copy of the Dragon and processes a shard of data.
        """
        # Determine device - in a real cluster, we might check for CUDA
        self.device = 'cpu' # Default to CPU for distribution test unless GPUs explicitly requested
        if torch.cuda.is_available():
            self.device = 'cuda'

        # Initialize Engine with Sparse Components
        self.engine = DragonEngineTorch(num_nodes, indices, values, biases, taus, dt=0.01, device=self.device)
        
        # Architecture Components
        # Input: [N, 256] - Projects one-hot byte to node currents
        self.input_weights = torch.tensor(input_weights, dtype=torch.float32, device=self.device)
        
        # Readout: [256, N] - Projects node state to output logits
        # We allow this to be updated/set from master
        self.readout_weights = torch.tensor(readout_weights, dtype=torch.float32, device=self.device)
        self.readout_bias = torch.zeros(256, dtype=torch.float32, device=self.device)
        
        self.io = ByteSensoryInterface(input_offset=0, output_offset=256, device=self.device)
        
        # Slice the data
        self.data_bytes = full_data_bytes[data_offset : data_offset + data_len_chunk]
        self.data_len = len(self.data_bytes)
        self.current_idx = 0
        
    def get_weights(self):
        """Returns weights for synchronization."""
        # We now return internal weights AND readout weights
        # But for now, let's assume internal weights are changing too (EqProp)
        return (self.engine.weight_values.cpu().numpy(), 
                self.engine.biases.cpu().numpy(),
                self.readout_weights.cpu().numpy(),
                self.readout_bias.cpu().numpy())
        
    def set_weights(self, weight_values, biases, readout_weights, readout_bias):
        """Updates internal weight values from global master."""
        self.engine.weight_values = torch.tensor(weight_values, device=self.device, dtype=torch.float32)
        self.engine.biases = torch.tensor(biases, device=self.device, dtype=torch.float32)
        self.readout_weights = torch.tensor(readout_weights, device=self.device, dtype=torch.float32)
        self.readout_bias = torch.tensor(readout_bias, device=self.device, dtype=torch.float32)

    def train_step(self, steps=10):
        """
        Runs 'steps' of training with Readout Layer.
        """
        start_weight_values = self.engine.weight_values.clone()
        start_biases = self.engine.biases.clone()
        
        # Gradients for Readout
        grad_readout_w = torch.zeros_like(self.readout_weights)
        grad_readout_b = torch.zeros_like(self.readout_bias)
        
        loss_accum = 0
        correct_count = 0
        readout_lr = 0.01
        
        for _ in range(steps):
            if self.current_idx >= self.data_len - 1:
                self.current_idx = 0
                
            input_byte = self.data_bytes[self.current_idx]
            target_byte = self.data_bytes[self.current_idx + 1]
            self.current_idx += 1
            
            # 1. Input Projection
            # One-hot input: we just pick the column from input_weights
            input_idx = int(input_byte)
            input_vec = self.input_weights[:, input_idx]  # [N]
            
            # 2. Settle (Free Phase)
            # We assume EqProp internal learning is secondary for now, 
            # or we can keep it. Let's keep it but with low LR.
            self.engine.settle(input_vec, duration_steps=10)
            state_free = self.engine.state.clone() # [N]
            
            # 3. Readout Prediction
            # logits = W_out @ state + b
            logits = torch.mv(self.readout_weights, state_free) + self.readout_bias # [256]
            
            # 4. Compute Loss (Cross Entropy)
            # Softmax
            probs = torch.softmax(logits, dim=0)
            target_idx = int(target_byte)
            
            # Neg Log Likelihood of target
            curr_loss = -torch.log(probs[target_idx] + 1e-8)
            loss_accum += curr_loss.item()
            
            # Accuracy Check
            predicted_idx = torch.argmax(probs).item()
            if predicted_idx == target_idx:
                correct_count += 1
            
            # 5. Backprop for Readout (Manual or Autograd)
            # dL/dLogits = probs - one_hot(target)
            d_logits = probs.clone()
            d_logits[target_idx] -= 1.0
            
            # d_readout_w = d_logits outer state
            # d_readout_b = d_logits
            
            # Accumulate gradients
            grad_readout_w += torch.outer(d_logits, state_free)
            grad_readout_b += d_logits
            
            # 6. EqProp (Internal Weights) - Disabled
            
        # Compute Averaged Gradients/Deltas
        delta_readout_w = -readout_lr * grad_readout_w
        delta_readout_b = -readout_lr * grad_readout_b
        
        # Internal weights didn't change (frozen)
        delta_w = np.zeros_like(start_weight_values.cpu().numpy())
        delta_b = np.zeros_like(start_biases.cpu().numpy())
        
        # Re-calculate correct count for return
        # Actually efficient way is to count in loop. 
        # I'll edit the loop above to count correct.
        
        return delta_w, delta_b, delta_readout_w.cpu().numpy(), delta_readout_b.cpu().numpy(), loss_accum, correct_count

def main():
    print("=== Initializing Distributed Dragon Training (Ray) ===")
    
    # Init Ray - will connect to local cluster or start one
    ray.init(ignore_reinit_error=True)
    
    # Configuration
    num_nodes = 5000 
    m_edges = 10
    p_triad = 0.1
    iterations = 500
    steps_per_iter = 200
    num_workers = 48
    
    print("=== Configuration ===")
    print(f"Nodes: {num_nodes}")
    print(f"Edges/Node (m): {m_edges}")
    print(f"Triad Prob: {p_triad}")
    print(f"Workers: {num_workers}")
    print(f"Iterations: {iterations}")
    print(f"Steps/Iter: {steps_per_iter}")
    print("=====================")

    seed = 42
    data_path = 'ndcd/data/sherlock.txt'
    
    if not os.path.exists(data_path):
        print("Data not found locally. Downloading Sherlock Holmes...")
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        import urllib.request
        # Project Gutenberg URL for The Adventures of Sherlock Holmes
        url = "https://www.gutenberg.org/files/1661/1661-0.txt" 
        try:
             urllib.request.urlretrieve(url, data_path)
             print("Data downloaded.")
        except Exception as e:
            print(f"Failed to download from primary url: {e}")
            # Fallback or alternative
            url = "https://raw.githubusercontent.com/kylerrussell/Living-Brain/main/ndcd/data/sherlock.txt" # Placeholder if we had one, or try another gutenberg mirror
            # let's try a reliable usually available mirror or just fail with message
            print("Please ensure internet access or provide 'ndcd/data/sherlock.txt' manually.")
            raise e
        
    total_data_size = os.path.getsize(data_path)

    # Read all data into memory on driver
    with open(data_path, 'rb') as f:
        all_data_bytes = f.read()

    # Put data into Plasma Object Store
    data_ref = ray.put(all_data_bytes)
    
    # Master Weights source (initialized locally first)
    # Generate graph ONCE on driver
    print("Generating Master Graph...")
    master_graph = DynamicGraph(num_nodes=num_nodes, m_edges=m_edges, p_triad=p_triad, seed=seed)
    
    # Export Sparse Components
    indices, values = master_graph.export_sparse_components()
    biases = master_graph.biases
    taus = master_graph.taus
    # --- Architectural Refinement: Spectral Radius Tuning ---
    print("Tuning Spectral Radius (Criticality)...")
    # 1. Construct sparse matrix for eigenvalue calculation
    num_edges = len(values)
    # COO format
    sparse_weights = sp.coo_matrix((values, (indices[0], indices[1])), shape=(num_nodes, num_nodes))
    
    # 2. Calculate largest magnitude eigenvalue
    # k=1, which='LM' (Largest Magnitude)
    # This can be slow, but for 5000 nodes it's okay (seconds).
    try:
        # Use simple 'LR' (Largest Real) part if complex (assuming recurrent, it is asymmetric usually, but here we made it symmetric? 
        # graph.py makes it symmetric. So eigenvalues are real.
        eigvals = eigs(sparse_weights, k=1, which='LM', return_eigenvectors=False)
        max_eig = np.abs(eigvals[0])
        print(f"Current Spectral Radius: {max_eig:.4f}")
        
        # 3. Scale to target
        target_radius = 1.1 # Edge of Chaos
        scale_factor = target_radius / (max_eig + 1e-8)
        values = values * scale_factor
        
        print(f"Scaled Spectral Radius to: {target_radius}")
    except Exception as e:
        print(f"Spectral Radius tuning failed: {e}. Using default weights.")

    # --- Architectural Refinement: Hierarchical Taus ---
    print("Setting Hierarchical Time Constants...")
    # Hierarchy: Fast (Sensory), Medium (Word), Slow (Context)
    # dt = 0.01
    # Fast: tau=0.02 (decay ~ 2 steps) - 10%
    # Medium: tau=0.2 (decay ~ 20 steps) - 80%
    # Slow: tau=2.0 (decay ~ 200 steps) - 10%
    
    taus = np.ones(num_nodes, dtype=np.float32)
    
    # Indices
    n_fast = int(0.1 * num_nodes)
    n_slow = int(0.1 * num_nodes)
    n_med = num_nodes - n_fast - n_slow
    
    perm = np.random.permutation(num_nodes)
    idx_fast = perm[:n_fast]
    idx_slow = perm[n_fast:n_fast+n_slow]
    idx_med = perm[n_fast+n_slow:]
    
    taus[idx_fast] = 0.02
    taus[idx_med] = 0.2
    taus[idx_slow] = 2.0
    
    # Put into Ray Store (Zero-Copy)
    indices_ref = ray.put(indices)
    values_ref = ray.put(values) # This is the master copy of weights
    biases_ref = ray.put(biases)
    taus_ref = ray.put(taus)

    
    # Generate Architectures Matrices
    print("Generating Readout/Input Matrices...")
    # Input: fixed random projection. 256 inputs -> N nodes
    # We make it sparse: each input connects to ~10% of nodes? Or dense?
    # Dense is fine for 256x5000 (1.2M floats = 5MB).
    input_weights = np.random.randn(num_nodes, 256).astype(np.float32) * 1.0
    
    # Readout: N nodes -> 256 outputs.
    # Initialize near zero
    readout_weights = np.random.randn(256, num_nodes).astype(np.float32) * 0.01
    readout_bias = np.zeros(256, dtype=np.float32)
    
    input_w_ref = ray.put(input_weights)
    readout_w_ref = ray.put(readout_weights)
    
    
    # Create Workers
    print(f"Spawning {num_workers} Workers...")
    
    workers = []
    chunk_size = total_data_size // num_workers
    
    for i in range(num_workers):
        offset = i * chunk_size
        # Pass the object references. Ray resolves them in the worker.
        worker = DragonWorker.remote(num_nodes, indices_ref, values_ref, biases_ref, taus_ref, input_w_ref, readout_w_ref, offset, chunk_size, data_ref)
        workers.append(worker)
    
    # Broadcast initial (ensure everyone starts identical)
    # NOTE: Workers already initialized with master weights via __init__ refs.
    # explicit broadcast not strictly needed for init, but good for reset.
    # We skip it here since we just spawned them with correct weights.
    
    # Global containers for aggregation (Driver side)
    # Note: Driver keeps dense weights for simple averaging if needed, 
    # OR we can keep values only.
    # Let's keep `values` (1D array) as the master weights.
    global_weight_values = values.copy()
    global_biases = biases.copy()
    global_readout_w = readout_weights.copy()
    global_readout_b = readout_bias.copy()
    
    # Training Loop
    # Instantiate local engine on driver for periodic generation
    # local_engine also needs input/readout
    local_engine = DragonEngineTorch(num_nodes, indices, global_weight_values, global_biases, taus, dt=0.01, device='cpu')
    local_input_w = torch.tensor(input_weights, dtype=torch.float32)
    local_readout_w = torch.tensor(global_readout_w, dtype=torch.float32)
    local_readout_b = torch.tensor(global_readout_b, dtype=torch.float32)

    print(f"Starting {iterations} sync iterations...")
    
    start_time = time.time()
    
    start_time = time.time()
    
    for i in range(iterations):
        # 1. Trigger training on all workers
        # They run for N steps (e.g., 50 bytes)
        futures = [w.train_step.remote(steps=steps_per_iter) for w in workers]
        
        # 2. Collect results (Barrier)
        results = ray.get(futures)
        
        # 3. Aggregate gradients
        # Optimization: In real huge scale, use DistributedOptimizer or AllReduce
        # Here: Parameter Server style averaging
        avg_delta_w = np.mean([r[0] for r in results], axis=0)
        avg_delta_b = np.mean([r[1] for r in results], axis=0)
        avg_delta_rw = np.mean([r[2] for r in results], axis=0)
        avg_delta_rb = np.mean([r[3] for r in results], axis=0)
        total_loss = sum([r[4] for r in results])
        total_correct = sum([r[5] for r in results])
        
        # 4. Apply Update
        # Results are delta_w_values (sparse structure assumed identical)
        global_weight_values += avg_delta_w
        global_biases += avg_delta_b
        global_readout_w += avg_delta_rw
        global_readout_b += avg_delta_rb
        
        # 5. Broadcast new weights
        # We broadcast VALUES only (indices are static)
        w_vals_ref = ray.put(global_weight_values)
        b_ref = ray.put(global_biases)
        rw_ref = ray.put(global_readout_w)
        rb_ref = ray.put(global_readout_b)
        
        ray.get([w.set_weights.remote(w_vals_ref, b_ref, rw_ref, rb_ref) for w in workers])
        
        # Calculate Accuracy
        total_predictions = num_workers * steps_per_iter
        accuracy = total_correct / total_predictions
        

        
        print(f"Iter {i} Complete. Loss: {total_loss:.2f} | Acc: {accuracy:.2%}")
        
        # Periodic Generation
        if i % 20 == 0:
            print(f"\n--- Generation Iter {i} ---")
            # Update local engine
            local_engine.weight_values = torch.tensor(global_weight_values, dtype=torch.float32)
            local_engine.biases = torch.tensor(global_biases, dtype=torch.float32)
            local_readout_w = torch.tensor(global_readout_w, dtype=torch.float32)
            local_readout_b = torch.tensor(global_readout_b, dtype=torch.float32)
            
            gen_text = generate_text_readout(local_engine, local_input_w, local_readout_w, local_readout_b, length=100, start_text="Sherlock:")
            print(f"{gen_text}\n-----------------------")
        

        
    print(f"Distributed Training Done. Time: {time.time() - start_time:.2f}s")

    # --- Generation Phase ---
    print("\n=== Generating Text from Global Weights ===")
    
    # Instantiate local engine on driver
    local_engine = DragonEngineTorch(num_nodes, indices, global_weight_values, global_biases, taus, dt=0.01, device='cpu')
    # No IO needed, we have matrices
    
    local_input_w = torch.tensor(input_weights, dtype=torch.float32)
    local_readout_w = torch.tensor(global_readout_w, dtype=torch.float32)
    local_readout_b = torch.tensor(global_readout_b, dtype=torch.float32)

    generated_text = generate_text_readout(local_engine, local_input_w, local_readout_w, local_readout_b, length=200, start_text="Sherlock:")
    print(f"\nGenerated Output:\n{generated_text}\n")
    print("===========================================")

    ray.shutdown()

def generate_text_readout(engine, input_w, readout_w, readout_b, length=100, start_text="A"):
    """
    Generates text using the trained Readout Layer.
    """
    current_text = start_text
    
    # 1. Prime state
    last_char = start_text[-1]
    input_idx = int(ord(last_char))
    input_vec = input_w[:, input_idx]
    
    engine.settle(input_vec, duration_steps=20)
    
    print(f"Prompt: '{start_text}'")
    
    for _ in range(length):
        # 1. Predict
        state = engine.state.clone()
        logits = torch.mv(readout_w, state) + readout_b
        probs = torch.softmax(logits, dim=0)
        
        # Sample or Argmax
        # next_byte_val = torch.argmax(probs).item()
        # Sampling is more interesting for generation
        next_byte_val = torch.multinomial(probs, 1).item()
        
        next_char = chr(next_byte_val) if 0 <= next_byte_val < 128 else '?'
        current_text += next_char
        
        # 2. Feedback (Input next char)
        input_vec = input_w[:, next_byte_val]
        engine.settle(input_vec, duration_steps=10) # Shorter settle for stream
        
    return current_text

if __name__ == "__main__":
    main()
