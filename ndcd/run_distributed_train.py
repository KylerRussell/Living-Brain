
import ray
import torch
import numpy as np
import os
import time
from ndcd.graph import DynamicGraph
from ndcd.engine_torch import DragonEngineTorch
from ndcd.sensory import ByteSensoryInterface

@ray.remote(num_cpus=1)
class DragonWorker:
    def __init__(self, num_nodes, seed, data_offset, data_len_chunk, full_data_bytes):
        """
        Worker actor that holds a copy of the Dragon and processes a shard of data.
        """
        # Re-initialize graph with same seed to ensure identical topology
        self.graph = DynamicGraph(num_nodes=num_nodes, m_edges=5, p_triad=0.1, seed=seed)
        
        # Determine device - in a real cluster, we might check for CUDA
        self.device = 'cpu' # Default to CPU for distribution test unless GPUs explicitly requested
        if torch.cuda.is_available():
            # Basic logic: if GPU available, try to use it. 
            # In Ray, better to use ray.get_gpu_ids(), but keeping it simple.
            self.device = 'cuda'

        self.engine = DragonEngineTorch(self.graph, dt=0.01, device=self.device)
        self.io = ByteSensoryInterface(device=self.device)
        
        # Slice the data from the shared memory object (or copy passed)
        self.data_bytes = full_data_bytes[data_offset : data_offset + data_len_chunk]
            
        self.data_len = len(self.data_bytes)
        self.current_idx = 0
        
    def get_weights(self):
        """Returns weights for synchronization."""
        return self.engine.weights.cpu().numpy(), self.engine.biases.cpu().numpy()
        
    def set_weights(self, weights, biases):
        """Updates internal weights from global master."""
        self.engine.weights = torch.tensor(weights, device=self.device, dtype=torch.float32)
        self.engine.biases = torch.tensor(biases, device=self.device, dtype=torch.float32)

    def train_step(self, steps=10):
        """
        Runs 'steps' of training. Returns the accumulated weight update (delta) 
        OR simply lets the trainer fetch weights later. 
        Better strategy for continuous EqProp: 
        Run X steps, return gradient approximation or just new weights.
        
        To simplify averaging:
        1. Worker receives Global Weights W_global.
        2. Worker does N updates. W_local = W_global + Delta.
        3. Worker returns Delta = W_local - W_global.
        """
        start_weights = self.engine.weights.clone()
        start_biases = self.engine.biases.clone()
        
        loss = 0
        
        for _ in range(steps):
            if self.current_idx >= self.data_len - 1:
                self.current_idx = 0
                
            input_byte = self.data_bytes[self.current_idx]
            target_byte = self.data_bytes[self.current_idx + 1]
            self.current_idx += 1
            
            # Encode
            input_vec = self.io.encode(input_byte, self.engine.num_nodes)
            
            # Free Phase
            self.engine.settle(input_vec, duration_steps=20)
            state_free = self.engine.state.clone()
            
            # Measurement
            if self.io.decode(state_free) != target_byte:
                loss += 1
            
            # Nudged Phase
            target_nudge = torch.zeros(self.engine.num_nodes, device=self.device)
            idx = int(target_byte) % 256 + 256
            target_nudge[idx] = 1.0
            
            self.engine.settle(input_vec, duration_steps=10, nudge_target=target_nudge, beta=0.5)
            state_nudged = self.engine.state.clone()
            
            # Update
            self.engine.update_weights_eq_prop(state_free, state_nudged, beta=0.5, learning_rate=0.01)

        # Compute Delta
        delta_w = (self.engine.weights - start_weights).cpu().numpy()
        delta_b = (self.engine.biases - start_biases).cpu().numpy()
        
        return delta_w, delta_b, loss

def main():
    print("=== Initializing Distributed Dragon Training (Ray) ===")
    
    # Init Ray - will connect to local cluster or start one
    ray.init(ignore_reinit_error=True)
    
    num_nodes = 2000 # Smaller for distribution test speed
    seed = 42
    data_path = 'ndcd/data/input.txt'
    
    if not os.path.exists(data_path):
        print("Data not found locally. Downloading...")
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, data_path)
        print("Data downloaded.")
        
    total_data_size = os.path.getsize(data_path)

    # Read all data into memory on driver
    with open(data_path, 'rb') as f:
        all_data_bytes = f.read()

    # Put data into Plasma Object Store
    data_ref = ray.put(all_data_bytes)
    
    # Create Workers
    num_workers = 48 # Safe margin below 54 threads (allowing for Head + System overhead)
    print(f"Spawning {num_workers} Workers...")
    
    workers = []
    chunk_size = total_data_size // num_workers
    
    for i in range(num_workers):
        offset = i * chunk_size
        # Pass the object reference `data_ref`. Ray resolves this to the actual bytes in the worker.
        worker = DragonWorker.remote(num_nodes, seed, offset, chunk_size, data_ref)
        workers.append(worker)
    
    # Master Weights source (initialized locally first)
    # We create a dummy graph just to get initial weights
    tmp_graph = DynamicGraph(num_nodes=num_nodes, m_edges=5, p_triad=0.1, seed=seed)
    global_weights = tmp_graph.weights
    global_biases = tmp_graph.biases
    
    # Broadcast initial (ensure everyone starts identical)
    # Using ray.put for large object efficiency
    w_ref = ray.put(global_weights)
    b_ref = ray.put(global_biases)
    
    print("Broadcasting initial weights...")
    ray.get([w.set_weights.remote(w_ref, b_ref) for w in workers])
    
    # Training Loop
    iterations = 10 
    print(f"Starting {iterations} sync iterations...")
    
    start_time = time.time()
    
    for i in range(iterations):
        # 1. Trigger training on all workers
        # They run for N steps (e.g., 50 bytes)
        futures = [w.train_step.remote(steps=500) for w in workers]
        
        # 2. Collect results (Barrier)
        results = ray.get(futures)
        
        # 3. Aggregate gradients
        # Optimization: In real huge scale, use DistributedOptimizer or AllReduce
        # Here: Parameter Server style averaging
        avg_delta_w = np.mean([r[0] for r in results], axis=0)
        avg_delta_b = np.mean([r[1] for r in results], axis=0)
        total_loss = sum([r[2] for r in results])
        
        # 4. Apply Update
        global_weights += avg_delta_w
        global_biases += avg_delta_b
        
        # 5. Broadcast new weights
        w_ref = ray.put(global_weights)
        b_ref = ray.put(global_biases)
        ray.get([w.set_weights.remote(w_ref, b_ref) for w in workers])
        
        print(f"Iter {i} Complete. Total Loss (Batch): {total_loss}")
        
    print(f"Distributed Training Done. Time: {time.time() - start_time:.2f}s")

    # --- Generation Phase ---
    print("\n=== Generating Text from Global Weights ===")
    
    # Instantiate local engine on driver
    # Note: In a real large-scale setting, we'd use a dedicated inference actor or service.
    local_graph = DynamicGraph(num_nodes=num_nodes, m_edges=5, p_triad=0.1, seed=seed)
    local_engine = DragonEngineTorch(local_graph, dt=0.01, device='cpu')
    local_engine.weights = torch.tensor(global_weights, dtype=torch.float32)
    local_engine.biases = torch.tensor(global_biases, dtype=torch.float32)
    local_io = ByteSensoryInterface(device='cpu')

    generated_text = generate_text(local_engine, local_io, length=200, start_text="The")
    print(f"\nGenerated Output:\n{generated_text}\n")
    print("===========================================")

    ray.shutdown()

def generate_text(engine, io_interface, length=100, start_text="A"):
    """
    Generates text using the trained engine "free dreaming".
    """
    current_text = start_text
    # Seed the state with start_text
    # For simplicity, we just run the last char to set state, 
    # but ideally we'd sequence them all.
    
    last_char = start_text[-1]
    input_vec = io_interface.encode(ord(last_char), engine.num_nodes)
    
    # Prime the engine
    engine.settle(input_vec, duration_steps=20)
    
    print(f"Prompt: '{start_text}'")
    
    for _ in range(length):
        # 1. Predict next state (Free phase only)
        # Input is the PREVIOUS output (Autoregressive)
        # We re-encode the last char effectively
        
        # Settle to find next attractor
        engine.settle(input_vec, duration_steps=20)
        state = engine.state.clone()
        
        # 2. Decode
        next_byte_val = io_interface.decode(state)
        next_char = chr(next_byte_val) if 0 <= next_byte_val < 128 else '?'
        
        current_text += next_char
        
        # 3. Feedback loop
        # The new input is what we just hallucinated
        input_vec = io_interface.encode(next_byte_val, engine.num_nodes)
        
    return current_text

if __name__ == "__main__":
    main()
