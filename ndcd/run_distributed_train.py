
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
    def __init__(self, num_nodes, seed, data_offset, data_len, data_path):
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
        
        # Load Dataset and seek to offset
        with open(data_path, 'rb') as f:
            f.seek(data_offset)
            self.data_bytes = f.read(data_len)
            
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
    
    num_nodes = 1000 # Smaller for distribution test speed
    seed = 42
    data_path = 'ndcd/data/input.txt'
    
    if not os.path.exists(data_path):
        print("Data not found!")
        return
        
    total_data_size = os.path.getsize(data_path)
    
    # Create Workers
    num_workers = 2 # Simulation of using 2 nodes/CPUs
    print(f"Spawning {num_workers} Workers...")
    
    workers = []
    chunk_size = total_data_size // num_workers
    
    for i in range(num_workers):
        offset = i * chunk_size
        worker = DragonWorker.remote(num_nodes, seed, offset, chunk_size, data_path)
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
    iterations = 5 
    print(f"Starting {iterations} sync iterations...")
    
    start_time = time.time()
    
    for i in range(iterations):
        # 1. Trigger training on all workers
        # They run for N steps (e.g., 50 bytes)
        futures = [w.train_step.remote(steps=50) for w in workers]
        
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
    ray.shutdown()

if __name__ == "__main__":
    main()
