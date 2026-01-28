
import numpy as np
import torch
import sys
import time
import os

from graph import DynamicGraph
from engine_torch import DragonEngineTorch
from sensory import ByteSensoryInterface

def main():
    print("=== Initializing Native Scale-Up Dragon (PyTorch) ===")
    
    # 1. Setup
    # GPT-2 small has ~117M parameters.
    # N^2 = 117 * 10^6 => N ~ 10,800.
    num_nodes = 10800 # Manually requested by user for GPT-2 scale equivalence
    
    print(f"Creating Brain with {num_nodes} neurons...")
    # Seed for reproducibility
    graph = DynamicGraph(num_nodes=num_nodes, m_edges=5, p_triad=0.1, seed=42)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    engine = DragonEngineTorch(graph, dt=0.01, device=device)
    io_interface = ByteSensoryInterface(device=device)
    
    # 2. Data
    data_path = 'ndcd/data/input.txt'
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return

    print(f"Loading dataset from {data_path}...")
    with open(data_path, 'rb') as f:
        bytes_data = f.read()
            
    data_len = len(bytes_data)
    print(f"Training Data Size: {data_len} bytes")
    
    # 3. Training Loop (Next-Byte Prediction)
    epochs = 1 # One epoch is plenty for 1MB data given the speed
    learning_rate = 0.01
    beta = 0.5 
    checkpoint_interval = 100
    save_path = 'ndcd/model_checkpoint.pt'
    
    start_time = time.time()
    total_loss = 0
    
    print("Starting Main Training Loop...")
    
    for epoch in range(epochs):
        for t in range(data_len - 1):
            # Input: Current Byte
            input_byte = bytes_data[t]
            target_byte = bytes_data[t+1]
            
            # Encode
            input_vec = io_interface.encode(input_byte, num_nodes)
            
            # Phase 1: Free (Predicting next state)
            engine.settle(input_vec, duration_steps=20)
            state_free = engine.state.clone()
            
            # Measure prediction
            predicted_byte = io_interface.decode(state_free)
            if predicted_byte != target_byte:
                total_loss += 1
                
            # Phase 2: Nudged (Teach it to be the target)
            target_nudge_vector = torch.zeros(num_nodes, device=device)
            # Offset 256 for Output nodes
            target_idx = int(target_byte) % 256 + 256 
            target_nudge_vector[target_idx] = 1.0
            
            engine.settle(input_vec, duration_steps=10, nudge_target=target_nudge_vector, beta=beta)
            state_nudged = engine.state.clone()
            
            # Update Weights
            engine.update_weights_eq_prop(state_free, state_nudged, beta, learning_rate)
            
            if t % 10 == 0:
                print(f"Propagating step {t}/{data_len} | Loss: {total_loss}")
                
            # Checkpoint
            if t % checkpoint_interval == 0:
                print(f"Saving checkpoint to {save_path}...")
                torch.save({
                    'weights': engine.weights,
                    'biases': engine.biases,
                    'step': t,
                    'loss': total_loss
                }, save_path)
                
    elapsed = time.time() - start_time
    print(f"Training Complete. Time: {elapsed:.2f}s")

if __name__ == "__main__":
    main()
