
import torch
import numpy as np
import sys
import os

# Ensure we can import from local modules
sys.path.append(os.getcwd())

from ndcd.graph import DynamicGraph
from ndcd.engine_torch import DragonEngineTorch
from ndcd.sensory import ByteSensoryInterface

def main():
    print("=== Debugging Dragon Training Dynamics ===")
    
    # Setup - Small scale for debugging
    num_nodes = 600 # Need 512+ for I/O separation
    seed = 42
    
    # 1. Identity / Sequence task
    # If it repeats "A", it learns Identity (Input A -> Output A).
    # We want Input A -> Output B.
    data = "ABC" * 20 
    data_bytes = data.encode('utf-8')
    
    print(f"Task: Learn sequence '{data}'")
    
    # Init Brain
    graph = DynamicGraph(num_nodes=num_nodes, m_edges=5, p_triad=0.1, seed=seed)
    # Using 'cpu' for debug simplicity and visibility provided numpy conversions
    device = 'cpu' 
    engine = DragonEngineTorch(graph, dt=0.01, device=device)
    # Separate Input (0-255) and Output (256-511)
    # Ensure num_nodes is enough! We set 500. 256+256 = 512.
    # Oops, 500 is too small for full separation. Let's bump num_nodes to 600.
    io = ByteSensoryInterface(input_offset=0, output_offset=256, device=device)
    
    # Parameters to tune
    beta = 0.5          # Nudging strength (0.0 = no nudge, 1.0 = hard clamp)
    learning_rate = 0.01 # Step size
    free_steps = 30     # How long to think before reading output
    nudge_steps = 20    # How long to dream the target
    
    optimizer = torch.optim.SGD([engine.weights, engine.biases], lr=learning_rate)
    
    print(f"Params: beta={beta}, lr={learning_rate}")
    
    # Training Loop
    total_loss = 0
    history = []
    
    for i in range(len(data_bytes) - 1):
        input_char = data_bytes[i]
        target_char = data_bytes[i+1]
        
        # 1. Encode Input
        input_vec = io.encode(input_char, num_nodes)
        
        # 2. Free Phase (Settling)
        engine.settle(input_vec, duration_steps=free_steps)
        state_free = engine.state.clone()
        
        # 3. Measurement (Prediction)
        predicted_byte = io.decode(state_free)
        loss = 1 if predicted_byte != target_char else 0
        total_loss += loss
        
        # Log prediction
        idx_in_seq = i % 3
        expected = chr(target_char)
        got = chr(predicted_byte) if 32 <= predicted_byte < 127 else '?'
        
        print(f"Step {i:02d} | Input: {chr(input_char)} | Target: {expected} | Pred: {got} | Loss: {loss}")
        
        # 4. Nudged Phase (Teaching)
        # Create a target pattern (one-hot ish)
        target_nudge = torch.zeros(num_nodes, device=device)
        # Adjusted logic for small brain:
        # Use sensory output offset!
        target_idx = (int(target_char) % 256) + 256 
        target_nudge[target_idx] = 1.0
        
        # We need to know WHICH nodes the 'decode' function looks at.
        # ByteSensoryInterface needs to be checked. Assuming checking View for now.
        
        # Settle with nudge
        engine.settle(input_vec, duration_steps=nudge_steps, nudge_target=target_nudge, beta=beta)
        state_nudged = engine.state.clone()
        
        # 5. Weight Update (EqProp)
        # Delta = (Free - Nudged) ? No, EqProp usually:
        # Gradient ~ epsilon * (State_Nudged - State_Free)
        # Implementation check needed: engine.update_weights_eq_prop
        engine.update_weights_eq_prop(state_free, state_nudged, beta=beta, learning_rate=learning_rate)
        
    print(f"Total Loss: {total_loss} / {len(data_bytes)-1}")
    
    # Test Generation
    print("\n--- Generation Test ---")
    start_char = 'A'
    curr_char = start_char
    gen_text = curr_char
    
    input_vec = io.encode(ord(curr_char), num_nodes)
    
    for _ in range(10):
        engine.settle(input_vec, duration_steps=free_steps)
        state = engine.state.clone()
        next_byte = io.decode(state)
        next_c = chr(next_byte) if 32 <= next_byte < 127 else '?'
        gen_text += next_c
        
        # Close loop
        input_vec = io.encode(next_byte, num_nodes)
        
    print(f"Generated: {gen_text}")
    print("Expected:  ABCABCABC...")
    
if __name__ == "__main__":
    main()
