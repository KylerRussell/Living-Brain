
import torch
import numpy as np
import os
import time
import argparse
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from ndcd.graph import DynamicGraph
from ndcd.engine_torch import DragonEngineTorch
from ndcd.curriculum_gen import generate_chars, generate_toddler_words, generate_quotes

def ensure_data(data_path, phase_name):
    if not os.path.exists(data_path):
        print(f"Data {data_path} not found. Generating for {phase_name}...")
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        if "level1" in data_path: generate_chars(data_path)
        elif "level2" in data_path: generate_toddler_words(data_path)
        elif "level3" in data_path: generate_quotes(data_path)
        elif "sherlock" in data_path:
            # Check if we can download it, otherwise warn
             import urllib.request
             url = "https://www.gutenberg.org/files/1661/1661-0.txt"
             try:
                 urllib.request.urlretrieve(url, data_path)
             except Exception as e:
                 print(f"Failed to download Sherlock: {e}")

class SequentialTrainer:
    def __init__(self, num_nodes=10000, device='cpu'):
        self.device = device
        self.num_nodes = num_nodes
        
        # 1. Initialize Graph
        print("Initializing Dynamic Graph...")
        self.graph = DynamicGraph(num_nodes=num_nodes, m_edges=10, p_triad=0.1, seed=42)
        indices, values = self.graph.export_sparse_components()
        biases = self.graph.biases
        biases = self.graph.biases
        taus = self.graph.taus
        
        # Spectral Radius Tuning
        print("Tuning Spectral Radius to 0.95 (Stable)...")
        # Construct sparse matrix depending on format
        # values are from graph.weights[rows, cols]
        # We need to construct Scipy sparse matrix
        # indices is 2xE, values is 1xE
        row = indices[0]
        col = indices[1]
        w_sparse = sp.csr_matrix((values, (row, col)), shape=(num_nodes, num_nodes))
        
        try:
            # Calculate largest eigs
            eigvals = eigs(w_sparse, k=1, which='LM', return_eigenvectors=False)
            max_eig = np.abs(eigvals[0])
            print(f"Original Spectral Radius: {max_eig:.4f}")
            
            target_radius = 0.95 # Stable for EqProp
            scale_factor = target_radius / (max_eig + 1e-8)
            values = values * scale_factor
            print(f"Scaled weights by {scale_factor:.4f}")
        except Exception as e:
            print(f"Warning: Spectral tuning failed ({e}). Using default.")
        
        # 2. Initialize Engine
        # Continuous state is maintained in self.engine.state
        self.engine = DragonEngineTorch(num_nodes, indices, values, biases, taus, dt=0.01, device=device)
        
        # 3. Define I/O Masks
        # Nodes 0-255: Input
        # Nodes 256-511: Output
        self.input_indices = list(range(0, 256))
        self.output_indices = list(range(256, 512))
        
        # Pre-compute One-Hot Identity Matrices for fast I/O
        # Input Projection: mapping byte 0-255 to node 0-255 is just Identity
        # But technically we inject current into these nodes.
        self.eye = torch.eye(256, device=device)
        
    def train_babbling(self, iterations=1000, lr=0.01):
        print(f"\n=== Starting Phase 0: Hebbian Babbling ===")
        # Input Mask: Clamp inputs
        input_mask = torch.zeros(self.num_nodes, device=self.device)
        input_mask[self.input_indices] = 1.0
        
        start_time = time.time()
        
        for i in range(iterations):
            # Random Static Input
            input_vals = torch.rand(len(self.input_indices), device=self.device)
            input_vec = torch.zeros(self.num_nodes, device=self.device)
            input_vec[self.input_indices] = input_vals
            
            # Settle (Free logic, but inputs clamped)
            self.engine.settle(input_vec, input_mask=input_mask)
            
            # Hebbian Update
            self.engine.update_weights_hebbian(learning_rate=lr)
            
            if i % 100 == 0:
                print(f"Babbling Step {i}/{iterations}", end='\r')
                
        print(f"\nBabbling Complete. Time: {time.time()-start_time:.2f}s")
        
    def train_phase(self, phase_name, data_path, iterations, steps_per_iter, beta=0.1, lr=0.01, use_rl=False):
        print(f"\n=== Starting Phase: {phase_name} ===")
        ensure_data(data_path, phase_name)
        
        if not os.path.exists(data_path):
            print(f"Skipping {phase_name} (Data missing)")
            return

        with open(data_path, 'rb') as f:
            data = f.read()
            
        data_len = len(data)
        curr_idx = 0
        
        start_time = time.time()
        
        total_steps = iterations * steps_per_iter
        
        loss_accum = 0.0
        correct_count = 0
        
        # EqProp Hyperparams
        # free_phase_steps = 10 # Settling time
        # nudged_phase_steps = 10 
        
        for step in range(total_steps):
            # 1. Get Data Stream
            if curr_idx >= data_len - 1:
                curr_idx = 0
            
            input_byte = data[curr_idx]
            target_byte = data[curr_idx + 1]
            curr_idx += 1
            
            # 2. Input Setup
            # Clamp Input Nodes (0-255)
            
            # Hard Clamp Mask
            input_mask = torch.zeros(self.num_nodes, device=self.device)
            input_mask[self.input_indices] = 1.0

            input_vec = torch.zeros(self.num_nodes, device=self.device)
            # One-hot input current (used as value for clamping)
            input_vec[input_byte] = 5.0 
            
            # 3. Free Phase (Dream)
            # Run dynamics. Result is state_free.
            # Using default convergence (removed duration_steps)
            self.engine.settle(input_vec, input_mask=input_mask)
            state_free = self.engine.state.clone()
            
            # Measure Prediction (Readout) during Free Phase
            # Output is nodes 256-511
            output_activity = state_free[256:512]
            probs = torch.softmax(output_activity, dim=0)
            pred_idx = torch.argmax(probs).item()
            
            reward = 0.0
            if pred_idx == target_byte:
                correct_count += 1
                reward = 1.0
            else:
                reward = -0.1
                
            loss = -torch.log(probs[target_byte] + 1e-8).item()
            loss_accum += loss
            
            if use_rl:
                # 4a. RL Update (Dopamine)
                self.engine.update_weights_dopamine(reward, lr)
            else:
                # 4b. Nudged Phase (EqProp)
                # We want to pull the Output Nodes towards the target.
                
                # Construct Nudge Target Vector
                # Target for these nodes is One-Hot(target_byte).
                
                # Nudge Mask: Only Output Nodes
                nudge_mask = torch.zeros(self.num_nodes, device=self.device)
                nudge_mask[self.output_indices] = 1.0
                
                # Initialize target to slightly negative (suppress incorrect classes)
                nudge_target = torch.ones(self.num_nodes, device=self.device) * -0.1 
                # Zero out the non-output nodes so we don't suppress the brain!
                nudge_target[:256] = 0.0 
                nudge_target[512:] = 0.0
    
                # Pull the correct answer UP strongly
                nudge_target[256 + target_byte] = 1.0
                
                # Run Nudged
                self.engine.settle(input_vec, nudge_target=nudge_target, beta=beta, nudge_mask=nudge_mask, input_mask=input_mask)
                state_nudged = self.engine.state.clone()
                
                # 5. Weight Update (EqProp)
                self.engine.update_weights_eq_prop(state_free, state_nudged, beta, lr)
            
            if step % 100 == 0:
                 print(f"Step {step}/{total_steps} | Loss: {loss:.4f} | Acc: {correct_count/(step+1):.2%}", end='\r')
                 
        print(f"\nPhase Complete. Avg Loss: {loss_accum/total_steps:.4f} | Final Acc: {correct_count/total_steps:.2%}")
        
    def generate(self, start_text="The", length=100):
        print(f"\n--- Generating: {start_text} ... ---")
        curr_text = start_text
        
        # Prime
        for char in start_text:
            val = ord(char)
            if val > 255: val = 0
            input_vec = torch.zeros(self.num_nodes, device=self.device)
            input_vec[val] = 5.0
            self.engine.settle(input_vec, duration_steps=30)
            
        for _ in range(length):
            # 1. Free run (with last input still fading? No, we need to feed ... nothing? or Silence?)
            # The autoregressive nature: We feed the LAST output as NEXT input?
            # Or we just let the network run?
            # Standard RNN generation: Input = Predicted Char.
            
            # Read current output
            state = self.engine.state
            out_act = state[256:512]
            probs = torch.softmax(out_act, dim=0)
            
            # Sample
            next_byte = torch.multinomial(probs, 1).item()
            char = chr(next_byte) if 0 <= next_byte < 128 else '?'
            curr_text += char
            
            # Feedback
            input_vec = torch.zeros(self.num_nodes, device=self.device)
            input_vec[next_byte] = 5.0
            self.engine.settle(input_vec, duration_steps=30)
            
        print(curr_text)
        print("--------------------------------------")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--nodes", type=int, default=10000) # Small for testing, 5000 for real
    args = parser.parse_args()
    
    device = args.device
    if torch.backends.mps.is_available() and device == 'cpu':
        device = 'mps'
    if torch.cuda.is_available() and device == 'cpu':
        device = 'cuda'
        
    print(f"Using device: {device}")
    
    trainer = SequentialTrainer(num_nodes=args.nodes, device=device)
    
    # Curriculum
    # Learning Rates need to be small for EqProp
    
    # Phase 0: Babbling (Warmup)
    trainer.train_babbling(iterations=1000)
    
    # Phase 1: Chars
    trainer.train_phase("Chars", "ndcd/data/level1_chars.txt", iterations=100, steps_per_iter=100, beta=0.5, lr=0.1, use_rl=False)
    trainer.generate(start_text="A")
    
    # Phase 2: Words
    trainer.train_phase("Words", "ndcd/data/level2_words.txt", iterations=200, steps_per_iter=100, beta=0.5, lr=0.05, use_rl=True)
    trainer.generate()
    
    # Phase 3: Quotes
    trainer.train_phase("Quotes", "ndcd/data/level3_quotes.txt", iterations=200, steps_per_iter=200, beta=0.5, lr=0.02, use_rl=True)
    trainer.generate()
    
    # Phase 4: Literature
    trainer.train_phase("Literature", "ndcd/data/sherlock.txt", iterations=500, steps_per_iter=500, beta=1.0, lr=0.01)
    trainer.generate(start_text="Sherlock", length=200)

if __name__ == "__main__":
    main()
