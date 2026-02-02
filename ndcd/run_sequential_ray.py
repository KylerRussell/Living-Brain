
import ray
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
             import urllib.request
             url = "https://www.gutenberg.org/files/1661/1661-0.txt"
             try:
                 urllib.request.urlretrieve(url, data_path)
             except Exception as e:
                 print(f"Failed to download Sherlock: {e}")

@ray.remote(num_cpus=1)
class RemoteTrainer:
    def __init__(self, num_nodes=5000, device='cpu'):
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
        row = indices[0]
        col = indices[1]
        w_sparse = sp.csr_matrix((values, (row, col)), shape=(num_nodes, num_nodes))
        
        try:
            eigvals = eigs(w_sparse, k=1, which='LM', return_eigenvectors=False)
            max_eig = np.abs(eigvals[0])
            print(f"Original Spectral Radius: {max_eig:.4f}")
            
            target_radius = 0.95 
            scale_factor = target_radius / (max_eig + 1e-8)
            values = values * scale_factor
            print(f"Scaled weights by {scale_factor:.4f}")
        except Exception as e:
            print(f"Warning: Spectral tuning failed ({e}). Using default.")
        
        # 2. Initialize Engine
        self.engine = DragonEngineTorch(num_nodes, indices, values, biases, taus, dt=0.01, device=device)
        self.eye = torch.eye(256, device=device)
        self.input_indices = list(range(0, 256))
        self.output_indices = list(range(256, 512))
        
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
                print(f"Babbling Step {i}/{iterations}")
                
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
        total_steps = iterations * steps_per_iter
        
        loss_accum = 0.0
        correct_count = 0
        
        for step in range(total_steps):
            if curr_idx >= data_len - 1:
                curr_idx = 0
            
            input_byte = data[curr_idx]
            target_byte = data[curr_idx + 1]
            curr_idx += 1
            
            # Input Setup
            input_mask = torch.zeros(self.num_nodes, device=self.device)
            input_mask[self.input_indices] = 1.0
            
            input_vec = torch.zeros(self.num_nodes, device=self.device)
            input_vec[input_byte] = 5.0 
            
            self.engine.settle(input_vec, input_mask=input_mask)
            state_free = self.engine.state.clone()
            
            # Measure Prediction
            output_activity = state_free[256:512]
            probs = torch.softmax(output_activity, dim=0)
            pred_idx = torch.argmax(probs).item()
            
            reward = 0.0
            if pred_idx == target_byte:
                correct_count += 1
                reward = 1.0
            else:
                reward = -0.1
                
            loss_accum += -torch.log(probs[target_byte] + 1e-8).item()
            
            if use_rl:
                 # RL Update (Dopamine)
                 self.engine.update_weights_dopamine(reward, lr)
            else:
                # Nudged Phase (Contrastive)
                # Initialize target to slightly negative (suppress incorrect classes)
                nudge_target = torch.ones(self.num_nodes, device=self.device) * -0.1 
                # Zero out the non-output nodes!
                nudge_target[:256] = 0.0 
                nudge_target[512:] = 0.0
    
                # Pull the correct answer UP strongly
                nudge_target[256 + target_byte] = 1.0
    
                nudge_mask = torch.zeros(self.num_nodes, device=self.device)
                nudge_mask[self.output_indices] = 1.0
                
                self.engine.settle(input_vec, nudge_target=nudge_target, beta=beta, nudge_mask=nudge_mask, input_mask=input_mask)
                state_nudged = self.engine.state.clone()
                
                # Update
                self.engine.update_weights_eq_prop(state_free, state_nudged, beta, lr)
            
            if step % 100 == 0:
                 # Print to Ray logs
                 print(f"Step {step}/{total_steps} | Acc: {correct_count/(step+1):.2%}")
                 
        print(f"Phase Complete. Final Acc: {correct_count/total_steps:.2%}")
        return correct_count/total_steps
        
    def generate(self, start_text="The", length=100):
        print(f"\n--- Generating: {start_text} ... ---")
        curr_text = start_text
        
        for char in start_text:
            val = ord(char)
            if val > 255: val = 0
            input_vec = torch.zeros(self.num_nodes, device=self.device)
            input_vec[val] = 5.0
            self.engine.settle(input_vec, duration_steps=30)
            
        for _ in range(length):
            state = self.engine.state
            input_mask = torch.zeros(self.num_nodes, device=self.device)
            input_mask[self.input_indices] = 1.0
            out_act = state[256:512]
            probs = torch.softmax(out_act, dim=0)
            
            next_byte = torch.multinomial(probs, 1).item()
            char = chr(next_byte) if 0 <= next_byte < 128 else '?'
            curr_text += char
            
            input_vec = torch.zeros(self.num_nodes, device=self.device)
            input_vec[next_byte] = 5.0
            self.engine.settle(input_vec, input_mask=input_mask)
            
        print(curr_text)
        print("--------------------------------------")
        return curr_text

def main():
    print("Connecting to Ray...")
    ray.init(ignore_reinit_error=True)
    
    # Instantiate Remote Actor
    # We use a single actor to maintain state!
    trainer = RemoteTrainer.remote(num_nodes=1000, device='cpu')
    
    # Run Phases
    print("Starting Sequential Training on Ray...")
    
    # Phase 0: Babbling
    ray.get(trainer.train_babbling.remote(iterations=1000))

    # Phase 1: Chars
    ray.get(trainer.train_phase.remote("Chars", "ndcd/data/level1_chars.txt", iterations=100, steps_per_iter=100, beta=0.5, lr=0.1, use_rl=False))
    print(ray.get(trainer.generate.remote(start_text="A")))
    
    # Phase 2: Words
    ray.get(trainer.train_phase.remote("Words", "ndcd/data/level2_words.txt", iterations=200, steps_per_iter=100, beta=0.5, lr=0.05, use_rl=True))
    print(ray.get(trainer.generate.remote()))
    
    # Phase 3: Quotes
    ray.get(trainer.train_phase.remote("Quotes", "ndcd/data/level3_quotes.txt", iterations=200, steps_per_iter=200, beta=0.5, lr=0.02, use_rl=True))
    print(ray.get(trainer.generate.remote()))
    
    # Phase 4: Literature
    ray.get(trainer.train_phase.remote("Literature", "ndcd/data/sherlock.txt", iterations=500, steps_per_iter=500, beta=1.0, lr=0.01))
    print(ray.get(trainer.generate.remote(start_text="Sherlock", length=200)))
    
    print("Done!")
    ray.shutdown()

if __name__ == "__main__":
    main()
