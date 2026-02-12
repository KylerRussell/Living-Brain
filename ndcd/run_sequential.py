
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
    def __init__(self, num_nodes=2000, device='cpu'):
        self.device = device
        self.num_nodes = num_nodes
        if num_nodes < 512:
            raise ValueError(f"num_nodes ({num_nodes}) must be >= 512 to support 256 input + 256 output nodes.")
        
        # 1. Initialize Graph
        print("Initializing Dynamic Graph...")
        self.graph = DynamicGraph(num_nodes=num_nodes, m_edges=20, p_triad=0.1, seed=42)
        indices, values = self.graph.export_sparse_components()
        biases = self.graph.biases
        taus = self.graph.taus
        
        self.indices = indices
        self.initial_values = values
        
        # Matched Input Scaling Factor
        # Maintains REA (Relative Effective Amplitude) when weights are scaled
        self.input_scale_factor = 1.0
        
        
    # Spectral Radius Tuning
        self.tune_spectral_radius(target_radius=0.95)
        
        # 2. Initialize Engine
        # Continuous state is maintained in self.engine.state
        self.engine = DragonEngineTorch(num_nodes, indices, self.initial_values, biases, taus, positions=self.graph.pos, dt=0.01, device=device)
        
        # 3. Define I/O Masks
        # Nodes 0-255: Input
        # Nodes 256-511: Output
        self.input_indices = list(range(0, 256))
        self.output_indices = list(range(256, 512))
        
        # Pre-compute One-Hot Identity Matrices for fast I/O
        self.eye = torch.eye(256, device=device)

    def tune_spectral_radius(self, target_radius=0.95):
        """
        Tunes the spectral radius of the weight matrix to a target value.
        Updates self.initial_values.
        If self.engine exists, it also updates self.engine.weight_values.
        """
        print(f"Tuning Spectral Radius to {target_radius:.2f} (Strict)...")
        
        if hasattr(self, 'engine'):
            # Pull from GPU if needed
            w_tensor = self.engine.weight_values.cpu().numpy()
            indices = self.engine.indices.cpu().numpy() 
        else:
             if not hasattr(self, 'initial_values'):
                 pass
             w_tensor = self.initial_values
             indices = self.indices
             
        row = indices[0]
        col = indices[1]
        w_sparse = sp.csr_matrix((w_tensor, (row, col)), shape=(self.num_nodes, self.num_nodes))
        
        try:
            # Calculate largest eigs
            eigvals = eigs(w_sparse, k=1, which='LM', return_eigenvectors=False)
            max_eig = np.abs(eigvals[0])
            print(f"Current Spectral Radius: {max_eig:.4f}")
            
            scale_factor = target_radius / (max_eig + 1e-8)
            w_tensor = w_tensor * scale_factor
            print(f"Scaled weights by {scale_factor:.4f}")
            
            # --- REMOVED MATCHED SCALING ---
            # Input scale fixed to 1.0 to avoid linear collapse
            self.input_scale_factor = 1.0
            print(f"Input Scale Factor fixed to {self.input_scale_factor:.4f}")
            
            if hasattr(self, 'engine'):
                self.engine.weight_values = torch.tensor(w_tensor, dtype=torch.float32, device=self.device)
                # Do not scale biases blindly
            else:
                self.initial_values = w_tensor

        except Exception as e:
            print(f"Warning: Spectral tuning failed ({e}). Using default.")
        
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
            input_vec[self.input_indices] = input_vals * self.input_scale_factor
            
            # Settle (Free logic, but inputs clamped)
            self.engine.settle(input_vec, input_mask=input_mask)
            
            # Hebbian Update
            # CRITICAL FIX 1: Much smaller learning rate to prevent explosion
            self.engine.update_weights_hebbian(learning_rate=lr * 0.01)
            
            # CRITICAL FIX 2: Clip weights after each update
            self.engine.weight_values.clamp_(-1.0, 1.0)
            
            # CRITICAL FIX 3: Monitor spectral radius during training
            if i % 100 == 0:
                W_sparse = torch.sparse_coo_tensor(
                    self.engine.indices, 
                    self.engine.weight_values,
                    (self.num_nodes, self.num_nodes)
                )
                W_dense = W_sparse.to_dense().cpu().numpy()
                current_rho = np.max(np.abs(np.linalg.eigvals(W_dense)))
                print(f"Babbling {i}/{iterations} - Spectral Radius: {current_rho:.4f}")
                
                # Emergency brake
                if current_rho > 1.5:
                    print(f"WARNING: Spectral radius too high, applying damping")
                    self.engine.weight_values *= 0.8
        
        # RE-TUNE SPECTRAL RADIUS
        print("\nRe-tuning after Babbling...")
        self.tune_spectral_radius(target_radius=0.95)
        
    def train_phase(self, phase_name, data_path, iterations, steps_per_iter, beta=0.1, lr=0.01, use_rl=False):
        print(f"\n=== Starting Phase: {phase_name} ===")
        print(f"Run started at: {time.ctime()}")
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
        # Rolling Accuracy Window
        acc_window = []
        acc_top3_window = []
        
        for step in range(total_steps):
            
            # --- Sleep / Remodeling Cycle ---
            if step > 0 and step % 1000 == 0:
                print("\n--- Initiating Sleep Phase (Homeostasis & Restructuring) ---")
                protected = self.input_indices + self.output_indices
                self.engine.remodel_structure(turnover_rate=0.05, protected_nodes=protected)
                self.tune_spectral_radius(target_radius=0.95)
            
            # 1. Get Data Stream
            if curr_idx >= data_len - 1:
                curr_idx = 0
            
            input_byte = data[curr_idx]
            target_byte = data[curr_idx + 1]
            curr_idx += 1
            
            # 2. Input Setup
            input_mask = torch.zeros(self.num_nodes, device=self.device)
            input_mask[self.input_indices] = 1.0

            input_vec = torch.zeros(self.num_nodes, device=self.device)
            input_vec[input_byte] = 1.0 * self.input_scale_factor 
            
            # 3. Free Phase (Monitor Only - or use as pivot if doing one-sided)
            # For Symmetric Nudging, we don't strictly *need* the free phase state for gradient,
            # but we need it to calculate the prediction loss!
            
            inhib_mask = torch.zeros(self.num_nodes, device=self.device)
            inhib_mask[self.output_indices] = 1.0
            
            # Run Free Phase
            self.engine.settle(input_vec, input_mask=input_mask, inhibition_mask=inhib_mask, inhibition_beta=1.0)
            state_free = self.engine.state.clone()
            
            # Measure Prediction
            # Fix: Use activated state (tanh) for output probability calculation
            # Use activation from engine state directly or consistent with settling
            output_activity = torch.tanh(state_free[256:512])
            probs = torch.softmax(output_activity, dim=0)
            pred_idx = torch.argmax(probs).item()
            
            # Metrics
            # Check state norm to ensure non-linear regime
            state_norm = torch.norm(state_free) / np.sqrt(self.num_nodes)
            
            is_correct = (pred_idx == target_byte)
            if is_correct: acc_window.append(1.0)
            else: acc_window.append(0.0)
            
            _, top3_indices = torch.topk(probs, 3)
            if target_byte in top3_indices.tolist(): acc_top3_window.append(1.0)
            else: acc_top3_window.append(0.0)
                
            loss = -torch.log(probs[target_byte] + 1e-8).item()
            loss_accum += loss
            
            if use_rl:
                # RL Update
                reward = 1.0 if is_correct else -0.1
                # Increase attention on error
                attention_val = 1.0 + (loss * 0.1) 
                self.engine.apply_neuromodulators(reward=reward, attention=attention_val, mood=0.0)
            else:
                # --- SYMMETRIC EQUILIBRIUM PROPAGATION ---
                
                # Nudge Targets
                nudge_mask = torch.zeros(self.num_nodes, device=self.device)
                nudge_mask[self.output_indices] = 1.0
                
                # Target Vector construction
                # We want to pull correct answer UP, incorrect DOWN? NO.
                # Nudged Phase - gentler target encoding
                # Target: +1.0 for correct class
                # Other outputs: 0.0 (not -0.1) -> More stable
                target_vec = torch.zeros(self.num_nodes, device=self.device)
                target_vec[256 + target_byte] = 1.0
                
                # Positive Phase (+beta)
                # s_pos = settle(x, beta, target)
                # We start from free state? Or input state? Starting from free state is faster.
                self.engine.state = state_free.clone() 
                self.engine.settle(input_vec, nudge_target=target_vec, beta=beta, nudge_mask=nudge_mask, input_mask=input_mask, inhibition_mask=inhib_mask, inhibition_beta=1.0)
                state_pos = self.engine.state.clone()
                
                # Negative Phase (-beta)
                # s_neg = settle(x, -beta, target)
                # Start from free state again
                self.engine.state = state_free.clone()
                self.engine.settle(input_vec, nudge_target=target_vec, beta=-beta, nudge_mask=nudge_mask, input_mask=input_mask, inhibition_mask=inhib_mask, inhibition_beta=1.0)
                state_neg = self.engine.state.clone()
                
                # Weight Update
                self.engine.update_weights_eq_prop(state_pos, state_neg, beta, lr, decay=1e-5)
            
            if step % 100 == 0:
                 if len(acc_window) > 100: acc_window = acc_window[-100:]
                 if len(acc_top3_window) > 100: acc_top3_window = acc_top3_window[-100:]
                 
                 rolling_acc = sum(acc_window) / len(acc_window) if len(acc_window) > 0 else 0.0
                 rolling_acc3 = sum(acc_top3_window) / len(acc_top3_window) if len(acc_top3_window) > 0 else 0.0
                 
                 elapsed = time.time() - start_time
                 print(f"Step {step}/{total_steps} | Time: {elapsed:.0f}s | Loss: {loss:.4f} | Acc: {rolling_acc:.2%} | Top3: {rolling_acc3:.2%} | ||s||: {state_norm:.3f}", end='\r')
                 
        final_acc = sum(acc_window)/len(acc_window) if len(acc_window) > 0 else 0.0
        final_acc3 = sum(acc_top3_window)/len(acc_top3_window) if len(acc_top3_window) > 0 else 0.0
        print(f"\nPhase Complete. Avg Loss: {loss_accum/total_steps:.4f} | Final Acc: {final_acc:.2%}")
        
    def generate(self, start_text="The", length=100):
        print(f"\n--- Generating: {start_text} ... ---")
        curr_text = start_text
        
        # Prime
        for char in start_text:
            val = ord(char)
            if val > 255: val = 0
            input_vec = torch.zeros(self.num_nodes, device=self.device)
            input_vec[val] = 1.0 * self.input_scale_factor
            self.engine.settle(input_vec, max_steps=30)
            
        for _ in range(length):
            state = self.engine.state
            # Fix: Use activated state
            out_act = torch.tanh(state[256:512])
            probs = torch.softmax(out_act, dim=0)
            
            # Sample
            next_byte = torch.multinomial(probs, 1).item()
            char = chr(next_byte) if 0 <= next_byte < 128 else '?'
            curr_text += char
            
            # Feedback
            input_vec = torch.zeros(self.num_nodes, device=self.device)
            input_vec[next_byte] = 1.0 * self.input_scale_factor
            self.engine.settle(input_vec, max_steps=30)
            
        print(curr_text)
        print("--------------------------------------")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--nodes", type=int, default=2000) # Small for testing, 5000 for real
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
    # Reduced Beta to 0.05, lr to 0.01 for stability
    trainer.train_phase("Chars", "ndcd/data/level1_chars.txt", iterations=500, steps_per_iter=100, beta=0.05, lr=0.01, use_rl=False)
    trainer.generate(start_text="A")
    
    # Phase 2: Words
    trainer.train_phase("Words", "ndcd/data/level2_words.txt", iterations=200, steps_per_iter=100, beta=0.05, lr=0.01, use_rl=True)
    trainer.generate()
    
    # Phase 3: Quotes
    trainer.train_phase("Quotes", "ndcd/data/level3_quotes.txt", iterations=200, steps_per_iter=200, beta=0.05, lr=0.01, use_rl=True)
    trainer.generate()
    
    # Phase 4: Literature
    trainer.train_phase("Literature", "ndcd/data/sherlock.txt", iterations=500, steps_per_iter=500, beta=0.1, lr=0.005)
    trainer.generate(start_text="Sherlock", length=200)

if __name__ == "__main__":
    main()
