
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

class DebugSequentialTrainer:
    def __init__(self, num_nodes=1000, device='cpu'):
        self.device = device
        self.num_nodes = num_nodes
        
        # 1. Initialize Graph
        print("Initializing Dynamic Graph...")
        self.graph = DynamicGraph(num_nodes=num_nodes, m_edges=20, p_triad=0.1, seed=42)
        indices, values = self.graph.export_sparse_components()
        biases = self.graph.biases
        taus = self.graph.taus
        
        self.indices = indices
        self.initial_values = values
        
        # Spectral Radius Tuning
        self.tune_spectral_radius(target_radius=0.99)
        
        # 2. Initialize Engine
        self.engine = DragonEngineTorch(num_nodes, indices, self.initial_values, biases, taus, positions=self.graph.pos, dt=0.01, device=device)
        self.input_indices = list(range(0, 256))
        self.output_indices = list(range(256, 512))

    def tune_spectral_radius(self, target_radius=0.95):
        print(f"Tuning Spectral Radius to {target_radius:.2f} (Stable)...")
        if hasattr(self, 'engine'):
            w_tensor = self.engine.weight_values.cpu().numpy()
            indices = self.engine.indices.cpu().numpy() 
        else:
             w_tensor = self.initial_values
             indices = self.indices
             
        row = indices[0]
        col = indices[1]
        w_sparse = sp.csr_matrix((w_tensor, (row, col)), shape=(self.num_nodes, self.num_nodes))
        
        try:
            eigvals = eigs(w_sparse, k=1, which='LM', return_eigenvectors=False)
            max_eig = np.abs(eigvals[0])
            print(f"Current Spectral Radius: {max_eig:.4f}")
            
            scale_factor = target_radius / (max_eig + 1e-8)
            w_tensor = w_tensor * scale_factor
            print(f"Scaled weights by {scale_factor:.4f}")
            
            if hasattr(self, 'engine'):
                self.engine.weight_values = torch.tensor(w_tensor, dtype=torch.float32, device=self.device)
            else:
                self.initial_values = w_tensor

        except Exception as e:
            print(f"Warning: Spectral tuning failed ({e}). Using default.")
        
    def train_babbling(self, iterations=100, lr=0.01):
        print(f"\n=== Starting Phase 0: Hebbian Babbling (Debug) ===")
        input_mask = torch.zeros(self.num_nodes, device=self.device)
        input_mask[self.input_indices] = 1.0
        
        for i in range(iterations):
            input_vals = torch.rand(len(self.input_indices), device=self.device)
            input_vec = torch.zeros(self.num_nodes, device=self.device)
            input_vec[self.input_indices] = input_vals
            
            self.engine.settle(input_vec, input_mask=input_mask)
            self.engine.update_weights_hebbian(learning_rate=lr)
            
            if i % 20 == 0:
                print(f"Babbling Step {i}/{iterations}")
                
        print("\nRe-tuning after Babbling...")
        self.tune_spectral_radius(target_radius=0.99)
        
    def train_phase_debug(self, phase_name, data_path, iterations, steps_per_iter, beta=0.1, lr=0.01):
        print(f"\n=== Starting Phase: {phase_name} (DEBUG) ===")
        ensure_data(data_path, phase_name)
        
        with open(data_path, 'rb') as f:
            data = f.read()
            
        data_len = len(data)
        curr_idx = 0
        total_steps = iterations * steps_per_iter
        
        print(f"DEBUG: Initial Weight Mean: {self.engine.weight_values.mean().item():.6f}, Std: {self.engine.weight_values.std().item():.6f}")
        
        for step in range(total_steps):
            if curr_idx >= data_len - 1: curr_idx = 0
            
            input_byte = data[curr_idx]
            target_byte = data[curr_idx + 1]
            curr_idx += 1
            
            input_mask = torch.zeros(self.num_nodes, device=self.device)
            input_mask[self.input_indices] = 1.0

            input_vec = torch.zeros(self.num_nodes, device=self.device)
            input_vec[input_byte] = 1.0 
            
            # Free Phase
            # Enable Lateral Inhibition to force a decision (Winner-Take-All)
            inhib_mask = torch.zeros(self.num_nodes, device=self.device)
            inhib_mask[self.output_indices] = 1.0
            
            self.engine.settle(input_vec, input_mask=input_mask, inhibition_mask=inhib_mask, inhibition_beta=0.5)
            state_free = self.engine.state.clone()
            
            # Predict
            output_activity = state_free[256:512]
            probs = torch.softmax(output_activity, dim=0)
            pred_idx = torch.argmax(probs).item()
            is_correct = (pred_idx == target_byte)
            
            if step % 20 == 0:
                print(f"\n--- STEP {step} ---")
                print(f"Stats - Free State: Mean {state_free.abs().mean():.4f}, Max {state_free.max():.4f}")
                print(f"Output Activity: Mean {output_activity.mean():.4f}, Std {output_activity.std():.4f}, Max {output_activity.max():.4f}")
                print(f"Probs max: {probs.max().item():.4f}, Min: {probs.min().item():.4f}. Target prob: {probs[target_byte].item():.4f}")
                print(f"Target Byte: {target_byte}, Pred: {pred_idx} ({'CORRECT' if is_correct else 'WRONG'})")
            
            # Nudged Phase
            nudge_target = torch.ones(self.num_nodes, device=self.device) * -0.8
            nudge_target[:256] = 0.0 
            nudge_target[512:] = 0.0
            nudge_target[256 + target_byte] = 1.0
            
            nudge_mask = torch.zeros(self.num_nodes, device=self.device)
            nudge_mask[self.output_indices] = 1.0
            
            inhib_mask = torch.zeros(self.num_nodes, device=self.device)
            inhib_mask[self.output_indices] = 1.0
            
            self.engine.settle(input_vec, nudge_target=nudge_target, beta=beta, nudge_mask=nudge_mask, input_mask=input_mask, inhibition_mask=inhib_mask, inhibition_beta=1.0)
            state_nudged = self.engine.state.clone()
            
            # Manual Gradient Calculation to Inspect
            rho_free = torch.tanh(state_free)
            rho_nudged = torch.tanh(state_nudged)
            
            idx_i = self.indices[0]
            idx_j = self.indices[1]
            rf_i = rho_free[idx_i]
            rf_j = rho_free[idx_j]
            rn_i = rho_nudged[idx_i]
            rn_j = rho_nudged[idx_j]
            grad_values = ((rn_i * rn_j) - (rf_i * rf_j)) / beta
            
            if step % 20 == 0:
                print(f"Gradient Stats: Mean {grad_values.mean().item():.6e}, Max {grad_values.max().item():.6e}, Min {grad_values.min().item():.6e}")
                print(f"Weight Values: Mean {self.engine.weight_values.mean().item():.6f}")

            # Update
            self.engine.update_weights_eq_prop(state_free, state_nudged, beta, lr)
                 
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    trainer = DebugSequentialTrainer(num_nodes=1000, device=device)
    
    # Short Babble
    trainer.train_babbling(iterations=50)
    
    # Train Phase
    trainer.train_phase_debug("Chars", "ndcd/data/level1_chars.txt", iterations=5, steps_per_iter=20, beta=1.0, lr=0.1)

if __name__ == "__main__":
    main()
