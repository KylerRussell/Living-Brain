
import torch
import numpy as np
import os
import time
import argparse
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from ndcd.graph import DynamicGraph
from ndcd.engine_torch import DragonEngineTorch
from ndcd.curriculum_gen import generate_simple_lowercase

def ensure_simple_data(data_path):
    if not os.path.exists(data_path):
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        generate_simple_lowercase(data_path)

class SimpleTrainer:
    def __init__(self, num_nodes=1000, device='cpu'):
        self.device = device
        self.num_nodes = num_nodes
        
        # 1. Initialize Graph
        print("Initializing Dynamic Graph (Small)...")
        # Higher connectivity for small graph to ensure paths exist
        self.graph = DynamicGraph(num_nodes=num_nodes, m_edges=30, p_triad=0.2, seed=42)
        indices, values = self.graph.export_sparse_components()
        biases = self.graph.biases
        taus = self.graph.taus
        
        self.indices = indices
        self.initial_values = values
        
        # Spectral Radius Tuning
        self.tune_spectral_radius(target_radius=1.1)
        
        # 2. Initialize Engine
        self.engine = DragonEngineTorch(num_nodes, indices, self.initial_values, biases, taus, positions=self.graph.pos, dt=0.01, device=device)
        self.input_indices = list(range(0, 256))
        self.output_indices = list(range(256, 512))

        # Stats
        self.loss_history = []
        self.acc_history = []

    def tune_spectral_radius(self, target_radius=1.1):
        print(f"Tuning Spectral Radius to {target_radius:.2f}...")
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
            scale_factor = target_radius / (max_eig + 1e-8)
            w_tensor = w_tensor * scale_factor
            
            if hasattr(self, 'engine'):
                self.engine.weight_values = torch.tensor(w_tensor, dtype=torch.float32, device=self.device)
            else:
                self.initial_values = w_tensor
            print(f"  -> Scaled by {scale_factor:.4f} (Radius: {max_eig:.4f} -> {target_radius})")

        except Exception as e:
            print(f"Warning: Spectral tuning failed ({e}).")
        
    def train(self, data_path, iterations=1000, lr=0.05, beta=1.0):
        print(f"\n=== Starting Simple Training ===")
        ensure_simple_data(data_path)
        
        with open(data_path, 'r') as f:
            data_str = f.read()
            
        # Filter to only valid ascii
        labels = [ord(c) for c in data_str if ord(c) < 256]
        data_len = len(labels)
        curr_idx = 0
        
        total_steps = iterations
        start_time = time.time()
        
        acc_window = []
        
        for step in range(total_steps):
            if curr_idx >= data_len - 1: curr_idx = 0
            
            input_byte = labels[curr_idx]
            target_byte = labels[curr_idx + 1]
            curr_idx += 1
            
            # Setup Inputs
            input_mask = torch.zeros(self.num_nodes, device=self.device)
            input_mask[self.input_indices] = 1.0
            input_vec = torch.zeros(self.num_nodes, device=self.device)
            input_vec[input_byte] = 1.0 
            
            # --- Free Phase ---
            # Enable Lateral Inhibition for Output Competition (Consistent with Nudge)
            inhib_mask = torch.zeros(self.num_nodes, device=self.device)
            inhib_mask[self.output_indices] = 1.0
            
            self.engine.settle(input_vec, input_mask=input_mask, inhibition_mask=inhib_mask, inhibition_beta=1.0)
            state_free = self.engine.state.clone()
            
            # Predict
            out_act = state_free[256:512]
            probs = torch.softmax(out_act, dim=0)
            pred_idx = torch.argmax(probs).item()
            is_correct = (pred_idx == target_byte)
            acc_window.append(1.0 if is_correct else 0.0)
            if len(acc_window) > 100: acc_window.pop(0)
            
            loss = -torch.log(probs[target_byte] + 1e-8).item()
            
            # --- Nudged Phase ---
            # Strong Soft Nudge
            nudge_target = torch.ones(self.num_nodes, device=self.device) * -0.5 # Suppression
            nudge_target[:256] = 0.0 
            nudge_target[512:] = 0.0 
            nudge_target[256 + target_byte] = 1.0 # Pull Target Up
            
            nudge_mask = torch.zeros(self.num_nodes, device=self.device)
            nudge_mask[self.output_indices] = 1.0
            
            # Keep inhibition in nudged phase
            inhib_mask = torch.zeros(self.num_nodes, device=self.device)
            inhib_mask[self.output_indices] = 1.0
            
            self.engine.settle(input_vec, nudge_target=nudge_target, beta=beta, nudge_mask=nudge_mask, input_mask=input_mask, inhibition_mask=inhib_mask, inhibition_beta=1.0)
            state_nudged = self.engine.state.clone()
            
            # Debug: Check if Nudge worked
            out_act_nudge = state_nudged[256:512]
            pred_nudge = torch.argmax(out_act_nudge).item()
            
            # --- Update ---
            # Use Weight Decay!
            self.engine.update_weights_eq_prop(state_free, state_nudged, beta, lr, decay=1e-5)
            
            if step % 50 == 0:
                roll_acc = sum(acc_window)/len(acc_window) if acc_window else 0.0
                elapsed = time.time() - start_time
                print(f"Step {step}/{total_steps} | Loss: {loss:.4f} | Acc: {roll_acc:.2%}")
                print(f"  Free  -> Mean: {out_act.mean():.4f}, Max: {out_act.max():.4f}, Pred: {pred_idx}")
                print(f"  Nudge -> Mean: {out_act_nudge.mean():.4f}, Max: {out_act_nudge.max():.4f}, Pred: {pred_nudge} (Target: {target_byte})")
                
        print(f"\nTraining Complete. Final Accuracy: {sum(acc_window)/len(acc_window):.2%}")
        
    def generate(self, start_char='a', length=20):
        print(f"\n--- Generating from '{start_char}' ---")
        curr = ord(start_char)
        text = start_char
        
        for _ in range(length):
            input_vec = torch.zeros(self.num_nodes, device=self.device)
            input_vec[curr] = 1.0
            # input_mask = torch.zeros(self.num_nodes, device=self.device)
            # input_mask[self.input_indices] = 1.0
            
            self.engine.settle(input_vec) #, input_mask=input_mask)
            out_act = self.engine.state[256:512]
            probs = torch.softmax(out_act, dim=0)
            
            next_byte = torch.argmax(probs).item()
            char = chr(next_byte) if 32 <= next_byte < 127 else '?'
            text += char
            curr = next_byte
            
        print(f"Generated: {text}")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Tiny model for rapid testing
    trainer = SimpleTrainer(num_nodes=1000, device=device)
    
    # Train heavily on simple pattern
    # High LR and BETA for testing
    trainer.train("ndcd/data/simple_lower.txt", iterations=2000, lr=0.2, beta=5.0)
    
    trainer.generate('a')
    trainer.generate('m')

if __name__ == "__main__":
    main()
