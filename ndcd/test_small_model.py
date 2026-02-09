
import torch
import numpy as np
import time
import sys
import os

# Ensure we can import from local ndcd
sys.path.append(os.getcwd())

from ndcd.graph import DynamicGraph
from ndcd.engine_torch import DragonEngineTorch
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs

class SmallModelTrainer:
    def __init__(self, num_nodes=100, device='cpu'):
        self.device = device
        self.num_nodes = num_nodes
        
        # 1. Initialize Graph
        # Using DynamicGraph to get identifying topology + biological constants
        # 10% sensory (10 nodes), 10% motor (10 nodes)
        print("Initializing Dynamic Graph (Small)...")
        self.graph = DynamicGraph(num_nodes=num_nodes, m_edges=10, p_triad=0.2, seed=42)
        
        indices, values = self.graph.export_sparse_components()
        biases = self.graph.biases
        taus = self.graph.taus
        
        self.indices = indices
        self.initial_values = values
        
        # We need to ensure we save the arrays if we need to reconstruct or tune before engine init
        self.biases_np = biases
        self.taus_np = taus
        
        # Define Input/Output Mapping
        # indices 0-9 are Sensory (Input)
        # indices 10-19 are Motor (Output)
        self.input_indices = list(range(0, 10))
        self.output_indices = list(range(10, 20))
        
        print(f"Inputs: {self.input_indices}")
        print(f"Outputs: {self.output_indices}")
        
        # 2. Tune Spectral Radius
        self.tune_spectral_radius(target_radius=0.9)
        
        # 3. Initialize Engine
        self.engine = DragonEngineTorch(
            num_nodes, 
            self.indices, 
            self.initial_values, 
            biases, 
            taus, 
            positions=self.graph.pos, 
            dt=0.01, 
            device=device
        )
        
        # Bias initialization for sparsity (as seen in run_simple)
        self.engine.biases.fill_(-0.5)

        # Monitoring
        self.loss_history = []
        
    def tune_spectral_radius(self, target_radius=0.9):
        print(f"Tuning Spectral Radius to {target_radius:.2f}...")
        w_tensor = self.initial_values
        indices = self.indices
        
        row = indices[0]
        col = indices[1]
        
        # Construct dense/sparse matrix for scipy
        w_sparse = csr_matrix((w_tensor, (row, col)), shape=(self.num_nodes, self.num_nodes))
        
        try:
            # Calculate spectral radius
            eigvals = eigs(w_sparse, k=1, which='LM', return_eigenvectors=False)
            max_eig = np.abs(eigvals[0])
            
            if max_eig == 0:
                print("Warning: Max eigenvalue is 0. Cannot scale.")
                return
                
            scale_factor = target_radius / (max_eig + 1e-8)
            self.initial_values = w_tensor * scale_factor
            print(f"  -> Scaled by {scale_factor:.4f} (Radius: {max_eig:.4f} -> {target_radius})")
            
            # If engine exists, update it too
            if hasattr(self, 'engine'):
                self.engine.weight_values = torch.tensor(self.initial_values, dtype=torch.float32, device=self.device)
                
        except Exception as e:
            print(f"Warning: Spectral tuning failed ({e}).")

    def train(self, iterations=1000, lr=0.05, beta=1.0):
        print("\n=== Starting Training: Number Sequence (0->1->...->9->0) ===")
        
        start_time = time.time()
        
        # Data Generator: Infinite loop of 0..9
        def data_gen():
            while True:
                for i in range(10):
                    yield i
        
        data = data_gen()
        
        # Initial input
        curr_num = next(data)
        
        correct_count = 0
        window_size = 50
        acc_window = []
        
        for step in range(iterations):
            target_num = next(data) # The next number in sequence
            
            # 1. Create Inputs
            # One-hot input vector at the specific input neuron
            input_vec = torch.zeros(self.num_nodes, device=self.device)
            input_idx = self.input_indices[curr_num]
            input_vec[input_idx] = 1.0
            
            # Input Mask: Clamp inputs so they aren't overwritten by feedback
            input_mask = torch.zeros(self.num_nodes, device=self.device)
            input_mask[self.input_indices] = 1.0
            
            # 2. Free Phase (Observation)
            # Run the engine to seeing what it predicts
            
            # Helper for lateral inhibition on outputs (Softmax-like competition)
            inhib_mask = torch.zeros(self.num_nodes, device=self.device)
            inhib_mask[self.output_indices] = 1.0 # Only inhibit outputs
            
            self.engine.settle(
                input_vec, 
                input_mask=input_mask, 
                inhibition_mask=inhib_mask, 
                inhibition_beta=0.5 # Moderate inhibition
            )
            state_free = self.engine.state.clone()
            
            # Check Prediction
            # Get activities of output neurons
            output_acts = state_free[self.output_indices]
            pred_idx = torch.argmax(output_acts).item() # 0-9 relative to output_indices
            
            is_correct = (pred_idx == target_num)
            acc_window.append(1.0 if is_correct else 0.0)
            if len(acc_window) > window_size:
                acc_window.pop(0)
            
            # 3. Nudged Phase (Teaching)
            # We want the output neuron corresponding to 'target_num' to be active
            target_output_idx = self.output_indices[target_num]
            
            # Nudge Target: Everything 0 (suppressed) except the correct one
            nudge_target = torch.full((self.num_nodes,), -0.5, device=self.device)
            # Don't nudge inputs, let them be clamped
            nudge_target[target_output_idx] = 1.0 # Pull up correct answer
            
            # Nudge Mask: Only apply nudge to output neurons
            nudge_mask = torch.zeros(self.num_nodes, device=self.device)
            nudge_mask[self.output_indices] = 1.0 
            
            self.engine.settle(
                input_vec,
                input_mask=input_mask,
                nudge_target=nudge_target,
                nudge_mask=nudge_mask,
                beta=beta,
                inhibition_mask=inhib_mask,
                inhibition_beta=0.5
            )
            state_nudged = self.engine.state.clone()
            
            # 4. Weight Update (EqProp)
            self.engine.update_weights_eq_prop(state_free, state_nudged, beta, lr, decay=1e-5)
            
            # Update current for next step
            curr_num = target_num
            
            # Logs
            if step % 50 == 0:
                acc = sum(acc_window) / len(acc_window) if acc_window else 0.0
                print(f"Step {step:04d} | Acc: {acc:.2%} | Target: {target_num} Pred: {pred_idx}")
                # print(f"   Free Output: {output_acts.detach().cpu().numpy().round(2)}")
                
        print(f"Final Accuracy: {sum(acc_window)/len(acc_window):.2%}")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    trainer = SmallModelTrainer(num_nodes=100, device=device)
    trainer.train(iterations=2000, lr=0.1, beta=1.0)

if __name__ == "__main__":
    main()
