
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
        
        # 1. Initialize Graph
        print("Initializing Dynamic Graph...")
        self.graph = DynamicGraph(num_nodes=num_nodes, m_edges=20, p_triad=0.1, seed=42)
        indices, values = self.graph.export_sparse_components()
        biases = self.graph.biases
        biases = self.graph.biases
        taus = self.graph.taus
        
        self.indices = indices
        self.initial_values = values
        
        
        # Spectral Radius Tuning
        self.tune_spectral_radius(target_radius=0.99)
        
        # 2. Initialize Engine
        # Continuous state is maintained in self.engine.state
        self.engine = DragonEngineTorch(num_nodes, indices, self.initial_values, biases, taus, positions=self.graph.pos, dt=0.01, device=device)
                
        # 3. Define I/O Masks
        # Nodes 0-255: Input
        # Nodes 256-511: Output
        self.input_indices = list(range(0, 256))
        self.output_indices = list(range(256, 512))
        
        # Pre-compute One-Hot Identity Matrices for fast I/O
        # Input Projection: mapping byte 0-255 to node 0-255 is just Identity
        # But technically we inject current into these nodes.
        self.eye = torch.eye(256, device=device)

    def tune_spectral_radius(self, target_radius=0.95):
        """
        Tunes the spectral radius of the weight matrix to a target value.
        Updates self.initial_values.
        If self.engine exists, it also updates self.engine.weight_values.
        """
        print(f"Tuning Spectral Radius to {target_radius:.2f} (Stable)...")
        # Re-construct sparse matrix from components
        # We need to use what is currently available. 
        # If engine exists, use engine weights (as they change during babbling).
        # But we need numpy for scipy eigs.
        
        if hasattr(self, 'engine'):
            # Pull from GPU if needed
            w_tensor = self.engine.weight_values.cpu().numpy()
            indices = self.engine.indices.cpu().numpy() # Use dynamic indices from engine
        else:
             # Initial construction
             if not hasattr(self, 'initial_values'):
                 # Need to get from graph if not yet saved?
                 # Actually __init__ flow shows we have 'values' from graph export
                 # We need to save 'values' to self.initial_values in __init__
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
            
            if hasattr(self, 'engine'):
                self.engine.weight_values = torch.tensor(w_tensor, dtype=torch.float32, device=self.device)
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
            input_vec[self.input_indices] = input_vals
            
            # Settle (Free logic, but inputs clamped)
            self.engine.settle(input_vec, input_mask=input_mask)
            
            # Hebbian Update
            self.engine.update_weights_hebbian(learning_rate=lr)
            
            if i % 100 == 0:
                print(f"Babbling Step {i}/{iterations}", end='\r')
                
        print(f"\nBabbling Complete. Time: {time.time()-start_time:.2f}s")
        
        # RE-TUNE SPECTRAL RADIUS
        # Hebbian learning likely exploded the weights. We need to normalize back to 0.95
        # so EqProp starts in a stable regime.
        print("\nRe-tuning after Babbling...")
        self.tune_spectral_radius(target_radius=0.99)
        
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
        loss_accum = 0.0
        # Rolling Accuracy Window
        acc_window = []
        acc_top3_window = []
        
        # EqProp Hyperparams
        # free_phase_steps = 10 # Settling time
        # nudged_phase_steps = 10 
        
        for step in range(total_steps):
            
            # --- NEW: Sleep / Remodeling Cycle ---
            # Run this periodically (e.g., every 1000 steps)
            if step > 0 and step % 1000 == 0:
                print("\n--- Initiating Sleep Phase (Structural Plasticity) ---")
                
                # 1. Prune weak, grow new
                # growth_rate: How many new connections to try per cycle
                self.engine.remodel_structure(prune_threshold=0.005, growth_rate=200)
                
                # 2. Re-Stabilize (CRITICAL)
                # This ensures the new random weights don't push eigenvalues > 1.0
                self.tune_spectral_radius(target_radius=0.99)
                
                # Optional: Reset optimizer momentum if you were using Adam (not used here)
            
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
            input_vec[input_byte] = 1.0 
            
            # 3. Free Phase (Dream)
            # Run dynamics. Result is state_free.
            # Using default convergence (removed duration_steps)
            self.engine.settle(input_vec, input_mask=input_mask)
            state_free = self.engine.state.clone()
            
            # Measure Prediction (Readout) during Free Phase
            # Output is nodes 256-511
            output_activity = state_free[256:512]
            probs = torch.softmax(output_activity, dim=0)
            
            # Top-1
            pred_idx = torch.argmax(probs).item()
            
            # Top-3
            _, top3_indices = torch.topk(probs, 3)
            is_top3 = (target_byte in top3_indices.tolist())
            if is_top3:
                acc_top3_window.append(1.0)
            else:
                acc_top3_window.append(0.0)
            
            reward = 0.0
            is_correct = (pred_idx == target_byte)
            if is_correct:
                acc_window.append(1.0)
                reward = 1.0
            else:
                acc_window.append(0.0)
                reward = -0.1
                
            loss = -torch.log(probs[target_byte] + 1e-8).item()
            loss_accum += loss
            
            if use_rl:
                # 4a. RL Update (Dopamine)
                # 4a. RL Update (Multi-Factor Neuromodulation)
                # D(t) = Reward
                # A(t) = Attention (e.g. 1.0 + |Error|) -> Higher plasticity on error
                # S(t) = Mood (e.g. 0.0)
                
                attention_val = 1.0 + (loss * 0.1) # Simple heuristic: Pay attention when confused
                mood_val = 0.0 # Neural baseline
                
                self.engine.apply_neuromodulators(reward=reward, attention=attention_val, mood=mood_val)
            else:
                # 4b. Nudged Phase (EqProp)
                # We want to pull the Output Nodes towards the target.
                
                # Construct Nudge Target Vector
                # Target for these nodes is One-Hot(target_byte).
                
                # Nudge Mask: Only Output Nodes
                nudge_mask = torch.zeros(self.num_nodes, device=self.device)
                nudge_mask[self.output_indices] = 1.0
                
                # Initialize target to output suppression (strong negative to force decision)
                nudge_target = torch.ones(self.num_nodes, device=self.device) * -0.8
                # Zero out the non-output nodes so we don't suppress the brain!
                nudge_target[:256] = 0.0 
                nudge_target[512:] = 0.0
    
                # Pull the correct answer UP strongly
                nudge_target[256 + target_byte] = 1.0
                
                # Lateral Inhibition Mask (Output Nodes Only)
                inhib_mask = torch.zeros(self.num_nodes, device=self.device)
                inhib_mask[self.output_indices] = 1.0
                
                # Run Nudged (with Lateral Inhibition)
                # Apply inhibition_beta=1.0 to enforce Winner-Take-All
                # Run Nudged (with Lateral Inhibition)
                # Apply inhibition_beta=1.0 to enforce Winner-Take-All
                # Attention: High attention during supervised phase?
                self.engine.settle(input_vec, nudge_target=nudge_target, beta=beta, nudge_mask=nudge_mask, input_mask=input_mask, inhibition_mask=inhib_mask, inhibition_beta=1.0, attention_factor=1.2)
                state_nudged = self.engine.state.clone()
                
                # 5. Weight Update (EqProp)
                self.engine.update_weights_eq_prop(state_free, state_nudged, beta, lr)
            
            if step % 100 == 0:
                 # Rolling Acc
                 if len(acc_window) > 100: acc_window = acc_window[-100:]
                 if len(acc_top3_window) > 100: acc_top3_window = acc_top3_window[-100:]
                 
                 rolling_acc = sum(acc_window) / len(acc_window) if len(acc_window) > 0 else 0.0
                 rolling_acc3 = sum(acc_top3_window) / len(acc_top3_window) if len(acc_top3_window) > 0 else 0.0
                 
                 elapsed = time.time() - start_time
                 print(f"Step {step}/{total_steps} | Time: {elapsed:.0f}s | Loss: {loss:.4f} | Roll Acc: {rolling_acc:.2%} | Top3: {rolling_acc3:.2%}", end='\r')
                 
        final_acc = sum(acc_window)/len(acc_window) if len(acc_window) > 0 else 0.0
        final_acc3 = sum(acc_top3_window)/len(acc_top3_window) if len(acc_top3_window) > 0 else 0.0
        print(f"\nPhase Complete. Avg Loss: {loss_accum/total_steps:.4f} | Final Roll Acc: {final_acc:.2%} | Final Top3: {final_acc3:.2%}")
        
    def generate(self, start_text="The", length=100):
        print(f"\n--- Generating: {start_text} ... ---")
        curr_text = start_text
        
        # Prime
        for char in start_text:
            val = ord(char)
            if val > 255: val = 0
            input_vec = torch.zeros(self.num_nodes, device=self.device)
            input_vec[val] = 1.0
            self.engine.settle(input_vec, max_steps=30)
            
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
            input_vec[next_byte] = 1.0
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
    # Increased Beta and LR for stronger learning signal
    # Increased iterations for optimization (500)
    trainer.train_phase("Chars", "ndcd/data/level1_chars.txt", iterations=500, steps_per_iter=100, beta=1.0, lr=0.1, use_rl=False)
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
