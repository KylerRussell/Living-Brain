
import torch
import numpy as np
import os
import time
import argparse
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
    def __init__(self, num_nodes=5000, device='cpu'):
        self.device = device
        self.num_nodes = num_nodes
        
        # 1. Initialize Graph
        print("Initializing Dynamic Graph...")
        self.graph = DynamicGraph(num_nodes=num_nodes, m_edges=10, p_triad=0.1, seed=42)
        indices, values = self.graph.export_sparse_components()
        biases = self.graph.biases
        taus = self.graph.taus
        
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
        
    def train_phase(self, phase_name, data_path, iterations, steps_per_iter, beta=0.1, lr=0.01):
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
            # We add external input current 'I'.
            # For clamping behavior in a dynamical system, we usually drive it hard 
            # or we just set the state? 
            # The engine adds 'input_vector' to the derivative. 
            # To "clamp", we can provide a strong input.
            
            input_vec = torch.zeros(self.num_nodes, device=self.device)
            # One-hot input current
            input_vec[input_byte] = 5.0 # Strong input to drive the node high
            
            # 3. Free Phase (Dream)
            # Run dynamics. Result is state_free.
            # We do NOT reset state. We continue from where we left off.
            self.engine.settle(input_vec, duration_steps=15)
            state_free = self.engine.state.clone()
            
            # Measure Prediction (Readout) during Free Phase
            # Output is nodes 256-511
            output_activity = state_free[256:512]
            probs = torch.softmax(output_activity, dim=0)
            pred_idx = torch.argmax(probs).item()
            
            if pred_idx == target_byte:
                correct_count += 1
                
            loss = -torch.log(probs[target_byte] + 1e-8).item()
            loss_accum += loss
            
            # 4. Nudged Phase (Reality)
            # We want to pull the Output Nodes towards the target.
            # Nudge Target: A vector size [N].
            # We only care about nudging nodes 256-511.
            # Target for these nodes is One-Hot(target_byte).
            # But 'Target' in EquilProp usually means the desired state configuration.
            # A one-hot target means we want target_node=1.0, others=-1.0 (tanh)?
            # Or just nudge towards correct class?
            
            # Construct Nudge Target Vector
            # We rely on engine's logic: nudge_force = beta * (nudge_target - rho_s)
            # So nudge_target should be the desired Activations (rho).
            # For output nodes: Target is 1.0 for correct char, -1.0 for others.
            
            # We need a mask to only nudge output nodes. 
            # The current engine applies nudge to ALL nodes if passed. 
            # We need to construct a target that is "current state" for hidden nodes?
            # NO, that stops them from changing.
            # We should probably modify engine or just pass zeros for others 
            # AND handle the beta masking there. 
            # OR simpler: The engine takes `nudge_target` and `beta`.
            # If we simply pass a nudge_target that has 0s for hidden, and we want hidden to be free...
            # The formula is F = beta * (T - rho). If T=0 and rho!=0, we pull hidden to 0. Bad.
            
            # WORKAROUND: In this script, we can't easily change engine logic without editing engine_torch.py.
            # I will modify engine_torch.py to accept a 'nudge_mask' or similar, 
            # OR I will just "read" the current hidden state, set it as target, 
            # effectively "clamping" them to stay same? No, Nudged phase must allow hidden to relax.
            
            # PROPER FIX: Modify engine_torch.py to allow nudging only specific nodes.
            # For now, let's assume I will fix engine_torch.py to handle sparse/masked nudging.
            # Let's say I pass a tuple or use a mask arg.
            # I'll stick to: modify engine to accept `nudge_mask`.
            
            nudge_target = torch.zeros(self.num_nodes, device=self.device)
            # Default everything to current state? No.
            # Default to 0? No.
            # Let's handle this by implementing `nudge_indices` in engine or similar.
            # Plan: Set target for output nodes 256-511.
            nudge_target[256 + target_byte] = 1.0 # Pull correct up
            # We might want to pull incorrect down? 
            # nudge_target[256:512] = -0.5
            # nudge_target[256 + target_byte] = 1.0
            
            # Assume I add 'nudge_mask' to settle().
            nudge_mask = torch.zeros(self.num_nodes, device=self.device)
            nudge_mask[256:512] = 1.0
            
            # Run Nudged
            self.engine.settle(input_vec, duration_steps=15, nudge_target=nudge_target, beta=beta, nudge_mask=nudge_mask)
            state_nudged = self.engine.state.clone()
            
            # 5. Weight Update (EqProp)
            # update ~ (rho_nudged * rho_nudged - rho_free * rho_free) ? No.
            # update ~ (rho_n_i * rho_n_j - rho_f_i * rho_f_j)
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
            self.engine.settle(input_vec, duration_steps=15)
            
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
            self.engine.settle(input_vec, duration_steps=15)
            
        print(curr_text)
        print("--------------------------------------")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--nodes", type=int, default=1000) # Small for testing, 5000 for real
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
    # Just run random inputs and Hebbian? Or just skip.
    # Let's skip for pure EqProp focus.
    
    # Phase 1: Chars
    trainer.train_phase("Chars", "ndcd/data/level1_chars.txt", iterations=100, steps_per_iter=100, beta=0.5, lr=0.1)
    trainer.generate()
    
    # Phase 2: Words
    trainer.train_phase("Words", "ndcd/data/level2_words.txt", iterations=200, steps_per_iter=100, beta=0.5, lr=0.05)
    trainer.generate()
    
    # Phase 3: Quotes
    trainer.train_phase("Quotes", "ndcd/data/level3_quotes.txt", iterations=200, steps_per_iter=200, beta=0.5, lr=0.02)
    trainer.generate()
    
    # Phase 4: Literature
    trainer.train_phase("Literature", "ndcd/data/sherlock.txt", iterations=500, steps_per_iter=500, beta=1.0, lr=0.01)
    trainer.generate(start_text="Sherlock", length=200)

if __name__ == "__main__":
    main()
