
import torch
import sys
import os
import numpy as np

# Ensure we can import ndcd
sys.path.append(os.getcwd())

from ndcd.run_sequential import SequentialTrainer

class DebugTrainer(SequentialTrainer):
    def train_debug(self, phase_name, data_path, iterations, steps_per_iter, beta=0.1, lr=0.01):
        print(f"\n=== DEBUG Phase: {phase_name} ===")
        # Check data
        if not os.path.exists(data_path):
             from ndcd.run_sequential import ensure_data
             ensure_data(data_path, phase_name)
             
        with open(data_path, 'rb') as f:
            data = f.read()
        
        print(f"Data length: {len(data)}")
        print(f"First 10 bytes: {list(data[:10])}")
        
        curr_idx = 0
        
        # We'll run just a few steps
        total_steps = 10 
        
        for step in range(total_steps):
            if curr_idx >= len(data) - 1: curr_idx = 0
            
            input_byte = data[curr_idx]
            target_byte = data[curr_idx + 1]
            curr_idx += 1
            
            print(f"\n--- Step {step} ---")
            print(f"Input Byte: {input_byte} ('{chr(input_byte) if 32<=input_byte<127 else '?'}')")
            print(f"Target Byte: {target_byte} ('{chr(target_byte) if 32<=target_byte<127 else '?'}')")
            
            # Input Setup
            input_mask = torch.zeros(self.num_nodes, device=self.device)
            input_mask[self.input_indices] = 1.0
            input_vec = torch.zeros(self.num_nodes, device=self.device)
            input_vec[input_byte] = 1.0 
            
            # Free Phase
            print("Running Free Phase...")
            self.engine.settle(input_vec, input_mask=input_mask)
            state_free = self.engine.state.clone()
            
            # Debug State
            print(f"State Statistics: Min={state_free.min():.4f}, Max={state_free.max():.4f}, Mean={state_free.mean():.4f}")
            print(f"State StdDev: {state_free.std():.4f}")
            
            # Debug Output
            output_activity = state_free[256:512]
            print(f"Output Activity: Min={output_activity.min():.4f}, Max={output_activity.max():.4f}, Mean={output_activity.mean():.4f}")
            
            probs = torch.softmax(output_activity, dim=0)
            pred_idx = torch.argmax(probs).item()
            prob_target = probs[target_byte].item()
            prob_pred = probs[pred_idx].item()
            
            print(f"Prediction: {pred_idx} (Prob: {prob_pred:.4f})")
            print(f"Target Prob: {prob_target:.4f}")
            print(f"Correct: {pred_idx == target_byte}")
            
            # Nudge Phase to see if gradients work
            nudge_mask = torch.zeros(self.num_nodes, device=self.device)
            nudge_mask[self.output_indices] = 1.0
            nudge_target = torch.ones(self.num_nodes, device=self.device) * -0.1 
            nudge_target[:256] = 0.0 
            nudge_target[512:] = 0.0
            nudge_target[256 + target_byte] = 1.0
            
            self.engine.settle(input_vec, nudge_target=nudge_target, beta=beta, nudge_mask=nudge_mask, input_mask=input_mask)
            state_nudged = self.engine.state.clone()
            
            diff = state_nudged - state_free
            print(f"Nudge Effect (Mean Abs Diff): {diff.abs().mean():.6f}")
            
            # Update
            self.engine.update_weights_eq_prop(state_free, state_nudged, beta, lr)

def main():
    # Use CPU to avoid issues if GPU is busy, or maybe GPU if user is using it.
    # User used GPU. I'll use CPU for debug unless it crashes.
    device = 'cpu'
    if torch.cuda.is_available(): device = 'cuda'
    
    print(f"Using device: {device}")
    
    trainer = DebugTrainer(num_nodes=2000, device=device) # Smaller graph for debug
    
    # Run a bit of babbling
    trainer.train_babbling(iterations=50) # Short warmup
    trainer.train_phase("Chars", "ndcd/data/level1_chars.txt", iterations=10, steps_per_iter=100, beta=0.5, lr=0.1, use_rl=False)
    
    # Run debug phase
    trainer.train_debug("Chars", "ndcd/data/level1_chars.txt", iterations=1, steps_per_iter=10, beta=0.5, lr=0.1)

if __name__ == "__main__":
    main()
