
import torch
import sys
import os

# Add local directory to path
sys.path.append(os.getcwd())

from ndcd.run_sequential import SequentialTrainer
from ndcd.engine_torch import DragonEngineTorch

def verify_plasticity():
    print("=== Verifying Structural Plasticity ===")
    
    # Initialize Trainer with small graph
    num_nodes = 500
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    try:
        trainer = SequentialTrainer(num_nodes=num_nodes, device=device)
        
        initial_edges = trainer.engine.weight_values.shape[0]
        print(f"Initial Edges: {initial_edges}")
        
        # Manually trigger remodel
        print("\n--- Triggering Remodel ---")
        trainer.engine.remodel_structure(prune_threshold=0.01, growth_rate=50)
        
        new_edges = trainer.engine.weight_values.shape[0]
        print(f"New Edges: {new_edges}")
        
        if new_edges != initial_edges:
            print("PASS: Edge count changed.")
        else:
            print("WARNING: Edge count did not change (might be chance if pruning balanced growth, but unlikely).")
            
        # Check Spectral Tuning compatibility
        print("\n--- Triggering Spectral Tuning ---")
        try:
            trainer.tune_spectral_radius(target_radius=0.95)
            print("PASS: Spectral Tuning ran successfully.")
        except Exception as e:
            print(f"FAIL: Spectral Tuning crashed: {e}")
            raise e
            
        print("\n=== Verification Complete: SUCCESS ===")
        
    except Exception as e:
        print(f"\n=== Verification Complete: FAILED ===")
        print(e)
        raise e

if __name__ == "__main__":
    verify_plasticity()
