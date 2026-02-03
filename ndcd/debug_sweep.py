
import ray
import time
from ndcd.run_sequential_ray import RemoteTrainer

def run_sweep():
    ray.init(ignore_reinit_error=True)
    print("Starting Expanded Hyperparameter Sweep...")
    
    # Define Configurations
    configs = {
        "HighGain":   {"m": 20, "r": 3.0, "s": 5.0, "c": "hard", "lr": 0.1, "beta": 0.5},
        "SoftClamp":  {"m": 20, "r": 3.0, "s": 5.0, "c": "soft", "lr": 0.1, "beta": 0.5},
        "LowInput":   {"m": 20, "r": 3.0, "s": 1.0, "c": "hard", "lr": 0.1, "beta": 0.5},
        "Baseline":   {"m": 10, "r": 1.5, "s": 5.0, "c": "hard", "lr": 0.1, "beta": 0.5}
    }
    
    trainers = {}
    futures = {}
    
    # Initialize Actors (Smaller graph for speed)
    for name, cfg in configs.items():
        print(f"Initializing {name}...")
        trainers[name] = RemoteTrainer.remote(
            num_nodes=1000, 
            device='cpu',
            m_edges=cfg["m"],
            target_radius=cfg["r"],
            input_scale=cfg["s"],
            clamp_mode=cfg["c"]
        )
        
    # Launch training in parallel
    for name, trainer in trainers.items():
        cfg = configs[name]
        futures[name] = trainer.train_phase.remote(
            phase_name=f"Sweep_{name}",
            data_path="ndcd/data/level1_chars.txt",
            iterations=10, 
            steps_per_iter=100, 
            beta=cfg["beta"], 
            lr=cfg["lr"],
            use_rl=False
        )
        
    print("\nTraining Running... Waiting for results.\n")
    
    # Collect Results
    results = {}
    for name, future in futures.items():
        try:
            stats = ray.get(future)
            results[name] = stats
            print(f"Finished {name}: Acc={stats['acc']:.2%} | Act={stats['activity']:.4f} | W={stats['weights']:.4f}")
        except Exception as e:
            print(f"Failed {name}: {e}")
            
    print("\n=== Sweep Results ===")
    for name, stats in results.items():
        if isinstance(stats, dict):
            print(f"{name}: Acc={stats['acc']:.2%} | Act={stats['activity']:.4f}")
        else:
            print(f"{name}: Failed")
        
    ray.shutdown()

if __name__ == "__main__":
    run_sweep()
