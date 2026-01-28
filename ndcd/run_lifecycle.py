
import numpy as np
from graph import DynamicGraph
from engine import DragonEngine
from curriculum import Curriculum

class SimpleBanditEnv:
    """
    A simple 2-armed bandit environment for Imitation/RL phase.
    Action 0: Reward -1 (Punishment)
    Action 1: Reward +1 (Reward)
    Action 2+: Reward 0 (Neutral)
    """
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        
    def get_observation(self):
        # Returns a random noise vector as "context"
        obs = np.zeros(self.num_nodes)
        obs[:5] = np.random.uniform(0, 0.5, 5) # Weak context
        return obs
        
    def step(self, action):
        if action == 0:
            return -1.0 # Bad action
        elif action == 1:
            return 1.0 # Good action
        else:
            return 0.0

def generate_grounding_dataset(num_nodes, size=20):
    """
    Generates a simple association task dataset.
    Pattern A (Nodes 0-5 active) -> Symbol A (Node -1 active)
    Pattern B (Nodes 5-10 active) -> Symbol B (Node -2 active)
    """
    dataset = []
    for _ in range(size):
        # Class 0
        input_vec = np.zeros(num_nodes)
        input_vec[0:5] = np.random.uniform(0.8, 1.0, 5) # Pattern A
        target_vec = np.zeros(num_nodes)
        target_vec[-1] = 1.0 # Symbol A
        dataset.append((input_vec, target_vec))
        
        # Class 1
        input_vec = np.zeros(num_nodes)
        input_vec[5:10] = np.random.uniform(0.8, 1.0, 5) # Pattern B
        target_vec = np.zeros(num_nodes)
        target_vec[-2] = 1.0 # Symbol B
        dataset.append((input_vec, target_vec))
        
    return dataset

def main():
    print("=== Initializing Non-Directional Continuous Dragon ===")
    num_nodes = 100
    graph = DynamicGraph(num_nodes=num_nodes, m_edges=3, p_triad=0.2, seed=1337)
    # Use small dt for stability
    engine = DragonEngine(graph, dt=0.01)
    curriculum = Curriculum(engine)
    
    print("\n=== Phase 1: Babbling (Unsupervised) ===")
    # Spontaneous activity shapes the attractors
    curriculum.run_babbling(steps=500, learning_rate=0.005)
    
    print("\n=== Phase 2: Grounding (Supervised) ===")
    # Bind sensory patterns to abstract symbols
    dataset = generate_grounding_dataset(num_nodes)
    curriculum.run_grounding(dataset, epochs=5, beta=0.1, learning_rate=0.01)
    
    print("\n=== Phase 3: Imitation (Reinforcement) ===")
    # Learn to pick the right action (Action 1)
    env = SimpleBanditEnv(num_nodes)
    curriculum.run_imitation(env, steps=500, learning_rate=0.05)
    
    print("\n=== Lifecycle Complete ===")
    print("The Dragon has been raised.")

if __name__ == "__main__":
    main()
