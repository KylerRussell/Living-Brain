
import numpy as np
import matplotlib.pyplot as plt
from graph import DynamicGraph
from engine import DragonEngine

def calculate_energy(graph, activation_func, input_vector):
    """
    Calculates the Hopfield Energy:
    E = 0.5 * sum(s^2) - 0.5 * sum(W_ij * rho(s_i) * rho(s_j)) - sum(b_i * rho(s_i)) - sum(I_i * rho(s_i))
    """
    s = graph.states
    rho = activation_func(s)
    
    term1 = 0.5 * np.sum(s**2)
    term2 = 0.5 * np.sum(graph.weights * np.outer(rho, rho))
    term3 = np.sum(graph.biases * rho)
    term4 = np.sum(input_vector * rho)
    
    return term1 - term2 - term3 - term4

def run_experiment():
    print("Initializing Non-Directional Continuous Dragon...")
    
    # 1. Setup
    num_nodes = 50
    graph = DynamicGraph(num_nodes=num_nodes, m_edges=2, p_triad=0.2, seed=42)
    engine = DragonEngine(graph, dt=0.001)
    
    print(f"Graph generated: {num_nodes} nodes.")
    
    # 2. Phase 1: Free Settling (Energy Minimization)
    print("\n[Phase 1] Free Settling (Dreaming)...")
    
    # Random sensory input
    sensory_input = np.zeros(num_nodes)
    sensory_input[:5] = np.random.uniform(-1, 1, 5) # Stimulate first 5 nodes
    
    # Track energy
    energies = []
    
    # Run settling manually to track energy
    initial_s = graph.states.copy()
    current_s = initial_s
    
    for t in range(500):
        current_s = engine.rk4_step(current_s, sensory_input)
        graph.states = current_s # Update graph state for energy calc
        e = calculate_energy(graph, engine.activation_function, sensory_input)
        energies.append(e)
        
    print("Settling complete.")
    print(f"Initial Energy: {energies[0]:.4f}")
    print(f"Final Energy: {energies[-1]:.4f}")
    
    if energies[-1] < energies[0]:
        print("SUCCESS: Energy decreased over time.")
    else:
        print("WARNING: Energy did not decrease. Check dynamics parameters.")

    # 3. Phase 2: Learning (Equilibrium Propagation)
    print("\n[Phase 2] Learning (EqProp)...")
    
    state_free = graph.states.copy()
    
    # Nudged phase: "Teach" the output nodes (last 5 nodes) to be active
    target = np.zeros(num_nodes)
    target[-5:] = 1.0 # Target outputs to be 1.0 (Active)
    
    # We create a specific nudge mask or just pass the full target vector 
    # and let the engine handle the force (assuming beta handles magnitude)
    
    print("Running Nudged Phase...")
    # Reset state to free state (or continue? EqProp usually continues)
    # Continuing from free state is better for infinitesimal diff
    engine.settle(sensory_input, duration_steps=100, nudge_target=target, beta=0.1)
    state_nudged = graph.states.copy()
    
    # Update weights
    print("Updating Weights...")
    old_weights_norm = np.linalg.norm(graph.weights)
    engine.update_weights_eq_prop(state_free, state_nudged, beta=0.1, learning_rate=0.01)
    new_weights_norm = np.linalg.norm(graph.weights)
    
    print(f"Weights updated. Norm change: {new_weights_norm - old_weights_norm:.6f}")
    
    if abs(new_weights_norm - old_weights_norm) > 1e-9:
        print("SUCCESS: Weights changed.")
    else:
        print("FAILURE: Weights did not change.")

if __name__ == "__main__":
    run_experiment()
