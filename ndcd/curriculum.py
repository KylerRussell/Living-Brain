
import numpy as np

class Curriculum:
    def __init__(self, engine):
        self.engine = engine
        self.graph = engine.graph
        
    def run_babbling(self, steps=1000, learning_rate=0.01, decay=0.01):
        """
        Phase 1: Babbling (Unsupervised Self-Organization)
        Learns statistical structure of input space via Hebbian learning.
        """
        print(f"--- Starting Phase 1: Babbling ({steps} steps) ---")
        num_sensory = int(self.graph.num_nodes * 0.2) # First 20% are sensory
        
        for t in range(steps):
            # 1. Random Input (White Noise)
            sensory_input = np.zeros(self.graph.num_nodes)
            sensory_input[:num_sensory] = np.random.uniform(-1, 1, num_sensory)
            
            # 2. Settle (Dreaming/Perception)
            # Short settling time for babbling
            self.engine.settle(sensory_input, duration_steps=50)
            
            # 3. Hebbian Update
            self.engine.update_weights_hebbian(self.graph.states, learning_rate, decay)
            
            if t % 100 == 0:
                print(f"Babbling Step {t}/{steps}")

    def run_grounding(self, dataset, epochs=10, beta=0.1, learning_rate=0.01):
        """
        Phase 2: Grounding (Supervised Symbol Association)
        Uses Equilibrium Propagation to map inputs to targets.
        
        Args:
            dataset: List of tuples (input_vector, target_vector)
            input_vector: Full node vector with sensory nodes set
            target_vector: Full node vector with symbol/output nodes set
        """
        print(f"--- Starting Phase 2: Grounding ({epochs} epochs) ---")
        
        for epoch in range(epochs):
            total_error = 0
            for idx, (input_vec, target_vec) in enumerate(dataset):
                # 1. Phase 1: Free Phase (Dreaming)
                # Clamp input only
                self.engine.settle(input_vec, duration_steps=50)
                state_free = self.graph.states.copy()
                
                # Check error (just for logging)
                # Assuming target_vec is 0 where not target, we check deviation on target nodes
                # Identify target indices (non-zero entries in target_vec)
                target_indices = np.where(target_vec != 0)[0]
                if len(target_indices) > 0:
                    output_vals = self.engine.activation_function(state_free)[target_indices]
                    target_vals = target_vec[target_indices]
                    error = np.mean((output_vals - target_vals)**2)
                    total_error += error
                
                # 2. Phase 2: Nudged Phase (Teaching)
                # Clamp input AND weakly clamp/nudge output
                # We assume settle supports nudge_target vector
                self.engine.settle(input_vec, duration_steps=20, nudge_target=target_vec, beta=beta)
                state_nudged = self.graph.states.copy()
                
                # 3. EqProp Update
                self.engine.update_weights_eq_prop(state_free, state_nudged, beta, learning_rate)
            
            avg_error = total_error / len(dataset)
            print(f"Grounding Epoch {epoch+1}/{epochs}, Average Error: {avg_error:.6f}")

    def run_imitation(self, environment, steps=1000, learning_rate=0.01):
        """
        Phase 3: Imitation (Reinforcement Learning)
        Uses Dopamine-modulated STDP.
        
        Args:
           environment: Object with step(action) -> reward method.
        """
        print(f"--- Starting Phase 3: Imitation ({steps} steps) ---")
        num_motor = 5 # Last 5 nodes are motor
        
        current_s = self.graph.states # Carry over state
        
        for t in range(steps):
            # 1. Perception / Action Generation
            # No specific clamped input, or environment provides observation
            observation = environment.get_observation() 
            # environment.get_observation() should return a vector of size num_nodes (or we map it)
            
            # Settle briefly to generate action
            self.engine.settle(observation, duration_steps=30)
            
            # Read motor output
            motor_activity = self.engine.activation_function(self.graph.states)[-num_motor:]
            action = np.argmax(motor_activity) # Discrete action logic for simplicity
            
            # 2. Interact with Environment
            reward = environment.step(action)
            
            # 3. Update Eligibility Traces
            # Trace accumulates Hebbian coincidence: dZ = -Z + rho * rho^T
            # We assume continuous update, but here we do discrete update step.
            rho = self.engine.activation_function(self.graph.states)
            self.graph.traces += 0.1 * (-self.graph.traces + np.outer(rho, rho))
            
            # 4. Dopamine Update
            self.engine.update_weights_dopamine(reward, learning_rate)
            
            if t % 100 == 0:
                 print(f"Imitation Step {t}/{steps}, Last Reward: {reward}")
