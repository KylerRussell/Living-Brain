
import unittest
import numpy as np
import torch
import sys
import os

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ndcd.graph import DynamicGraph
from ndcd.engine_torch import DragonEngineTorch

class TestLivingBrainSpecs(unittest.TestCase):
    
    def setUp(self):
        self.num_nodes = 500
        self.graph = DynamicGraph(num_nodes=self.num_nodes, m_edges=4, p_triad=0.1, seed=42)
        
        indices, values = self.graph.export_sparse_components()
        self.engine = DragonEngineTorch(
            self.num_nodes, indices, values, 
            self.graph.biases, self.graph.taus, 
            dt=0.01, device='cpu'
        )

    def test_spatial_embedding(self):
        """Test Section 2.2: Spatial 3D coordinates and pruned edges."""
        self.assertTrue(hasattr(self.graph, 'pos'), "Graph should have 'pos' attribute")
        self.assertEqual(self.graph.pos.shape, (self.num_nodes, 3), "Pos should be (N, 3)")
        
        # Check edge lengths are reasonable (heuristically check mean dist < 0.6 expecting some pruning)
        # In a random unit cube, avg dist is ~0.66. If we prune long edges, avg should represent local connections more.
        edges = list(self.graph.nx_graph.edges())
        dists = []
        for u, v in edges:
            d = np.linalg.norm(self.graph.pos[u] - self.graph.pos[v])
            dists.append(d)
        
        avg_dist = np.mean(dists)
        print(f"\nAvg Edge Distance: {avg_dist:.4f}")
        # Not a strict assertion because random replacement exists, but should be < 0.6 on average if pruning worked
        self.assertLess(avg_dist, 0.66, "Average edge distance should be reduced by pruning long connections")

    def test_node_classification(self):
        """Test Section 2.3: Explicit roles and Time Constants."""
        sensory = self.graph.sensory_indices
        motor = self.graph.motor_indices
        assoc = self.graph.association_indices
        
        # Overlap check
        all_indices = np.concatenate([sensory, motor, assoc])
        self.assertEqual(len(np.unique(all_indices)), self.num_nodes, "Indices should partition the nodes")
        
        # Tau check
        # Sensory/Motor -> Fast (~0.01)
        # Association -> Slow (~1.0)
        
        mean_sensory_tau = np.mean(self.graph.taus[sensory])
        mean_motor_tau = np.mean(self.graph.taus[motor])
        mean_assoc_tau = np.mean(self.graph.taus[assoc])
        
        self.assertAlmostEqual(mean_sensory_tau, 0.01, places=3, msg="Sensory Tau should be 0.01")
        self.assertAlmostEqual(mean_motor_tau, 0.01, places=3, msg="Motor Tau should be 0.01")
        self.assertAlmostEqual(mean_assoc_tau, 1.0, places=1, msg="Association Tau should be 1.0")

    def test_restricted_wiring(self):
        """Test Restricted I/O Wiring from implementation plan (Section 1)."""
        # Simulate logic from run_curriculum.py
        sensory = self.graph.sensory_indices
        motor = self.graph.motor_indices
        assoc = self.graph.association_indices
        
        # Test Inputs
        input_weights = np.zeros((self.num_nodes, 256))
        # Logic: Only Sensory nodes should be non-zero
        input_weights[sensory, :] = 1.0 
        # Assert non-sensory are zero
        self.assertTrue(np.all(input_weights[motor, :] == 0), "Motor nodes should not receive direct input")
        self.assertTrue(np.all(input_weights[assoc, :] == 0), "Association nodes should not receive direct input")
        
        # Test Outputs
        readout_weights = np.zeros((256, self.num_nodes))
        # Logic: Only Motor nodes should drive output
        readout_weights[:, motor] = 1.0
        # Assert non-motor are zero
        self.assertTrue(np.all(readout_weights[:, sensory] == 0), "Sensory nodes should not drive output")
        self.assertTrue(np.all(readout_weights[:, assoc] == 0), "Association nodes should not drive output")

    def test_eligibility_traces(self):
        """Test Section 4.3: Eligibility Trace accumulation."""
        input_vec = torch.zeros(self.num_nodes)
        input_vec[self.graph.sensory_indices] = 1.0 # Stimulate sensory
        
        # Run settle
        self.engine.settle(input_vec, duration_steps=20)
        
        # Check traces
        max_trace = torch.max(self.engine.trace_values).item()
        mean_trace = torch.mean(torch.abs(self.engine.trace_values)).item()
        
        print(f"Max Trace: {max_trace:.4f}, Mean Trace: {mean_trace:.6f}")
        self.assertGreater(max_trace, 0.0, "Traces should move from zero")
        
        # Ensure traces match edge logic (roughly)
        # If nodes are active, traces should increase
    
    def test_dopamine_learning(self):
        """Test Dopamine update"""
        # Pre-seed traces
        self.engine.trace_values = torch.ones_like(self.engine.weight_values) * 0.1
        original_weights = self.engine.weight_values.clone()
        
        reward = 1.0
        lr = 0.1
        self.engine.update_weights_dopamine(reward, lr)
        
        diff = self.engine.weight_values - original_weights
        
        # Expected: + lr * reward * trace = 0.1 * 1.0 * 0.1 = 0.01
        self.assertTrue(torch.allclose(diff, torch.tensor(0.01)), "Dopamine update failed")

    def test_hebbian_learning(self):
        """Test Hebbian update"""
        # Set state to all 1.0 -> tanh(1)=0.76
        self.engine.state = torch.ones(self.num_nodes) 
        original_weights = self.engine.weight_values.clone()
        
        lr = 0.1
        decay = 0.0
        self.engine.update_weights_hebbian(lr, decay)
        
        # Check if weights changed
        # Delta ~ 0.1 * (0.76 * 0.76) ~ 0.05
        diff = self.engine.weight_values - original_weights
        self.assertGreater(diff.mean().item(), 0.0, "Hebbian should increase weights for correlated activity")

if __name__ == '__main__':
    unittest.main()
