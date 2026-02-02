
import networkx as nx
import numpy as np

class DynamicGraph:
    def __init__(self, num_nodes, m_edges=2, p_triad=0.1, seed=None):
        """
        Initializes the Substrate using Holme-Kim Scale-Free Small-World model.
        
        Args:
           num_nodes: Total neurons in the brain.
           m_edges: Number of edges to add per new node (sparsity control).
           p_triad: Probability of forming a triangle (Clustering control).
           seed: Random seed for reproducibility.
        """
        self.num_nodes = num_nodes
        if seed is not None:
            np.random.seed(seed)
            
        # Generate topology using Holme-Kim algorithm
        # m must be <= m_edges in original paper, here networkx uses m as number of random edges
        self.nx_graph = nx.powerlaw_cluster_graph(n=num_nodes, m=m_edges, p=p_triad, seed=seed)
        
        # --- 1. Spatial Embedding (Section 2.2) ---
        # Assign random 3D coordinates in [0,1]^3
        self.pos = np.random.rand(num_nodes, 3)
        
        # Wiring Cost / Pruning
        # We want to minimize long-distance connections.
        # Simple heuristic: Prune top 20% longest edges and rewire to nearest neighbors?
        # Or just prune. Let's prune for now to simple enforce "Wiring Cost".
        # Better: Rewire long edges to closer nodes to maintain connectivity.
        
        # Calculate edge lengths
        edges = list(self.nx_graph.edges())
        edge_lengths = []
        for u, v in edges:
            dist = np.linalg.norm(self.pos[u] - self.pos[v])
            edge_lengths.append((dist, u, v))
            
        # Sort by length
        edge_lengths.sort(key=lambda x: x[0])
        
        # Threshold: Prune edges longer than 0.5 (in unit cube)
        # This is aggressive. Let's say top 10% are "too expensive" -> rewire.
        num_prune = int(len(edges) * 0.1)
        long_edges = edge_lengths[-num_prune:]
        
        for _, u, v in long_edges:
            if self.nx_graph.has_edge(u, v):
                self.nx_graph.remove_edge(u, v)
                # Rewire u to a closer node w (that isn't v and not already connected)
                # Find nearest neighbors of u
                # This could be slow for large N. Just random re-attach to a spatially close candidate?
                # Optimization: Just pick a random node w, if dist(u,w) < dist(u,v), connect.
                for _ in range(5): # Try 5 times
                    w = np.random.randint(0, num_nodes)
                    if w != u and not self.nx_graph.has_edge(u, w):
                         d_new = np.linalg.norm(self.pos[u] - self.pos[w])
                         if d_new < 0.5: # Arbitrary "close" threshold
                             self.nx_graph.add_edge(u, w)
                             break

        # --- 4. Node Classification (Section 2.3) ---
        # Explicit Roles
        # Define ratios: 10% Sensory, 10% Motor, 80% Association
        n_sensory = int(num_nodes * 0.1)
        n_motor = int(num_nodes * 0.1)
        n_association = num_nodes - n_sensory - n_motor
        
        self.sensory_indices = np.arange(0, n_sensory)
        self.motor_indices = np.arange(n_sensory, n_sensory + n_motor)
        self.association_indices = np.arange(n_sensory + n_motor, num_nodes)
        
        # Initialize Sparse Weight Matrix
        # W must be symmetric for Equilibrium Propagation energy definition
        self.weights = np.zeros((num_nodes, num_nodes))
        adj = nx.to_numpy_array(self.nx_graph)
        
        # Initialize weights with small random values, symmetric
        random_weights = np.random.uniform(-0.1, 0.1, (num_nodes, num_nodes))
        # Mask with adjacency to maintain sparsity
        self.weights = adj * random_weights
        self.weights = (self.weights + self.weights.T) / 2 # Enforce symmetry
        
        # Eligibility Traces Matrix (for RL)
        self.traces = np.zeros((num_nodes, num_nodes))
        
        # Initialize Neuron Parameters (Heterogeneous Tau)
        # Fast (Sensory/Motor) vs Slow (Association)
        self.taus = np.ones(num_nodes) * 1.0 # Default Slow (1000ms = 1.0s)
        
        # Assign Fast Tau to Sensory and Motor
        self.taus[self.sensory_indices] = 0.01 # 10ms
        self.taus[self.motor_indices] = 0.01   # 10ms
        
        # State Vectors
        self.states = np.zeros(num_nodes) # Internal potential s
        self.biases = np.zeros(num_nodes)
        
    def export_sparse_components(self):
        """
        Exports the graph weights as sparse components (indices, values).
        Returns:
            indices (np.ndarray): 2xE array of edge indices.
            values (np.ndarray): 1xE array of edge weights.
        """
        # Get indices of non-zero weights
        rows, cols = np.nonzero(self.weights)
        values = self.weights[rows, cols]
        indices = np.stack([rows, cols])
        return indices, values
