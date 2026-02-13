
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
        
        # Initialize Sparse Weight Matrix
        # W must be symmetric for Equilibrium Propagation energy definition
        self.weights = np.zeros((num_nodes, num_nodes))
        adj = nx.to_numpy_array(self.nx_graph)
        
        # Orthogonal Initialization for better gradient flow
        # We create a random orthogonal matrix and mask it
        # Since we need symmetry, we can use a symmetric orthogonal matrix or just
        # stabilize the random normal one.
        # Let's use standard normal scaled by 1/sqrt(connectivity) for now, then spectral norm.
        # But to be "Orthogonal-like", we want singular values ~ 1.
        
        # Random Normal Initialization (Xavier-like but sparse context)
        # Variance = 2 / (fan_in + fan_out)? For sparse, just 1/sqrt(k)
        # Average degree k ~ 2*m_edges
        k = 2 * m_edges
        std_dev = 1.0 / np.sqrt(k)
        
        random_weights = np.random.normal(0, std_dev, (num_nodes, num_nodes))
        
        # Mask with adjacency to maintain sparsity
        self.weights = adj * random_weights
        
        # Enforce Symmetry for EqProp
        self.weights = (self.weights + self.weights.T) / 2 
        
        # Enforce Spectral Radius = 0.95 (Edge of Chaos)
        try:
            current_radius = np.max(np.abs(np.linalg.eigvals(self.weights)))
            if current_radius > 0:
                self.weights *= (0.95 / current_radius)
        except:
             # If eigvals fail (too slow?), just normalize by frobenius?
             # For 2000 nodes, it should be fine.
             pass
        
        # Eligibility Traces Matrix (for RL)
        self.traces = np.zeros((num_nodes, num_nodes))
        
        # Initialize Neuron Parameters (Multi-timescale Tau Hierarchy)
        # Fast nodes handle bytes/chars, medium handle words, slow handle
        # sentences, ultra-slow handle topic/context.
        # Input/output nodes (0-511) are always fast.
        n_io = min(512, num_nodes)  # I/O nodes are always fast
        n_remaining = num_nodes - n_io

        # Distribute remaining nodes across timescales (cortical ratios)
        # Fast: ~30% of remaining, Medium: ~40%, Slow: ~20%, Ultra-slow: ~10%
        n_fast_extra = int(n_remaining * 0.3)
        n_medium = int(n_remaining * 0.4)
        n_slow = int(n_remaining * 0.2)
        n_ultra = n_remaining - n_fast_extra - n_medium - n_slow

        self.taus = np.concatenate([
            np.ones(n_io) * 0.1,              # I/O nodes: τ ≈ 2 steps effective
            np.ones(n_fast_extra) * 0.1,       # Extra fast association nodes
            np.ones(n_medium) * 0.75,          # τ ≈ 15 steps
            np.ones(n_slow) * 5.0,             # τ ≈ 100 steps
            np.ones(n_ultra) * 25.0,           # τ ≈ 500 steps
        ])
        
        # State Vectors
        self.states = np.zeros(num_nodes) # Internal potential s
        # Initialize biases with small noise to break dead states
        self.biases = np.random.uniform(-0.01, 0.01, num_nodes)
        
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
