
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
        # 20% Fast (Sensory), 80% Slow (Association)
        self.taus = np.ones(num_nodes) * 0.1 # Default 100ms
        
        # Randomly select 20% of nodes to be "Fast"
        num_fast = int(num_nodes * 0.2)
        fast_indices = np.random.choice(num_nodes, num_fast, replace=False)
        self.taus[fast_indices] = 0.01 # 10ms for sensory
        
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
