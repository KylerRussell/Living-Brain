import numpy as np
import torch
import itertools

def generate_120_vertex_600_cell():
    """
    Generates the 120 vertices of the 600-cell polytope in 4D space.
    The vertices are given by:
    - 16 vertices: (±1, ±1, ±1, ±1)/2
    - 8 vertices: permutations of (±1, 0, 0, 0)
    - 96 vertices: even permutations of (±phi, ±1, ±1/phi, 0)/2
    """
    phi = (1 + np.sqrt(5)) / 2
    inv_phi = 1 / phi
    
    vertices = []
    
    # 1. 16 vertices: (+-0.5, +-0.5, +-0.5, +-0.5)
    for signs in itertools.product([-1, 1], repeat=4):
        vertices.append(np.array(signs) * 0.5)
        
    # 2. 8 vertices: permutations of (+-1, 0, 0, 0)
    for i in range(4):
        for sign in [-1, 1]:
            v = np.zeros(4)
            v[i] = sign
            vertices.append(v)
            
    # 3. 96 vertices: even permutations of (+-phi, +-1, +-inv_phi, 0) / 2
    base = [phi/2, 1/2, inv_phi/2, 0]
    for p in itertools.permutations([0,1,2,3]):
        # Check if permutation is even by counting inversions
        inv = sum(1 for i in range(4) for j in range(i+1, 4) if p[i] > p[j])
        if inv % 2 == 0:
            for signs in itertools.product([-1, 1], repeat=3):
                v = np.zeros(4)
                v[p[0]] = base[0] * signs[0]
                v[p[1]] = base[1] * signs[1]
                v[p[2]] = base[2] * signs[2]
                v[p[3]] = base[3] # 0 has no sign
                vertices.append(v)
                
    # Remove floating point duplicates
    unique_vertices = []
    for v in vertices:
        if not any(np.allclose(v, uv) for uv in unique_vertices):
            unique_vertices.append(v)
            
    V = np.array(unique_vertices)
    
    # Connect nearest neighbors (distance = 1/phi ≈ 0.618)
    dist_matrix = np.linalg.norm(V[:, None, :] - V[None, :, :], axis=2)
    dist_matrix[dist_matrix < 1e-5] = 100 # ignore self-distance
    
    min_dist = np.min(dist_matrix)
    # A threshold slightly larger than min_dist to account for floating point
    adjacency = (dist_matrix < min_dist + 1e-3).astype(np.float32)
    
    # The 600-cell vertices each have degree 12
    return V, adjacency

def get_geometric_target_weights(n_nodes, device='cpu'):
    """
    Creates a symmetric constraint field matrix. We tile the 120-vertex adjacency
    matrix to fit the required n_nodes subset (e.g., exactly 120 or a multiple).
    """
    _, adj = generate_120_vertex_600_cell() # shape (120, 120)
    
    # If n_nodes is smaller or larger, we can block-tile or truncate.
    # Ideally, we constrain exactly a multiple of 120 nodes.
    target_weights = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    
    n_tiles = n_nodes // 120
    remainder = n_nodes % 120
    
    # Tile it diagonally (disconnected 600-cell manifolds) or globally
    # For a unified constraint field, we add random cross-links or just tile.
    for i in range(n_tiles):
        start = i * 120
        end = start + 120
        target_weights[start:end, start:end] = adj
        
    # Truncated remainder (breaks perfect symmetry but allows flexible node counts)
    if remainder > 0:
        target_weights[n_tiles*120:, n_tiles*120:] = adj[:remainder, :remainder]
        
    return torch.tensor(target_weights, device=device)

def thermodynamic_sleep_phase(engine, geometric_indices, geometric_prior, sleep_steps=50, relaxation_rate=0.01):
    """
    Thermodynamic Reset: Pulls the network's diverging synaptic weights back
    toward the symmetrical 4D constraint field.
    """
    with torch.no_grad():
        # Find which edges in engine.indices connect nodes within geometric_indices
        # Create a boolean mask for nodes in geometric_indices
        in_geom = torch.zeros(engine.num_nodes, dtype=torch.bool, device=engine.device)
        in_geom[geometric_indices] = True
        
        # Mask for edges where both source and destination are in the geometric manifold
        edge_mask = in_geom[engine.indices[0]] & in_geom[engine.indices[1]]
        
        geom_edges = engine.indices[:, edge_mask]
        current_weights = engine.weight_values[edge_mask]
        
        # Map the 1D edge indices to the 2D geometric_prior matrix
        # First, create a reverse mapping from global node ID to geometric index ID
        global_to_geom = torch.full((engine.num_nodes,), -1, dtype=torch.long, device=engine.device)
        for i, idx in enumerate(geometric_indices):
            global_to_geom[idx] = i
            
        src_geom = global_to_geom[geom_edges[0]]
        dst_geom = global_to_geom[geom_edges[1]]
        
        # We only want to pull weights that exist in the engine's topology
        # target_weights is 1 if it's an edge in the 120-cell, 0 otherwise
        target_weights = geometric_prior[src_geom, dst_geom]
        
        # Scale target weights to match the current mean magnitude to maintain energy balance
        current_mag = current_weights.abs().mean()
        # To avoid collapse, if target is 0, we decay to 0; if target is 1, we pull to current_mag
        target_weights = target_weights * current_mag
        
        # Apply relaxation (geometric error correction)
        # Pull weights back
        geometric_error = target_weights - current_weights
        
        # Only update the base weight values (or the effective weights if you prefer)
        engine.weight_values[edge_mask] += relaxation_rate * sleep_steps * geometric_error
        
        # Clear out accumulating prediction errors (thermodynamic tension)
        # engine.zero_states() is not a method in this engine. It's usually handled per-trial,
        # but we can reset the state variables.
        engine.state.zero_()
        engine.state_basal.zero_()
        engine.state_apical.zero_()
        
    return geometric_error.abs().mean().item()