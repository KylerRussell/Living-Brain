import torch
import numpy as np
import scipy.sparse as sp
from ndcd.graph import DynamicGraph
from ndcd.engine_torch import PredictiveCodingEngine

print("Starting SR test...")
graph = DynamicGraph(num_nodes=2000, num_modules=20)
indices, values = graph.export_sparse_components()

W = sp.csr_matrix((values, (indices[0], indices[1])), shape=(2000, 2000))
try:
    from scipy.sparse.linalg import eigs
    eigvals = eigs(W.astype(np.float64), k=1, which='LM', return_eigenvectors=False)
    sr = np.abs(eigvals[0])
    print(f"Graph exported Full SR: {sr}")
except Exception as e:
    print(f"Error computing full SR: {e}")

# What if we just take free_sparse alone?
input_proj_mask = (indices[0] < 256) | (indices[1] < 256)
free_mask = ~input_proj_mask

W_free = sp.csr_matrix((values[free_mask], (indices[0][free_mask], indices[1][free_mask])), shape=(2000, 2000))
try:
    eigvals = eigs(W_free.astype(np.float64), k=1, which='LM', return_eigenvectors=False)
    print(f"Graph exported Free SR: {np.abs(eigvals[0])}")
except Exception as e:
    print(f"Error computing free SR: {e}")

# Let's check engine initial SR
try:
    engine = PredictiveCodingEngine(2000, indices, values, graph.module_ranges, graph.module_levels, graph.taus, "cpu")
    w = engine.effective_weights.cpu().numpy()
    W_eng = sp.csr_matrix((w, (indices[0], indices[1])), shape=(2000, 2000))
    eigvals = eigs(W_eng.astype(np.float64), k=1, which='LM', return_eigenvectors=False)
    print(f"Engine SR (Effective Weights): {np.abs(eigvals[0])}")
except Exception as e:
    print(f"Error computing Engine SR: {e}")
