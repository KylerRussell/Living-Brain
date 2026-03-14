import torch
import numpy as np
from engine_torch import PredictiveCodingEngine
from graph import DynamicGraph

print("Testing environment...")
device = torch.device("cpu")
num_nodes = 3500
num_modules = 10
graph = DynamicGraph(num_nodes, num_modules=num_modules)

indices, values = graph.export_sparse_components()
biases = np.zeros(num_nodes)
taus = np.ones(num_nodes) * 5.0

print("Initializing engine...")
engine = PredictiveCodingEngine(
    num_nodes=num_nodes,
    indices=indices,
    values=values,
    biases=biases,
    taus=taus,
    module_ranges=graph.get_module_ranges(),
    module_levels=np.zeros(num_modules),
    hier_pairs=[],
    modules=graph.modules,
    device=device,
    is_inhibitory=graph.is_inhibitory,
    is_pv=graph.is_pv,
    is_sst=graph.is_sst,
    is_vip=graph.is_vip,
    temporal_alpha=0.05
)

print("Running one settle step...")
input_vec = torch.zeros(num_nodes)
input_vec[0] = 10.0
engine.settle(input_vec, max_steps=10, tol=0.001)

print(f"Success! Firing rate: {engine.lifetime_firing.mean().item():.4f}")
