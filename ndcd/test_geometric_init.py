import torch
import numpy as np
from graph import DynamicGraph
from engine_torch import PredictiveCodingEngine
from geometric_constraint import get_geometric_target_weights, thermodynamic_sleep_phase

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_nodes = 2048 # Reduced for faster initialization in test
num_modules = 10
num_levels = 2

print(f"Building DynamicGraph with {num_nodes} nodes on {device}...")
graph = DynamicGraph(
    num_nodes,
    num_modules=num_modules,
    num_levels=num_levels,
)

indices, values = graph.export_sparse_components()
edge_index = torch.tensor(indices, dtype=torch.long)
edge_weight = torch.tensor(values, dtype=torch.float32)

actual_nodes = graph.num_nodes
biases = torch.zeros(actual_nodes, dtype=torch.float32)
taus = torch.ones(actual_nodes, dtype=torch.float32) * 20.0

is_inhibitory = graph.is_inhibitory
is_pv = graph.is_pv
is_sst = graph.is_sst
is_vip = graph.is_vip
is_lts = getattr(graph, 'is_lts', None)

mod_levels = np.array([m["level"] for m in graph.modules])

print("Initializing Engine...")
engine = PredictiveCodingEngine(
    num_nodes=actual_nodes,
    indices=edge_index.numpy(),
    values=edge_weight.numpy(),
    biases=biases.numpy(),
    taus=taus.numpy(),
    module_ranges=graph.get_module_ranges(),
    module_levels=mod_levels,
    hier_pairs=graph.hier_pairs,
    modules=graph.modules,
    device=device,
    is_inhibitory=is_inhibitory,
    is_pv=is_pv,
    is_sst=is_sst,
    is_vip=is_vip,
    is_lts=is_lts,
    dg_indices=graph.dg_indices,
    temporal_alpha=0.05,
    gc_indices=graph.granule_cell_indices,
    purkinje_indices=graph.purkinje_indices,
)

print("Initializing Geometric Constraint Field (600 nodes)...")
geom_idx_list = []
for mod in graph.modules:
    geom_idx_list.extend(mod['l56_indices'].tolist())
    if len(geom_idx_list) >= 600:
        break

if len(geom_idx_list) < 600:
    print("WARNING: Not enough L5/6 nodes to form 600-node geometric manifold.")
    geometric_indices = torch.tensor(geom_idx_list, dtype=torch.long, device=device)
    geometric_prior = get_geometric_target_weights(len(geom_idx_list), device=device)
else:
    geometric_indices = torch.tensor(geom_idx_list[:600], dtype=torch.long, device=device)
    geometric_prior = get_geometric_target_weights(600, device=device)

print(f"Prior shape: {geometric_prior.shape}")
print(f"Indices shape: {geometric_indices.shape}")

print("Testing thermodynamic sleep phase...")
error = thermodynamic_sleep_phase(engine, geometric_indices, geometric_prior, sleep_steps=10, relaxation_rate=0.05)
print(f"Geometric Error Correction applied. Tension: {error:.6f}")

print("Success!")
