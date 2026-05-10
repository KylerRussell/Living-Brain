mod_levels = np.array([m["level"] for m in graph.modules])

print("Initializing Engine...")
engine = PredictiveCodingEngine(
    num_nodes=num_nodes,
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
