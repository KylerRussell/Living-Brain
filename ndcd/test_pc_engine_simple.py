import os
import sys
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import random
from collections import deque
from engine_torch import PredictiveCodingEngine
from graph import DynamicGraph


def create_training_graph(
    num_modules=10, num_levels=2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Builds a smaller hierarchical graph suitable for simple PC engine testing.
    Adapts logic from micro_scale_diagnostics and DynamicGraph.
    """
    total_nodes = num_modules * 350  # Increased to 3500 nodes total

    # We need at least 512 nodes for I/O (256 in, 256 out)
    if total_nodes < 600:
        total_nodes = 600

    print(
        f"Building DynamicGraph with {total_nodes} nodes, {num_modules} modules, {num_levels} levels..."
    )
    # Use DynamicGraph to build the structure
    graph = DynamicGraph(
        total_nodes,
        m_edges=2,
        p_triad=0.1,
        num_modules=num_modules,
        num_levels=num_levels,
    )

    indices, values = graph.export_sparse_components()
    edge_index = torch.tensor(indices, dtype=torch.long)
    edge_weight = torch.tensor(values, dtype=torch.float32)

    # Default biases and taus
    biases = torch.zeros(total_nodes, dtype=torch.float32)
    taus = torch.ones(total_nodes, dtype=torch.float32) * 5.0  # default tau

    return edge_index, edge_weight, biases, taus, graph


def boost_output_connectivity(graph, edge_index, edge_weight, num_nodes):
    """
    Add additional output projections from ALL modules' L5/6 to motor nodes.
    The original graph only projects from Level 0 L5/6 → motor. This adds
    projections from ALL levels, giving the output nodes broader visibility.
    """
    n_sensory = 256
    n_motor = 256

    new_rows = []
    new_cols = []

    for mod in graph.modules:
        # Skip level 0 modules (already have output projections)
        if mod['level'] == 0:
            continue
        idx = mod['l56_indices']
        if len(idx) == 0:
            continue
        # Sparse projection: 10% of L5/6 nodes per output
        n_proj = max(1, len(idx) // 10)
        for m in range(n_motor):
            sources = np.random.choice(idx, n_proj, replace=True)
            motor_node = n_sensory + m
            new_rows.extend(sources.tolist())
            new_cols.extend([motor_node] * n_proj)

    if new_rows:
        new_rows = np.array(new_rows, dtype=np.int64)
        new_cols = np.array(new_cols, dtype=np.int64)

        # Initialize with small random weights
        new_weights = np.random.normal(0, 0.05, len(new_rows)).astype(np.float32)

        # Merge with existing
        old_indices = edge_index.numpy()
        old_weights = edge_weight.numpy()

        merged_rows = np.concatenate([old_indices[0], new_rows])
        merged_cols = np.concatenate([old_indices[1], new_cols])
        merged_weights = np.concatenate([old_weights, new_weights])

        # Deduplicate
        pairs = np.stack([merged_rows, merged_cols], axis=1)
        _, unique_idx = np.unique(pairs, axis=0, return_index=True)
        merged_rows = merged_rows[unique_idx]
        merged_cols = merged_cols[unique_idx]
        merged_weights = merged_weights[unique_idx]

        edge_index = torch.tensor(np.stack([merged_rows, merged_cols]), dtype=torch.long)
        edge_weight = torch.tensor(merged_weights, dtype=torch.float32)
        print(f"Output boost: added {len(new_rows)} new output projections from higher-level modules")

    return edge_index, edge_weight


def add_thalamocortical_loop(graph, edge_index, edge_weight, biases, taus, num_nodes,
                              n_thalamic=128, device='cpu'):
    """
    Thalamocortical stabilization loop (NOT an output pathway).

    L5/6 → Thalamic → L4 (same modules) → re-excites L5/6.
    Creates reverberating attractor dynamics that temporally smooth
    the L5/6 representation, solving the non-stationarity problem.

    Thalamic relay neurons:
    - Fast tau (5ms) for rapid relay
    - No recurrent connections (biological constraint)
    - Tonically inhibited (basal ganglia default state)
    - Reciprocal projections BACK to L5/6 (not to motor)
    """
    thal_start = num_nodes
    new_num_nodes = num_nodes + n_thalamic

    # Extend biases: TONIC INHIBITION (basal ganglia default state)
    # Relay neurons are silent by default, only fire when gate opens.
    # We bypass the loop for sequence generation by setting deep tonic inhibition (-50.0).
    thal_biases = torch.ones(n_thalamic, dtype=torch.float32) * -50.0  # Deep tonic inhibition
    thal_taus = torch.ones(n_thalamic, dtype=torch.float32) * 5.0     # Fast relay

    biases = torch.cat([biases, thal_biases])
    taus = torch.cat([taus, thal_taus])

    new_rows = []
    new_cols = []
    new_weights = []

    # Gather all L5/6 and L4 indices
    all_l56 = []
    all_l4 = []
    for mod in graph.modules:
        all_l56.extend(mod['l56_indices'].tolist())
        all_l4.extend(mod['l4_indices'].tolist())
    all_l56 = np.array(all_l56, dtype=np.int64)
    all_l4 = np.array(all_l4, dtype=np.int64)

    # 1. L5/6 → Thalamic (sparse, ~5% connectivity)
    n_proj_per_thal = max(1, len(all_l56) // 20)
    for t in range(n_thalamic):
        sources = np.random.choice(all_l56, n_proj_per_thal, replace=False)
        thal_node = thal_start + t
        for s in sources:
            new_rows.append(s)
            new_cols.append(thal_node)
            new_weights.append(np.random.normal(0.1, 0.05))

    # 2. Thalamic → L4 (reciprocal feedback, sparse)
    #    This is the cortico-thalamo-cortical loop that stabilizes representations.
    n_proj_per_thal_back = max(1, len(all_l4) // 20)
    for t in range(n_thalamic):
        thal_node = thal_start + t
        targets = np.random.choice(all_l4, n_proj_per_thal_back, replace=False)
        for tgt in targets:
            new_rows.append(thal_node)
            new_cols.append(tgt)
            new_weights.append(np.random.normal(0.05, 0.02))

    # NO thalamic → motor projections (that's the cerebellum's job now)
    # NO thalamic ↔ thalamic recurrence (biological constraint)

    new_rows = np.array(new_rows, dtype=np.int64)
    new_cols = np.array(new_cols, dtype=np.int64)
    new_weights = np.array(new_weights, dtype=np.float32)

    # Merge with existing edges
    old_indices = edge_index.numpy()
    old_weights = edge_weight.numpy()
    merged_rows = np.concatenate([old_indices[0], new_rows])
    merged_cols = np.concatenate([old_indices[1], new_cols])
    merged_weights = np.concatenate([old_weights, new_weights])

    pairs = np.stack([merged_rows, merged_cols], axis=1)
    _, unique_idx = np.unique(pairs, axis=0, return_index=True)
    merged_rows = merged_rows[unique_idx]
    merged_cols = merged_cols[unique_idx]
    merged_weights = merged_weights[unique_idx]

    edge_index = torch.tensor(np.stack([merged_rows, merged_cols]), dtype=torch.long)
    edge_weight = torch.tensor(merged_weights, dtype=torch.float32)

    print(f"Thalamocortical loop: {n_thalamic} relay neurons (stabilization only, no motor output)")
    thalamic_indices = np.arange(thal_start, thal_start + n_thalamic, dtype=np.int64)
    return edge_index, edge_weight, biases, taus, new_num_nodes, thalamic_indices


def add_cerebellar_module(graph, num_nodes, n_granule=16384, sparsity=0.05, device='cpu', 
                            thalamic_indices=None, broca_indices=None, wernicke_indices=None):
    """
    Cerebellar output module: Pontine compression + GC expansion + Purkinje readout.
    """
    N_PONTINE = 128
    N_INPUTS_PER_GC = 4

    all_l56 = []
    for mod in graph.modules:
        all_l56.extend(mod['l56_indices'].tolist())
    all_l56 = np.array(all_l56, dtype=np.int64)
    n_l56 = len(all_l56)

    # Pontine: mixed-sign random projection (fixed, no Oja)
    pontine_weights = torch.randn(N_PONTINE, n_l56, device=device) * (1.0 / (n_l56 ** 0.5))

    # Sparse 4-input GC connectivity
    mossy_mask = torch.zeros(n_granule, N_PONTINE, dtype=torch.bool, device=device)
    for g in range(n_granule):
        sel = np.random.choice(N_PONTINE, N_INPUTS_PER_GC, replace=False)
        mossy_mask[g, sel] = True
    mossy_weights = torch.randn(n_granule, N_PONTINE, device=device) * mossy_mask.float()
    row_norms = mossy_weights.norm(dim=1, keepdim=True).clamp(min=1e-6)
    mossy_weights = mossy_weights / row_norms

    # Signed PK weights (no non-negative constraint)
    purkinje_weights = torch.randn(256, n_granule, device=device) * (1.0 / np.sqrt(n_granule))

    cerebellum = {
        'n_granule': n_granule,
        'n_pontine': N_PONTINE,
        'l56_indices': torch.tensor(all_l56, dtype=torch.long, device=device),
        'pontine_weights': pontine_weights,
        'mossy_weights': mossy_weights,
        'mossy_weights_baseline_norm': mossy_weights.norm().item(),
        'purkinje_weights': purkinje_weights,
        'purkinje_tonic_rate': 0.5,
        'purkinje_intrinsic_excitability': torch.ones(256, device=device),
        'lateral_weights': torch.zeros(256, 256, device=device),
        'lateral_lr': 0.001,
        'golgi_inhibition_ema': torch.zeros(1, device=device),
        'golgi_alpha': 0.01,
        'delta_lr': 0.01,
        'gc_target_sparsity': 0.02,
        'calcium_threshold': torch.ones(256, device=device) * 0.5,
        'gc_active_mask': torch.zeros(n_granule, device=device),
        'error_ema': 0.0,
        'last_climbing_fiber_error': torch.zeros(256, device=device),
        'mli_inhibition_scale': torch.ones(n_granule, device=device),
        'purkinje_bias': torch.zeros(256, device=device),
    }
    init_row_norms = cerebellum['purkinje_weights'].norm(dim=1)
    cerebellum['pk_target_row_norm'] = init_row_norms.clone()

    print(f"Cerebellum: {n_granule} GCs ({N_INPUTS_PER_GC} mossy inputs each), "
          f"{N_PONTINE} pontine neurons, 256 Purkinje outputs")
    print(f"  delta_lr={cerebellum['delta_lr']}, gc_sparsity={cerebellum['gc_target_sparsity']}")
    return cerebellum


def cerebellar_forward(engine, cerebellum):
    """Cerebellar forward: L5/6 → Pontine → GC (divisive Golgi + kWTA) → PK → logits."""
    with torch.no_grad():
        l56_acts = engine.state[cerebellum['l56_indices']]

        # Pontine compression + normalize
        pontine_raw_relu = torch.relu(cerebellum['pontine_weights'] @ l56_acts)
        pontine_norm = pontine_raw_relu.norm()
        if pontine_norm > 1e-8:
            pontine_acts = pontine_raw_relu / pontine_norm
        else:
            pontine_acts = torch.relu(torch.randn_like(pontine_raw_relu) * 0.01)

        # Mossy fiber → GC
        granule_pre = cerebellum['mossy_weights'] @ pontine_acts

        # Divisive Golgi inhibition
        population_input = torch.relu(granule_pre).mean()
        cerebellum['golgi_inhibition_ema'] = (
            (1 - cerebellum['golgi_alpha']) * cerebellum['golgi_inhibition_ema']
            + cerebellum['golgi_alpha'] * population_input
        )
        g_golgi = cerebellum['golgi_inhibition_ema'] * 5.0
        granule_acts_raw = torch.relu(granule_pre) / (1.0 + g_golgi)

        # kWTA sparsity
        k_winners = max(1, int(granule_pre.size(0) * cerebellum['gc_target_sparsity']))
        if k_winners < granule_acts_raw.size(0):
            topk_vals, topk_indices = torch.topk(granule_acts_raw, k_winners)
            granule_acts = torch.zeros_like(granule_acts_raw)
            granule_acts.scatter_(0, topk_indices, topk_vals)
        else:
            granule_acts = granule_acts_raw

        # Normalize GC vector
        gc_norm = granule_acts.norm()
        if gc_norm > 1e-6:
            granule_acts = granule_acts * (5.0 / gc_norm)

        # Purkinje readout with tonic baseline
        purkinje_output = cerebellum['purkinje_weights'] @ granule_acts
        logits = cerebellum['purkinje_tonic_rate'] - purkinje_output
        logits = logits - logits.mean()

        # Store for learning
        cerebellum['gc_active_mask'] = (granule_acts > 0).float()
        cerebellum['_last_gc_for_purkinje'] = granule_acts
        cerebellum['_last_pontine_acts'] = pontine_acts
        cerebellum['_last_pontine_acts_raw'] = pontine_raw_relu

        # Gate for diagnostics only
        gate_value = torch.tensor(1.0, device=logits.device)
        return logits, granule_acts, gate_value


def prune_and_rewire_output(engine, prune_ratio=0.05):
    """
    Hebbian Structural Plasticity for direct L5/6 -> Motor projections.
    Consolidates successful correlations and prunes weak ones.
    """
    with torch.no_grad():
        if not hasattr(engine, 'output_edge_mask') or not engine.output_edge_mask.any():
            return
            
        mask = engine.output_edge_mask
        weights = engine.w_surface[mask]
        abs_weights = weights.abs()
        
        # Determine pruning threshold (bottom X% of weights)
        n_edges = mask.sum().item()
        n_prune = int(n_edges * prune_ratio)
        
        if n_prune == 0:
            return
            
        threshold = torch.kthvalue(abs_weights, n_prune).values
        
        # Identifiers for pruning
        prune_indices = abs_weights <= threshold
        
        # Reassign pruned edges to new random L5/6 -> Motor targets
        # Motor nodes are 256-511. L5/6 indices are engine.is_l56
        l56_indices = torch.where(engine.is_l56)[0]
        motor_indices = torch.arange(256, 512, device=engine.device)
        
        # Randomly sample new source/dest pairs
        new_src = l56_indices[torch.randint(0, len(l56_indices), (n_prune,), device=engine.device)]
        new_dst = motor_indices[torch.randint(0, len(motor_indices), (n_prune,), device=engine.device)]
        
        # Apply pruning and rewiring to global index/weight tensors
        # Find global edge indices that match the output mask and pruning mask
        global_indices = torch.where(mask)[0][prune_indices]
        
        engine.indices[0, global_indices] = new_src
        engine.indices[1, global_indices] = new_dst
        engine._weight_values_raw[global_indices] = torch.randn(n_prune, device=engine.device) * 0.1
        engine.w_surface[global_indices] = engine._weight_values_raw[global_indices].clone()
        
    return n_prune


def cerebellar_learn(cerebellum, logits, granule_acts, target_byte, gate_value, engine, settle_diff=0.0):
    """Online cross-entropy delta rule. No temperature, no sparse update."""
    device = logits.device
    
    target_one_hot = torch.zeros(256, device=device)
    target_one_hot[target_byte] = 1.0
    probs = torch.softmax(logits, dim=0)
    cf_error = target_one_hot - probs

    gc_acts = cerebellum.get('_last_gc_for_purkinje', granule_acts)
    
    # Full delta rule: Δw = -lr * error ⊗ gc_acts
    delta_lr = cerebellum['delta_lr']
    ltd_update = -delta_lr * cf_error.unsqueeze(1) * gc_acts.unsqueeze(0)
    cerebellum['purkinje_weights'] += ltd_update

    # No non-negative clamp (signed weights for anti-correlation)
    # No default-to-LTP (delta rule handles all plasticity)

    # Soft 2% row normalization
    _apply_pk_row_normalization(cerebellum)

    # Anti-Hebbian laterals (corrected sign: co-fire → increase inhibition)
    with torch.no_grad():
        y = torch.relu(logits)
        y_norm = y / (y.norm() + 1e-8)
        co_fire = cerebellum['lateral_lr'] * y_norm.unsqueeze(1) * y_norm.unsqueeze(0)
        co_fire.fill_diagonal_(0.0)
        cerebellum['lateral_weights'] += co_fire
        cerebellum['lateral_weights'] *= 0.999
        cerebellum['lateral_weights'].clamp_(min=0.0, max=0.1)

    cerebellum['last_climbing_fiber_error'] = cf_error
    cerebellum['error_ema'] = (
        0.99 * cerebellum.get('error_ema', 0.0)
        + 0.01 * cf_error.abs().mean().item()
    )
    return cf_error.abs().mean().item()


def _apply_pk_row_normalization(cerebellum):
    """Soft 2% exponential pull-back toward initial row norms."""
    with torch.no_grad():
        w = cerebellum['purkinje_weights']
        target_norms = cerebellum['pk_target_row_norm']
        current_norms = w.norm(dim=1).clamp(min=1e-6)
        scale = 0.98 + 0.02 * (target_norms / current_norms)
        cerebellum['purkinje_weights'] *= scale.unsqueeze(1)


def compute_cerebellar_cortical_feedback(cerebellum, output_error):
    """Feedback through pontine: output_error → PK^T → GC → Mossy^T → Pontine^T → L5/6."""
    with torch.no_grad():
        gc_error = cerebellum['purkinje_weights'].T @ output_error
        gc_mask = cerebellum.get('_last_gc_for_purkinje', cerebellum['gc_active_mask'])
        gc_error_masked = gc_error * (gc_mask > 0).float()
        pontine_error = cerebellum['mossy_weights'].T @ gc_error_masked
        l56_gradient = cerebellum['pontine_weights'].T @ pontine_error
        return l56_gradient


def create_pfa_matrices(engine, n_output=256, n_hidden_pfa=128, device='cpu'):
    """
    Create fixed random feedback matrices for Product Feedback Alignment (PFA).

    PFA replaces DFA's single random matrix B with a product of two matrices
    R and B, such that W ∝ (RB)^T. This introduces an additional population
    of n_hidden_pfa neurons in the backward pathway, allowing any pair of
    connected neurons to be unidirectionally connected while closely
    approximating backpropagation's gradient descent dynamics.

    Empirical results show PFA eliminates the need for separate learning
    phases and achieves BP-level performance in deep hierarchical substrates.

    Returns:
        pfa_matrices: dict mapping level -> (R, B, level_mask) where
            R: [level_size, n_hidden_pfa] first feedback matrix
            B: [n_hidden_pfa, n_output] second feedback matrix
            level_mask: [N] boolean mask for this level's nodes
    """
    # Compute per-node motor connectivity strength
    motor_connectivity = torch.zeros(engine.num_nodes, device=device)
    dst = engine.indices[1]
    src = engine.indices[0]
    motor_dst_mask = (dst >= 256) & (dst < 512)
    if motor_dst_mask.any():
        motor_src = src[motor_dst_mask]
        motor_w = engine.effective_weights[motor_dst_mask].abs()
        motor_connectivity.scatter_add_(0, motor_src, motor_w)

    mc_max = motor_connectivity.max().item()
    if mc_max > 0:
        motor_connectivity = 0.1 + 0.9 * (motor_connectivity / mc_max)
    else:
        motor_connectivity[:] = 1.0

    pfa_matrices = {}
    for level in range(engine.max_level + 1):
        level_mask = (engine.node_to_level == level)
        level_size = level_mask.sum().item()
        if level_size > 0:
            # PFA: Two fixed random matrices forming the product feedback path
            # B maps output error into the hidden backward population
            B = torch.randn(n_hidden_pfa, n_output, device=device) / np.sqrt(n_output)
            # R maps the hidden backward population into the level's nodes
            R = torch.randn(level_size, n_hidden_pfa, device=device) / np.sqrt(n_hidden_pfa)
            # Scale R rows by motor connectivity (same weighting as DFA)
            mc_level = motor_connectivity[level_mask]
            R *= mc_level.unsqueeze(1)
            pfa_matrices[level] = (R, B, level_mask)

    total_nodes = sum(m.sum().item() for _, (_, _, m) in pfa_matrices.items())
    avg_mc = motor_connectivity[512:].mean().item()
    print(f"PFA: Created product feedback alignment for {len(pfa_matrices)} levels, "
          f"{total_nodes} nodes, {n_hidden_pfa} hidden PFA neurons, "
          f"avg motor connectivity: {avg_mc:.3f}")
    return pfa_matrices


def run_linear_probe(engine, data, batch_size=200, free_steps=100, device='cpu'):
    """
    Diagnostic: Freeze the engine, collect latent L5/6 activations for 200 samples,
    and train a temporary linear probe to see if the internal representation
    is actually learning anything, bypassing the complex motor readout.
    """
    print(f"\n[DIAGNOSTIC] Running Linear Probe on {batch_size} samples...")
    engine.state.zero_()
    engine.state_basal.zero_()
    engine.state_apical.zero_()

    l56_indices = torch.where(engine.is_l56)[0]
    num_features = len(l56_indices)
    
    X = []
    Y = []
    
    # 1. Collect Dataset
    # We sample random indices from the data (avoiding the very end)
    sample_indices = np.random.randint(0, len(data) - 1, size=batch_size)
    
    input_mask = torch.zeros(engine.num_nodes, device=device)
    input_mask[:256] = 1.0

    with torch.no_grad():
        for idx in sample_indices:
            curr_byte = int(data[idx])
            next_byte = int(data[idx+1])
            
            input_vec = torch.zeros(engine.num_nodes, device=device)
            input_vec[curr_byte] = 10.0
            
            # Settle engine (FREE phase only)
            engine.settle(
                input_vec, input_mask=input_mask, max_steps=free_steps, 
                tol=0.0, sigma_noise=0.0, damping=0.8
            )
            
            X.append(engine.state[l56_indices].clone())
            Y.append(next_byte)

    X = torch.stack(X) # [batch_size, num_features]
    Y = torch.tensor(Y, device=device) # [batch_size]

    # 2. Train Probe
    # Simple linear readout: X -> logits -> CrossEntropy
    probe = torch.nn.Linear(num_features, 256).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.01)
    
    for ep in range(50):
        logits = probe(X)
        loss = F.cross_entropy(logits, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 3. Eval Probe Accuracy
    with torch.no_grad():
        preds = torch.argmax(probe(X), dim=1)
        acc = (preds == Y).float().mean().item()
    
    return acc


def run_eval(engine, cerebellum, eval_data, free_steps=100, n_samples=500, 
             leak_factor=0.95, device='cpu'):
    """
    Run the cerebellar readout on held-out eval data (no learning).
    Processes data SEQUENTIALLY with LEAK_FACTOR to replicate training dynamics.
    Tests whether the model generalizes or just memorizes training sequences.
    """
    input_mask = torch.zeros(engine.num_nodes, device=device)
    input_mask[:256] = 1.0
    
    # Save engine state
    saved_state = engine.state.clone()
    saved_basal = engine.state_basal.clone()
    saved_apical = engine.state_apical.clone()
    
    correct = 0
    total = 0
    cs_correct = cs_total = 0
    cc_correct = cc_total = 0
    sc_correct = sc_total = 0
    
    # Run sequentially through a random chunk of eval data
    n_samples = min(n_samples, len(eval_data) - 2)
    start = np.random.randint(0, max(1, len(eval_data) - n_samples - 2))
    
    with torch.no_grad():
        engine.state.zero_()
        engine.state_basal.zero_()
        engine.state_apical.zero_()
        
        for j in range(n_samples):
            idx = start + j
            if idx >= len(eval_data) - 1:
                break
            
            prev_byte = int(eval_data[idx - 1]) if idx > 0 else 0
            curr_byte = int(eval_data[idx])
            next_byte = int(eval_data[idx + 1])
            
            # Replicate training dynamics
            engine.state *= leak_factor
            engine.state_basal *= leak_factor
            engine.state_apical *= leak_factor
            
            input_vec = torch.zeros(engine.num_nodes, device=device)
            input_vec[curr_byte] = 10.0
            
            engine.settle(
                input_vec, input_mask=input_mask, max_steps=free_steps,
                tol=0.0, sigma_noise=0.0, damping=0.8
            )
            
            logits, _, _ = cerebellar_forward(engine, cerebellum)
            pred = torch.argmax(logits).item()
            is_correct = (pred == next_byte)
            
            # Skip sentence boundaries (not learnable)
            is_sentence_boundary = (curr_byte == 32 and prev_byte in (ord('.'), ord('?'), ord('!')))
            
            if not is_sentence_boundary:
                total += 1
                if is_correct: correct += 1
                
                if next_byte == 32:
                    cs_total += 1
                    if is_correct: cs_correct += 1
                elif curr_byte == 32:
                    sc_total += 1
                    if is_correct: sc_correct += 1
                else:
                    cc_total += 1
                    if is_correct: cc_correct += 1
    
    # Restore engine state
    engine.state = saved_state
    engine.state_basal = saved_basal
    engine.state_apical = saved_apical
    
    acc = correct / total if total > 0 else 0
    cs_acc = cs_correct / cs_total if cs_total > 0 else 0
    cc_acc = cc_correct / cc_total if cc_total > 0 else 0
    sc_acc = sc_correct / sc_total if sc_total > 0 else 0
    
    return acc, cs_acc, cc_acc, sc_acc


def show_readout(engine, cerebellum, data, current_pos=0, free_steps=100, n_chars=80, 
                 leak_factor=0.95, device='cpu'):
    """
    Display model predictions from the CURRENT training position using 
    the CURRENT engine state. No zeroing, no warmup — shows what the 
    model is actually doing during training.
    """
    input_mask = torch.zeros(engine.num_nodes, device=device)
    input_mask[:256] = 1.0
    
    saved_state = engine.state.clone()
    saved_basal = engine.state_basal.clone()
    saved_apical = engine.state_apical.clone()
    
    actual_chars = []
    predicted_chars = []
    match_markers = []
    
    with torch.no_grad():
        for j in range(n_chars):
            idx = current_pos + j
            if idx >= len(data) - 1:
                break
            
            curr_byte = int(data[idx])
            next_byte = int(data[idx + 1])
            
            engine.state *= leak_factor
            engine.state_basal *= leak_factor
            engine.state_apical *= leak_factor
            
            input_vec = torch.zeros(engine.num_nodes, device=device)
            input_vec[curr_byte] = 10.0
            
            engine.settle(
                input_vec, input_mask=input_mask, max_steps=free_steps,
                tol=0.0, sigma_noise=0.0, damping=0.8
            )
            
            logits, _, _ = cerebellar_forward(engine, cerebellum)
            pred_byte = torch.argmax(logits).item()
            
            actual_ch = chr(next_byte) if 32 <= next_byte < 127 else '.'
            pred_ch = chr(pred_byte) if 32 <= pred_byte < 127 else '.'
            
            actual_chars.append(actual_ch)
            predicted_chars.append(pred_ch)
            match_markers.append('+' if pred_byte == next_byte else '-')
    
    engine.state = saved_state
    engine.state_basal = saved_basal
    engine.state_apical = saved_apical
    
    actual_str = ''.join(actual_chars)
    pred_str = ''.join(predicted_chars)
    match_str = ''.join(match_markers)
    n_correct = match_markers.count('+')
    
    print(f"\n[READOUT] ({n_correct}/{len(match_markers)} correct)")
    print(f"  Target: {actual_str}")
    print(f"  Output: {pred_str}")
    print(f"  Match:  {match_str}")


def ensure_curriculum_data(data_dir):
    """Generate curriculum data if it doesn't exist."""
    train_dir = os.path.join(data_dir, "train")
    eval_dir = os.path.join(data_dir, "eval")
    
    needed_files = [
        os.path.join(train_dir, "level1_holophrases.txt"),
        os.path.join(eval_dir, "level1_holophrases.txt"),
        os.path.join(train_dir, "level2_slot_frame.txt"),
        os.path.join(eval_dir, "level2_slot_frame.txt"),
    ]
    
    if all(os.path.exists(f) for f in needed_files):
        return  # All data exists
    
    print("Curriculum data not found, generating...")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    
    # Import and run curriculum generator
    try:
        from curriculum_gen import (
            generate_holophrases, generate_slot_and_frame,
            generate_complex_constructions, generate_contextual_continuity
        )
        generate_holophrases(
            train_file=os.path.join(train_dir, "level1_holophrases.txt"),
            test_file=os.path.join(eval_dir, "level1_holophrases.txt"),
        )
        generate_slot_and_frame(
            train_file=os.path.join(train_dir, "level2_slot_frame.txt"),
            test_file=os.path.join(eval_dir, "level2_slot_frame.txt"),
        )
        generate_complex_constructions(
            train_file=os.path.join(train_dir, "level3_complex.txt"),
            test_file=os.path.join(eval_dir, "level3_complex.txt"),
        )
        generate_contextual_continuity(
            train_file=os.path.join(train_dir, "level4_contextual.txt"),
            test_file=os.path.join(eval_dir, "level4_contextual.txt"),
        )
        print("Curriculum data generated successfully.")
    except ImportError:
        print("WARNING: curriculum_gen.py not found. Please generate data manually.")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_modules = 10
    num_levels = 2
    edge_index, edge_weight, biases, taus, graph = create_training_graph(
        num_modules=num_modules, num_levels=num_levels
    )
    num_nodes = biases.shape[0]

    # =====================================================================
    # ARCHITECTURAL ENHANCEMENT 1: Boost output connectivity
    # =====================================================================
    edge_index, edge_weight = boost_output_connectivity(
        graph, edge_index, edge_weight, num_nodes
    )

    # Thalamocortical stabilization loop (no motor output)
    # DISABLED for high-frequency linguistic sequences to eliminate temporal smearing
    N_THALAMIC = 0
    graph.thalamic_indices = None

    # Pad cell-type arrays for thalamic relay neurons (excitatory, non-interneuron)
    thal_pad = np.zeros(N_THALAMIC, dtype=bool)
    is_inhibitory = np.concatenate([graph.is_inhibitory, thal_pad])
    is_pv = np.concatenate([graph.is_pv, thal_pad])
    is_sst = np.concatenate([graph.is_sst, thal_pad])
    is_vip = np.concatenate([graph.is_vip, thal_pad])
    is_lts_base = getattr(graph, "is_lts", None)
    if is_lts_base is not None:
        is_lts = np.concatenate([is_lts_base, thal_pad])
    else:
        is_lts = None

    mod_starts = torch.tensor([m["start"] for m in graph.modules], dtype=torch.long)
    mod_ends = torch.tensor([m["end"] for m in graph.modules], dtype=torch.long)
    mod_levels = torch.tensor([m["level"] for m in graph.modules], dtype=torch.long)

    print("Initializing PredictiveCodingEngine...")
    engine = PredictiveCodingEngine(
        num_nodes=num_nodes,
        indices=edge_index.numpy(),
        values=edge_weight.numpy(),
        biases=biases.numpy(),
        taus=taus.numpy(),
        module_ranges=graph.get_module_ranges(),
        module_levels=mod_levels.numpy(),
        hier_pairs=graph.hier_pairs,
        modules=graph.modules,
        device=device,
        is_inhibitory=is_inhibitory,
        is_pv=is_pv,
        is_sst=is_sst,
        is_vip=is_vip,
        is_lts=is_lts,
        dg_indices=graph.dg_indices,
        ca3_indices=graph.ca3_indices,
        temporal_alpha=0.05,
    )

    # Cerebellar output module (replaces FRNL + lateral inhibition + terminal Adam optimizer)
    # Cerebellar output module (replaces FRNL + lateral inhibition + terminal Adam optimizer)
    thal_indices = np.arange(num_nodes - N_THALAMIC, num_nodes)
    
    # Gather Broca and Wernicke indices for SAL partitioning
    broca_indices = []
    wernicke_indices = []
    for mod in graph.modules:
        if mod['id'] in graph.broca_modules:
            broca_indices.extend(mod['l56_indices'].tolist())
        elif mod['id'] in graph.wernicke_modules:
            wernicke_indices.extend(mod['l56_indices'].tolist())
    
    cerebellum = add_cerebellar_module(
        graph, num_nodes, n_granule=1024, device=device)

    # =====================================================================
    # ARCHITECTURAL ENHANCEMENT 3: Direct Feedback Alignment matrices
    # =====================================================================
    pfa_matrices = create_pfa_matrices(engine, n_output=256, n_hidden_pfa=128, device=device)

    # Simplified Dummy Curriculum for Testing
    phases = [
        {
            "name": "SimpleSequenceTracking",
            "lr": 0.05,
            "epochs": 1,
            "text": "abc abc abc abc " * 200,
            "eval_text": "abc abc abc abc " * 50
        }
    ]

    # Input Mask: clamp input nodes only during settling
    input_mask = torch.zeros(num_nodes, device=device)
    input_mask[:256] = 1.0

    # =================================================================
    # Hyperparameters (Round 5 — homeostatic PK + DCN inversion)
    # =================================================================
    FREE_STEPS = 40
    NUDGE_STEPS = 12
    nudge_strength = 5.0
    lr = 0.01
    sigma_noise_free = 0.01
    sigma_noise_nudge = 0.01
    LEAK_FACTOR = 0.95
    ALPHA_PFA = 0.15
    ALPHA_CEREBELLAR = 0.05
    BURST_APICAL_GAIN = 2.0

    for phase_info in phases:
        phase_name = phase_info["name"]
        text = phase_info["text"]
        eval_text = phase_info["eval_text"]

        print(f"\n--- Starting Phase: {phase_name} ---")
        print(f"  Steps: Free={FREE_STEPS}, Nudge={NUDGE_STEPS}")
        print(f"  Learning rate: {lr}, Nudge strength: {nudge_strength}")
        print(f"  PFA alpha: {ALPHA_PFA}, CB alpha: {ALPHA_CEREBELLAR}, Burst apical gain: {BURST_APICAL_GAIN}")
        print(f"  Thalamocortical loop: {N_THALAMIC} neurons")
        print(f"  Cerebellum: {cerebellum['n_granule']} granule cells")

        data = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
        seq_len = len(data)
        eval_data = np.frombuffer(eval_text.encode("utf-8"), dtype=np.uint8)
        print(f"  Eval data: {len(eval_data)} bytes")

        # Metrics tracking
        spatial_errors = []
        accuracies = []       # Excludes sentence boundaries
        accuracies_all = []   # Includes everything (for raw logging)
        burst_strengths = []
        # Structural Acc Metrics (excluding sentence-boundary S→C)
        cs_total, cs_correct = 0, 0   # Char->Space (learnable)
        cc_total, cc_correct = 0, 0   # Char->Char (learnable)
        sc_total, sc_correct = 0, 0   # Space->Char within sentence (learnable)
        sb_total = 0                   # Sentence boundary transitions (not learnable, just counted)
        
        settle_history = deque(maxlen=50) # Track settle diffs for adaptive gating

        # Reset weights for the tuned run to ensure no stale associations
        engine.reset_plastic_weights()

        start_time = time.time()
        for i in range(seq_len - 1):
            current_byte = int(data[i])
            next_byte = int(data[i + 1])

            # PHASE 1: Leaky Persistence (no reset)
            engine.state *= LEAK_FACTOR
            engine.state_basal *= LEAK_FACTOR
            engine.state_apical *= LEAK_FACTOR
            
            # Target 5: Thalamocortical Gating - rapidly reset thalamic state at each sequence token
            if getattr(graph, 'thalamic_indices', None) is not None:
                engine.state[graph.thalamic_indices] = 0.0
                engine.state_basal[graph.thalamic_indices] = 0.0
                engine.state_apical[graph.thalamic_indices] = 0.0
                
            engine.previous_state = engine.state.clone()

            # === FREE PHASE ===
            input_vec = torch.zeros(num_nodes, device=device)
            input_vec[current_byte] = 10.0

            # Forward Updates: store initial prediction μ_l^0 before settling
            initial_prediction = engine.state.clone()

            engine.settle(
                input_vec, input_mask=input_mask, max_steps=FREE_STEPS, tol=0.0,
                sigma_noise=sigma_noise_free, damping=0.8
            )

            # Forward Updates: compute forward error ε̃ = x^T - μ^0
            # This measures how much the network's state changed during settling.
            # Large forward errors indicate surprising inputs (high prediction error).
            forward_error = engine.state - initial_prediction
            
            # Adaptive Gating: "Leaky" Basal Ganglia logic
            # If settle has stabilized (low variance), slowly raise threshold to allow output
            settle_history.append(engine.last_settle_diff)
            # BG gate threshold fixed at 0.3 — no drift
            # Previous drift logic (0.3→0.8) caused permissive gating

            free_state = engine.state.clone()
            free_basal = engine.state_basal.clone()
            free_apical = engine.state_apical.clone()

            # === CEREBELLAR READOUT (replaces FRNL + lateral inhibition) ===
            logits, granule_acts, gate_value = cerebellar_forward(engine, cerebellum)
            pred_byte = torch.argmax(logits).item()
            is_correct = pred_byte == next_byte
            
            # Detect sentence boundaries: space after . ? !
            # These predict the first char of a RANDOM next sentence — not learnable
            prev_byte = int(data[i - 1]) if i > 0 else 0
            is_sentence_boundary = (current_byte == 32 and prev_byte in (ord('.'), ord('?'), ord('!')))
            
            # Track all predictions for raw window accuracy
            accuracies_all.append(1.0 if is_correct else 0.0)
            
            # Track learnable predictions only (exclude sentence boundaries)
            if not is_sentence_boundary:
                accuracies.append(1.0 if is_correct else 0.0)

            # Update structural stats (learnable transitions only)
            if is_sentence_boundary:
                sb_total += 1  # Count but don't score
            elif next_byte == 32:  # Char->Space
                cs_total += 1
                if is_correct: cs_correct += 1
            elif current_byte == 32:  # Space->Char (within sentence)
                sc_total += 1
                if is_correct: sc_correct += 1
            else:  # Char->Char
                cc_total += 1
                if is_correct: cc_correct += 1

            # === CEREBELLAR LEARNING (replaces terminal Adam optimizer) ===
            cf_error = cerebellar_learn(cerebellum, logits, granule_acts, next_byte, gate_value,
                                        engine=engine, settle_diff=engine.last_settle_diff)

            # Prediction errors
            spatial_err = engine.compute_prediction_errors()
            spatial_errors.append(spatial_err)

            # === NUDGED PHASE (with DFA + Burst Coincidence) ===
            target_one_hot = torch.zeros(256, device=device)
            target_one_hot[next_byte] = 1.0

            # Compute output error for DFA (Fix D: Sync with Softmax)
            # Use Softmax to match the internal cerebellar learning rule.
            output_error = target_one_hot - torch.softmax(logits, dim=0)

            # Restore to free state before nudge
            engine.state = free_state.clone()
            engine.state_basal = free_basal.clone()
            engine.state_apical = free_apical.clone()

            # Build nudge vector with PFA feedback
            nudge_vec = torch.zeros(num_nodes, device=device)
            nudge_vec[current_byte] = 10.0
            # NOTE: No direct motor node clamping — the cerebellum handles output

            # ENHANCEMENT 3d: Forward Updates — inject prediction surprise into nudge
            # ΔW_l = η · ε̃_l^T · f(x_{l-1}^T)^T
            # Forward error = how much state changed during settling.
            # Scaled at 0.15 to properly propagate deep prediction error.
            nudge_vec += 0.15 * forward_error

            # ENHANCEMENT 3a: PFA — inject output error into all hidden levels
            for level, (R, B, level_mask) in pfa_matrices.items():
                # PFA: R @ (B @ output_error) — product of two feedback matrices
                pfa_signal = R @ (B @ output_error)  # [level_size]
                nudge_vec[level_mask] += ALPHA_PFA * pfa_signal

            # ENHANCEMENT 3c: Cerebello-thalamo-cortical co-adaptation
            if (i + 1) % 50 == 0:
                l56_grad = compute_cerebellar_cortical_feedback(cerebellum, output_error)
                nudge_vec[cerebellum['l56_indices']] += ALPHA_CEREBELLAR * l56_grad

            # ENHANCEMENT 3b: Burst Coincidence — inject target into apical
            engine.inject_apical_nudge(target_one_hot, strength=BURST_APICAL_GAIN)

            engine.settle(
                nudge_vec,
                input_mask=input_mask,
                max_steps=NUDGE_STEPS,
                tol=0.0,
                sigma_noise=sigma_noise_nudge,
                damping=0.3
            )

            # BUG 2 FIX: Corticospinal Motor Nudge via L5/6 Burst Propagation
            # Motor nodes (256-511) receive NO direct teaching signal during nudge.
            # EqProp needs (nudge - free) ≠ 0 at motor nodes to learn output weights.
            # Bio pathway: L5/6 BAC firing (from inject_apical_nudge) → burst propagates
            # down axon through output projection weights → drives motor nodes.
            # This creates a bootstrap loop: as output weights learn, burst propagation
            # strengthens, which creates stronger EqProp gradients, which further
            # improves the weights — matching developmental motor learning.
            with torch.no_grad():
                burst = engine.compute_burst_coincidence()  # L5/6 burst from apical nudge
                out_mask = engine.output_edge_mask
                if out_mask.any():
                    burst_signal = burst[engine.indices[0][out_mask]] * engine.effective_weights[out_mask]
                    burst_drive = torch.zeros(num_nodes, device=device)
                    burst_drive.scatter_add_(0, engine.indices[1][out_mask], burst_signal)
                    # Inject burst-propagated signal into motor nodes only
                    engine.state[256:512] += 2.0 * burst_drive[256:512]

            nudge_pos = engine.state.clone()

            # Reuse burst computed in propagation block for monitoring
            avg_burst = burst[engine.is_l56].mean().item()
            burst_strengths.append(avg_burst)

            # Biological EqProp/DFA for all deep layers
            engine.update_weights_phase2(
                free_state,
                nudge_pos,
                learning_rate=lr,
            )

            # Restore free-phase state after nudge+weight update.
            # Without this, the nudge state (containing target info) leaks 
            # into the next step via LEAK_FACTOR.
            engine.state = free_state.clone()
            engine.state_basal = free_basal.clone()
            engine.state_apical = free_apical.clone()

            # Periodic Structural Plasticity (Hebbian Rewiring)
            if (i + 1) % 1000 == 0:
                n_pruned = prune_and_rewire_output(engine, prune_ratio=0.03)
                print(f"\n  [STRUCTURAL] Pruned and rewired {n_pruned} weak output projections.")

            # Logging and periodic tasks
            if (i + 1) % 10 == 0:
                window = min(500, len(accuracies))
                avg_acc = np.mean(accuracies[-window:])
                avg_err = np.mean(spatial_errors[-100:])
                avg_bst = np.mean(burst_strengths[-100:])

                w_surf_norm = engine.w_surface.norm().item()
                w_max = engine.w_surface.abs().max().item()

                elapsed = time.time() - start_time
                tps = (i + 1) / elapsed

                # Diagnostics
                mean_firing = engine.lifetime_firing[512:].mean().item()
                bias_norm = engine.biases.norm().item()
                inh_w_norm = engine.w_surface[engine.is_inhibitory[engine.indices[0]]].norm().item()

                # Granule cell sparsity (should be ~5-10% active)
                gc_sparsity = (granule_acts > 0).float().mean().item()
                # Purkinje weight norms
                pk_w_norm = cerebellum['purkinje_weights'].norm().item()
                pk_row_norms = cerebellum['purkinje_weights'].norm(dim=1)
                pk_row_max = pk_row_norms.max().item()
                pk_row_mean = pk_row_norms.mean().item()
                
                # Softmax-relevant diagnostics (NOT sigmoid — learning uses softmax)
                probs = torch.softmax(logits, dim=0)
                target_prob = probs[next_byte].item()  # P(correct class)
                pred_prob = probs.max().item()          # P(predicted class)
                logit_range = (logits.max() - logits.min()).item()
                
                # Target prob tracking (most informative learning metric)
                if not hasattr(cerebellar_learn, '_target_probs'):
                    cerebellar_learn._target_probs = []
                cerebellar_learn._target_probs.append(target_prob)
                tp_window = cerebellar_learn._target_probs[-window:]
                avg_target_prob = np.mean(tp_window)

                print(
                    f"Step {i+1}/{seq_len} | Acc: {avg_acc:.2%} | "
                    f"P(target): {avg_target_prob:.3f} | "
                    f"PK_w: {pk_w_norm:.2f} (row mean: {pk_row_mean:.2f} max: {pk_row_max:.2f}) | "
                    f"GC_spars: {gc_sparsity:.2%}"
                )
                print(
                    f"  Logit range: {logit_range:.4f} | "
                    f"CF_err: {cf_error:.4f} | "
                    f"Settle: {engine.last_settle_diff:.4f}"
                )
                
                # New diagnostics for Bug 1/2/3 fixes
                out_w_norm = engine.w_surface[engine.output_edge_mask].norm().item() if engine.output_edge_mask.any() else 0.0
                out_w_mean = engine.w_surface[engine.output_edge_mask].abs().mean().item() if engine.output_edge_mask.any() else 0.0
                cf_surprise = cerebellum.get('error_ema', 0.0)
                
                # Burst selectivity: fraction of L5/6 neurons with burst > 0.5
                # Should be ~10-30% with selective burst, was ~100% with .abs() bug
                with torch.no_grad():
                    burst_diag = engine.compute_burst_coincidence()
                    l56_burst = burst_diag[engine.is_l56]
                    burst_frac = (l56_burst > 0.5).float().mean().item()
                    burst_mean = l56_burst.mean().item()
                
                # Actual IO Dimensionality (% of non-zero teaching signals)
                last_cf = cerebellum.get('last_climbing_fiber_error', torch.zeros(256))
                io_dim = (last_cf.abs() > 1e-5).float().mean().item() * 100
                
                print(
                    f"  OutProj w_surf: norm={out_w_norm:.4f} mean={out_w_mean:.6f} | "
                    f"Burst: mean={burst_mean:.4f} frac>0.5={burst_frac:.2%} | "
                    f"CF_ema: {cf_surprise:.4f} | IO_space: {io_dim:.1f}%"
                )
                
                # Cerebellar homeostasis diagnostics
                pk_silent = (cerebellum['purkinje_weights'].abs() < 1e-6).float().mean().item()
                mossy_norm = cerebellum['mossy_weights'].norm().item()
                mossy_baseline = cerebellum['mossy_weights_baseline_norm']
                mossy_ratio = mossy_norm / max(mossy_baseline, 1e-6)
                mli_mean = cerebellum['mli_inhibition_scale'].mean().item()
                ca_thresh_mean = cerebellum['calcium_threshold'].mean().item()
                pk_row_ratio = (cerebellum['purkinje_weights'].norm(dim=1) / 
                               cerebellum['pk_target_row_norm'].clamp(min=1e-6)).mean().item()
                
                print(
                    f"  PK_silent: {pk_silent:.1%} | "
                    f"MF_gain: {mossy_ratio:.2f}x | "
                    f"MLI: {mli_mean:.3f} | "
                    f"Ca_θ: {ca_thresh_mean:.4f} | "
                    f"PK_row_ratio: {pk_row_ratio:.3f}"
                )
                
                # GC pattern discriminability: overlap between current and previous GC pattern
                # If centering works, overlap should be <100% (different GCs for different inputs)
                gc_mask_now = cerebellum.get('gc_active_mask', torch.zeros(cerebellum['n_granule'], device=device))
                gc_prev = cerebellum.get('_prev_gc_mask', gc_mask_now)
                if gc_mask_now.sum() > 0 and gc_prev.sum() > 0:
                    intersection = (gc_mask_now * gc_prev).sum().item()
                    union = ((gc_mask_now + gc_prev) > 0).float().sum().item()
                    gc_jaccard = intersection / max(union, 1.0)
                else:
                    gc_jaccard = 1.0
                cerebellum['_prev_gc_mask'] = gc_mask_now.clone()
                
                # Pontine compression diagnostics
                if '_last_pontine_acts' in cerebellum:
                    pontine_acts = cerebellum['_last_pontine_acts']
                    pontine_range = (pontine_acts.max() - pontine_acts.min()).item()
                    pontine_sparsity = (pontine_acts > 0).float().mean().item()
                else:
                    pontine_range = 0.0
                    pontine_sparsity = 0.0
                
                print(
                    f"  GC_overlap: {gc_jaccard:.2%} | "
                    f"Pontine: range={pontine_range:.3f} spars={pontine_sparsity:.1%}"
                )


                # Diagnostic Linear Probe + Readout every 500 steps
                if (i + 1) % 500 == 0:
                    probe_acc = run_linear_probe(
                        engine, data, batch_size=200, 
                        free_steps=FREE_STEPS, device=device
                    )
                    print(f"  >>> LINEAR PROBE ACCURACY: {probe_acc:.2%} (Internal Representation Quality)")
                    if probe_acc > avg_acc * 2:
                        print(f"  [!] ALERT: Representation ({probe_acc:.1%}) >> Readout ({avg_acc:.1%}). Readout is the bottleneck.")
                    else:
                        print(f"  [!] NOTE: Representation ({probe_acc:.1%}) ~= Readout ({avg_acc:.1%}). Hebbian learning is the bottleneck.")
                    
                    # Show what the model is actually outputting
                    show_readout(engine, cerebellum, data, current_pos=i+1, free_steps=FREE_STEPS, n_chars=80, leak_factor=LEAK_FACTOR, device=device)
                    
                    # Prediction diversity check (class collapse detector)
                    # Sample 100 chars and count unique predictions
                    with torch.no_grad():
                        pred_counts = torch.zeros(256, device=device)
                        saved_s = engine.state.clone()
                        saved_b = engine.state_basal.clone()
                        saved_a = engine.state_apical.clone()
                        div_indices = np.random.randint(0, len(data)-1, size=100)
                        for di in div_indices:
                            cb = int(data[di])
                            iv = torch.zeros(num_nodes, device=device)
                            iv[cb] = 10.0
                            engine.settle(iv, input_mask=input_mask, max_steps=FREE_STEPS, tol=0.0, sigma_noise=0.0, damping=0.8)
                            lg, _, _ = cerebellar_forward(engine, cerebellum)
                            pred_counts[torch.argmax(lg)] += 1
                        engine.state = saved_s
                        engine.state_basal = saved_b
                        engine.state_apical = saved_a
                        n_unique = (pred_counts > 0).sum().item()
                        top_pred_byte = torch.argmax(pred_counts).item()
                        top_pred_frac = pred_counts.max().item() / 100.0
                        top_ch = chr(top_pred_byte) if 32 <= top_pred_byte < 127 else f'\\x{top_pred_byte:02x}'
                        print(f"  [DIVERSITY] {n_unique} unique predictions / 100 samples | "
                              f"Top: '{top_ch}' ({top_pred_frac:.0%})"
                              f"{' ← CLASS COLLAPSE' if n_unique <= 3 else ''}")
                
                # Eval on held-out data every 2500 steps
                if (i + 1) % 2500 == 0 and eval_data is not None:
                    eval_acc, eval_cs, eval_cc, eval_sc = run_eval(
                        engine, cerebellum, eval_data,
                        free_steps=FREE_STEPS, n_samples=300, leak_factor=LEAK_FACTOR, device=device
                    )
                    print(f"  >>> EVAL (held-out): {eval_acc:.2%} | C->S: {eval_cs:.2%} | C->C: {eval_cc:.2%} | S->C: {eval_sc:.2%}")

                # Structural Diagnostics (learnable transitions only)
                cs_acc = (cs_correct / cs_total) if cs_total > 0 else 0
                cc_acc = (cc_correct / cc_total) if cc_total > 0 else 0
                sc_acc = (sc_correct / sc_total) if sc_total > 0 else 0
                print(f"  Structural Acc | C->S: {cs_acc:.2%} ({cs_total}) | C->C: {cc_acc:.2%} ({cc_total}) | S->C: {sc_acc:.2%} ({sc_total}) | SentBound: {sb_total}")

        elapsed = time.time() - start_time
        final_acc = (
            np.mean(accuracies[-1000:])
            if len(accuracies) >= 1000
            else np.mean(accuracies)
        )
        print(
            f"\nPhase {phase_name} complete in {elapsed:.1f}s | "
            f"Final Acc: {final_acc:.2%}"
        )


if __name__ == "__main__":
    main()