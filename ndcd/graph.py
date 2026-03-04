
import networkx as nx
import numpy as np
import scipy.sparse as sp

class DynamicGraph:
    def __init__(self, num_nodes, m_edges=2, p_triad=0.1, seed=None,
                 num_modules=50, num_levels=4):
        """
        Initializes the Substrate using a Hierarchical Modular topology
        for Predictive Coding.

        Instead of a flat Holme-Kim graph, creates explicit modules with:
        - Dense internal connectivity (~60% within module)
        - Sparse inter-module connectivity at same level (~2-5%)
        - Sparse hierarchical connections between levels (~1-3%)

        Modules are assigned to hierarchical levels:
        - Level 0: Closest to input (fast tau, receives input projections)
        - Level 1: Low-level features (word-internal sequences)
        - Level 2: Mid-level features (word-to-word transitions)
        - Level 3: Most abstract (topic persistence, ultra-slow tau)

        Args:
            num_nodes: Total neurons in the brain (should be >= 512 + num_modules*2).
            m_edges: Legacy parameter (unused in modular topology).
            p_triad: Legacy parameter (unused in modular topology).
            seed: Random seed for reproducibility.
            num_modules: Number of learning modules.
            num_levels: Number of hierarchical levels (default 4).
        """
        self.num_nodes = num_nodes
        self.num_modules = num_modules
        self.num_levels = num_levels
        if seed is not None:
            np.random.seed(seed)

        # --- Node Classification ---
        # 256 input + 256 output nodes, rest are association nodes in modules
        n_sensory = 256
        n_motor = 256
        n_io = n_sensory + n_motor
        n_association = num_nodes - n_io

        self.sensory_indices = np.arange(0, n_sensory)
        self.motor_indices = np.arange(n_sensory, n_io)
        self.association_indices = np.arange(n_io, num_nodes)

        # --- Assign modules to hierarchical levels ---
        # Distribution: flattened to expand L2/L3 representational capacity
        # for contextual and semantic persistence (was pyramidal [40,30,20,10])
        level_fractions = [0.25, 0.25, 0.25, 0.25]
        level_module_counts = []
        remaining = num_modules
        for i in range(num_levels - 1):
            count = max(1, int(num_modules * level_fractions[i]))
            level_module_counts.append(count)
            remaining -= count
        level_module_counts.append(max(1, remaining))

        # Build module metadata
        self.modules = []  # List of dicts: {level, start_idx, end_idx, node_indices, l4_indices, l23_indices, l56_indices}
        self.module_levels = []  # level per module
        self.level_modules = {l: [] for l in range(num_levels)}  # modules per level

        # Distribute association nodes across modules
        nodes_per_module = n_association // num_modules
        extra_nodes = n_association % num_modules

        current_node = n_io  # Start after I/O nodes
        module_id = 0
        for level in range(num_levels):
            for _ in range(level_module_counts[level]):
                # Distribute extra nodes to first modules
                n_in_module = nodes_per_module + (1 if module_id < extra_nodes else 0)
                start = current_node
                end = current_node + n_in_module
                node_indices = np.arange(start, end)

                # Laminar Architecture sub-populations
                # Layer 4 (L4): ~20% of nodes (sensory/bottom-up recipient)
                # Layer 2/3 (L2/3): ~40% of nodes (error/surprisal)
                # Layer 5/6 (L5/6): ~40% of nodes (predictions/top-down source)
                n_l4 = max(1, int(n_in_module * 0.2))
                n_l23 = max(1, int(n_in_module * 0.4))
                n_l56 = n_in_module - n_l4 - n_l23

                l4_start = start
                l23_start = l4_start + n_l4
                l56_start = l23_start + n_l23

                l4_indices = np.arange(l4_start, l4_start + n_l4)
                l23_indices = np.arange(l23_start, l23_start + n_l23)
                l56_indices = np.arange(l56_start, l56_start + n_l56)

                self.modules.append({
                    'id': module_id,
                    'level': level,
                    'start': start,
                    'end': end,
                    'indices': node_indices,
                    'size': n_in_module,
                    'l4_indices': l4_indices,
                    'l23_indices': l23_indices,
                    'l56_indices': l56_indices
                })
                self.module_levels.append(level)
                self.level_modules[level].append(module_id)

                current_node = end
                module_id += 1

        self.num_actual_modules = module_id
        self.module_levels = np.array(self.module_levels)

        # --- Dale's Law E/I Assignment ---
        self.is_inhibitory = np.zeros(num_nodes, dtype=bool)
        for mod in self.modules:
            # 20% of nodes in each module are inhibitory
            n_inh = int(mod['size'] * 0.2)
            if n_inh > 0:
                inh_idx = np.random.choice(mod['indices'], n_inh, replace=False)
                self.is_inhibitory[inh_idx] = True

        # --- Functional Lateralization ---
        # Selectively designate ~15% of intermediate modules as Broca and ~15% as Wernicke
        self.broca_modules = set()
        self.wernicke_modules = set()
        for level in range(1, num_levels):
            level_mods = self.level_modules[level]
            n_special = max(1, int(len(level_mods) * 0.15))
            
            # First batch -> Broca
            for mod_id in level_mods[:n_special]:
                self.broca_modules.add(mod_id)
            # Second batch -> Wernicke
            for mod_id in level_mods[n_special:2*n_special]:
                self.wernicke_modules.add(mod_id)
        
        print(f"Lateralization: {len(self.broca_modules)} Broca (seq), {len(self.wernicke_modules)} Wernicke (sem)")

        # --- Spatial Embedding ---
        self.pos = np.random.rand(num_nodes, 3)

        # --- Build Sparse Weight Matrix ---
        # We build edge lists directly for efficiency
        edge_rows = []
        edge_cols = []

        # 1. INTRA-module connectivity
        # Structural sparsity constraint: Target < 1% connectivity internally
        # L4 -> L2/3, L2/3 -> L5/6, L5/6 -> L5/6 (recurrent), L5/6 -> L4
        target_connections_per_node = 10  # Reduced significantly for <1% sparsity
        for mod in self.modules:
            # We want connections between specific layers
            l4 = mod['l4_indices']
            l23 = mod['l23_indices']
            l56 = mod['l56_indices']
            
            # L4 -> L2/3
            if len(l4) > 0 and len(l23) > 0:
                n_edges = int(len(l4) * target_connections_per_node)
                src = np.random.choice(l4, n_edges, replace=True)
                dst = np.random.choice(l23, n_edges, replace=True)
                edge_rows.extend(src.tolist())
                edge_cols.extend(dst.tolist())
                
            # L2/3 -> L5/6
            if len(l23) > 0 and len(l56) > 0:
                n_edges = int(len(l23) * target_connections_per_node)
                src = np.random.choice(l23, n_edges, replace=True)
                dst = np.random.choice(l56, n_edges, replace=True)
                edge_rows.extend(src.tolist())
                edge_cols.extend(dst.tolist())
                
            # L5/6 -> L5/6 (recurrent)
            if len(l56) > 0:
                n_edges = int(len(l56) * target_connections_per_node)
                src = np.random.choice(l56, n_edges, replace=True)
                dst = np.random.choice(l56, n_edges, replace=True)
                valid = src != dst
                edge_rows.extend(src[valid].tolist())
                edge_cols.extend(dst[valid].tolist())
                
            # L5/6 -> L4 (feedback within column)
            if len(l56) > 0 and len(l4) > 0:
                n_edges = int(len(l56) * target_connections_per_node)
                src = np.random.choice(l56, n_edges, replace=True)
                dst = np.random.choice(l4, n_edges, replace=True)
                edge_rows.extend(src.tolist())
                edge_cols.extend(dst.tolist())

        # 2. Sparse INTER-module connectivity at same level (~1%)
        inter_same_density_default = 0.01
        for level in range(num_levels):
            inter_same_density = 0.02 if level == 0 else inter_same_density_default
            mods_at_level = self.level_modules[level]
            for i in range(len(mods_at_level)):
                for j in range(i + 1, len(mods_at_level)):
                    mod_i = self.modules[mods_at_level[i]]
                    mod_j = self.modules[mods_at_level[j]]
                    
                    pair_density = inter_same_density
                    if mod_i['id'] in getattr(self, 'wernicke_modules', set()) or mod_j['id'] in getattr(self, 'wernicke_modules', set()):
                        pair_density *= 2.0
                        
                    # Lateral connections primarily go L2/3 <-> L2/3 and L5/6 <-> L5/6
                    for layer in ['l23_indices', 'l56_indices']:
                        idx_i = mod_i[layer]
                        idx_j = mod_j[layer]
                        
                        if len(idx_i) == 0 or len(idx_j) == 0: continue
                        
                        n_possible = len(idx_i) * len(idx_j)
                        n_connections = max(1, int(n_possible * pair_density))
                        n_connections = min(n_connections, max(5, min(len(idx_i), len(idx_j))))
                        
                        src_picks = np.random.choice(idx_i, n_connections, replace=True)
                        dst_picks = np.random.choice(idx_j, n_connections, replace=True)
                        edge_rows.extend(src_picks.tolist())
                        edge_cols.extend(dst_picks.tolist())
                        # Bidirectional
                        edge_rows.extend(dst_picks.tolist())
                        edge_cols.extend(src_picks.tolist())

        # 3. Hierarchical connections between levels.
        # Bottom-up: L2/3 (lower) -> L4 (upper)
        # Top-down: L5/6 (upper) -> L2/3 and L5/6 (lower)
        hier_density = 0.02  # Sparsified from 0.08
        for level in range(num_levels - 1):
            lower_mods = self.level_modules[level]
            upper_mods = self.level_modules[level + 1]
            for lm_id in lower_mods:
                for um_id in upper_mods:
                    mod_lower = self.modules[lm_id]
                    mod_upper = self.modules[um_id]
                    
                    pair_density = hier_density
                    if mod_lower['id'] in getattr(self, 'wernicke_modules', set()) or mod_upper['id'] in getattr(self, 'wernicke_modules', set()):
                        pair_density *= 2.0
                        
                    # Bottom-up (lower L2/3 -> upper L4)
                    l_l23 = mod_lower['l23_indices']
                    u_l4 = mod_upper['l4_indices']
                    if len(l_l23) > 0 and len(u_l4) > 0:
                        n_possible = len(l_l23) * len(u_l4)
                        n_connections = max(5, int(n_possible * pair_density))
                        src = np.random.choice(l_l23, n_connections, replace=True)
                        dst = np.random.choice(u_l4, n_connections, replace=True)
                        edge_rows.extend(src.tolist())
                        edge_cols.extend(dst.tolist())
                        
                    # Top-down (upper L5/6 -> lower L2/3 and L5/6)
                    u_l56 = mod_upper['l56_indices']
                    l_l2356 = np.concatenate([mod_lower['l23_indices'], mod_lower['l56_indices']])
                    if len(u_l56) > 0 and len(l_l2356) > 0:
                        n_possible = len(u_l56) * len(l_l2356)
                        n_connections = max(5, int(n_possible * pair_density))
                        src = np.random.choice(u_l56, n_connections, replace=True)
                        dst = np.random.choice(l_l2356, n_connections, replace=True)
                        edge_rows.extend(src.tolist())
                        edge_cols.extend(dst.tolist())

        # 4. Input projections: sensory nodes -> level-0 modules (specifically L4)
        level0_mods = self.level_modules[0]
        for mod_id in level0_mods:
            mod = self.modules[mod_id]
            idx = mod['l4_indices']  # Project into L4
            if len(idx) == 0: continue
            n_proj = max(1, len(idx) // 2)
            for s in range(n_sensory):
                targets = np.random.choice(idx, n_proj, replace=False)
                edge_rows.extend([s] * n_proj)
                edge_cols.extend(targets.tolist())
                # Allow feedback from L4 to input nodes to stabilize inputs
                edge_rows.extend(targets.tolist())
                edge_cols.extend([s] * n_proj)

        # 5. Output projections: level-0 modules (L5/6) -> motor nodes
        for mod_id in level0_mods:
            mod = self.modules[mod_id]
            idx = mod['l56_indices']  # Output comes from L5/6
            if len(idx) == 0: continue
            n_proj = max(1, len(idx) // 4)
            for m in range(n_motor):
                sources = np.random.choice(idx, n_proj, replace=False)
                motor_node = n_sensory + m
                edge_rows.extend(sources.tolist())
                edge_cols.extend([motor_node] * n_proj)
                edge_rows.extend([motor_node] * n_proj)
                edge_cols.extend(sources.tolist())

        # Convert to numpy arrays and remove duplicates
        edge_rows = np.array(edge_rows, dtype=np.int64)
        edge_cols = np.array(edge_cols, dtype=np.int64)

        # Remove self-loops
        valid = edge_rows != edge_cols
        edge_rows = edge_rows[valid]
        edge_cols = edge_cols[valid]

        # Deduplicate edges
        edge_pairs = np.stack([edge_rows, edge_cols], axis=1)
        edge_pairs = np.unique(edge_pairs, axis=0)
        edge_rows = edge_pairs[:, 0]
        edge_cols = edge_pairs[:, 1]

        num_edges = len(edge_rows)
        print(f"Hierarchical graph: {num_nodes} nodes, {num_edges} edges, "
              f"{self.num_actual_modules} modules across {num_levels} levels")
        for level in range(num_levels):
            n_mods = len(self.level_modules[level])
            mod_sizes = [self.modules[m]['size'] for m in self.level_modules[level]]
            print(f"  Level {level}: {n_mods} modules, "
                  f"avg size {np.mean(mod_sizes):.0f} nodes")

        # Initialize weights (sparse — never allocate dense num_nodes × num_nodes)
        # Xavier-like: std = sqrt(2 / (avg_fan_in + avg_fan_out))
        avg_degree = num_edges / num_nodes
        std_dev = np.sqrt(2.0 / (avg_degree + avg_degree))

        weight_vals = np.abs(np.random.normal(0, std_dev, num_edges)).astype(np.float32)
        src_is_inh = self.is_inhibitory[edge_rows]
        weight_vals[src_is_inh] = -weight_vals[src_is_inh]

        # Input nodes (0-255) are CLAMPED during settling — edges from/to
        # them are external forcing, not autonomous recurrence. Tuning
        # SR uniformly across all edges lets I/O edges (~60% of total)
        # dominate the spectrum, crushing recurrent weights to ~20%.
        input_proj_mask = (edge_rows < 256) | (edge_cols < 256)
        free_mask = ~input_proj_mask

        # Tune spectral radius of FREE edges only (non-input-projection)
        free_sparse = sp.csr_matrix(
            (weight_vals[free_mask], (edge_rows[free_mask], edge_cols[free_mask])),
            shape=(num_nodes, num_nodes)
        )

        target_sr = 0.90
        try:
            from scipy.sparse.linalg import eigs as sp_eigs
            eigvals = sp_eigs(free_sparse.astype(np.float64),
                              k=1, which='LM', return_eigenvectors=False)
            free_radius = np.abs(eigvals[0])
            if free_radius > 0:
                scale = np.float32(target_sr / free_radius)
                weight_vals[free_mask] *= scale
                print(f"Free-edge spectral radius tuned: {free_radius:.4f} -> {target_sr}")
        except Exception as e:
            print(f"Spectral tuning failed ({e}), using Frobenius fallback")
            frob = sp.linalg.norm(free_sparse, 'fro')
            if frob > 0:
                weight_vals[free_mask] *= np.float32(target_sr * np.sqrt(num_nodes) / frob)

        # Set input projection edges independently.
        # These are external forcing (clamped nodes), not recurrence.
        # CRITICAL FIX for One-Hot Input Drive:
        # Since the input is one-hot (only 1 out of 256 nodes active), standard
        # Xavier limits the sum to just one single weight instead of a random sum
        # over the fan-in. To achieve order ~1.0 variance in the receiving Level 0 nodes,
        # we must multiply the input weights by sqrt(fan_in) for the input projection.
        # With avg fan-in ~250 from I/O nodes, sqrt(250) ≈ 15.8.
        # We cap it at 15.0 to maintain solver stability.
        # Dynamically calibrate input boost against lateral recurrence strength
        # instead of hardcoded 6.5x which overwhelmed lateral recurrence
        lateral_rms = np.sqrt(np.mean(weight_vals[free_mask] ** 2)) if free_mask.sum() > 0 else 0.1
        input_target_rms = 3.0 * lateral_rms  # 3x lateral (reduced from 6.5x)
        input_current_rms = np.sqrt(np.mean(weight_vals[input_proj_mask] ** 2)) if input_proj_mask.sum() > 0 else 0.1
        input_boost = input_target_rms / max(input_current_rms, 1e-8)
        input_boost = np.clip(input_boost, 1.0, 6.0)  # Safety bounds
        weight_vals[input_proj_mask] *= input_boost
        print(f"Input boost: {input_boost:.2f}x (calibrated to 3x lateral RMS={lateral_rms:.4f})")

        n_input = int(input_proj_mask.sum())
        n_free = int(free_mask.sum())
        print(f"Edges: {n_input} input-proj (2x, outside SR), {n_free} free (SR-tuned to {target_sr})")
        print(f"Weight magnitudes: input={np.abs(weight_vals[input_proj_mask]).mean():.4f}, "
              f"free={np.abs(weight_vals[free_mask]).mean():.4f}")

        # Build node-to-module lookup
        self.node_to_module = np.full(num_nodes, -1, dtype=np.int64)
        for mod in self.modules:
            self.node_to_module[mod['start']:mod['end']] = mod['id']

        # Balance Hierarchies: Shift fan-in normalization to Level 0 lateral connections.
        node_levels = np.zeros(num_nodes, dtype=int)
        node_levels[:n_io] = -1  # I/O nodes conceptually at level -1
        for mod in self.modules:
            node_levels[mod['start']:mod['end']] = mod['level']
        src_levels_arr = node_levels[edge_rows]
        dst_levels_arr = node_levels[edge_cols]
        
        # Lateral Level 0 edges (excluding I/O nodes mapping)
        lat_l0_mask = (src_levels_arr == dst_levels_arr) & (src_levels_arr == 0) & (edge_rows >= n_io) & (edge_cols >= n_io)

        # Fan-in normalized Level-0 lateral weights to prevent input-saturation cascade
        lat_dst_nodes = edge_cols[lat_l0_mask]
        unique_dst, dst_counts = np.unique(lat_dst_nodes, return_counts=True)
        fan_in_map = dict(zip(unique_dst, dst_counts))

        lat_fan_in = np.array([fan_in_map.get(d, 1) for d in lat_dst_nodes], dtype=np.float32)
        # Bound fan-in normalization to prevent weights decaying to 0.00
        max_effective_fan_in_lat = 64.0
        bounded_lat_fan_in = np.minimum(lat_fan_in, max_effective_fan_in_lat)
        
        lat_src_inh = self.is_inhibitory[edge_rows[lat_l0_mask]]
        lat_signs = np.where(lat_src_inh, -1.0, 1.0)
        weight_vals[lat_l0_mask] = np.abs(weight_vals[lat_l0_mask]) / np.sqrt(bounded_lat_fan_in) * lat_signs
        
        # Massive Recurrent Excitation for Level 2 and Level 3 intra-module connections (Prefrontal)
        l23_intra_mask = (src_levels_arr == dst_levels_arr) & (src_levels_arr >= 2) & free_mask
        # Only boost connections within the *same* module for working memory 
        l23_same_mod = self.node_to_module[edge_rows[l23_intra_mask]] == self.node_to_module[edge_cols[l23_intra_mask]]
        
        # Ensure we can update the correct edges
        l23_indices = np.where(l23_intra_mask)[0]
        same_mod_indices = l23_indices[l23_same_mod]
        
        # Level 2 gets 3x boost, Level 3 gets 5x boost for working memory persistence
        for idx in same_mod_indices:
            lvl = src_levels_arr[idx]
            boost = 3.0 if lvl == 2 else 5.0
            sign = -1.0 if self.is_inhibitory[edge_rows[idx]] else 1.0
            weight_vals[idx] = np.abs(weight_vals[idx]) * boost * sign
            
        print(f"Boosted recurrent excitation for {len(same_mod_indices)} high-level intra-module working memory edges.")

        # Enforce minimum weight magnitude for lateral signal propagation
        lat_min_magnitude = 0.01
        weight_vals[lat_l0_mask] = np.maximum(np.abs(weight_vals[lat_l0_mask]), lat_min_magnitude) * lat_signs
        
        n_lat = int(lat_l0_mask.sum())
        if len(fan_in_map) > 0:
            avg_fan_in = np.mean(list(fan_in_map.values()))
            print(f"Level 0 lateral edges: {n_lat} made positive, fan-in normalized (avg fan-in={avg_fan_in:.0f})")

        # Boost Bottom-Up edges to prevent vanishing activations across hierarchy
        # Without this, the SR=0.85 tuning makes the signal decay at each level,
        # leaving L1-L3 completely dead initially so they can never learn.
        # Only boost internal hierarchy (src >= 512), not I/O feedback.
        bu_mask = (src_levels_arr < dst_levels_arr) & free_mask & (edge_rows >= 512)
        
        # Fan-in normalize the bottom-up edges without artificial static amplification.
        bu_dst_nodes = edge_cols[bu_mask]
        unique_bu_dst, bu_dst_counts = np.unique(bu_dst_nodes, return_counts=True)
        bu_fan_in_map = dict(zip(unique_bu_dst, bu_dst_counts))
        bu_fan_in = np.array([bu_fan_in_map.get(d, 1) for d in bu_dst_nodes], dtype=np.float32)
        
        # Bound fan-in normalization to prevent weights decaying to 0.00
        # for high fan-in nodes. Cap the effective fan-in divisor at 64.
        max_effective_fan_in_bu = 64.0
        bounded_bu_fan_in = np.minimum(bu_fan_in, max_effective_fan_in_bu)
        weight_vals[bu_mask] = weight_vals[bu_mask] / np.sqrt(bounded_bu_fan_in)
        
        # THALAMIC BOOST: Amplify Bottom-Up signals feeding into higher levels
        # to prevent Signal Death before they reach Level 3 working memory
        bu_dst_levels_arr = src_levels_arr[edge_cols[bu_mask]]
        for i, idx in enumerate(np.where(bu_mask)[0]):
            lvl = dst_levels_arr[idx]
            if lvl == 1:
                weight_vals[idx] *= 2.0
            elif lvl == 2:
                weight_vals[idx] *= 4.0
            elif lvl == 3:
                weight_vals[idx] *= 8.0
                
        # Enforce minimum weight magnitude to guarantee upward signal flow
        bu_min_magnitude = 0.01
        bu_signs = np.sign(weight_vals[bu_mask])
        bu_signs[bu_signs == 0] = 1.0  # default to positive for zero weights
        weight_vals[bu_mask] = np.maximum(np.abs(weight_vals[bu_mask]), bu_min_magnitude) * bu_signs
        print(f"Bottom-up edges: {int(bu_mask.sum())} fan-in normalized (bounded, min_mag={bu_min_magnitude})")

        # FIX 4: Decouple & enforce symmetric motor connections (Motor -> Level 0)
        fwd_motor_mask = (edge_rows >= 512) & (edge_cols >= 256) & (edge_cols < 512)
        fb_motor_mask = (edge_rows >= 256) & (edge_rows < 512) & (edge_cols >= 512)

        fwd_idx = np.where(fwd_motor_mask)[0]
        fb_idx = np.where(fb_motor_mask)[0]

        fwd_ids = edge_rows[fwd_idx] * num_nodes + edge_cols[fwd_idx]
        fb_ids = edge_cols[fb_idx] * num_nodes + edge_rows[fb_idx]  # transpose to match

        fwd_sort = np.argsort(fwd_ids)
        fb_sort = np.argsort(fb_ids)

        weight_vals[fb_idx[fb_sort]] = weight_vals[fwd_idx[fwd_sort]]
        print(f"Tied {len(fb_idx)} motor feedback edges to their forward counterparts (W_fb = W_fwd^T).")

        # Build final sparse matrix
        self.weight_sparse = sp.csr_matrix(
            (weight_vals, (edge_rows, edge_cols)),
            shape=(num_nodes, num_nodes)
        )

        # --- Timescale Assignment (per-module, per-level) ---
        # Massive Range of Intrinsic Timescales corresponding to NMDA vs AMPA
        tau_by_level = {0: 1.0, 1: 5.0, 2: 25.0, 3: 100.0}

        self.taus = np.zeros(num_nodes)
        # I/O nodes: fast
        self.taus[:n_io] = 0.1
        # Association nodes: per-module by level
        for mod in self.modules:
            level = mod['level']
            self.taus[mod['start']:mod['end']] = tau_by_level[level]

        # State and bias initialization
        self.states = np.zeros(num_nodes)
        self.biases = np.random.uniform(-0.01, 0.01, num_nodes)

        # Store hierarchical connection metadata for predictive coding
        # Top-down weight indices: for each (upper_mod, lower_mod) pair,
        # store the edge indices in the sparse representation
        self._build_hierarchical_index(edge_rows, edge_cols)

        # --- Complementary Learning Systems ---
        # Mark ~20% of modules at each level as hippocampal (fast learners)
        # Remaining 80% are neocortical (slow learners that extract regularities)
        self.hippocampal_modules = set()
        self.neocortical_modules = set()
        for level in range(num_levels):
            level_mods = self.level_modules[level]
            n_hippo = max(1, int(len(level_mods) * 0.2))
            for mod_id in level_mods[:n_hippo]:
                self.hippocampal_modules.add(mod_id)
            for mod_id in level_mods[n_hippo:]:
                self.neocortical_modules.add(mod_id)
        print(f"CLS: {len(self.hippocampal_modules)} hippocampal modules, "
              f"{len(self.neocortical_modules)} neocortical modules")

        # Build connectivity map between hippocampal and neocortical modules
        self._build_cls_connectivity(edge_rows, edge_cols)

    def _build_cls_connectivity(self, edge_rows, edge_cols):
        """Build map of which neocortical modules each hippocampal module connects to."""
        self.hippo_to_neo = {h: set() for h in self.hippocampal_modules}
        for r, c in zip(edge_rows, edge_cols):
            src_mod = int(self.node_to_module[r])
            dst_mod = int(self.node_to_module[c])
            if src_mod in self.hippocampal_modules and dst_mod in self.neocortical_modules:
                self.hippo_to_neo[src_mod].add(dst_mod)
            elif dst_mod in self.hippocampal_modules and src_mod in self.neocortical_modules:
                self.hippo_to_neo[dst_mod].add(src_mod)

    def get_connected_neocortical(self, hippo_mod_id):
        """Return set of neocortical module IDs connected to a hippocampal module."""
        return self.hippo_to_neo.get(hippo_mod_id, set())

    def _build_hierarchical_index(self, edge_rows, edge_cols):
        """
        Build index structures for predictive coding:
        - topdown_pairs: list of (upper_mod_id, lower_mod_id) pairs
        - For each pair, the set of node indices in upper and lower modules
        """
        self.hier_pairs = []  # (upper_mod_id, lower_mod_id)
        for level in range(self.num_levels - 1):
            lower_mods = self.level_modules[level]
            upper_mods = self.level_modules[level + 1]
            for lm_id in lower_mods:
                for um_id in upper_mods:
                    self.hier_pairs.append((um_id, lm_id))

    def get_module_ranges(self):
        """
        Returns list of (start, end) tuples for each module's node range.
        Used by the engine for computing per-module prediction errors.
        """
        return [(mod['start'], mod['end']) for mod in self.modules]

    def get_module_level(self, module_id):
        return self.modules[module_id]['level']

    def export_sparse_components(self):
        """
        Exports the graph weights as sparse components (indices, values).
        Returns:
            indices (np.ndarray): 2xE array of edge indices (int64).
            values (np.ndarray): 1xE array of edge weights (float32).
        """
        coo = self.weight_sparse.tocoo()
        indices = np.stack([coo.row.astype(np.int64), coo.col.astype(np.int64)])
        values = coo.data.astype(np.float32)
        return indices, values

    @property
    def hippocampal_modules_set(self):
        """Returns the set of hippocampal module IDs."""
        return self.hippocampal_modules

    @property
    def neocortical_modules_set(self):
        """Returns the set of neocortical module IDs."""
        return self.neocortical_modules

    @property
    def hippo_to_neo_mapping(self):
        """Returns the mapping of hippocampal to connected neocortical module IDs."""
        return self.hippo_to_neo
