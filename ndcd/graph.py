
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
        # Distribution: more modules at lower levels (pyramid shape)
        # Level 0: 40%, Level 1: 30%, Level 2: 20%, Level 3: 10%
        level_fractions = [0.40, 0.30, 0.20, 0.10]
        level_module_counts = []
        remaining = num_modules
        for i in range(num_levels - 1):
            count = max(1, int(num_modules * level_fractions[i]))
            level_module_counts.append(count)
            remaining -= count
        level_module_counts.append(max(1, remaining))

        # Build module metadata
        self.modules = []  # List of dicts: {level, start_idx, end_idx, node_indices}
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

                self.modules.append({
                    'id': module_id,
                    'level': level,
                    'start': start,
                    'end': end,
                    'indices': node_indices,
                    'size': n_in_module
                })
                self.module_levels.append(level)
                self.level_modules[level].append(module_id)

                current_node = end
                module_id += 1

        self.num_actual_modules = module_id
        self.module_levels = np.array(self.module_levels)

        # --- Spatial Embedding ---
        self.pos = np.random.rand(num_nodes, 3)

        # --- Build Sparse Weight Matrix ---
        # We build edge lists directly for efficiency
        edge_rows = []
        edge_cols = []

        # 1. INTRA-module connectivity
        # Reduced from 200 connections/node (60% density) to 50 (25%).
        # At 200, lateral recurrence outnumbered inter-level edges 25:1,
        # making the hierarchy structurally disconnected — higher levels
        # were completely input-invariant. Target ~3:1 lateral:hierarchical.
        target_connections_per_node = 50
        for mod in self.modules:
            idx = mod['indices']
            n = len(idx)
            if n < 2:
                continue
            # Adaptive density: min(0.25, target_conn / (n-1))
            intra_density = min(0.25, target_connections_per_node / max(1, n - 1))

            if n <= 300:
                # Small module: use dense random matrix
                conn = np.random.rand(n, n) < intra_density
                np.fill_diagonal(conn, False)
                local_rows, local_cols = np.nonzero(conn)
            else:
                # Large module: sample edges directly to avoid n*n memory
                n_edges_target = int(n * target_connections_per_node)
                local_rows = np.random.randint(0, n, n_edges_target)
                local_cols = np.random.randint(0, n, n_edges_target)
                # Remove self-loops
                valid = local_rows != local_cols
                local_rows = local_rows[valid]
                local_cols = local_cols[valid]

            edge_rows.extend(idx[local_rows].tolist())
            edge_cols.extend(idx[local_cols].tolist())

        # 2. Sparse INTER-module connectivity at same level (~3%)
        inter_same_density = 0.03
        for level in range(num_levels):
            mods_at_level = self.level_modules[level]
            for i in range(len(mods_at_level)):
                for j in range(i + 1, len(mods_at_level)):
                    mod_i = self.modules[mods_at_level[i]]
                    mod_j = self.modules[mods_at_level[j]]
                    idx_i = mod_i['indices']
                    idx_j = mod_j['indices']
                    # Sparse random connections between module pairs
                    n_possible = len(idx_i) * len(idx_j)
                    n_connections = max(1, int(n_possible * inter_same_density))
                    # Cap to avoid excessive edges between large modules
                    n_connections = min(n_connections, max(10, len(idx_i) + len(idx_j)))
                    src_picks = np.random.choice(idx_i, n_connections, replace=True)
                    dst_picks = np.random.choice(idx_j, n_connections, replace=True)
                    edge_rows.extend(src_picks.tolist())
                    edge_cols.extend(dst_picks.tolist())
                    # Bidirectional
                    edge_rows.extend(dst_picks.tolist())
                    edge_cols.extend(src_picks.tolist())

        # 3. Hierarchical connections between levels — MUCH denser than lateral.
        # At 2% density with aggressive capping, each module pair had only
        # ~2-3 inter-level edges, making the hierarchy structurally disconnected.
        # L1-L3 settled to input-invariant states (cos_sim=1.0) because signal
        # couldn't climb or descend. 15% density gives real pathways.
        hier_density = 0.15
        for level in range(num_levels - 1):
            lower_mods = self.level_modules[level]
            upper_mods = self.level_modules[level + 1]
            for lm_id in lower_mods:
                for um_id in upper_mods:
                    mod_lower = self.modules[lm_id]
                    mod_upper = self.modules[um_id]
                    idx_lower = mod_lower['indices']
                    idx_upper = mod_upper['indices']
                    n_possible = len(idx_lower) * len(idx_upper)
                    n_connections = max(5, int(n_possible * hier_density))
                    # Let density control the count — no aggressive cap
                    n_connections = min(n_connections, n_possible)
                    # Top-down: upper -> lower
                    src_picks = np.random.choice(idx_upper, n_connections, replace=True)
                    dst_picks = np.random.choice(idx_lower, n_connections, replace=True)
                    edge_rows.extend(src_picks.tolist())
                    edge_cols.extend(dst_picks.tolist())
                    # Bottom-up: lower -> upper
                    edge_rows.extend(dst_picks.tolist())
                    edge_cols.extend(src_picks.tolist())

        # 4. Input projections: sensory nodes -> level-0 modules
        level0_mods = self.level_modules[0]
        for mod_id in level0_mods:
            mod = self.modules[mod_id]
            idx = mod['indices']
            # Each sensory node connects to a subset of level-0 module nodes
            n_proj = max(1, len(idx) // 4)
            for s in range(n_sensory):
                targets = np.random.choice(idx, n_proj, replace=False)
                edge_rows.extend([s] * n_proj)
                edge_cols.extend(targets.tolist())
                # Bidirectional for settling
                edge_rows.extend(targets.tolist())
                edge_cols.extend([s] * n_proj)

        # 5. Output projections: level-0 modules -> motor nodes
        for mod_id in level0_mods:
            mod = self.modules[mod_id]
            idx = mod['indices']
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

        weight_vals = np.random.normal(0, std_dev, num_edges).astype(np.float32)

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

        target_sr = 0.95
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

        # Set input projection edges independently (5x Xavier).
        # These are external forcing (clamped nodes), not recurrence.
        weight_vals[input_proj_mask] *= 5.0

        n_input = int(input_proj_mask.sum())
        n_free = int(free_mask.sum())
        print(f"Edges: {n_input} input-proj (5x, outside SR), {n_free} free (SR-tuned to {target_sr})")
        print(f"Weight magnitudes: input={np.abs(weight_vals[input_proj_mask]).mean():.4f}, "
              f"free={np.abs(weight_vals[free_mask]).mean():.4f}")

        # Balance Hierarchies: Shift fan-in normalization to Level 0 lateral connections.
        node_levels = np.zeros(num_nodes, dtype=int)
        node_levels[:n_io] = 0
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
        weight_vals[lat_l0_mask] = np.abs(weight_vals[lat_l0_mask]) / np.sqrt(lat_fan_in)
        
        n_lat = int(lat_l0_mask.sum())
        if len(fan_in_map) > 0:
            avg_fan_in = np.mean(list(fan_in_map.values()))
            print(f"Level 0 lateral edges: {n_lat} made positive, fan-in normalized (avg fan-in={avg_fan_in:.0f})")

        # Build final sparse matrix
        self.weight_sparse = sp.csr_matrix(
            (weight_vals, (edge_rows, edge_cols)),
            shape=(num_nodes, num_nodes)
        )

        # --- Timescale Assignment (per-module, per-level) ---
        # Reduced from {0.1, 0.75, 5.0, 25.0} to {0.1, 0.5, 2.0, 8.0}.
        # With dt=0.5 and tau=25.0, each IMEX step moves L3 by only
        # 0.02× its input signal — after 20 steps L3 integrates just 0.4×.
        # Tau ratio 80:1 still gives meaningful timescale separation while
        # letting L3 respond within reasonable settle step counts.
        tau_by_level = {0: 0.1, 1: 0.5, 2: 2.0, 3: 8.0}

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

        # Build node-to-module lookup
        self.node_to_module = np.full(num_nodes, -1, dtype=np.int64)
        for mod in self.modules:
            self.node_to_module[mod['start']:mod['end']] = mod['id']

        # Build connectivity map between hippocampal and neocortical modules
        self._build_cls_connectivity(edge_rows, edge_cols)

    def _build_cls_connectivity(self, edge_rows, edge_cols):
        """Build map of which neocortical modules each hippocampal module connects to."""
        self.hippo_to_neo = {h: set() for h in self.hippocampal_modules}
        for r, c in zip(edge_rows, edge_cols):
            src_mod = self.node_to_module[r]
            dst_mod = self.node_to_module[c]
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
