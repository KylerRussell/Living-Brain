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
        self.modules = []
        self.module_levels = []
        self.level_modules = {i: [] for i in range(num_levels)}

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
        
        # --- Total Capacity Check for 5:1 Expansion ---
        # We need to ensure we have enough association nodes for the 5:1 expansion of the DG layer.
        # If DG is 1 module and neocortex has 49 modules, we need to balance the node count.

        # Build module metadata
        # --- Complementary Learning Systems (CLS) Module Designation ---
        # Pick 1 module at Level 2 (medial temporal lobe equivalent) to be Hippocampus.
        nodes_per_module = n_association // num_modules
        extra_nodes = n_association % num_modules
        
        self.hippocampal_modules = set()
        self.neocortical_modules = set()

        # Build node-to-module lookup (pre-allocate)
        self.node_to_module = np.full(num_nodes, -1, dtype=np.int64)

        self.dg_indices = []
        self.ca3_indices = []
        
        current_node = n_io  # Start after I/O nodes
        module_id = 0
        hippo_module_id = -1
        
        # Pre-determine which module is hippocampal (first module of level 2)
        target_hippo_idx = sum(level_module_counts[:2]) # First module index in level 2
        
        for level in range(num_levels):
            for _ in range(level_module_counts[level]):
                if module_id == target_hippo_idx:
                    self.hippocampal_modules.add(module_id)
                    hippo_module_id = module_id
                else:
                    self.neocortical_modules.add(module_id)
                    
                # Distribute extra nodes
                if module_id == hippo_module_id:
                    # Item 3: DG-Inspired Sparse Expansion Layer (10:1 Expansion for DG)
                    # Entorhinal (Input) is 256 nodes. DG should be 2560 nodes (10:1).
                    n_l4 = 2560 
                    n_l23 = 256  # CA3
                    n_l56 = 256  # CA1
                    n_in_module = n_l4 + n_l23 + n_l56
                else:
                    # Uniform Neocortical Lamination
                    n_in_module = nodes_per_module + (1 if module_id < extra_nodes else 0)
                    n_l4 = max(1, int(n_in_module * 0.2))
                    n_l23 = max(1, int(n_in_module * 0.4))
                    n_l56 = n_in_module - n_l4 - n_l23

                l4_start = current_node
                l23_start = l4_start + n_l4
                l56_start = l23_start + n_l23
                end = l56_start + n_l56

                l4_indices = np.arange(l4_start, l4_start + n_l4)
                l23_indices = np.arange(l23_start, l23_start + n_l23)
                l56_indices = np.arange(l56_start, l56_start + n_l56)

                if module_id == hippo_module_id:
                    self.dg_indices.extend(l4_indices.tolist())
                    self.ca3_indices.extend(l23_indices.tolist())

                self.modules.append({
                    'id': module_id,
                    'level': level,
                    'start': l4_start,
                    'end': end,
                    'indices': np.arange(l4_start, end),
                    'size': end - l4_start,
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
        print(f"CLS: {len(self.hippocampal_modules)} hippocampal modules, "
              f"{len(self.neocortical_modules)} neocortical modules")

        # --- Dale's Law E/I Assignment & Tripartite Interneurons ---
        self.is_inhibitory = np.zeros(num_nodes, dtype=bool)
        self.is_pv = np.zeros(num_nodes, dtype=bool)
        self.is_sst = np.zeros(num_nodes, dtype=bool)
        self.is_vip = np.zeros(num_nodes, dtype=bool)
        self.is_neg_pe = np.zeros(num_nodes, dtype=bool)
        
        for mod in self.modules:
            # 20% of nodes in each module are inhibitory
            n_inh = int(mod['size'] * 0.2)
            if n_inh > 0:
                inh_idx = np.random.choice(mod['indices'], n_inh, replace=False)
                self.is_inhibitory[inh_idx] = True
                
                # Subdivide inhibitory nodes into 3 groups (PV, SST, VIP)
                # Typically PV (~40%), SST (~30%), VIP (~30%)
                n_pv = int(n_inh * 0.4)
                n_sst = int(n_inh * 0.3)
                
                np.random.shuffle(inh_idx)
                if n_pv > 0:
                    self.is_pv[inh_idx[:n_pv]] = True
                if n_sst > 0:
                    self.is_sst[inh_idx[n_pv:n_pv+n_sst]] = True
                if n_pv + n_sst < n_inh:
                    self.is_vip[inh_idx[n_pv+n_sst:]] = True
            
            # --- Item 4: Asymmetric Predictive Coding (PEONs) ---
            # Designate half of L2/3 as Negative Prediction Error neurons (PEONs)
            l23_idx = mod['l23_indices']
            if len(l23_idx) > 0:
                n_neg = len(l23_idx) // 2
                self.is_neg_pe[l23_idx[:n_neg]] = True

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
        # Structural sparsity constraint: Target < 1% connectivity internally for large networks,
        # but scales dynamically to support CA3-style specific recurrence in small microcircuits.
        # L4 -> L2/3, L2/3 -> L5/6, L5/6 -> L5/6 (recurrent), L5/6 -> L4
        for mod in self.modules:
            # We want connections between specific layers
            l4 = mod['l4_indices']
            l23 = mod['l23_indices']
            l56 = mod['l56_indices']
            
            if mod['id'] in self.hippocampal_modules:
                # Hippocampal Specific Microcircuitry (Topological Divergence)
                # DG (L4) -> CA3 (L23) -> CA1 (L56)
                
                # 1. DG (L4) receives inputs (from EC, constructed later in hierarchical edges)
                
                # 2. DG Mossy Fibers -> CA3 (L4 -> L2/3). Extreme sparsity (e.g. 1%)
                if len(l4) > 0 and len(l23) > 0:
                    target_conns = max(2, int(len(l23) * 0.01))
                    n_edges = int(len(l4) * target_conns)
                    src = np.random.choice(l4, n_edges, replace=True)
                    dst = np.random.choice(l23, n_edges, replace=True)
                    edge_rows.extend(src.tolist())
                    edge_cols.extend(dst.tolist())
                    
                # 3. CA3 Recurrent Collaterals (L2/3 -> L2/3). 
                # "Diluted connectivity" (sparse recurrent attractor)
                if len(l23) > 0:
                    density = 0.15 # Diluted relative to dense neocortex, but strong enough for pattern completion
                    target_conns = max(5, int(len(l23) * density))
                    n_edges = int(len(l23) * target_conns)
                    src = np.random.choice(l23, n_edges, replace=True)
                    dst = np.random.choice(l23, n_edges, replace=True)
                    valid = src != dst
                    edge_rows.extend(src[valid].tolist())
                    edge_cols.extend(dst[valid].tolist())
                    
                # 4. CA3 Schaffer Collaterals -> CA1 (L2/3 -> L5/6).
                if len(l23) > 0 and len(l56) > 0:
                    target_conns = max(5, int(len(l56) * 0.20))
                    n_edges = int(len(l23) * target_conns)
                    src = np.random.choice(l23, n_edges, replace=True)
                    dst = np.random.choice(l56, n_edges, replace=True)
                # 5. CA3 (L23) -> DG (L4) Backprojections (Inhibitory for pattern separation)
                if len(l23) > 0 and len(l4) > 0:
                    target_conns = max(2, int(len(l4) * 0.05))
                    n_edges = int(len(l23) * target_conns)
                    src = np.random.choice(l23, n_edges, replace=True)
                    dst = np.random.choice(l4, n_edges, replace=True)
                    # These will be marked as inhibitory later based on is_inhibitory,
                    # but for this specific microcircuit, we force them to be inhibitory
                    # by adding to the lists and then ensuring signs are correct later.
                    edge_rows.extend(src.tolist())
                    edge_cols.extend(dst.tolist())
                    # Mark sources as inhibitory if not already 
                    # (Better approach: just ensure the weights will be negative)
                    # For now, we'll rely on the src_is_inh check in DynamicGraph.
                # Standard Neocortical Microcircuitry
                # L4 -> L2/3
                if len(l4) > 0 and len(l23) > 0:
                    # Broca modules: +10% L4 density for faster sensory throughput
                    l4_density = 0.33 if mod['id'] in getattr(self, 'broca_modules', set()) else 0.3
                    target_conns = max(10, int(len(l23) * l4_density))
                    n_edges = int(len(l4) * target_conns)
                    src = np.random.choice(l4, n_edges, replace=True)
                    dst = np.random.choice(l23, n_edges, replace=True)
                    edge_rows.extend(src.tolist())
                    edge_cols.extend(dst.tolist())
                    
                # L2/3 -> L5/6
                if len(l23) > 0 and len(l56) > 0:
                    target_conns = max(10, int(len(l56) * 0.3))
                    n_edges = int(len(l23) * target_conns)
                    src = np.random.choice(l23, n_edges, replace=True)
                    dst = np.random.choice(l56, n_edges, replace=True)
                    edge_rows.extend(src.tolist())
                    edge_cols.extend(dst.tolist())
                    
                # L5/6 -> L5/6 (recurrent)
                if len(l56) > 0:
                    depth_factor = mod['level'] / max(1, self.num_levels - 1)
                    # Level 3 gets ultra-dense recurrent excitation (up to 80%)
                    recurrent_density = 0.1 + 0.70 * depth_factor
                    # Wernicke modules: boost L5/6 recurrence to 50% base for
                    # enhanced semantic persistence (was 40% = 0.1 + 0.70*0)
                    if mod['id'] in getattr(self, 'wernicke_modules', set()):
                        recurrent_density = max(recurrent_density, 0.50)
                    target_conns = max(10, int(len(l56) * recurrent_density))
                    n_edges = int(len(l56) * target_conns)
                    src = np.random.choice(l56, n_edges, replace=True)
                    dst = np.random.choice(l56, n_edges, replace=True)
                    valid = src != dst
                    edge_rows.extend(src[valid].tolist())
                    edge_cols.extend(dst[valid].tolist())
                    
                # L5/6 -> L4 (feedback within column)
                if len(l56) > 0 and len(l4) > 0:
                    target_conns = max(10, int(len(l4) * 0.3))
                    n_edges = int(len(l56) * target_conns)
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
                    
                    # --- Item 4: PEON Lateral Inhibition (Omission Signaling) ---
                    # PEON streams in L2/3 connect laterally via inhibitory synapses
                    pe_i = mod_i['l23_indices'][self.is_neg_pe[mod_i['l23_indices']]]
                    pe_j = mod_j['l23_indices'][self.is_neg_pe[mod_j['l23_indices']]]
                    if len(pe_i) > 0 and len(pe_j) > 0:
                        n_peon = max(2, int(len(pe_i) * 0.15)) # 15% lateral PEON connectivity
                        src_p = np.random.choice(pe_i, n_peon, replace=True)
                        dst_p = np.random.choice(pe_j, n_peon, replace=True)
                        edge_rows.extend(src_p.tolist())
                        edge_cols.extend(dst_p.tolist())
                        edge_rows.extend(dst_p.tolist())
                        edge_cols.extend(src_p.tolist())

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

        # 4. Input projections: sensory nodes -> level-0 modules AND Hippo DG
        # Item 4: Input (EC) projects to expansion layer (DG) with 1:5 ratio (1280 nodes)
        target_input_mods = set(self.level_modules[0])
        if hippo_module_id != -1:
            target_input_mods.add(hippo_module_id)
            
        for mod_id in target_input_mods:
            mod = self.modules[mod_id]
            idx = mod['l4_indices']  # Project into L4
            if len(idx) == 0: continue
            
            # Sparse Random weights for expansion (Item 4)
            # n_proj set to ~5% density for pattern separation
            n_proj = max(1, int(len(idx) * 0.05)) 
            for s in range(n_sensory):
                targets = np.random.choice(idx, n_proj, replace=False)
                edge_rows.extend([s] * n_proj)
                edge_cols.extend(targets.tolist())
                # Reciprocal feedback for stability
                edge_rows.extend(targets.tolist())
                edge_cols.extend([s] * n_proj)

        # 5. Output projections: level-0 modules (L5/6) -> motor nodes
        for mod_id in self.level_modules[0]:
            mod = self.modules[mod_id]
            idx = mod['l56_indices']  # Output comes from L5/6
            if len(idx) == 0: continue
            n_proj = max(1, len(idx) // 4)
            for m in range(n_motor):
                sources = np.random.choice(idx, n_proj, replace=False)
                motor_node = n_sensory + m
                edge_rows.extend(sources.tolist())
                edge_cols.extend([motor_node] * n_proj)
                
        # 6. DKP-PC: Direct feedback from output (motor) to ALL hidden modules
        # This allows O(1) error propagation from output to deep layers.
        dkp_feedback_density = 0.05
        for mod in self.modules:
            # Feedback goes to L4 (input recipient) or L2/3 (error units)
            idx = np.concatenate([mod['l4_indices'], mod['l23_indices']])
            if len(idx) == 0: continue
            n_proj = max(1, int(len(idx) * dkp_feedback_density))
            for m in range(n_motor):
                motor_node = n_sensory + m
                targets = np.random.choice(idx, n_proj, replace=False)
                edge_rows.extend([motor_node] * n_proj)
                edge_cols.extend(targets.tolist())

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
        
        # --- Item 2: Lateral Inhibition Hierarchy ---
        # Enforce a strict hierarchy where lateral inhibition (PN -> IN -> neighbor PN) 
        # is weighted 10 times more heavily than recurrent inhibition (PN -> IN -> same PN).
        # We approximate this by looking at inter-module vs intra-module inhibitory edges.
        src_modules = self.node_to_module[edge_rows]
        dst_modules = self.node_to_module[edge_cols]
        is_lateral_inh = src_is_inh & (src_modules != dst_modules) & (src_modules != -1) & (dst_modules != -1)
        is_recurrent_inh = src_is_inh & (src_modules == dst_modules) & (src_modules != -1)
        
        weight_vals[is_lateral_inh] *= 10.0
        weight_vals[is_recurrent_inh] *= 1.0 # Base weight for recurrent
        
        # --- Solution 2: Spatial Lateral Inhibition (Mexican-hat mask) ---
        # Apply spatial mask to PV+ interneuron connectivity
        # Ensures strong lateral inhibition to neighboring nodes while minimizing recurrent self-inhibition
        src_is_pv = self.is_pv[edge_rows]
        if src_is_pv.any():
            pos_src = self.pos[edge_rows[src_is_pv]]
            pos_dst = self.pos[edge_cols[src_is_pv]]
            distances = np.linalg.norm(pos_src - pos_dst, axis=1)
            
            # Mexican-hat ring: r^2 * exp(-r^2 / sigma^2)
            # Peaks at distance = sigma.
            sigma_spatial = 0.2
            spatial_mask = (distances**2 / sigma_spatial**2) * np.exp(-distances**2 / sigma_spatial**2) * np.e
            
            # Apply mask to PV+ weights, amplifying surround ring and suppressing center
            weight_vals[src_is_pv] *= spatial_mask * 5.0

        weight_vals[src_is_inh] = -np.abs(weight_vals[src_is_inh]) * 0.1

        # --- Item 4: Lateral PEON Inhibition ---
        src_is_peon = self.is_neg_pe[edge_rows]
        dst_is_peon = self.is_neg_pe[edge_cols]
        # Lateral inhibitory connections between PEON streams across different modules
        is_lat_peon = src_is_peon & dst_is_peon & (src_modules != dst_modules) & (src_modules != -1) & (dst_modules != -1)
        weight_vals[is_lat_peon] = -np.abs(weight_vals[is_lat_peon]) * 0.5 # Strong lateral inhibition

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

        target_sr = 1.15
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
        input_target_rms = 6.0 * lateral_rms  # 6.0x lateral (MAX boost for firing breakout)
        input_current_rms = np.sqrt(np.mean(weight_vals[input_proj_mask] ** 2)) if input_proj_mask.sum() > 0 else 0.1
        input_boost = input_target_rms / max(input_current_rms, 1e-8)
        input_boost = np.clip(input_boost, 0.25, 6.0)  # Safety bounds
        weight_vals[input_proj_mask] *= input_boost
        print(f"Input boost: {input_boost:.2f}x (calibrated to 3.0x lateral RMS={lateral_rms:.4f})")

        n_input = int(input_proj_mask.sum())
        n_free = int(free_mask.sum())
        print(f"Edges: {n_input} input-proj (2x, outside SR), {n_free} free (SR-tuned to {target_sr})")

        # --- Scale-Invariant E/I Balancing (1/sqrt(K)) ---
        # Scale synaptic strengths proportionally to 1/sqrt(K), where K is the in-degree.
        # This ensures that nodes with high fan-in don't saturate.
        unique_dst, dst_counts = np.unique(edge_cols, return_counts=True)
        fan_in_map = dict(zip(unique_dst, dst_counts))
        fan_in = np.array([fan_in_map.get(d, 1) for d in edge_cols], dtype=np.float32)
        
        # Apply 1/sqrt(K) scaling to free edges (SR-tuned weights already have some scaling)
        # We blend this with the existing weights to maintain the spectral radius property
        # while enforcing the fan-in balance.
        k_scaling = 1.0 / np.sqrt(np.maximum(fan_in[free_mask], 1.0))
        # Normalize k_scaling to preserve mean magnitude
        k_scaling /= (np.mean(k_scaling) + 1e-8)
        weight_vals[free_mask] *= k_scaling
        
        print(f"Scale-invariant E/I balancing (1/sqrt(K)) applied to {n_free} free edges.")
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
        
        # Hierarchical Recurrent Excitation Gradient (Chaudhuri 2015)
        # w_EE(l) = w_base * (1 + η * h_l) where η ≈ 0.68
        # This increases topic persistence and reverberatory activity in higher layers.
        recurrent_e_mask = (src_levels_arr == dst_levels_arr) & (~src_is_inh) & free_mask
        # Only boost connections within the *same* module for working memory
        same_mod_mask = self.node_to_module[edge_rows] == self.node_to_module[edge_cols]
        target_indices = np.where(recurrent_e_mask & same_mod_mask)[0]
        
        eta = 0.68
        max_h_lvl = float(self.num_levels - 1)
        for idx in target_indices:
            h_l = src_levels_arr[idx] / max_h_lvl if max_h_lvl > 0 else 0.0
            boost = 1.0 + eta * h_l
            weight_vals[idx] *= boost
            
        print(f"Applied hierarchical recurrent excitatory gradient (eta={eta}) to {len(target_indices)} edges.")

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

        # Fix 4 removed: Uncoupled Product Feedback Alignment (PFA)
        # Weights are left completely asymmetric.
        fwd_motor_mask = (edge_rows >= 512) & (edge_cols >= 256) & (edge_cols < 512)
        fb_motor_mask = (edge_rows >= 256) & (edge_rows < 512) & (edge_cols >= 512)
        print(f"PFA: {fb_motor_mask.sum()} independent feedback edges enabling Product Feedback Alignment.")

        # Ensure DG/CA3 indices are arrays
        self.dg_indices = np.array(self.dg_indices, dtype=np.int64)
        self.ca3_indices = np.array(self.ca3_indices, dtype=np.int64)

        # Apply Detonator Synapses Boost (DG -> CA3)
        dg_mask = np.isin(edge_rows, self.dg_indices)
        ca3_mask = np.isin(edge_cols, self.ca3_indices)
        detonator_mask = dg_mask & ca3_mask
        # Mossy Fibers are extremely powerful ("detonators")
        weight_vals[detonator_mask] *= 30.0
        if detonator_mask.sum() > 0:
            print(f"Applied 10x detonator boost to {detonator_mask.sum()} DG->CA3 synapses.")

        # Build final sparse matrix
        self.weight_sparse = sp.csr_matrix(
            (weight_vals, (edge_rows, edge_cols)),
            shape=(num_nodes, num_nodes)
        )

        # --- Timescale Assignment (per-module, per-level) ---
        # Eliminate homogeneous parameters, progressive scaling of tau
        self.taus = np.zeros(num_nodes)
        self.taus[:n_io] = 1.0  # I/O nodes rapid transient
        
        for mod in self.modules:
            level = mod['level']
            # Item 2: Hierarchical Gradient of Intrinsic Neural Timescales (INTs)
            # τm scales from transient in level 0 to prolonged in level 3.
            # Sensory Layers (L0): 20-35ms. Associative Layers (L3): 150-200ms.
            tau_min = 25.0
            tau_max = 175.0
            base_tau = tau_min + (tau_max - tau_min) * (level / max(1, self.num_levels - 1))
            self.taus[mod['start']:mod['end']] = np.random.normal(base_tau, base_tau * 0.1, mod['size']).astype(np.float32)

        # Enforce biological bounds [20ms, 2000ms]
        self.taus = np.clip(self.taus, 20.0, 2000.0)

        # State and bias initialization
        self.states = np.zeros(num_nodes)
        self.biases = np.random.uniform(-0.01, 0.01, num_nodes)
        
        # Intense tonic inhibition for Dentate Gyrus (DG) pattern separation
        if len(self.dg_indices) > 0:
            self.biases[self.dg_indices] = -2.0

        # Store hierarchical connection metadata for predictive coding
        # Top-down weight indices: for each (upper_mod, lower_mod) pair,
        # store the edge indices in the sparse representation
        self._build_hierarchical_index(edge_rows, edge_cols)

        # (CLS sets are now initialized at the top of the function)

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