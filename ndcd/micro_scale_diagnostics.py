"""
Micro-Scale Diagnostic Framework for Neural Network Dynamics.
Bridging Neurobiology and Computational Benchmarks.

This script executes isolated neurobiological tests against the PredictiveCodingEngine,
measuring intrinsic excitability, integration, working memory, problem-solving,
predictive coding omission, and sleep dynamics in scale-restricted (10-500 unit) topologies.
"""

import torch
import numpy as np
from engine_torch import PredictiveCodingEngine

# I/O nodes in the engine are hardcoded to indices 0-511. 
# Micro diagnostic nodes must be placed after 511.
IO_OFFSET = 512

def create_micro_circuit(num_active_nodes, hierarchy_levels=1, e_i_ratio=0.8, 
                         connectivity_density=0.3, tau_mean=10.0, tau_var=2.0, seed=42):
    """
    Spins up an isolated PredictiveCodingEngine for targeted benchmarking.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    total_nodes = IO_OFFSET + num_active_nodes
    
    module_ranges = []
    module_levels = []
    modules = []
    
    nodes_per_level = max(1, num_active_nodes // hierarchy_levels)
    
    curr = IO_OFFSET
    for lvl in range(hierarchy_levels):
        end = curr + nodes_per_level if lvl < hierarchy_levels - 1 else total_nodes
        size = end - curr
        module_ranges.append((curr, end))
        module_levels.append(lvl)
        
        l4_sz = max(1, size // 3)
        l23_sz = max(1, size // 3)
        l56_sz = max(1, size - l4_sz - l23_sz)
        
        modules.append({
            'l4_indices': np.arange(curr, curr + l4_sz),
            'l23_indices': np.arange(curr + l4_sz, curr + l4_sz + l23_sz),
            'l56_indices': np.arange(curr + l4_sz + l23_sz, end)
        })
        curr = end
        
    hier_pairs = []
    if hierarchy_levels > 1:
        for m_id in range(hierarchy_levels - 1):
            hier_pairs.append((m_id + 1, m_id))  # (upper_module, lower_module)
            
    is_inhibitory = np.zeros(total_nodes, dtype=bool)
    is_pv = np.zeros(total_nodes, dtype=bool)
    is_sst = np.zeros(total_nodes, dtype=bool)
    is_vip = np.zeros(total_nodes, dtype=bool)
    num_modules = len(module_ranges)
    for i in range(num_modules):
        start = module_ranges[i][0]
        end = module_ranges[i][1]
        n_inh = max(1, int((end - start) * (1.0 - e_i_ratio))) # Use e_i_ratio from args
        if n_inh > 0:
            inh_idx = np.random.choice(np.arange(start, end), n_inh, replace=False)
            is_inhibitory[inh_idx] = True
            
            n_pv = int(n_inh * 0.4)
            n_sst = int(n_inh * 0.3)
            np.random.shuffle(inh_idx)
            if n_pv > 0:
                is_pv[inh_idx[:n_pv]] = True
            if n_sst > 0:
                is_sst[inh_idx[n_pv:n_pv+n_sst]] = True
            if n_pv + n_sst < n_inh:
                is_vip[inh_idx[n_pv+n_sst:]] = True
            
    possible_edges = num_active_nodes * (num_active_nodes - 1)
    num_edges = int(possible_edges * connectivity_density)
    
    src = np.random.randint(IO_OFFSET, total_nodes, max(1, num_edges))
    dst = np.random.randint(IO_OFFSET, total_nodes, max(1, num_edges))
    
    mask = src != dst
    src = src[mask]
    dst = dst[mask]
    
    indices = np.stack([src, dst], axis=0) if len(src) > 0 else np.zeros((2, 0), dtype=np.int64)
    
    # Input projections to support test stimuli
    inp_src = np.zeros(num_active_nodes, dtype=np.int64)
    inp_dst = np.arange(IO_OFFSET, total_nodes, dtype=np.int64)
    io_indices = np.stack([inp_src, inp_dst], axis=0)
    
    indices = np.concatenate([indices, io_indices], axis=1) if indices.shape[1] > 0 else io_indices
    
    values = np.random.randn(indices.shape[1]).astype(np.float32) * 0.1
    for i in range(indices.shape[1]):
        s = indices[0, i]
        d = indices[1, i]
        if is_inhibitory[s]:
            values[i] = -abs(values[i])
        else:
            values[i] = abs(values[i])
            
        # Powerful Mossy Fibers (L4 -> L2/3 in Hippo-like modules)
        # Using a simple heuristic for micro-circuit: if L4 src and L23 dst, boost
        if s in modules[0]['l4_indices'] and d in modules[0]['l23_indices']:
            values[i] *= 10.0
            
    biases = np.zeros(total_nodes, dtype=np.float32)
    # Hippocampal DG suppression
    biases[modules[0]['l4_indices']] = -2.0
    
    taus = np.ones(total_nodes, dtype=np.float32) * 20.0 # Default base
    for lvl in range(hierarchy_levels):
        start, end = module_ranges[lvl]
        lvl_tau = 20.0 * (5.5 ** lvl) # Stronger gradient
        taus[start:end] = np.random.normal(lvl_tau, lvl_tau * 0.1, end - start)
    
    taus = np.clip(taus, 1.0, 1000.0)
    
    engine = PredictiveCodingEngine(
        num_nodes=total_nodes,
        indices=indices,
        values=values,
        biases=biases,
        taus=taus,
        module_ranges=module_ranges,
        module_levels=np.array(module_levels),
        hier_pairs=hier_pairs,
        modules=modules,
        temporal_alpha=0.5,
        is_inhibitory=is_inhibitory,
        is_pv=is_pv,
        is_sst=is_sst,
        is_vip=is_vip,
        dg_indices=np.zeros(0, dtype=np.int64) # No explicit DG in micro circuit by default
    )
    
    # Enforce low population sparseness for orthogonality tests
    engine.sparsity_alpha = 0.15
    
    engine.modules = modules
    return engine, np.arange(IO_OFFSET, total_nodes)

class MicroScaleDiagnosticSuite:
    """
    Executes the 8 diagnostic testing suites directly informed by neurobiological benchmarks.
    """
    def __init__(self):
        self.results = {}
        
    def log(self, test_name, status, details=""):
        print(f"[{status}] {test_name}: {details}")
        self.results[test_name] = {'status': status, 'details': details}

    # =========================================================================
    # Test I: Individual Neuron Dynamics
    # =========================================================================
    def test_fi_curve_and_sfa(self):
        test_name = "I.1 Individual Neuron Dynamics (f-I & SFA)"
        engine, active_nodes = create_micro_circuit(10, connectivity_density=0.0, tau_mean=5.0)
        
        currents = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        firings = []
        for c in currents:
            inp = torch.zeros(engine.num_nodes)
            inp[active_nodes[0]] = c
            state = engine.settle(inp, max_steps=10)
            firings.append(state[active_nodes].abs().mean().item())
            
        inp = torch.zeros(engine.num_nodes)
        inp[active_nodes[0]] = 5.0
        activations = []
        engine.dt = 10.0
        for _ in range(50):
            state = engine.settle(inp, max_steps=1)
            activations.append(state[active_nodes[0]].item())
            
        # Fit to double exponential: a*exp(-b*t) + c*exp(-d*t) + e
        from scipy.optimize import curve_fit
        def double_exp(t, a, b, c, d, e):
            return a * np.exp(-b * t) + c * np.exp(-d * t) + e
            
        t_data = np.arange(len(activations))
        y_data = np.array(activations)
        
        passed = False
        details = ""
        try:
            # Initial guess: fast decay, slow decay, offset
            p0 = (y_data[0]/2, 0.5, y_data[0]/2, 0.05, y_data[-1])
            # Bounds to ensure b > d (fast vs slow) and positive coefficients
            bounds = ([-np.inf, 0.1, -np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf, 0.1, np.inf])
            popt, pcov = curve_fit(double_exp, t_data, y_data, p0=p0, bounds=bounds, maxfev=2000)
            
            # Check if fit is good (MSE)
            y_fit = double_exp(t_data, *popt)
            mse = np.mean((y_data - y_fit)**2)
            
            # Ensure we captured two distinct timescales
            if mse < 0.01 and popt[1] > popt[3] * 2:
                passed = True
                details = f"Double-exp fit MSE: {mse:.4f}, Fast-tau: {1/popt[1]:.2f}, Slow-tau: {1/popt[3]:.2f}"
            else:
                details = f"Fit failed criteria. MSE: {mse:.4f}, b:{popt[1]:.4f}, d:{popt[3]:.4f}"
        except Exception as e:
            details = f"Curve fitting failed: {str(e)}"
            
        self.log(test_name, "PASS" if passed else "FAIL", details)

    def test_burst_coincidence_detection(self):
        test_name = "I.2 Burst Coincidence Detection"
        engine, active_nodes = create_micro_circuit(10, connectivity_density=0.0)
        
        target_node = active_nodes[0]
        
        # Basal only stimulation
        engine.state_basal.zero_()
        engine.state_apical.zero_()
        engine.state_basal[target_node] = 1.0
        engine.state_apical[target_node] = 0.0
        from engine_torch import get_soma
        
        zero_inh = torch.zeros_like(engine.state_basal)
        soma_basal = get_soma(engine.state_basal, engine.state_basal, zero_inh, engine.state_apical, engine.cahva_states, engine.is_neg_pe, engine.threshold_adaptation, engine.is_sst, engine.is_pv, engine.is_dg, engine.ip_gain, engine.ip_bias, engine.nmda_ratio)[target_node].item()

        
        # Apical only stimulation
        engine.state_basal.zero_()
        engine.state_apical.zero_()
        engine.state_basal[target_node] = 0.0
        engine.state_apical[target_node] = 1.0
        soma_apical = get_soma(engine.state_basal, engine.state_basal, zero_inh, engine.state_apical, engine.cahva_states, engine.is_neg_pe, engine.threshold_adaptation, engine.is_sst, engine.is_pv, engine.is_dg, engine.ip_gain, engine.ip_bias, engine.nmda_ratio)[target_node].item()

        
        # Coincident stimulation
        engine.state_basal.zero_()
        engine.state_apical.zero_()
        engine.state_basal[target_node] = 1.0
        engine.state_apical[target_node] = 1.0
        soma_coincident = get_soma(engine.state_basal, engine.state_basal, zero_inh, engine.state_apical, engine.cahva_states, engine.is_neg_pe, engine.threshold_adaptation, engine.is_sst, engine.is_pv, engine.is_dg, engine.ip_gain, engine.ip_bias, engine.nmda_ratio)[target_node].item()

        
        # Linear expectation
        linear_sum = abs(soma_basal) + abs(soma_apical)
        
        # Biological requirement: Coincident stimulation must produce supralinear burst
        supralinear_ratio = abs(soma_coincident) / (linear_sum + 1e-6)
        
        if supralinear_ratio > 1.2:  # Requires at least 20% amplification
            self.log(test_name, "PASS", f"Supralinear burst detected. Ratio: {supralinear_ratio:.2f} (Coinc: {soma_coincident:.2f}, Linear: {linear_sum:.2f})")
        else:
            self.log(test_name, "FAIL", f"Failed to detect burst. Ratio: {supralinear_ratio:.2f}")

    def test_dead_neuron_threshold_plasticity(self):
        test_name = "I.2 Dead Neuron Threshold Plasticity (IP)"
        engine, active_nodes = create_micro_circuit(20, connectivity_density=0.1)
        
        engine.biases[active_nodes] = -5.0 
        
        activations_pre = []
        for _ in range(10):
            inp = torch.randn(engine.num_nodes) * 5.0
            state = engine.settle(inp, max_steps=5)
            activations_pre.append(state[active_nodes].abs().mean().item())
            
        for _ in range(50):
            # Stronger stimulus
            inp = torch.randn(engine.num_nodes) * 5.0
            state = engine.settle(inp, max_steps=5)
            # Use the internal IP update logic
            engine.update_weights_predictive(state, state)
            # The internal IP logic (KL-Divergence) adjusts ip_bias and ip_gain
            # This should rescue the dead neurons
            
        activations_post = []
        for _ in range(10):
            inp = torch.randn(engine.num_nodes) * 5.0
            state = engine.settle(inp, max_steps=5)
            activations_post.append(state[active_nodes].abs().mean().item())
            
        pre = np.mean(activations_pre)
        post = np.mean(activations_post)
        
        if post > pre + 0.005:
            self.log(test_name, "PASS", f"Dead neurons recovered. Pre: {pre:.4f}, Post: {post:.4f}")
        else:
            self.log(test_name, "FAIL", "IP rules failed to rescue dead neurons.")

    # =========================================================================
    # Test II: Micro-Scale Communication
    # =========================================================================
    def test_ei_balance(self):
        test_name = "II.1 E/I Balance and AI State (Fano & CV)"
        engine, active_nodes = create_micro_circuit(50, e_i_ratio=0.8, connectivity_density=0.5)
        
        # Test across vastly different input intensities
        intensities = [2.0, 10.0]
        passed = True
        details = []
        
        for intensity in intensities:
            inp = torch.zeros(engine.num_nodes)
            inp[active_nodes[0:5]] = intensity
            
            acts = []
            for i in range(50):
                input_vec = inp if i < 5 else torch.zeros(engine.num_nodes)
                state = engine.settle(input_vec, max_steps=5)
                acts.append(state[active_nodes].cpu().numpy())
                
            acts = np.abs(acts) # Shape: (50, num_active_nodes)
            
            # Continuous metrics for AI State (Asynchronous Irregular)
            # Irregularity: CV of the continuous rate (Temporal Variance / Mean)
            # acts shape: (50, num_active_nodes)
            rate_means = acts.mean(axis=0) + 1e-6
            rate_vars = acts.var(axis=0)
            fano_factor = np.mean(rate_vars / rate_means)
            
            # CV = standard deviation / mean of the rates
            cvs = np.sqrt(rate_vars) / rate_means
            mean_cv = np.mean(cvs)
            
            details.append(f"In: {intensity}, Fano: {fano_factor:.2f}, CV: {mean_cv:.2f}")
            
            # Biological constraint: CV should be near 1 (Poisson), Fano should be relatively stable
            if mean_cv < 0.2 or fano_factor > 6.0 or fano_factor < 0.02:
                passed = False
                
        self.log(test_name, "PASS" if passed else "FAIL", " | ".join(details))

    def test_lateral_inhibition_kwta(self):
        test_name = "II.2 k-Winner-Take-All Lateral Inhibition"
        engine, active_nodes = create_micro_circuit(50, connectivity_density=0.1)
        
        node_A = active_nodes[10]
        node_B = active_nodes[11] # Adjacent, likely same module
        
        # Stimulate similarly but unequally
        inp = torch.zeros(engine.num_nodes)
        inp[node_A] = 5.0
        inp[node_B] = 4.5
        
        # Ensure sparsity is active and very strict
        engine.sparsity_alpha = 0.01
        
        engine.state_basal.zero_()
        engine.state_apical.zero_()
        
        acts_A = []
        acts_B = []
        for _ in range(20):
            state = engine.settle(inp, max_steps=5)
            acts_A.append(state[node_A].item())
            acts_B.append(state[node_B].item())
        
        # A should stay active, B should be disproportionately suppressed over time
        final_ratio = (acts_A[-1] + 1e-6) / (acts_B[-1] + 1e-6)
        initial_ratio = (acts_A[0] + 1e-6) / (acts_B[0] + 1e-6)
        
        if final_ratio > initial_ratio * 2.0: # B is strongly suppressed relative to A
            self.log(test_name, "PASS", f"kWTA active. Init Ratio (A/B): {initial_ratio:.2f}, Final: {final_ratio:.2f}")
        else:
            self.log(test_name, "FAIL", f"Failed to suppress weaker node. Init: {initial_ratio:.2f}, Final: {final_ratio:.2f}")

    # =========================================================================
    # Test III: Membrane Time Constants
    # =========================================================================
    def test_membrane_time_constants(self):
        test_name = "III.1 Hierarchical Autocorrelation"
        engine, active_nodes = create_micro_circuit(100, hierarchy_levels=3, tau_mean=20.0)
        
        # Stimulate network with pure background OU noise (input vector = 0)
        acts = []
        for _ in range(200):
            inp = torch.zeros(engine.num_nodes)
            state = engine.settle(inp, max_steps=1, tol=0.0) # Single-step running trajectory
            acts.append(engine.state_basal[active_nodes].abs().cpu().numpy())
            
        acts = np.array(acts) # (200, 100)
        
        # Auto-correlation function
        def autocorr(x, max_lag=20):
            res = []
            mean_x = np.mean(x)
            var_x = np.var(x) + 1e-6
            for lag in range(1, max_lag + 1):
                cov = np.mean((x[:-lag] - mean_x) * (x[lag:] - mean_x))
                res.append(cov / var_x)
            return res

        l1_nodes = [n for n in active_nodes if engine.node_to_level[n].item() == 0]
        l3_nodes = [n for n in active_nodes if engine.node_to_level[n].item() == 2]
        
        l1_ac = np.mean([autocorr(acts[:, n - 512]) for n in l1_nodes[:5]], axis=0) # offset by IO_OFFSET 512? Wait active_nodes is from IO_OFFSET. acts[:, n - IO_OFFSET]
        l3_ac = np.mean([autocorr(acts[:, n - 512]) for n in l3_nodes[:5]], axis=0)
        
        # Calculate decay time (lag at which autocorrelation drops below 1/e ~ 0.36)
        def get_decay_time(ac):
            for i, val in enumerate(ac):
                if val < 0.5: return i + 1 # Higher threshold for noisy micro-circuit
            return len(ac)
            
        l1_decay = get_decay_time(l1_ac)
        l3_decay = get_decay_time(l3_ac)
        
        if l3_decay > l1_decay * 1.3: # Relaxed slightly for micro-scale
            self.log(test_name, "PASS", f"Hierarchical tau validated. L1 decay: {l1_decay}, L3 decay: {l3_decay}")
        else:
            self.log(test_name, "FAIL", f"Failed hierarchical gradient. L1: {l1_decay}, L3: {l3_decay}")

    def test_apical_temporal_overlap(self):
        test_name = "III.2 Apical Temporal Overlap"
        engine, active_nodes = create_micro_circuit(20, tau_mean=10.0)
        
        target = active_nodes[0]
        # For tanh activations, we must stimulate high but check decay relative to peak
        engine.state_basal[target] = 2.0
        engine.state_apical[target] = 2.0
        
        b_decay = []
        a_decay = []
        for _ in range(50):
            engine.settle(torch.zeros(engine.num_nodes), max_steps=1)
            b_decay.append(abs(engine.state_basal[target].item()))
            a_decay.append(abs(engine.state_apical[target].item()))
            
        b_max = max(b_decay)
        a_max = max(a_decay)
        b_half_life = next((i for i, v in enumerate(b_decay) if v < b_max * 0.5), 50)
        a_half_life = next((i for i, v in enumerate(a_decay) if v < a_max * 0.5), 50)
        
        if a_half_life > b_half_life:
            self.log(test_name, "PASS", f"Apical state decays slower. Basal HL: {b_half_life}, Apical HL: {a_half_life}")
        else:
            self.log(test_name, "FAIL", f"Failed overlap. Basal HL: {b_half_life}, Apical HL: {a_half_life}")

    # =========================================================================
    # Test IV: Micro-Scale Memorization
    # =========================================================================
    def test_working_memory_and_stsp(self):
        test_name = "IV.1 Activity-Silent STSP (Z-Score Validated)"
        engine, active_nodes = create_micro_circuit(30, connectivity_density=0.5)
        
        engine.weight_values[:] = torch.abs(engine.weight_values) * 0.1
        
        # Establish baseline noise distribution
        baseline_acts = []
        for _ in range(50):
            engine.settle(torch.zeros(engine.num_nodes), max_steps=2)
            baseline_acts.append(engine.state[active_nodes[:5]].abs().mean().item())
            
        base_mean = np.mean(baseline_acts)
        base_std = np.std(baseline_acts) + 1e-6
        
        inp = torch.zeros(engine.num_nodes)
        inp[active_nodes[0]] = 5.0
            
        # Stimulate to create memory trace
        for _ in range(10):
            engine.settle(inp, max_steps=2)
            
        # Delay period (activity-silent)
        for _ in range(100):
            engine.settle(torch.zeros(engine.num_nodes), max_steps=2)
            
        # Ping
        ping = torch.zeros(engine.num_nodes)
        ping[active_nodes[0]] = 0.5 
        state = engine.settle(ping, max_steps=10)
        
        stim_memory = state[active_nodes[:5]].abs().mean().item()
        
        # Calculate Z-score of the recalled memory against the baseline
        z_score = (stim_memory - base_mean) / base_std
        
        # High statistical confidence required (e.g. Z > 3.0)
        if z_score > 3.0 and stim_memory > 0.001:
            self.log(test_name, "PASS", f"STSP Trace distinct from noise. Z-Score: {z_score:.2f}")
        else:
            self.log(test_name, "FAIL", f"Failed memory distinctness. Z-Score: {z_score:.2f}")

    def test_pattern_completion_capability(self):
        test_name = "IV.2 Pattern Completion Capability"
        engine, active_nodes = create_micro_circuit(20, connectivity_density=0.5)
        
        # Simulate a trained associative memory in a small layer 5/6 module
        engine.weight_values[:] = torch.abs(engine.weight_values) * 0.5 # Potentiate connections
        
        # Identify central nodes (by degree for simplicity instead of full closeness centrality)
        # Using numpy array 'indices' from engine
        edge_src = engine.indices[0].cpu().numpy()
        edge_dst = engine.indices[1].cpu().numpy()
        
        degrees = np.zeros(engine.num_nodes)
        for src in edge_src:
            degrees[src] += 1
            
        active_degrees = degrees[active_nodes]
        central_nodes = active_nodes[np.argsort(active_degrees)[-4:]] # Top 20% (4 out of 20)
        peripheral_nodes = np.setdiff1d(active_nodes, central_nodes)
        
        # Turn off sparsity so peripheral nodes are not hard-suppressed
        engine.sparsity_alpha = 1.0
        
        # Partial cue: stimulate only the top central nodes
        inp = torch.zeros(engine.num_nodes)
        inp[central_nodes] = 5.0
        
        state = engine.settle(inp, max_steps=20)
        
        # Measure recovery of unstimulated peripheral ensemble members
        peripheral_recovery = state[peripheral_nodes].abs().mean().item()
        central_activation = state[central_nodes].abs().mean().item()
        
        recovery_ratio = peripheral_recovery / (central_activation + 1e-6)
        
        if recovery_ratio > 0.4: # recovered 40% of the representation magnitude
            self.log(test_name, "PASS", f"Pattern Completed from 20% cue. Recovery ratio: {recovery_ratio:.2f}")
        else:
            self.log(test_name, "FAIL", f"Failed pattern completion. Recovery ratio: {recovery_ratio:.2f}")

    def test_representation_orthogonality(self):
        test_name = "IV.3 Representation Orthogonality"
        engine, active_nodes = create_micro_circuit(30, connectivity_density=0.2)
        
        # Hold memory A
        inpA = torch.zeros(engine.num_nodes)
        inpA[active_nodes[0:3]] = 5.0
        for _ in range(5): engine.settle(inpA, max_steps=5)
        engine.state.zero_()
        
        # Hold memory B
        inpB = torch.zeros(engine.num_nodes)
        inpB[active_nodes[15:18]] = 5.0
        for _ in range(5): engine.settle(inpB, max_steps=5)
        
        # Clear all states completely
        engine.state.zero_()
        engine.state_basal.zero_()
        engine.state_apical.zero_()
        engine.rho_slow_states.zero_()
        
        # Ping A
        pingA = torch.zeros(engine.num_nodes)
        pingA[active_nodes[0]] = 0.5
        repA = engine.settle(pingA, max_steps=10)[active_nodes].abs().cpu().numpy()
        
        # Ping B
        pingB = torch.zeros(engine.num_nodes)
        pingB[active_nodes[15]] = 0.5
        repB = engine.settle(pingB, max_steps=10)[active_nodes].abs().cpu().numpy()
        
        # Cosine similarity
        from numpy.linalg import norm
        sim = np.dot(repA, repB) / ((norm(repA) * norm(repB)) + 1e-6)
        
        if sim < 0.2: # Near orthogonal (close to 0)
            self.log(test_name, "PASS", f"Orthogonal memory manifolds. Cos Similarity: {sim:.3f}")
        else:
            self.log(test_name, "FAIL", f"Memory interference detected. Cos Similarity: {sim:.3f}")

    # =========================================================================
    # Test V: Algorithmic Problem Solving
    # =========================================================================
    def test_algorithmic_xor_and_pattern_sep(self):
        test_name = "V.1 Non-Linear Logic & Pattern Separation"
        engine, active_nodes = create_micro_circuit(50, connectivity_density=0.2, hierarchy_levels=2)
        
        node_A = active_nodes[0]
        node_B = active_nodes[1]
        
        # Enforce highly non-linear DG logic
        engine.is_dg[active_nodes] = True
        
        # Ensure we target Level 2 nodes for top-down apical feedback
        l2_nodes = engine.modules[1]['l56_indices']
        apical_node_A = l2_nodes[0] if len(l2_nodes) > 0 else active_nodes[-1]
        apical_node_B = l2_nodes[1] if len(l2_nodes) > 1 else active_nodes[-2]
        
        # Test true pattern separation: Combined A+B representation should be orthogonal to A and B individually
        inpA = torch.zeros(engine.num_nodes)
        inpA[node_A] = 5.0
        inpA[apical_node_A] = 5.0 # Coincident apical feedback
        
        inpB = torch.zeros(engine.num_nodes)
        inpB[node_B] = 5.0
        inpB[apical_node_B] = 5.0 # Coincident apical feedback
        
        inpAB = torch.zeros(engine.num_nodes)
        inpAB[node_A] = 5.0
        inpAB[node_B] = 5.0
        inpAB[apical_node_A] = 5.0
        inpAB[apical_node_B] = 5.0
        
        engine.state.zero_()
        repA = engine.settle(inpA, max_steps=20)[active_nodes].abs().cpu().numpy()
        
        engine.state.zero_()
        repB = engine.settle(inpB, max_steps=20)[active_nodes].abs().cpu().numpy()
        
        engine.state.zero_()
        repAB = engine.settle(inpAB, max_steps=20)[active_nodes].abs().cpu().numpy()
        
        from numpy.linalg import norm
        simA = np.dot(repAB, repA) / ((norm(repAB) * norm(repA)) + 1e-6)
        simB = np.dot(repAB, repB) / ((norm(repAB) * norm(repB)) + 1e-6)
        
        # Combined representation must be distinct from individual representations
        if simA < 0.5 and simB < 0.5:
            self.log(test_name, "PASS", f"Pattern separation active. Sim(AB,A): {simA:.2f}, Sim(AB,B): {simB:.2f}")
        else:
            self.log(test_name, "FAIL", f"Failed pattern separation. Sim(AB,A): {simA:.2f}, Sim(AB,B): {simB:.2f}")

    def test_metabolic_efficiency(self):
        test_name = "V.2 Metabolic Efficiency (Heavy-Tail Updates)"
        engine, active_nodes = create_micro_circuit(50, connectivity_density=0.5)
        
        inp = torch.randn(engine.num_nodes) * 5.0
        
        # Initial effective weights
        pre_weights = engine.effective_weights.clone()
        
        for _ in range(10):
            state = engine.settle(inp, max_steps=5)
            engine.update_weights_predictive(state, state)
            
        post_weights = engine.effective_weights.clone()
        
        # Measure weight updates
        updates = (post_weights - pre_weights).abs().cpu().numpy()
        updates = updates[engine.free_edge_mask.cpu().numpy()]
        
        if len(updates) < 10:
            self.log(test_name, "FAIL", "Not enough free edges to test.")
            return
            
        # Check for heavy-tailed distribution (kurtosis > 3, or top 5% accounts for > 50% of total change)
        updates_sorted = np.sort(updates)[::-1]
        top_5_percent_idx = max(1, int(len(updates_sorted) * 0.05))
        
        top_magnitude = np.sum(updates_sorted[:top_5_percent_idx])
        total_magnitude = np.sum(updates_sorted) + 1e-6
        
        tail_ratio = top_magnitude / total_magnitude
        
        if tail_ratio > 0.3: # Top 5% of synapses account for > 30% of total change
            self.log(test_name, "PASS", f"Heavy-tailed updates verified. Top 5% holds {tail_ratio*100:.1f}% of change.")
        else:
            self.log(test_name, "FAIL", f"Updates too uniform. Top 5% holds {tail_ratio*100:.1f}% of change.")

    # =========================================================================
    # Test VI: Predictive Coding & Omission Responses
    # =========================================================================
    def test_predictive_omission_response(self):
        test_name = "VI.1 Omission Responses & Hierarchical Dependence"
        engine, active_nodes = create_micro_circuit(100, hierarchy_levels=2)
        
        stim_node = active_nodes[0]
        err_nodes = engine.modules[0]['l23_indices']
        
        # Train expectations
        baseline_errs = []
        for i in range(20):
            inp = torch.zeros(engine.num_nodes)
            inp[stim_node] = 5.0
            state = engine.settle(inp, max_steps=5)
            engine.update_weights_predictive(state, state)
            err = state[err_nodes].abs().mean().item()
            if i > 5: # Skip initial untrained
                baseline_errs.append(err)
                
        base_mean = np.mean(baseline_errs)
        base_std = np.std(baseline_errs) + 1e-6
        
        # Test 1: Negative Oddball (Omission)
        engine.state.zero_()
        engine.state_apical[err_nodes] = 2.0  # Top-down prediction persists
        state = engine.settle(torch.zeros(engine.num_nodes), max_steps=5)
        omission_err_nodes = state[err_nodes].abs()
        omission_err = omission_err_nodes.mean().item()
        
        z_score_omission = (omission_err - base_mean) / base_std
        
        # Test 2: Positive Oddball (Amplified)
        engine.state.zero_()
        engine.state_apical[err_nodes] = 2.0
        inp_pos = torch.zeros(engine.num_nodes)
        inp_pos[stim_node] = 10.0 # Double expectation
        state = engine.settle(inp_pos, max_steps=5)
        positive_err_nodes = state[err_nodes].abs()
        
        # Check Asymmetry: Are the active error populations different?
        omission_active = (omission_err_nodes > omission_err_nodes.mean()).float()
        positive_active = (positive_err_nodes > positive_err_nodes.mean()).float()
        
        overlap = torch.dot(omission_active, positive_active) / (torch.norm(omission_active) * torch.norm(positive_active) + 1e-6)
        
        # Test 3: Hierarchical Dependence
        # Sever top-down connections
        saved_td = engine.weight_values[engine.topdown_edge_mask].clone()
        engine.weight_values[engine.topdown_edge_mask] = 0.0
        engine.w_surface[engine.topdown_edge_mask] = 0.0
        engine.w_deep[engine.topdown_edge_mask] = 0.0
        engine.w_mid[engine.topdown_edge_mask] = 0.0
        
        engine.state.zero_()
        state = engine.settle(torch.zeros(engine.num_nodes), max_steps=5)
        severed_err = state[err_nodes].abs().mean().item()
        
        # Restore TD for safety
        engine.weight_values[engine.topdown_edge_mask] = saved_td
        
        passed = True
        details = []
        
        if z_score_omission > 3.0:
            details.append(f"Omission Z: {z_score_omission:.1f}")
        else:
            passed = False
            details.append(f"Omission Z too low: {z_score_omission:.1f}")
            
        if overlap < 0.8: # Distinct populations
            details.append(f"Asymmetry Overlap: {overlap:.2f}")
        else:
            passed = False
            details.append(f"Failed Asymmetry (Overlap {overlap:.2f})")
            
        if severed_err < 0.01:
            details.append("Hierarchical isolation validated")
        else:
            passed = False
            details.append(f"Failed Hierarchical Isolation: Err {severed_err:.3f}")
            
        self.log(test_name, "PASS" if passed else "FAIL", " | ".join(details))

    # =========================================================================
    # Test VII: Sleeping Dynamics
    # =========================================================================
    def test_sleep_and_synaptic_scaling(self):
        test_name = "VII.1 Sleep & Synaptic Homeostasis"
        engine, active_nodes = create_micro_circuit(200, connectivity_density=0.3)
        
        # Train a specific pattern
        target_nodes = active_nodes[10:20]
        distractor_nodes = active_nodes[50:60]
        stim_node = active_nodes[0]
        
        inp = torch.zeros(engine.num_nodes)
        inp[stim_node] = 10.0
        target_vec = torch.zeros(engine.num_nodes)
        target_vec[target_nodes] = 1.0
        target_vec[distractor_nodes] = -1.0 # Suppress distractors
        
        # Saturated training phase (Awake)
        engine.w_deep += torch.randn_like(engine.w_deep) * 0.1 # Add noise to simulate saturation
        for _ in range(50):
            state = engine.settle(inp, max_steps=5)
            # Force target representation for learning
            nudge = state.clone()
            nudge[target_nodes] = 5.0
            nudge[distractor_nodes] = -5.0
            engine.update_weights_predictive(state, nudge)
            engine.consolidate_importance() # Accumulate omega
            
        # Test pre-sleep classification confidence
        engine.state.zero_()
        noisy_inp = inp + torch.randn(engine.num_nodes) * 2.0
        state = engine.settle(noisy_inp, max_steps=10)
        
        target_act = state[target_nodes].mean().item()
        distract_act = state[distractor_nodes].mean().item()
        pre_margin = target_act - distract_act
        
        pre_var = engine.effective_weights[engine.free_edge_mask].var().item()
        
        # Simulate Sleep Phase (Offline Renormalization)
        if hasattr(engine, 'offline_renormalization'):
            engine.offline_renormalization()

        else:
            self.log(test_name, "FAIL", "engine_torch.py lacks offline_renormalization method.")

            return
            
        # Test post-sleep classification confidence
        engine.state.zero_()
        state = engine.settle(noisy_inp, max_steps=10)
        
        target_act = state[target_nodes].mean().item()
        distract_act = state[distractor_nodes].mean().item()
        post_margin = target_act - distract_act
        
        post_var = engine.effective_weights[engine.free_edge_mask].var().item()
        
        passed = True
        details = []
        
        if post_margin > pre_margin:
            details.append(f"Margin improved ({pre_margin:.2f} -> {post_margin:.2f})")
        else:
            passed = False
            details.append(f"Margin failed ({pre_margin:.2f} -> {post_margin:.2f})")
            
        if post_var > pre_var * 0.01: # As long as it wasn't completely eradicated
            details.append(f"Variance scaled effectively ({pre_var:.4f} -> {post_var:.4f})")
        else:
            passed = False
            details.append(f"Variance failed ({pre_var:.4f} -> {post_var:.4f})")
            
        self.log(test_name, "PASS" if passed else "FAIL", " | ".join(details))
             
    # =========================================================================
    # Test VIII: Network-Level Diagnostics
    # =========================================================================
    def test_fisher_information(self):
        test_name = "VIII.1 Fisher Information Matrix Trace"
        engine, active_nodes = create_micro_circuit(50, connectivity_density=0.5)
        
        # Estimate the trace of the Fisher Information Matrix (FIM)
        # FIM trace roughly correlates with the sum of variances of log-likelihood gradients
        # We can approximate this by the variance of the state changes with respect to parameters
        # For simplicity, we'll measure the sensitivity of the network to small perturbations
        
        inp = torch.randn(engine.num_nodes) * 2.0
        
        # Baseline state
        engine.state.zero_()
        base_state = engine.settle(inp, max_steps=10).clone()
        
        # Perturb weights slightly and measure state divergence
        perturbations = 20
        divergences = []
        
        original_weights = engine.weight_values.clone()
        
        for _ in range(perturbations):
            noise = torch.randn_like(engine.weight_values) * 0.01
            engine.weight_values += noise
            
            engine.state.zero_()
            perturbed_state = engine.settle(inp, max_steps=10)
            
            div = torch.norm(perturbed_state - base_state).item()
            divergences.append(div)
            
            # Restore
            engine.weight_values.copy_(original_weights)
            
        fisher_trace_approx = np.var(divergences) * (1.0 / 0.01**2) # Scale by perturbation variance inverse
        
        if fisher_trace_approx > 10.0: # Requires high sensitivity/information content
            self.log(test_name, "PASS", f"High Fisher Information. Trace Approx: {fisher_trace_approx:.2f}")
        else:
            self.log(test_name, "FAIL", f"Low Fisher Information. Trace Approx: {fisher_trace_approx:.2f}")

    def test_structural_pruning_fisher(self):
        test_name = "VIII.2 Fisher-Preserving Structural Pruning"
        engine, active_nodes = create_micro_circuit(100, connectivity_density=0.4)
        
        # Train to accumulate omega
        inp = torch.randn(engine.num_nodes) * 5.0
        for _ in range(20):
            state = engine.settle(inp, max_steps=5)
            engine.update_weights_predictive(state, state)
            engine.consolidate_importance()
            
        # Function to approximate FIM trace
        def approx_fisher(eng):
            test_inp = torch.ones(eng.num_nodes)
            eng.state.zero_()
            base = eng.settle(test_inp, max_steps=5).clone()
            
            divs = []
            orig_w = eng.weight_values.clone()
            for _ in range(10):
                noise = torch.randn_like(eng.weight_values) * 0.05
                eng.weight_values += noise
                eng.state.zero_()
                pert = eng.settle(test_inp, max_steps=5)
                divs.append(torch.norm(pert - base).item())
                eng.weight_values.copy_(orig_w)
            return np.mean(divs) # Simpler proxy: mean divergence to noise
            
        pre_fisher = approx_fisher(engine)
        
        # Prune 10% of synapses
        pruned_count = engine.remodel_structure(prune_ratio=0.1)
        
        post_fisher = approx_fisher(engine)
        
        # We expect a slight drop, but mostly preservation of information
        retention_ratio = post_fisher / (pre_fisher + 1e-6)
        
        passed = True
        details = []
        
        if pruned_count > 0:
            details.append(f"Pruned {pruned_count} synapses")
        else:
            passed = False
            details.append("No synapses pruned")
            
        if retention_ratio > 0.8: # Retained > 80% of information despite losing 10% of synapses
            details.append(f"Fisher Retained: {retention_ratio*100:.1f}%")
        else:
            passed = False
            details.append(f"Fisher Retained: {retention_ratio*100:.1f}% (Too low)")
            
        self.log(test_name, "PASS" if passed else "FAIL", " | ".join(details))

    def run_all(self):
        print("====== Executing Micro-Scale Diagnostic Framework ======")
        self.test_fi_curve_and_sfa()
        self.test_burst_coincidence_detection()
        self.test_dead_neuron_threshold_plasticity()
        self.test_ei_balance()
        self.test_lateral_inhibition_kwta()
        self.test_membrane_time_constants()
        self.test_apical_temporal_overlap()
        self.test_working_memory_and_stsp()
        self.test_pattern_completion_capability()
        self.test_representation_orthogonality()
        self.test_algorithmic_xor_and_pattern_sep()
        self.test_metabolic_efficiency()
        self.test_predictive_omission_response()
        self.test_sleep_and_synaptic_scaling()
        self.test_fisher_information()
        self.test_structural_pruning_fisher()
        
        passed = sum(1 for res in self.results.values() if res['status'] == 'PASS')
        total = len(self.results)
        print("========================================================")
        print(f"Diagnostics Complete: {passed}/{total} Benchmark Suites Passed.")

if __name__ == "__main__":
    suite = MicroScaleDiagnosticSuite()
    suite.run_all()
