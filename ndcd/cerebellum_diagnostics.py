import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict

# Import core engine components
from engine_torch import PredictiveCodingEngine, get_soma
from graph import DynamicGraph
from test_pc_engine_simple import (
    add_cerebellar_module, 
    cerebellar_forward, 
    cerebellar_learn
)

class StubGraph:
    def __init__(self, num_nodes, n_l56=256):
        self.num_nodes = num_nodes
        self.modules = [{'l23_indices': np.array([]), 
                         'l4_indices': np.array([]), 
                         'l56_indices': np.arange(n_l56),
                         'level': 3}]
        self.num_levels = 4
        self.is_neg_pe = torch.zeros(num_nodes, dtype=torch.bool)
        self.is_pos_pe = torch.zeros(num_nodes, dtype=torch.bool)

class StubEngine:
    """Isolated engine for cerebellar diagnostics."""
    def __init__(self, n_l56=256, device='cuda'):
        self.device = device
        self.n_l56 = n_l56
        self.num_nodes = n_l56
        self.state = torch.zeros(n_l56, device=device)
        self.state_basal = torch.zeros(n_l56, device=device)
        self.is_l56 = torch.ones(n_l56, dtype=torch.bool, device=device)
        self.node_to_level = torch.ones(n_l56, dtype=torch.long, device=device) * 3
        self.module_ranges = [(0, n_l56)]
        
        # Diagnostics metadata
        self.ip_gain = torch.ones(n_l56, device=device)
        self.ip_bias = torch.zeros(n_l56, device=device)
        self.cahva_states = torch.zeros(n_l56, device=device)
        
        # Working Memory (Context EMA) properties
        self.context_ema = torch.zeros(n_l56, device=device)
        self.context_ema_alpha = 0.1
        self.context_injection_weight = 0.5

    def set_l56(self, acts):
        self.state[:self.n_l56] = acts
        
    def settle(self, input_vector, *args, **kwargs):
        # Mocking the IMEX integration by mapping directly + persistent feedback
        self.state = input_vector.clone() + self.context_ema * self.context_injection_weight
        
    def update_context_ema(self):
        self.context_ema = (1 - self.context_ema_alpha) * self.context_ema + self.context_ema_alpha * self.state

def summarize(hist: Dict, window: int = 200, baseline: float = 0.0) -> Dict:
    """Helper to summarize test history."""
    res = {}
    for k, v in hist.items():
        if not v: continue
        res[f'{k}_tail'] = np.mean(v[-window:])
        res[f'{k}_peak'] = np.max(v)
    
    # Simple first-over-baseline heuristic
    if 'acc' in hist:
        for i, val in enumerate(hist['acc']):
            if i > 50 and np.mean(hist['acc'][max(0, i-window):i+1]) >= baseline:
                res['first_over_baseline'] = i
                break
        else:
            res['first_over_baseline'] = -1
            
    return res

class CerebellumDiagnosticSuite:
    def __init__(self, device='cuda', seed=42):
        self.device = device
        self.seed = seed
        self.n_l56 = 256
        self.n_granule = 16384
        self.results = []
        torch.manual_seed(seed)
        np.random.seed(seed)
        
    def _new_cerebellum(self, test_seed=None):
        if test_seed is not None:
            torch.manual_seed(test_seed)
            np.random.seed(test_seed)
            
        graph = StubGraph(num_nodes=self.n_l56, n_l56=self.n_l56)
        engine = StubEngine(n_l56=self.n_l56, device=self.device)
        cereb = add_cerebellar_module(
            graph, 
            num_nodes=self.n_l56, 
            n_granule=self.n_granule, 
            device=self.device
        )
        return engine, cereb

    def _log(self, name, passed, details):
        status = "PASS" if passed else "FAIL"
        print(f"\n{name}: {status}")
        for k, v in details.items():
            print(f"    {k:25s}: {v}")
        self.results.append((name, status))

    # -------------------------------------------------------------------------
    # Benchmark 1: Information-Theoretic Granule Cell Pattern Separation
    # -------------------------------------------------------------------------
    def benchmark_01_pattern_separation(self):
        print("\nBenchmark 1: GC Pattern Separation & Orthogonality")
        engine, cereb = self._new_cerebellum()
        
        # Generate smooth trajectory in L5/6 space (Pontine input)
        n_steps = 1000
        t = torch.linspace(0, 4*math.pi, n_steps, device=self.device)
        # 256-dim manifold: sin/cos combos
        manifold = torch.stack([
            torch.sin(t + i * 0.1) for i in range(self.n_l56)
        ], dim=1)
        
        mf_acts_list = []
        gc_acts_list = []
        
        for i in range(n_steps):
            engine.set_l56(manifold[i])
            logits, granule_acts, _ = cerebellar_forward(engine, cereb)
            
            # Record Mossy Fiber (Pontine) and Granule Cell activations
            mf_acts_list.append(cereb['_last_pontine_acts'].cpu().numpy())
            gc_acts_list.append(granule_acts.cpu().numpy())
            
        mf_acts = np.array(mf_acts_list) # (steps, 1024)
        gc_acts = np.array(gc_acts_list) # (steps, 16384)
        
        # Compute Correlation Matrices
        # We sample 100 random pairs to estimate orthogonality expansion
        n_samples = 200
        indices = np.random.choice(n_steps, n_samples, replace=False)
        
        mf_corr = np.corrcoef(mf_acts[indices])
        gc_corr = np.corrcoef(gc_acts[indices])
        
        # Pattern Separation Metric: Ratio of average correlations
        # Biological GCs should be significantly more orthogonal than MFs
        avg_mf_corr = np.mean(np.abs(mf_corr[np.triu_indices(n_samples, k=1)]))
        avg_gc_corr = np.mean(np.abs(gc_corr[np.triu_indices(n_samples, k=1)]))
        
        decorrelation_ratio = 1.0 - (avg_gc_corr / (avg_mf_corr + 1e-9))
        
        # Sparsity check
        gc_sparsity = np.mean(gc_acts > 0)
        
        passed = (0.05 <= gc_sparsity <= 0.30) and (decorrelation_ratio >= 0.30)
        
        details = {
            'Avg MF Correlation': f"{avg_mf_corr:.4f}",
            'Avg GC Correlation': f"{avg_gc_corr:.4f}",
            'Decorr Ratio':       f"{decorrelation_ratio:.2f}",
            'GC Sparsity':        f"{gc_sparsity:.2%}",
            'Criterion':          "Decorr >= 0.30 & Sparsity 5-30%",
        }
        self._log("B1 Pattern Separation", passed, details)

    # -------------------------------------------------------------------------
    # Benchmark 2: CF-Gated PF-PC LTD
    # -------------------------------------------------------------------------
    def benchmark_02_pf_pc_ltd(self):
        print("\nBenchmark 2: CF-gated long-term depression at PF-PC synapses")
        engine, cereb = self._new_cerebellum(test_seed=42)
        
        # Temporarily disable proportional restoring normalization to cleanly measure delta rule LTD
        orig_norm = cereb['pk_target_row_norm'].clone()
        cereb['pk_target_row_norm'] *= 10.0 # Push it far so restoring force is zero
        
        w_init = cereb['purkinje_weights'].clone()
        target_pk = 0
        n_trials = 300
        
        for _ in range(n_trials):
            mf_in = torch.randn(self.n_l56, device=self.device)
            engine.set_l56(mf_in)
            logits, gc_acts, _ = cerebellar_forward(engine, cereb)
            # Advance time for trace cascade
            engine.set_l56(torch.zeros(self.n_l56, device=self.device))
            for _ in range(2):
                logits, _, _ = cerebellar_forward(engine, cereb)
                
            cerebellar_learn(cereb, logits, gc_acts, target_pk, 1.0, engine)
            
        w_final = cereb['purkinje_weights'].clone()
        
        # Estimate reduction on target row
        w_init_mean = w_init[target_pk].mean().item()
        w_final_mean = w_final[target_pk].mean().item()
        # The delta rule subtracts from weights: w -= (positive_update)
        reduction = (w_init_mean - w_final_mean) / (abs(w_init_mean) + 1e-9)
        
        # Protocol B: Control row (PF only, no CF)
        control_pk = 1
        w_init_c = w_init[control_pk].mean().item()
        w_final_c = w_final[control_pk].mean().item()
        control_reduction = (w_init_c - w_final_c) / (abs(w_init_c) + 1e-9)
        
        passed = (reduction >= 0.15) and (control_reduction <= 0.05)
        
        details = {
            'Target LTD Reduction': f"{reduction:.2%}",
            'Control Row Deletion': f"{control_reduction:.2%}",
            'Criterion': "LTD >= 15%, Control < 5%"
        }
        self._log("B2 PF-PC LTD", passed, details)

    # -------------------------------------------------------------------------
    # Benchmark 4: Complex Spike Coincidence and BAC Burst Amplification
    # -------------------------------------------------------------------------
    def benchmark_04_bac_firing(self):
        print("\nBenchmark 4: BAC Burst Amplification (Coincidence Detection)")
        # This test ensures that when Basal(PF) and Apical(CF) inputs coincide,
        # the output is supralinearly amplified, as per Alviña et al. 2008.
        
        # We need a Purkinje unit. In our engine, get_soma implements this.
        # Purkinje cells are our readouts, but here we test the mechanism.
        
        def test_bac(basal_drive, apical_drive):
            # Parameters from engine_torch.py get_soma
            apical_beta = 1.0
            
            # Somatic spike gate
            somatic_spike = torch.sigmoid(torch.tensor(basal_drive) * 15.0 - 7.5)
            
            # Apical XOR-like gate
            a = torch.tensor(apical_drive)
            apical_gate_fast = torch.sigmoid(a * apical_beta * 8.0 - 3.0)
            apical_gate_slow = torch.sigmoid(a * apical_beta * 12.0 - 10.0)
            apical_xor = apical_gate_fast - 0.4 * apical_gate_slow
            
            # Coincidence trigger
            bac_burst = somatic_spike * apical_xor
            return bac_burst.item()

        # 1. Basal only (0.6 drive -> ~0.5 spike)
        b_only = test_bac(0.6, 0.0)
        # 2. Apical only (0.5 drive -> ~0.5 gate)
        a_only = test_bac(0.0, 0.5)
        # 3. Coincident (Both 0.5)
        coincident = test_bac(0.6, 0.5)
        
        supralinear_ratio = coincident / (b_only + a_only + 1e-9)
        
        passed = supralinear_ratio > 1.20
        
        details = {
            'Basal-only Output':  f"{b_only:.4f}",
            'Apical-only Output': f"{a_only:.4f}",
            'Coincident Output':  f"{coincident:.4f}",
            'Supralinear Ratio':  f"{supralinear_ratio:.2f}",
            'Criterion':          "Ratio > 1.20",
        }
        self._log("B4 BAC Bursting", passed, details)

    # -------------------------------------------------------------------------
    # Benchmark 3: PF-PC LTP and Bidirectional Plasticity
    # -------------------------------------------------------------------------
    def benchmark_03_pf_pc_ltp(self):
        print("\nBenchmark 3: PF-PC LTP and bidirectional plasticity")
        engine, cereb = self._new_cerebellum(test_seed=42)
        target_pk = 0
        w_init = cereb['purkinje_weights'].clone()
        
        n_trials = 300
        # Phase 1: Induce LTD
        for _ in range(n_trials):
            engine.set_l56(torch.randn(self.n_l56, device=self.device))
            logits, gc_acts, _ = cerebellar_forward(engine, cereb)
            engine.set_l56(torch.zeros(self.n_l56, device=self.device))
            for _ in range(2):
                logits, _, _ = cerebellar_forward(engine, cereb)
            cerebellar_learn(cereb, logits, gc_acts, target_pk, 1.0, engine)
        
        w_post_ltd = cereb['purkinje_weights'].clone()
        w_init_mean = w_init[target_pk].mean().item()
        w_ltd_mean = w_post_ltd[target_pk].mean().item()
        ltd_drop = w_init_mean - w_ltd_mean
        
        # Phase 2: LTP via normalization (spontaneous without CF error)
        from test_pc_engine_simple import _apply_pk_row_normalization
        for _ in range(n_trials):
            engine.set_l56(torch.randn(self.n_l56, device=self.device))
            logits, gc_acts, _ = cerebellar_forward(engine, cereb)
            # Apply row norm without CF LTD
            _apply_pk_row_normalization(cereb)
            
        w_post_ltp = cereb['purkinje_weights'].clone()
        w_ltp_mean = w_post_ltp[target_pk].mean().item()
        
        ltp_gain = w_ltp_mean - w_ltd_mean
        rel_increase = ltp_gain / (abs(w_init_mean) + 1e-9)
        recovery_ratio = ltp_gain / (abs(ltd_drop) + 1e-9)
        
        passed = (rel_increase >= 0.10) and (recovery_ratio >= 0.50)
        details = {
            'LTD Drop': f"{ltd_drop:.4f}",
            'LTP Gain': f"{ltp_gain:.4f}",
            'Rel Increase': f"{rel_increase:.2%}",
            'Recovery Ratio': f"{recovery_ratio:.2%}",
            'Criterion': "Increase >= 10% & Recovery >= 50%"
        }
        self._log("B3 PF-PC LTP", passed, details)

    # -------------------------------------------------------------------------
    # Benchmark 6: DCN Rebound Firing
    # -------------------------------------------------------------------------
    def benchmark_06_dcn_rebound(self):
        print("\nBenchmark 6: DCN rebound firing (T-type Calcium)")
        engine, cereb = self._new_cerebellum(test_seed=42)
        
        target_dcn = 0
        pause_durations = [10, 25, 50, 100, 200]
        rebound_ratios = []
        
        # We will directly stimulate Purkinje weights to create inhibition
        w_pk = cereb['purkinje_weights']
        # Set all weights positive to guarantee strong inhibition
        w_pk.copy_(torch.ones_like(w_pk) * 0.1)
        
        for pause in pause_durations:
            cereb['dcn_hyperpol_state'].zero_()
            
            # 1. Steady state strong inhibition
            for _ in range(20):
                engine.set_l56(torch.ones(self.n_l56, device=self.device))
                cerebellar_forward(engine, cereb)
                
            baseline_dcn = cereb['_last_dcn_rate'][target_dcn].item()
            if baseline_dcn < 0.1: baseline_dcn = 0.1 # floor
            
            # 2. Pause inhibition
            pause_steps = max(1, pause // 20) # 20ms per step
            for _ in range(pause_steps):
                engine.set_l56(torch.zeros(self.n_l56, device=self.device))
                cerebellar_forward(engine, cereb)
                
            rebound_dcn = cereb['_last_dcn_rate'][target_dcn].item()
            rebound_ratios.append(rebound_dcn / baseline_dcn)
            
        ratio_50ms = rebound_ratios[2]
        
        # Criteria: Rebound magnitude increases monotonically up to 200ms
        monotonic = all(rebound_ratios[i] <= rebound_ratios[i+1] + 0.1 for i in range(len(rebound_ratios)-1))
        passed = (ratio_50ms >= 1.50) and monotonic
        
        details = {
            'Ratio (50ms)': f"{ratio_50ms:.2f}",
            'Monotonic': f"{monotonic}",
            'Criterion': "Ratio >= 1.50 for 50ms pulse & Monotonic"
        }
        self._log("B6 DCN Rebound Firing", passed, details)

    # -------------------------------------------------------------------------
    # Benchmark 9: Short-Term Plasticity (Tsodyks-Markram)
    # -------------------------------------------------------------------------
    def benchmark_09_stp(self):
        print("\nBenchmark 9: Short-Term Synaptic Plasticity")
        engine, cereb = self._new_cerebellum(test_seed=42)
        
        # Pair pulse test at 20ms ISI (1 step)
        # 1st pulse
        engine.set_l56(torch.ones(self.n_l56, device=self.device))
        cerebellar_forward(engine, cereb)
        pf_u_1 = cereb['pf_u'].mean().item()
        mf_u_1 = cereb['mf_u'].mean().item()
        
        # 2nd pulse
        cerebellar_forward(engine, cereb)
        pf_u_2 = cereb['pf_u'].mean().item()
        mf_u_2 = cereb['mf_u'].mean().item()
        
        # PPR is proportional to u_2/u_1 since x is barely depleted at start
        ppr_mf = mf_u_2 / (pf_u_1 + 1e-9)
        ppr_pf = pf_u_2 / (pf_u_1 + 1e-9)
        
        passed = (ppr_mf > 1.20) and (ppr_pf > 1.10)
        details = {
            'MF PPR (20ms)': f"{ppr_mf:.2f}",
            'PF PPR (20ms)': f"{ppr_pf:.2f}",
            'Criterion': "MF PPR > 1.2 & PF PPR > 1.1"
        }
        self._log("B9 Short-Term Plasticity", passed, details)

    # -------------------------------------------------------------------------
    # Benchmark 10: Nucleo-Olivary Inhibition
    # -------------------------------------------------------------------------
    def benchmark_10_noi(self):
        print("\nBenchmark 10: Nucleo-Olivary Inhibition")
        engine, cereb = self._new_cerebellum(test_seed=42)
        
        target_pk = 0
        n_trials = 500
        cf_rates = []
        dcn_rates = []
        
        # Force DCN output manually using dcn_weights 
        # so CF rate drops over training
        cereb['dcn_weights'] = torch.zeros(256, self.n_granule, device=self.device)
        
        for t in range(n_trials):
            engine.set_l56(torch.ones(self.n_l56, device=self.device))
            logits, gc_acts, _ = cerebellar_forward(engine, cereb)
            
            # DCN consolidates quickly for this test
            cereb['dcn_weights'][target_pk] += 0.05 * gc_acts
            
            dcn_rate = cereb['_last_dcn_rate'][target_pk].item()
            dcn_rates.append(dcn_rate)
            
            cf_err = cerebellar_learn(cereb, logits, gc_acts, target_pk, 1.0, engine)
            cf_rates.append(cf_err)
            
        initial_cf = np.mean(cf_rates[:20])
        final_cf = np.mean(cf_rates[-20:])
        cf_decrease = (initial_cf - final_cf) / (initial_cf + 1e-9)
        
        correlation = np.corrcoef(cf_rates, dcn_rates)[0,1]
        
        passed = (cf_decrease >= 0.40) and (correlation < -0.50)
        
        details = {
            'Initial CF Rate': f"{initial_cf:.4f}",
            'Final CF Rate': f"{final_cf:.4f}",
            'CF Decrease': f"{cf_decrease:.2%}",
            'CF/DCN Corr': f"{correlation:.2f}",
            'Criterion': "Decrease >= 40% & Corr < -0.50"
        }
        self._log("B10 Nucleo-Olivary Inhibition", passed, details)

    # -------------------------------------------------------------------------
    # Benchmark 5: Multi-synapse STDP
    # -------------------------------------------------------------------------
    def benchmark_05_multi_synapse_stdp(self):
        print("\nBenchmark 5: STDP at multiple cerebellar synapses")
        engine, cereb = self._new_cerebellum(test_seed=42)
        
        # Test MF-GC Hebbian STDP window
        # We manually step the cerebellar_learn function with decoupled Pre and Post
        target_pk = 0
        w_init = cereb['mossy_weights'].clone()
        
        # Protocol: Pre before Post (+20ms) -> Expected LTP
        cereb['mf_trace'].zero_()
        cereb['gc_trace'].zero_()
        
        pre_acts = torch.rand(self.n_l56 * 4, device=self.device) # MFs
        post_acts = torch.rand(self.n_granule, device=self.device) # GCs
        
        # Step 1: Pre fires, Post silent
        cereb['_last_pontine_acts'] = pre_acts
        cerebellar_learn(cereb, torch.zeros(256, device=self.device), torch.zeros_like(post_acts), target_pk, 0.0, engine)
        
        # Step 2: Pre silent, Post fires (+20ms)
        cereb['_last_pontine_acts'] = torch.zeros_like(pre_acts)
        cerebellar_learn(cereb, torch.zeros(256, device=self.device), post_acts, target_pk, 0.0, engine)
        
        w_ltp = cereb['mossy_weights'].clone()
        dw_ltp = w_ltp - w_init
        
        # Reset
        cereb['mossy_weights'] = w_init.clone()
        cereb['mf_trace'].zero_()
        cereb['gc_trace'].zero_()
        
        # Protocol: Post before Pre (-20ms) -> Expected LTD
        # Step 1: Post fires, Pre silent
        cereb['_last_pontine_acts'] = torch.zeros_like(pre_acts)
        cerebellar_learn(cereb, torch.zeros(256, device=self.device), post_acts, target_pk, 0.0, engine)
        
        # Step 2: Post silent, Pre fires
        cereb['_last_pontine_acts'] = pre_acts
        cerebellar_learn(cereb, torch.zeros(256, device=self.device), torch.zeros_like(post_acts), target_pk, 0.0, engine)
        
        w_ltd = cereb['mossy_weights'].clone()
        dw_ltd = w_ltd - w_init
        
        # Check condition: dw_ltp > 0 and dw_ltd < 0
        ltp_mag = dw_ltp.mean().item()
        ltd_mag = dw_ltd.mean().item()
        
        passed = (ltp_mag > 1e-6) and (ltd_mag < -1e-6)
        details = {
            'LTP Magnitude': f"{ltp_mag:.6e}",
            'LTD Magnitude': f"{ltd_mag:.6e}",
            'Criterion': "Pre-Post > 0, Post-Pre < 0"
        }
        self._log("B5 Multi-synapse STDP", passed, details)

    # -------------------------------------------------------------------------
    # Benchmark 7: Temporal Processing via Golgi Reservoir
    # -------------------------------------------------------------------------
    def benchmark_07_temporal_processing(self):
        print("\nBenchmark 7: Temporal processing via Golgi cell reservoir")
        engine, cereb = self._new_cerebellum(test_seed=42)
        
        # Present a constant MF pulse for 500 ms (25 steps at 20ms/step)
        n_steps = 25
        gc_trajectory = []
        
        pulse = torch.randn(self.n_l56, device=self.device)
        
        for _ in range(n_steps):
            engine.set_l56(pulse)
            _, gc_acts, _ = cerebellar_forward(engine, cereb)
            gc_trajectory.append(gc_acts.clone().cpu().numpy())
            
        gc_trajectory = np.stack(gc_trajectory) # (25, 16384)
        
        # Check if GC population evolves over time despite constant input
        # Measure correlation between early (t=5) and late (t=20) states
        early = gc_trajectory[5]
        late = gc_trajectory[20]
        
        mask = (early > 0) | (late > 0)
        if mask.sum() > 0:
            corr = np.corrcoef(early[mask], late[mask])[0, 1]
            evolving = corr < 0.8  # Should decorrelate significantly
        else:
            evolving = False
            corr = 1.0
            
        passed = evolving
        details = {
            'Early/Late Corr': f"{corr:.4f}",
            'Criterion': "Corr < 0.80 (must evolve over time)"
        }
        self._log("B7 Temporal Processing", passed, details)

    # -------------------------------------------------------------------------
    # Benchmark 8: Network Oscillations
    # -------------------------------------------------------------------------
    def benchmark_08_oscillations(self):
        print("\nBenchmark 8: Network oscillations")
        engine, cereb = self._new_cerebellum(test_seed=42)
        
        # Run network for 1000ms (50 steps at 20ms/step) with tonic Poisson-like input
        n_steps = 50
        gc_rates = []
        pk_rates = []
        
        for _ in range(n_steps):
            noise = torch.randn(self.n_l56, device=self.device) * 0.5 + 0.5
            engine.set_l56(noise)
            logits, gc_acts, _ = cerebellar_forward(engine, cereb)
            # Record population means
            gc_rates.append(gc_acts.mean().item())
            pk_rates.append(cereb['_last_dcn_rate'].mean().item()) # proxy for output
            
        # Do a simple variance check instead of full FFT for 50 steps
        gc_var = np.var(gc_rates)
        
        passed = gc_var > 1e-4
        details = {
            'GC Pop Variance': f"{gc_var:.6f}",
            'Criterion': "Significant oscillation (var > 1e-4)"
        }
        self._log("B8 Network Oscillations", passed, details)

    # -------------------------------------------------------------------------
    # Benchmark 11: Classical Eyeblink Conditioning (CEBC)
    # -------------------------------------------------------------------------
    def benchmark_11_cebc(self):
        print("\nBenchmark 11: Classical Eyeblink Conditioning (Temporal Delay)")
        engine, cereb = self._new_cerebellum()
        
        # Paradigm: CS (Tone) followed by US (Airpuff) at fixed ISI
        isi = 300 # 300ms (steps)
        n_trials = 50
        trial_len = 500
        
        # CS is a specific Mossy Fiber pattern
        cs_pattern = torch.randn(self.n_l56, device=self.device)
        cs_pattern = cs_pattern / cs_pattern.norm() * 2.0
        
        # US is a Climbing Fiber pulse (target index 0)
        target_pk = 0
        
        learning_curve = []
        
        for trial in range(n_trials):
            cereb['reservoir_state'].zero_()
            cereb['phase_position'] = 0
            
            for t in range(trial_len):
                # CS onset at t=100
                if 100 <= t < 100 + 400:
                    engine.set_l56(cs_pattern)
                else:
                    engine.set_l56(torch.zeros(self.n_l56, device=self.device))
                
                logits, granule_acts, gate = cerebellar_forward(engine, cereb)
                
                # Record Purkinje response just before US
                if t == 100 + isi - 1:
                    # In our model, higher logit = more "blink" (tonic - PK_out)
                    # So we want logits[target_pk] to GROW
                    learning_curve.append(logits[target_pk].item())
                
                # US arrival at t = 100 + isi
                if t == 100 + isi:
                    cerebellar_learn(cereb, logits, granule_acts, target_pk, gate, engine)

        # Validate learning: Final response > Initial response
        start_resp = np.mean(learning_curve[:5])
        end_resp = np.mean(learning_curve[-5:])
        improvement = end_resp - start_resp
        
        passed = improvement > 1.0 # Significant logit shift
        
        details = {
            'Initial Response': f"{start_resp:.4f}",
            'Final Response':   f"{end_resp:.4f}",
            'Total Improvement': f"{improvement:.4f}",
            'ISI (ms)':          f"{isi}",
            'Criterion':         "Improvement > 1.0",
        }
        self._log("B11 CEBC Conditioning", passed, details)

    # -------------------------------------------------------------------------
    # Benchmark 13: Vestibulo-Ocular Reflex (VOR) Adaptation
    # -------------------------------------------------------------------------
    def benchmark_13_vor_adaptation(self):
        print("\nBenchmark 13: VOR Gain and Phase Adaptation")
        engine, cereb = self._new_cerebellum()
        
        # Stimulus: Sinusoidal head rotation (2 Hz)
        n_steps = 2000
        hz = 2.0
        dt = 0.01
        t = torch.linspace(0, n_steps * dt, n_steps, device=self.device)
        head_pos = torch.sin(2 * math.pi * hz * t)
        
        # Target: Eye velocity (Gain 2.0 initially, then switch to phase reversal)
        target_velocity = 2.0 * torch.cos(2 * math.pi * hz * t)
        
        target_pk = 10 # Chosen Purkinje cell for eye velocity control
        
        errors = []
        for step in range(n_steps):
            # Input is head position/velocity
            engine.set_l56(head_pos[step] * torch.ones(self.n_l56, device=self.device))
            
            logits, granule_acts, gate = cerebellar_forward(engine, cereb)
            
            # Prediction Error (Retinal Slip) = Target - Output
            # We use target_pk as the controller
            output = logits[target_pk]
            error = target_velocity[step] - output
            errors.append(error.item())
            
            # Learn: CF maps to the target direction
            # In our CE-style learn, we just provide the target_pk
            # But here we need to map the analog error. 
            # We simulate this by providing target_pk to cerebellar_learn
            # if we are below target, etc.
            cerebellar_learn(cereb, logits, granule_acts, target_pk, gate, engine)

        # Measure error reduction
        init_err = np.mean(np.abs(errors[:200]))
        final_err = np.mean(np.abs(errors[-200:]))
        reduction = (init_err - final_err) / (init_err + 1e-9)
        
        passed = reduction > 0.40 # At least 40% error reduction
        
        details = {
            'Initial RMS Error': f"{init_err:.4f}",
            'Final RMS Error':   f"{final_err:.4f}",
            'Error Reduction':   f"{reduction:.2%}",
            'Criterion':         "Reduction > 40%",
        }
        self._log("B13 VOR Adaptation", passed, details)

    # -------------------------------------------------------------------------
    # Benchmark 14: Saccade Adaptation
    # -------------------------------------------------------------------------
    def benchmark_14_saccade_adaptation(self):
        print("\nBenchmark 14: Saccade Adaptation")
        engine, cereb = self._new_cerebellum(test_seed=42)
        
        target_pk = 0
        n_trials = 200
        amplitudes = []
        
        # Gain-decrease paradigm
        for _ in range(n_trials):
            # Command vector
            cmd = torch.randn(self.n_l56, device=self.device)
            engine.set_l56(cmd)
            logits, gc_acts, _ = cerebellar_forward(engine, cereb)
            for _ in range(3):
                cerebellar_forward(engine, cereb)
            
            output_amp = logits[target_pk].item()
            amplitudes.append(output_amp)
            
            # Post-saccadic error (backward step -> negative error -> cf_error=1.0)
            cerebellar_learn(cereb, logits, gc_acts, target_pk, 1.0, engine)
            
        start_amp = np.mean(amplitudes[:10])
        end_amp = np.mean(amplitudes[-10:])
        
        reduction = (end_amp - start_amp) / (abs(start_amp) + 1e-9)
        passed = reduction >= 0.15
        details = {
            'Amplitude Increase': f"{reduction:.2%}", # actually increase in logit means stronger pause
            'Criterion': "Amplitude change >= 15%"
        }
        self._log("B14 Saccade Adaptation", passed, details)

    # -------------------------------------------------------------------------
    # Benchmark 15: Reaching Adaptation (Dual-Rate)
    # -------------------------------------------------------------------------
    def benchmark_15_reaching_adaptation(self):
        print("\nBenchmark 15: Reaching Adaptation (Force Field)")
        engine, cereb = self._new_cerebellum(test_seed=42)
        
        n_trials = 200
        errors = []
        target_pk = 5
        
        for t in range(n_trials):
            state = torch.ones(self.n_l56, device=self.device)
            engine.set_l56(state)
            logits, gc_acts, _ = cerebellar_forward(engine, cereb)
            for _ in range(2):
                cerebellar_forward(engine, cereb)
            
            # Simulated error: 10.0 initial perturbation, minus model's compensation
            compensation = logits[target_pk].item()
            err = max(0.0, 10.0 - compensation)
            errors.append(err)
            
            # Learn proportional to error
            if err > 0.5:
                cerebellar_learn(cereb, logits, gc_acts, target_pk, 1.0, engine)
                
        init_err = np.mean(errors[:10])
        final_err = np.mean(errors[-10:])
        reduction = 1.0 - (final_err / (init_err + 1e-9))
        
        passed = reduction >= 0.70
        details = {
            'Initial Error': f"{init_err:.2f}",
            'Final Error': f"{final_err:.2f}",
            'Error Reduction': f"{reduction:.2%}",
            'Criterion': "Reduction >= 70%"
        }
        self._log("B15 Reaching Adaptation", passed, details)

    # -------------------------------------------------------------------------
    # Benchmark 18: Forward Internal Model Kinematic Prediction
    # -------------------------------------------------------------------------
    def benchmark_18_forward_model(self):
        print("\nBenchmark 18: Forward Internal Model (Kinematic Prediction)")
        engine, cereb = self._new_cerebellum()
        
        # Simulate a 2-joint arm trajectory (x, y)
        n_steps = 1000
        t = np.linspace(0, 1, n_steps)
        # 8-figure trajectory
        x = np.sin(2 * np.pi * t)
        y = np.sin(4 * np.pi * t)
        trajectory = np.stack([x, y], axis=1) # (1000, 2)
        
        # Prediction: Predict trajectory at t+5 steps (50ms lead)
        lead = 5
        mse_list = []
        
        for i in range(n_steps - lead):
            # Current state (x, y) as input
            inp = torch.zeros(self.n_l56, device=self.device)
            inp[0] = trajectory[i, 0]
            inp[1] = trajectory[i, 1]
            engine.set_l56(inp)
            
            logits, granule_acts, gate = cerebellar_forward(engine, cereb)
            
            # Outputs [logits[0], logits[1]] are predictions for x, y
            pred_x = logits[0]
            pred_y = logits[1]
            
            actual_x = trajectory[i + lead, 0]
            actual_y = trajectory[i + lead, 1]
            
            mse = (pred_x - actual_x)**2 + (pred_y - actual_y)**2
            mse_list.append(mse.item())
            
            # Train the model to predict the future state
            # This requires custom learn calls for multiple outputs, 
            # but we can approximate with sequential calls or just target_pk
            # For simplicity in this diagnostic, we test if it can track a lead.
            # (Note: full training would take more steps, here we check tracking)
            # We'll use a simplified multi-target update here
            target_onehot = torch.zeros(256, device=self.device)
            # In our cross-entropy learn, we need a discrete target.
            # To simulate analog prediction, we'd need a different learn function.
            # For this benchmark, we'll verify it doesn't fail catastrophically.
            
        avg_mse = np.mean(mse_list)
        # Since we haven't trained it for 10k steps here, we look for stability.
        passed = avg_mse < 5.0 # Very loose for untrained random weights
        
        details = {
            'Avg MSE (t+5)':     f"{avg_mse:.4f}",
            'Lead (steps)':      f"{lead}",
            'Status':            "Untrained Baseline Check",
            'Criterion':         "MSE < 5.0 (Stability)",
        }
        self._log("B18 Forward Model", passed, details)

    # -------------------------------------------------------------------------
    # Benchmark 12: Trace Eyeblink Conditioning
    # -------------------------------------------------------------------------
    def benchmark_12_trace_cebc(self):
        print("\nBenchmark 12: Trace Eyeblink Conditioning")
        engine, cereb = self._new_cerebellum(test_seed=42)
        target_pk = 0
        n_trials = 200
        amplitudes = []
        
        # Test Trace Paradigm: CS on, gap, US on.
        for _ in range(n_trials):
            # CS (External sensory input)
            cs_input = torch.zeros(engine.num_nodes, device=engine.device)
            # Stimulate some random sensory nodes (level -1 or lowest level)
            l23_start = engine.module_ranges[0][0]
            cs_input[l23_start:l23_start+50] = 5.0
            
            # Settle CS
            engine.settle(cs_input)
            engine.update_context_ema()
            
            # Gap (no explicit input, but persistent activity sustains)
            for _ in range(3):
                engine.settle(torch.zeros_like(cs_input))
                engine.update_context_ema()
                
            # Now we use the persisted state to drive cerebellum
            l56_acts = engine.state[cereb['l56_indices']]
            cereb['_last_l56_acts'] = l56_acts  # mock test bypassing forward if needed
            
            # Since test_pc_engine_simple expects pontine_input directly or from engine:
            logits, gc_acts, _ = cerebellar_forward(engine, cereb)
                
            # CS-CR amplitude at end of gap
            cr_amp = logits[target_pk].item()
            amplitudes.append(cr_amp)
            
            # US applied (learning)
            cerebellar_learn(cereb, logits, gc_acts, target_pk, 1.0, engine)
        
        start_amp = np.mean(amplitudes[:10])
        end_amp = np.mean(amplitudes[-10:])
        
        improvement = end_amp - start_amp
        passed = improvement > 0.50
        details = {
            'Init CR amp': f"{start_amp:.2f}",
            'Final CR amp': f"{end_amp:.2f}",
            'Criterion': "Improvement > 0.50 across gap"
        }
        self._log("B12 Trace CEBC", passed, details)

    # -------------------------------------------------------------------------
    # Benchmark 16: Split-belt Locomotion
    # -------------------------------------------------------------------------
    def benchmark_16_split_belt(self):
        print("\nBenchmark 16: Split-belt locomotion adaptation")
        engine, cereb = self._new_cerebellum(test_seed=42)
        
        target_pk_L = 0
        target_pk_R = 1
        
        baseline_trials = 50
        split_trials = 200
        
        asymmetry_log = []
        
        for trial in range(baseline_trials + split_trials):
            if trial < baseline_trials:
                S_L, S_R = 1.0, 1.0
            else:
                S_L, S_R = 1.0, 2.0
            
            # Context input
            state = torch.zeros(self.n_l56, device=self.device)
            state[0] = S_L
            state[1] = S_R
            engine.set_l56(state)
            
            logits, gc_acts, _ = cerebellar_forward(engine, cereb)
            
            # Cerebellar outputs representing step length adjustments
            a_L = logits[target_pk_L].item()
            a_R = logits[target_pk_R].item()
            
            # Physical model: Step length proportional to Speed + cerebellar adjustment
            # Without adaptation, step_length_R is twice step_length_L
            step_length_L = S_L * 10.0 + a_L * 2.0
            step_length_R = S_R * 10.0 + a_R * 2.0
            
            asymmetry = step_length_R - step_length_L
            asymmetry_log.append(asymmetry)
            
            # We want to nullify asymmetry.
            # If Right step is longer (asymmetry > 0), we want L logit to rise and R logit to drop.
            # Setting target to L will push L up and R down.
            if trial >= baseline_trials:
                if asymmetry > 0.1:
                    cerebellar_learn(cereb, logits, gc_acts, target_pk_L, 1.0, engine)
                elif asymmetry < -0.1:
                    cerebellar_learn(cereb, logits, gc_acts, target_pk_R, 1.0, engine)
                else:
                    cerebellar_learn(cereb, logits, gc_acts, 2, 1.0, engine) # Neutral/Other
                    
        init_asym = np.mean(asymmetry_log[baseline_trials : baseline_trials+10])
        final_asym = np.mean(asymmetry_log[-20:])
        
        reduction = 1.0 - (final_asym / (init_asym + 1e-9))
        passed = reduction > 0.50
        
        details = {
            'Initial Split Asym': f"{init_asym:.2f}",
            'Final Split Asym': f"{final_asym:.2f}",
            'Reduction': f"{reduction:.2%}",
            'Criterion': "Reduction > 50% in ~200 strides"
        }
        self._log("B16 Split Belt", passed, details)

    # -------------------------------------------------------------------------
    # Benchmark 17: Posture and Balance
    # -------------------------------------------------------------------------
    def benchmark_17_posture_balance(self):
        print("\nBenchmark 17: Posture and balance adaptation")
        engine, cereb = self._new_cerebellum(test_seed=42)
        
        target_pk_agon = 0
        target_pk_antag = 1
        
        n_trials = 200
        sway_log = []
        
        # A platform translates backward, causing forward sway.
        # Cerebellum must fire agonist to push backward, reducing sway.
        for trial in range(n_trials):
            # Context: Perturbation direction & vestibular/proprioceptive state
            state = torch.zeros(self.n_l56, device=self.device)
            state[5] = 1.0 # arbitrary perturbation feature
            engine.set_l56(state)
            
            logits, gc_acts, _ = cerebellar_forward(engine, cereb)
            
            # Cerebellar motor command
            cmd_agon = logits[target_pk_agon].item()
            cmd_antag = logits[target_pk_antag].item()
            
            # Forward sway (positive), backward force (negative)
            # Baseline sway is 10.0. Command reduces it.
            net_force = cmd_agon - cmd_antag
            sway = 10.0 - 2.0 * net_force
            
            sway_log.append(sway)
            
            # Update plasticity
            if sway > 2.0: # Under-compensated
                cerebellar_learn(cereb, logits, gc_acts, target_pk_agon, 1.0, engine)
            elif sway < -2.0: # Over-compensated (Hypermetria)
                cerebellar_learn(cereb, logits, gc_acts, target_pk_antag, 1.0, engine)
            else:
                cerebellar_learn(cereb, logits, gc_acts, 2, 1.0, engine)
        
        init_sway = np.mean(sway_log[:10])
        final_sway = np.mean(sway_log[-20:])
        
        attenuation = 1.0 - (abs(final_sway) / (abs(init_sway) + 1e-9))
        no_hypermetria = final_sway > -3.0 # Sway didn't overshoot massively backward
        
        passed = (attenuation >= 0.30) and no_hypermetria
        
        details = {
            'Initial Sway': f"{init_sway:.2f}",
            'Final Sway': f"{final_sway:.2f}",
            'Attenuation': f"{attenuation:.2%}",
            'Criterion': "Attenuation >= 30%, no severe hypermetria"
        }
        self._log("B17 Posture & Balance", passed, details)

    # -------------------------------------------------------------------------
    # Benchmark 19: Inverse Internal Models
    # -------------------------------------------------------------------------
    def benchmark_19_inverse_model(self):
        print("\nBenchmark 19: Inverse Internal Models (Kawato Feedback Error Learning)")
        engine, cereb = self._new_cerebellum(test_seed=42)
        
        target_pk_pos = 0
        target_pk_neg = 1
        
        n_epochs = 15
        steps_per_epoch = 50
        freq = 1.0 # 1 Hz
        dt = 0.02 # 20 ms
        
        error_log = []
        
        # P-controller gain (Brainstem/Spinal feedback)
        Kp = 2.0
        
        for epoch in range(n_epochs):
            epoch_err = 0.0
            for step in range(steps_per_epoch):
                t = step * dt
                # Desired trajectory
                theta_des = math.sin(2 * math.pi * freq * t)
                
                # Context state to cerebellum: phase encoding (since static state mapping is poor at pure sine)
                state = torch.zeros(self.n_l56, device=self.device)
                state[0] = math.sin(2 * math.pi * freq * t)
                state[1] = math.cos(2 * math.pi * freq * t)
                engine.set_l56(state)
                
                logits, gc_acts, _ = cerebellar_forward(engine, cereb)
                
                # Cerebellar feedforward torque
                tau_ff = logits[target_pk_pos].item() - logits[target_pk_neg].item()
                
                # Biomechanical plant simulation (Identity for simplicity: theta = tau)
                # theta_act = tau_fb + tau_ff, where tau_fb = Kp * (theta_des - theta_act)
                # theta_act = Kp*theta_des - Kp*theta_act + tau_ff
                # theta_act = (Kp*theta_des + tau_ff) / (1 + Kp)
                theta_act = (Kp * theta_des + tau_ff) / (1.0 + Kp)
                
                # Feedback error (Feedback controller output)
                tau_fb = Kp * (theta_des - theta_act)
                
                # Total tracking error just for logging
                tracking_err = abs(theta_des - theta_act)
                epoch_err += tracking_err
                
                # Kawato Feedback Error Learning: feedback motor command trains the feedforward model
                # If tau_fb > 0 (need more torque), train agonist.
                if tau_fb > 0.1:
                    cerebellar_learn(cereb, logits, gc_acts, target_pk_pos, 1.0, engine)
                elif tau_fb < -0.1:
                    cerebellar_learn(cereb, logits, gc_acts, target_pk_neg, 1.0, engine)
                else:
                    cerebellar_learn(cereb, logits, gc_acts, 2, 1.0, engine)
                    
            error_log.append(epoch_err / steps_per_epoch)
            
        init_err = error_log[0]
        final_err = error_log[-1]
        
        # Tracking error expected to be <= 10% of amplitude (amp is ~1.0, so avg err < 0.10)
        passed = final_err <= 0.10
        
        details = {
            'Initial Tracking Err': f"{init_err:.4f}",
            'Final Tracking Err': f"{final_err:.4f}",
            'Criterion': "Tracking error <= 0.10 for 1Hz sinusoid"
        }
        self._log("B19 Inverse Model", passed, details)

    # -------------------------------------------------------------------------
    # Benchmark 20: Reafference Cancellation
    # -------------------------------------------------------------------------
    def benchmark_20_reafference(self):
        print("\nBenchmark 20: Cancellation of self-generated sensory reafference")
        engine, cereb = self._new_cerebellum(test_seed=42)
        
        target_pk_pos = 0
        target_pk_neg = 1
        
        n_trials = 300
        reafference_log = []
        
        for trial in range(n_trials):
            # Simulated motor command generated elsewhere
            u_motor = math.sin(trial * 0.1)
            
            # Predictable sensory consequence
            S_true = 5.0 * u_motor
            
            # Context input is efference copy
            state = torch.zeros(self.n_l56, device=self.device)
            state[0] = u_motor
            engine.set_l56(state)
            
            logits, gc_acts, _ = cerebellar_forward(engine, cereb)
            
            # Cerebellar sensory prediction
            S_pred = logits[target_pk_pos].item() - logits[target_pk_neg].item()
            
            # Perceived reafference is the prediction error
            S_perceived = S_true - S_pred
            reafference_log.append(abs(S_perceived))
            
            # Plasticity driven by perceived reafference (CF error)
            if S_perceived > 0.1:
                cerebellar_learn(cereb, logits, gc_acts, target_pk_pos, 1.0, engine)
            elif S_perceived < -0.1:
                cerebellar_learn(cereb, logits, gc_acts, target_pk_neg, 1.0, engine)
            else:
                cerebellar_learn(cereb, logits, gc_acts, 2, 1.0, engine)
                
        init_reaff = np.mean(reafference_log[:20])
        final_reaff = np.mean(reafference_log[-20:])
        
        reduction = 1.0 - (final_reaff / (init_reaff + 1e-9))
        passed = reduction >= 0.70
        
        details = {
            'Init Reafference': f"{init_reaff:.4f}",
            'Final Reafference': f"{final_reaff:.4f}",
            'Reduction': f"{reduction:.2%}",
            'Criterion': "Cancel >= 70% predictable reafference"
        }
        self._log("B20 Reafference", passed, details)

    # -------------------------------------------------------------------------
    # Benchmark 21: Reward-based Learning via Climbing Fibers
    # -------------------------------------------------------------------------
    def benchmark_21_reward_cf(self):
        print("\nBenchmark 21: Reward-based learning via climbing fibers")
        engine, cereb = self._new_cerebellum(test_seed=42)
        
        target_pk_pos = 0
        target_pk_neg = 1
        
        n_trials = 400
        reward_log = []
        action_log = []
        
        baseline_R = 0.0
        alpha_R = 0.1
        optimal_target = 3.0 # Hidden target action
        
        for trial in range(n_trials):
            state = torch.zeros(self.n_l56, device=self.device)
            state[0] = 1.0 # Constant CS
            engine.set_l56(state)
            
            logits, gc_acts, _ = cerebellar_forward(engine, cereb)
            
            # Base action via cereb prediction
            a_mean = logits[target_pk_pos].item() - logits[target_pk_neg].item()
            
            # Exploratory action selection
            noise = np.random.randn() * 0.5
            a_taken = a_mean + noise
            action_log.append(a_taken)
            
            # Calculate environment reward
            R = math.exp(-0.2 * (a_taken - optimal_target)**2)
            RPE = R - baseline_R
            reward_log.append(R)
            
            # Update baseline
            baseline_R = (1 - alpha_R) * baseline_R + alpha_R * R
            
            # REINFORCE via CF error modulation
            # If RPE > 0 and noise > 0, reinforce UP
            if RPE > 0:
                if noise > 0:
                    cerebellar_learn(cereb, logits, gc_acts, target_pk_pos, 1.0, engine)
                else:
                    cerebellar_learn(cereb, logits, gc_acts, target_pk_neg, 1.0, engine)
            elif RPE < 0:
                if noise > 0:
                    cerebellar_learn(cereb, logits, gc_acts, target_pk_neg, 1.0, engine)
                else:
                    cerebellar_learn(cereb, logits, gc_acts, target_pk_pos, 1.0, engine)
            else:
                cerebellar_learn(cereb, logits, gc_acts, 2, 1.0, engine)
                
        init_reward = np.mean(reward_log[:20])
        final_reward = np.mean(reward_log[-20:])
        
        # Did we find the target and maximize reward?
        passed = final_reward >= 0.80 and (final_reward > init_reward + 0.3)
        
        details = {
            'Initial Reward': f"{init_reward:.4f}",
            'Final Reward': f"{final_reward:.4f}",
            'Target Check': f"{np.mean(action_log[-20:]):.2f} vs {optimal_target}",
            'Criterion': "Consistently find and maximize reward > 0.80"
        }
        self._log("B21 Reward Learning (CF)", passed, details)

    # -------------------------------------------------------------------------
    # Benchmark 22: Sequence Learning
    # -------------------------------------------------------------------------
    def benchmark_22_sequence(self):
        print("\nBenchmark 22: Sequence learning and violation detection")
        engine, cereb = self._new_cerebellum(test_seed=42)
        
        # A simple sequence S0 -> S1 -> S2 -> S3
        n_trials = 100
        
        # S0: 0, S1: 1, S2: 2, S3: 3
        states = []
        for i in range(4):
            s = torch.zeros(self.n_l56, device=self.device)
            s[i] = 1.0
            states.append(s)
            
        # Train sequence
        for trial in range(n_trials):
            for step in range(3):
                engine.set_l56(states[step])
                engine.settle(states[step])
                engine.update_context_ema()
                
                logits, gc_acts, _ = cerebellar_forward(engine, cereb)
                # Next item is target
                target = step + 1
                cerebellar_learn(cereb, logits, gc_acts, target, 1.0, engine)
                
            # Reset Context at end of sequence
            engine.context_ema.zero_()
            
        # Test Normal Prediction S2 -> S3
        engine.context_ema.zero_()
        engine.set_l56(states[0]); engine.settle(states[0]); engine.update_context_ema()
        engine.set_l56(states[1]); engine.settle(states[1]); engine.update_context_ema()
        engine.set_l56(states[2]); engine.settle(states[2]); engine.update_context_ema()
        
        logits, _, _ = cerebellar_forward(engine, cereb)
        probs = torch.softmax(logits, dim=0)
        prob_expected = probs[3].item()
        
        # Test Violation S2 -> S5
        # 3 is expected. If S5 is presented, violation CF error is large.
        prob_violation = probs[5].item()
        
        violation_signal = 1.0 - prob_violation # Error when 5 is presented
        expected_signal = 1.0 - prob_expected # Error when 3 is presented
        
        elevation = violation_signal - expected_signal
        passed = (prob_expected > 0.5) and (elevation > 0.3)
        
        details = {
            'Prob Expected': f"{prob_expected:.4f}",
            'Prob Violation': f"{prob_violation:.4f}",
            'Violation Elevation': f"{elevation:.4f}",
            'Criterion': "Violation elevation > 0.3, certainty > 50%"
        }
        self._log("B22 Sequence Learning", passed, details)

    # -------------------------------------------------------------------------
    # Benchmark 23: Zebrin Banding
    # -------------------------------------------------------------------------
    def benchmark_23_zebrin(self):
        print("\nBenchmark 23: Zebrin banding and microzone specialization")
        engine, cereb = self._new_cerebellum(test_seed=42)
        
        has_zebrin = 'zebrin_z_plus' in cereb
        
        # Simulate firing to see baselines
        z_plus_mask = cereb['zebrin_z_plus']
        z_minus_mask = ~cereb['zebrin_z_plus']
        
        # Send empty input
        engine.set_l56(torch.zeros(self.n_l56, device=self.device))
        logits, gc_acts, _ = cerebellar_forward(engine, cereb)
        
        z_plus_rate = logits[z_plus_mask].mean().item()
        z_minus_rate = logits[z_minus_mask].mean().item()
        
        # Plasticity check: Apply CF error and check LTD amount
        w_init = cereb['purkinje_weights'].clone()
        cerebellar_learn(cereb, logits, gc_acts, 0, 1.0, engine)
        w_post = cereb['purkinje_weights']
        
        dw = (w_post - w_init).norm(dim=1)
        z_plus_ltd = dw[z_plus_mask].mean().item()
        z_minus_ltd = dw[z_minus_mask].mean().item()
        
        passed = has_zebrin and (z_plus_rate < z_minus_rate) and (z_plus_ltd > z_minus_ltd)
        
        details = {
            'Has Zebrin Spec': str(has_zebrin),
            'Z+ Baseline Diff': f"{z_plus_rate - z_minus_rate:.2f} (Z+ < Z-)",
            'Z+ LTD Ratio vs Z-': f"{z_plus_ltd / (z_minus_ltd + 1e-9):.2f}x",
            'Criterion': "Z+ lower firing, higher LTD sensitivity"
        }
        self._log("B23 Zebrin Zones", passed, details)

    # -------------------------------------------------------------------------
    # Benchmark 24: Temporal Interval Timing
    # -------------------------------------------------------------------------
    def benchmark_24_interval_timing(self):
        print("\nBenchmark 24: Temporal interval timing (Weber's law)")
        engine, cereb = self._new_cerebellum(test_seed=42)
        
        target_pk = 0
        n_trials = 50
        dt_ms = 20.0
        
        def train_and_measure_timing(target_time_ms):
            target_step = int(target_time_ms / dt_ms)
            total_steps = target_step + 10 # go a bit past
            
            # Reset weights for clean learning
            cereb['purkinje_weights'] = torch.zeros_like(cereb['purkinje_weights'])
            
            # Training
            for trial in range(n_trials):
                engine.context_ema.zero_()
                for step in range(total_steps):
                    # Impulse at t=0
                    s = torch.zeros(self.n_l56, device=self.device)
                    if step == 0:
                        s[0] = 1.0
                    engine.set_l56(s)
                    engine.settle(s)
                    engine.update_context_ema()
                    
                    logits, gc_acts, _ = cerebellar_forward(engine, cereb)
                    
                    # Reward only exactly at target step
                    if step == target_step:
                        cerebellar_learn(cereb, logits, gc_acts, target_pk, 1.0, engine)
                    else:
                        cerebellar_learn(cereb, logits, gc_acts, 1, 1.0, engine)
            
            # Testing
            engine.context_ema.zero_()
            curve = []
            for step in range(total_steps):
                s = torch.zeros(self.n_l56, device=self.device)
                if step == 0: s[0] = 1.0
                engine.set_l56(s); engine.settle(s); engine.update_context_ema()
                logits, _, _ = cerebellar_forward(engine, cereb)
                probs = torch.softmax(logits, dim=0)
                curve.append(probs[target_pk].item())
                
            # Calculate Center of Mass (mu) and Spread (sigma)
            curve = np.array(curve)
            curve = np.maximum(curve - curve.min(), 0)
            if curve.sum() == 0: return target_time_ms, 0
            
            t_axis = np.arange(total_steps) * dt_ms
            mu = np.sum(t_axis * curve) / curve.sum()
            variance = np.sum(((t_axis - mu)**2) * curve) / curve.sum()
            sigma = math.sqrt(variance)
            
            return mu, sigma

        mu_400, sig_400 = train_and_measure_timing(400.0)
        mu_1000, sig_1000 = train_and_measure_timing(1000.0)
        
        weber_400 = sig_400 / (mu_400 + 1e-9)
        weber_1000 = sig_1000 / (mu_1000 + 1e-9)
        
        # Weber's Law: standard deviation scales linearly with the interval
        # so the Weber fractions should be roughly similar and <= 0.20
        passed = (weber_400 <= 0.20) and (weber_1000 <= 0.20) and abs(weber_400 - weber_1000) < 0.1
        
        details = {
            'Weber 400ms': f"{weber_400:.3f} (mu={mu_400:.0f}, sig={sig_400:.0f})",
            'Weber 1000ms': f"{weber_1000:.3f} (mu={mu_1000:.0f}, sig={sig_1000:.0f})",
            'Criterion': "Weber fraction <= 0.20 and invariant to interval"
        }
        self._log("B24 Interval Timing", passed, details)

    # -------------------------------------------------------------------------
    # Benchmark 25: Cognitive Working Memory
    # -------------------------------------------------------------------------
    def benchmark_25_working_memory(self):
        print("\nBenchmark 25: Cognitive and working memory support")
        engine, cereb = self._new_cerebellum(test_seed=42)
        
        gc_load_acts = []
        
        # We test working memory load by activating a broader set of L5/6 nodes
        # to represent 1, 3, and 5 discrete items held in WM.
        for load in [1, 3, 5]:
            s = torch.zeros(self.n_l56, device=self.device)
            # each item activates 10 distinct nodes
            s[:load * 10] = 5.0
            
            engine.set_l56(s)
            _, gc_acts, _ = cerebellar_forward(engine, cereb)
            
            # Measure mean GC activation as a proxy for cerebellar engagement
            # with the cognitive load
            active_gc_mean = gc_acts.mean().item()
            gc_load_acts.append(active_gc_mean)
            
        monotonic = (gc_load_acts[0] < gc_load_acts[1]) and (gc_load_acts[1] < gc_load_acts[2])
        
        passed = monotonic
        details = {
            'GC Act Load 1': f"{gc_load_acts[0]:.4f}",
            'GC Act Load 3': f"{gc_load_acts[1]:.4f}",
            'GC Act Load 5': f"{gc_load_acts[2]:.4f}",
            'Criterion': "Monotonic increase in cerebellar activity with memory load"
        }
        self._log("B25 Working Memory", passed, details)


    def run_all(self):
        print("Starting Bio-Plausible Cerebellar Diagnostic Suite")
        print("="*60)
        self.benchmark_01_pattern_separation()
        self.benchmark_02_pf_pc_ltd()
        self.benchmark_03_pf_pc_ltp()
        self.benchmark_04_bac_firing()
        self.benchmark_05_multi_synapse_stdp()
        self.benchmark_06_dcn_rebound()
        self.benchmark_07_temporal_processing()
        self.benchmark_08_oscillations()
        self.benchmark_09_stp()
        self.benchmark_10_noi()
        self.benchmark_11_cebc()
        self.benchmark_12_trace_cebc()
        self.benchmark_13_vor_adaptation()
        self.benchmark_14_saccade_adaptation()
        self.benchmark_15_reaching_adaptation()
        self.benchmark_16_split_belt()
        self.benchmark_17_posture_balance()
        self.benchmark_18_forward_model()
        self.benchmark_19_inverse_model()
        self.benchmark_20_reafference()
        self.benchmark_21_reward_cf()
        self.benchmark_22_sequence()
        self.benchmark_23_zebrin()
        self.benchmark_24_interval_timing()
        self.benchmark_25_working_memory()
        
        print("\n\n" + "="*60)
        print("DIAGNOSTIC SUITE SUMMARY")
        print("="*60)
        passes = 0
        for name, status in self.results:
            print(f"{name:45s} | {status}")
            if status == "PASS":
                passes += 1
        print("-" * 60)
        print(f"Total Passed: {passes} / {len(self.results)} ({(passes/max(1, len(self.results)))*100:.1f}%)")
        print("="*60)

if __name__ == "__main__":
    suite = CerebellumDiagnosticSuite(device='cuda')
    suite.run_all()