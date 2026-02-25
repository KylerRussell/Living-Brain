"""
Comprehensive diagnostic tests for the Living-Brain predictive coding network.

These tests systematically probe 10 potential failure modes identified from
training run analysis. Each test category targets a specific mechanism and
reports quantitative metrics to diagnose whether that mechanism is functional.

Usage:
    python -m ndcd.diagnostic_tests --device cpu --nodes 2000 --modules 20

    Or import and call individual tests after training:
        from ndcd.diagnostic_tests import ModelDiagnostics
        diag = ModelDiagnostics(trainer)
        diag.run_all()
"""

import torch
import numpy as np
import os
import argparse
import time
from typing import Dict, List, Optional, Tuple


class ModelDiagnostics:
    """Runs all diagnostic tests against a trained SequentialTrainer instance."""

    def __init__(self, trainer):
        """
        Args:
            trainer: A SequentialTrainer instance (trained or freshly initialized).
        """
        self.trainer = trainer
        self.engine = trainer.engine
        self.device = trainer.device
        self.num_nodes = trainer.num_nodes
        self.results = {}

    def run_all(self):
        """Run all diagnostic tests and print a summary."""
        print("\n" + "=" * 70)
        print("LIVING-BRAIN DIAGNOSTIC TEST SUITE")
        print("=" * 70)

        test_methods = [
            ("1A", "Output Node Prediction (PC hierarchy vs readout)", self.test_1a_output_node_prediction),
            ("1B", "Top-Down Prediction Quality", self.test_1b_topdown_prediction_quality),
            ("1C", "Edge Type Counts", self.test_1c_count_edge_types),
            ("2", "Gradient Alignment by Edge Type", self.test_2_gradient_alignment),
            ("3A", "Cascade Flow Dynamics", self.test_3a_cascade_dynamics),
            ("3B", "Deep Weight Stability (diversity reg impact)", self.test_3b_deep_stability),
            ("4A", "Replay Attractor Sparsity", self.test_4a_replay_sparsity),
            ("4B", "Replay Pattern Diversity", self.test_4b_replay_diversity),
            ("5A", "State Distribution (saturation check)", self.test_5a_state_distribution),
            ("5B", "Input Sensitivity", self.test_5b_input_sensitivity),
            ("6", "Temporal Prediction Usefulness", self.test_6_temporal_contribution),
            ("7", "CLS Hippocampal vs Neocortical Dynamics", self.test_7_cls_dynamics),
            ("8", "Retention with Re-trained Readout", self.test_8_true_retention),
            ("9", "Runtime Spectral Radius", self.test_9_runtime_spectral_radius),
            ("10", "IMEX Settle Convergence", self.test_10_settle_convergence),
        ]

        for test_id, name, method in test_methods:
            print(f"\n{'─' * 70}")
            print(f"TEST {test_id}: {name}")
            print(f"{'─' * 70}")
            try:
                result = method()
                self.results[test_id] = result
            except Exception as e:
                print(f"  ERROR: {e}")
                self.results[test_id] = {"error": str(e)}

        self._print_summary()
        return self.results

    def _print_summary(self):
        """Print a summary table of all test results with pass/warn/fail indicators."""
        print(f"\n{'=' * 70}")
        print("DIAGNOSTIC SUMMARY")
        print(f"{'=' * 70}")

        assessments = {
            "1A": self._assess_1a,
            "1B": self._assess_1b,
            "1C": self._assess_1c,
            "2": self._assess_2,
            "3A": self._assess_3a,
            "3B": self._assess_3b,
            "4A": self._assess_4a,
            "4B": self._assess_4b,
            "5A": self._assess_5a,
            "5B": self._assess_5b,
            "6": self._assess_6,
            "7": self._assess_7,
            "8": self._assess_8,
            "9": self._assess_9,
            "10": self._assess_10,
        }

        for test_id, assess_fn in assessments.items():
            result = self.results.get(test_id, {})
            if "error" in result:
                status = "ERROR"
                detail = result["error"]
            else:
                status, detail = assess_fn(result)
            indicator = {"PASS": "[OK]", "WARN": "[!!]", "FAIL": "[XX]", "ERROR": "[??]"}.get(status, "[??]")
            print(f"  {indicator} Test {test_id}: {detail}")

    # ─────────────────────────────────────────────────────────────────────
    # TEST 1A: Ablation — readout vs output-node prediction
    # ─────────────────────────────────────────────────────────────────────

    def test_1a_output_node_prediction(self, data_path="ndcd/data/level1_chars.txt",
                                        num_samples=2000) -> Dict:
        """Do the output nodes themselves predict anything, or only the readout?

        If output nodes ~ chance (0.4%) while readout >> chance, the PC
        hierarchy isn't contributing to prediction — only the linear readout is.
        """
        if not os.path.exists(data_path):
            print(f"  Data not found at {data_path}, using random data")
            data = bytes(np.random.randint(0, 256, 5000).tolist())
        else:
            with open(data_path, 'rb') as f:
                data = f.read()

        num_samples = min(num_samples, len(data) - 1)
        correct_output = 0
        correct_readout = 0

        saved_state = self.engine.state.clone()

        for i in range(num_samples):
            input_byte = data[i]
            target_byte = data[i + 1]

            # Settle
            input_vec = torch.zeros(self.num_nodes, device=self.device)
            input_vec[0:256] = self.trainer.eye[input_byte] * 1.0
            input_mask = torch.zeros(self.num_nodes, device=self.device)
            input_mask[list(range(256))] = 1.0
            self.engine.settle(input_vec, input_mask=input_mask, max_steps=50)

            # Output node prediction (raw network)
            output_state = self.engine.state[256:512]
            output_probs = torch.softmax(output_state, dim=0)
            if torch.argmax(output_probs).item() == target_byte:
                correct_output += 1

            # Readout prediction
            level0_acts = torch.tanh(self.engine.state[self.trainer.level0_indices])
            features = level0_acts
            logits = self.trainer.readout_W @ features + self.trainer.readout_b
            probs = torch.softmax(logits, dim=0)
            if torch.argmax(probs).item() == target_byte:
                correct_readout += 1

        self.engine.state = saved_state

        output_acc = correct_output / num_samples
        readout_acc = correct_readout / num_samples
        chance = 1.0 / 256

        print(f"  Output nodes accuracy:  {output_acc:.2%} (chance = {chance:.2%})")
        print(f"  Readout accuracy:       {readout_acc:.2%}")
        print(f"  Ratio (readout/output): {readout_acc / max(output_acc, 1e-6):.1f}x")

        if output_acc < chance * 2:
            print(f"  >> OUTPUT NODES AT CHANCE — PC hierarchy not contributing to prediction")
        elif output_acc > readout_acc * 0.5:
            print(f"  >> Output nodes carrying significant signal")

        return {
            "output_acc": output_acc,
            "readout_acc": readout_acc,
            "chance": chance,
        }

    def _assess_1a(self, r):
        if not r:
            return "WARN", "No data"
        output_acc = r.get("output_acc", 0)
        readout_acc = r.get("readout_acc", 0)
        chance = r.get("chance", 1 / 256)
        
        if readout_acc > 0.8 and output_acc < chance * 5:
            return "FAIL", f"MEMORIZATION DETECTED - Output nodes at chance ({output_acc:.1%}) but Readout is perfect ({readout_acc:.1%}). Network is an Echo State Network."
        elif output_acc > chance * 3 and output_acc > readout_acc * 0.3:
            return "PASS", f"Output nodes functional ({output_acc:.1%} acc, readout {readout_acc:.1%})"
        elif output_acc > chance * 2:
            return "WARN", f"Output nodes marginal ({output_acc:.1%} acc, readout {readout_acc:.1%})"
        else:
            return "FAIL", f"Output nodes at chance ({output_acc:.1%}), only readout works ({readout_acc:.1%})"

    # ─────────────────────────────────────────────────────────────────────
    # TEST 1B: Top-down prediction quality
    # ─────────────────────────────────────────────────────────────────────

    def test_1b_topdown_prediction_quality(self) -> Dict:
        """Measure how well top-down predictions match actual lower-level states.

        If prediction variance ~ 0 or correlation ~ 0, top-down isn't predicting.
        """
        rho = torch.tanh(self.engine.state)

        # Build top-down prediction
        cascade_weights = self.engine.effective_weights
        td_vals = (cascade_weights[self.engine.topdown_edge_mask] *
                   self.engine.facilitation[self.engine.topdown_edge_mask] *
                   self.engine.depression[self.engine.topdown_edge_mask])
        td_sparse = torch.sparse_coo_tensor(
            self.engine.topdown_indices, td_vals,
            (self.engine.num_nodes, self.engine.num_nodes))
        topdown_pred = torch.mv(td_sparse, rho)

        results_by_level = {}
        for level in range(self.engine.max_level):
            mask = self.engine.node_to_level == level
            if mask.sum() == 0:
                continue
            actual = self.engine.state[mask]
            pred = topdown_pred[mask]
            pred_var = pred.var().item()
            actual_var = actual.var().item()

            # Compute correlation safely
            if pred_var < 1e-10 or actual_var < 1e-10:
                corr = 0.0
            else:
                stacked = torch.stack([actual, pred])
                corr = torch.corrcoef(stacked)[0, 1].item()
                if np.isnan(corr):
                    corr = 0.0

            results_by_level[level] = {
                "correlation": corr,
                "pred_variance": pred_var,
                "actual_variance": actual_var,
            }
            print(f"  Level {level}: correlation={corr:.4f}, "
                  f"pred_var={pred_var:.6f}, actual_var={actual_var:.6f}")

        if all(v["pred_variance"] < 1e-4 for v in results_by_level.values()):
            print(f"  >> TOP-DOWN PREDICTIONS NEAR ZERO — hierarchy not generating predictions")
        elif all(abs(v["correlation"]) < 0.05 for v in results_by_level.values()):
            print(f"  >> TOP-DOWN PREDICTIONS UNCORRELATED WITH STATE — predictions not informative")

        return {"by_level": results_by_level}

    def _assess_1b(self, r):
        levels = r.get("by_level", {})
        if not levels:
            return "WARN", "No level data"
        avg_corr = np.mean([abs(v["correlation"]) for v in levels.values()])
        avg_pred_var = np.mean([v["pred_variance"] for v in levels.values()])
        if avg_corr > 0.1 and avg_pred_var > 1e-3:
            return "PASS", f"Top-down functional (avg |corr|={avg_corr:.3f}, pred_var={avg_pred_var:.4f})"
        elif avg_pred_var < 1e-4:
            return "FAIL", f"Top-down predictions near zero (pred_var={avg_pred_var:.6f})"
        else:
            return "WARN", f"Top-down weak (avg |corr|={avg_corr:.3f}, pred_var={avg_pred_var:.4f})"

    # ─────────────────────────────────────────────────────────────────────
    # TEST 1C: Count edge types and their proportions
    # ─────────────────────────────────────────────────────────────────────

    def test_1c_count_edge_types(self) -> Dict:
        """Count top-down, bottom-up, lateral, and I/O edge proportions.

        If top-down < 5% of edges, predictions are starved of bandwidth.
        """
        src_levels = self.engine.node_to_level[self.engine.indices[0]]
        dst_levels = self.engine.node_to_level[self.engine.indices[1]]

        topdown = (src_levels > dst_levels).sum().item()
        bottomup = (src_levels < dst_levels).sum().item()
        lateral = (src_levels == dst_levels).sum().item()
        total = len(self.engine.indices[0])

        # Further break down lateral into intra-IO and intra-module
        io_mask = (src_levels == 0) & (dst_levels == 0)
        io_edges = io_mask.sum().item()
        non_io_lateral = lateral - io_edges

        print(f"  Total edges: {total:,}")
        print(f"  Top-down (higher→lower): {topdown:,} ({topdown / total:.1%})")
        print(f"  Bottom-up (lower→higher): {bottomup:,} ({bottomup / total:.1%})")
        print(f"  Lateral (same level):     {lateral:,} ({lateral / total:.1%})")
        print(f"    - I/O level:            {io_edges:,} ({io_edges / total:.1%})")
        print(f"    - Non-I/O lateral:      {non_io_lateral:,} ({non_io_lateral / total:.1%})")

        if topdown / total < 0.05:
            print(f"  >> TOP-DOWN EDGES < 5% — predictions may be starved of bandwidth")

        return {
            "total": total,
            "topdown": topdown,
            "bottomup": bottomup,
            "lateral": lateral,
            "io_edges": io_edges,
            "topdown_pct": topdown / total,
        }

    def _assess_1c(self, r):
        pct = r.get("topdown_pct", 0)
        if pct >= 0.05:
            return "PASS", f"Top-down edges = {pct:.1%} of total (sufficient bandwidth)"
        elif pct >= 0.02:
            return "WARN", f"Top-down edges = {pct:.1%} (may be low)"
        else:
            return "FAIL", f"Top-down edges = {pct:.1%} (starved, <2%)"

    # ─────────────────────────────────────────────────────────────────────
    # TEST 2: Gradient alignment by edge type
    # ─────────────────────────────────────────────────────────────────────

    def test_2_gradient_alignment(self, data_path="ndcd/data/level1_chars.txt") -> Dict:
        """Check whether gradients are meaningful for each edge type.

        Top-down should have coherent signal (|mean/std| > 0.1).
        Others should be noise (|mean/std| ~ 0) if they're irrelevant.
        If all three look like noise, the error signal isn't informative.
        """
        # Save state
        saved_state = self.engine.state.clone()
        
        # Get one sample to calculate a real EqProp gradient
        if not os.path.exists(data_path):
            input_byte, target_byte = 0, 1
        else:
            with open(data_path, 'rb') as f:
                data = f.read(2)
                input_byte, target_byte = data[0], data[1]

        input_vec = torch.zeros(self.engine.num_nodes, device=self.device)
        input_vec[0:256] = self.trainer.eye[input_byte] * 1.0
        input_mask = torch.zeros(self.engine.num_nodes, device=self.device)
        input_mask[list(range(256))] = 1.0

        # Free phase
        self.engine.settle(input_vec, input_mask=input_mask, max_steps=50)
        rho_free = torch.tanh(self.engine.state.clone())

        # Nudge phase
        beta = 0.5
        nudge_vec = input_vec.clone()
        nudge_vec[256:512] = self.trainer.eye[target_byte] * beta
        self.engine.settle(nudge_vec, input_mask=input_mask, max_steps=50)
        rho_nudge = torch.tanh(self.engine.state.clone())

        idx_i = self.engine.indices[0]
        idx_j = self.engine.indices[1]
        
        # Top-down uses EqProp gradient
        td_grad = (rho_nudge[idx_i] * rho_nudge[idx_j] - rho_free[idx_i] * rho_free[idx_j]) / beta

        src_levels = self.engine.node_to_level[idx_i]
        dst_levels = self.engine.node_to_level[idx_j]

        results = {}
        for name, mask in [
            ("top-down", src_levels > dst_levels),
            ("bottom-up", src_levels < dst_levels),
            ("lateral", src_levels == dst_levels),
        ]:
            if name == "top-down":
                g = td_grad[mask]
            else:
                # BU/Lat use associative rules, but we test spatial cross-talk here.
                # Use standard spatial_errors * rho_free just to see if it leaked.
                self.engine.state = torch.atanh(rho_free.clamp(-0.99, 0.99))
                self.engine.compute_prediction_errors()
                g = (self.engine.spatial_errors[idx_j] * rho_free[idx_i])[mask]
            if len(g) == 0:
                results[name] = {"mean": 0, "std": 0, "snr": 0}
                continue
            mean = g.mean().item()
            std = g.std().item()
            snr = abs(mean / std) if std > 1e-10 else 0.0
            results[name] = {"mean": mean, "std": std, "snr": snr}
            print(f"  {name:>10}: mean_grad={mean:.6f}, std={std:.6f}, |mean/std|={snr:.4f}")

        td_snr = results.get("top-down", {}).get("snr", 0)
        bu_snr = results.get("bottom-up", {}).get("snr", 0)
        lat_snr = results.get("lateral", {}).get("snr", 0)

        if td_snr < 0.01:
            print(f"  >> TOP-DOWN GRADIENT IS NOISE — spatial error signal is uninformative")
        else:
            print(f"  >> Top-down gradient coherent. Non-top-down edges do not use spatial error.")

        self.engine.state = saved_state
        return results

    def _assess_2(self, r):
        td = r.get("top-down", {}).get("snr", 0)
        if td > 0.01:
            return "PASS", f"Top-down gradient coherent (SNR={td:.3f})"
        else:
            return "FAIL", f"Top-down gradient is noise (SNR={td:.4f})"

    # ─────────────────────────────────────────────────────────────────────
    # TEST 3A: Cascade flow dynamics
    # ─────────────────────────────────────────────────────────────────────

    def test_3a_cascade_dynamics(self, data_path="ndcd/data/level1_chars.txt",
                                  num_steps=500) -> Dict:
        """Track how fast weight magnitude flows through cascade levels.

        If surface never accumulates above 0.01, learning drains to deep
        before useful transient representations form. If deep grows
        monotonically regardless of task, it's accumulating noise.
        """
        if not os.path.exists(data_path):
            print(f"  Data not found at {data_path}, using synthetic steps")
            data = bytes(list(range(256)) * 20)
        else:
            with open(data_path, 'rb') as f:
                data = f.read()

        snapshots = {'surface': [], 'mid': [], 'deep': [], 'step': []}

        # Save state
        saved_surface = self.engine.w_surface.clone()
        saved_mid = self.engine.w_mid.clone()
        saved_deep = self.engine.w_deep.clone()
        saved_state = self.engine.state.clone()

        for step in range(num_steps):
            idx = step % (len(data) - 1)
            input_byte = data[idx]
            target_byte = data[idx + 1]

            input_vec = torch.zeros(self.num_nodes, device=self.device)
            input_vec[0:256] = self.trainer.eye[input_byte] * 1.0
            input_mask = torch.zeros(self.num_nodes, device=self.device)
            input_mask[list(range(256))] = 1.0

            self.engine.store_previous_state()
            self.engine.settle(input_vec, input_mask=input_mask, max_steps=50)
            free_state = self.engine.state.clone()
            self.engine.compute_prediction_errors()

            # EqProp Nudge Phase
            beta = 0.5
            output_target = self.trainer.eye[target_byte]
            nudge_vec = input_vec.clone()
            nudge_vec[256:512] = output_target * beta
            
            self.engine.settle(nudge_vec, input_mask=input_mask, max_steps=50)
            nudge_state = self.engine.state.clone()

            self.engine.update_weights_predictive(free_state, nudge_state, beta=beta, learning_rate=0.05)
            if step >= 50:
                self.engine.cascade_transfer()

            if step % 50 == 0:
                snapshots['surface'].append(self.engine.w_surface.abs().mean().item())
                snapshots['mid'].append(self.engine.w_mid.abs().mean().item())
                snapshots['deep'].append(self.engine.w_deep.abs().mean().item())
                snapshots['step'].append(step)

        # Report
        print(f"  Cascade dynamics over {num_steps} steps:")
        for i, step in enumerate(snapshots['step']):
            if i % 2 == 0 or i == len(snapshots['step']) - 1:
                print(f"    Step {step:>4d}: surface={snapshots['surface'][i]:.6f}, "
                      f"mid={snapshots['mid'][i]:.6f}, deep={snapshots['deep'][i]:.6f}")

        surface_max = max(snapshots['surface'])
        deep_growth = snapshots['deep'][-1] - snapshots['deep'][0]
        print(f"  Surface peak: {surface_max:.6f}")
        print(f"  Deep growth:  {deep_growth:+.6f}")

        if surface_max < 0.01:
            print(f"  >> SURFACE NEVER ACCUMULATES — learning drains to deep too fast")

        # Restore state
        self.engine.w_surface = saved_surface
        self.engine.w_mid = saved_mid
        self.engine.w_deep = saved_deep
        self.engine.state = saved_state

        return {
            "snapshots": snapshots,
            "surface_peak": surface_max,
            "deep_growth": deep_growth,
        }

    def _assess_3a(self, r):
        peak = r.get("surface_peak", 0)
        growth = r.get("deep_growth", 0)
        if peak > 0.01:
            return "PASS", f"Surface accumulates (peak={peak:.4f}), deep growth={growth:+.4f}"
        else:
            return "FAIL", f"Surface peak={peak:.6f} (drains too fast), deep growth={growth:+.4f}"

    # ─────────────────────────────────────────────────────────────────────
    # TEST 3B: Diversity regularization impact on w_deep
    # ─────────────────────────────────────────────────────────────────────

    def test_3b_deep_stability(self) -> Dict:
        """Measure how much the diversity reg changes w_deep per step.

        If change/magnitude ratio > 0.01, the reg is significantly eroding
        deep weights every step.
        """
        deep_before = self.engine.w_deep.clone()

        # Run one weight update (zero gradient to isolate diversity reg)
        self.engine.update_weights_predictive(
            free_state=self.engine.state,
            nudge_state=self.engine.state,
            beta=0.5,
            learning_rate=0.05
        )

        deep_after = self.engine.w_deep.clone()
        delta = (deep_after - deep_before).abs()

        mean_delta = delta.mean().item()
        max_delta = delta.max().item()
        deep_mag = self.engine.w_deep.abs().mean().item()
        ratio = mean_delta / max(deep_mag, 1e-8)

        print(f"  w_deep change from one update step:")
        print(f"    Mean |Δ|: {mean_delta:.6f}")
        print(f"    Max  |Δ|: {max_delta:.6f}")
        print(f"    |w_deep| mean: {deep_mag:.4f}")
        print(f"    Change/magnitude ratio: {ratio:.4f}")

        if ratio > 0.01:
            print(f"  >> DIVERSITY REG ERODING DEEP WEIGHTS ({ratio:.3f} per step)")

        # Restore
        self.engine.w_deep = deep_before

        return {
            "mean_delta": mean_delta,
            "max_delta": max_delta,
            "deep_magnitude": deep_mag,
            "ratio": ratio,
        }

    def _assess_3b(self, r):
        ratio = r.get("ratio", 0)
        if ratio < 0.001:
            return "PASS", f"Deep weights stable (change/mag ratio={ratio:.5f})"
        elif ratio < 0.01:
            return "WARN", f"Deep weights slightly eroded (ratio={ratio:.4f})"
        else:
            return "FAIL", f"Deep weights being eroded (ratio={ratio:.3f} per step)"

    # ─────────────────────────────────────────────────────────────────────
    # TEST 4A: Attractor sparsity during replay
    # ─────────────────────────────────────────────────────────────────────

    def test_4a_replay_sparsity(self, thresholds=None) -> Dict:
        """How sparse are replay attractors at different activation thresholds?

        A healthy attractor should have <30% of nodes strongly active.
        If >80% are active at threshold 0.5, attractors are not sparse.
        """
        if thresholds is None:
            thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]

        saved_state = self.engine.state.clone()

        # Settle from noise without input
        noise = torch.randn(self.num_nodes, device=self.device) * 0.1
        self.engine.state = noise
        input_vec = torch.zeros(self.num_nodes, device=self.device)
        self.engine.settle(input_vec, max_steps=50)
        rho = torch.tanh(self.engine.state)

        results = {}
        print(f"  Attractor sparsity (settled from noise, no input):")
        for t in thresholds:
            active = (rho.abs() > t).sum().item()
            frac = active / self.num_nodes
            results[f"thresh_{t}"] = {"active": active, "fraction": frac}
            print(f"    |tanh(x)| > {t}: {active}/{self.num_nodes} ({frac:.1%})")

        frac_05 = results.get("thresh_0.5", {}).get("fraction", 1.0)
        if frac_05 > 0.8:
            print(f"  >> ATTRACTORS NOT SPARSE — {frac_05:.0%} active at threshold 0.5")
            print(f"     Sleep replay Hebbian update will strengthen everything uniformly")

        self.engine.state = saved_state

        return results

    def _assess_4a(self, r):
        frac = r.get("thresh_0.5", {}).get("fraction", 1.0)
        if frac < 0.3:
            return "PASS", f"Attractors are sparse ({frac:.0%} active at |tanh|>0.5)"
        elif frac < 0.6:
            return "WARN", f"Moderate sparsity ({frac:.0%} active at |tanh|>0.5)"
        else:
            return "FAIL", f"Attractors not sparse ({frac:.0%} active, should be <30%)"

    # ─────────────────────────────────────────────────────────────────────
    # TEST 4B: Replay pattern diversity
    # ─────────────────────────────────────────────────────────────────────

    def test_4b_replay_diversity(self, num_replays=20) -> Dict:
        """Do different noise initializations converge to different attractors?

        If mean similarity > 0.9, network has essentially one attractor.
        Healthy: mean 0.3–0.6 with std > 0.1 (diverse attractors).
        """
        saved_state = self.engine.state.clone()
        patterns = []

        for i in range(num_replays):
            noise = torch.randn(self.num_nodes, device=self.device) * 0.1
            self.engine.state = noise
            input_vec = torch.zeros(self.num_nodes, device=self.device)
            self.engine.settle(input_vec, max_steps=50)
            patterns.append(torch.tanh(self.engine.state).clone())

        # Pairwise cosine similarity
        sims = []
        for i in range(len(patterns)):
            for j in range(i + 1, len(patterns)):
                sim = torch.cosine_similarity(patterns[i], patterns[j], dim=0).item()
                sims.append(sim)

        mean_sim = np.mean(sims)
        std_sim = np.std(sims)
        min_sim = np.min(sims)
        max_sim = np.max(sims)

        print(f"  Replay pattern similarity ({num_replays} random inits):")
        print(f"    Mean:  {mean_sim:.4f}")
        print(f"    Std:   {std_sim:.4f}")
        print(f"    Range: [{min_sim:.4f}, {max_sim:.4f}]")

        if mean_sim > 0.9:
            print(f"  >> NETWORK HAS ~1 ATTRACTOR — all replays converge to same state")
        elif mean_sim > 0.6:
            print(f"  >> LOW ATTRACTOR DIVERSITY — few distinct memories stored")
        else:
            print(f"  >> Good attractor diversity")

        self.engine.state = saved_state

        return {
            "mean_similarity": mean_sim,
            "std_similarity": std_sim,
            "min_similarity": min_sim,
            "max_similarity": max_sim,
        }

    def _assess_4b(self, r):
        mean_sim = r.get("mean_similarity", 1.0)
        std = r.get("std_similarity", 0.0)
        if mean_sim < 0.6 and std > 0.1:
            return "PASS", f"Diverse attractors (mean_sim={mean_sim:.3f}, std={std:.3f})"
        elif mean_sim < 0.8:
            return "WARN", f"Moderate diversity (mean_sim={mean_sim:.3f}, std={std:.3f})"
        else:
            return "FAIL", f"Near-single attractor (mean_sim={mean_sim:.3f})"

    # ─────────────────────────────────────────────────────────────────────
    # TEST 5A: State distribution (saturation check)
    # ─────────────────────────────────────────────────────────────────────

    def test_5a_state_distribution(self) -> Dict:
        """Check whether states are saturated near clamp limits.

        If >50% near clamp (|s|>1.4), the network is saturated.
        tanh(1.4) = 0.885, tanh'(1.4) = 0.035 — almost zero gradient.
        """
        s = self.engine.state.detach()
        total = len(s)

        near_clamp = (s.abs() > 1.4).sum().item() / total
        in_linear = (s.abs() < 0.5).sum().item() / total
        moderate = ((s.abs() >= 0.5) & (s.abs() <= 1.4)).sum().item() / total

        print(f"  State distribution ({total} nodes):")
        print(f"    Near clamp (|s|>1.4):      {near_clamp:.1%} (tanh' < 0.035)")
        print(f"    Moderate (0.5<|s|<1.4):    {moderate:.1%}")
        print(f"    Linear region (|s|<0.5):   {in_linear:.1%} (tanh' > 0.79)")

        # Histogram
        edges = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
        print(f"  State histogram:")
        for edge in edges:
            lo, hi = edge - 0.25, edge + 0.25
            count = ((s > lo) & (s <= hi)).sum().item()
            bar = "#" * int(count / total * 100)
            print(f"    [{lo:+5.2f}, {hi:+5.2f}]: {count:>5d} ({count / total:5.1%}) {bar}")

        if near_clamp > 0.5:
            print(f"  >> STATE SATURATED — {near_clamp:.0%} nodes near clamp limits")
            print(f"     Gradients are near-zero for half the network")

        return {
            "near_clamp": near_clamp,
            "in_linear": in_linear,
            "moderate": moderate,
        }

    def _assess_5a(self, r):
        near_clamp = r.get("near_clamp", 0)
        in_linear = r.get("in_linear", 0)
        if near_clamp < 0.2:
            return "PASS", f"State not saturated ({near_clamp:.0%} near clamp, {in_linear:.0%} linear)"
        elif near_clamp < 0.5:
            return "WARN", f"Partial saturation ({near_clamp:.0%} near clamp)"
        else:
            return "FAIL", f"State saturated ({near_clamp:.0%} near clamp, gradients dead)"

    # ─────────────────────────────────────────────────────────────────────
    # TEST 5B: Input sensitivity
    # ─────────────────────────────────────────────────────────────────────

    def test_5b_input_sensitivity(self, test_bytes=None) -> Dict:
        """Measure how much the settled state differs across different inputs.

        If cos_sim > 0.99 for different inputs, the network converges to the
        same attractor regardless of input. Higher levels should show MORE
        similarity (abstract), lower LESS.
        """
        if test_bytes is None:
            test_bytes = [ord('a'), ord('z'), ord(' '), ord('e')]

        saved_state = self.engine.state.clone()

        states = {}
        for b in test_bytes:
            self.engine.state.zero_()
            input_vec = torch.zeros(self.num_nodes, device=self.device)
            input_vec[0:256] = self.trainer.eye[b] * 1.0
            input_mask = torch.zeros(self.num_nodes, device=self.device)
            input_mask[list(range(256))] = 1.0
            self.engine.settle(input_vec, input_mask=input_mask, max_steps=50)
            states[b] = self.engine.state.clone()

        comparisons = []
        for i, b1 in enumerate(test_bytes):
            for b2 in test_bytes[i + 1:]:
                diff = (states[b1] - states[b2]).norm().item()
                cos = torch.cosine_similarity(states[b1], states[b2], dim=0).item()

                print(f"  '{chr(b1)}' vs '{chr(b2)}': L2_diff={diff:.4f}, cos_sim={cos:.4f}")

                # Per-level breakdown
                level_sims = {}
                for level in range(self.engine.max_level + 1):
                    mask = self.engine.node_to_level == level
                    if mask.sum() < 2:
                        continue
                    s1 = states[b1][mask]
                    s2 = states[b2][mask]
                    lvl_cos = torch.cosine_similarity(s1, s2, dim=0).item()
                    level_sims[level] = lvl_cos
                    print(f"    L{level} cos_sim: {lvl_cos:.4f}")

                comparisons.append({
                    "pair": (chr(b1), chr(b2)),
                    "l2_diff": diff,
                    "cos_sim": cos,
                    "level_sims": level_sims,
                })

        avg_cos = np.mean([c["cos_sim"] for c in comparisons])
        if avg_cos > 0.99:
            print(f"  >> NETWORK CONVERGES TO SAME STATE regardless of input (avg cos={avg_cos:.4f})")
        elif avg_cos > 0.95:
            print(f"  >> Low input sensitivity (avg cos={avg_cos:.4f})")

        self.engine.state = saved_state

        return {
            "comparisons": comparisons,
            "avg_cosine_similarity": avg_cos,
        }

    def _assess_5b(self, r):
        avg_cos = r.get("avg_cosine_similarity", 1.0)
        if avg_cos < 0.9:
            return "PASS", f"Input-sensitive (avg cos_sim={avg_cos:.3f})"
        elif avg_cos < 0.95:
            return "WARN", f"Low input sensitivity (avg cos_sim={avg_cos:.3f})"
        else:
            return "FAIL", f"Input-insensitive (avg cos_sim={avg_cos:.3f}), same attractor for all inputs"

    # ─────────────────────────────────────────────────────────────────────
    # TEST 6: Temporal prediction usefulness
    # ─────────────────────────────────────────────────────────────────────

    def test_6_temporal_contribution(self) -> Dict:
        """Is temporal prediction doing anything beyond predicting persistence?

        If A ~ 0.9 +/- 0.05 everywhere, temporal prediction hasn't learned
        anything beyond its initialization. The diagonal approximation can't
        capture cross-node temporal dependencies.
        """
        results = {}
        print(f"  Temporal transition matrices A_l (diagonal approximation):")
        for mod_idx, (start, end) in enumerate(self.engine.module_ranges):
            a = self.engine.temporal_A[mod_idx]
            level = self.engine.module_levels[mod_idx]
            a_mean = a.mean().item()
            a_std = a.std().item()
            a_min = a.min().item()
            a_max = a.max().item()

            # Check deviation from initialization (0.9)
            deviation = (a - 0.9).abs().mean().item()

            results[mod_idx] = {
                "level": int(level),
                "mean": a_mean,
                "std": a_std,
                "min": a_min,
                "max": a_max,
                "deviation_from_init": deviation,
            }

        # Aggregate by level
        level_stats = {}
        for mod_idx, data in results.items():
            level = data["level"]
            if level not in level_stats:
                level_stats[level] = {"means": [], "stds": [], "deviations": []}
            level_stats[level]["means"].append(data["mean"])
            level_stats[level]["stds"].append(data["std"])
            level_stats[level]["deviations"].append(data["deviation_from_init"])

        for level in sorted(level_stats.keys()):
            stats = level_stats[level]
            avg_mean = np.mean(stats["means"])
            avg_std = np.mean(stats["stds"])
            avg_dev = np.mean(stats["deviations"])
            print(f"  Level {level} ({len(stats['means'])} modules): "
                  f"A_mean={avg_mean:.4f}, A_std={avg_std:.4f}, "
                  f"deviation_from_0.9={avg_dev:.4f}")

        all_devs = [d["deviation_from_init"] for d in results.values()]
        avg_deviation = np.mean(all_devs)

        if avg_deviation < 0.05:
            print(f"  >> TEMPORAL A NEAR INITIALIZATION (avg dev={avg_deviation:.4f})")
            print(f"     Temporal prediction hasn't learned beyond 'predict persistence'")
            print(f"     Diagonal approximation may be inherently too limited")

        return {
            "per_module": results,
            "level_stats": level_stats,
            "avg_deviation_from_init": avg_deviation,
        }

    def _assess_6(self, r):
        dev = r.get("avg_deviation_from_init", 0)
        if dev > 0.1:
            return "PASS", f"Temporal A learned (avg deviation from init = {dev:.3f})"
        elif dev > 0.05:
            return "WARN", f"Temporal A barely changed (deviation = {dev:.3f})"
        else:
            return "FAIL", f"Temporal A at initialization (deviation = {dev:.4f}), not learning"

    # ─────────────────────────────────────────────────────────────────────
    # TEST 7: CLS hippocampal vs neocortical dynamics
    # ─────────────────────────────────────────────────────────────────────

    def test_7_cls_dynamics(self) -> Dict:
        """Compare weight evolution in hippocampal vs neocortical modules.

        Ratio should be ~3-10x for healthy fast learning.
        If hippo surface ~ 0 or wildly larger than neo, the CLS dynamics are broken.
        """
        hippo_mask = self.trainer.hippo_edge_mask

        hippo_surface = self.engine.w_surface[hippo_mask].abs().mean().item()
        neo_surface = self.engine.w_surface[~hippo_mask].abs().mean().item()
        hippo_mid = self.engine.w_mid[hippo_mask].abs().mean().item()
        neo_mid = self.engine.w_mid[~hippo_mask].abs().mean().item()
        hippo_deep = self.engine.w_deep[hippo_mask].abs().mean().item()
        neo_deep = self.engine.w_deep[~hippo_mask].abs().mean().item()

        surface_ratio = hippo_surface / max(neo_surface, 1e-8)
        mid_ratio = hippo_mid / max(neo_mid, 1e-8)
        deep_ratio = hippo_deep / max(neo_deep, 1e-8)

        print(f"  Hippocampal edges: {hippo_mask.sum().item():,}")
        print(f"  Neocortical edges: {(~hippo_mask).sum().item():,}")
        print(f"  Surface: hippo={hippo_surface:.6f}, neo={neo_surface:.6f}, ratio={surface_ratio:.2f}x")
        print(f"  Mid:     hippo={hippo_mid:.6f}, neo={neo_mid:.6f}, ratio={mid_ratio:.2f}x")
        print(f"  Deep:    hippo={hippo_deep:.6f}, neo={neo_deep:.6f}, ratio={deep_ratio:.2f}x")

        if hippo_surface < 1e-6 and neo_surface < 1e-6:
            print(f"  >> Both hippo and neo surface ~ 0 (before training or after reset)")
        elif surface_ratio > 50:
            print(f"  >> HIPPO SURFACE WILDLY LARGER ({surface_ratio:.0f}x) — CLS boost may be unstable")
        elif surface_ratio < 1.5 and hippo_surface > 1e-4:
            print(f"  >> HIPPO NOT DIFFERENTIATED — 10x boost not creating fast learning")

        return {
            "hippo_surface": hippo_surface,
            "neo_surface": neo_surface,
            "hippo_deep": hippo_deep,
            "neo_deep": neo_deep,
            "surface_ratio": surface_ratio,
            "deep_ratio": deep_ratio,
        }

    def _assess_7(self, r):
        ratio = r.get("surface_ratio", 0)
        hippo_s = r.get("hippo_surface", 0)
        neo_s = r.get("neo_surface", 0)
        if hippo_s < 1e-6 and neo_s < 1e-6:
            return "WARN", "Both surface ~0 (likely before training or after reset)"
        elif 1.5 <= ratio <= 50:
            return "PASS", f"CLS differentiated (hippo/neo surface ratio = {ratio:.1f}x)"
        elif ratio > 50:
            return "FAIL", f"CLS unstable (ratio = {ratio:.0f}x, too large)"
        else:
            return "WARN", f"CLS not differentiated (ratio = {ratio:.2f}x, expected 3-10x)"

    # ─────────────────────────────────────────────────────────────────────
    # TEST 8: Retention with re-trained readout
    # ─────────────────────────────────────────────────────────────────────

    def test_8_true_retention(self, data_path="ndcd/data/level1_chars.txt",
                               num_train=2000, num_test=1000) -> Dict:
        """Train a fresh readout on old data, THEN measure accuracy.

        This isolates reservoir quality from readout alignment. If accuracy is
        still low with a re-trained readout, the reservoir truly lost the info.
        If accuracy is high, only the readout was miscalibrated.
        """
        if not os.path.exists(data_path):
            print(f"  Data not found at {data_path}, skipping")
            return {"error": "data_not_found"}

        with open(data_path, 'rb') as f:
            data = f.read()

        data_len = len(data)
        num_train = min(num_train, data_len - num_test - 1)
        num_test = min(num_test, data_len - num_train - 1)

        if num_train < 100 or num_test < 50:
            print(f"  Insufficient data for test (need > 150 bytes)")
            return {"error": "insufficient_data"}

        saved_state = self.engine.state.clone()

        # Get feature dimensions
        n_reservoir = len(self.trainer.level0_indices)
        n_features = n_reservoir

        # Initialize temporary readout
        tmp_W = torch.randn(256, n_features, device=self.device) * (1.0 / np.sqrt(n_features))
        tmp_b = torch.zeros(256, device=self.device)

        # Train temporary readout
        print(f"  Training fresh readout ({num_train} steps)...")
        for i in range(num_train):
            idx = i % (data_len - 1)
            input_byte = data[idx]
            target_byte = data[idx + 1]

            input_vec = torch.zeros(self.num_nodes, device=self.device)
            input_vec[0:256] = self.trainer.eye[input_byte] * 1.0
            input_mask = torch.zeros(self.num_nodes, device=self.device)
            input_mask[list(range(256))] = 1.0
            self.engine.settle(input_vec, input_mask=input_mask, max_steps=50)

            with torch.no_grad():
                level0_acts = torch.tanh(self.engine.state[self.trainer.level0_indices])
                features = level0_acts
                logits = tmp_W @ features + tmp_b
                probs = torch.softmax(logits, dim=0)
                target_one_hot = self.trainer.eye[target_byte]
                grad = probs - target_one_hot
                tmp_W -= 0.05 * torch.outer(grad, features)
                tmp_b -= 0.05 * grad

        # Evaluate with re-trained readout
        correct_retrained = 0
        correct_original = 0
        for i in range(num_test):
            idx = (num_train + i) % (data_len - 1)
            input_byte = data[idx]
            target_byte = data[idx + 1]

            input_vec = torch.zeros(self.num_nodes, device=self.device)
            input_vec[0:256] = self.trainer.eye[input_byte] * 1.0
            input_mask = torch.zeros(self.num_nodes, device=self.device)
            input_mask[list(range(256))] = 1.0
            self.engine.settle(input_vec, input_mask=input_mask, max_steps=50)

            with torch.no_grad():
                level0_acts = torch.tanh(self.engine.state[self.trainer.level0_indices])
                features = level0_acts

                # Re-trained readout
                logits_new = tmp_W @ features + tmp_b
                if torch.argmax(logits_new).item() == target_byte:
                    correct_retrained += 1

                # Original readout
                logits_orig = self.trainer.readout_W @ features + self.trainer.readout_b
                if torch.argmax(logits_orig).item() == target_byte:
                    correct_original += 1

        self.engine.state = saved_state

        retrained_acc = correct_retrained / num_test
        original_acc = correct_original / num_test
        chance = 1.0 / 256

        print(f"  Re-trained readout accuracy:  {retrained_acc:.2%}")
        print(f"  Original readout accuracy:    {original_acc:.2%}")
        print(f"  Chance:                       {chance:.2%}")

        if retrained_acc > original_acc * 2 and retrained_acc > chance * 3:
            print(f"  >> RESERVOIR RETAINS INFO but current readout is miscalibrated")
            print(f"     Phase boundary reset destroyed readout, not reservoir")
        elif retrained_acc < chance * 2:
            print(f"  >> RESERVOIR HAS LOST THE INFORMATION — not just readout issue")

        return {
            "retrained_acc": retrained_acc,
            "original_acc": original_acc,
            "chance": chance,
        }

    def _assess_8(self, r):
        if "error" in r:
            return "WARN", f"Skipped ({r['error']})"
        retrained = r.get("retrained_acc", 0)
        original = r.get("original_acc", 0)
        chance = r.get("chance", 1 / 256)
        if retrained > chance * 5:
            return "PASS", f"Reservoir retains info (retrained={retrained:.1%}, original={original:.1%})"
        elif retrained > chance * 2:
            return "WARN", f"Weak retention (retrained={retrained:.1%})"
        else:
            return "FAIL", f"Reservoir lost info (retrained={retrained:.1%}, ~chance)"

    # ─────────────────────────────────────────────────────────────────────
    # TEST 9: Runtime spectral radius
    # ─────────────────────────────────────────────────────────────────────

    def test_9_runtime_spectral_radius(self) -> Dict:
        """Check actual spectral radius of effective weights.

        If this is >10, input signal is negligible compared to recurrence.
        The network is operating as a fixed-point attractor, not a driven system.
        """
        import scipy.sparse as sp
        from scipy.sparse.linalg import eigs

        w = self.engine.effective_weights.cpu().numpy()
        idx = self.engine.indices.cpu().numpy()
        n = self.engine.num_nodes
        
        # Filter for only recurrent (free) edges
        # Input projection edges (source < 256) are clamped and scaled 15x
        # to drive one-hot inputs, so they would artificially inflate SR.
        free_mask = idx[0] >= 512
        w_free = w[free_mask]
        idx_free = idx[:, free_mask]

        W = sp.csr_matrix((w_free, (idx_free[0], idx_free[1])), shape=(n, n))

        try:
            eigvals = eigs(W, k=1, which='LM', return_eigenvectors=False)
            sr = np.abs(eigvals[0])
        except Exception as e:
            print(f"  Eigenvalue computation failed: {e}")
            # Fallback: Frobenius norm upper bound
            sr = sp.linalg.norm(W, 'fro') / np.sqrt(n)
            print(f"  Using Frobenius estimate: {sr:.2f}")

        # Also compute with STP modulation
        w_stp = (self.engine.effective_weights *
                 self.engine.facilitation *
                 self.engine.depression).cpu().numpy()[free_mask]
        W_stp = sp.csr_matrix((w_stp, (idx_free[0], idx_free[1])), shape=(n, n))
        try:
            eigvals_stp = eigs(W_stp, k=1, which='LM', return_eigenvectors=False)
            sr_stp = np.abs(eigvals_stp[0])
        except Exception:
            sr_stp = sp.linalg.norm(W_stp, 'fro') / np.sqrt(n)

        print(f"  Effective spectral radius (cascade sum): {sr:.2f}")
        print(f"  With STP modulation:                     {sr_stp:.2f}")
        print(f"  Initial target was:                      0.80")

        if sr > 100:
            print(f"  >> SPECTRAL RADIUS >> 100 — recurrence completely dominates input")
            print(f"     Network is a fixed-point attractor, not a responsive system")
        elif sr > 10:
            print(f"  >> SPECTRAL RADIUS > 10 — input signal significantly diluted")
        elif sr < 0.5:
            print(f"  >> SPECTRAL RADIUS < 0.5 — network is too damped, no recurrence")

        return {
            "spectral_radius": sr,
            "spectral_radius_stp": sr_stp,
        }

    def _assess_9(self, r):
        sr = r.get("spectral_radius", 0)
        if 0.5 <= sr <= 10:
            return "PASS", f"Spectral radius = {sr:.2f} (reasonable range)"
        elif sr < 0.5:
            return "WARN", f"Spectral radius = {sr:.2f} (too damped)"
        elif sr <= 50:
            return "WARN", f"Spectral radius = {sr:.2f} (high, input diluted)"
        else:
            return "FAIL", f"Spectral radius = {sr:.2f} (unbounded, input negligible)"

    # ─────────────────────────────────────────────────────────────────────
    # TEST 10: IMEX settle convergence
    # ─────────────────────────────────────────────────────────────────────

    def test_10_settle_convergence(self, test_inputs=None) -> Dict:
        """Track per-step state change during settling to verify convergence.

        If diff never drops below tolerance, the network isn't settling —
        it's oscillating or being clamped every step.
        """
        if test_inputs is None:
            test_inputs = [ord('a'), ord(' '), ord('z')]

        saved_state = self.engine.state.clone()

        results = {}
        for input_byte in test_inputs:
            self.engine.state.zero_()

            input_vec = torch.zeros(self.num_nodes, device=self.device)
            input_vec[0:256] = self.trainer.eye[input_byte] * 1.0
            input_mask = torch.zeros(self.num_nodes, device=self.device)
            input_mask[list(range(256))] = 1.0

            # Manually run IMEX steps to track convergence
            max_steps = 50
            diffs = []

            # Build effective weights with STP
            cascade_weights = self.engine.effective_weights
            effective_w = cascade_weights * self.engine.facilitation * self.engine.depression

            # Build sparse matrix
            W = torch.sparse_coo_tensor(
                self.engine.indices, effective_w,
                (self.num_nodes, self.num_nodes))
            imex_denom = 1.0 + self.engine.dt / self.engine.taus

            current_s = self.engine.state.clone()

            for step in range(max_steps):
                # Hard clamp input
                current_s = current_s * (1.0 - input_mask) + input_vec * input_mask

                old = current_s.clone()

                rho = torch.tanh(current_s)
                synaptic = torch.mv(W, rho)
                nonlinear = synaptic + self.engine.biases + input_vec
                current_s = (current_s + self.engine.dt * nonlinear / self.engine.taus) / imex_denom
                current_s = current_s.clamp(-1.5, 1.5)

                # Hard clamp after
                current_s = current_s * (1.0 - input_mask) + input_vec * input_mask

                diff = (current_s - old).norm().item()
                diffs.append(diff)

            char = chr(input_byte)
            converged = diffs[-1] < 1e-3
            print(f"  Input '{char}': {diffs[0]:.4f} → {diffs[-1]:.4f} "
                  f"(converged={converged}, min_diff={min(diffs):.4f})")
            print(f"    Steps 0-4:  {[f'{d:.3f}' for d in diffs[:5]]}")
            print(f"    Steps 45-49: {[f'{d:.3f}' for d in diffs[45:50]]}")

            results[char] = {
                "diffs": diffs,
                "initial_diff": diffs[0],
                "final_diff": diffs[-1],
                "min_diff": min(diffs),
                "converged": converged,
            }

        any_converged = any(r["converged"] for r in results.values())
        avg_final = np.mean([r["final_diff"] for r in results.values()])

        if not any_converged:
            print(f"  >> NO INPUT CONVERGED — network may be oscillating or diverging")
            print(f"     avg final diff = {avg_final:.4f} (tol = 1e-3)")

        self.engine.state = saved_state

        return {
            "per_input": results,
            "any_converged": any_converged,
            "avg_final_diff": avg_final,
        }

    def _assess_10(self, r):
        converged = r.get("any_converged", False)
        avg_final = r.get("avg_final_diff", float('inf'))
        if converged and avg_final < 0.01:
            return "PASS", f"IMEX converges (avg final diff = {avg_final:.4f})"
        elif avg_final < 0.1:
            return "WARN", f"IMEX partially converges (avg final diff = {avg_final:.4f})"
        else:
            return "FAIL", f"IMEX not converging (avg final diff = {avg_final:.4f})"


# ═════════════════════════════════════════════════════════════════════════
# Standalone runner
# ═════════════════════════════════════════════════════════════════════════

def run_diagnostics_standalone(args):
    """Initialize a trainer, optionally train, then run diagnostics."""
    from ndcd.run_sequential import SequentialTrainer

    device = args.device
    if torch.backends.mps.is_available() and device == 'cpu':
        device = 'mps'
    if torch.cuda.is_available() and device == 'cpu':
        device = 'cuda'
    print(f"Using device: {device}")

    trainer = SequentialTrainer(
        num_nodes=args.nodes, device=device, num_modules=args.modules)

    if args.train_first:
        print(f"\nRunning Full Curriculum ({args.train_iters} steps per phase) before diagnostics...")
        trainer.engine.state.zero_()
        
        trainer.train_phase("Chars", "ndcd/data/level1_chars.txt",
                            iterations=args.train_iters, steps_per_iter=100,
                            lr=0.05, settle_steps=8, input_gain=5.0)
                            
        trainer.train_phase("Words", "ndcd/data/level2_words.txt",
                            iterations=args.train_iters, steps_per_iter=100,
                            lr=0.05, settle_steps=8, input_gain=5.0)
                            
        trainer.train_phase("Quotes", "ndcd/data/level3_quotes.txt",
                            iterations=args.train_iters, steps_per_iter=100,
                            lr=0.05, settle_steps=8, input_gain=5.0)

    diag = ModelDiagnostics(trainer)
    results = diag.run_all()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run diagnostic tests on Living-Brain model")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use (cpu/cuda/mps)")
    parser.add_argument("--nodes", type=int, default=2000,
                        help="Number of nodes in the model")
    parser.add_argument("--modules", type=int, default=20,
                        help="Number of modules")
    parser.add_argument("--train-first", action="store_true",
                        help="Train on Chars before running diagnostics")
    parser.add_argument("--train-iters", type=int, default=100,
                        help="Number of training iterations if --train-first")
    args = parser.parse_args()

    run_diagnostics_standalone(args)


if __name__ == "__main__":
    main()
