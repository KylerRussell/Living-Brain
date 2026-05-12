"""
B24 production sweep.

Runs ONLY benchmark_24_interval_timing across a grid of MICRO_LR_SCALE
values and CF kernel choices. Skips all other benchmarks to keep wall
time manageable (~1-2 min per cell on a 3090 at n_granule=16384).

Two kernels tested:
  - "mexican_hat": LTD@T (1.0), LTP@T±1 (-0.5)  — production default
  - "variant_I":  LTD@T (1.0), broad LTP across [1, T-2] totaling -0.6,
                  single LTP@T+1 (-0.4)  — net-non-zero per cell

Scale grid is centered tight around the production default (0.2),
extending up to where sandbox sweeps showed saturation.

Outputs a table to stdout. Two intervals tested: 400ms and 1000ms.
Same seed (42) every cell, so differences are purely from SCALE / kernel.

Usage (from the directory containing test_pc_engine_simple.py and
cerebellum_diagnostics.py):

    python b24_sweep.py

Optionally, pass a different scale grid as space-separated args:

    python b24_sweep.py 0.1 0.2 0.3 0.5

The CF kernel sweep is fixed at 2 options. Edit KERNELS below to
change that.
"""
import sys
import os
import time
import math
import argparse
import gc
import io
import contextlib

import torch
import numpy as np

import test_pc_engine_simple as tps
from cerebellum_diagnostics import CerebellumDiagnosticSuite
from test_pc_engine_simple import (
    cerebellar_forward,
    cerebellar_learn,
    reset_reservoir_cascade,
)


# ----------------------------------------------------------------------
# CF kernel builders. Each returns a dict {step: cf_gain} for a given
# target_step.
# ----------------------------------------------------------------------

def kernel_mexican_hat(target_step, total_steps):
    """Production default: LTD@T, LTP@T±1."""
    k = {target_step: 1.0}
    if target_step - 1 >= 0:
        k[target_step - 1] = -0.5
    if target_step + 1 < total_steps:
        k[target_step + 1] = -0.5
    return k


def kernel_variant_I(target_step, total_steps):
    """Variant I: LTD@T, broad pre-LTP across [1, T-2], single post-LTP@T+1."""
    k = {target_step: 1.0}
    pre_window = list(range(1, max(2, target_step - 1)))
    if pre_window:
        ltp_each = -0.6 / len(pre_window)
        for s in pre_window:
            k[s] = ltp_each
    if target_step + 1 < total_steps:
        k[target_step + 1] = -0.4
    return k


def kernel_variant_K(target_step, total_steps):
    """Variant K: variant_I structure but post-LTP DISTRIBUTED across the
    full post-T test window instead of just step T+1.

    Hypothesis: variant_I's response curve undershoots mu (mu < T) more
    severely at long intervals (mu=789 vs T=1000 = 79%). This suggests
    the post-T region is poorly constrained — variant_I has 18-48 pre-LTP
    steps but only 1 post-LTP step. Spreading the post-LTP across the
    full [T+1, total-1] window applies symmetric-style suppression on
    both sides of T, which should tighten the response curve at the
    expense of more total LTP events per trial.

    Pre-LTP magnitudes kept identical to variant_I; only the post
    distribution differs.
    """
    k = {target_step: 1.0}
    pre_window = list(range(1, max(2, target_step - 1)))
    if pre_window:
        ltp_each = -0.6 / len(pre_window)
        for s in pre_window:
            k[s] = ltp_each
    post_window = list(range(target_step + 1, total_steps))
    if post_window:
        ltp_each = -0.4 / len(post_window)
        for s in post_window:
            k[s] = ltp_each
    return k


def kernel_variant_K_balanced(target_step, total_steps):
    """Variant K_balanced: rebalances pre/post LTP equally (-0.5 each)
    across their windows. Tests whether the 0.6/0.4 pre/post split from
    variant_I is suboptimal once the post window is broadened."""
    k = {target_step: 1.0}
    pre_window = list(range(1, max(2, target_step - 1)))
    if pre_window:
        ltp_each = -0.5 / len(pre_window)
        for s in pre_window:
            k[s] = ltp_each
    post_window = list(range(target_step + 1, total_steps))
    if post_window:
        ltp_each = -0.5 / len(post_window)
        for s in post_window:
            k[s] = ltp_each
    return k


KERNELS = {
    "mexican_hat":        kernel_mexican_hat,
    "variant_I":          kernel_variant_I,
    "variant_K":          kernel_variant_K,
    "variant_K_balanced": kernel_variant_K_balanced,
}


# ----------------------------------------------------------------------
# Single B24 run with a given SCALE and kernel.
# ----------------------------------------------------------------------

def run_b24_once(scale, kernel_fn, target_time_ms,
                 instant_elig=False, s3_weight=0.4, s1_weight=0.8, s2_weight=0.4,
                 deep_weights=None, normalize_peaks=True,
                 device='cuda', n_trials=50, dt_ms=20.0, n_l56=256, test_seed=42):
    """Returns (mu, sigma, weber, mu_pct_of_target, wall_time_s).

    deep_weights: list of floats of length deep_K, or None. When non-None,
        configures the engine to allocate a deep cascade of stages 4..3+deep_K
        and blend them into the microzone readout. Weights are in the
        "normalized" sense when normalize_peaks=True — each unit of weight
        contributes at the same peak amplitude as stage 1.
    """
    tps._MICRO_LR_SCALE_OVERRIDE = scale
    tps._MICRO_INSTANT_ELIGIBILITY = instant_elig
    tps._MICRO_S1_WEIGHT = s1_weight
    tps._MICRO_S2_WEIGHT = s2_weight
    tps._MICRO_S3_WEIGHT = s3_weight

    # Configure deep cascade BEFORE building the cerebellum so the reservoir
    # tensor gets allocated at init time. The cerebellum dict construction
    # reads _MICRO_CASCADE_DEEP_K to decide whether to allocate.
    if deep_weights is not None:
        tps._MICRO_CASCADE_DEEP_K = len(deep_weights)
        tps._MICRO_CASCADE_DEEP_WEIGHTS = list(deep_weights)
        tps._MICRO_NORMALIZE_DEEP_PEAKS = normalize_peaks
    else:
        tps._MICRO_CASCADE_DEEP_K = 0
        tps._MICRO_CASCADE_DEEP_WEIGHTS = None
        tps._MICRO_NORMALIZE_DEEP_PEAKS = True

    # Build a fresh suite and cerebellum each cell so state doesn't carry
    # over between cells. The cerebellar module prints a noisy setup
    # block on construction (and again on each _new_cerebellum call) that
    # garbles the sweep table. Capture and discard.
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        suite = CerebellumDiagnosticSuite(device=device, seed=42)
        engine, cereb = suite._new_cerebellum(test_seed=test_seed)

    target_pk = 32  # microzone
    target_step = int(target_time_ms / dt_ms)
    total_steps = target_step + 10

    cf_kernel = kernel_fn(target_step, total_steps)

    t0 = time.time()

    # Training (mirrors benchmark_24_interval_timing's training loop).
    # Also wrapped in stdout suppression in case anything inside the
    # engine prints during settling.
    with contextlib.redirect_stdout(captured):
        for trial in range(n_trials):
            engine.context_ema.zero_()
            reset_reservoir_cascade(cereb)
            for step in range(total_steps):
                s = torch.zeros(suite.n_l56, device=device)
                if step == 0:
                    s[0] = 1.0
                engine.set_l56(s)
                engine.settle(s)
                engine.update_context_ema()

                logits, gc_acts, gate = cerebellar_forward(engine, cereb)

                cf_gain = cf_kernel.get(step, 0.0)
                if cf_gain != 0.0:
                    cf_signal = torch.zeros(256, device=device)
                    cf_signal[target_pk] = cf_gain
                    cerebellar_learn(cereb, logits, gc_acts, target_pk,
                                     gate, engine, cf_signal=cf_signal)

        # Test (no plasticity)
        engine.context_ema.zero_()
        reset_reservoir_cascade(cereb)
        curve = []
        for step in range(total_steps):
            s = torch.zeros(suite.n_l56, device=device)
            if step == 0:
                s[0] = 1.0
            engine.set_l56(s)
            engine.settle(s)
            engine.update_context_ema()
            logits, _, _ = cerebellar_forward(engine, cereb)
            probs = torch.softmax(logits, dim=0)
            curve.append(probs[target_pk].item())

    wall_time_s = time.time() - t0

    curve = np.array(curve)
    curve = np.maximum(curve - curve.min(), 0)
    if curve.sum() < 1e-9:
        return target_time_ms, 0.0, float('nan'), 0.0, wall_time_s

    t_axis = np.arange(total_steps) * dt_ms
    mu = float(np.sum(t_axis * curve) / curve.sum())
    sigma = float(math.sqrt(np.sum(((t_axis - mu) ** 2) * curve) / curve.sum()))
    weber = sigma / (mu + 1e-9)
    mu_pct = mu / target_time_ms

    # Clear the overrides before returning so we don't leak into anything else.
    # Restore the production defaults (post-May-2026 sweep): microzone uses
    # pure-s3 blend at total magnitude 1.6, deep cascade OFF.
    tps._MICRO_LR_SCALE_OVERRIDE = None
    tps._MICRO_INSTANT_ELIGIBILITY = False
    tps._MICRO_S1_WEIGHT = 0.0
    tps._MICRO_S2_WEIGHT = 0.0
    tps._MICRO_S3_WEIGHT = 1.6
    tps._MICRO_CASCADE_DEEP_K = 0
    tps._MICRO_CASCADE_DEEP_WEIGHTS = None
    tps._MICRO_NORMALIZE_DEEP_PEAKS = True

    return mu, sigma, weber, mu_pct, wall_time_s


# ----------------------------------------------------------------------
# Sweep driver.
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scales", nargs="*", type=float,
                    default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7],
                    help="MICRO_LR_SCALE values to sweep (default: 0.05 0.1 0.2 0.3 0.5 0.7)")
    ap.add_argument("--kernels", nargs="*", default=["mexican_hat", "variant_I"],
                    choices=list(KERNELS.keys()),
                    help="Which CF kernels to test")
    ap.add_argument("--instant_elig", nargs="*", type=int, default=[0],
                    help="Microzone instant-eligibility toggle. 0=cascade-smoothed (default),"
                         " 1=instantaneous granule_blend. Pass both '0 1' to A/B test.")
    ap.add_argument("--blend", nargs="*", default=["0.0,0.0,1.6"],
                    help="Microzone (s1,s2,s3) blend weight triples to sweep. Default "
                         "'0.0,0.0,1.6' is the production-validated optimum (pure s3, "
                         "total magnitude matching global blend's 1.6). Pass "
                         "'0.8,0.4,0.4' to compare against the pre-redistribution "
                         "baseline. Total weight (s1+s2+s3) ~1.6 preserves overall "
                         "signal magnitude calibration.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_trials", type=int, default=50)
    ap.add_argument("--target_times", nargs="*", type=float,
                    default=[400.0, 1000.0])
    # ---------- Deep cascade extension (May 2026 architectural push) ----------
    ap.add_argument("--deep", nargs="*", default=[],
                    help="Deep cascade microzone-blend weights for stages 4..3+K. "
                         "Each value is a comma-separated list of floats — length "
                         "is the deep cascade depth K. Example: '0,0,0,0,0.45' "
                         "puts everything on stage 8 (deep_K=5). '0.09,0.09,0.09,0.09,0.09' "
                         "spreads 0.45 evenly across stages 4-8. Multiple values "
                         "to this flag run each as a separate cell. Weights are in "
                         "NORMALIZED units when --normalize=1 (default): w=0.45 "
                         "means peak contribution equal to 0.45 × stage_1_peak. "
                         "If empty, runs the K=3 baseline only (no deep extension). "
                         "Recommended budget: total normalized weight ≈ 0.45 to "
                         "match the current pure-s3@1.6 peak contribution.")
    ap.add_argument("--normalize", nargs="*", type=int, default=[1],
                    help="Per-stage peak compensation. 1 = multiply each deep "
                         "stage weight by √(2π(k-1)) so deep stages contribute "
                         "at stage-1 peak amplitude. 0 = use raw weights. Pass "
                         "'0 1' to A/B both.")
    ap.add_argument("--seeds", nargs="*", type=int, default=[42],
                    help="Test seeds. With multiple seeds, each cell runs once "
                         "per seed and the per-interval Weber values are "
                         "aggregated to a MEDIAN across seeds (matching the B7 "
                         "5-seed protocol from the May 2026 handoff). Single "
                         "seed [42] preserves single-seed exploration mode. "
                         "Recommended for verification: '42 7 13 99 2024' (5x "
                         "wall time per cell).")
    args = ap.parse_args()

    # Parse blend triples
    blends = []
    for b in args.blend:
        try:
            s1, s2, s3 = [float(x) for x in b.split(",")]
            blends.append((s1, s2, s3))
        except ValueError:
            print(f"Error: --blend arg '{b}' must be 's1,s2,s3' triple (e.g. '0.8,0.4,0.4')")
            return

    # Parse deep weight lists. Empty list = baseline only (no deep extension).
    # We always include a "no deep" cell as None so the baseline shows up in
    # the table for comparison even when --deep is passed.
    deep_configs = [None]
    for d in args.deep:
        try:
            w = [float(x) for x in d.split(",")]
            if len(w) == 0:
                raise ValueError("empty list")
            deep_configs.append(w)
        except ValueError:
            print(f"Error: --deep arg '{d}' must be comma-separated floats (e.g. '0,0,0,0,0.45')")
            return

    print(f"B24 sweep: scales={args.scales}, kernels={args.kernels}, "
          f"instant_elig={args.instant_elig}, blends={blends}, "
          f"target_times={args.target_times}, device={args.device}")
    print(f"Seeds: {args.seeds} "
          f"({'median across seeds' if len(args.seeds) > 1 else 'single seed'})")
    print(f"Deep cascade configs: {len(deep_configs)} (incl. baseline) — "
          f"normalize={args.normalize}")
    for i, dc in enumerate(deep_configs):
        if dc is None:
            print(f"  [{i}] baseline (K=3, no deep extension)")
        else:
            stages = ', '.join(f'k={4+j}:{w:.3f}' for j, w in enumerate(dc) if w != 0.0)
            print(f"  [{i}] deep_K={len(dc)}, weights={dc}, nonzero=({stages})")
    print("Reference (production variant_I / SCALE=0.125 / cascade-elig / blend=0.0,0.0,1.6): "
          "Weber 400≈0.327, 1000≈0.316")
    print()

    header = (
        f"{'kernel':<14} {'SCALE':>6} {'inst':>5} {'blend':>14} {'nrm':>4} "
        f"{'deep':<28} "
        f"{'W(400)':>7} {'mu(400)':>8} {'%T':>5} "
        f"{'W(1000)':>8} {'mu(1000)':>9} {'%T':>5} "
        f"{'time':>6}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for kname in args.kernels:
        kfn = KERNELS[kname]
        for instant in args.instant_elig:
            for (s1, s2, s3) in blends:
                for deep_w in deep_configs:
                    for nrm in args.normalize:
                        # When deep is OFF, normalize knob is meaningless;
                        # skip the duplicate.
                        if deep_w is None and nrm != args.normalize[0]:
                            continue
                        for scale in args.scales:
                            row = {'kernel': kname, 'scale': scale,
                                   'instant': bool(instant),
                                   's1': s1, 's2': s2, 's3': s3,
                                   'deep': deep_w, 'normalize': bool(nrm),
                                   'seeds': list(args.seeds)}
                            total_time = 0.0
                            # Per-interval, collect (mu, sigma, weber, pct) tuples
                            # across all seeds, then take the median Weber as the
                            # reported value. Seed-to-seed variation is ~0.01-0.02
                            # Weber at this regime; the median is robust to a single
                            # outlier seed. Individual seed values are kept so the
                            # verbose-mode print can surface them.
                            per_interval = {int(tt): [] for tt in args.target_times}
                            for seed in args.seeds:
                                for tt in args.target_times:
                                    mu, sig, weber, mu_pct, wt = run_b24_once(
                                        scale, kfn, tt,
                                        instant_elig=bool(instant),
                                        s1_weight=s1, s2_weight=s2, s3_weight=s3,
                                        deep_weights=deep_w,
                                        normalize_peaks=bool(nrm),
                                        device=args.device, n_trials=args.n_trials,
                                        test_seed=seed,
                                    )
                                    per_interval[int(tt)].append(
                                        {'seed': seed, 'mu': mu, 'sig': sig,
                                         'weber': weber, 'pct': mu_pct, 'wt': wt}
                                    )
                                    total_time += wt

                            # Reduce: median across seeds for each interval.
                            for tt in args.target_times:
                                vals = per_interval[int(tt)]
                                webers = sorted([v['weber'] for v in vals if not math.isnan(v['weber'])])
                                mus = sorted([v['mu'] for v in vals])
                                pcts = sorted([v['pct'] for v in vals])
                                # numpy median to handle even-N cases cleanly.
                                row[f'weber_{int(tt)}'] = float(np.median(webers)) if webers else float('nan')
                                row[f'mu_{int(tt)}'] = float(np.median(mus))
                                row[f'pct_{int(tt)}'] = float(np.median(pcts))
                                row[f'sig_{int(tt)}'] = float(np.median([v['sig'] for v in vals]))
                                row[f'per_seed_{int(tt)}'] = vals
                            row['time'] = total_time
                            results.append(row)

                            blend_str = f"{s1:.2f},{s2:.2f},{s3:.2f}"
                            if deep_w is None:
                                deep_str = "K=3 (baseline)"
                            else:
                                deep_str = f"K={3+len(deep_w)}:[" + ",".join(f"{w:.2f}" for w in deep_w) + "]"
                                deep_str = deep_str[:28]
                            print(
                                f"{kname:<14} {scale:>6.3f} {int(instant):>5d} {blend_str:>14} {int(nrm):>4} "
                                f"{deep_str:<28} "
                                f"{row.get('weber_400', float('nan')):>7.3f} "
                                f"{row.get('mu_400', float('nan')):>8.1f} "
                                f"{row.get('pct_400', 0.0)*100:>4.0f}% "
                                f"{row.get('weber_1000', float('nan')):>8.3f} "
                                f"{row.get('mu_1000', float('nan')):>9.1f} "
                                f"{row.get('pct_1000', 0.0)*100:>4.0f}% "
                                f"{row['time']:>5.0f}s",
                                flush=True,
                            )
                            # When multi-seed, show the per-seed Weber values
                            # so we can see the spread, not just the median.
                            if len(args.seeds) > 1:
                                for tt in args.target_times:
                                    ws = [v['weber'] for v in row[f'per_seed_{int(tt)}']]
                                    ws_str = " ".join(f"{w:.3f}" for w in ws)
                                    print(f"    seeds W({int(tt)}): [{ws_str}]  "
                                          f"min={min(ws):.3f} max={max(ws):.3f} "
                                          f"spread={max(ws)-min(ws):.3f}",
                                          flush=True)

                            # Reclaim memory between cells so the sweep can run many
                            # configs back-to-back without accumulation.
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

    # Summary
    print()
    print("Looking for: minimum max(W_400, W_1000) — balanced low Weber across intervals.")
    best = min(results,
               key=lambda r: max(r.get('weber_400', 1e9), r.get('weber_1000', 1e9)))
    deep_repr = "baseline" if best['deep'] is None else f"K={3+len(best['deep'])}:{best['deep']}"
    print(f"Best balanced cell: kernel={best['kernel']}, SCALE={best['scale']}, "
          f"blend=({best['s1']:.2f},{best['s2']:.2f},{best['s3']:.2f}), "
          f"deep={deep_repr}, normalize={best['normalize']}, "
          f"W=({best.get('weber_400', float('nan')):.3f}, "
          f"{best.get('weber_1000', float('nan')):.3f}) "
          f"[{'median across ' + str(len(args.seeds)) + ' seeds' if len(args.seeds) > 1 else 'single seed'}]")
    # Multi-seed pass check using per-seed values: a candidate passes only if
    # the median is below 0.20 AND no individual seed regresses too far past
    # the ceiling. Single-seed mode just reports the criterion as it stands.
    if len(args.seeds) > 1:
        ps_400 = best.get('per_seed_400', [])
        ps_1000 = best.get('per_seed_1000', [])
        all_passing = all(v['weber'] <= 0.20 for v in ps_400) and all(v['weber'] <= 0.20 for v in ps_1000)
        median_passing = best.get('weber_400', 1.0) <= 0.20 and best.get('weber_1000', 1.0) <= 0.20
        invariance = abs(best.get('weber_400', 1.0) - best.get('weber_1000', 1.0)) < 0.10
        print(f"  multi-seed verdict:  median≤0.20={median_passing}  "
              f"all_seeds≤0.20={all_passing}  invariance={invariance}")
    print("Pass criterion (Weber ≤ 0.20 at both intervals) needs both columns ≤ 0.20.")


if __name__ == "__main__":
    main()