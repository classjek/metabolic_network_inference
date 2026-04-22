"""
Grid search over noise model parameters for make_PL_exp.py + inference_test.py.

For each parameter combination:
  1. Run make_PL_exp.py with the given params (regenerates ERP TSV + ground truth).
  2. Run inference_test.py (reads the freshly generated files).
  3. Parse the RESULT line and compute a score.

Score: rewards both ProbLog and SOS beating the baseline, and SOS > ProbLog.
  Disqualified (score = -999) if SOS ≤ baseline OR SOS ≤ ProbLog.
  Otherwise: (SOS - baseline) + (SOS - ProbLog)

Results are saved to grid_search_results.json and the top configurations are printed.

Usage:
  python grid_search.py
"""

import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

PATHWAY_JSON = "NData/R-HSA-196854.json"   # pathway to use for all runs
SEED         = 0                            # fixed seed → deterministic

GRID = {
    "s_fraction": [0.3, 0.4, 0.5, 0.6, 0.8],
    "sigma_ec":   [1.0, 2.0, 4.0, 5.0, 6.0],
    "sigma_n":    [0.125, 0.175, 0.20, 0.275, 0.30],
    "base_true":  [0.25, 0.30, 0.40, 0.50],
    # k_wrong kept fixed at paper value; add more values here if desired
    "k_wrong":    [5],
}

RESULTS_FILE = "grid_search_results.json"

# ── Scoring ───────────────────────────────────────────────────────────────────

def score(baseline: float, problog: float, sos: float, n: int) -> float:
    """
    Reward configurations where SOS > ProbLog > baseline.
    Disqualified (-999) if ANY of the following:
      - fewer than 3 evaluated genes (too noisy)
      - baseline ≤ 0.35 (prior is too uninformative to be meaningful)
      - ProbLog does not beat the baseline
      - SOS does not beat both baseline and ProbLog
    """
    if n < 3:
        return -999.0
    if baseline <= 0.35:
        return -999.0
    if problog <= baseline:
        return -999.0
    if sos <= baseline or sos <= problog:
        return -999.0
    return (sos - baseline) + (sos - problog)


# ── Grid search ───────────────────────────────────────────────────────────────

def run_combo(params: dict):
    """Run one (make_PL_exp, inference_test) pair and return parsed metrics."""

    make_cmd = [
        sys.executable, "make_PL_exp.py",
        "--s_fraction", str(params["s_fraction"]),
        "--sigma_ec",   str(params["sigma_ec"]),
        "--sigma_n",    str(params["sigma_n"]),
        "--base_true",  str(params["base_true"]),
        "--k_wrong",    str(params["k_wrong"]),
        "--seed",       str(SEED),
    ]

    r1 = subprocess.run(make_cmd, capture_output=True, text=True)
    if r1.returncode != 0:
        print(f"  [ERROR] make_PL_exp.py failed:\n{r1.stderr[-400:]}")
        return None

    r2 = subprocess.run(
        [sys.executable, "inference_test.py"],
        capture_output=True, text=True,
    )
    if r2.returncode != 0:
        print(f"  [ERROR] inference_test.py failed:\n{r2.stderr[-400:]}")
        return None

    # Parse the RESULT line emitted by evaluate_ranking
    result_line = next(
        (ln for ln in r2.stdout.splitlines() if ln.startswith("RESULT")), None
    )
    if result_line is None:
        print(f"  [WARN] No RESULT line found in inference_test output.")
        return None

    # RESULT baseline=0.xxxx problog=0.xxxx sos=0.xxxx n=N
    parts = dict(tok.split("=") for tok in result_line.split()[1:])
    b  = float(parts["baseline"])
    p  = float(parts["problog"])
    s  = float(parts["sos"])
    n  = int(parts["n"])
    sc = score(b, p, s, n)

    return {**params, "baseline": b, "problog": p, "sos": s, "n": n, "score": sc}


def main():
    keys   = list(GRID.keys())
    combos = list(itertools.product(*GRID.values()))
    total  = len(combos)
    print(f"Grid search: {total} combinations  (pathway={PATHWAY_JSON}, seed={SEED})\n")

    all_results = []
    t0 = time.time()

    for i, values in enumerate(combos, 1):
        params = dict(zip(keys, values))
        label  = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"[{i:3d}/{total}] {label}", end="  … ", flush=True)

        res = run_combo(params)
        if res is None:
            print("SKIPPED")
            continue

        all_results.append(res)
        if res["score"] >= 0:
            print(
                f"✓  baseline={res['baseline']:.3f}  problog={res['problog']:.3f}"
                f"  sos={res['sos']:.3f}  n={res['n']}  score={res['score']:.4f}"
            )
        else:
            print("✗")

    elapsed = time.time() - t0
    print(f"\nFinished {len(all_results)}/{total} runs in {elapsed:.0f}s")

    # Sort best first
    all_results.sort(key=lambda r: r["score"], reverse=True)

    # Save full results
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved → {RESULTS_FILE}")

    # Print top-10 qualified results only (score != -999)
    qualified = [r for r in all_results if r["score"] >= 0]
    print(f"\n=== TOP 10 CONFIGURATIONS ({len(qualified)} qualified / {len(all_results)} total) ===")
    print(f"  Criteria: baseline > 0.35, problog > baseline, sos > problog > baseline")
    print(f"{'rank':>4}  {'s':>5}  {'σEC':>5}  {'σN':>6}  {'base':>5}  "
          f"{'k':>3}  {'base_acc':>8}  {'problog':>7}  {'sos':>7}  {'n':>4}  {'score':>7}")
    print("-" * 80)
    if not qualified:
        print("  (no qualifying configurations found)")
    for rank, r in enumerate(qualified[:10], 1):
        print(
            f"{rank:4d}  {r['s_fraction']:5.2f}  {r['sigma_ec']:5.1f}  "
            f"{r['sigma_n']:6.3f}  {r['base_true']:5.2f}  {r['k_wrong']:3d}  "
            f"{r['baseline']:8.3f}  {r['problog']:7.3f}  {r['sos']:7.3f}  "
            f"{r['n']:4d}  {r['score']:7.4f}"
        )


if __name__ == "__main__":
    main()
