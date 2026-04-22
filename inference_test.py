"""
inference_test.py — simulate ProbLog's query2(G, E) using erp values from TSV.

For each unique (G, E) pair, computes:
    P(query2(G, E)) = 1 - prod(1 - erp_value_i)
over all erp rows sharing that (G, E).
"""

import csv
import json
import pandas as pd
from collections import defaultdict

PATHWAY_ID = '196854'
TSV_FILE = f'./experiments/array_erp_{PATHWAY_ID}.tsv'
GROUNDTRUTH_JSON = f"./experiments/groundtruth_{PATHWAY_ID}.json"

def _parse_erp(row):
    """Parse erp(G, E, E2, G2) → returns (e, g2, e2)."""
    inner = row['ground_relation'][len('erp('):-1]
    parts = inner.split(',')
    e  = parts[1][3:].replace('_', '.')
    e2 = parts[2][3:].replace('_', '.')
    g2 = parts[3].lstrip('g')
    return e, g2, e2

def compute_query2_PL(tsv_file):
    groups = defaultdict(list)
    with open(tsv_file) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            e, g2, e2 = _parse_erp(row)
            val = float(row['erp_value'])
            groups[(row['fixed_gene'], e)].append(val)
            groups[(g2, e2)].append(val)

    results = {}
    for (gene, enzyme), erp_values in groups.items():
        prob = 1.0
        for v in erp_values:
            prob *= (1.0 - v)
        results[(gene, enzyme)] = (1.0 - prob, len(erp_values))

    return results

def compute_query2_SOS(tsv_file):
    groups = defaultdict(list)
    with open(tsv_file) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            e, g2, e2 = _parse_erp(row)
            val = float(row['erp_value'])
            groups[(row['fixed_gene'], e)].append(val)
            groups[(g2, e2)].append(val)

    results = {}
    for (gene, enzyme), erp_values in groups.items():
        results[(gene, enzyme)] = (max(erp_values), len(erp_values))

    return results

def evaluate_ranking(json_file, results_PL, results_SOS):
    """
    For each gene in the JSON, find the candidate enzyme with the highest
    predicted probability in PL, SOS, and baseline noisy_prob.
    Check if it matches true_enzyme.
    """
    with open(json_file) as f:
        data = json.load(f)

    pl_correct = 0
    sos_correct = 0
    baseline_correct = 0
    baseline_correct_erp = 0
    total_with_erp = 0
    total_baseline = len(data)

    for entry in data:
        gene = entry['gene'].lstrip('g')
        true_enzyme = entry['true_enzyme'][3:].replace('_', '.')

        # Baseline: Precision@1 — correct only when the true enzyme is the UNIQUE top scorer.
        noisy_scores = {c['enzyme'][3:].replace('_', '.'): c['noisy_prob'] for c in entry['candidates']}
        noisy_best = max(noisy_scores.values())
        baseline_top = [e for e, p in noisy_scores.items() if p == noisy_best]
        baseline_enzyme = baseline_top[0] if len(baseline_top) == 1 else None
        if len(baseline_top) == 1 and baseline_enzyme == true_enzyme:
            baseline_correct += 1

        pl_scores  = {}
        sos_scores = {}

        for candidate in entry['candidates']:
            enzyme = candidate['enzyme'][3:].replace('_', '.')
            key = (gene, enzyme)

            # Missing from ERP → no pathway support → score 0 (not a skip)
            pl_scores[enzyme]  = results_PL[key][0]  if key in results_PL  else 0.0
            sos_scores[enzyme] = results_SOS[key][0] if key in results_SOS else 0.0

        # Precision@1: correct only when the true enzyme is the UNIQUE top scorer.
        # Ties (e.g. ProbLog saturating at 1.0 for multiple candidates) count as wrong.
        pl_best  = max(pl_scores.values())
        sos_best = max(sos_scores.values())
        pl_top   = [e for e, p in pl_scores.items()  if p == pl_best]
        sos_top  = [e for e, p in sos_scores.items() if p == sos_best]

        total_with_erp += 1
        if baseline_enzyme == true_enzyme:
            baseline_correct_erp += 1
        if len(pl_top) == 1 and pl_top[0] == true_enzyme:
            pl_correct += 1
        if len(sos_top) == 1 and sos_top[0] == true_enzyme:
            sos_correct += 1

    b = baseline_correct / total_baseline if total_baseline else 0.0
    p = pl_correct / total_with_erp      if total_with_erp  else 0.0
    s = sos_correct / total_with_erp     if total_with_erp  else 0.0

    print(f"\n=== Evaluation Results (Precision@1, ties = wrong) ===")
    print(f"Baseline (noisy_prob):  {baseline_correct}/{total_baseline} = {b:.3f}")
    print(f"ProbLog (Q2):           {pl_correct}/{total_with_erp} = {p:.3f}")
    print(f"SOS     (max-ERP):      {sos_correct}/{total_with_erp} = {s:.3f}")
    print(f"RESULT baseline={b:.4f} problog={p:.4f} sos={s:.4f} n={total_with_erp}")


def spot_check(json_file, results_PL, results_SOS, target_gene=None, n=3):
    """
    Print a side-by-side comparison of noisy_prob, pl_query2, and sos_lb
    for each candidate enzyme of genes that have full ERP coverage.
    """
    with open(json_file) as f:
        data = json.load(f)

    shown = 0
    for entry in data:
        gene = entry['gene'].lstrip('g')
        if target_gene and gene != str(target_gene):
            continue

        true_enzyme = entry['true_enzyme'][3:].replace('_', '.')

        # Build rows for this gene; missing ERP entries get score 0
        rows = []
        for candidate in entry['candidates']:
            enzyme = candidate['enzyme'][3:].replace('_', '.')
            key = (gene, enzyme)
            rows.append({
                'enzyme':     enzyme,
                'noisy_prob': candidate['noisy_prob'],
                'pl_query2':  results_PL[key][0]  if key in results_PL  else 0.0,
                'sos_lb':     results_SOS[key][0] if key in results_SOS else 0.0,
                'true':       enzyme == true_enzyme,
            })

        print(f"\n--- Gene {gene} | true enzyme: {true_enzyme} ---")
        print(f"  {'enzyme':<14} {'noisy_prob':>12} {'pl_query2':>12} {'sos_lb':>10}  true?")
        for r in sorted(rows, key=lambda x: x['noisy_prob'], reverse=True):
            marker = ' <--' if r['true'] else ''
            print(f"  {r['enzyme']:<14} {r['noisy_prob']:>12.6f} {r['pl_query2']:>12.6f} {r['sos_lb']:>10.6f}{marker}")

        shown += 1
        if shown >= n:
            break


def load_erp_raw(tsv_file):
    """
    Load all ERP rows from the TSV, grouped by (gene, enzyme).

    Each ERP row  erp(gFG, ecE1, ecFE, gG2)  contributes to two (gene, enzyme) pairs:
      - 'left'  endpoint: (FG, E1)  — fixed_gene has function(FG, E1)
      - 'right' endpoint: (G2, FE)  — g2 has function(G2, FE)

    Returns:
      dict { (gene, enzyme): [ { 'erp_str', 'direction', 'bridge_e', 'other_gene',
                                  'erp_value', 'problog_value' }, ... ] }
    """
    raw = defaultdict(list)
    with open(tsv_file) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            e1, g2, fe = _parse_erp(row)   # e1=bridging, fe=fixed_enzyme, g2=other gene
            fg  = row['fixed_gene']
            erp_val     = float(row['erp_value'])
            problog_val = float(row['problog_value'])
            erp_str     = row['ground_relation']

            # Left endpoint: gene=FG scored for enzyme=E1
            raw[(fg, e1)].append({
                'erp_str':     erp_str,
                'direction':   'left',
                'bridge_e':    fe,          # the enzyme connecting via enzyme_pair
                'other_gene':  g2,
                'erp_value':   erp_val,
                'problog_value': problog_val,
            })

            # Right endpoint: gene=G2 scored for enzyme=FE
            raw[(g2, fe)].append({
                'erp_str':     erp_str,
                'direction':   'right',
                'bridge_e':    e1,
                'other_gene':  fg,
                'erp_value':   erp_val,
                'problog_value': problog_val,
            })

    return raw


def spot_check_verbose(json_file, results_PL, results_SOS, erp_raw,
                       target_gene=None, n=3, max_paths=5):
    """
    Like spot_check but shows the individual ERP paths that produced each
    pl_query2 and sos_lb score.
    """
    with open(json_file) as f:
        data = json.load(f)

    shown = 0
    for entry in data:
        gene = entry['gene'].lstrip('g')
        if target_gene and gene != str(target_gene):
            continue

        true_enzyme = entry['true_enzyme'][3:].replace('_', '.')

        # Title card + summary table (same as spot_check)
        print(f"\n{'='*70}")
        print(f"Gene {gene}  |  true enzyme: {true_enzyme}")
        print(f"{'='*70}")

        candidates = []
        for candidate in entry['candidates']:
            enzyme = candidate['enzyme'][3:].replace('_', '.')
            key = (gene, enzyme)
            pl_prob  = results_PL[key][0]  if key in results_PL  else 0.0
            sos_prob = results_SOS[key][0] if key in results_SOS else 0.0
            candidates.append((enzyme, candidate['noisy_prob'], pl_prob, sos_prob))

        candidates_sorted = sorted(candidates, key=lambda x: x[1], reverse=True)

        # Summary table
        print(f"\n  {'enzyme':<14} {'noisy_prob':>12} {'pl_query2':>12} {'sos_lb':>10}  true?")
        for enzyme, noisy_prob, pl_prob, sos_prob in candidates_sorted:
            marker = ' <--' if enzyme == true_enzyme else ''
            print(f"  {enzyme:<14} {noisy_prob:>12.6f} {pl_prob:>12.6f} {sos_prob:>10.6f}{marker}")

        # Per-candidate ERP path breakdown
        for enzyme, noisy_prob, pl_prob, sos_prob in candidates_sorted:
            marker = '  <-- TRUE' if enzyme == true_enzyme else ''
            print(f"\n  Candidate: {enzyme}{marker}")
            print(f"    noisy_prob : {noisy_prob:.6f}")
            print(f"    pl_query2  : {pl_prob:.6f}")
            print(f"    sos_lb     : {sos_prob:.6f}")

            paths = erp_raw.get((gene, enzyme), [])
            if not paths:
                print(f"    ERP paths  : (none — scored as 0)")
                continue

            paths_sorted = sorted(paths, key=lambda p: p['erp_value'], reverse=True)
            n_paths = len(paths_sorted)
            print(f"    ERP paths  : {n_paths} total  (showing top {min(max_paths, n_paths)})")
            print(f"    {'sos_contribution':>10} {'problog_val':>12}  erp_str")
            for p in paths_sorted[:max_paths]:
                print(f"    {p['erp_value']:>10.6f} {p['problog_value']:>12.6f}  {p['erp_str']}")
            if n_paths > max_paths:
                print(f"    ... ({n_paths - max_paths} more paths not shown)")

        shown += 1
        if shown >= n:
            break


if __name__ == '__main__':
    results_PL  = compute_query2_PL(TSV_FILE)
    results_SOS = compute_query2_SOS(TSV_FILE)
    rows = []
    for (gene, enzyme) in sorted(results_PL.keys()):
        pl_val,  pl_count  = results_PL[(gene, enzyme)]
        sos_val, sos_count = results_SOS[(gene, enzyme)]
        rows.append({
            'gene':        gene,
            'enzyme':      enzyme,
            'sos_lb':      sos_val,
            'pl_query2':   pl_val,
            'erp_count':   pl_count,
        })
    df = pd.DataFrame(rows)
    df['sos_minus_pl'] = (df['sos_lb'] - df['pl_query2']).round(6)
    df_sorted = df.sort_values('sos_minus_pl', ascending=False)

    print("The shape: ", df_sorted.shape)
    
    # filter 
    df_sorted = df_sorted[df_sorted['erp_count'] >= 4]

    print(f"Filtering by erp_count >= 4 yields {len(df_sorted)} (G,E) predictions.")
    # print(df_sorted.head(238).to_string(index=False))

    evaluate_ranking(GROUNDTRUTH_JSON, results_PL, results_SOS)
    # spot_check(GROUNDTRUTH_JSON, results_PL, results_SOS, n=10)

    # print("\n\n" + "="*70)
    # print("VERBOSE SPOT CHECK (ERP path breakdown)")
    # print("="*70)
    # erp_raw = load_erp_raw(TSV_FILE)
    # spot_check_verbose(GROUNDTRUTH_JSON, results_PL, results_SOS, erp_raw, n=4)