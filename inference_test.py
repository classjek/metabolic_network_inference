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

PATHWAY_ID = '1483249'
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

        # Baseline: pick highest noisy_prob directly from JSON (no skips possible)
        best_candidate = max(entry['candidates'], key=lambda c: c['noisy_prob'])
        baseline_enzyme = best_candidate['enzyme'][3:].replace('_', '.')
        if baseline_enzyme == true_enzyme:
            baseline_correct += 1

        pl_best_prob, pl_best_enzyme = -1, None
        sos_best_prob, sos_best_enzyme = -1, None
        skip = False

        for candidate in entry['candidates']:
            enzyme = candidate['enzyme'][3:].replace('_', '.')
            key = (gene, enzyme)

            if key not in results_PL:
                print(f"  [SKIP] ({gene}, {enzyme}) not in results_PL")
                skip = True
                break
            if key not in results_SOS:
                print(f"  [SKIP] ({gene}, {enzyme}) not in results_SOS")
                skip = True
                break

            pl_prob  = results_PL[key][0]
            sos_prob = results_SOS[key][0]

            if pl_prob > pl_best_prob:
                pl_best_prob, pl_best_enzyme = pl_prob, enzyme
            if sos_prob > sos_best_prob:
                sos_best_prob, sos_best_enzyme = sos_prob, enzyme

        if skip:
            continue

        total_with_erp += 1
        if baseline_enzyme == true_enzyme:
            baseline_correct_erp += 1
        if pl_best_enzyme == true_enzyme:
            pl_correct += 1
        if sos_best_enzyme == true_enzyme:
            sos_correct += 1

    print(f"\n=== Evaluation Results ===")
    print(f"Baseline (noisy_prob) all genes:      {baseline_correct}/{total_baseline} = {baseline_correct/total_baseline:.3f}")
    print(f"Baseline (noisy_prob) ERP subset:     {baseline_correct_erp}/{total_with_erp} = {baseline_correct_erp/total_with_erp:.3f}")
    print(f"ProbLog accuracy:                     {pl_correct}/{total_with_erp} = {pl_correct/total_with_erp:.3f}")
    print(f"SOS     accuracy:                     {sos_correct}/{total_with_erp} = {sos_correct/total_with_erp:.3f}")


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

        # Check full ERP coverage for this gene
        rows = []
        skip = False
        for candidate in entry['candidates']:
            enzyme = candidate['enzyme'][3:].replace('_', '.')
            key = (gene, enzyme)
            if key not in results_PL or key not in results_SOS:
                skip = True
                break
            rows.append({
                'enzyme':     enzyme,
                'noisy_prob': candidate['noisy_prob'],
                'pl_query2':  results_PL[key][0],
                'sos_lb':     results_SOS[key][0],
                'true':       enzyme == true_enzyme,
            })

        if skip:
            continue

        print(f"\n--- Gene {gene} | true enzyme: {true_enzyme} ---")
        print(f"  {'enzyme':<14} {'noisy_prob':>12} {'pl_query2':>12} {'sos_lb':>10}  true?")
        for r in sorted(rows, key=lambda x: x['noisy_prob'], reverse=True):
            marker = ' <--' if r['true'] else ''
            print(f"  {r['enzyme']:<14} {r['noisy_prob']:>12.6f} {r['pl_query2']:>12.6f} {r['sos_lb']:>10.6f}{marker}")

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
    
    # filter 
    df_sorted = df_sorted[df_sorted['erp_count'] >= 4]

    print(f"Filtering by erp_count >= 4 yields {len(df_sorted)} (G,E) predictions.")
    # print(df_sorted.head(238).to_string(index=False))

    evaluate_ranking(GROUNDTRUTH_JSON, results_PL, results_SOS)
    spot_check(GROUNDTRUTH_JSON, results_PL, results_SOS, n=3)