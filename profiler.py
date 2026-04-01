"""
Pathway Profiler: Find the sweet spot of small + highly connected pathways
"""
import json
import glob
from pathlib import Path
import pandas as pd
from collections import defaultdict

from ec_utils import load_rcr_from_cr_pairs, build_ortholog_pairs, ec_distance
from problog_writer import rank_ge_by_q2_support, rank_ge_by_q2_gene_paths, rank_ge_by_q3_support

# === SHARED DATA (load once) ===
RE_CSV       = "NData/RE.csv"
CR_JSON      = "NData/CR_pairs.json"
GE_EXPANDED  = "NData/GE_expanded.tsv"
SPECIES_UNIPROT = "NData/species_uniprot.json"
SPECIES = "9606"  # Human
WELL_DOCUMENTED_THRESHOLD = 50  # Minimum (G,E) pairs to count as well-documented

BANNED = {
    '30616','456216','456215','15377','15378','43474','15379','16526',
    '58189','37565','33019','58349','57783','57540','57945','57287',
    '59789','57856','57288'
}

print("Loading shared data...")
rec = pd.read_csv(RE_CSV, dtype=str, keep_default_na=False)
df_ge = pd.read_csv(GE_EXPANDED, sep="\t", dtype=str).fillna("")
df_ge["uniprot"] = df_ge["Entry"].str.strip().str.upper()

species_data = json.loads(Path(SPECIES_UNIPROT).read_text(encoding="utf-8-sig"))
df_sp = pd.DataFrame(species_data)
df_sp["uniprot"] = df_sp["uniprot"].astype(str).str.strip().str.upper()

merged = df_ge.merge(df_sp, on="uniprot", how="left", validate="m:1")
merged["EC number"] = (
    merged["EC number"].fillna("")
    .astype(str).str.replace(r"\s+", "", regex=True)
    .str.replace(",", ";")
)
ec_expanded = (
    merged.assign(EC_list=lambda d: d["EC number"].str.split(";"))
    .explode("EC_list")
    .rename(columns={"EC_list": "EC"})
)
ec_expanded = ec_expanded[ec_expanded["EC"].ne("")].drop_duplicates()
ec_expanded['GeneID'] = ec_expanded['GeneID'].astype(str).str.strip().str.rstrip(';').str.upper()
bad_gene_mask = (
    ec_expanded['GeneID'].eq('') | 
    ec_expanded['GeneID'].isin({'NA','N/A','NONE','NULL','NAN'})
)
ec_expanded = ec_expanded[~bad_gene_mask].copy()

banned_df = pd.DataFrame({'C': list(BANNED)})
print("Shared data loaded.\n")


def compute_ec_metrics(enzymes, confusion_threshold=4):
    """
    Compute EC clustering metrics for a set of enzymes.
    
    Returns:
    - mean_ec_dist: mean pairwise EC distance (lower = more clustered = harder noise)
    - confusion_potential: avg number of other enzymes within threshold distance per enzyme
    """
    enzymes = list(enzymes)
    n = len(enzymes)
    
    if n < 2:
        return 0.0, 0.0
    
    # Compute all pairwise distances
    total_dist = 0
    n_pairs = 0
    confusion_counts = [0] * n
    
    for i in range(n):
        for j in range(i + 1, n):
            d = ec_distance(enzymes[i], enzymes[j])
            total_dist += d
            n_pairs += 1
            
            # Count confusion potential (within threshold)
            if d <= confusion_threshold:
                confusion_counts[i] += 1
                confusion_counts[j] += 1
    
    mean_ec_dist = total_dist / n_pairs if n_pairs > 0 else 0
    confusion_potential = sum(confusion_counts) / n if n > 0 else 0
    
    return mean_ec_dist, confusion_potential


def build_indices_n_graph(rcr_df, re_df, ge_df, banned_df):
    """Build lookup indices and reaction adjacency graph."""
    E_to_R = defaultdict(set)
    R_to_E = defaultdict(set)
    for _, row in re_df.iterrows():
        E_to_R[row['EC']].add(row['R'])
        R_to_E[row['R']].add(row['EC'])
    
    E_to_G = defaultdict(set)
    for _, row in ge_df.iterrows():
        E_to_G[row['EC']].add(row['G'])
    
    all_compounds = set(rcr_df['C'].tolist()) if not rcr_df.empty else set()
    banned_compounds = set(banned_df['C'])
    accepted_compounds = all_compounds - banned_compounds

    adj = defaultdict(set)
    all_reactions = set()
    for _, row in rcr_df.iterrows():
        R1, C, R2 = row['R'], row['C'], row['R2']
        if C in accepted_compounds:
            adj[R1].add(R2)
            adj[R2].add(R1)
            all_reactions.update([R1, R2])
    
    for R in re_df['R'].unique():
        all_reactions.add(R)
        adj.setdefault(R, set())

    return {
        'E_to_R': E_to_R, 'R_to_E': R_to_E, 'E_to_G': E_to_G,
        'accepted_compounds': accepted_compounds, 'adj': adj,
        'all_reactions': all_reactions,
    }


def profile_pathway(pathway_json):
    """Profile a single pathway and return stats dict."""
    name = Path(pathway_json).stem
    
    # Load pathway reactions
    pathway_res = json.loads(Path(pathway_json).read_text(encoding="utf-8-sig"))
    Rset = {str(d['R']) for d in pathway_res}
    
    # Filter RE to pathway
    re_filtered = rec[rec['R'].isin(Rset)].copy()
    if re_filtered.empty:
        return None  # Skip pathways with no enzyme data
    
    # Build RCR from CR pairs
    rcr_df = load_rcr_from_cr_pairs(CR_JSON, Rset, BANNED)
    
    # Filter GE to pathway enzymes + documented species
    ec_filt = ec_expanded[ec_expanded['EC'].isin(re_filtered['EC'])]
    species_counts = ec_filt["species"].value_counts()
    documented_species = species_counts[species_counts >= WELL_DOCUMENTED_THRESHOLD].index
    ec_filt = ec_filt[ec_filt["species"].isin(documented_species)]
    
    if SPECIES not in documented_species:
        return None  # Skip if human isn't well-documented for this pathway
    
    # Count OTHER well-documented species (excluding target)
    n_other_species = len(documented_species) - 1  # subtract human
    
    # Build ortholog data
    ortholog_df = build_ortholog_pairs(ec_filt, target_species=SPECIES)
    ortholog_genes = set(ortholog_df['G2']) if not ortholog_df.empty else set()
    ge_orthologs = ec_filt[ec_filt['GeneID'].isin(ortholog_genes)][['GeneID', 'EC']].drop_duplicates()
    ge_orthologs.rename(columns={'GeneID': 'G'}, inplace=True)
    
    # Target species genes
    ge_filtered = (ec_filt[ec_filt['species'] == SPECIES])[['GeneID', 'EC']].drop_duplicates()
    ge_filtered.rename(columns={'GeneID': 'G'}, inplace=True)
    
    if ge_filtered.empty:
        return None
    
    ge_all = pd.concat([ge_filtered, ge_orthologs]).drop_duplicates()
    
    # Build index
    idx = build_indices_n_graph(rcr_df, re_filtered, ge_all, banned_df)
    
    # === SIZE METRICS ===
    n_reactions = len(Rset)
    n_enzymes = re_filtered['EC'].nunique()
    n_genes = ge_filtered['G'].nunique()
    n_ge_pairs = len(ge_filtered)
    n_rcr = len(rcr_df)
    n_orthologs = len(ortholog_df)
    
    # Ortholog additions
    n_genes_orth = ge_orthologs['G'].nunique()
    n_ge_orth = len(ge_orthologs)
    
    # === EC CLUSTERING METRICS ===
    pathway_enzymes = set(re_filtered['EC'].unique())
    mean_ec_dist, confusion_potential = compute_ec_metrics(pathway_enzymes)
    
    # === Q2 METRICS ===
    ranked_q2 = rank_ge_by_q2_support(ge_filtered, idx)
    q2_with_support = (ranked_q2['q2_paths'] > 0).sum()
    q2_pct = 100 * q2_with_support / len(ranked_q2) if len(ranked_q2) > 0 else 0
    q2_mean = ranked_q2['q2_paths'].mean() if len(ranked_q2) > 0 else 0
    
    # Q2 gene paths (more detailed)
    ranked_q2_gene = rank_ge_by_q2_gene_paths(ge_filtered, idx)
    q2_gene_with_support = (ranked_q2_gene['q2_paths'] > 0).sum()
    q2_gene_pct = 100 * q2_gene_with_support / len(ranked_q2_gene) if len(ranked_q2_gene) > 0 else 0
    
    # === Q3 METRICS ===
    ranked_q3 = rank_ge_by_q3_support(ge_filtered, idx, ortholog_df, ge_orthologs)
    q3_with_orth = ranked_q3['has_orth_support'].sum()
    q3_with_paths = (ranked_q3['q3_paths'] > 0).sum()
    q3_orth_pct = 100 * q3_with_orth / len(ranked_q3) if len(ranked_q3) > 0 else 0
    q3_pct = 100 * q3_with_paths / len(ranked_q3) if len(ranked_q3) > 0 else 0
    q3_mean = ranked_q3['q3_paths'].mean() if len(ranked_q3) > 0 else 0
    
    return {
        'name': name,
        'n_reactions': n_reactions,
        'n_enzymes': n_enzymes,
        'n_genes': n_genes,
        'n_ge_pairs': n_ge_pairs,
        'n_genes_orth': n_genes_orth,
        'n_ge_orth': n_ge_orth,
        'n_rcr': n_rcr,
        'n_orthologs': n_orthologs,
        'n_other_species': n_other_species,
        'mean_ec_dist': mean_ec_dist,
        'confusion_potential': confusion_potential,
        'q2_pct': q2_pct,
        'q2_gene_pct': q2_gene_pct,
        'q2_mean': q2_mean,
        'q3_orth_pct': q3_orth_pct,
        'q3_pct': q3_pct,
        'q3_mean': q3_mean,
    }


def print_profile(stats):
    """Pretty print pathway profile."""
    print(f"=== {stats['name']} ===")
    print(f"  Size: {stats['n_reactions']} reactions, {stats['n_enzymes']} enzymes, {stats['n_genes']} genes ({stats['n_ge_pairs']} GE pairs)")
    print(f"  Orthologs: +{stats['n_genes_orth']} genes, +{stats['n_ge_orth']} GE pairs from {stats['n_other_species']} other species")
    print(f"  RCR: {stats['n_rcr']}, Ortholog links: {stats['n_orthologs']}")
    print(f"  EC clustering: mean_dist={stats['mean_ec_dist']:.1f}, confusion={stats['confusion_potential']:.1f}")
    print(f"  Q2: {stats['q2_pct']:.1f}% support, {stats['q2_gene_pct']:.1f}% gene-paths, mean={stats['q2_mean']:.1f}")
    print(f"  Q3: {stats['q3_orth_pct']:.1f}% orth, {stats['q3_pct']:.1f}% paths, mean={stats['q3_mean']:.1f}")
    print()


if __name__ == "__main__":
    # Find all HSA (human) pathway JSONs
    pathway_files = sorted(glob.glob("NData/R-HSA-*.json"))
    print(f"Found {len(pathway_files)} HSA pathways")
    print(f"Well-documented threshold: >= {WELL_DOCUMENTED_THRESHOLD} (G,E) pairs\n")
    
    results = []
    for pf in pathway_files:
        stats = profile_pathway(pf)
        if stats:
            print_profile(stats)
            results.append(stats)
        else:
            print(f"=== {Path(pf).stem} === SKIPPED (no data for human)\n")
    
        print("\n" + "="*95)


    # Summary: sort by Q3 connectivity (best candidates)
    print("\n" + "="*115)
    print("TOP CANDIDATES (sorted by Q3% with paths, smallest first within tier)")
    print("="*115)
    print(f"{'Pathway':25} | {'Enz':>4} | {'Genes':>5} | {'GE':>5} | {'ECdist':>6} | {'Confus':>6} | {'Spp':>3} | {'Q2%':>5} | {'Q2g%':>5} | {'Q3%':>5}")
    print("-"*125)
    results.sort(key=lambda x: (-x['q3_pct'], x['n_genes']))
    for r in results[:30]:
        print(f"{r['name']:25} | {r['n_enzymes']:4} | {r['n_genes']:5} | {r['n_ge_pairs']:5} | {r['mean_ec_dist']:6.1f} | {r['confusion_potential']:6.1f} | {r['n_other_species']:3} | {r['q2_pct']:5.1f} | {r['q2_gene_pct']:5.1f} | {r['q3_pct']:5.1f}")
    # print(f"{'Pathway':25} | {'Enz':>4} | {'Genes':>5} | {'GE':>5} | {'ECdist':>6} | {'Confus':>6} | {'Spp':>3} | {'Q2%':>5} | {'Q3%':>5}")
    # print("-"*115)
    # results.sort(key=lambda x: (-x['q3_pct'], x['n_genes']))
    # for r in results[:30]:
    #     print(f"{r['name']:25} | {r['n_enzymes']:4} | {r['n_genes']:5} | {r['n_ge_pairs']:5} | {r['mean_ec_dist']:6.1f} | {r['confusion_potential']:6.1f} | {r['n_other_species']:3} | {r['q2_pct']:5.1f} | {r['q3_pct']:5.1f}")
