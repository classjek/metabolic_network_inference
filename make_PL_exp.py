import json
import random
from pathlib import Path
import pandas as pd
from collections import defaultdict

from ec_utils import norm_ec, ec_is_leaf, load_rcr_from_cr_pairs, build_ortholog_pairs
from noise_models import make_agnostic_prior, make_noisy_prior
from problog_writer import (compute_automorphism_orbits, write_single_problog, counts_by_kind, rank_ge_by_q2_support, rank_ge_by_q2_gene_paths, rank_ge_by_q3_support)

random.seed(0)

######################
### Configuration ###
######################

# R-BTA-1430728   species = 9913     bos taurus
# R-HSA-162582    species = 9606     homo sapiens
# R-HSA-1430728   species = 9606     homo sapiens
# R-MMU-168256    species = 10090    mus musculus   P(g,e) = 0.40
# R-MMU-1430728   species = 10090    mus musculus
# R-CFA-168256    species = 9615     canis familiaris <- Don't use, too small

# # New Pathways ##     nReactions   nSpeciesLinked   Name
# R-HSA-597592.json    998          12               Metabolism of Lipids
# R-HSA-9006934.json   929          11               Metabolism of Proteins
# R-HSA-392499.json    759          9                Signaling by Receptor Tyrosine Kinsases
# R-HSA-556833.json    606          8                Post-translational protein modification

# # More Pathways ##    nReactions   nSpeciesLinked   Name
# R-HSA-372790.json    791          7                Cytokine Signaling in Immune System
# R-HSA-1640170.json   762          7                Innate Immune System
# R-HSA-449147.json    505          6                Signaling by Interleukins
# R-HSA-168249.json    474          6                Cell Cycle
# R-HSA-1280215.json   394          6                Signaling buy GPCR

SPECIES = "9606"
# PATHWAY_JSON = "NData/R-HSA-162582.json"   
# PATHWAY_JSON = "NData/R-HSA-597592.json"   # L400 -> 2,  L250 -> 8 well documented species
# PATHWAY_JSON = "NData/R-HSA-9006934.json"  # L400 -> 1,  L250 -> 4 well documented species     pretty good 
# PATHWAY_JSON = "NData/R-HSA-392499.json"   # L400 -> 3,  L250 -> 10 well documented species
PATHWAY_JSON = "NData/R-HSA-556833.json"   # L400 -> 5,  L250 -> 10 well documented species    pretty good

# PATHWAY_JSON = "NData/R-HSA-372790.json"     # L250 -> 3 well documented species

# PATHWAY_JSON = "NData/R-HSA-1640170.json"    # L250 -> 4 well documented species  very few rcr
# PATHWAY_JSON = "NData/R-HSA-449147.json"     # Weird -> Signaling Pathway
# PATHWAY_JSON = "NData/R-HSA-168249.json"       # L250 -> 4 well documented species
# PATHWAY_JSON = "NData/R-HSA-1280215.json"    # Weird -> Signaling Pathway 


RE_CSV       = "NData/RE.csv"               # reaction_enzyme(R,E)
RCR_JSON     = "NData/new_RCR.json"         # reaction_compound_reaction(R1,C,R2)
CR_JSON      = "NData/CR_pairs.json"        # compound-reaction pairs for building RCR on the fly
GE_EXPANDED  = "NData/GE_expanded.tsv"
SPECIES_UNIPROT = "NData/species_uniprot.json"

# Very common compounds to exclude
BANNED = {
    '30616','456216','456215','15377','15378','43474','15379','16526',
    '58189','37565','33019','58349','57783','57540','57945','57287',
    '59789','57856','57288'
}

################################
### Load Data and Preprocess ###
################################

# Load pathway reactions
txt = Path(PATHWAY_JSON).read_text(encoding="utf-8-sig")  
pathway_res = json.loads(txt)
R_list = [d['R'] for d in pathway_res]
Rset = {str(x) for x in R_list}  # ensure all strings

# Create RCR (reaction-compound-reaction) triples
#rcr_from_cr = load_rcr_from_cr_pairs(CR_JSON, Rset, BANNED)
rcr_filtered = load_rcr_from_cr_pairs(RCR_JSON, Rset, BANNED)

# Load and filter RE (reaction-enzyme)
rec = pd.read_csv(RE_CSV, dtype=str, keep_default_na=False)
re_filtered = rec[rec['R'].isin(Rset)].copy()

# Load and process GE (gene-enzyme) data
df_ge = pd.read_csv(GE_EXPANDED, sep="\t", dtype=str).fillna("")
df_ge["uniprot"] = df_ge["Entry"].str.strip().str.upper()

txt = Path(SPECIES_UNIPROT).read_text(encoding="utf-8-sig") 
species = json.loads(txt)
df_sp = pd.DataFrame(species)
df_sp["uniprot"] = df_sp["uniprot"].astype(str).str.strip().str.upper()

merged = df_ge.merge(df_sp, on="uniprot", how="left", validate="m:1")

# Clean up merged df 
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
if bad_gene_mask.any():
    ec_expanded = ec_expanded[~bad_gene_mask].copy()

# Filter by enzymes present in pathway and well-documented species
ec_filt = ec_expanded[ec_expanded['EC'].isin(re_filtered['EC'])]
documented_species = (ec_filt["species"].value_counts()[lambda s: s >= 200]).index
print(f"We have {len(documented_species)} well documented species")
ec_filt = ec_filt[ec_filt["species"].isin(documented_species)]


# Ortholog relation is:  ortholog_support(G,E) :- ortholog(G,G2), function(G2,E).
# get ortholog pairs: ortholog(G,G2) where G is from target species
# only consider the well documented species 
ortholog_df = build_ortholog_pairs(ec_filt, target_species=SPECIES)
print(f"Ortholog pairs (target↔other): {len(ortholog_df)}")

# now use genes from non-target species to get function(G2,E) pairs
ortholog_genes = set(ortholog_df['G2']) 
ge_orthologs = ec_filt[ec_filt['GeneID'].isin(ortholog_genes)][['GeneID', 'EC']].drop_duplicates()
ge_orthologs.rename(columns={'GeneID': 'G'}, inplace=True)
print(f"(G,E) for ortholog genes: {len(ge_orthologs)}")

# Filter to target species
ge_filtered = (ec_filt[ec_filt['species'] == SPECIES])[['GeneID', 'EC']].drop_duplicates() 
ge_filtered.rename(columns={'GeneID':'G'}, inplace=True)

ge_all = pd.concat([ge_filtered, ge_orthologs]).drop_duplicates()
print(f"(G,E) total for ProbLog: {len(ge_all)}")

# Banned compounds dataframe
banned_df = pd.DataFrame({'C': list(BANNED)})

# Print summary
print(f"Num Reactions: {len(Rset)}")
print(f"(R1,C,R2):     {len(rcr_filtered)}")
print(f"(R,E):         {len(re_filtered)}")
print(f"(G,E):         {len(ge_filtered)}")
print(f"Banned:        {len(banned_df)}\n")


##############################
### Index & Graph Building ###
##############################

def build_indices_n_graph(rcr_filtered, re_filtered, ge_filtered, banned_df):
    """Build lookup indices and reaction adjacency graph."""
    # Build E -> R and R -> E maps
    E_to_R = defaultdict(set)
    R_to_E = defaultdict(set)
    for _, row in re_filtered.iterrows():
        R = row['R']
        EC = row['EC']
        E_to_R[EC].add(R)
        R_to_E[R].add(EC)
    
    # Build E -> G map
    E_to_G = defaultdict(set)
    for _, row in ge_filtered.iterrows():
        G = row['G']
        EC = row['EC']
        E_to_G[EC].add(G)
    
    all_compounds = set(rcr_filtered['C'].to_list())
    banned_compounds = set(banned_df['C'])
    accepted_compounds = all_compounds - banned_compounds

    # Build reaction-only graph via accepted compounds
    adj = defaultdict(set)
    all_reactions = set()

    for _, row in rcr_filtered.iterrows():
        R1 = row['R']
        C  = row['C']
        R2 = row['R2']
        if C in accepted_compounds:
            adj[R1].add(R2)
            adj[R2].add(R1)
            all_reactions.add(R1)
            all_reactions.add(R2)
    
    for R in re_filtered['R'].unique():
        all_reactions.add(R)
        adj.setdefault(R, set())
    
    degree = {R: len(adj[R]) for R in all_reactions}
    terminal_reactions = {R for R, d in degree.items() if d <= 1}
    filler_reactions = all_reactions - terminal_reactions

    return {
        'E_to_R': E_to_R,
        'R_to_E': R_to_E,
        'E_to_G': E_to_G,
        'accepted_compounds': accepted_compounds,
        'adj': adj,
        'all_reactions': all_reactions,
        'degree': degree,
        'terminal_reactions': terminal_reactions,
        'filler_reactions': filler_reactions,
    }

##################
### Evaluation ###
##################

def compute_auprc_from_prior(noisy_prior, GE_gold):
    """Compute AUPRC (Area Under Precision-Recall Curve) for the prior."""
    gold = set(GE_gold)

    pairs  = list(noisy_prior.keys())
    scores = [noisy_prior[p] for p in pairs]
    labels = [1 if p in gold else 0 for p in pairs]

    n_pos = sum(labels)
    if n_pos == 0:
        raise ValueError("No positives found in the prior; cannot compute AUCPR.")

    ranked = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)

    tp = fp = 0
    sum_prec_at_pos = 0.0
    for _, y in ranked:
        if y == 1:
            tp += 1
            sum_prec_at_pos += tp / (tp + fp)
        else:
            fp += 1

    auprc = sum_prec_at_pos / n_pos
    return auprc

######################
### Main Execution ###
######################

if __name__ == "__main__":

    # ## TESTING ##
    # print("=== Q2 Support Diagnostic (rank_ge_by_q2_support) ===")
    
    # # Build idx for OLD method
    # idx_old = build_indices_n_graph(rcr_filtered, re_filtered, ge_filtered, banned_df)
    # ranked_old = rank_ge_by_q2_support(ge_filtered, idx_old)
    
    # # Build idx for NEW method  
    # idx_new = build_indices_n_graph(rcr_from_cr, re_filtered, ge_filtered, banned_df)
    # ranked_new = rank_ge_by_q2_support(ge_filtered, idx_new)
    
    # # Compute summary stats
    # def q2_summary(df, label):
    #     n_total = len(df)
    #     n_with_support = (df['q2_paths'] > 0).sum()
    #     pct_support = 100 * n_with_support / n_total if n_total else 0
    #     mean_paths = df['q2_paths'].mean()
    #     total_paths = df['q2_paths'].sum()
    #     print(f"{label}:")
    #     print(f"  Pairs with support: {n_with_support}/{n_total} ({pct_support:.1f}%)")
    #     print(f"  Mean q2_paths:      {mean_paths:.2f}")
    #     print(f"  Total paths:        {total_paths}")
    
    # q2_summary(ranked_old, "OLD (rcr_filtered)")
    # q2_summary(ranked_new, "NEW (rcr_from_cr)")
    # exit()

    ## MAIN FUNCTION ##
    idx = build_indices_n_graph(rcr_filtered, re_filtered, ge_all, banned_df)

    # Build EC pool (leaf-level enzymes in this pathway)
    EC_pool = {norm_ec(e) for e in re_filtered['EC'].astype(str)}
    EC_pool = {e for e in EC_pool if ec_is_leaf(e)}

    # GE_gold = target species only (for evaluation)
    GE_gold = {(str(g), norm_ec(e)) for g, e in ge_filtered[['G','EC']].itertuples(index=False)}
    GE_gold = {(g,e) for (g,e) in GE_gold if e in EC_pool}

    # === AUPRC Evaluation (target species only) ===
    print("\n=== Creating Noisy Prior (for AUPRC) ===")
    eval_prior = make_noisy_prior(GE_gold, EC_pool)
    print(f"Eval prior has {len(eval_prior)} (G,E) pairs (target species)")
    
    initial_auprc = compute_auprc_from_prior(eval_prior, GE_gold)
    print(f"Baseline AUPRC: {initial_auprc:.4f}")

    # === Noisy prior for ProbLog (includes orthologs) ===
    print("\n=== Creating Full Prior (for ProbLog) ===")
    GE_all_set = {(str(g), norm_ec(e)) for g, e in ge_all[['G','EC']].itertuples(index=False)}
    GE_all_set = {(g, e) for (g, e) in GE_all_set if e in EC_pool}
    noisy_prior = make_noisy_prior(GE_all_set, EC_pool)
    print(f"Full prior has {len(noisy_prior)} (G,E) pairs (target + orthologs)")

    # === Q3 Support Diagnostic ===
    print("\n=== Q3 Support Diagnostic ===")
    ranked_q3 = rank_ge_by_q3_support(
        ge_df=ge_filtered,
        idx=idx,
        ortholog_df=ortholog_df,
        ge_orthologs_df=ge_orthologs
    )
    
    n_total = len(ranked_q3)
    n_with_orth = ranked_q3['has_orth_support'].sum()
    n_with_q3 = (ranked_q3['q3_paths'] > 0).sum()
    
    print(f"Pairs with ortholog support: {n_with_orth}/{n_total} ({100*n_with_orth/n_total:.1f}%)")
    print(f"Pairs with Q3 paths:         {n_with_q3}/{n_total} ({100*n_with_q3/n_total:.1f}%)")
    print(f"Mean Q3 paths:               {ranked_q3['q3_paths'].mean():.2f}")
    print(f"Total Q3 paths:              {ranked_q3['q3_paths'].sum()}")
    
    print("\n--- Top 10 by Q3 support ---")
    print(ranked_q3[['G','EC','q3_paths','unique_G2','has_orth_support']].head(10))

    # # Compute automorphism orbits
    # print("\n=== Computing Automorphism Orbits ===")
    # entity_orbits = compute_automorphism_orbits(
    #     prior=noisy_prior,
    #     rcr_df=rcr_filtered[['R','C','R2']].copy(),
    #     re_df=re_filtered[['R','EC']].copy(),
    #     accepted_compounds=idx['accepted_compounds'],
    #     ortholog_df=ortholog_df, 
    #     p_round=6
    # )

    # counts = counts_by_kind(entity_orbits)
    # print(f"Unique orbits per kind: {counts}")
    # print(f"Total unique entities: {sum(counts.values())}")

    # # Write ProbLog file
    # print("\n=== Writing ProbLog File ===")
    # write_single_problog(
    #     out_path="NData/q3_single.pl",
    #     prior=noisy_prior,
    #     rcr_df=rcr_filtered[['R','C','R2']].copy(),
    #     re_df=re_filtered[['R','EC']].copy(),
    #     accepted_compounds=idx['accepted_compounds'],
    #     ortholog_df=ortholog_df, 
    #     targets=None            
    # )