import json
import re, math, random
from pathlib import Path
import pandas as pd
from collections import defaultdict
from collections import Counter
from typing import Optional, List, Tuple, Dict
from functools import lru_cache
from datetime import datetime
import pynauty as pn
from collections import defaultdict

random.seed(0)

# R-BTA-1430728   species = 9913     bos taurus
# R-HSA-162582    species = 9606     homo sapiens
# R-HSA-1430728   species = 9606     homo sapiens
# R-MMU-168256    species = 10090    mus musculus   P(g,e) = 0.40
# R-MMU-1430728   species = 10090    mus musculus
# R-CFA-168256    species = 9615     canis familiaris <- Don't use, too small. Or bug?

SPECIES = "10090"
PATHWAY_JSON = "NData/R-MMU-168256.json"   
# SPECIES = "9913"
# PATHWAY_JSON = "NData/R-BTA-1430728.json"
RE_CSV       = "NData/RE.csv"               # reaction_enzyme(R,E)
RCR_JSON     = "NData/new_RCR.json"    # reaction_compound_reaction(R1,C,R2), very huge this time
GE_EXPANDED = "NData/GE_expanded.tsv"
SPECIES_UNIPROT = "NData/species_uniprot.json"

# Very common compounds to exclude
BANNED = {'30616','456216','456215','15377','15378','43474','15379','16526','58189','37565','33019','58349','57783','57540','57945','57287','59789','57856', '58189','37565','33019','58349','57783','57540','57945','57287','59789','57856'}
BANNED.add('57288')
# BANNED = {'15377', '15378', '57783', '58349', '57287', '30616', '43474'}

# Extract List of Dicts of all relevant reactions
# Ex: [{'R': 'IN=10723;15377|OUT=28325'}, {'R': 'IN=10723;15378;57783|OUT=28493;58349'}, {'R': 'IN=10983;57540|OUT=13705;15378;57945'}]
txt = Path(PATHWAY_JSON).read_text(encoding="utf-8-sig")  
pathway_res = json.loads(txt)
R_list = [d['R'] for d in pathway_res] # extract second term, get list of strings
#Rset = set(R_list) # for fast lookup 
Rset = {str(x) for x in R_list}  # ensure all strings

# Now filter other files based on the resulting set of reactions
# reaction_compound_reaction(R1,C,R2)
rcr = Path(RCR_JSON).read_text(encoding="utf-8-sig")
rcr_list = json.loads(rcr)
for d in rcr_list: # convert from int to str
    d['R'] = str(d['R'])
    d['R2'] = str(d['R2'])

#print(f"Length of rcr_list = {len(rcr_list)} and first entry is: {type(rcr_list[0]['R']), type(rcr_list[0]['C']), type(rcr_list[0]['R2'])}")

rcr_filtered = []
for d in rcr_list:
    # d is a dict with keys R1,C,R2. Only keep if both R1 and R2 are in Rset, and C is not in BANNED
    if d['R'] in Rset and d['R2'] in Rset and d['C'][6:] not in BANNED:
        rcr_filtered.append(d)
rcr_filtered = pd.DataFrame(rcr_filtered) # convert from list to dataframe
rcr_filtered['C'] = rcr_filtered['C'].str.split(':').str[1]  # remove "CHEBI:"

rec = pd.read_csv(RE_CSV, dtype=str, keep_default_na=False)
re_filtered = rec[rec['R'].isin(Rset)].copy()

# Now get function(G,E) and Ortholog(G1,G2) relation
df_ge = pd.read_csv(GE_EXPANDED, sep="\t", dtype=str).fillna("")
df_ge["uniprot"] = df_ge["Entry"].str.strip().str.upper()
txt = Path(SPECIES_UNIPROT).read_text(encoding="utf-8-sig") 
species = json.loads(txt)
df_sp = pd.DataFrame(species)
df_sp["uniprot"] = df_sp["uniprot"].astype(str).str.strip().str.upper()

merged = df_ge.merge(df_sp, on="uniprot", how="left", validate="m:1")

# Clean up merged df 
merged["EC number"] = (merged["EC number"].fillna("") .astype(str).str.replace(r"\s+", "", regex=True) .str.replace(",", ";"))
ec_expanded = (merged.assign(EC_list=lambda d: d["EC number"].str.split(";")).explode("EC_list").rename(columns={"EC_list": "EC"}))
ec_expanded = ec_expanded[ec_expanded["EC"].ne("")].drop_duplicates()
ec_expanded['GeneID']  = ec_expanded['GeneID'].astype(str).str.strip().str.rstrip(';').str.upper()

bad_gene_mask = (ec_expanded['GeneID'].eq('') | ec_expanded['GeneID'].isin({'NA','N/A','NONE','NULL','NAN'}))
if bad_gene_mask.any():
    #print("Dropping empty/placeholder GeneID rows:", bad_gene_mask.sum())
    ec_expanded = ec_expanded[~bad_gene_mask].copy()

# Filter ec_expanded by the enzymes(EC numbers) present
ec_filt = ec_expanded[ec_expanded['EC'].isin(re_filtered['EC'])]
documented_species = (ec_filt["species"].value_counts()[lambda s: s >= 200]).index # only consider well documented species. 400 is arbitrary threshold
print(f"We have {len(documented_species)} well documented species")
ec_filt = ec_filt[ec_filt["species"].isin(documented_species)]

# Now we can filter it by species 
# inspect filtering by different species
# specs = ec_filt['species'].unique()
# for spe in specs: 
#     print(f"species {spe}: {(ec_filt[ec_filt['species'] == spe])[['GeneID', 'EC']].drop_duplicates().shape}")
#ge_filtered = (ec_filt[ec_filt['species'] == '10116'])[['GeneID', 'EC']].drop_duplicates() # Rat is 10116
ge_filtered = (ec_filt[ec_filt['species'] == SPECIES])[['GeneID', 'EC']].drop_duplicates() 
ge_filtered.rename(columns={'GeneID':'G'}, inplace=True)

# turn banned compounds into dataframe 
banned_df = pd.DataFrame({'C': list(BANNED)})

# # Results
print(f"Num Reactions: {len(Rset)}")
print(f"(R1,C,R2):     {len(rcr_filtered)},    from {len(rcr_list)}")
print(f"(R,E):         {len(re_filtered)},     from {len(rec)}")
print(f"(G,E):         {len(ge_filtered)},    ")
print(f"Banned:        {len(banned_df)},      \n")


# Works on filtered relations
def build_indices_n_graph(rcr_filtered, re_filtered, ge_filtered, banned_df):
    # Build E -> R and R -> E maps for fast lookups 
    E_to_R = defaultdict(set)
    R_to_E = defaultdict(set)
    for _, row in re_filtered.iterrows():
        R = row['R']
        EC = row['EC']
        E_to_R[EC].add(R)
        R_to_E[R].add(EC)
    
    # Build E -> G map for fast lookup
    E_to_G = defaultdict(set)
    for _, row in ge_filtered.iterrows():
        G = row['G']
        EC = row['EC']
        E_to_G[EC].add(G)
    
    all_compounds = set(rcr_filtered['C'].to_list())
    banned_compounds = set(banned_df['C'])
    accepted_compounds = all_compounds - banned_compounds

    # build reaction-only graph via accepted compounds
    adj = defaultdict(set)
    all_reactions = set()

    # see which reactions connect through allowed compounds
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
    
    # degree[R]: how many neighbors R has in the reaction-only graph
    degree = {R: len(adj[R]) for R in all_reactions}

    terminal_reactions = {R for R, d in degree.items() if d <= 1}
    filler_reactions   = all_reactions - terminal_reactions

    return {
        # core indices
        'E_to_R': E_to_R,
        'R_to_E': R_to_E,
        'E_to_G': E_to_G,

        # compound filter
        'accepted_compounds': accepted_compounds,

        # reaction-only graph
        'adj': adj,
        'all_reactions': all_reactions,
        'degree': degree,
        'terminal_reactions': terminal_reactions,
        'filler_reactions': filler_reactions,
    }

#region TESTING MAP
def rank_ge_by_q2_support(ge_df, idx):
    E_to_R  = idx['E_to_R']
    R_to_E  = idx['R_to_E']
    E_to_G  = idx['E_to_G']
    adj     = idx['adj']

    # Precompute which E' have any genes (faster membership test)
    E_has_gene = {e for e, genes in E_to_G.items() if genes}

    rows = []
    for G, E in ge_df[['G','EC']].itertuples(index=False):
        if E not in E_to_R:
            #rows.append((G, E, 0, 0, 0))
            continue

        seen_paths = set()
        seen_E2    = set()
        seen_R2    = set()

        # For each reaction Ra using E, walk one RCR step to Rb, then to enzymes E2 at Rb
        for Ra in E_to_R[E]:
            for Rb in adj.get(Ra, ()):
                for E2 in R_to_E.get(Rb, ()):
                    if E2 in E_has_gene:
                        seen_paths.add((Ra, Rb, E2))
                        seen_E2.add(E2)
                        seen_R2.add(Rb)

        rows.append((G, E, len(seen_paths), len(seen_E2), len(seen_R2)))

    out = (pd.DataFrame(rows, columns=['G','EC','q2_paths','unique_E2','unique_R2'])
             .sort_values(['q2_paths','unique_E2','unique_R2'], ascending=False, kind='mergesort')
             .reset_index(drop=True))
    return out

def rank_ge_by_q2_gene_paths(ge_df, idx, exclude_same_gene=True, exclude_same_ec=False):

    E_to_R  = idx['E_to_R']   # EC -> {R}
    R_to_E  = idx['R_to_E']   # R  -> {EC}
    E_to_G  = idx['E_to_G']   # EC -> {G}
    adj     = idx['adj']      # R  -> {R'}

    rows = []
    for G, E in ge_df[['G','EC']].itertuples(index=False):
        # If this EC has no reactions, there's no Q2 support
        if E not in E_to_R:
            #rows.append((G,E,0,0,0,0))
            continue

        seen_paths = set()  # (Ra, Rb, E2, G2)
        seen_G2    = set()
        seen_E2    = set()
        seen_R2    = set()

        for Ra in E_to_R[E]:
            for Rb in adj.get(Ra, ()):
                for E2 in R_to_E.get(Rb, ()):
                    if exclude_same_ec and E2 == E:
                        continue
                    # For Q2, we require a gene G' for E2 on the other end
                    for G2 in E_to_G.get(E2, ()):
                        if exclude_same_gene and G2 == G:
                            continue
                        seen_paths.add((Ra, Rb, E2, G2))
                        seen_G2.add(G2)
                        seen_E2.add(E2)
                        seen_R2.add(Rb)

        rows.append((G, E, len(seen_paths), len(seen_G2), len(seen_E2), len(seen_R2)))

    out = (pd.DataFrame(rows, columns=['G','EC','q2_paths','unique_G2','unique_E2','unique_R2'])
             .sort_values(['q2_paths','unique_G2','unique_E2','unique_R2'],
                          ascending=False, kind='mergesort')
             .reset_index(drop=True))
    return out
#endregion

idx = build_indices_n_graph(rcr_filtered, re_filtered, ge_filtered, banned_df)

###################
### Noise Model ###
###################

S_FRACTION = 0.50    # s: fraction of genes to corrupt
K_WRONG    = 5       # k: wrong ECs per corrupted gene
SIGMA_EC   = 1.5     # how "local" wrong ECs are in EC tree
SIGMA_N    = 0.125   # Gaussian jitter added to probabilities
BASE_TRUE  = 0.40    # prior for true links before noise

## PREPROCESSING FOR NOISE MODEL ##
def norm_ec(x:str) -> str: # clean EC string 
    return re.sub(r'^\s*EC\s*', '', str(x).strip())

@lru_cache(None)
def ec_levels(ec:str):
    # ECs are of the form X.X.X.X where each X is an integer or '-'
    # Return a tuple of 4 levels, with None for missing levels 
    parts = norm_ec(ec).split('.')
    parts += ['-'] * (4 - len(parts))
    out = []
    for p in parts[:4]:
        if p in ('', '-', None):
            out.append(None)
        else:
            out.append(int(p) if p.isdigit() else None)
    return tuple(out)

@lru_cache(None)
def ec_distance(a:str, b:str) -> int:
    # Compute distance between two EC numbers in the EC hierarchy
    A, B = ec_levels(a), ec_levels(b)
    for i in range(4):
        if A[i] != B[i]:
            # distance = up from A to the lowest common ancestor, then down to B
            return (4 - i) * 2
    return 0

def ec_is_leaf(ec:str) -> bool:
    return all(isinstance(x, int) for x in ec_levels(ec))

# Need this because Problog atoms cannot have '.' or '-' or start with a digit
def r_atom(r: str) -> str: return 'r' + ''.join(ch for ch in str(r) if ch.isdigit())
def g_atom(g:str) -> str: return 'g' + re.sub(r'\W+', '', str(g))
def ec_atom(ec:str) -> str:
    lev = [str(x) for x in ec_levels(ec) if isinstance(x,int)]
    return 'ec_' + '_'.join(lev) if lev else 'ec'
def c_atom(c: str) -> str: return 'c' + re.sub(r'\W+', '', str(c))
######## END PREPROCESSING ########

######### BUILD EC POOL #########
EC_pool = {norm_ec(e) for e in re_filtered['EC'].astype(str)} # Collect ECs present in pathway
EC_pool = {e for e in EC_pool if ec_is_leaf(e)}  # Filter to leaf ECs(have all 4 parts) only

# Okay so we are primarily going to copy Experiment 4.2 in the paper, which uses enzymes in its noise model that aren't 
# necessarily present in the pathway. We can try this, but it seems weird and may be something they just forgot to explicitly
# mention. Instead, we'll begin with enzymes present in the pathway only.
GE_gold = {(str(g), norm_ec(e)) for g, e in ge_filtered[['G','EC']].itertuples(index=False)} # (G,E) pairs from species-filtered GE data
GE_gold = {(g,e) for (g,e) in GE_gold if e in EC_pool} # Keep only ECs associated with pathway
######## END EC POOL #########

#region NOISE MODEL FUNCTIONS
# For first experiment: Agnostic Noise Model 

def _ec_prefix_tuple(ec: str, depth: int):
    """Tuple key of the first `depth` integer EC levels, consistent with ec_atom/ec_levels."""
    levels = [x for x in ec_levels(ec) if isinstance(x, int)]
    return tuple(levels[:depth])

def make_agnostic_prior(GE_gold, EC_pool, s_fraction=0.01, lcp_depth=2, rng=None):
    if rng is None:
        rng = random

    gold_links = list(GE_gold)
    rng.shuffle(gold_links)
    n_select = max(0, int(round(s_fraction * len(gold_links))))
    selected = sorted(gold_links[:n_select], key=lambda t: (t[0], t[1])) 

    prior = {(g, e): 1.0 for (g, e) in gold_links}

    # Pre-index EC_pool by LCP prefix for quick bucket lookup
    buckets_by_prefix = defaultdict(list)
    for e in EC_pool:
        if not ec_is_leaf(e):
            continue
        buckets_by_prefix[_ec_prefix_tuple(e, lcp_depth)].append(e)

    # Diagnostics
    bucket_sizes = Counter()
    n_buckets, n_injected = 0, 0

    gold_set = set(GE_gold)           
    perturbations = []   

    for (g, e_true) in selected:
        pref = _ec_prefix_tuple(e_true, lcp_depth)
        bucket = buckets_by_prefix.get(pref, [])
        if not bucket:
            continue

        p = 1.0 / len(bucket)
        bucket_sizes[len(bucket)] += 1
        n_buckets += 1

        # Assign uniform weight to all ECs (including true one)
        for e_prime in bucket:
            if (g, e_prime) not in prior:
                n_injected += 1
            prior[(g, e_prime)] = p

        prior[(g, e_true)] = p # redundant 

        # Generate info on perturbations
        true_in_bucket = sorted(e for e in bucket if (g, e) in gold_set)      
        injected_in_bucket = sorted(e for e in bucket if (g, e) not in gold_set)  
        perturbations.append({'g': g, 'prefix': pref, 'p': p, 'bucket': sorted(bucket), 'true_in_bucket': true_in_bucket, 'injected_in_bucket': injected_in_bucket})

    # Stats
    n_gold = len(gold_links)
    n_total = len(prior)
    print(f"[§4.1 agnostic] Perturbed gold links: {n_select}/{n_gold} (s={s_fraction})")
    print(f"[§4.1 agnostic] Buckets created: {n_buckets}  size_hist={dict(sorted(bucket_sizes.items()))}")
    print(f"[§4.1 agnostic] Prior size: {n_total}  (gold {n_gold}, injected {n_total - n_gold})")

    return prior, perturbations

agnostic_prior, perturbations = make_agnostic_prior(GE_gold, EC_pool)

# "Sampling selects enzymes using a normal distribution 𝒩(0, σ_EC) over the topological distance" Pg.423
def _gauss_w(d, sigmaEC):  # weight ∝ exp(-d^2 / (2 σ^2))
    return math.exp(-(d*d)/(2.0*sigmaEC*sigmaEC))

def sample_wrong_ecs(target_ec, pool, k, sigmaEC):
    # Pick k ECs from pool that are 'close' to target_ec 
    # Closeness is biased by a Gaussian over ec_distance
    cands = [e for e in pool if e != target_ec]
    if not cands:
        return []

    dists   = [ec_distance(target_ec, e) for e in cands]
    weights = [_gauss_w(d, sigmaEC) for d in dists]
    total   = sum(weights)
    if total == 0:
        return []

    probs = [w/total for w in weights]
    picks = random.choices(range(len(cands)), weights=weights, k=k)  # indices
    # map picked indices to (EC, prob) tuples
    return [(cands[i], probs[i]) for i in picks]

# For the second experiment: Unreliable Predictions
# TODO: Consider changing this so that prior probabilities are symmetric, i.e. equal across distance levels
def make_noisy_prior(GE_gold, EC_pool, s_fraction=S_FRACTION, k_wrong=K_WRONG, sigma_ec=SIGMA_EC, sigma_n=SIGMA_N, base_true=BASE_TRUE):
    # 1) Select a fraction s of genes to corrupt.
    # 2) For each (g, E_true) of those genes, add k_wrong links (g, E_wrong), where E_wrong is sampled with probability ∝ N(0, σ_EC) over the topological distance.
    # 3) Set belief for each (g, E) to: selection_probability(g,E) + N(0, σ_N), clipped to [0,1]. For true links, we use a high base (base_true) before the same N(0, σ_N) jitter.
    # Returns: dict {(G, EC) -> probability}
    def clip01(x): return 0.0 if x < 0 else (1.0 if x > 1 else x)

    G_to_trueE = defaultdict(set)
    for g, e in GE_gold:
        G_to_trueE[g].add(e)
    
    genes = list(G_to_trueE.keys())
    random.shuffle(genes)
    n_corrupt = int(s_fraction * len(genes))
    print(f"Corrupting {n_corrupt} out of {len(genes)} genes (s={s_fraction})")
    corrupt = set(genes[:n_corrupt])

    # Start with true links at a high base probability
    prior = {(g, e): base_true for (g, e) in GE_gold}

    injected_dist = Counter()   
    injected_cnt  = 0

    for g in corrupt:
        for e_true in G_to_trueE[g]:
            picks = sample_wrong_ecs(e_true, EC_pool, k_wrong, sigma_ec)
            for e_wrong, p_select in picks: 
                if (g, e_wrong) in GE_gold:
                    # It's already a true link—leave it as base_true; still count distance for stats
                    pass
                else:
                    # If multiple true ECs of g inject the same wrong EC, keep the max selection prob (most generous)
                    prior[(g, e_wrong)] = max(prior.get((g, e_wrong), 0.0), p_select)
                    injected_cnt += 1
                # diagnostics: measure distance from this true EC (one proxy)
                d = ec_distance(e_true, e_wrong)
                injected_dist[d] += 1
    
    # Add N(0, σ_N) noise to *every* prior entry and clip to [0,1]
    for key, p in list(prior.items()):
        prior[key] = clip01(p + random.gauss(0.0, sigma_n))
    
    print(f"[§4.2] Corrupted genes: {n_corrupt} / {len(genes)}")
    print(f"[§4.2] Injected wrong links (draws): {injected_cnt}")
    if injected_cnt == 0:
        print("[§4.2] No injected links—check EC_pool/GE_gold sizes.")
    else:
        # show how local the injected ECs are (should skew to small distances when σ_EC is small)
        print("[§4.2] Injected EC distance histogram (d -> count):", dict(sorted(injected_dist.items())))
        # rough share of non-gold links in the prior
        n_gold = len(GE_gold)
        n_total = len(prior)
        print(f"[§4.2] Prior size: {n_total} entries  (gold {n_gold}, injected {n_total-n_gold})")

    return prior

noisy_prior = make_noisy_prior(GE_gold, EC_pool)
#endregion

#region TESTING THE NOISE MODEL
## Checks for Noisy Prior ##
G_to_trueE = defaultdict(set)
for g,e in GE_gold:
    G_to_trueE[g].add(e)

def nearest_true_distance(g, e_wrong):
    """Distance from injected EC to the nearest true EC of gene g."""
    if g not in G_to_trueE: 
        return None
    return min(ec_distance(e_wrong, e_true) for e_true in G_to_trueE[g])

def split_true_injected(noisy_prior, GE_gold):
    """Partition prior into true vs injected sets."""
    true_set     = {(g,e): p for (g,e),p in noisy_prior.items() if (g,e) in GE_gold}
    injected_set = {(g,e): p for (g,e),p in noisy_prior.items() if (g,e) not in GE_gold}
    return true_set, injected_set

#true_set, injected_set = split_true_injected(noisy_prior, GE_gold)

def spot_check_gene(g, noisy_prior, topn=10):
    """Print a readable summary for one gene g."""
    true_ecs = sorted(G_to_trueE.get(g, []))
    inj = [(e, noisy_prior[(g,e)], nearest_true_distance(g, e))
           for (gg,e) in noisy_prior.keys() if gg == g and (gg,e) not in GE_gold]
    inj.sort(key=lambda x: (-x[1], x[2]))  # highest prob first, then smaller distance

    print(f"\n=== Gene {g} ===")
    print(f"True ECs ({len(true_ecs)}): {true_ecs}")
    print(f"Injected ECs ({len(inj)} total). Top {min(topn,len(inj))} by prior p:")
    for e,p,d in inj[:topn]:
        print(f"  e_wrong={e:>10s}  p={p:0.3f}  d_to_nearest_true={d}")

    # small distance histogram (how local the noise was for this gene)
    dh = Counter(d for _,_,d in inj if d is not None)
    if dh:
        print("Distance histogram (d -> count):", dict(sorted(dh.items())))
    else:
        print("No injected links for this gene.")

def spot_check_random_genes(noisy_prior, k=5, seed=0):
    """Pick k genes that actually have injected links and show them."""
    random.seed(seed)
    genes_with_inj = sorted({g for (g,e) in injected_set.keys()})
    sample = random.sample(genes_with_inj, min(k, len(genes_with_inj)))
    for g in sample:
        spot_check_gene(g, noisy_prior, topn=10)

#spot_check_random_genes(noisy_prior, k=5)

def debug_one_gene(g, G_to_trueE, EC_pool, sigma_ec=SIGMA_EC):
    for e_true in sorted(G_to_trueE[g]):
        # build candidate set and weights
        cands = [e for e in EC_pool if e != e_true]
        dists = [ec_distance(e_true, e) for e in cands]
        weights = [math.exp(-(d*d)/(2*sigma_ec*sigma_ec)) for d in dists]
        total = sum(weights) or 1.0
        probs = [w/total for w in weights]

        # summarize by distance
        from collections import defaultdict
        by_d = defaultdict(list)
        for e,d,p in zip(cands, dists, probs):
            by_d[d].append(p)

        print(f"\nGene {g} true EC {e_true}")
        for d in sorted(by_d):
            ps = by_d[d]
            print(f"  d={d}: n={len(ps)}  mean={sum(ps)/len(ps):.5f}  max={max(ps):.5f}")

#debug_one_gene('129642', G_to_trueE, EC_pool)
#debug_one_gene('4538', G_to_trueE, EC_pool)
# spot_check_random_genes(noisy_prior, k=10)

def _ec_prefix_tuple(ec: str, depth: int):
    levels = [x for x in ec_levels(ec) if isinstance(x, int)]
    return tuple(levels[:depth])

def build_true_map(GE_gold):
    G_to_trueE = defaultdict(set)
    for g, e in GE_gold:
        G_to_trueE[g].add(e)
    return G_to_trueE

## Checks for Agnostic Prior ##
def validate_agnostic_prior(prior, GE_gold, EC_pool, lcp_depth=2, verbose=True):
    EC_pool = set(EC_pool)
    G_to_trueE = build_true_map(GE_gold)
    gold_set = set(GE_gold)

    # Pre-index EC_pool by prefix for fast lookup
    buckets_by_prefix = defaultdict(list)
    for e in EC_pool:
        if ec_is_leaf(e):
            buckets_by_prefix[_ec_prefix_tuple(e, lcp_depth)].append(e)

    # 1) Sanity: only ECs from EC_pool and only leafs
    bad_pool = [(g,e,p) for (g,e),p in prior.items() if e not in EC_pool]
    bad_leaf = [(g,e,p) for (g,e),p in prior.items() if not ec_is_leaf(e)]
    assert not bad_pool, f"Found {len(bad_pool)} (g,E) not in EC_pool"
    assert not bad_leaf, f"Found {len(bad_leaf)} (g,E) that are not leaf ECs"

    # 2) Classify per gold link: unselected vs selected (uniform bucket)
    selected_links = {}   
    unselected = 0

    for (g, e_true) in gold_set:
        p_true = prior.get((g, e_true), None)
        if p_true is None:
            raise AssertionError(f"Missing gold link {(g,e_true)} in prior")

        if abs(p_true - 1.0) < 1e-12:
            unselected += 1
            continue

        # Construct the expected bucket and verify uniformity
        pref   = _ec_prefix_tuple(e_true, lcp_depth)
        bucket = buckets_by_prefix.get(pref, [])
        if not bucket:
            raise AssertionError(f"No bucket candidates found in EC_pool for {e_true} at depth {lcp_depth}")

        # All bucket members must exist for this gene with the same p
        ps = []
        missing = []
        for e_prime in bucket:
            p = prior.get((g, e_prime), None)
            if p is None:
                missing.append(e_prime)
            else:
                ps.append(p)

        if missing:
            raise AssertionError(f"For selected {(g,e_true)} missing {len(missing)} bucket members: {missing[:5]} ...")

        # Uniform check
        p0 = ps[0]
        if any(abs(p - p0) > 1e-10 for p in ps):
            raise AssertionError(f"Non-uniform probs for {(g,e_true)}: min={min(ps):.6f}, max={max(ps):.6f}")

        # p should be 1/|bucket|
        expected = 1.0 / len(bucket)
        if abs(p0 - expected) > 1e-10:
            raise AssertionError(f"Wrong uniform value for {(g,e_true)}: got {p0:.6f}, expected {expected:.6f} (|B|={len(bucket)})")
        selected_links[(g, e_true)] = {"p": p0, "bucket": list(bucket)}

    # 3) More Info
    if verbose:
        n_gold = len(gold_set)
        n_sel  = len(selected_links)
        bucket_hist = Counter(len(v["bucket"]) for v in selected_links.values())
        injected_non_gold = sum(1 for (g,e) in prior.keys() if (g,e) not in gold_set)
        print(f"[validate §4.1] gold links: {n_gold} | selected: {n_sel} | unselected: {unselected}")
        print(f"[validate §4.1] bucket size histogram (over selected): {dict(sorted(bucket_hist.items()))}")
        print(f"[validate §4.1] total prior entries: {len(prior)} | injected (non-gold): {injected_non_gold}")

    return selected_links

#validated_buckets = validate_agnostic_prior(agnostic_prior, GE_gold, EC_pool)

#endregion

#region INSPECTING RESULTING PATHWAY GRAPH
# print("\n=== Build summary ===")
# print(f"Reactions in graph: {len(idx['all_reactions'])}")
# print(f"Accepted compounds: {len(idx['accepted_compounds'])}")
# print(f"Adjacency entries:  {sum(len(v) for v in idx['adj'].values())} (undirected counts)")

# print("\n=== Degree stats ===")
# deg_vals = list(idx['degree'].values())
# print(f"min/median/max degree: {min(deg_vals) if deg_vals else 0} / "
#       f"{(sorted(deg_vals)[len(deg_vals)//2] if deg_vals else 0)} / "
#       f"{max(deg_vals) if deg_vals else 0}")
# print(f"terminal reactions: {len(idx['terminal_reactions'])}")
# print(f"filler reactions:   {len(idx['filler_reactions'])}")

# print("\n=== Map sizes ===")
# print(f"E_to_R enzymes: {len(idx['E_to_R'])}")
# print(f"R_to_E reactions with EC: {len(idx['R_to_E'])}")
# print(f"E_to_G enzymes: {len(idx['E_to_G'])}")

# # Unique reactions coming from RCR
# R_from_rcr = set(rcr_filtered['R']).union(set(rcr_filtered['R2']))
# print("Unique reactions in RCR:", len(R_from_rcr))

# # Unique reactions from RE (have an EC mapping)
# R_from_re = set(re_filtered['R'])
# print("Unique reactions in RE:", len(R_from_re))

# # How many R are in both?
# print("Overlap RCR ∩ RE:", len(R_from_rcr & R_from_re))

# # Reactions in graph but missing ECs?
# R_in_graph = set(idx['all_reactions'])
# print("In graph but not in RE (no EC):", len(R_in_graph - R_from_re))

# allC = set(rcr_filtered['C'])
# banC = set(banned_df['C'])
# accC = allC - banC
# print("Compounds total:", len(allC), "| banned:", len(banC), "| accepted:", len(accC))

# EC_RE = set(re_filtered['EC'])
# EC_GE = set(ge_filtered['EC'])
# print("Orphan GE's:", len(EC_RE-EC_GE), len(EC_GE-EC_RE))

# print("R with EC but not in graph:", len(set(re_filtered['R']) - set(idx['all_reactions'])))

# # 0) Normalize R & EC columns
# re = re_filtered.copy()
# re['R']  = re['R'].astype(str).str.strip()
# re['EC'] = re['EC'].astype(str).str.strip()

# rcr = rcr_filtered.copy()
# rcr['R']  = rcr['R'].astype(str).str.strip()
# rcr['R2'] = rcr['R2'].astype(str).str.strip()
# rcr['C']  = rcr['C'].astype(str).str.strip()

q2_ranked = rank_ge_by_q2_support(ge_filtered, idx)
q2_ranked_GG = rank_ge_by_q2_gene_paths(ge_filtered, idx, exclude_same_gene=False, exclude_same_ec=False)
#print(len(q2_ranked), len(q2_ranked_GG))
#print(q2_ranked_GG[['EC','q2_paths']].drop_duplicates().head(15))

#endregion 

#region initial estimate (before Problog)
# Discussing the Noise Model for Unreliable Predictions, they say:
#   When using the initial (perturbed) estimate for the gene-enzyme link we achieve an AUCPR of 0.69
# At the same time, they don't indicate what the probability of (gene, true enzyme) pair should be
# So we will use the following function to find an initial probability that achieves similar AUCPR on our data
def compute_auprc_from_prior(noisy_prior, GE_gold):
    gold = set(GE_gold)

    # Build label/score lists over the keys of noisy_prior
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

    auprc = sum_prec_at_pos / n_pos  # mean of precision@k over all positive ranks

    return auprc

# initial_auprc = compute_auprc_from_prior(noisy_prior, GE_gold)
# print("\n Testing AUCPR of noisy prior:", initial_auprc)
#endregion

#region CREATING PROBLOG FILES
# Get queries from agnostic model
def queries_from_perturbations_str(perturbations, which="query2", include_true=True, include_injected=True):
    # allow which to be a single string or a list/tuple
    which_list = (which,) if isinstance(which, str) else tuple(which)

    lines = []
    for rec in perturbations:
        g = rec["g"]
        ecs = []
        if include_true:
            ecs.extend(rec["true_in_bucket"])
        if include_injected:
            ecs.extend(rec["injected_in_bucket"])
        # de-dup per record to keep it simple
        ecs = sorted(set(ecs))
        for e in ecs:
            for q in which_list:
                lines.append(f"query({q}({g_atom(g)},{ec_atom(e)})).")

    return "\n".join(lines)

# perturbed = queries_from_perturbations_str(perturbations, which="query1")
# print(perturbed)

def true_pairs_for_perturbed_genes(GE_gold, perturbations):
    """
    All ground-truth (g,e) for genes that were perturbed,
    including true ECs that may lie OUTSIDE the affected bucket(s).
    """
    T = set(GE_gold)
    pert_g = set(sorted({rec['g'] for rec in perturbations}))
    return sorted([(g,e) for (g,e) in T if g in pert_g])

# true_pairs_for_perturbed_genes_list = true_pairs_for_perturbed_genes(GE_gold, perturbations)
# for truep in true_pairs_for_perturbed_genes_list:
#     print(truep)

# print('\n\n', perturbed) 

# Convert relational data into a graph and calculate automorphism orbits
# This gives us our equivalence classes
def compute_automorphism_orbits(
    prior,                   # dict {(G,E) -> p}  
    rcr_df,                  # DataFrame  ['R','C','R2']
    re_df,                   # DataFrame  ['R','EC']
    accepted_compounds,      # compounds
    ortholog_df=None,        # optional ['G','G2']
    p_round=6                # round probabilities to p_round decimal places
):
    """
    Build a typed, labeled graph from your ground facts and compute strict
    equivalence classes as automorphism orbits. Returns:
      - orbit_of: dict { 'R:...'/ 'G:...'/ 'E:...'/ 'C:...' -> orbit_id (int) }
    """

    COLOR = {
        "R": 1,
        "C_accept": 2,
        "C_other": 3,
        "E": 4,
        "G": 5,
        # relation label nodes start at 100+; function's color encodes p-bin
        "L_react": 100,
        "L_rxnEnz": 101,
        "L_orth": 102,
    }

    names = []      # igraph vertex "name" attribute
    colors = []     # igraph vertex color partition (list of ints)
    v_index = {}    # map "typed name" -> vertex id

    def add_vertex(key, color_id):
        """Create a vertex if missing; key is a unique string like 'R:R1234'."""
        if key in v_index:
            return v_index[key]
        vid = len(names)
        v_index[key] = vid
        names.append(key)
        colors.append(color_id)
        return vid

    def add_entity(kind, raw):
        """
        Add/return a vertex id for an entity:
          kind ∈ {'R','C','E','G'}
          raw  = original id/string
        Compounds get split by accept flag to two colors.
        """
        if kind == "C":
            color = COLOR["C_accept"] if raw in accepted_compounds else COLOR["C_other"]
        else:
            color = COLOR[kind]
        return add_vertex(f"{kind}:{raw}", color)

    def func_label_color(p):
        """
        Encode the function(G,E) probability into the label node color.
        Different rounded p's => different colors => cannot be swapped.
        """
        # bin by rounding to p_round decimals (consistent with write_single_problog)
        p_key = int(round(float(round(p, p_round)) * (10 ** p_round)))
        return 1100000 + p_key  # large base so bins don't collide with other colors


    # --- 1) First pass: ensure all entity nodes exist -----------------------------
    # From RCR (reactions & compounds)
    for R1, C, R2 in rcr_df[['R','C','R2']].itertuples(index=False):
        add_entity("R", R1); add_entity("R", R2); add_entity("C", C)

    # From RE (reactions & enzymes)
    for R, E in re_df[['R','EC']].itertuples(index=False):
        add_entity("R", R); add_entity("E", E)

    # From function prior (genes & enzymes)
    for (G, E), p in prior.items():
        add_entity("G", G); add_entity("E", E)

    # From orthologs (genes)
    if ortholog_df is not None and not ortholog_df.empty:
        for G, G2 in ortholog_df[['G','G2']].itertuples(index=False):
            add_entity("G", G); add_entity("G", G2)

    # --- 2) Edges: encode each fact X rel Y as X — L_rel — Y ---------------------
    edges = []

    def add_gadget(u_vid, label_color, v_vid):
        """Insert a degree-2 label node between u and v: u--L--v with L colored."""
        L_vid = add_vertex(f"L:{len(names)}", label_color)  # unique label-node key
        edges.append((u_vid, L_vid))
        edges.append((L_vid, v_vid))

    # react edges from RCR: split (R1,C,R2) into two R--C links
    for R1, C, R2 in rcr_df[['R','C','R2']].itertuples(index=False):
        r1 = add_entity("R", R1); c = add_entity("C", C); r2 = add_entity("R", R2)
        add_gadget(r1, COLOR["L_react"], c)
        add_gadget(r2, COLOR["L_react"], c)

    # reaction_enzyme edges
    for R, E in re_df[['R','EC']].itertuples(index=False):
        r = add_entity("R", R); e = add_entity("E", E)
        add_gadget(r, COLOR["L_rxnEnz"], e)

    # function(G,E) edges with p-aware label colors
    for (G, E), p in prior.items():
        g = add_entity("G", G); e = add_entity("E", E)
        add_gadget(g, func_label_color(p), e)

    # ortholog edges (undirected; one gadget per pair is enough)
    if ortholog_df is not None and not ortholog_df.empty:
        for G, G2 in ortholog_df[['G','G2']].itertuples(index=False):
            g1 = add_entity("G", G); g2 = add_entity("G", G2)
            add_gadget(g1, COLOR["L_orth"], g2)

    # --- 3) Build graph and compute automorphism orbits --------------------------
    n = len(names)
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v); adj[v].append(u)

    # build vertex-color partition
    by = defaultdict(list)
    for vid, c in enumerate(colors):
        by[c].append(vid)
    # partition = [by[c] for c in sorted(by)]
    partition = [set(by[c]) for c in sorted(by)]

    Gp = pn.Graph(number_of_vertices=n, adjacency_dict=adj, vertex_coloring=partition, directed=False)
    result = pn.autgrp(Gp)

    # ## TESTING ##
    # n = Gp.number_of_vertices
    # avg_deg = 45336 / n
    # print("n =", n, "avg degree ≈", avg_deg)
    # # Verify coloring covers all vertices exactly once
    # sizes = [len(c) for c in partition]
    # print("sum(colors) =", sum(sizes), "== n?", sum(sizes) == n)
    # print("largest class =", max(sizes))
    # deg = {v: len(adj.get(v, ())) for v in range(n)}
    # print("min/max/total degree:", min(deg.values()), max(deg.values()), sum(deg.values()))
    # # 2) Coloring sanity
    # print("num color classes:", len(partition), "first sizes:", [len(c) for c in partition[:10]])
    # leaf = next(v for v, d in deg.items() if d == 1)
    # hub  = next(v for v, d in deg.items() if d == 334)
    # # print(leaf, hub)
    # # key_of = {vid: key for key, vid in v_index.items()}
    # # print("leaf key:", key_of.get(leaf))
    # # print("hub  key:", key_of.get(hub))
    # ## END TESTING ##

    labels = result[3]

    keep = [k for k in v_index if not k.startswith("L:")]
    n_before = len(keep)
    n_after  = len({ int(labels[v_index[k]]) for k in keep })
    print("before:", n_before, "after:", n_after, "reduction:", n_before - n_after)

    entity_orbits = {}
    for key, vid in v_index.items():
        if not key.startswith("L:"):
            entity_orbits[key] = int(labels[vid])

    return entity_orbits

def write_single_problog(
    out_path: str,
    prior: dict,                      # {(G,E)->p}
    rcr_df: pd.DataFrame,             # columns ['R','C','R2'] (C = bare CHEBI number string)
    re_df: pd.DataFrame,              # columns ['R','EC']
    accepted_compounds: set,          # from idx['accepted_compounds']
    ortholog_df: Optional[pd.DataFrame] = None,   
    targets: Optional[List[Tuple[str, str]]] = None 
):
    lines = []

    lines += [
        "% ==============================================================\n",
        "% Q3 single-file corpus (facts + rules)\n",
        "% Only 'function/2' is probabilistic; others are deterministic.\n",
        "% Ground relations included (as requested):\n",
        "%   reaction_compound_reaction(R1,C,R2)\n",
        "%   reaction_enzyme(R,E)\n",
        "%   function(G,E)\n",
        "%   ortholog(G1,G2)\n",
        "%   accept_compound(C)\n",
        "% Rules included (as requested):\n",
        "%   reaction(R1,R2) :- reaction_compound_reaction(R1,C,R2), accept_compound(C).\n",
        "%   (plus symmetric rule to make reaction undirected)\n",
        "%   enzyme_reaction_path(G1,E1,E2,G2) :- function(G1,E1), reaction_enzyme(R1,E1),\n",
        "%       reaction(R1,R2), reaction_enzyme(R2,E2), function(G2,E2).\n",
        "%   ortholog_support(G,E) :- ortholog(G,G2), function(G2,E).\n",
        "%   query3(G,E) :- enzyme_reaction_path(G,E,E2,G2), ortholog_support(G,E), ortholog_support(G2,E2).\n",
        "% ==============================================================\n\n"
    ]

    lines.append("% --- Probabilistic gene-enzyme priors ---\n")
    for (G,E), p in sorted(prior.items(), key=lambda kv:(kv[0][0], kv[0][1])):
        lines.append(f"{p:.6f}::function({g_atom(G)},{ec_atom(E)}).\n")
    lines.append("\n")

    lines.append("% --- Acceptable compounds (used to gate RCR edges) ---\n")
    for C in sorted(accepted_compounds, key=str):
        lines.append(f"accept_compound({c_atom(C)}).\n")
    lines.append("\n")

    lines.append("% --- Reaction-Compound-Reaction edges (raw, undirected in facts) ---\n")
    seen = set()
    for R1, C, R2 in rcr_df[['R','C','R2']].itertuples(index=False):
        key = (str(R1), str(C), str(R2))
        if key in seen: 
            continue
        seen.add(key)
        lines.append(f"reaction_compound_reaction({r_atom(R1)},{c_atom(C)},{r_atom(R2)}).\n")
        lines.append(f"reaction_compound_reaction({r_atom(R2)},{c_atom(C)},{r_atom(R1)}).\n")
    lines.append("\n")

    lines.append("% --- Reaction -> Enzyme assignments ---\n")
    for R, E in re_df[['R','EC']].itertuples(index=False):
        lines.append(f"reaction_enzyme({r_atom(R)},{ec_atom(E)}).\n")
    lines.append("\n")

    if ortholog_df is not None and not ortholog_df.empty:
        lines.append("% --- Ortholog pairs ---\n")
        for G, G2 in ortholog_df[['G','G2']].itertuples(index=False):
            lines.append(f"ortholog({g_atom(G)},{g_atom(G2)}).\n")
        lines.append("\n")
    else:
        lines.append("% --- Ortholog pairs ---\n% (none provided; Q3 will evaluate to 0 without orthologs)\n\n")
    
    lines += [
        "% ================= RULES =================\n",
        "% Make reaction one-hop edges filtered by accepted compounds.\n",
        "reaction(R1,R2) :- reaction_compound_reaction(R1,C,R2), accept_compound(C).\n",
        "% Symmetry is already encoded by emitting both directions above; this rule suffices.\n",
        "\n",
        "% Enzyme-reaction path (one-hop) with probabilistic endpoints.\n",
        "enzyme_reaction_path(G1,E1,E2,G2) :-\n",
        "    function(G1,E1),\n",
        "    reaction_enzyme(R1,E1),\n",
        "    reaction(R1,R2),\n",
        "    reaction_enzyme(R2,E2),\n",
        "    function(G2,E2).\n",
        "\n",
        "% Ortholog support: gene has ortholog whose function supports the enzyme.\n",
        "ortholog_support(G,E) :- ortholog(G,G2), function(G2,E).\n",
        "\n",
        "% Q1: direct function + at least one reaction annotated with E exists.\n",
        "query1(G,E) :-\n",
        "    function(G,E),\n",
        "    reaction_enzyme(R,E).\n",
        "\n",
        "% Q2: exists a one-hop enzyme-reaction path starting at (G,E).\n",
        "query2(G,E) :-\n",
        "    enzyme_reaction_path(G,E,E2,G2).\n",
        "\n",
        "% Q3 (most complex): Q2 plus ortholog support on both endpoints.\n",
        "query3(G,E) :-\n",
        "    enzyme_reaction_path(G,E,E2,G2),\n",
        "    ortholog_support(G,E),\n",
        "    ortholog_support(G2,E2).\n",
        "% ==========================================\n\n"
    ]

    lines.append("% --- Add your queries below (examples):\n")
    lines.append("% query(query3(g12345, ec_3_1_1_4)).\n\n")

    Path(out_path).write_text(''.join(lines), encoding='utf-8')
    print(f"Wrote single-file ProbLog corpus -> {out_path} (lines: {len(lines)})")

# Calculate Equivalences
entity_orbits = compute_automorphism_orbits(
    prior=noisy_prior,
    rcr_df=rcr_filtered[['R','C','R2']].copy(),
    re_df=re_filtered[['R','EC']].copy(),
    accepted_compounds=idx['accepted_compounds'],
    ortholog_df=None,   
    p_round=6
)

print(f"Type of entity_orbits: {type(entity_orbits)}")
print(f"size of entity_orbits: {len(entity_orbits)}")


# Count number of orbits per kind
def counts_by_kind(entity_orbits):
    kind_to_orbits = defaultdict(set)
    for key, oid in entity_orbits.items():
        kind = key.split(':', 1)[0]         # e.g., 'G'
        kind_to_orbits[kind].add(int(oid))  # merge by orbit id

    return {kind: len(orbits) for kind, orbits in kind_to_orbits.items()}

counts = counts_by_kind(entity_orbits)
print("unique per kind:", counts)
print("total unique entities:", sum(counts.values()))

# Write out to Problog File
# write_single_problog(
#     out_path="NData/q3_single_test.pl",
#     prior=noisy_prior,
#     #prior=agnostic_prior,
#     rcr_df=rcr_filtered[['R','C','R2']].copy(),
#     re_df=re_filtered[['R','EC']].copy(),
#     accepted_compounds=idx['accepted_compounds'],
#     ortholog_df=None,        
#     targets=None            
# )

