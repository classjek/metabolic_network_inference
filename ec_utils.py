import re
import json
import pandas as pd
from pathlib import Path
from functools import lru_cache
from collections import defaultdict

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


def _ec_prefix_tuple(ec: str, depth: int):
    """Tuple key of the first `depth` integer EC levels, consistent with ec_atom/ec_levels."""
    levels = [x for x in ec_levels(ec) if isinstance(x, int)]
    return tuple(levels[:depth])

# Use CR-pairs to build RCR pairs on the fly 
# significantly reduces memory usage for large pathways
def load_rcr_from_cr_pairs(cr_pairs_path, Rset, BANNED):
    # for each compound, find all reactions that touch it
    # filter to pathway reactions, then generate RCR triples

    cr_list = json.loads(Path(cr_pairs_path).read_text(encoding="utf-8-sig"))
    
    # Build compound -> [reactions] index
    C_to_Rs = defaultdict(set)
    for d in cr_list:
        C = d['C']
        R = str(d['R'])
        # limit to reactions in pathway of interest
        if R in Rset:
            C_to_Rs[C].add(R)
    
    rcr_rows = []
    for C, reactions in C_to_Rs.items():
        chebi_id = C.split(':')[1] if ':' in C else C
        if chebi_id in BANNED:
            continue
        if len(reactions) < 2:
            continue  # Need at least 2 reactions to form a pair
        
        reactions = list(reactions)
        for i, R1 in enumerate(reactions):
            for R2 in reactions[i+1:]:
                rcr_rows.append({'R': R1, 'C': chebi_id, 'R2': R2})
    
    return pd.DataFrame(rcr_rows)

# Use information in GE_expanded to build ortholog pairs (filtering out paralogs)
def build_ortholog_pairs(ec_expanded, target_species, ortholog_col='OrthoDB', max_group_size=50):
    df = ec_expanded[ec_expanded[ortholog_col].ne('') & ec_expanded['GeneID'].ne('')]
    
    grouped = df.groupby(ortholog_col).apply(
        lambda g: list(zip(g['GeneID'], g['species'])),
        include_groups=False
    )
    
    pairs = []
    for gene_species_list in grouped:
        if len(gene_species_list) < 2 or len(gene_species_list) > max_group_size:
            continue
        # Separate target vs other species genes
        target_genes = [(g, sp) for g, sp in gene_species_list if sp == target_species]
        other_genes = [(g, sp) for g, sp in gene_species_list if sp != target_species]
        # Pair each target gene with each other-species gene
        for g1, _ in target_genes:
            for g2, _ in other_genes:
                pairs.append({'G': g1, 'G2': g2})
    
    return pd.DataFrame(pairs)

def compute_enzyme_pairs(rcr_df, re_df, accepted_compounds):
    """
    Compute enzyme_pair(E1,E2) from pathway connectivity.
    Returns a set of (E1, E2) tuples.
    """
    # Build R -> [E] lookup
    R_to_E = defaultdict(list)
    for _, row in re_df.iterrows():
        R_to_E[row['R']].append(row['EC'])
    
    enzyme_pairs = set()
    for _, row in rcr_df.iterrows():
        R1, C, R2 = row['R'], row['C'], row['R2']
        if C not in accepted_compounds:
            continue
        
        E1s = R_to_E.get(R1, [])
        E2s = R_to_E.get(R2, [])
        
        for e1 in E1s:
            for e2 in E2s:
                enzyme_pairs.add((str(e1), str(e2)))
                # Add reverse direction (undirected)
                enzyme_pairs.add((str(e2), str(e1)))
    
    return enzyme_pairs