import pynauty as pn
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Optional, List, Tuple

from ec_utils import r_atom, g_atom, ec_atom, c_atom

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

def true_pairs_for_perturbed_genes(GE_gold, perturbations):
    """
    All ground-truth (g,e) for genes that were perturbed,
    including true ECs that may lie OUTSIDE the affected bucket(s).
    """
    T = set(GE_gold)
    pert_g = set(sorted({rec['g'] for rec in perturbations}))
    return sorted([(g,e) for (g,e) in T if g in pert_g])

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

def counts_by_kind(entity_orbits):
    kind_to_orbits = defaultdict(set)
    for key, oid in entity_orbits.items():
        kind = key.split(':', 1)[0]         # e.g., 'G'
        kind_to_orbits[kind].add(int(oid))  # merge by orbit id

    return {kind: len(orbits) for kind, orbits in kind_to_orbits.items()}