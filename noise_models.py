import math
import random
from collections import defaultdict, Counter

from ec_utils import ec_is_leaf, ec_distance, ec_levels, _ec_prefix_tuple


# Default noise model parameters (from paper §4.2)
S_FRACTION = 0.50    # s: fraction of genes to corrupt
K_WRONG    = 5       # k: wrong ECs per corrupted gene
SIGMA_EC   = 1.5     # how "local" wrong ECs are in EC tree
SIGMA_N    = 0.125   # Gaussian jitter added to probabilities
BASE_TRUE  = 0.40    # prior for true links before noise


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
    #print(f"Corrupting {n_corrupt} out of {len(genes)} genes (s={s_fraction})")
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
    
    #print(f"[§4.2] Corrupted genes: {n_corrupt} / {len(genes)}")
    #print(f"[§4.2] Injected wrong links (draws): {injected_cnt}")
    if injected_cnt == 0:
        print("[§4.2] No injected links—check EC_pool/GE_gold sizes.")
    else:
        # show how local the injected ECs are (should skew to small distances when σ_EC is small)
        #print("[§4.2] Injected EC distance histogram (d -> count):", dict(sorted(injected_dist.items())))
        # rough share of non-gold links in the prior
        n_gold = len(GE_gold)
        n_total = len(prior)
        #print(f"[§4.2] Prior size: {n_total} entries  (gold {n_gold}, injected {n_total-n_gold})")

    return prior

def build_true_map(GE_gold):
    G_to_trueE = defaultdict(set)
    for g, e in GE_gold:
        G_to_trueE[g].add(e)
    return G_to_trueE

def split_true_injected(noisy_prior, GE_gold):
    """Partition prior into true vs injected sets."""
    true_set     = {(g,e): p for (g,e),p in noisy_prior.items() if (g,e) in GE_gold}
    injected_set = {(g,e): p for (g,e),p in noisy_prior.items() if (g,e) not in GE_gold}
    return true_set, injected_set


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

def nearest_true_distance(g, e_wrong, G_to_trueE):
    """Distance from injected EC to the nearest true EC of gene g."""
    if g not in G_to_trueE: 
        return None
    return min(ec_distance(e_wrong, e_true) for e_true in G_to_trueE[g])

def spot_check_gene(g, noisy_prior, G_to_trueE, GE_gold, topn=10):
    """Print a readable summary for one gene g."""
    true_ecs = sorted(G_to_trueE.get(g, []))
    inj = [(e, noisy_prior[(g,e)], nearest_true_distance(g, e, G_to_trueE))
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

def spot_check_random_genes(noisy_prior, GE_gold, k=5, seed=0):
    """Pick k genes that actually have injected links and show them."""
    random.seed(seed)
    G_to_trueE = build_true_map(GE_gold)
    _, injected_set = split_true_injected(noisy_prior, GE_gold)
    
    genes_with_inj = sorted({g for (g,e) in injected_set.keys()})
    sample = random.sample(genes_with_inj, min(k, len(genes_with_inj)))
    for g in sample:
        spot_check_gene(g, noisy_prior, G_to_trueE, GE_gold, topn=10)

def debug_one_gene(g, G_to_trueE, EC_pool, sigma_ec):
    """Debug the sampling weights for one gene."""
    for e_true in sorted(G_to_trueE[g]):
        # build candidate set and weights
        cands = [e for e in EC_pool if e != e_true]
        dists = [ec_distance(e_true, e) for e in cands]
        weights = [math.exp(-(d*d)/(2*sigma_ec*sigma_ec)) for d in dists]
        total = sum(weights) or 1.0
        probs = [w/total for w in weights]

        # summarize by distance
        by_d = defaultdict(list)
        for e,d,p in zip(cands, dists, probs):
            by_d[d].append(p)

        print(f"\nGene {g} true EC {e_true}")
        for d in sorted(by_d):
            ps = by_d[d]
            print(f"  d={d}: n={len(ps)}  mean={sum(ps)/len(ps):.5f}  max={max(ps):.5f}")