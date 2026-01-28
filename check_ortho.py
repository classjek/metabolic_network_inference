import pandas as pd
from collections import Counter

df = pd.read_csv("NData/GE_expanded.tsv", sep="\t", dtype=str).fillna("")

# Clean up columns
df['GeneID'] = df['GeneID'].str.strip().str.rstrip(';').str.upper()
df['species'] = df['Organism (ID)'].astype(str).str.strip()

# Filter to genes with valid GeneID
df = df[df['GeneID'].ne('') & ~df['GeneID'].isin({'NA','N/A','NONE','NULL','NAN'})]

print(f"Total rows: {len(df)}")
print(f"Unique genes: {df['GeneID'].nunique()}")
print(f"Unique species: {df['species'].nunique()}")

# === 1. COVERAGE: How many genes have values? ===
print("\n=== COVERAGE ===")
for col in ['OMA', 'eggNOG', 'OrthoDB']:
    has_value = df[col].ne('').sum()
    pct = 100 * has_value / len(df)
    print(f"{col}: {has_value}/{len(df)} ({pct:.1f}%)")

# === 2. GROUP SIZES: How many genes per group? ===
print("\n=== GROUP SIZE DISTRIBUTION ===")
for col in ['OMA', 'eggNOG', 'OrthoDB']:
    groups = df[df[col].ne('')].groupby(col)['GeneID'].apply(set)
    sizes = [len(g) for g in groups]
    if sizes:
        print(f"{col}:")
        print(f"  Groups: {len(sizes)}")
        print(f"  Mean size: {sum(sizes)/len(sizes):.2f}")
        print(f"  Median size: {sorted(sizes)[len(sizes)//2]}")
        print(f"  Max size: {max(sizes)}")
        print(f"  Singletons (useless): {sum(1 for s in sizes if s == 1)} ({100*sum(1 for s in sizes if s == 1)/len(sizes):.1f}%)")
        print(f"  Size 2-10 (good): {sum(1 for s in sizes if 2 <= s <= 10)}")
        print(f"  Size >100 (too broad?): {sum(1 for s in sizes if s > 100)}")

# === 3. CROSS-SPECIES GROUPS ===
print("\n=== CROSS-SPECIES GROUPS ===")
for col in ['OMA', 'eggNOG', 'OrthoDB']:
    groups = df[df[col].ne('')].groupby(col)['species'].apply(set)
    multi_species = [g for g in groups if len(g) > 1]
    print(f"{col}: {len(multi_species)} groups span multiple species")

# === 4. TRUE ORTHOLOG PAIRS (cross-species only, NO paralogs) ===
print("\n=== TRUE ORTHOLOG PAIRS (cross-species only) ===")

def count_cross_species_pairs(df, col, max_group_size=50):
    """Count only cross-species pairs, excluding same-species paralogs."""
    # Group by ortholog ID, keeping gene and species together
    grouped = df[df[col].ne('')].groupby(col).apply(
        lambda g: list(zip(g['GeneID'], g['species']))
    )
    
    total_pairs = 0
    paralog_pairs = 0
    ortholog_pairs = 0
    
    for gene_species_list in grouped:
        if len(gene_species_list) < 2:
            continue
        if len(gene_species_list) > max_group_size:
            continue  # Skip overly broad groups
            
        for i, (g1, sp1) in enumerate(gene_species_list):
            for g2, sp2 in gene_species_list[i+1:]:
                total_pairs += 1
                if sp1 == sp2:
                    paralog_pairs += 1
                else:
                    ortholog_pairs += 1
    
    return ortholog_pairs, paralog_pairs, total_pairs

for col in ['OMA', 'eggNOG', 'OrthoDB']:
    ortho, para, total = count_cross_species_pairs(df, col)
    print(f"{col}:")
    print(f"  Total pairs (before filtering): {total:,}")
    print(f"  Paralogs (same-species, excluded): {para:,} ({100*para/total if total else 0:.1f}%)")
    print(f"  True orthologs (cross-species): {ortho:,} ({100*ortho/total if total else 0:.1f}%)")

# === 5. SAMPLE TRUE ORTHOLOG PAIRS ===
print("\n=== SAMPLE TRUE ORTHOLOG PAIRS ===")

def get_sample_ortholog_pairs(df, col, n_samples=5, max_group_size=20):
    """Get sample cross-species ortholog pairs."""
    grouped = df[df[col].ne('')].groupby(col).apply(
        lambda g: list(zip(g['GeneID'], g['species'], g['Organism']))
    )
    
    pairs = []
    for gene_species_list in grouped:
        if len(gene_species_list) < 2 or len(gene_species_list) > max_group_size:
            continue
        for i, (g1, sp1, org1) in enumerate(gene_species_list):
            for g2, sp2, org2 in gene_species_list[i+1:]:
                if sp1 != sp2:  # Cross-species only
                    pairs.append((g1, org1, g2, org2))
                    if len(pairs) >= n_samples:
                        return pairs
    return pairs

for col in ['OMA', 'eggNOG', 'OrthoDB']:
    print(f"\n{col} sample pairs:")
    samples = get_sample_ortholog_pairs(df, col)
    for g1, org1, g2, org2 in samples[:3]:
        print(f"  {g1} ({org1[:20]}...) <-> {g2} ({org2[:20]}...)")

# === 6. SUMMARY ===
print("\n" + "="*60)
print("SUMMARY: Which database to use?")
print("="*60)
for col in ['OMA', 'eggNOG', 'OrthoDB']:
    ortho, para, total = count_cross_species_pairs(df, col)
    coverage = df[col].ne('').sum() / len(df) * 100
    paralog_pct = 100*para/total if total else 0
    print(f"{col:10} | Coverage: {coverage:5.1f}% | Orthologs: {ortho:>7,} | Paralog noise: {paralog_pct:4.1f}%")