import json
from pathlib import Path
import pandas as pd

# these two share a Uniprot column
GE_EXPANDED = "NData/GE_expanded.tsv"
SPECIES_UNIPROT = "NData/species_uniprot.json"  

df_ge = pd.read_csv(GE_EXPANDED, sep="\t", dtype=str).fillna("")
df_ge["uniprot"] = df_ge["Entry"].str.strip().str.upper()

txt = Path(SPECIES_UNIPROT).read_text(encoding="utf-8-sig") 
species = json.loads(txt)
df_sp = pd.DataFrame(species)
df_sp["uniprot"] = df_sp["uniprot"].astype(str).str.strip().str.upper()

print("df_ge shape:", df_ge.shape)
print("df_sp shape:", df_sp.shape, '\n')

merged = df_ge.merge(df_sp, on="uniprot", how="left", validate="m:1")




