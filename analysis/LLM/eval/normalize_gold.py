# analysis/LLM/eval/normalize_gold.py
import pandas as pd, re
from pathlib import Path

G = Path("analysis/LLM/artifacts/gold/gold_150.csv")
df = pd.read_csv(G)

# 1) canonicalize primary_label
canon = {
    "FORESTS_LAND":"FORESTS_LAND","FOREST":"FORESTS_LAND","FOREST_LAND":"FORESTS_LAND",
    "WATER_MARINE":"WATER_MARINE","WATER":"WATER_MARINE","MARINE":"WATER_MARINE",
    "WILDLIFE_SPECIES":"WILDLIFE_SPECIES","WILDLIFE":"WILDLIFE_SPECIES","SPECIES":"WILDLIFE_SPECIES",
    "CROSS_CUTTING":"CROSS_CUTTING","CROSSCUTTING":"CROSS_CUTTING",
    "NONE":"NONE","": "NONE"
}
pl = df["primary_label"].astype(str).str.strip().str.upper()
df["primary_label"] = pl.map(canon).fillna("NONE")

# 2) normalize sdg15_targets → "SDG15.x" (no spaces, deduped)
def norm_targets(s: str) -> str:
    if pd.isna(s) or not str(s).strip(): return ""
    toks = re.split(r"[,\s]+", str(s).strip())
    out = []
    for t in toks:
        t = t.strip().upper()
        if not t: continue
        t = t if t.startswith("SDG15.") else (f"SDG15.{t}" if re.fullmatch(r"15\.[1-9]", t) else t)
        if re.fullmatch(r"SDG15\.(1|2|3|4|5|7|8|9)", t):
            out.append(t)
    return ",".join(sorted(set(out)))
df["sdg15_targets"] = df["sdg15_targets"].map(norm_targets)

# 3) dedupe by activity_id (keep first)
df = df.drop_duplicates("activity_id", keep="first").reset_index(drop=True)

# 4) write back
df.to_csv(G, index=False, encoding="utf-8")
print("gold rows:", len(df))
print("class counts:", df["primary_label"].value_counts().to_dict())
