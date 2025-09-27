# analysis/LLM/eval/compute_metrics.py
import argparse, pandas as pd, numpy as np
from sklearn.metrics import classification_report, f1_score, jaccard_score

VALID = {"FORESTS_LAND","WATER_MARINE","WILDLIFE_SPECIES","CROSS_CUTTING","NONE"}
MAP = {
    "FOREST":"FORESTS_LAND","FOREST_LAND":"FORESTS_LAND",
    "WATER":"WATER_MARINE","MARINE":"WATER_MARINE",
    "WILDLIFE":"WILDLIFE_SPECIES","SPECIES":"WILDLIFE_SPECIES",
    "CROSSCUTTING":"CROSS_CUTTING",
}

def canon_label(s: str) -> str:
    if not isinstance(s, str): return "NONE"
    t = "".join(s.split()).upper()  # drop all spaces (incl. non-breaking), uppercase
    t = MAP.get(t, t)
    return t if t in VALID else "NONE"

def explode_targets(s):
    if pd.isna(s) or s=="":
        return []
    return [t.strip() for t in str(s).split(",") if t.strip()]

def binarize_targets(series, classes):
    M = np.zeros((len(series), len(classes)), dtype=int)
    for i, s in enumerate(series):
        for t in explode_targets(s):
            if t in classes: M[i, classes.index(t)] = 1
    return M

ap = argparse.ArgumentParser()
ap.add_argument("--gold", required=True)
ap.add_argument("--pred", required=True)
args = ap.parse_args()

g = pd.read_csv(args.gold)[["activity_id","primary_label","sdg15_targets"]].copy()
p = pd.read_csv(args.pred)[["activity_id","llm_primary_label","llm_sdg15_targets"]].copy()

# --- normalize labels
g["primary_label"]     = g["primary_label"].map(canon_label)
p["llm_primary_label"] = p["llm_primary_label"].map(canon_label)

m = g.merge(p, on="activity_id", how="inner")

print("== sanity ==")
print("gold classes:", sorted(m["primary_label"].unique().tolist()))
print("pred classes:", sorted(m["llm_primary_label"].unique().tolist()))
print("rows:", len(m))

print("\n== primary_label ==")
print(classification_report(
    m["primary_label"], m["llm_primary_label"], digits=3, zero_division=0
))

# multi-label targets
CLZ = [f"SDG15.{i}" for i in [1,2,3,4,5,7,8,9]]
Y_true = binarize_targets(m["sdg15_targets"], CLZ)
Y_pred = binarize_targets(m["llm_sdg15_targets"], CLZ)

print("== sdg15_targets ==")
print("micro-F1:", f1_score(Y_true, Y_pred, average="micro", zero_division=0))
print("macro-F1:", f1_score(Y_true, Y_pred, average="macro", zero_division=0))
print("micro-Jaccard:", jaccard_score(Y_true, Y_pred, average="micro", zero_division=0))
print("macro-Jaccard:", jaccard_score(Y_true, Y_pred, average="macro", zero_division=0))
