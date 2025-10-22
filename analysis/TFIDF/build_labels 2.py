#!/usr/bin/env python3
# analysis/TFIDF/tfidf_build_labels_clean.py
import re
import pandas as pd
from pathlib import Path

ROOT   = Path(__file__).resolve().parents[2]
TABDIR = ROOT / "reports" / "tables"; TABDIR.mkdir(parents=True, exist_ok=True)

# Inputs produced by your baseline
summary   = pd.read_csv(TABDIR / "tfidf_cluster_summary.csv")          # cluster, share_projects, top_terms, (maybe) majority_llm_label
terms     = pd.read_csv(TABDIR / "tfidf_kmeans_top_terms.csv")         # cluster, top_terms
maj_path  = TABDIR / "tfidf_cluster_majority_llm.csv"                  # tfidf_cluster, majority_llm_label (optional)

#  Stopword / boilerplate lists 
EN_STOP = {
    "the","and","of","for","to","in","on","with","by","from","at","or","as","an","a","be",
    "into","per","over","under","through"
}
FR_STOP = {"de","des","du","les","la","le","et","pour","dans","avec","par","au","aux","en"}
BOILER  = {
    "project","projects","programme","program","programmes","programs",
    "activities","activity","management","support","implementation","contribution",
    "usaid","undp","unicef","adb","world","bank","fund","grant","tc","et","le","la"
}

def clean_terms(s: str) -> list[str]:
    # split on commas produced earlier
    raw = [t.strip() for t in str(s).split(",")]
    out = []
    for t in raw:
        # keep only alphabetic tokens (allow hyphen), length >= 3
        t = t.lower()
        t = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ\- ]", "", t)
        t = t.strip("- ").replace("  "," ")
        if len(t) < 3:
            continue
        if t in EN_STOP or t in FR_STOP or t in BOILER:
            continue
        out.append(t)
    # de-dup preserving order
    seen = set(); cleaned=[]
    for t in out:
        if t not in seen:
            cleaned.append(t); seen.add(t)
    return cleaned

# Join to ensure we have top_terms and majority label
df = summary.merge(terms, on="cluster", how="left", suffixes=("","_dup"))
df["top_terms"] = df["top_terms"].fillna(df.get("top_terms_dup"))
df = df.drop(columns=[c for c in df.columns if c.endswith("_dup")])

if maj_path.exists():
    maj = pd.read_csv(maj_path)
    df = df.merge(maj[["tfidf_cluster","majority_llm_label"]],
                  left_on="cluster", right_on="tfidf_cluster", how="left").drop(columns=["tfidf_cluster"])

# Build readable label: PrimaryLabel — term1, term2 (cleaned)
def make_pretty(row):
    primary = row.get("majority_llm_label")
    terms = clean_terms(row.get("top_terms",""))
    hint  = ", ".join(terms[:3]) if terms else ""
    if isinstance(primary, str) and primary:
        return f"{primary.replace('_','/')} — {hint}" if hint else primary.replace("_","/")
    return hint if hint else f"Cluster {row['cluster']}"

df["pretty_label"] = df.apply(make_pretty, axis=1)

# Save mapping + “pretty” prevalence and timeseries
mapping = df[["cluster","pretty_label"]]
mapping.to_csv(TABDIR / "tfidf_cluster_label_map.csv", index=False)

prev = pd.read_csv(TABDIR / "tfidf_cluster_prevalence.csv")
prev = prev.merge(mapping, on="cluster", how="left")
prev.to_csv(TABDIR / "tfidf_cluster_prevalence_pretty.csv", index=False)

ts = pd.read_csv(TABDIR / "tfidf_cluster_timeseries.csv")
ts = ts.merge(mapping, left_on="tfidf_cluster", right_on="cluster", how="left")
ts.to_csv(TABDIR / "tfidf_cluster_timeseries_pretty.csv", index=False)

print("[TFIDF] Wrote:")
print("  - tfidf_cluster_label_map.csv")
print("  - tfidf_cluster_prevalence_pretty.csv")
print("  - tfidf_cluster_timeseries_pretty.csv")
