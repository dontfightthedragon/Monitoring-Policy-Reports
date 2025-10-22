#!/usr/bin/env python3
# analysis/TFIDF/make_cluster_review_pack.py
import pandas as pd
from pathlib import Path

ROOT   = Path(__file__).resolve().parents[2]
TABDIR = ROOT / "reports" / "tables"; TABDIR.mkdir(parents=True, exist_ok=True)

# Inputs from your pipeline
labels = pd.read_parquet(TABDIR / "tfidf_kmeans_labels.parquet")   # activity_id, year, tfidf_cluster
terms  = pd.read_csv(TABDIR / "tfidf_kmeans_top_terms.csv")        # cluster, top_terms
prev   = pd.read_csv(TABDIR / "tfidf_cluster_prevalence.csv")      # cluster, share_projects

# Load a text source for examples (pick the one you used)
# If you used the normalized English file:
DATA = ROOT / "dataset/temporaryFileToWorkWith/CRS_enfr_language_normalized_use_this.parquet"
df = pd.read_parquet(DATA)[["activity_id","year","donor_name","recipient_name","text_en","text_original"]]

# Join to get examples per cluster
j = labels.merge(df, on=["activity_id","year"], how="left")
j["text_preview"] = j["text_en"].fillna(j["text_original"]).astype(str).str.slice(0, 260).str.replace("\n"," ")

# One CSV per cluster with 25 examples to read quickly
out_dir = TABDIR / "tfidf_cluster_samples"; out_dir.mkdir(exist_ok=True, parents=True)
for c, g in j.groupby("tfidf_cluster"):
    g.sort_values("activity_id").head(25)[
        ["activity_id","donor_name","recipient_name","text_preview"]
    ].to_csv(out_dir / f"cluster_{c:02d}_samples.csv", index=False)

# A single template for labeling
label_template = (prev.merge(terms, left_on="cluster", right_on="cluster", how="left")
                     .sort_values("share_projects", ascending=False))
label_template["suggested_label"] = ""  # you will fill this manually
label_template.to_csv(TABDIR / "tfidf_cluster_labels_template.csv", index=False)

print("[REVIEW PACK] Wrote:")
print("  - reports/tables/tfidf_cluster_samples/cluster_XX_samples.csv  (per-cluster examples)")
print("  - reports/tables/tfidf_cluster_labels_template.csv              (fill in your labels)")
