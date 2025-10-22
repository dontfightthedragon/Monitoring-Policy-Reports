#!/usr/bin/env python3
# analysis/TFIDF/apply_labels_and_plots.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT   = Path(__file__).resolve().parents[2]
TABDIR = ROOT / "reports" / "tables"
FIGDIR = ROOT / "reports" / "figures"; FIGDIR.mkdir(parents=True, exist_ok=True)

prev   = pd.read_csv(TABDIR / "tfidf_cluster_prevalence.csv")         # cluster, share_projects
ts     = pd.read_csv(TABDIR / "tfidf_cluster_timeseries.csv")         # year, tfidf_cluster, share
labels = pd.read_csv(TABDIR / "tfidf_cluster_labels_template.csv")    # cluster, suggested_label

# Fallback: if a label is blank, use the first 2–3 top terms
def fallback_label(row):
    if isinstance(row.get("suggested_label"), str) and row["suggested_label"].strip():
        return row["suggested_label"].strip()
    # compact top-terms fallback
    terms = [t.strip() for t in str(row.get("top_terms","")).split(",") if t.strip()]
    return ", ".join(terms[:3]) if terms else f"Cluster {row['cluster']}"

labels["pretty_label"] = labels.apply(fallback_label, axis=1)
label_map = labels.set_index("cluster")["pretty_label"].to_dict()

# -------- Figure A: share of projects by cluster (top 12) --------
top = (prev.assign(pretty_label=lambda d: d["cluster"].map(label_map))
            .sort_values("share_projects", ascending=False).head(12))

plt.figure(figsize=(11, 6))
bars = plt.bar(top["pretty_label"], top["share_projects"]*100)
for b, v in zip(bars, top["share_projects"]*100):
    plt.text(b.get_x()+b.get_width()/2, b.get_height(), f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
plt.ylabel("% of projects")
plt.title("TF–IDF clusters: share of projects (top 12)")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(FIGDIR / "tfidf_share_by_cluster_labeled.pdf")
plt.close()

#  Figure B: top-6 clusters over time 
ts["pretty_label"] = ts["tfidf_cluster"].map(label_map)
top6_ids = (ts.groupby("tfidf_cluster")["share"].sum()
              .sort_values(ascending=False).head(6).index.tolist())
sub = ts[ts["tfidf_cluster"].isin(top6_ids)].copy()

plt.figure(figsize=(11,6))
for cid, g in sub.groupby("tfidf_cluster"):
    g = g.sort_values("year")
    plt.plot(g["year"], g["share"]*100, marker="o", label=label_map.get(cid, f"Cluster {cid}"))
plt.legend(title="Cluster", fontsize=9)
plt.xlabel("Year"); plt.ylabel("% of projects")
plt.title("TF–IDF clusters over time (top 6)")
plt.tight_layout()
plt.savefig(FIGDIR / "tfidf_top6_timeseries_labeled.pdf")
plt.close()

print("[PLOTS] Wrote:")
print("  - figures/tfidf_share_by_cluster_labeled.pdf")
print("  - figures/tfidf_top6_timeseries_labeled.pdf")
