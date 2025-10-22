#!/usr/bin/env python3
# analysis/01_tfidf_baseline.py
# TF–IDF + k-means baseline focused on classification (project-level metrics)

from __future__ import annotations
import argparse, numpy as np, pandas as pd
from pathlib import Path
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
# Make repository root importable when running by path
_FILE = Path(__file__).resolve()
_REPO_ROOT = _FILE.parents[2]
try:
    from analysis.Other.common import load_textprep, ensure_outdir, seed_all
except ModuleNotFoundError:
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from analysis.Other.common import load_textprep, ensure_outdir, seed_all

ROOT = _REPO_ROOT
TABDIR = ROOT / "reports" / "tables"; ensure_outdir(str(TABDIR))

# Optional inputs produced elsewhere (used if present)
KEYWORD_LABELS = TABDIR / "sdg_rule_labels_per_project.parquet"  # activity_id, sdg_target,...
LLM_PRIMARY    = TABDIR / "llm_primary_labels.parquet"           # activity_id, primary_label (FORESTS/LAND, ...)

def get_text(df: pd.DataFrame, both_fields: bool) -> pd.Series:
    # Prefer normalized English, optionally append original
    if "text_en" in df.columns:
        s = df["text_en"].fillna("").astype(str)
        if both_fields and "text_original" in df.columns:
            s = (s + " " + df["text_original"].fillna("").astype(str)).str.strip()
        return s
    # Fallbacks
    for c in ("text_for_nlp","text"):
        if c in df.columns:
            return df[c].fillna("").astype(str)
    raise KeyError("No usable text column found (expected text_en/text_original or text_for_nlp/text).")

def dedupe_longest(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    if df["activity_id"].duplicated().any():
        tlen = df[text_col].str.len()
        df = df.assign(_tlen=tlen).sort_values("_tlen", ascending=False)
        df = df.drop_duplicates(subset="activity_id", keep="first").drop(columns="_tlen")
    return df

def main():
    ap = argparse.ArgumentParser(description="TF–IDF + k-means baseline (classification-first).")
    ap.add_argument("--input", type=str,
        default=str(ROOT / "dataset/temporaryFileToWorkWith/CRS_textprep_FINAL.parquet"))
    ap.add_argument("--k", type=int, default=20, help="Number of clusters.")
    ap.add_argument("--min-df", type=int, default=30, help="Min doc freq for TF–IDF.")
    ap.add_argument("--max-df", type=float, default=0.90, help="Max doc freq fraction for TF–IDF.")
    ap.add_argument("--max-feats", type=int, default=40000, help="Max TF–IDF features.")
    ap.add_argument("--bigrams", action="store_true", help="Use bigrams (1–2-grams) instead of unigrams only.")
    ap.add_argument("--mbkm", action="store_true", help="Use MiniBatchKMeans (memory/speed).")
    ap.add_argument("--both-fields", action="store_true", help="Append text_original to text_en.")
    args = ap.parse_args()

    seed_all(1)

    # ---------- Load base
    df = load_textprep(args.input)
    txt = get_text(df, both_fields=args.both_fields)
    df = df.assign(text_for_nlp=txt)
    df = dedupe_longest(df, "text_for_nlp")

    # ---------- Vectorize (TF–IDF)
    ngram = (1,2) if args.bigrams else (1,1)
    vect = TfidfVectorizer(
        ngram_range=ngram,
        min_df=args.min_df,
        max_df=args.max_df,
        max_features=args.max_feats,
        strip_accents="unicode",
        lowercase=True,
        sublinear_tf=True,
    )
    X = vect.fit_transform(df["text_for_nlp"])
    vocab = np.array(vect.get_feature_names_out())

    # ---------- Clustering
    k = args.k
    Clusterer = MiniBatchKMeans if args.mbkm else KMeans
    km = Clusterer(n_clusters=k, random_state=1, n_init="auto" if args.mbkm else 10)
    labels = km.fit_predict(X)

    # Labels per project
    lab = df[["activity_id","year"]].copy()
    lab["tfidf_cluster"] = labels
    lab.to_parquet(TABDIR / "tfidf_kmeans_labels.parquet", index=False)

    # Top terms per cluster (by centroid weight)
    rows = []
    centers = km.cluster_centers_
    topn = 15
    for c in range(k):
        idx = centers[c].argsort()[-topn:][::-1]
        rows.append({"cluster": c, "top_terms": ", ".join(vocab[idx])})
    pd.DataFrame(rows).to_csv(TABDIR / "tfidf_kmeans_top_terms.csv", index=False)

    # ---------- Classification-first outputs

    # 1) Prevalence (% of projects per cluster)
    prev = (lab["tfidf_cluster"].value_counts(normalize=True)
            .rename("share_projects")
            .rename_axis("cluster")
            .reset_index()
            .sort_values("cluster"))
    prev.to_csv(TABDIR / "tfidf_cluster_prevalence.csv", index=False)

    # 2) Time series: share of projects per cluster by year
    base_by_year = lab.groupby("year")["activity_id"].nunique().rename("n_year")
    counts = (lab.groupby(["year","tfidf_cluster"])["activity_id"]
              .nunique().rename("n").reset_index())
    ts = counts.merge(base_by_year.reset_index(), on="year", how="left")
    ts["share"] = ts["n"] / ts["n_year"]
    ts.to_csv(TABDIR / "tfidf_cluster_timeseries.csv", index=False)

    # 3) Cluster quality: silhouette on a sample (skip if too small)
    sil = np.nan
    try:
        # sample up to 20k rows for speed
        n = min(20000, X.shape[0])
        rnd = np.random.RandomState(1).choice(X.shape[0], size=n, replace=False)
        sil = silhouette_score(X[rnd], labels[rnd], metric="cosine")
    except Exception:
        pass
    pd.DataFrame([{"k": k, "silhouette_cosine": sil}]).to_csv(TABDIR / "tfidf_cluster_quality.csv", index=False)

    # 4) Alignment with other methods (if available)

    # 4a) Keyword hit rate per cluster (any SDG15 tag)
    if KEYWORD_LABELS.exists():
        kw = pd.read_parquet(KEYWORD_LABELS)[["activity_id"]].drop_duplicates()
        kw["kw_hit"] = 1
        j = lab.merge(kw, on="activity_id", how="left").fillna({"kw_hit":0})
        kh = (j.groupby("tfidf_cluster")["kw_hit"].mean()
                .rename("keyword_hit_rate")
                .reset_index()
                .sort_values("tfidf_cluster"))
        kh.to_csv(TABDIR / "tfidf_cluster_keyword_hit_rate.csv", index=False)
    else:
        kh = None

    # 4b) Majority LLM primary label per cluster
    if LLM_PRIMARY.exists():
        llm = pd.read_parquet(LLM_PRIMARY)[["activity_id","primary_label"]]
        j = lab.merge(llm, on="activity_id", how="inner")
        maj = (j.groupby(["tfidf_cluster","primary_label"])["activity_id"]
                 .count().rename("n").reset_index())
        # pick argmax label per cluster
        maj_idx = maj.groupby("tfidf_cluster")["n"].idxmax()
        maj_lab = (maj.loc[maj_idx, ["tfidf_cluster","primary_label","n"]]
                      .rename(columns={"primary_label":"majority_llm_label","n":"maj_count"})
                      .sort_values("tfidf_cluster"))
        maj_lab.to_csv(TABDIR / "tfidf_cluster_majority_llm.csv", index=False)
    else:
        maj_lab = None

    # 5) Consolidated cluster summary table (for the paper)
    summary = prev.merge(pd.DataFrame(rows), on="cluster", how="left")
    if kh is not None:
        summary = summary.merge(kh, on="tfidf_cluster", left_on=None, right_on=None, how="left").rename(columns={"tfidf_cluster":"cluster"})
    if maj_lab is not None:
        summary = summary.merge(maj_lab, on="cluster", how="left")
    # Clean order
    summary = summary.sort_values("share_projects", ascending=False)
    summary.to_csv(TABDIR / "tfidf_cluster_summary.csv", index=False)

    print("[TFIDF] Wrote:")
    print("  - tfidf_kmeans_labels.parquet, tfidf_kmeans_top_terms.csv")
    print("  - tfidf_cluster_prevalence.csv, tfidf_cluster_timeseries.csv, tfidf_cluster_quality.csv")
    print("  - tfidf_cluster_keyword_hit_rate.csv (if keywords present)")
    print("  - tfidf_cluster_majority_llm.csv (if LLM labels present)")
    print("  - tfidf_cluster_summary.csv")

if __name__ == "__main__":
    main()
