# analysis/01_tfidf_baseline.py
# Quick TF–IDF vector baseline for project texts

#!/usr/bin/env python3
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from common import load_textprep, ensure_outdir, seed_all

ROOT = Path(__file__).resolve().parents[1]
INP  = ROOT / "dataset/temporaryFileToWorkWith/CRS_textprep.parquet"
OUTD = ROOT / "reports" / "tables"; ensure_outdir(str(OUTD))

def main():
    seed_all(1)
    df = load_textprep(str(INP))
    # Focus on biodiversity-marked projects to keep it lean:
    if "biodiversity" in df.columns:
        df = df[pd.to_numeric(df["biodiversity"], errors="coerce").fillna(0) >= 1].copy()

    texts = df["text_for_nlp"].fillna("").astype(str)

    vect = TfidfVectorizer(
        ngram_range=(1,2),
        min_df=30, max_df=0.9,  # small & robust
        max_features=40000
    )
    X = vect.fit_transform(texts)

    # Top terms overall
    vocab = np.array(vect.get_feature_names_out())
    means = np.asarray(X.mean(axis=0)).ravel()
    top = pd.DataFrame({"term": vocab, "tfidf": means}).sort_values("tfidf", ascending=False).head(200)
    top.to_csv(OUTD / "tfidf_top_terms.csv", index=False)

    # Quick k-means
    k = 20
    km = KMeans(n_clusters=k, random_state=1, n_init=10)
    labels = km.fit_predict(X)
    df_out = df[["activity_id","year"]].copy()
    df_out["tfidf_cluster"] = labels
    df_out.to_parquet(OUTD / "tfidf_kmeans_labels.parquet", index=False)

    # Top terms per cluster
    rows=[]
    for c in range(k):
        idx = np.where(labels==c)[0]
        if len(idx)==0: continue
        centroid = km.cluster_centers_[c]
        top_idx = centroid.argsort()[-15:][::-1]
        rows.append({"cluster": c, "top_terms": ", ".join(vocab[top_idx])})
    pd.DataFrame(rows).to_csv(OUTD / "tfidf_kmeans_top_terms.csv", index=False)

    # Optional: compare to LDA (if you have lda_doc_topics.parquet)
    lda_path = OUTD / "lda_doc_topics.parquet"
    if lda_path.exists():
        lda = pd.read_parquet(lda_path)
        merged = df_out.merge(lda, on=["activity_id","year"], how="inner")
        ari = adjusted_rand_score(merged["tfidf_cluster"], merged["topic"])
        nmi = normalized_mutual_info_score(merged["tfidf_cluster"], merged["topic"])
        print(f"[TFIDF] ARI vs LDA: {ari:.3f} | NMI: {nmi:.3f}")

    print("[TFIDF] Wrote: tfidf_top_terms.csv, tfidf_kmeans_labels.parquet, tfidf_kmeans_top_terms.csv")

if __name__ == "__main__":
    main()
