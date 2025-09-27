# analysis/04_sbert_cluster.py
# BERT embeddings + clustering (modern)

#!/usr/bin/env python3
from common import seed_all, load_textprep, ensure_outdir
import numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import umap

INP  = "dataset/temporaryFileToWorkWith/CRS_en_textprep.parquet"
OUTD = "reports/tables"; ensure_outdir(OUTD)

def main(model_name="sentence-transformers/all-MiniLM-L6-v2", k=50):
    seed_all(1)
    df = load_textprep(INP)
    texts = df["text_for_nlp"].fillna("").tolist()

    enc = SentenceTransformer(model_name)
    Z = enc.encode(texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    km = KMeans(n_clusters=k, random_state=1, n_init="auto")
    lab = km.fit_predict(Z)

    um = umap.UMAP(n_components=2, random_state=1)
    emb2 = um.fit_transform(Z)

    out = df[["activity_id","year"]].copy()
    out["cluster_kmeans"] = lab
    out["umap_x"] = emb2[:,0]; out["umap_y"] = emb2[:,1]
    out.to_parquet(f"{OUTD}/sbert_clusters.parquet", index=False)

    # Quick theme summary: top words per cluster using your cleaned tokens
    df_tokens = df.assign(cluster=lab)
    def top_terms(group, n=10):
        from collections import Counter
        toks = " ".join(group["text_for_nlp"]).split()
        return ", ".join([w for w,_ in Counter(toks).most_common(n)])
    summary = df_tokens.groupby("cluster").apply(top_terms).reset_index(name="top_terms")
    summary.to_csv(f"{OUTD}/sbert_cluster_terms.csv", index=False)
    print(summary.head())

if __name__ == "__main__": main()
