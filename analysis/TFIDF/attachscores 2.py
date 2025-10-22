#!/usr/bin/env python3
import numpy as np, pandas as pd
from pathlib import Path
import argparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", default="reports/figures/artifacts/sdg15_scores.npy")
    p.add_argument("--top_terms", default="reports/figures/artifacts/sdg15_top_terms.txt")
    p.add_argument("--data", default="dataset/temporaryFileToWorkWith/CRS_textprep.parquet")
    p.add_argument("--out", default="reports/figures/artifacts/sdg15_scored_projects.csv")
    p.add_argument("--index", default="reports/figures/artifacts/index_order.npy",
                   help="Optional index file saved by tfidf_baseline for exact alignment")
    return p.parse_args()


def coalesce_text(df: pd.DataFrame) -> pd.Series:
    cols = ["text_clean", "text_en", "text_original"]
    have = [c for c in cols if c in df.columns]
    if not have:
        raise ValueError("No text column found (expected one of: text_clean, text_en, text_original)")
    txt = df[have[0]].astype(str)
    for c in have[1:]:
        mask = txt.isna() | (txt.str.strip() == "") | (txt.str.lower().isin(["nan", "none"]))
        txt = txt.where(~mask, df[c].astype(str))
    return txt.rename("text")


def main():
    args = parse_args()
    scores_path = Path(args.scores)
    top_terms_path = Path(args.top_terms)
    out_path = Path(args.out)

    scores = np.load(scores_path)
    top_terms = top_terms_path.read_text().splitlines() if top_terms_path.exists() else []

    df = pd.read_parquet(args.data)

    keep = df["biodiversity"].isin([0, 1, 2])
    meta_cols = [c for c in ["year", "activity_id", "biodiversity"] if c in df.columns]
    df_keep = df.loc[keep].copy()
    texts = df_keep.loc[:, meta_cols].copy()
    texts["text"] = coalesce_text(df_keep)

    # If an index_order file exists, use it to select and order rows exactly
    index_path = Path(args.index)
    if index_path.exists():
        try:
            order = np.load(index_path, allow_pickle=True)
            # Ensure all indices exist; filter to intersection while preserving order
            mask = pd.Index(order).isin(df_keep.index)
            order = pd.Index(order)[mask]
            texts = texts.loc[order]
        except Exception as e:
            print(f"[warn] Failed to use index order file {index_path}: {e}")

    # Align lengths conservatively if needed
    n = min(len(texts), int(scores.shape[0]))
    if n != len(texts) or n != int(scores.shape[0]):
        print(f"[warn] Length mismatch: texts={len(texts)} scores={scores.shape[0]}; truncating to {n}")
    texts = texts.iloc[:n].copy()
    texts["sdg15_score"] = scores[:n]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    texts.to_csv(out_path, index=False)
    print(f"[OK] Wrote {n} rows with SDG-15 scores to {out_path}")


if __name__ == "__main__":
    main()
