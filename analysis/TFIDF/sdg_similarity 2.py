#!/usr/bin/env python3
# analysis/TFIDF/sdg_similarity.py
# Post-process SDG-15 cosine scores: thresholding, overlap vs keywords, examples, plots

from pathlib import Path
import argparse, numpy as np, pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scored_csv", required=True,
                   help="CSV with sdg15_score and project text (e.g., outputs/tfidf_baseline/artifacts/sdg15_scored_projects.csv)")
    p.add_argument("--kw_csv", default="", 
                   help="Optional CSV with keyword flags (must include activity_id and kw_any)")
    p.add_argument("--outdir", default="outputs/sdg15_report",
                   help="Where to write summary tables and figures")
    p.add_argument("--top_pct", type=float, default=1.0,
                   help="If no keyword Top-K is used, select top X percent by score (e.g., 1.0 = top 1%%)")
    p.add_argument("--match_keyword_k", action="store_true",
                   help="If kw_csv provided, set K equal to the number of keyword-tagged projects")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()

def main():
    args = parse_args()
    outdir = Path(args.outdir); (outdir/"figures").mkdir(parents=True, exist_ok=True); (outdir/"artifacts").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.scored_csv)
    if "sdg15_score" not in df.columns:
        raise ValueError("Expected column 'sdg15_score' in scored_csv")

    # Merge keyword flags if provided
    have_kw = False
    if args.kw_csv:
        kw = pd.read_csv(args.kw_csv)
        need = {"activity_id","kw_any"}
        if not need.issubset(kw.columns):
            raise ValueError(f"kw_csv must contain columns: {need}")
        df = df.merge(kw[list(need)], on="activity_id", how="left")
        df["kw_any"] = df["kw_any"].fillna(False).astype(bool)
        have_kw = True

    # Decide selection rule
    if have_kw and args.match_keyword_k:
        K = int(df["kw_any"].sum())
        if K == 0:
            # fallback to top_pct if no keywords matched
            K = max(1000, int(len(df) * (args.top_pct / 100.0)))
        cut = df["sdg15_score"].nlargest(K).min()
        rule = {"type":"topk_match_keywords", "K": K, "threshold": float(cut)}
        df["sdg15_selected"] = df["sdg15_score"] >= cut
    else:
        # top percentile (e.g., 1.0 => top 1%)
        frac = max(0.0001, float(args.top_pct) / 100.0)
        K = max(1, int(len(df) * frac))
        cut = df["sdg15_score"].nlargest(K).min()
        rule = {"type":"top_percent", "percent": args.top_pct, "K": K, "threshold": float(cut)}
        df["sdg15_selected"] = df["sdg15_score"] >= cut

    # Overlap summary 
    if have_kw:
        a = df["kw_any"]
        b = df["sdg15_selected"]
        overlap_tbl = pd.DataFrame({
            "set": ["Keyword only","SDG-sim only","Both","Either"],
            "count": [(a & ~b).sum(), (~a & b).sum(), (a & b).sum(), (a | b).sum()]
        })
        overlap_tbl.to_csv(outdir/"artifacts"/"sdg15_overlap_summary.csv", index=False)
        print("\nOverlap summary:\n", overlap_tbl)
    else:
        overlap_tbl = None
        print("\nNo kw_csv provided → skipping overlap summary.")

    # Example snippets the similarity finds but keywords miss
    if have_kw:
        examples = df.loc[(~a & df["sdg15_selected"]), ["activity_id","year","sdg15_score","text"]].copy()
        examples = examples.sort_values("sdg15_score", ascending=False).head(10)
        examples.to_csv(outdir/"artifacts"/"sdg15_examples_similarity_only.csv", index=False)
        print("\nSaved examples (similarity-only) → artifacts/sdg15_examples_similarity_only.csv")

    # Histogram with threshold
    plt.figure(figsize=(6,4))
    df["sdg15_score"].hist(bins=40)
    plt.axvline(cut, linestyle="--", linewidth=2)
    plt.xlabel("SDG-15 similarity score"); plt.ylabel("Projects")
    plt.tight_layout()
    plt.savefig(outdir/"figures"/"sdg15_hist_with_cut.pdf"); plt.close()

    # Overlap bars (if available)
    if have_kw:
        vals = [(a & ~df["sdg15_selected"]).sum(), (~a & df["sdg15_selected"]).sum(), (a & df["sdg15_selected"]).sum()]
        labels = ["Keyword only","SDG-sim only","Both"]
        plt.figure(figsize=(6,4))
        plt.bar(labels, vals)
        plt.ylabel("Projects")
        plt.tight_layout()
        plt.savefig(outdir/"figures"/"sdg15_overlap_bars.pdf"); plt.close()

    # Year trend (share selected)
    if "year" in df.columns:
        year_share = df.groupby("year")["sdg15_selected"].mean().rename("share_selected").reset_index()
        year_share.to_csv(outdir/"artifacts"/"sdg15_topk_by_year.csv", index=False)

    # Save enriched CSV + rule used
    df.to_csv(outdir/"artifacts"/"sdg15_scored_projects_with_selection.csv", index=False)
    (outdir/"artifacts"/"sdg15_selection_rule.json").write_text(pd.Series(rule).to_json(indent=2))

    # --- Boxplot by Rio marker (clean + zoomed) ---
    # ECDF of SDG-15 scores by Rio marker (log x-scale)
    g0 = df.loc[df["biodiversity"].isin([0]), "sdg15_score"].dropna().values
    g1 = df.loc[df["biodiversity"].isin([1,2]), "sdg15_score"].dropna().values

    def ecdf(x):
        x = np.sort(x)
        y = np.arange(1, x.size+1) / x.size
        return x, y

    x0, y0 = ecdf(g0); x1, y1 = ecdf(g1)

    plt.figure(figsize=(6,4))
    plt.plot(x0, y0, label="Rio 0 (non-bio)")
    plt.plot(x1, y1, label="Rio 1/2 (bio)")
    plt.xscale("log")                 # scores are tiny → log helps
    plt.xlabel("SDG-15 similarity score (log scale)")
    plt.ylabel("Cumulative share of projects")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outdir/"figures"/"sdg15_ecdf.pdf"); plt.close()

    print("\n[OK] Wrote:")
    print(" - figures/sdg15_hist_with_cut.pdf")
    if have_kw:
        print(" - figures/sdg15_overlap_bars.pdf")
        print(" - artifacts/sdg15_overlap_summary.csv")
        print(" - artifacts/sdg15_examples_similarity_only.csv")
    print(" - artifacts/sdg15_topk_by_year.csv (if year found)")
    print(" - artifacts/sdg15_scored_projects_with_selection.csv")
    print(" - artifacts/sdg15_selection_rule.json")

if __name__ == "__main__":
    main()
