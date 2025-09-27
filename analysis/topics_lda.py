#!/usr/bin/env python3
# Classical topic modeling (LDA) for CRS_textprep
from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction import text as sk_text
from joblib import dump

from common import seed_all, load_textprep, ensure_outdir

ROOT = Path(__file__).resolve().parents[1]

def parse_args():
    p = argparse.ArgumentParser(description="LDA topic modeling on CRS_textprep.")
    p.add_argument("--input", default="dataset/temporaryFileToWorkWith/CRS_textprep.parquet")
    p.add_argument("--outdir", default="reports/tables")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--max-features", type=int, default=50000)
    p.add_argument("--min-df", type=int, default=10)
    p.add_argument("--max-df", type=float, default=0.95)
    p.add_argument("--ngram", type=int, default=1, choices=[1,2,3])
    p.add_argument("--online", action="store_true")
    p.add_argument("--bio-only", action="store_true")
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()

def main():
    args = parse_args()
    outd = ROOT / args.outdir
    ensure_outdir(str(outd))
    seed_all(args.seed)

    # --- Load & (optional) biodiversity filter ---
    df = load_textprep(str(ROOT / args.input))
    if args.bio_only and "biodiversity" in df.columns:
        mask_bio = pd.to_numeric(df["biodiversity"], errors="coerce").fillna(0) >= 1
        df = df.loc[mask_bio].copy()

    if args.sample and args.sample > 0 and len(df) > args.sample:
        df = df.sample(args.sample, random_state=args.seed).copy()

    texts = df["text_for_nlp"].fillna("").astype(str)
    print(f"[LDA] Docs: {len(texts):,} | bio_only={args.bio_only} | sample={args.sample}")

    # --- Stopwords & biodiversity anchor (ALL inside main) ---
    DOMAIN_STOP = {
        "project","programme","program","support","implementation","capacity","strengthening",
        "activity","activities","component","phase","plan","policy","strategic","approach","framework",
        "administrative","administration","accountability","transparency","cost","expenses",
        "report","reporting","monitoring","evaluation","logframe",
        "usaid","world","bank","un","unicef","undp","eu","ifad","fao","giz","gef","gcf","dfid",
        "ngo","foundation","partnership","cooperation",
        "aid","assistance","fund","funding","grant","loan","credit",
        "emission","carbon","climate","resilience","adaptation","mitigation",
        "health","hiv","aids","malaria","tuberculosis","immunization","vaccine",
        "coffee","cocoa","cotton","sugar",
        "les","des","une","ssnc", "sida", "progreen", "vietnam", 
        "rri", "wwf", "ilrg", "sek", "million", "initiative", "charitable", "wales", 
        "prince", "canada", "dan", "africa", "asia", "caribbean", "projet", "pour", "sur", 
        "aux", "dans", "par", "femmes", "civil", "civil society", "organisation", "society", "base", "technical", 
        "solution", "initiative", "improve", "mexico", "strengthen", "include", "services", "service", "initiative", 
        "charitable", "sida", "ssnc", "progreen", "rri", "ilrg", "million"}
    STOPWORDS = sorted(set(sk_text.ENGLISH_STOP_WORDS) | DOMAIN_STOP)

    ANCHOR = re.compile(
        r"\b(biodiversit|ecosystem|conservation|protected area|wildlife|forest|"
        r"wetland|ramsar|land degradation|desertification|invasive species|"
        r"poaching|cites|reforestation|afforestation|restoration|watershed|mangrove)\b",
        re.I
    )

    # Apply anchor once
    mask_anchor = texts.str.contains(ANCHOR, na=False)
    df = df.loc[mask_anchor].copy()
    texts = df["text_for_nlp"].fillna("").astype(str)
    print(f"[LDA] After biodiversity anchor: {len(df):,} docs")

    # --- Vectorize ---
    ngram_range = (1, args.ngram)
    vect = CountVectorizer(
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b",
        max_df=args.max_df, min_df=args.min_df,
        ngram_range=ngram_range,
        max_features=args.max_features,
        stop_words=STOPWORDS
    )
    X = vect.fit_transform(texts)
    vocab = np.array(vect.get_feature_names_out())
    print(f"[LDA] Vocab size: {len(vocab):,} | ngram_range={ngram_range}")

    # --- Train LDA ---
    lda_kwargs = dict(
        n_components=args.k,
        learning_method="online" if args.online else "batch",
        random_state=args.seed,
        n_jobs=-1,
    )
    if args.online:
        lda_kwargs["batch_size"] = 2048
    lda = LatentDirichletAllocation(**lda_kwargs)
    W = lda.fit_transform(X)
    H = lda.components_

    # --- Top terms ---
    top_k = 15
    rows = []
    for t in range(args.k):
        idx = H[t].argsort()[-top_k:][::-1]
        terms = vocab[idx]
        weights = H[t, idx]
        rows.append({
            "topic": t,
            "top_terms": ", ".join(terms),
            "top_terms_with_weights": ", ".join(f"{w:.0f}:{term}" for term, w in zip(terms, weights))
        })
    top_df = pd.DataFrame(rows)

    # Optional labels
    LABEL_HINTS = {
        "forest": "Forest management",
        "illegal logging": "Illegal logging",
        "protected area": "Protected areas",
        "invasive": "Invasive species",
        "poach": "Poaching & wildlife",
        "cites": "Wildlife trade (CITES)",
        "ranger": "Anti-poaching/rangers",
        "ecosystem service": "Ecosystem services",
        "natural capital": "Natural capital",
        "desertification": "Desertification",
        "land degradation": "Land degradation",
        "wetland": "Wetlands",
    }
    def label_topic(row):
        terms = row["top_terms"]
        for k, v in LABEL_HINTS.items():
            if k in terms:
                return v
        return "General biodiversity"
    top_df["topic_label"] = top_df.apply(label_topic, axis=1)
    top_df.to_csv(outd / "lda_top_terms.csv", index=False)

    # --- Assignments & saves ---
    assign = df[["activity_id","year"]].copy()
    assign["topic"] = W.argmax(axis=1)
    assign.to_parquet(outd / "lda_doc_topics.parquet", index=False)

    np.save(outd / "lda_doc_topic.npy", W)
    np.save(outd / "lda_topic_term.npy", H)
    dump(vect, outd / "lda_vectorizer.joblib")
    dump(lda,  outd / "lda_model.joblib")

    # --- Prevalence by year ---
    prev = (assign.groupby(["year","topic"])
                  .size().rename("doc_count").reset_index()
                  .pivot(index="year", columns="topic", values="doc_count")
                  .fillna(0).astype(int))
    prev.to_csv(outd / "lda_topic_prevalence_by_year.csv")

    # --- Donor / recipient by topic ---
    meta_cols = set(df.columns)
    if {"donor_name", "usd_disbursement_defl"}.issubset(meta_cols):
        donor_topic = (assign.join(
                df[["activity_id","donor_name","usd_disbursement_defl"]].set_index("activity_id"),
                on="activity_id", how="left")
            .groupby(["topic","donor_name"], as_index=False, observed=True)["usd_disbursement_defl"].sum()
            .sort_values(["topic","usd_disbursement_defl"], ascending=[True, False]))
        donor_topic.groupby("topic").head(15).to_csv(outd / "lda_top_donors_per_topic.csv", index=False)

    if {"recipient_name", "usd_disbursement_defl"}.issubset(meta_cols):
        recip_topic = (assign.join(
                df[["activity_id","recipient_name","usd_disbursement_defl"]].set_index("activity_id"),
                on="activity_id", how="left")
            .groupby(["topic","recipient_name"], as_index=False, observed=True)["usd_disbursement_defl"].sum()
            .sort_values(["topic","usd_disbursement_defl"], ascending=[True, False]))
        recip_topic.groupby("topic").head(15).to_csv(outd / "lda_top_recipients_per_topic.csv", index=False)

    # --- Console preview ---
    print("[LDA] Top terms per topic (first 5):")
    print(top_df.head())

    # === QA LOOP: topic sizes, money per topic, and sample docs ===
    qa_dir = outd / "lda_qa"
    qa_dir.mkdir(parents=True, exist_ok=True)

    # Merge assignments with metadata/text for convenient QA
    merge_cols = ["activity_id", "year", "usd_disbursement_defl"]
    if "donor_name" in df.columns: merge_cols.append("donor_name")
    if "recipient_name" in df.columns: merge_cols.append("recipient_name")
    # Prefer English text if present; fall back to cleaned text
    text_col = "text_en" if "text_en" in df.columns else "text_for_nlp"
    merge_cols.append(text_col)

    merged = (
        assign.merge(
            df[merge_cols],
            on="activity_id",
            how="left"
        )
    )

    # 1) Topic sizes (doc counts)
    topic_sizes = merged["topic"].value_counts().sort_index()
    print("\n[LDA-QA] Topic sizes (docs per topic):")
    print(topic_sizes)
    topic_sizes.to_csv(qa_dir / "topic_sizes.csv", header=["doc_count"])

    # 2) Disbursement totals by topic
    if "usd_disbursement_defl" in merged.columns:
        money_by_topic = (merged.groupby("topic", as_index=True)["usd_disbursement_defl"]
                                .sum()
                                .sort_index())
        print("\n[LDA-QA] Disbursements by topic (sum of usd_disbursement_defl):")
        print(money_by_topic.round(1))
        money_by_topic.to_csv(qa_dir / "topic_disbursements.csv", header=["usd_disb"])

    # 3) Sample docs per topic (quick text sanity check)
    def _snippet(s: str, n=240) -> str:
        s = (s or "") if isinstance(s, str) else ""
        s = re.sub(r"\s+", " ", s).strip()
        return s[:n]

    samples_path = qa_dir / "topic_samples.csv"
    samples = []  # collect in-memory then write once

    # Pick top 3 by money (if available), otherwise first 3 by index
    for t in sorted(merged["topic"].unique()):
        sub = merged[merged["topic"] == t].copy()
        if "usd_disbursement_defl" in sub.columns:
            sub["usd_disbursement_defl"] = pd.to_numeric(sub["usd_disbursement_defl"], errors="coerce").fillna(0.0)
            sub = sub.sort_values("usd_disbursement_defl", ascending=False)
        sub = sub.head(3)

        for _, r in sub.iterrows():
            samples.append({
                "topic": int(t),
                "year": r.get("year"),
                "usd_disbursement_defl": r.get("usd_disbursement_defl"),
                "donor_name": r.get("donor_name"),
                "recipient_name": r.get("recipient_name"),
                "text_snippet": _snippet(r.get(text_col, "")),
            })

    if samples:
        pd.DataFrame(samples).to_csv(samples_path, index=False)
        print(f"\n[LDA-QA] Wrote QA files to: {qa_dir}")
        print(f"  - topic_sizes.csv")
        print(f"  - topic_disbursements.csv")
        print(f"  - topic_samples.csv (3 example projects per topic)")


if __name__ == "__main__":
    main()

  