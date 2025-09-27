#!/usr/bin/env python3
# preprocessing/textPrep.py

from pathlib import Path
import pandas as pd
import spacy

# -------- Paths --------
INPUT = Path("dataset/temporaryFileToWorkWith/CRS_enfr_language_normalized.parquet")
OUTPUT = Path("dataset/temporaryFileToWorkWith/CRS_textprep.parquet")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# -------- Load data --------
print(f"[TEXTPREP] Reading: {INPUT}")
df = pd.read_parquet(INPUT, engine="pyarrow")

# -------- spaCy setup --------
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])  # only tokenizer + tagger + lemmatizer

def clean_text(doc):
    return " ".join(
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and token.is_alpha and len(token) > 2
    )

# -------- Process in batches --------
import time

if __name__ == "__main__":
    texts = df["text_en"].astype(str).tolist()
    cleaned = []
    total = len(texts)
    batch_size = 500
    start_time = time.time()
    processed = 0
    print(f"[TEXTPREP] Starting batch processing: {total} texts")
    for i, doc in enumerate(nlp.pipe(texts, batch_size=batch_size, n_process=2)):
        cleaned.append(clean_text(doc))
        processed += 1
        if processed % batch_size == 0 or processed == total:
            elapsed = time.time() - start_time
            avg_time = elapsed / processed
            remaining = total - processed
            est_left = avg_time * remaining
            print(f"[TEXTPREP] {processed}/{total} processed | Elapsed: {elapsed:.1f}s | ETA: {est_left:.1f}s")

    df["text_clean"] = cleaned

    # -------- Save --------
    df.to_parquet(OUTPUT, engine="pyarrow", index=False)

    print(f"[TEXTPREP] Saved -> {OUTPUT}")
    print(f"[TEXTPREP] Rows: {len(df):,}")
    print("[TEXTPREP] Sample (text_en → text_clean):")
    print(df[["text_en","text_clean"]].sample(3, random_state=42).to_string(index=False))
