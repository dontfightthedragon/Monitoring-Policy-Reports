#!/usr/bin/env python3
# preprocessing/LanguageNormalize.py
from __future__ import annotations
from pathlib import Path
import argparse
import re
import sys
import time
from typing import Callable, Optional, Dict

import pandas as pd

# --------- Paths (repo-root relative) ----------
ROOT   = Path(__file__).resolve().parents[1]
INPUT  = ROOT / "dataset/temporaryFileToWorkWith/CRS_filtered_dedup.parquet"
OUTPUT = ROOT / "dataset/temporaryFileToWorkWith/CRS_language_normalized.parquet"

# --------- Config defaults ----------
BATCH_SIZE = 50_000       # process in batches to keep memory predictable
MAX_TEXT_SLICE = 800      # detector speedup
TARGET_LANGS = {"en", "fr"}

# --------- Utilities ----------
def basic_clean(text: str) -> str:
    if not isinstance(text, str): return ""
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text

def build_language_detector() -> Callable[[str], str]:
    """
    Returns a function text -> {'en','fr','other','unknown'}
    Priority: lingua -> langdetect -> heuristic
    """
    # 1) lingua (fast/accurate)
    try:
        from lingua import Language, LanguageDetectorBuilder
        det = LanguageDetectorBuilder.from_languages(
            Language.ENGLISH, Language.FRENCH
        ).build()
        def detect_lang(s: str) -> str:
            if not isinstance(s, str) or not s.strip(): return "unknown"
            lang = det.detect_language_of(s[:MAX_TEXT_SLICE])
            if   lang == Language.ENGLISH: return "en"
            elif lang == Language.FRENCH:  return "fr"
            else: return "other"
        return detect_lang
    except Exception:
        pass

    # 2) langdetect (simple, permissive)
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        def detect_lang(s: str) -> str:
            if not isinstance(s, str) or not s.strip(): return "unknown"
            try:
                code = detect(s[:MAX_TEXT_SLICE])
                if code in TARGET_LANGS: return code
                return "other"
            except Exception:
                return "unknown"
        return detect_lang
    except Exception:
        pass

    # 3) heuristic fallback
    def detect_lang(s: str) -> str:
        s = (" " + (s or "").lower() + " ")
        fr_hits = any(w in s for w in (" le ", " la ", " et ", " est ", " des ", " une ", " biodiversité "))
        en_hits = any(w in s for w in (" the ", " and ", " is ", " of ", " for ", " biodiversity "))
        if fr_hits and not en_hits: return "fr"
        if en_hits and not fr_hits: return "en"
        if fr_hits and en_hits:     return "other"
        return "unknown"
    return detect_lang

def build_translator(kind: str) -> Optional[Callable[[str], str]]:
    """
    Returns a function fr_text -> en_text depending on chosen backend.
    kind in {"none","google","helsinki"}
    """
    if kind == "none":
        return None

    if kind == "google":
        try:
            from deep_translator import GoogleTranslator
            gt = GoogleTranslator(source="fr", target="en")
            def translate(text: str) -> str:
                if not text: return text
                try:
                    return gt.translate(text)
                except Exception:
                    return text
            return translate
        except Exception:
            print("[WARN] deep-translator not available; falling back to no translation.", file=sys.stderr)
            return None

    if kind == "helsinki":
        # Offline model via transformers (larger setup, but no external API)
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            import torch
            model_name = "Helsinki-NLP/opus-mt-fr-en"
            tok = AutoTokenizer.from_pretrained(model_name)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            mdl.to(device)

            def translate(text: str, max_len: int = 500) -> str:
                if not text: return text
                # Short safeguard for extremely long rows
                chunk = text if len(text) <= 2000 else text[:2000]
                try:
                    inputs = tok(chunk, return_tensors="pt", truncation=True, max_length=max_len).to(device)
                    outputs = mdl.generate(**inputs, max_length=max_len)
                    out = tok.decode(outputs[0], skip_special_tokens=True)
                    return out
                except Exception:
                    return text
            return translate
        except Exception:
            print("[WARN] transformers not available; falling back to no translation.", file=sys.stderr)
            return None

    print(f"[WARN] Unknown translator kind: {kind}; using none.", file=sys.stderr)
    return None

def summarize_counts(series: pd.Series, k: int = 10) -> Dict[str, int]:
    vc = series.value_counts(dropna=False)
    head = vc.head(k).to_dict()
    head["__TOTAL__"] = int(vc.sum())
    return head

# --------- Main pipeline ----------
def process(args) -> None:
    in_path  = Path(args.input or INPUT)
    out_path = Path(args.output or OUTPUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[LANG] Reading: {in_path}")
    df = pd.read_parquet(in_path, engine="pyarrow")

    # Ensure required columns
    for c in ["text", "language"]:
        if c not in df.columns:
            df[c] = ""
    df["text"] = df["text"].astype(str).map(basic_clean)

    # Build detector & translator
    detect_lang = build_language_detector()
    translator  = build_translator(args.translator)

    # Detect language (batch with caching)
    print("[LANG] Detecting languages (EN/FR/other/unknown)...")
    short = df["text"].str.slice(0, MAX_TEXT_SLICE)
    uniq  = short.drop_duplicates()
    lang_map = {s: detect_lang(s) for s in uniq}
    df["language"] = short.map(lambda s: lang_map.get(s, "unknown"))

    # Prepare text fields
    df["text_original"] = df["text"]
    df["text_en"] = df["text"]  # default: identity

    # Translate FR→EN if requested
    if translator is not None:
        mask_fr = df["language"].eq("fr")
        if mask_fr.any():
            print(f"[LANG] Translating FR->EN: {mask_fr.sum():,} rows via '{args.translator}'...")
            # Process in batches to avoid OOM or API throttling
            idx = df.index[mask_fr]
            for i in range(0, len(idx), BATCH_SIZE):
                batch_idx = idx[i:i+BATCH_SIZE]
                fr_texts = df.loc[batch_idx, "text_original"].astype(str).tolist()
                start = time.time()
                en_texts = [translator(t) for t in fr_texts]
                df.loc[batch_idx, "text_en"] = en_texts
                dur = time.time() - start
                print(f"  - translated {len(batch_idx):,} rows in {dur:.1f}s")
    else:
        print("[LANG] Translator disabled ('none'); keeping text_en = original text.")

    # Save
    keep_cols = list(dict.fromkeys([
        # ids & meta (keep what exists without failing)
        *(c for c in ["year","activity_id","donor_name","recipient_name","recipient_code",
                      "flow_name","flow_code","donor_code","biodiversity",
                      "usd_commitment_defl","usd_disbursement_defl"] if c in df.columns),
        # language/text
        "language","text_original","text_en"
    ]))
    out_df = df[keep_cols].copy()

    # Light compaction
    for c in ["donor_name","recipient_name","flow_name","language"]:
        if c in out_df.columns:
            out_df[c] = out_df[c].astype("category")

    out_df.to_parquet(out_path, engine="pyarrow", index=False)
    print(f"[LANG] Saved -> {out_path}")

    # Quick report
    print("\n[LANG] === Summary ===")
    print("Language counts:", summarize_counts(out_df["language"]))
    if "year" in out_df.columns:
        yrs = out_df["year"].value_counts().sort_index()
        print(f"Years: {yrs.index.min()}–{yrs.index.max()} | Unique years: {len(yrs)}")
    print(f"Rows: {len(out_df):,}")
    print("Sample text pairs (EN/FR):")
    sample = out_df.sample(min(5, len(out_df)), random_state=1)
    for _, r in sample.iterrows():
        print("—", r.get("language"))
        print("   original:", (r.get("text_original") or "")[:160])
        print("   text_en :", (r.get("text_en") or "")[:160])

def parse_args():
    p = argparse.ArgumentParser(description="Detect language and translate FR->EN for CRS dataset.")
    p.add_argument("--input",  type=str, default=None, help="Path to CRS_filtered_dedup.parquet")
    p.add_argument("--output", type=str, default=None, help="Path to save CRS_language_normalized.parquet")
    p.add_argument("--translator", type=str, default="none",
                   choices=["none","google","helsinki"],
                   help="Translation backend: 'none' (skip), 'google' (deep-translator), or 'helsinki' (offline HF model).")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process(args)


import pandas as pd
p="dataset/temporaryFileToWorkWith/CRS_language_normalized.parquet"
df=pd.read_parquet(p, engine="pyarrow")
print(df["language"].value_counts(dropna=False))            # should match your counts
fr=df[df["language"]=="fr"]
print("Translated share:", (fr["text_en"]!=fr["text_original"]).mean())  # should be 0.0 right now
