#!/usr/bin/env python3
# preprocessing/LanguageNormalize.py
from __future__ import annotations
from pathlib import Path
import argparse, re, sys, time
from typing import Callable, Optional, Dict, List

import pandas as pd

# --------- Paths (repo-root relative) ----------
ROOT   = Path(__file__).resolve().parents[1]
INPUT  = ROOT / "dataset/temporaryFileToWorkWith/CRS_filtered_dedup.parquet"
OUTPUT = ROOT / "dataset/temporaryFileToWorkWith/CRS_language_normalized.parquet"
CACHE_PATH = ROOT / "dataset/temporaryFileToWorkWith/translation_cache.parquet"

# --------- Config defaults ----------
MAX_TEXT_SLICE = 800            # detector speedup
TARGET_LANGS   = {"en", "fr"}
FR_HINT_RE     = re.compile(r"\b(le|la|les|des|du|de|et|au|aux|dans|sur|avec|pour|sans|entre|biodiversité)\b", re.I)

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

    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        def detect_lang(s: str) -> str:
            if not isinstance(s, str) or not s.strip(): return "unknown"
            try:
                code = detect(s[:MAX_TEXT_SLICE])
                return code if code in TARGET_LANGS else "other"
            except Exception:
                return "unknown"
        return detect_lang
    except Exception:
        pass

    def detect_lang(s: str) -> str:
        s = (" " + (s or "").lower() + " ")
        fr_hits = bool(FR_HINT_RE.search(s))
        en_hits = any(w in s for w in (" the ", " and ", " is ", " of ", " for ", " biodiversity "))
        if fr_hits and not en_hits: return "fr"
        if en_hits and not fr_hits: return "en"
        if fr_hits and en_hits:     return "other"
        return "unknown"
    return detect_lang

def build_translator(kind: str):
    """
    Returns tuple: (backend, translator_fn or (tok, mdl, device))
    backend in {"none","google","helsinki"}
    """
    if kind == "none":
        return "none", None

    if kind == "google":
        try:
            from deep_translator import GoogleTranslator
            gt = GoogleTranslator(source="fr", target="en")
            def translate(text: str) -> str:
                """Google"""
                if not text: return text
                try:
                    return gt.translate(text)
                except Exception:
                    return text
            return "google", translate
        except Exception:
            print("[WARN] deep-translator not available; falling back to 'none'.", file=sys.stderr)
            return "none", None

    if kind == "helsinki":
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            import torch
            model_name = "Helsinki-NLP/opus-mt-fr-en"
            tok = AutoTokenizer.from_pretrained(model_name)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            mdl.to(device).eval()
            return "helsinki", (tok, mdl, device)
        except Exception:
            print("[WARN] transformers not available; falling back to 'none'.", file=sys.stderr)
            return "none", None

    print(f"[WARN] Unknown translator kind: {kind}; using 'none'.", file=sys.stderr)
    return "none", None

def summarize_counts(series: pd.Series, k: int = 10) -> Dict[str, int]:
    vc = series.value_counts(dropna=False)
    head = vc.head(k).to_dict()
    head["__TOTAL__"] = int(vc.sum())
    return head

# --------- Caching helpers ----------
def load_cache() -> Dict[str,str]:
    if CACHE_PATH.exists():
        try:
            df = pd.read_parquet(CACHE_PATH)
            return dict(zip(df["src"], df["tgt"]))
        except Exception:
            pass
    return {}

def save_cache(cache: Dict[str,str]) -> None:
    if not cache: return
    pd.DataFrame({"src": list(cache.keys()), "tgt": list(cache.values())}).to_parquet(CACHE_PATH, index=False)

# --------- Translation cores ----------
def translate_google_unique(uniq_texts: List[str], translator, threads: int = 8) -> Dict[str,str]:
    """Translate unique strings with Google in parallel (with retries)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import random

    def robust_translate(t: str, tries=3) -> str:
        for k in range(tries):
            try:
                return translator(t)
            except Exception:
                time.sleep(1.2 * (k + 1) + random.random()*0.3)
        return t

    out: Dict[str,str] = {}
    with ThreadPoolExecutor(max_workers=max(1, threads)) as ex:
        futs = {ex.submit(robust_translate, s): s for s in uniq_texts}
        done = 0
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                out[s] = fut.result()
            except Exception:
                out[s] = s
            done += 1
            if done % 2000 == 0:
                save_cache(out)  # small safety dump
                print(f"[LANG][Google] {done:,}/{len(uniq_texts):,}")
    return out

def translate_helsinki_unique(uniq_texts: List[str], tok_mdl_device, batch: int = 64, max_len: int = 256) -> Dict[str,str]:
    import torch
    tok, mdl, device = tok_mdl_device
    out: Dict[str,str] = {}
    for i in range(0, len(uniq_texts), batch):
        chunk = uniq_texts[i:i+batch]
        with torch.no_grad():
            toks = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
            outs = mdl.generate(**toks, max_length=max_len)
            res = [tok.decode(o, skip_special_tokens=True) for o in outs]
        out.update(dict(zip(chunk, res)))
        if (i // batch) % 20 == 0:
            save_cache(out)
            print(f"[LANG][Helsinki] {min(i+batch, len(uniq_texts)):,}/{len(uniq_texts):,}")
    return out

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
    backend, translator = build_translator(args.translator)

    # Detect language (hashed cache over first 800 chars)
    print("[LANG] Detecting languages (EN/FR/other/unknown)...")
    short = df["text"].str.slice(0, MAX_TEXT_SLICE)
    uniq  = short.drop_duplicates()
    lang_map = {s: detect_lang(s) for s in uniq}
    df["language"] = short.map(lambda s: lang_map.get(s, "unknown"))

    # Prepare text fields
    df["text_original"] = df["text"]
    df["text_en"] = df["text"]  # default: identity

    # Translate FR→EN with unique-text cache
    if backend != "none":
        mask_fr = (df["language"] == "fr")
        n_fr = int(mask_fr.sum())
        if n_fr:
            print(f"[LANG] FR rows: {n_fr:,} | backend='{backend}'")
            cache = load_cache()
            fr_series = df.loc[mask_fr, "text_original"].astype(str)

            # dedupe sources & remove already-cached ones
            uniq_fr = fr_series.drop_duplicates()
            todo = [t for t in uniq_fr if t not in cache]

            print(f"[LANG] Unique FR strings: {len(uniq_fr):,} | To translate now: {len(todo):,}")
            new_map: Dict[str,str] = {}

            if todo:
                if backend == "google":
                    new_map = translate_google_unique(todo, translator, threads=args.threads)
                elif backend == "helsinki":
                    new_map = translate_helsinki_unique(todo, translator, batch=args.batch, max_len=args.max_len)

            # merge cache + new results
            cache.update(new_map)
            save_cache(cache)

            # map translations back
            df.loc[mask_fr, "text_en"] = fr_series.map(lambda s: cache.get(s, s))

            # --- Fix pass: re-translate unchanged, likely-French rows ---
            if args.fix_missed:
                still = df.loc[mask_fr & (df["text_en"] == df["text_original"]), ["text_original"]].astype(str)
                looks_fr = still["text_original"].str.contains(FR_HINT_RE) & (still["text_original"].str.len() >= 40)
                retry_src = list(still.loc[looks_fr, "text_original"].drop_duplicates())
                print(f"[LANG] Fix pass: candidates {len(retry_src):,}")
                if retry_src:
                    # translate only those not in cache or identical
                    need = [s for s in retry_src if cache.get(s, s) == s]
                    if need:
                        if backend == "google":
                            fix_map = translate_google_unique(need, translator, threads=max(2, args.threads//2))
                        elif backend == "helsinki":
                            fix_map = translate_helsinki_unique(need, translator, batch=max(16, args.batch//2), max_len=args.max_len)
                        cache.update(fix_map); save_cache(cache)
                    # apply fixes
                    df.loc[mask_fr, "text_en"] = df.loc[mask_fr, "text_original"].map(lambda s: cache.get(s, s))
                print("[LANG] Fix pass done.")
        else:
            print("[LANG] No FR rows found; skipping translation.")
    else:
        print("[LANG] Translator disabled ('none'); keeping text_en = original text.")

    # Save
    keep_cols = list(dict.fromkeys([
        *(c for c in ["year","activity_id","donor_name","recipient_name","recipient_code",
                      "flow_name","flow_code","donor_code","biodiversity",
                      "usd_commitment_defl","usd_disbursement_defl"] if c in df.columns),
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
    fr = out_df[out_df["language"]=="fr"]
    share_changed = (fr["text_en"] != fr["text_original"]).mean() if len(fr) else 1.0
    print(f"Translated share (FR rows): {share_changed:.3f}")

def parse_args():
    p = argparse.ArgumentParser(description="Detect language and translate FR->EN for CRS dataset with caching and fix-pass.")
    p.add_argument("--input",  type=str, default=None, help="Path to CRS_filtered_dedup.parquet")
    p.add_argument("--output", type=str, default=None, help="Path to save CRS_language_normalized.parquet")
    p.add_argument("--translator", type=str, default="none",
                   choices=["none","google","helsinki"],
                   help="Translation backend: 'none', 'google' (deep-translator), 'helsinki' (offline HF model).")
    # Performance knobs
    p.add_argument("--threads", type=int, default=8, help="Parallel threads for Google backend.")
    p.add_argument("--batch",   type=int, default=64, help="Batch size for Helsinki backend.")
    p.add_argument("--max_len", type=int, default=256, help="Max generation length for Helsinki backend.")
    # Quality knob
    p.add_argument("--fix_missed", action="store_true", help="Re-translate unchanged FR rows that look French.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process(args)

#python preprocessing/LanguageNormalize2.py --translator google --threads 8 --fix_missed
