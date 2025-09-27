# analysis/LLM/makeDevSet_pos_aware.py
import re, unicodedata, pandas as pd
from pathlib import Path
from typing import Iterable, List
import tiktoken

RANDOM_SEED = 7
MAX_TOKENS = 1200
HEAD_SENTENCES = 5     # keep first N sentences unconditionally
CTX_WINDOW = 1         # keep +/- this many sentences around trigger hits

DATA = Path("dataset/temporaryFileToWorkWith/CRS_textprep.parquet")
OUT  = Path("/Users/johannahofmann/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Bachelorarbeit/Thesis_rep/analysis/LLM/labels_dev.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

use_cols = ["activity_id","year","text_en","text_clean","donor_name","recipient_name","biodiversity"]
df = (
    pd.read_parquet(DATA, columns=use_cols)
      .rename(columns={"text_en":"text","donor_name":"donor","recipient_name":"recipient"})
)

# ---------- helpers
def normalize(s: str) -> str:
    if not isinstance(s, str): return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", s).strip()

def safe_sample(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    n = min(n, len(frame))
    return frame.iloc[0:0] if n <= 0 else frame.sample(n, random_state=seed)

def split_sentences(text: str) -> List[str]:
    # simple, robust-ish splitter (no external deps)
    text = text.replace("\r", " ").replace("\n", " ")
    parts = re.split(r"(?<=[\.\?!])\s+(?=[^\s])", text)
    # merge tiny fragments caused by abbreviations
    out = []
    for p in parts:
        if out and len(p) < 3:
            out[-1] += " " + p
        else:
            out.append(p)
    return [p.strip() for p in out if p.strip()]

# non-capturing outer groups to avoid pandas regex warnings
ANCHOR = re.compile(
    r"\b(?:"
    r"biodiversit(?:y|e)?|ecosystem(?:s)?|conservation|protected area(?:s)?|wildlife|forest(?:s)?|"
    r"wetland(?:s)?|ramsar|peatland(?:s)?|coral reef(?:s)?|seagrass|kba(?:s)?|natura 2000|"
    r"land degradation|desertification|invasive species|human[- ]wildlife conflict|"
    r"poaching|anti[- ]poaching|cites|redd\+|reforestation|afforestation|restoration(?: ecology)?|"
    r"watershed|mangrove(?:s)?|park(?:s)?|habitat(?:s)?|species|fisher(?:y|ies)|marine|coastal|"
    r"marine protected area(?:s)?|mpa(?:s)?|river(?:s)?|lake(?:s)?|agroforestry"
    r")\b",
    re.I
)
NEG_ADMIN = re.compile(
    r"\b(?:usaid|travel|transportation|administrative|operating expense|miscellaneous|"
    r"contract|procurement|audit|payroll|voucher|overhead|benefit(?:s)?|"
    r"information redacted|accordance.*accountability act)\b",
    re.I
)

# ---------- build text for LLM + normalized copy
df["text_for_llm"] = df["text"].fillna(df["text_clean"]).fillna("").astype(str)
df["text_norm"] = df["text_for_llm"].map(normalize)

# drop empties/very short
df = df[df["text_norm"].str.len() >= 30]

# ---------- flags
df["is_anchor"] = df["text_norm"].str.contains(ANCHOR, na=False)
df["is_admin"]  = df["text_norm"].str.contains(NEG_ADMIN, na=False)

# ---------- year buckets (optional)
def add_year_bucket(y):
    try: y = int(y)
    except Exception: return "unknown"
    if y <= 2016: return "≤2016"
    if y <= 2019: return "2017–2019"
    if y <= 2021: return "2020–2021"
    return "2022+"
df["year_bucket"] = df["year"].map(add_year_bucket)

# ---------- strata (disjoint)
rng = RANDOM_SEED
pos_pool  = df[(df["is_anchor"]) & (~df["is_admin"])].copy()
hard_pool = df[(df["is_anchor"]) & (df["is_admin"])].copy()
bg_pool   = df[~df["is_anchor"]].copy()

def remove_ids(pool: pd.DataFrame, ids) -> pd.DataFrame:
    return pool[~pool["activity_id"].isin(set(ids))]

def stratified_sample(frame: pd.DataFrame, by: str, n_per_group: int, seed: int) -> pd.DataFrame:
    parts = []
    for _, g in frame.groupby(by, dropna=False):
        parts.append(safe_sample(g, n_per_group, seed))
    return pd.concat(parts, ignore_index=True) if parts else frame.iloc[0:0]

pos  = stratified_sample(pos_pool,  "year_bucket", 40, rng)  # total ~160
pos_ids = pos["activity_id"]

hard_pool = remove_ids(hard_pool, pos_ids)
hard = stratified_sample(hard_pool, "year_bucket", 20, rng)  # total ~80
hard_ids = hard["activity_id"]

bg_pool = remove_ids(bg_pool, pd.concat([pos_ids, hard_ids], ignore_index=True))
rand = stratified_sample(bg_pool,   "year_bucket", 15, rng)  # total ~60

dev = pd.concat([pos, hard, rand], ignore_index=True)
dev = dev.drop_duplicates("activity_id").sample(frac=1.0, random_state=rng).reset_index(drop=True)

# ---------- export columns
cols_out = ["activity_id","year","donor","recipient","text_for_llm"]
dev_out = dev[cols_out].copy()

# ---------- keyword-aware shortening (preserve trigger sentences)
def get_encoder():
    try: return tiktoken.get_encoding("o200k_base")
    except Exception: return tiktoken.get_encoding("cl100k_base")
ENC = get_encoder()

def truncate_tokens(text: str, max_tokens: int = MAX_TOKENS) -> str:
    if not isinstance(text, str): return ""
    toks = ENC.encode(text)
    return text if len(toks) <= max_tokens else ENC.decode(toks[:max_tokens])

def shorten_preserve_triggers(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    sents = split_sentences(text)
    if len(sents) <= HEAD_SENTENCES:
        return text

    keep = set(range(min(HEAD_SENTENCES, len(sents))))  # head
    # find trigger sentences on normalized text
    norm_sents = [normalize(s) for s in sents]
    trigger_idx = [i for i, s in enumerate(norm_sents) if ANCHOR.search(s)]
    for i in trigger_idx:
        for j in range(max(0, i-CTX_WINDOW), min(len(sents), i+CTX_WINDOW+1)):
            keep.add(j)

    # keep in original order
    shortened = " ".join(sents[i] for i in sorted(keep))
    return shortened

# audit copy, then shorten + final token cap
dev_out["text_original"] = dev_out["text_for_llm"]
dev_out["text_for_llm"] = dev_out["text_for_llm"].map(shorten_preserve_triggers)

n_short = (dev_out["text_for_llm"] != dev_out["text_original"]).sum()
# final hard cap
_before = dev_out["text_for_llm"].copy()
dev_out["text_for_llm"] = dev_out["text_for_llm"].map(truncate_tokens)
n_trunc = (dev_out["text_for_llm"] != _before).sum()

print(f"Shortened {n_short} rows; then truncated {n_trunc} rows to ≤ {MAX_TOKENS} tokens")

# ---------- add blank label cols and export
dev_out["primary_label"] = ""
dev_out["sdg15_targets"] = ""
dev_out["rationale"] = ""

dev_out = dev_out.astype({
    "activity_id":"string","year":"string","donor":"string","recipient":"string",
    "text_for_llm":"string","primary_label":"string","sdg15_targets":"string","rationale":"string"
})
dev_out.to_csv(OUT, index=False, encoding="utf-8", lineterminator="\n")
print(f"Wrote {len(dev_out)} rows to {OUT}")
