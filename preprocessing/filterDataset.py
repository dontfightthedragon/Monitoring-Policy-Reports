# preprocessing/FilteredDataset.py
from __future__ import annotations
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT  = ROOT / "dataset/rawDataset/CRS.parquet"
OUTPUT = ROOT / "dataset/temporaryFileToWorkWith/CRS_filtered.parquet"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

KEEP_LANGUAGES = {"en", "fr"}
MIN_TEXT_CHARS = 20

def basic_clean(text: str) -> str:
    if not isinstance(text, str): return ""
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text

print(f"[FILTER] Reading: {INPUT}")

# Read only the columns we need
cols = [
    "year","flow_name","donor_name", "recipient_code", "recipient_name",
    "project_title","short_description","long_description",
    "biodiversity","crs_id","project_number" ,"usd_commitment_defl","usd_disbursement_defl"
]
df = pd.read_parquet(INPUT, engine="pyarrow", columns=cols)

# Robust year filter
y = pd.to_numeric(df["year"], errors="coerce")
df = df[y >= 2015]

# ODA only
df = df[df["flow_name"].isin(["ODA Grants", "ODA Loans"])]
print(f"[FILTER] Rows after year+ODA: {len(df):,}")

# Build unified text
for c in ["project_title","short_description","long_description"]:
    if c not in df.columns: df[c] = ""
df["text"] = (
    df["project_title"].fillna("").astype(str) + " " +
    df["short_description"].fillna("").astype(str) + " " +
    df["long_description"].fillna("").astype(str)
).str.strip().map(basic_clean)

# Drop too-short texts
before = len(df)
df = df[df["text"].str.len() >= MIN_TEXT_CHARS]
print(f"[FILTER] Removed short texts (<{MIN_TEXT_CHARS}): {before - len(df):,}")

# Language detection -> keep en/fr
def detect_langs(series: pd.Series) -> pd.Series:
    try:
        from lingua import Language, LanguageDetectorBuilder
        det = LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.FRENCH).build()
        def one(s: str) -> str:
            if not isinstance(s, str) or not s.strip(): return "unknown"
            lang = det.detect_language_of(s[:800])
            if lang == Language.ENGLISH: return "en"
            if lang == Language.FRENCH:  return "fr"
            return "unknown"
    except Exception:
        try:
            from langdetect import detect, DetectorFactory
            DetectorFactory.seed = 0
            def one(s: str) -> str:
                if not isinstance(s, str) or not s.strip(): return "unknown"
                try: return detect(s[:800])
                except Exception: return "unknown"
        except Exception:
            # last-resort heuristic
            def one(s: str) -> str:
                s = (" " + (s or "").lower() + " ")
                if any(w in s for w in (" le ", " la ", " et ", " est ", " des ", " une ")): return "fr"
                if any(w in s for w in (" the ", " and ", " is ", " of ", " for ")): return "en"
                return "unknown"

    short = series.astype(str).str.slice(0, 800)
    uniq  = short.drop_duplicates()
    m = uniq.map(one).to_dict()
    return short.map(lambda s: m.get(s, "unknown"))

# df["language"] = detect_langs(df["text"])
# before_lang = len(df)
# df = df[df["language"].isin(KEEP_LANGUAGES)]
# print(f"[FILTER] Kept {sorted(KEEP_LANGUAGES)}: removed {before_lang - len(df):,}")

df["language"] = "unknown"


# Ensure activity_id from crs_id/project_number (fallback to index)
df["activity_id"] = df["crs_id"].where(df["crs_id"].notna(), df["project_number"])
missing = df["activity_id"].isna()
if missing.any():
    df.loc[missing, "activity_id"] = df.index[missing].astype("int64")
df["activity_id"] = df["activity_id"].astype(str)

# FINAL COLUMNS (includes flow_name + raw text fields)

# Only keep columns that exist in df to avoid KeyError
keep_cols = [
    "year","activity_id","donor_name","recipient_name","recipient_code",
    "flow_name","flow_code","donor_code",  
    "project_title","short_description","long_description",
    "text","biodiversity","language","usd_commitment_defl","usd_disbursement_defl"
]
keep_cols = [col for col in keep_cols if col in df.columns]
df = df[keep_cols].copy()

# Light compaction
for c in ["donor_name","recipient_name","flow_name","language"]:
    df[c] = df[c].astype("category")

# Save
df.to_parquet(OUTPUT, engine="pyarrow", index=False)
print(f"[FILTER] Saved -> {OUTPUT}")
print(f"[FILTER] Columns: {list(df.columns)}")
print(f"[FILTER] Rows (pre-dedup): {len(df):,}")
