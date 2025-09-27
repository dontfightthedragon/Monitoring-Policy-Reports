# analysis/LLM/eval/make_review_sheet.py
import re, pandas as pd
from pathlib import Path

DEV = Path("analysis/LLM/artifacts/dev/labels_dev.csv")
LLM = Path("analysis/LLM/artifacts/dev/llm_predictions.csv")
KW  = Path("analysis/LLM/artifacts/dev/kw_predictions.csv")  # optional

# anchors and admin-ish cues (non-capturing groups)
ANCHOR = re.compile(
    r"\b(?:biodiversit|ecosystem|conservation|protected area|wildlife|forest|wetland|ramsar|"
    r"peatland|coral reef|seagrass|kba|natura 2000|land degradation|desertification|"
    r"invasive species|human[- ]wildlife conflict|poaching|anti[- ]poaching|cites|redd\+|"
    r"reforestation|afforestation|restoration|watershed|mangrove|park|habitat|species|"
    r"fisher(?:y|ies)|marine|coastal|mpa|river|lake|agroforestry)\b", re.I
)
NEG_ADMIN = re.compile(
    r"\b(?:usaid|travel|transportation|administrative|operating expense|miscellaneous|"
    r"contract|procurement|audit|payroll|voucher|overhead|benefits?|information redacted|"
    r"accordance.*accountability act)\b", re.I
)

d = pd.read_csv(DEV)
l = pd.read_csv(LLM)
m = d.merge(l, on="activity_id", how="inner")

# optional keyword baseline
if KW.exists():
    k = pd.read_csv(KW)
    m = m.merge(k, on="activity_id", how="left")
else:
    m["kw_primary_label"] = ""

def flags(row):
    txt = row.get("text_for_llm","") or ""
    fl = []
    if ANCHOR.search(txt) and row.get("llm_primary_label") == "NONE":
        fl.append("anchor_but_NONE")
    if NEG_ADMIN.search(txt):
        fl.append("adminish")
    if len(txt) < 200:
        fl.append("short")
    if row.get("kw_primary_label") and row["kw_primary_label"] != row["llm_primary_label"]:
        fl.append("kw_vs_llm")
    return "|".join(fl)

m["review_flags"] = m.apply(flags, axis=1)
m["n_flags"] = m["review_flags"].str.count(r"\|") + (m["review_flags"]!="").astype(int)

# priority: anchor_but_NONE first, then more flags, then shorter text (easier to review)
m["has_anchor_NONE"] = m["review_flags"].str.contains("anchor_but_NONE")
# sort_values cannot take a Series in 'by'; compute length first
m["text_len"] = m["text_for_llm"].astype(str).str.len()
m = m.sort_values(["has_anchor_NONE","n_flags","text_len"], ascending=[False, False, True])

cols = [
    "activity_id","year","donor","recipient",
    "llm_primary_label","llm_sdg15_targets","review_flags",
    "text_for_llm"
]
m[cols].to_csv("analysis/LLM/eval/review_sheet.csv", index=False)
print("Wrote analysis/LLM/eval/review_sheet.csv → focus on rows with flags")
