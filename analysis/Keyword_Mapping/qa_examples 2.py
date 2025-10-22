#!/usr/bin/env python3
# analysis/Keyword_Mapping/qa_examples.py
import re
import yaml
import pandas as pd
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[2]
TABLES  = ROOT / "reports" / "tables"
OUTDIR  = TABLES  # keep outputs with other tables

LABELED_PARQUET = TABLES / "sdg_rule_labels_per_project.parquet"
CRS_PARQUET     = ROOT / "dataset" / "temporaryFileToWorkWith" / "dataset/temporaryFileToWorkWith/CRS_textprep_FINAL.parquet"
RULES_YAML      = ROOT / "analysis" / "Keyword_Mapping" / "baselineKeywordMapping_1.yaml"

NEARMISS_PATTERNS = [
    r"\bgreen[- ]space connectivity\b",
    r"\bkeystone species\b",
    r"\becosystem[- ]based (planning|management|approach)\b",
    r"\bnature[- ]based solutions?\b",
]

def get_text_column(df: pd.DataFrame) -> str:
    for c in ("text_en", "text_original", "text_for_nlp", "text"):
        if c in df.columns:
            return c
    raise KeyError("No text column found (expected one of text_en, text_original, text_for_nlp, text).")

def compile_excludes(rules: dict) -> dict[str, re.Pattern]:
    """Return compiled exclude regex per target (only those with excludes)."""
    out = {}
    for tgt, block in rules.items():
        if isinstance(block, dict):
            exc_list = block.get("exclude") or []
            if exc_list:
                out[tgt] = re.compile(r"(?:%s)" % "|".join(exc_list), flags=re.I)
    return out

def main():
    # ---- Load inputs ----
    if not LABELED_PARQUET.exists():
        raise FileNotFoundError(f"Missing labels: {LABELED_PARQUET}")
    if not CRS_PARQUET.exists():
        raise FileNotFoundError(f"Missing CRS base: {CRS_PARQUET}")
    if not RULES_YAML.exists():
        raise FileNotFoundError(f"Missing rules YAML: {RULES_YAML}")

    long = pd.read_parquet(LABELED_PARQUET)  # activity_id, sdg_target, year, ...
    base = pd.read_parquet(CRS_PARQUET)
    text_col = get_text_column(base)
    base[text_col] = base[text_col].fillna("").astype(str)

    tagged_ids = set(long["activity_id"].unique())

    want_targets = ["SDG15.1", "SDG15.7", "SDG15.9"]
    tp_rows = []
    for tgt in want_targets:
        ids = (long.loc[long["sdg_target"] == tgt, "activity_id"]
                   .drop_duplicates()
                   .head(1)
                   .tolist())
        if not ids:
            continue
        j = base.loc[base["activity_id"].isin(ids), ["activity_id", text_col]].copy()
        if len(j):
            row = j.iloc[0]
            tp_rows.append({"target": tgt,
                            "activity_id": row["activity_id"],
                            "text_snippet": (row[text_col][:400] + "…") if len(row[text_col]) > 400 else row[text_col]})
    tp_df = pd.DataFrame(tp_rows)
    tp_out = OUTDIR / "qa_true_positive_examples.csv"
    tp_df.to_csv(tp_out, index=False)

    rules = yaml.safe_load(RULES_YAML.read_text())
    excludes = compile_excludes(rules)

    fp_rows = []
    # Search two targets that often need excludes (edit if desired)
    for tgt in ["SDG15.1", "SDG15.9"]:
        exc = excludes.get(tgt)
        if not exc:
            continue
        # candidates that match an exclude pattern
        cand = base.loc[base[text_col].str.contains(exc, na=False)]
        # …but are not tagged at all
        cand = cand.loc[~cand["activity_id"].isin(tagged_ids)]
        for _, r in cand.head(2).iterrows():  # keep it short; 2 per target
            fp_rows.append({
                "target": tgt,
                "activity_id": r["activity_id"],
                "why": "matches EXCLUDE pattern; correctly not tagged",
                "text_snippet": (r[text_col][:400] + "…") if len(r[text_col]) > 400 else r[text_col],
            })
    fp_df = pd.DataFrame(fp_rows)
    fp_out = OUTDIR / "qa_blocked_false_positive_examples.csv"
    fp_df.to_csv(fp_out, index=False)

    nm_mask = pd.Series(False, index=base.index)
    for pat in NEARMISS_PATTERNS:
        nm_mask = nm_mask | base[text_col].str.contains(pat, flags=re.I, na=False)
    near = base.loc[nm_mask & (~base["activity_id"].isin(tagged_ids)), ["activity_id", text_col]].head(3).copy()
    near["why"] = "indirect phrasing; not matched by current includes"
    near["text_snippet"] = near[text_col].apply(lambda s: (s[:400] + "…") if len(s) > 400 else s)
    near_df = near[["activity_id", "why", "text_snippet"]]
    near_out = OUTDIR / "qa_nearmiss_examples.csv"
    near_df.to_csv(near_out, index=False)

    print("[QA] Wrote examples:")
    print(f"  - {tp_out.name} (true positives)")
    print(f"  - {fp_out.name} (blocked false positives)")
    print(f"  - {near_out.name} (near-misses)")

if __name__ == "__main__":
    main()
