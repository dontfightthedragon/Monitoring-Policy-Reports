from __future__ import annotations
import argparse, yaml, re, pandas as pd
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import sys

_FILE = Path(__file__).resolve()
_REPO_ROOT = _FILE.parents[2]  # .../Thesis_rep
try:
    from analysis.Other.common import seed_all, load_textprep, ensure_outdir
except ModuleNotFoundError:
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from analysis.Other.common import seed_all, load_textprep, ensure_outdir

ROOT = _REPO_ROOT

def resolve_path(p: Optional[str], default_rel: str) -> Path:
    if p:
        q = Path(p).expanduser()
        return (q if q.is_absolute() else (ROOT / q)).resolve()
    return (ROOT / default_rel).resolve()

def compile_rules(d: Dict[str, dict]) -> Dict[str, Tuple[re.Pattern | None, re.Pattern | None]]:
    """
    YAML format:
      SDG15.1:
        include: ["wetlands?", "ramsar", "habitat restoration"]
        exclude: ["industrial parks?"]
    NOTE: entries are plain regex strings (already escaped as needed).
    """
    rx: Dict[str, Tuple[re.Pattern | None, re.Pattern | None]] = {}
    for k, v in d.items():
        if not isinstance(v, dict):
            continue  # defensively skip stray non-dict entries
        inc_list = v.get("include", []) or []
        exc_list = v.get("exclude", []) or []
        inc = re.compile(r"(?:%s)" % "|".join(inc_list), re.I) if inc_list else None
        exc = re.compile(r"(?:%s)" % "|".join(exc_list), re.I) if exc_list else None
        rx[k] = (inc, exc)
    return rx

def apply_rules_to_text(s: str, rx: dict) -> List[str]:
    if not isinstance(s, str) or not s:
        return []
    hits: List[str] = []
    for tag, (inc, exc) in rx.items():
        if inc and inc.search(s):
            if not (exc and exc.search(s)):
                hits.append(tag)
    return hits

def pick_money_column(df: pd.DataFrame) -> str:
    for c in ["usd_disbursement_defl","usd_commitment_defl","usd_disbursement","usd_commitment"]:
        if c in df.columns:
            return c
    raise KeyError("No usd_* money column found in input.")

def main():
    ap = argparse.ArgumentParser(description="Rule-based SDG keyword mapping (SDG15-focused).")
    ap.add_argument("--rules",  type=str,
                    default="analysis/Keyword_Mapping/baselineKeywordMapping_1.yaml",
                    help="Path to YAML rules file")
    ap.add_argument("--input",  type=str,
                    default="dataset/temporaryFileToWorkWith/CRS_enfr_language_normalized_use_this.parquet",
                    help="Input parquet")
    ap.add_argument("--outdir", type=str, default="reports/tables", help="Output directory")
    ap.add_argument("--both-fields", action="store_true",
                    help="If set and text_original exists: search on text_en + text_original (EN+FR).")
    ap.add_argument("--bio-coverage-only", action="store_true",
                    help="Also print coverage among biodiversity>=1 subset.")
    args = ap.parse_args()

    seed_all(1)

    # Resolve paths
    rules_path = resolve_path(args.rules, "rules/sdg_keywords.yaml")
    inp_path   = resolve_path(args.input, "dataset/temporaryFileToWorkWith/CRS_textprep_FINAL.parquet")
    out_dir    = resolve_path(args.outdir, "reports/tables")
    ensure_outdir(str(out_dir))

    if not rules_path.exists():
        raise FileNotFoundError(f"Rules YAML not found: {rules_path}")

    
    df = load_textprep(str(inp_path))  
    if "text_en" in df.columns:
        df["text_for_nlp"] = df["text_en"].fillna("").astype(str)
        if args.both_fields and "text_original" in df.columns:
            df["text_for_nlp"] = (df["text_for_nlp"] + " " + df["text_original"].fillna("").astype(str)).str.strip()
    elif "text_for_nlp" in df.columns:
        df["text_for_nlp"] = df["text_for_nlp"].fillna("").astype(str)
    elif "text" in df.columns:
        df["text_for_nlp"] = df["text"].fillna("").astype(str)
    else:
        raise KeyError("No suitable text column found (need text_en or text_for_nlp or text).")

    # ---- Deduplicate activity_id by keeping the row with longest text_for_nlp
    if df["activity_id"].duplicated().any():
        df["_tlen"] = df["text_for_nlp"].str.len()
        df = (df.sort_values("_tlen", ascending=False)
                .drop_duplicates(subset="activity_id", keep="first")
                .drop(columns="_tlen"))

    money_c = pick_money_column(df)

    
    rules = yaml.safe_load(rules_path.read_text())
    rx = compile_rules(rules)

    labels = df["text_for_nlp"].map(lambda s: apply_rules_to_text(s, rx))

    cols = ["activity_id","year", money_c]
    for c in ("donor_name","recipient_name"):
        if c in df.columns:
            cols.append(c)
    long = df[cols].copy()
    long["sdg_target"] = labels
    long = (long.explode("sdg_target")
                .dropna(subset=["sdg_target"])
                .drop_duplicates(subset=["activity_id","sdg_target"]))

    proj_path = out_dir / "sdg_rule_labels_per_project.parquet"
    long.to_parquet(proj_path, index=False)

    base_n = df["activity_id"].nunique()
    per_target_counts = (long.groupby("sdg_target")["activity_id"].nunique()
                             .sort_values(ascending=False))
    per_target_share = (per_target_counts / base_n).rename("project_share")
    proj_stats = (pd.concat([per_target_counts.rename("n_projects"), per_target_share], axis=1)
                    .reset_index())
    proj_stats.to_csv(out_dir / "keyword_label_share_overall.csv", index=False)

    # Time series: project share by year × target
    base_by_year = df.groupby("year")["activity_id"].nunique().rename("base_year_n")
    counts_by_year = (long.groupby(["year","sdg_target"])["activity_id"]
                          .nunique()
                          .rename("n_projects")
                          .reset_index())
    ts = counts_by_year.merge(base_by_year.reset_index(), on="year", how="left")
    ts["project_share"] = ts["n_projects"] / ts["base_year_n"]
    ts.to_csv(out_dir / "keyword_label_share_timeseries.csv", index=False)

    # Coverage (any SDG15 target matched)
    tagged_n = long["activity_id"].nunique()
    coverage = (tagged_n / base_n) if base_n else 0.0
    cov_by_year = (
        long.groupby("year")["activity_id"].nunique().rename("tagged_year_n").reset_index()
        .merge(base_by_year.reset_index(), on="year", how="right")
    )
    cov_by_year["coverage_share"] = cov_by_year["tagged_year_n"].fillna(0) / cov_by_year["base_year_n"]
    cov_by_year.to_csv(out_dir / "keyword_coverage_by_year.csv", index=False)

    if len(long) == 0:
        panel_money = pd.DataFrame(columns=["year","sdg_target","usd_disb"])
    else:
        panel_money = (
            long.groupby(["year","sdg_target"], as_index=False)[money_c]
                .sum().rename(columns={money_c:"usd_disb"})
        )
    panel_money.to_parquet(out_dir / "panel_by_target.parquet", index=False)
    panel_money.to_csv(out_dir / "timeseries_target_global.csv", index=False)

    # Console report
    print("[KEYWORDS] Using rules:", rules_path)
    print(f"[KEYWORDS] Coverage (any target): {coverage:.1%} ({tagged_n}/{base_n})")
    print("[KEYWORDS] Top targets by project share:")
    print(proj_stats.sort_values("project_share", ascending=False).head(10))

    # biodiversity-only coverage
    if args.bio_coverage_only and "biodiversity" in df.columns:
        bio_mask = pd.to_numeric(df["biodiversity"], errors="coerce").fillna(0) >= 1
        base_bio = df.loc[bio_mask, "activity_id"].nunique()
        tagged_bio = (
            df.loc[bio_mask, ["activity_id"]].merge(
                long[["activity_id"]].drop_duplicates(), on="activity_id", how="inner"
            )["activity_id"].nunique()
        )
        bio_cov = (tagged_bio / base_bio) if base_bio else 0.0
        print(f"[KEYWORDS] BIO-only coverage: {bio_cov:.1%} ({tagged_bio}/{base_bio})")

    # QA sample per target
    sample_csv = out_dir / "sdg_rule_sample10_per_target.csv"
    if len(long) > 0:
        # Prefer English text for inspection
        snippet_src = "text_en" if "text_en" in df.columns else "text_for_nlp"
        j = long.merge(df[["activity_id", snippet_src]], on="activity_id", how="left")
        samp = j.groupby("sdg_target").head(10).copy()
        samp = samp.rename(columns={snippet_src: "text_snippet"})
        samp.to_csv(sample_csv, index=False)
        print(f"[KEYWORDS] Wrote sample → {sample_csv}")

if __name__ == "__main__":
    main()
