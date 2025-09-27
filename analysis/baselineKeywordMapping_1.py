
from __future__ import annotations
import argparse, yaml, re, pandas as pd
from pathlib import Path
from common import seed_all, load_textprep, ensure_outdir
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]

def resolve_path(p: Optional[str], default_rel: str) -> Path:
    if p:
        q = Path(p).expanduser()
        return (q if q.is_absolute() else (ROOT / q)).resolve()
    return (ROOT / default_rel).resolve()

def compile_rules(d: dict[str, dict]) -> dict[str, tuple[re.Pattern | None, re.Pattern | None]]:
    """
    YAML format:
      SDG15.1:
        include: ["wetlands?", "ramsar", "habitat restoration"]
        exclude: ["industrial parks?"]
    NOTE: entries are treated as REGEX; do not escape here.
    """
    rx = {}
    for k, v in d.items():
        inc_list = v.get("include", []) or []
        exc_list = v.get("exclude", []) or []
        inc = re.compile(r"(?:%s)" % "|".join(inc_list), re.I) if inc_list else None
        exc = re.compile(r"(?:%s)" % "|".join(exc_list), re.I) if exc_list else None
        rx[k] = (inc, exc)
    return rx

def apply_rules_to_text(s: str, rx: dict) -> list[str]:
    if not isinstance(s, str) or not s:
        return []
    hits = []
    for tag, (inc, exc) in rx.items():
        if inc and inc.search(s):
            if not (exc and exc.search(s)):
                hits.append(tag)
    return hits

def pick_money_column(df: pd.DataFrame) -> str:
    for c in ["usd_disbursement_defl","usd_commitment_defl","usd_disbursement","usd_commitment"]:
        if c in df.columns: return c
    raise KeyError("No usd_* money column found in input.")

def main():
    ap = argparse.ArgumentParser(description="Rule-based SDG keyword mapping (SDG15-focused).")
    ap.add_argument("--rules",  type=str, default="rules/sdg_keywords.yaml", help="Path to YAML rules file")
    ap.add_argument("--input",  type=str, default="dataset/temporaryFileToWorkWith/CRS_textprep.parquet", help="Input parquet (uses text_clean)")
    ap.add_argument("--outdir", type=str, default="reports/tables", help="Output directory")
    ap.add_argument("--both-fields", action="store_true",
                    help="Match against BOTH text_clean and text_en (joined). Default: text_clean only.")
    ap.add_argument("--bio-coverage-only", action="store_true",
                    help="Also print coverage among biodiversity>=1 subset.")
    args = ap.parse_args()

    seed_all(1)

    # Resolve paths
    rules_path = resolve_path(args.rules, "rules/sdg_keywords.yaml")
    inp_path   = resolve_path(args.input, "dataset/temporaryFileToWorkWith/CRS_textprep.parquet")
    out_dir    = resolve_path(args.outdir, "reports/tables")
    ensure_outdir(str(out_dir))

    if not rules_path.exists():
        raise FileNotFoundError(f"Rules YAML not found: {rules_path}")

    # Load data (load_textprep ensures df['text_for_nlp'] = text_clean if present)
    df = load_textprep(str(inp_path))
    money_c = pick_money_column(df)

    # Select text to search
    if args.both_fields and "text_en" in df.columns:
        # Join clean + original English for robust recall
        search_text = (df["text_for_nlp"].fillna("") + " " + df["text_en"].fillna("")).astype(str)
    else:
        search_text = df["text_for_nlp"].astype(str)

    # Load & compile rules
    rules = yaml.safe_load(rules_path.read_text())
    rx = compile_rules(rules)

    # Apply rules
    labels = search_text.map(lambda s: apply_rules_to_text(s, rx))

    # Long table (explode multilabels)
    long = df[["activity_id","year", money_c]].copy()
    long["sdg_target"] = labels
    long = long.explode("sdg_target").dropna(subset=["sdg_target"])

    # Save per-project labels
    proj_path = out_dir / "sdg_rule_labels_per_project.parquet"
    long.to_parquet(proj_path, index=False)

    # Panels by year/target
    if len(long) == 0:
        panel = pd.DataFrame(columns=["year","sdg_target","usd_disb"])
    else:
        panel = (
            long.groupby(["year","sdg_target"], as_index=False)[money_c]
                .sum().rename(columns={money_c:"usd_disb"})
        )
    panel.to_parquet(out_dir / "panel_by_target.parquet", index=False)
    panel.to_csv(out_dir / "timeseries_target_global.csv", index=False)

    # Console report
    totals = (panel.groupby("sdg_target")["usd_disb"].sum()
                    .sort_values(ascending=False).round(1)) if len(panel) else pd.Series(dtype=float)
    base_n   = df["activity_id"].nunique()
    tagged_n = long["activity_id"].nunique()
    coverage = (tagged_n / base_n) if base_n else 0.0

    print("[KEYWORDS] Using rules:", rules_path)
    print("[KEYWORDS] Totals by target (USD):")
    print(totals if len(totals) else "(no matches)")
    print(f"[KEYWORDS] Coverage: {coverage:.1%} ({tagged_n}/{base_n}) over analysis base")

    # Optional: biodiversity-only coverage
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

    # Write a small sample per target for QA
    sample_csv = out_dir / "sdg_rule_sample10_per_target.csv"
    if len(long) > 0:
        # join back a snippet to inspect (use text_en if available, otherwise text_for_nlp)
        snippet_src = "text_en" if "text_en" in df.columns else "text_for_nlp"
        j = long.merge(df[["activity_id", snippet_src]], on="activity_id", how="left")
        samp = j.groupby("sdg_target").head(10).copy()
        samp = samp.rename(columns={snippet_src: "text_snippet"})
        samp.to_csv(sample_csv, index=False)

        print(f"[KEYWORDS] Wrote sample → {sample_csv}")

        print(long.groupby("sdg_target")["activity_id"].nunique().sort_values(ascending=False))


if __name__ == "__main__":
    main()




