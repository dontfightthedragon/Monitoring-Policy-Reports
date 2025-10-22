# analysis/tagProjectsWithSDG1415.py
from __future__ import annotations
from pathlib import Path
import argparse, re, unicodedata
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def strip_accents(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn").lower()

EXTRA = {
    "SDG14.5": [
        r"\bmarine (?:parks?|sanctuar(?:y|ies)|areas?)\b",
        r"\bmarine (?:biodiversity|ecosystem)s?\b",
        r"\bmangroves?\b",                     # coastal → 14.5
        r"\baires? marines? (?:protegees?|parcs?|sanctuaires?)\b",
    ],
    "SDG15.1": [
        r"\bnational parks?\b",
        r"\bnature reserves?\b",
        r"\bprotected landscapes?\b",
        r"\bwatersheds?\b|\briver basins?\b",
        r"\bwetlands?\b|\bramsar\b",           # inland bias → 15.1
        r"\bhabitat restoration\b",
        r"\bgestion forestiere\b|\bparc(?:s)? nationa(?:l|ux)\b",
    ],
    "SDG15.5": [
        r"\bwildlife (?:crime|traffick(?:ing)?)\b",
        r"\banti[- ]poaching\b",
        r"\btrade in endangered species\b|\bcites\b",
    ],
}
for k,v in EXTRA.items():
    PATTERNS.setdefault(k,[]).extend(v)

def make_slug(s: str) -> str:
    s = strip_accents(s).upper()
    s = re.sub(r"[^A-Z0-9]+", "_", s).strip("_")
    return s or "UNKNOWN"

def main():
    ap = argparse.ArgumentParser(description="Tag projects with SDG 14/15 targets (robust).")
    ap.add_argument("--source", default=str(ROOT / "dataset/temporaryFileToWorkWith/CRS_2015_2023_analysis_ready.parquet"))
    ap.add_argument("--outdir",  default=str(ROOT / "reports/tables"))
    ap.add_argument("--money",   default=None, help="Override money column (auto-detect if not set)")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.source).copy()

    # Limit labeling universe to biodiversity projects only
    if "biodiversity" in df.columns:
        df = df[pd.to_numeric(df["biodiversity"], errors="coerce") >= 1].copy()

    # Pick money column
    money_c = args.money
    if not money_c:
        for c in ["usd_disbursement_defl", "usd_commitment_defl", "usd_disbursement", "usd_commitment"]:
            if c in df.columns:
                money_c = c; break
    if not money_c:
        raise SystemExit("No usd_* money column found in the analysis-ready file.")

    # Ensure recipient_code exists
    if "recipient_code" not in df.columns:
        if "recipient_name" not in df.columns:
            raise SystemExit("Missing both recipient_code and recipient_name.")
        print("[TAGS] 'recipient_code' missing → creating temporary codes from recipient_name.")
        df["recipient_code"] = df["recipient_name"].astype(str).map(make_slug)

    # Normalize text once
    df["text_norm"] = df["text"].fillna("").map(strip_accents)

    # Compile regex patterns (ignore case)
    COMPILED = {
        sdg: [re.compile(pat, flags=re.IGNORECASE) for pat in pats]
        for sdg, pats in PATTERNS.items()
    }

    base_cols = [c for c in ["activity_id","recipient_code","recipient_name","year",money_c] if c in df.columns]
    rows = []
    for sdg, plist in COMPILED.items():
        mask = pd.Series(False, index=df.index)
        for rx in plist:
            # str.contains accepts compiled regex; na=False avoids NaN warnings
            mask |= df["text_norm"].str.contains(rx, regex=True, na=False)
        tmp = df.loc[mask, base_cols].copy()
        if not tmp.empty:
            tmp["sdg_target"] = sdg
            rows.append(tmp)

    long = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=base_cols + ["sdg_target"])
    if money_c in long.columns:
        long[money_c] = pd.to_numeric(long[money_c], errors="coerce")

    proj_path = outdir / "sdg_rule_labels_per_project.parquet"
    long.to_parquet(proj_path, index=False)

    # Aggregate safely and use observed=True
    keys = ["recipient_code","recipient_name","year","sdg_target"]
    if long.empty:
        by_target = pd.DataFrame(columns=keys + ["usd_disb","proj_n"])
    else:
        g = long.groupby(keys, observed=True, sort=False)
        money_sum = g[money_c].sum().rename("usd_disb")
        counts    = g.size().rename("proj_n")
        by_target = pd.concat([money_sum, counts], axis=1).reset_index()

    p_parquet = outdir / "panel_by_target.parquet"
    p_csv     = outdir / "panel_by_target.csv"
    by_target.to_parquet(p_parquet, index=False)
    by_target.to_csv(p_csv, index=False)

    # Global time series by target
    if by_target.empty:
        ts = pd.DataFrame(columns=["year","sdg_target","usd_disb"])
    else:
        ts = by_target.groupby(["year","sdg_target"], observed=True, sort=False)["usd_disb"].sum().reset_index()
    ts.pivot(index="year", columns="sdg_target", values="usd_disb").to_csv(outdir / "timeseries_target_global.csv")

    # quick sanity prints
    try:
        src_bio_n = df["activity_id"].nunique()
        tagged_n  = long["activity_id"].nunique()
        cov = (tagged_n / src_bio_n) if src_bio_n else 0.0
        print(f"[TAGS] Coverage among BIO projects: {cov:.1%} ({tagged_n}/{src_bio_n})")
        print("[TAGS] Totals by target (USD):")
        print(by_target.groupby("sdg_target", observed=True)["usd_disb"].sum().round(1).sort_values(ascending=False))
    except Exception:
        pass

    print(f"[TAGS] Wrote:\n  - {proj_path}\n  - {p_parquet} (+.csv)\n  - {outdir/'timeseries_target_global.csv'}")

if __name__ == "__main__":
    main()
