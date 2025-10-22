# analysis/createPanel.py
from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def pick(df: pd.DataFrame, names: list[str]) -> str:
    m = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in m:
            return m[n.lower()]
    raise SystemExit(f"Missing required column from {names}. Got: {list(df.columns)[:15]} ...")

def main():
    ap = argparse.ArgumentParser(description="Build recipient×year panel + summary tables.")
    ap.add_argument("--source",
        default=str(ROOT / "dataset/temporaryFileToWorkWith/CRS_2015_2023_analysis_ready.parquet"),
        help="Path to analysis-ready Parquet (from addBioWeights.py)")
    ap.add_argument("--outdir",
        default=str(ROOT / "reports/tables"),
        help="Directory for outputs")
    args = ap.parse_args()

    src = Path(args.source)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[PANEL] Reading: {src.resolve()}")
    if not src.exists():
        raise SystemExit(f"Source not found: {src}")

    df = pd.read_parquet(src)

    # Robust column picks
    year_c  = pick(df, ["year"])
    biod_c  = pick(df, ["biodiversity"])
    act_c   = pick(df, ["activity_id"])

    rec_code = pick(df, ["recipient_code", "recipient_name"])  # fallback to name if code missing
    rec_name = pick(df, ["recipient_name"])
    

    if rec_code == rec_name:
        group_cols = [rec_name, year_c]
        print(f"[PANEL] Using recipient_name as grouping key (recipient_code not available)")
    else:
        group_cols = [rec_code, rec_name, year_c]
    money_c = next((c for c in ["usd_disbursement_defl","usd_commitment_defl",
                                "usd_disbursement","usd_commitment"] if c in df.columns), None)
    if money_c is None:
        raise SystemExit("No money column (usd_*) found in the analysis-ready file.")

    print(f"[PANEL] Rows in source: {len(df):,}")
    print(f"[PANEL] Using money column: {money_c}")

    yr_tbl = (df.groupby(year_c)[["bio_weight_bin","bio_weight_wtd"]]
                .sum(numeric_only=True).round(1))
    yr_path = outdir / "bio_yearly_totals.csv"
    yr_tbl.to_csv(yr_path)
    mk_tbl = (df.groupby([year_c, biod_c]).size()
                .unstack(fill_value=0)
                .rename(columns={2:"principal(2)",1:"significant(1)",0:"none(0)"}))
    mk_path = outdir / "bio_marker_breakdown.csv"
    mk_tbl.to_csv(mk_path)
    print(f"[PANEL] Wrote:\n  - {yr_path.resolve()}\n  - {mk_path.resolve()}")

    bio = df[df[biod_c] >= 1].copy()
    print(f"[PANEL] Bio-slice rows: {len(bio):,}")
    
    # Data quality checks
    print(f"[PANEL] Data quality checks:")
    print(f"  - Missing {money_c}: {bio[money_c].isna().sum():,} ({bio[money_c].isna().mean()*100:.1f}%)")
    print(f"  - Zero {money_c}: {(bio[money_c] == 0).sum():,} ({(bio[money_c] == 0).mean()*100:.1f}%)")
    
    # Fix pandas groupby issue by using proper aggregation syntax
    panel = bio.groupby(group_cols, as_index=False, observed=True).agg({
        money_c: "sum",
        act_c: "count", 
        "bio_weight_bin": "sum",
        "bio_weight_wtd": "sum"
    }).rename(columns={
        money_c: "usd_disb",
        act_c: "proj_n",
        "bio_weight_bin": "bio_bin",
        "bio_weight_wtd": "bio_wtd"
    })
    
    # Add derived metrics
    panel['avg_project_size'] = panel['usd_disb'] / panel['proj_n']
    panel['bio_intensity'] = panel['bio_wtd'] / panel['proj_n']  # average bio weight per project

    panel_pq  = outdir / "panel_recipient_year.parquet"
    panel_csv = outdir / "panel_recipient_year.csv"
    panel.to_parquet(panel_pq, index=False)
    panel.to_csv(panel_csv, index=False)
    print(f"[PANEL] Saved panel ({panel.shape[0]:,} rows, {panel.shape[1]} cols):\n  - {panel_pq.resolve()}\n  - {panel_csv.resolve()}")
    
    # Additional summary statistics
    print(f"\n[PANEL] Summary:")
    print(f"  - Unique recipients: {panel['recipient_name'].nunique():,}")
    print(f"  - Year range: {panel['year'].min()}-{panel['year'].max()}")
    print(f"  - Total biodiversity projects: {panel['proj_n'].sum():,}")
    print(f"  - Total disbursements: ${panel['usd_disb'].sum():,.0f}M")
    print(f"  - Average projects per recipient-year: {panel['proj_n'].mean():.1f}")
    
    # Top recipients by total projects
    top_recipients = panel.groupby('recipient_name', observed=True)['proj_n'].sum().sort_values(ascending=False).head(5)
    print(f"\n[PANEL] Top 5 recipients by total projects:")
    for recipient, count in top_recipients.items():
        print(f"  - {recipient}: {count:,} projects")

if __name__ == "__main__":
    main()

