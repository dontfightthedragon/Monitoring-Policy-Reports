#!/usr/bin/env python3
# analysis/03_keywords_report.py

"""
Summary:
This script generates summary tables and figures for SDG15 keyword mapping results. It:
- Loads per-project and per-target label tables from previous rule-based mapping.
- Calculates total disbursements by SDG15 target and saves as CSV and bar plot.
- Creates time series of disbursements by target and saves as CSV and line plot.
- Identifies top donors and recipients for each SDG15 target (if available) and saves as CSV.
- Computes coverage statistics (share of projects tagged with SDG15 targets).
- All outputs are saved in the reports/tables and reports/figures directories.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

TABLES = Path("reports/tables")
FIGS   = Path("reports/figures")
FIGS.mkdir(parents=True, exist_ok=True)

def main():
    # --- Load data ---
    proj = pd.read_parquet(TABLES / "sdg_rule_labels_per_project.parquet")
    panel = pd.read_parquet(TABLES / "panel_by_target.parquet")

    # --- Totals by target ---
    totals = (
        panel.groupby("sdg_target")["usd_disb"]
        .sum()
        .sort_values(ascending=False)
        .round(1)
    )
    totals.to_csv(TABLES / "keyword_totals_sdg15.csv")
    print("[REPORT] Totals saved → keyword_totals_sdg15.csv")

    # --- Plot totals ---
    totals.plot(kind="bar", figsize=(8,5), title="Total Disbursements by SDG15 Target")
    plt.ylabel("USD (Millions)")
    plt.tight_layout()
    plt.savefig(FIGS / "keyword_totals_sdg15.png")
    plt.close()

    # --- Time series by target ---
    ts = panel.pivot(index="year", columns="sdg_target", values="usd_disb").fillna(0)
    ts.to_csv(TABLES / "keyword_timeseries_sdg15.csv")

    ts.plot(figsize=(10,6), marker="o", title="Disbursements over Time by SDG15 Target")
    plt.ylabel("USD (Millions)")
    plt.xlabel("Year")
    plt.tight_layout()
    plt.savefig(FIGS / "keyword_timeseries_sdg15.png")
    plt.close()

    # --- Top donors/recipients ---
    # Needs proj table with donor/recipient fields, if present
    cols = proj.columns
    if "donor_name" in cols:
        top_donors = (
            proj.groupby(["donor_name","sdg_target"])["usd_disbursement_defl"]
            .sum()
            .reset_index()
            .sort_values("usd_disbursement_defl", ascending=False)
            .groupby("sdg_target").head(10)
        )
        top_donors.to_csv(TABLES / "top10_donors_sdg15.csv", index=False)

    if "recipient_name" in cols:
        top_recipients = (
            proj.groupby(["recipient_name","sdg_target"])["usd_disbursement_defl"]
            .sum()
            .reset_index()
            .sort_values("usd_disbursement_defl", ascending=False)
            .groupby("sdg_target").head(10)
        )
        top_recipients.to_csv(TABLES / "top10_recipients_sdg15.csv", index=False)

    # --- Coverage stats ---
    base_n = proj["activity_id"].nunique()
    tagged_n = proj.dropna(subset=["sdg_target"])["activity_id"].nunique()
    cov = (tagged_n / base_n) if base_n else 0

    coverage_df = pd.DataFrame([{
        "base_projects": base_n,
        "tagged_projects": tagged_n,
        "coverage_share": cov
    }])
    coverage_df.to_csv(TABLES / "coverage_stats.csv", index=False)

    print("[REPORT] Coverage:", coverage_df)

if __name__ == "__main__":
    main()
