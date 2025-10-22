#!/usr/bin/env python3
# analysis/03_keywords_report.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

TABLES = Path("reports/tables"); TABLES.mkdir(parents=True, exist_ok=True)
FIGS   = Path("reports/figures"); FIGS.mkdir(parents=True, exist_ok=True)

def save_pct_bar(series, title, fname, ylabel="% of projects"):
    ax = (series * 100.0).round(2).plot(kind="bar", figsize=(8,5), title=title)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGS / fname)
    plt.close()

def save_pct_lines(df, x, title, fname, ylabel="% of projects"):
    (df.pivot(index=x, columns="sdg_target", values="project_share")
       .mul(100.0)
       .plot(figsize=(8,5), marker="o", title=title))
    plt.ylabel(ylabel); plt.xlabel(x.capitalize())
    plt.tight_layout(); plt.savefig(FIGS / fname); plt.close()

def must_have(*names):
    for n in names:
        p = TABLES / n
        if not p.exists():
            raise FileNotFoundError(f"Missing input: {p}. Run the keyword mapper first.")
    return [TABLES / n for n in names]

def main():
    # --- Inputs ---
    p_overall, p_ts, p_cov = must_have(
        "keyword_label_share_overall.csv",
        "keyword_label_share_timeseries.csv",
        "keyword_coverage_by_year.csv",
    )
    share_overall = pd.read_csv(p_overall)   # sdg_target, n_projects, project_share
    share_ts      = pd.read_csv(p_ts)        # year, sdg_target, n_projects, base_year_n, project_share
    cov_by_year   = pd.read_csv(p_cov)       # year, tagged_year_n, base_year_n, coverage_share

    # --- Figure A: % of ALL projects by target (tiny for rare targets) ---
    overall_series = (share_overall.set_index("sdg_target")["project_share"]
                      .sort_values(ascending=False))
    save_pct_bar(overall_series, "Keyword mapping: project share by SDG 15 target",
                 "kw_overall_project_share.pdf")

    # --- NEW Figure: composition among tagged (sums to 100%) ---
    tagged_total = share_overall["n_projects"].sum()
    comp = (share_overall.set_index("sdg_target")["n_projects"]
            .div(tagged_total).sort_values(ascending=False) * 100)
    plt.figure(figsize=(8,5))
    comp.plot(kind="bar")
    plt.ylabel("% of tagged projects")
    plt.title("Keyword mapping: composition of tagged projects by SDG 15 target")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGS / "kw_composition_among_tagged.pdf")
    plt.close()

    #  Figure B: time series of project shares (per target) 
    save_pct_lines(share_ts, x="year",
                   title="Keyword mapping: project share over time (by target)",
                   fname="kw_timeseries_project_share.pdf")

    #  Figure C: coverage over time (any SDG 15 hit)
    (cov_by_year.assign(coverage_pct=lambda d: d["coverage_share"]*100)
                .plot(x="year", y="coverage_pct", figsize=(8,5), marker="o",
                      title="Keyword mapping: coverage over time (any SDG 15 target)"))
    plt.ylabel("% of projects tagged")
    plt.tight_layout()
    plt.savefig(FIGS / "kw_coverage_over_time.pdf")
    plt.close()

    print("[REPORT] Wrote:",
          "kw_overall_project_share.pdf, kw_composition_among_tagged.pdf,",
          "kw_timeseries_project_share.pdf, kw_coverage_over_time.pdf")

if __name__ == "__main__":
    main()
