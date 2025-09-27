#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT   = Path(__file__).resolve().parents[1]
TABDIR = ROOT / "reports" / "tables"
FIGDIR = ROOT / "reports" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# ---- Load your CSVs ----
totals = pd.read_csv(TABDIR / "keyword_totals_sdg15.csv")           # sdg_target, usd_disb
ts     = pd.read_csv(TABDIR / "keyword_timeseries_sdg15.csv")       # year, SDG15.1,...

# ---- Table (LaTeX) for totals ----
# Make “USD M” with 1 decimal
totals_fmt = totals.copy()
totals_fmt["usd_disb"] = totals_fmt["usd_disb"].map(lambda x: f"{x:,.1f}")
(TABDIR / "keyword_totals_sdg15.tex").write_text(
    totals_fmt.to_latex(index=False, caption="Disbursements identified by keyword rules for SDG 15 (2015--2023), million constant USD.",
                        label="tab:sdg15_keyword_totals", escape=False)
)

# ---- Figure: totals bar chart ----
plt.figure(figsize=(8,4.5))
order = totals.sort_values("usd_disb", ascending=False)
plt.bar(order["sdg_target"], order["usd_disb"])
plt.ylabel("Disbursements (USD, millions)")
plt.title("SDG 15 disbursements identified via keyword mapping (2015–2023)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(FIGDIR / "keyword_totals_sdg15.png", dpi=200)
plt.close()

# ---- Figure: time series ----
plt.figure(figsize=(8.5,4.8))
for col in ts.columns:
    if col != "year":
        plt.plot(ts["year"], ts[col], label=col)
plt.legend(ncol=2, fontsize=8)
plt.xlabel("Year")
plt.ylabel("Disbursements (USD, millions)")
plt.title("SDG 15 disbursements over time (keyword mapping)")
plt.tight_layout()
plt.savefig(FIGDIR / "keyword_timeseries_sdg15.png", dpi=200)
plt.close()

print("[REPORT] Wrote:")
print("  - tables/keyword_totals_sdg15.tex")
print("  - figures/keyword_totals_sdg15.png")
print("  - figures/keyword_timeseries_sdg15.png")
