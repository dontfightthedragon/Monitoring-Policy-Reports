#!/usr/bin/env python3
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

try:
    import seaborn as sns
    _HAVE_SNS = True
except Exception:
    sns = None
    _HAVE_SNS = False

# Style setup with graceful fallback
try:
    plt.style.use('seaborn-v0_8')
except Exception:
    plt.style.use('ggplot')
if _HAVE_SNS:
    sns.set_palette("husl")

# Ensure white background for figures/axes regardless of style
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'


class BiodiversityAnalysis: 
    def __init__(self, data_path, output_dir="reports/figures"):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Biodiversity subset (Rio 1/2) for reference if needed
        self.df = pd.read_parquet(self.data_path)

        # Load pre-aggregated panels
        base = Path(__file__).resolve().parent
        self.panel_year   = pd.read_csv(base / "panel_year.csv").sort_values("year")
        self.recip_by_usd = pd.read_csv(base / "panel_recipient_by_usd.csv")
        self.recip_by_cnt = pd.read_csv(base / "panel_recipient_by_count.csv")
        self.donor_by_usd = pd.read_csv(base / "panel_donor_by_usd.csv")
        self.donor_by_cnt = pd.read_csv(base / "panel_donor_by_count.csv")

        # Ensure a Millions column exists for consistent plotting
        def to_millions(df: pd.DataFrame, in_col: str = "usd_disb", out_col: str = "usd_disb_m") -> pd.DataFrame:
            if out_col in df.columns:
                return df
            s = pd.to_numeric(df[in_col], errors="coerce").astype(float)
            # Heuristic: if max < 0.1 it's likely already divided by 1e6 → scale up
            if np.nanmax(s) < 0.1:
                df[out_col] = s * 1_000_000.0
            # if max > 1e5 it's raw USD → scale down
            elif np.nanmax(s) > 1e5:
                df[out_col] = s / 1_000_000.0
            else:
                df[out_col] = s
            return df

        self.panel_year   = to_millions(self.panel_year)
        self.recip_by_usd = to_millions(self.recip_by_usd)
        self.donor_by_usd = to_millions(self.donor_by_usd)

        print(
            f"Loaded {len(self.df):,} biodiversity rows; panels: "
            f"{len(self.panel_year)} years, "
            f"{self.recip_by_usd.shape[0]} recipients, "
            f"{self.donor_by_usd.shape[0]} donors"
        )

    def temporal_analysis(self):
        print("\n=== Temporal Analysis ===")
        y = self.panel_year.copy()
        y["avg_project_size_m"] = y["usd_disb_m"] / y["proj_n"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        primary_color = "#1f77b4"
        axes[0].plot(y['year'], y['usd_disb_m'], marker='o', linewidth=2, color=primary_color)
        axes[0].set_title('Biodiversity Disbursements by Year')
        axes[0].set_ylabel('USD (Millions)')
        axes[0].grid(True, alpha=0.3)

        axes[1].bar(y['year'], y['proj_n'], color=primary_color)
        axes[1].set_title('Projects by Year')
        axes[1].set_ylabel('Number of Projects')
        axes[1].tick_params(axis='x', rotation=45)

        axes[2].plot(y['year'], y['avg_project_size_m'], marker='s', linewidth=2, color=primary_color)
        axes[2].set_title('Average Project Size')
        axes[2].set_ylabel('USD per Project (Millions)')
        axes[2].grid(True, alpha=0.3)

        for ax in axes:
            for spine in ax.spines.values():
                spine.set_color("black")
                spine.set_linewidth(0.8)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'temporal_analysis.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        y.to_csv(self.output_dir / 'yearly_summary.csv', index=False)
        print(f"Temporal analysis saved to {self.output_dir}")
        return y

    def geographic_analysis(self):
        print("\n=== Geographic Analysis (Recipients) ===")
        top_usd = self.recip_by_usd.nlargest(15, 'usd_disb_m')
        top_cnt = self.recip_by_cnt.nlargest(15, 'proj_n')

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        primary_color = "#1f77b4"
        axes[0].barh(top_usd['recipient_name'][::-1], top_usd['usd_disb_m'][::-1], color=primary_color)
        axes[0].set_title('Top 15 Recipients by Disbursements')
        axes[0].set_xlabel('USD (Millions)')

        axes[1].barh(top_cnt['recipient_name'][::-1], top_cnt['proj_n'][::-1], color=primary_color)
        axes[1].set_title('Top 15 Recipients by Project Count')
        axes[1].set_xlabel('Number of Projects')

        for ax in axes:
            for spine in ax.spines.values():
                spine.set_color("black")
                spine.set_linewidth(0.8)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'recipients_top.pdf', dpi=300, bbox_inches='tight')
        plt.close()

        # Lorenz curve on recipient totals (Millions)
        vals = self.recip_by_usd['usd_disb_m'].sort_values(ascending=False).to_numpy()
        cum_f = np.cumsum(vals) / vals.sum()
        cum_n = np.arange(1, len(vals)+1) / len(vals)

        fig_lor, ax_lor = plt.subplots(figsize=(6, 5))
        primary_color = "#1f77b4"
        ax_lor.plot(cum_n, cum_f, linewidth=2, color=primary_color)
        ax_lor.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax_lor.set_title('Funding Concentration (Recipients)')
        ax_lor.set_xlabel('Cumulative Share of Recipients')
        ax_lor.set_ylabel('Cumulative Share of Funding')
        ax_lor.grid(True, alpha=0.3)
        for spine in ax_lor.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.8)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'recipients_lorenz.pdf', dpi=300, bbox_inches='tight')
        plt.close(fig_lor)

        top_usd.to_csv(self.output_dir / 'top_recipients_by_usd.csv', index=False)
        top_cnt.to_csv(self.output_dir / 'top_recipients_by_count.csv', index=False)
        print(f"Recipient analysis saved to {self.output_dir}")
        return {"top_usd": top_usd, "top_cnt": top_cnt}

    def donors_analysis(self):
        print("\n=== Donor Analysis ===")
        top_usd = self.donor_by_usd.nlargest(15, 'usd_disb_m')
        top_cnt = self.donor_by_cnt.nlargest(15, 'proj_n')

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        primary_color = "#1f77b4"
        axes[0].barh(top_usd['donor_name'][::-1], top_usd['usd_disb_m'][::-1], color=primary_color)
        axes[0].set_title('Top 15 Donors by Disbursements')
        axes[0].set_xlabel('USD (Millions)')

        axes[1].barh(top_cnt['donor_name'][::-1], top_cnt['proj_n'][::-1], color=primary_color)
        axes[1].set_title('Top 15 Donors by Project Count')
        axes[1].set_xlabel('Number of Projects')

        for ax in axes:
            for spine in ax.spines.values():
                spine.set_color("black")
                spine.set_linewidth(0.8)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'donors_top.pdf', dpi=300, bbox_inches='tight')
        plt.close()

        top_usd.to_csv(self.output_dir / 'top_donors_by_usd.csv', index=False)
        top_cnt.to_csv(self.output_dir / 'top_donors_by_count.csv', index=False)
        print(f"Donor analysis saved to {self.output_dir}")
        return {"top_usd": top_usd, "top_cnt": top_cnt}

    def generate_summary_report(self):
        print("\n=== Generating Summary Report ===")
        y = self.panel_year.sort_values('year')
        report = f"""
# Biodiversity Finance Analysis Summary Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Overview
- Range: {int(y['year'].min())}-{int(y['year'].max())}
- Total Biodiversity Projects: {int(y['proj_n'].sum()):,}

## Financial Overview (Millions USD)
- Total Disbursements (sum): ${y['usd_disb_m'].sum():,.1f} M
- Average Project Size (overall mean): ${(y['usd_disb_m'].sum()/y['proj_n'].sum()):.3f} M

## Temporal Trends
- Funding Growth: {((y['usd_disb_m'].iloc[-1]/y['usd_disb_m'].iloc[0]) - 1) * 100:.1f}% over period
- Project Count Growth: {((y['proj_n'].iloc[-1]/y['proj_n'].iloc[0]) - 1) * 100:.1f}% over period

(Amounts are constant 2023 USD; scope: Rio biodiversity marker ∈ {1,2}; negatives not present after aggregation.)
"""
        with open(self.output_dir / 'analysis_summary_report.md', 'w') as f:
            f.write(report)
        print(f"Summary report saved to {self.output_dir}/analysis_summary_report.md")

    def run_full_analysis(self):
        print("Starting comprehensive biodiversity finance analysis...")
        self.temporal_analysis()
        self.geographic_analysis()
        self.donors_analysis()
        self.generate_summary_report()
        # Debug magnitudes (Millions)
        print("DEBUG max yearly disb (M):", float(self.panel_year['usd_disb_m'].max()))
        print("DEBUG max recipient disb (M):", float(self.recip_by_usd['usd_disb_m'].max()))
        print("DEBUG max donor disb (M):", float(self.donor_by_usd['usd_disb_m'].max()))
        print(f"\n=== Analysis Complete ===\nAll outputs saved to: {self.output_dir}")


def main():
    analysis = BiodiversityAnalysis(
        data_path="analysis/analyticalOverview/CRS_biodiv_rio12.parquet",
        output_dir="analysis/analyticalOverview/figures"
    )
    analysis.run_full_analysis()


if __name__ == "__main__":
    main()
