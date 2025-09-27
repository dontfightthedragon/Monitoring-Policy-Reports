#!/usr/bin/env python3
"""
Comprehensive Biodiversity Finance Analysis
Builds on existing data pipeline to provide thesis-ready analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BiodiversityAnalysis:
    def __init__(self, data_path, output_dir="reports/figures"):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.df = pd.read_parquet(self.data_path)
        self.panel = pd.read_csv("reports/tables/panel_recipient_year.csv")
        
        print(f"Loaded {len(self.df):,} projects and {len(self.panel):,} panel observations")
    
    def temporal_analysis(self):
        """Analyze biodiversity funding trends over time"""
        print("\n=== Temporal Analysis ===")
        
        # Yearly totals
        yearly = self.panel.groupby('year').agg({
            'usd_disb': 'sum',
            'proj_n': 'sum',
            'bio_wtd': 'sum'
        }).reset_index()
        
        # Create time series plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total disbursements
        axes[0,0].plot(yearly['year'], yearly['usd_disb'], marker='o', linewidth=2)
        axes[0,0].set_title('Total Biodiversity Disbursements by Year')
        axes[0,0].set_ylabel('USD (Millions)')
        axes[0,0].grid(True, alpha=0.3)
        
        # Project counts
        axes[0,1].bar(yearly['year'], yearly['proj_n'], alpha=0.7)
        axes[0,1].set_title('Biodiversity Projects by Year')
        axes[0,1].set_ylabel('Number of Projects')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Average project size
        yearly['avg_size'] = yearly['usd_disb'] / yearly['proj_n']
        axes[1,0].plot(yearly['year'], yearly['avg_size'], marker='s', color='green', linewidth=2)
        axes[1,0].set_title('Average Project Size Over Time')
        axes[1,0].set_ylabel('USD per Project (Millions)')
        axes[1,0].grid(True, alpha=0.3)
        
        # Biodiversity intensity
        yearly['bio_intensity'] = yearly['bio_wtd'] / yearly['proj_n']
        axes[1,1].plot(yearly['year'], yearly['bio_intensity'], marker='^', color='red', linewidth=2)
        axes[1,1].set_title('Average Biodiversity Intensity Over Time')
        axes[1,1].set_ylabel('Bio Weight per Project')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'temporal_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save summary statistics
        yearly.to_csv(self.output_dir / 'yearly_summary.csv', index=False)
        print(f"Temporal analysis saved to {self.output_dir}")
        
        return yearly
    
    def geographic_analysis(self):
        """Analyze geographic distribution of biodiversity funding"""
        print("\n=== Geographic Analysis ===")
        
        # Top recipients by total funding
        top_recipients = self.panel.groupby('recipient_name').agg({
            'usd_disb': 'sum',
            'proj_n': 'sum'
        }).sort_values('usd_disb', ascending=False).head(15)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Top recipients by funding
        top_recipients['usd_disb'].plot(kind='barh', ax=axes[0,0])
        axes[0,0].set_title('Top 15 Recipients by Total Funding')
        axes[0,0].set_xlabel('Total Disbursements (USD Millions)')
        
        # Top recipients by project count
        top_recipients['proj_n'].plot(kind='barh', ax=axes[0,1])
        axes[0,1].set_title('Top 15 Recipients by Project Count')
        axes[0,1].set_xlabel('Number of Projects')
        
        # Funding concentration (Lorenz curve)
        sorted_funding = np.sort(self.panel['usd_disb'])[::-1]
        cumulative_funding = np.cumsum(sorted_funding) / np.sum(sorted_funding)
        cumulative_recipients = np.arange(1, len(sorted_funding) + 1) / len(sorted_funding)
        
        axes[1,0].plot(cumulative_recipients, cumulative_funding, linewidth=2)
        axes[1,0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[1,0].set_title('Funding Concentration (Lorenz Curve)')
        axes[1,0].set_xlabel('Cumulative Share of Recipients')
        axes[1,0].set_ylabel('Cumulative Share of Funding')
        axes[1,0].grid(True, alpha=0.3)
        
        # Regional analysis (if available)
        if 'recipient_name' in self.panel.columns:
            # Simple regional grouping based on common patterns
            regional_keywords = {
                'Africa': ['Africa', 'Sub-Saharan', 'Sahara', 'West Africa', 'East Africa'],
                'Asia': ['Asia', 'South Asia', 'Southeast Asia', 'India', 'China', 'Indonesia'],
                'Latin America': ['Latin America', 'Caribbean', 'Central America', 'South America'],
                'Europe': ['Europe', 'Eastern Europe', 'Balkans'],
                'Middle East': ['Middle East', 'Arab', 'Gulf']
            }
            
            # This would need more sophisticated regional mapping
            axes[1,1].text(0.5, 0.5, 'Regional Analysis\n(Requires detailed mapping)', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Regional Distribution')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'geographic_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save top recipients data
        top_recipients.to_csv(self.output_dir / 'top_recipients_analysis.csv')
        print(f"Geographic analysis saved to {self.output_dir}")
        
        return top_recipients
    
    def environmental_markers_analysis(self):
        """Compare different environmental markers"""
        print("\n=== Environmental Markers Analysis ===")
        
        # Load marker data
        markers = ['biodiversity', 'climate_mitigation', 'climate_adaptation', 'desertification']
        marker_data = {}
        
        for marker in markers:
            try:
                marker_df = pd.read_csv(f"reports/tables/marker_{marker}.csv")
                marker_data[marker] = marker_df
            except FileNotFoundError:
                print(f"Warning: {marker} marker data not found")
        
        if not marker_data:
            print("No marker data available for comparison")
            return None
        
        # Create comparison visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for i, (marker, data) in enumerate(marker_data.items()):
            if i >= 4:  # Only plot first 4 markers
                break
            
            row, col = i // 2, i % 2
            
            # Extract counts (assuming standard format)
            if 'count' in data.columns:
                counts = data.set_index(data.columns[0])['count']
            elif 'n' in data.columns:
                counts = data.set_index(data.columns[0])['n']
            else:
                continue
            
            # Remove NaN entries
            counts = counts.dropna()
            
            # Create pie chart for marker distribution
            axes[row, col].pie(counts.values, labels=counts.index, autopct='%1.1f%%')
            axes[row, col].set_title(f'{marker.replace("_", " ").title()} Marker Distribution')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'environmental_markers_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Environmental markers analysis saved to {self.output_dir}")
        return marker_data
    
    def correlation_analysis(self):
        """Analyze correlations between different variables"""
        print("\n=== Correlation Analysis ===")
        
        # Select numeric columns for correlation
        numeric_cols = ['usd_disb', 'proj_n', 'bio_bin', 'bio_wtd', 'avg_project_size', 'bio_intensity']
        available_cols = [col for col in numeric_cols if col in self.panel.columns]
        
        if len(available_cols) < 2:
            print("Not enough numeric columns for correlation analysis")
            return None
        
        # Calculate correlation matrix
        corr_matrix = self.panel[available_cols].corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f')
        plt.title('Correlation Matrix of Biodiversity Finance Variables')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save correlation matrix
        corr_matrix.to_csv(self.output_dir / 'correlation_matrix.csv')
        print(f"Correlation analysis saved to {self.output_dir}")
        
        return corr_matrix
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n=== Generating Summary Report ===")
        
        report = f"""
# Biodiversity Finance Analysis Summary Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Overview
- Total Projects: {len(self.df):,}
- Panel Observations: {len(self.panel):,}
- Unique Recipients: {self.panel['recipient_name'].nunique():,}
- Year Range: {self.panel['year'].min()}-{self.panel['year'].max()}

## Key Findings

### Financial Overview
- Total Biodiversity Disbursements: ${self.panel['usd_disb'].sum():,.0f}M
- Average Project Size: ${self.panel['avg_project_size'].mean():.2f}M
- Total Biodiversity Projects: {self.panel['proj_n'].sum():,}

### Geographic Distribution
- Top Recipient: {self.panel.groupby('recipient_name')['usd_disb'].sum().idxmax()}
- Top Recipient Funding: ${self.panel.groupby('recipient_name')['usd_disb'].sum().max():,.0f}M

### Temporal Trends
- Funding Growth: {((self.panel.groupby('year')['usd_disb'].sum().iloc[-1] / self.panel.groupby('year')['usd_disb'].sum().iloc[0]) - 1) * 100:.1f}% over period
- Project Count Growth: {((self.panel.groupby('year')['proj_n'].sum().iloc[-1] / self.panel.groupby('year')['proj_n'].sum().iloc[0]) - 1) * 100:.1f}% over period

## Files Generated
- Temporal Analysis: temporal_analysis.png
- Geographic Analysis: geographic_analysis.png
- Environmental Markers: environmental_markers_comparison.png
- Correlation Analysis: correlation_heatmap.png
- Data Tables: Various CSV files in reports/tables/

## Next Steps for Research
1. Statistical modeling (regression analysis)
2. Impact evaluation with external data
3. Text analysis of project descriptions
4. Comparative analysis with other environmental markers
5. Policy implications analysis
"""
        
        with open(self.output_dir / 'analysis_summary_report.md', 'w') as f:
            f.write(report)
        
        print(f"Summary report saved to {self.output_dir}/analysis_summary_report.md")
    
    def run_full_analysis(self):
        """Run all analysis components"""
        print("Starting comprehensive biodiversity finance analysis...")
        
        # Run all analysis components
        self.temporal_analysis()
        self.geographic_analysis()
        self.environmental_markers_analysis()
        self.correlation_analysis()
        self.generate_summary_report()
        
        print(f"\n=== Analysis Complete ===")
        print(f"All outputs saved to: {self.output_dir}")
        print(f"Check the summary report for key findings!")

def main():
    # Path to your analysis-ready dataset
    data_path = "dataset/temporaryFileToWorkWith/CRS_2015_2023_analysis_ready.parquet"
    
    # Create analysis instance
    analysis = BiodiversityAnalysis(data_path)
    
    # Run full analysis
    analysis.run_full_analysis()

if __name__ == "__main__":
    main()

