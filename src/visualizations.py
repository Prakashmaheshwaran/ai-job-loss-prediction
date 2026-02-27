"""
Visualizations Module
Creates charts and dashboards for AI job displacement analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_job_displacement_trends(wef_df, save_path=None):
    """
    Plot job displacement trends over time by region.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Global trends
    global_trends = wef_df.groupby('year').agg({
        'jobs_displaced_millions': 'sum',
        'net_change_millions': 'sum',
        'ai_adoption_rate': 'mean'
    }).reset_index()
    
    # Plot 1: Jobs Displaced
    ax1 = axes[0, 0]
    for region in wef_df['region'].unique():
        region_data = wef_df[wef_df['region'] == region].groupby('year')['jobs_displaced_millions'].sum()
        ax1.plot(region_data.index, region_data.values, marker='o', label=region, linewidth=2)
    ax1.set_title('Jobs Displaced by AI Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Jobs Displaced (Millions)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Net Job Change
    ax2 = axes[0, 1]
    for region in wef_df['region'].unique():
        region_data = wef_df[wef_df['region'] == region].groupby('year')['net_change_millions'].sum()
        ax2.plot(region_data.index, region_data.values, marker='s', label=region, linewidth=2)
    ax2.set_title('Net Job Change by AI Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Net Change (Millions)')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: AI Adoption Rate
    ax3 = axes[1, 0]
    for region in wef_df['region'].unique():
        region_data = wef_df[wef_df['region'] == region].groupby('year')['ai_adoption_rate'].mean()
        ax3.plot(region_data.index, region_data.values, marker='^', label=region, linewidth=2)
    ax3.set_title('AI Adoption Rate by Region', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('AI Adoption Rate (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Industry breakdown
    ax4 = axes[1, 1]
    industry_data = wef_df[wef_df['year'] == 2025].groupby('industry')['jobs_displaced_millions'].sum().sort_values(ascending=True)
    industry_data.plot(kind='barh', ax=ax4, color='steelblue')
    ax4.set_title('Projected Job Displacement by Industry (2025)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Jobs Displaced (Millions)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_automation_risk_distribution(mckinsey_df, save_path=None):
    """
    Plot automation risk distribution across occupations.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Automation potential histogram
    ax1 = axes[0, 0]
    mckinsey_df['automation_potential'].hist(bins=20, ax=ax1, color='coral', edgecolor='black', alpha=0.7)
    ax1.set_title('Distribution of Automation Potential', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Automation Potential (%)')
    ax1.set_ylabel('Number of Occupations')
    ax1.axvline(mckinsey_df['automation_potential'].mean(), color='red', linestyle='--', 
                label=f'Mean: {mckinsey_df["automation_potential"].mean():.1f}%')
    ax1.legend()
    
    # Plot 2: Risk by sector
    ax2 = axes[0, 1]
    sector_risk = mckinsey_df.groupby('sector')['automation_potential'].mean().sort_values(ascending=True)
    sector_risk.plot(kind='barh', ax=ax2, color='teal')
    ax2.set_title('Average Automation Potential by Sector', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Average Automation Potential (%)')
    
    # Plot 3: Wage vs Automation
    ax3 = axes[1, 0]
    scatter = ax3.scatter(mckinsey_df['median_wage_usd'], mckinsey_df['automation_potential'],
                         c=mckinsey_df['ai_impact_score'], cmap='viridis', alpha=0.6, s=50)
    ax3.set_title('Wage vs Automation Potential', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Median Annual Wage (USD)')
    ax3.set_ylabel('Automation Potential (%)')
    plt.colorbar(scatter, ax=ax3, label='AI Impact Score')
    
    # Plot 4: Employment size vs Automation
    ax4 = axes[1, 1]
    bubble_sizes = mckinsey_df['current_employment_us_millions'] * 500
    ax4.scatter(mckinsey_df['automation_potential'], mckinsey_df['ai_impact_score'],
               s=bubble_sizes, alpha=0.5, c='purple')
    ax4.set_title('Automation vs AI Impact\n(Bubble size = Employment)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Automation Potential (%)')
    ax4.set_ylabel('AI Impact Score')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_global_adoption_heatmap(global_df, save_path=None):
    """
    Create heatmap of AI adoption across countries and years.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Heatmap 1: AI Adoption Rate
    ax1 = axes[0]
    pivot_adoption = global_df.pivot(index='country', columns='year', values='ai_adoption_rate')
    sns.heatmap(pivot_adoption, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax1, cbar_kws={'label': 'AI Adoption Rate (%)'})
    ax1.set_title('AI Adoption Rate by Country and Year', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Country')
    
    # Heatmap 2: Net Job Impact
    ax2 = axes[1]
    pivot_impact = global_df.pivot(index='country', columns='year', values='net_job_impact')
    sns.heatmap(pivot_impact, annot=True, fmt='.0f', cmap='RdYlGn', ax=ax2, cbar_kws={'label': 'Net Job Impact'})
    ax2.set_title('Net Job Impact by Country and Year', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Country')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_employment_projections(bls_df, save_path=None):
    """
    Plot BLS employment projections.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Employment change by risk category
    ax1 = axes[0, 0]
    risk_change = bls_df.groupby('automation_risk')['percent_change'].mean()
    risk_change.plot(kind='bar', ax=ax1, color=['green', 'orange', 'red'])
    ax1.set_title('Projected Employment Change by Automation Risk', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Automation Risk')
    ax1.set_ylabel('Percent Change (%)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Wage distribution by risk
    ax2 = axes[0, 1]
    bls_df.boxplot(column='median_wage_annual', by='automation_risk', ax=ax2)
    ax2.set_title('Wage Distribution by Automation Risk', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Automation Risk')
    ax2.set_ylabel('Median Annual Wage (USD)')
    
    # Plot 3: Top growing occupations
    ax3 = axes[1, 0]
    top_growing = bls_df.nlargest(15, 'percent_change')[['occupation_title', 'percent_change']]
    top_growing.set_index('occupation_title').plot(kind='barh', ax=ax3, color='forestgreen', legend=False)
    ax3.set_title('Top 15 Fastest Growing Occupations', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Projected Growth (%)')
    
    # Plot 4: Top declining occupations
    ax4 = axes[1, 1]
    top_declining = bls_df.nsmallest(15, 'percent_change')[['occupation_title', 'percent_change']]
    top_declining.set_index('occupation_title').plot(kind='barh', ax=ax4, color='darkred', legend=False)
    ax4.set_title('Top 15 Declining Occupations', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Projected Change (%)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_feature_importance(importance_df, save_path=None):
    """
    Plot feature importance from ML model.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    importance_df.head(20).plot(x='feature', y='importance', kind='barh', 
                                ax=ax, color='steelblue', legend=False)
    ax.set_title('Top 20 Feature Importances', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_sector_comparison(mckinsey_df, save_path=None):
    """
    Create comprehensive sector comparison visualization.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    sector_stats = mckinsey_df.groupby('sector').agg({
        'automation_potential': 'mean',
        'ai_impact_score': 'mean',
        'current_employment_us_millions': 'sum',
        'median_wage_usd': 'mean'
    }).sort_values('automation_potential', ascending=False)
    
    # Plot 1: Automation potential by sector
    ax1 = axes[0, 0]
    sector_stats['automation_potential'].plot(kind='bar', ax=ax1, color='coral')
    ax1.set_title('Average Automation Potential by Sector', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Automation Potential (%)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Employment by sector
    ax2 = axes[0, 1]
    sector_stats['current_employment_us_millions'].plot(kind='bar', ax=ax2, color='teal')
    ax2.set_title('Total Employment by Sector', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Employment (Millions)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Wage vs Automation by sector
    ax3 = axes[1, 0]
    ax3.scatter(sector_stats['automation_potential'], sector_stats['median_wage_usd'],
               s=sector_stats['current_employment_us_millions']*200,
               alpha=0.6, c=range(len(sector_stats)), cmap='viridis')
    for i, sector in enumerate(sector_stats.index):
        ax3.annotate(sector, (sector_stats['automation_potential'].iloc[i], 
                             sector_stats['median_wage_usd'].iloc[i]),
                    fontsize=8, rotation=45)
    ax3.set_title('Wage vs Automation by Sector\n(Bubble size = Employment)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Automation Potential (%)')
    ax3.set_ylabel('Average Median Wage (USD)')
    
    # Plot 4: Risk categories
    ax4 = axes[1, 1]
    risk_counts = mckinsey_df['displacement_risk_score'].value_counts()
    colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red', 'Very High': 'darkred'}
    risk_counts.plot(kind='pie', ax=ax4, autopct='%1.1f%%', colors=[colors.get(x, 'gray') for x in risk_counts.index])
    ax4.set_title('Distribution of Displacement Risk', fontsize=14, fontweight='bold')
    ax4.set_ylabel('')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_comprehensive_dashboard(wef_df, mckinsey_df, bls_df, global_df, save_path=None):
    """
    Create a comprehensive dashboard with all visualizations.
    """
    # Create output directory
    reports_dir = Path(__file__).parent.parent / 'reports' / 'figures'
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all plots
    figures = []
    
    # Figure 1: WEF Trends
    fig1 = plot_job_displacement_trends(wef_df, save_path=reports_dir / 'wef_trends.png')
    figures.append(fig1)
    
    # Figure 2: Automation Risk
    fig2 = plot_automation_risk_distribution(mckinsey_df, save_path=reports_dir / 'automation_risk.png')
    figures.append(fig2)
    
    # Figure 3: Global Heatmap
    fig3 = plot_global_adoption_heatmap(global_df, save_path=reports_dir / 'global_heatmap.png')
    figures.append(fig3)
    
    # Figure 4: BLS Projections
    fig4 = plot_employment_projections(bls_df, save_path=reports_dir / 'bls_projections.png')
    figures.append(fig4)
    
    # Figure 5: Sector Comparison
    fig5 = plot_sector_comparison(mckinsey_df, save_path=reports_dir / 'sector_comparison.png')
    figures.append(fig5)
    
    return figures


def save_all_figures(figures, output_dir):
    """
    Save all figures to specified directory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i, fig in enumerate(figures):
        fig.savefig(output_path / f'figure_{i+1}.png', dpi=300, bbox_inches='tight')
    
    print(f"Saved {len(figures)} figures to {output_dir}")
