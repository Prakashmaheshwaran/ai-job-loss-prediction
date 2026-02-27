"""
Main Analysis Script
Run complete analysis pipeline for AI Job Loss Prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

from data_loader import load_all_data, get_high_risk_occupations
from feature_engineering import prepare_ml_features, create_risk_features
from models import JobDisplacementPredictor, predict_job_loss_risk, get_sector_risk_ranking
from visualizations import create_comprehensive_dashboard


def run_full_analysis():
    """
    Run the complete analysis pipeline.
    """
    print("=" * 70)
    print("AI JOB LOSS PREDICTION - COMPREHENSIVE ANALYSIS")
    print("=" * 70)
    print()
    
    # Step 1: Load Data
    print("Step 1: Loading datasets...")
    data = load_all_data()
    print(f"  ✓ WEF data: {data['wef'].shape}")
    print(f"  ✓ McKinsey data: {data['mckinsey'].shape}")
    print(f"  ✓ BLS data: {data['bls'].shape}")
    print(f"  ✓ Global AI data: {data['global_ai'].shape}")
    print()
    
    # Step 2: Create Visualizations
    print("Step 2: Generating visualizations...")
    reports_dir = Path(__file__).parent.parent / 'reports' / 'figures'
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    figures = create_comprehensive_dashboard(
        data['wef'], 
        data['mckinsey'], 
        data['bls'], 
        data['global_ai']
    )
    print(f"  ✓ Generated {len(figures)} visualizations")
    print(f"  ✓ Saved to {reports_dir}")
    print()
    
    # Step 3: High-Risk Occupations Analysis
    print("Step 3: Identifying high-risk occupations...")
    high_risk = get_high_risk_occupations()
    print(f"  ✓ Found {len(high_risk)} high-risk occupations")
    print("\n  Top 10 Highest Risk Occupations:")
    for idx, row in high_risk.head(10).iterrows():
        print(f"    {row['occupation_name']:50s} | Risk: {row['automation_potential']:.0f}%")
    print()
    
    # Step 4: Sector Risk Ranking
    print("Step 4: Analyzing sector risk...")
    sector_risk = get_sector_risk_ranking(data['mckinsey'])
    print("\n  Sector Risk Ranking (by automation potential):")
    for sector, row in sector_risk.head(10).iterrows():
        print(f"    {sector:30s} | Avg Risk: {row['automation_potential']:.1f}% | Employment: {row['current_employment_us_millions']:.2f}M")
    print()
    
    # Step 5: Machine Learning Model
    print("Step 5: Training predictive models...")
    X, y, feature_cols = prepare_ml_features(data['mckinsey'])
    
    predictor, results, _ = predict_job_loss_risk(data['mckinsey'])
    
    print(f"  ✓ Best model: {predictor.best_model_name.upper()}")
    print(f"  ✓ Cross-validation accuracy: {results[predictor.best_model_name]['cv_mean']:.2%}")
    print(f"  ✓ Test set accuracy: {results[predictor.best_model_name]['test_score']:.2%}")
    print()
    
    # Step 6: Feature Importance
    print("Step 6: Analyzing feature importance...")
    importance_df = predictor.get_feature_importance(X.columns)
    print("\n  Top 10 Most Important Features:")
    for idx, row in importance_df.head(10).iterrows():
        print(f"    {row['feature']:30s} | Importance: {row['importance']:.4f}")
    print()
    
    # Step 7: Generate Summary Statistics
    print("Step 7: Generating summary statistics...")
    
    # Global displacement by 2025
    total_displaced_2025 = data['wef'][data['wef']['year'] == 2025]['jobs_displaced_millions'].sum()
    total_net_change_2025 = data['wef'][data['wef']['year'] == 2025]['net_change_millions'].sum()
    
    print(f"  ✓ Projected jobs displaced by 2025: {total_displaced_2025:.1f} million")
    print(f"  ✓ Projected net change by 2025: {total_net_change_2025:.1f} million")
    
    # Average automation potential
    avg_automation = data['mckinsey']['automation_potential'].mean()
    print(f"  ✓ Average automation potential: {avg_automation:.1f}%")
    
    # High-risk percentage
    high_risk_pct = (data['mckinsey']['automation_potential'] >= 60).mean() * 100
    print(f"  ✓ Occupations at high risk (≥60%): {high_risk_pct:.1f}%")
    print()
    
    # Save results
    print("Step 8: Saving results...")
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    with open(models_dir / 'analysis_results.pkl', 'wb') as f:
        pickle.dump({
            'predictor': predictor,
            'results': results,
            'importance_df': importance_df,
            'high_risk': high_risk,
            'sector_risk': sector_risk
        }, f)
    
    print(f"  ✓ Results saved to {models_dir}/analysis_results.pkl")
    print()
    
    # Generate Report
    print("=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nKey Findings:")
    print(f"  • {total_displaced_2025:.1f} million jobs may be displaced by 2025")
    print(f"  • {total_net_change_2025:.1f} million net job change expected")
    print(f"  • {high_risk_pct:.1f}% of occupations are at high automation risk")
    print(f"  • Best model achieved {results[predictor.best_model_name]['cv_mean']:.1%} CV accuracy")
    print()
    print("Reports and visualizations available in:")
    print(f"  • {reports_dir}")
    print(f"  • {models_dir}")
    print()
    
    return {
        'data': data,
        'predictor': predictor,
        'results': results,
        'importance_df': importance_df,
        'high_risk': high_risk,
        'sector_risk': sector_risk
    }


def generate_executive_summary(results):
    """
    Generate an executive summary of the analysis.
    """
    print("\n" + "=" * 70)
    print("EXECUTIVE SUMMARY")
    print("=" * 70)
    
    summary = f"""
AI JOB DISPLACEMENT PREDICTION ANALYSIS

SCOPE:
This analysis examines the impact of AI and automation on global employment 
using data from WEF, McKinsey, and BLS.

KEY FINDINGS:

1. JOB DISPLACEMENT PROJECTIONS
   • By 2025: {results['high_risk'].head(20)['automation_potential'].mean():.0f}% average automation risk for top 20 occupations
   • High-risk sectors: Manufacturing, Administrative, Customer Service
   • Protected sectors: Healthcare, Education, Creative industries

2. PREDICTIVE MODEL PERFORMANCE
   • Best Model: {results['predictor'].best_model_name.upper()}
   • Accuracy: {results['results'][results['predictor'].best_model_name]['cv_mean']:.2%}
   • Top predictor: {results['importance_df'].iloc[0]['feature']}

3. SECTOR RANKING
   Highest Risk Sectors:
   {results['sector_risk'].head(5).reset_index().to_string(index=False)}

4. RECOMMENDATIONS
   • Focus reskilling programs on administrative and manufacturing workers
   • Invest in AI literacy for all workforce levels
   • Develop transition programs for high-risk occupations
   • Support creation of new AI-augmented roles

DATA SOURCES:
   • World Economic Forum (WEF) Future of Jobs Report
   • McKinsey Global Institute Automation Analysis
   • Bureau of Labor Statistics (BLS) Employment Projections
   • Global AI Adoption Survey Data

For detailed analysis, see the generated visualizations and Jupyter notebooks.
"""
    print(summary)
    
    # Save summary to file
    reports_dir = Path(__file__).parent.parent / 'reports'
    with open(reports_dir / 'executive_summary.txt', 'w') as f:
        f.write(summary)
    
    print(f"\nSummary saved to {reports_dir}/executive_summary.txt")


if __name__ == "__main__":
    # Run full analysis
    results = run_full_analysis()
    
    # Generate executive summary
    generate_executive_summary(results)
