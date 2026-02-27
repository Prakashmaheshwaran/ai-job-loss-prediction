"""
Data Loader Module
Loads and preprocesses datasets from WEF, McKinsey, BLS, and other sources.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_wef_data():
    """Load World Economic Forum job displacement data."""
    filepath = Path(__file__).parent.parent / 'data' / 'wef_job_displacement.csv'
    df = pd.read_csv(filepath)
    df['year'] = pd.to_numeric(df['year'])
    df['displacement_risk'] = df['displacement_risk'].astype('category')
    df['skill_level'] = df['skill_level'].astype('category')
    df['region'] = df['region'].astype('category')
    return df


def load_mckinsey_data():
    """Load McKinsey automation index data."""
    filepath = Path(__file__).parent.parent / 'data' / 'mckinsey_automation_index.csv'
    df = pd.read_csv(filepath)
    df['displacement_risk_score'] = df['displacement_risk_score'].astype('category')
    return df


def load_bls_data():
    """Load Bureau of Labor Statistics employment projections."""
    filepath = Path(__file__).parent.parent / 'data' / 'bls_employment_projections.csv'
    df = pd.read_csv(filepath)
    df['automation_risk'] = df['automation_risk'].astype('category')
    return df


def load_global_ai_adoption():
    """Load global AI adoption data."""
    filepath = Path(__file__).parent.parent / 'data' / 'global_ai_adoption.csv'
    df = pd.read_csv(filepath)
    df['year'] = pd.to_numeric(df['year'])
    df['region'] = df['region'].astype('category')
    return df


def load_all_data():
    """Load all datasets and return as dictionary."""
    data = {
        'wef': load_wef_data(),
        'mckinsey': load_mckinsey_data(),
        'bls': load_bls_data(),
        'global_ai': load_global_ai_adoption()
    }
    return data


def merge_for_analysis():
    """
    Merge datasets for comprehensive analysis.
    Returns merged dataframe with key features.
    """
    # Load data
    wef = load_wef_data()
    mckinsey = load_mckinsey_data()
    bls = load_bls_data()
    global_ai = load_global_ai_adoption()
    
    # Aggregate WEF data by industry and year
    wef_agg = wef.groupby(['year', 'industry', 'region']).agg({
        'jobs_displaced_millions': 'sum',
        'net_change_millions': 'sum',
        'ai_adoption_rate': 'mean',
        'automation_potential': 'mean'
    }).reset_index()
    
    # Aggregate global AI data by year
    global_agg = global_ai.groupby('year').agg({
        'ai_adoption_rate': 'mean',
        'automation_readiness': 'mean',
        'job_displacement_risk': 'mean',
        'job_creation_potential': 'mean',
        'net_job_impact': 'mean'
    }).reset_index()
    
    # Prepare McKinsey data - use automation potential as key metric
    mckinsey_clean = mckinsey[['occupation_name', 'sector', 'automation_potential', 
                                'ai_impact_score', 'displacement_risk_score', 
                                'current_employment_us_millions']].copy()
    
    # Prepare BLS data
    bls_clean = bls[['occupation_code', 'occupation_title', 'employment_2023_millions',
                     'projected_2033_millions', 'percent_change', 'median_wage_annual',
                     'automation_risk']].copy()
    
    return {
        'wef_agg': wef_agg,
        'global_agg': global_agg,
        'mckinsey_clean': mckinsey_clean,
        'bls_clean': bls_clean
    }


def get_high_risk_occupations():
    """Return occupations with highest automation risk."""
    mckinsey = load_mckinsey_data()
    high_risk = mckinsey[mckinsey['automation_potential'] >= 60]
    return high_risk.sort_values('automation_potential', ascending=False)


def get_job_trends_by_year():
    """Get job displacement trends over time."""
    wef = load_wef_data()
    trends = wef.groupby('year').agg({
        'jobs_displaced_millions': 'sum',
        'net_change_millions': 'sum',
        'ai_adoption_rate': 'mean'
    }).reset_index()
    return trends
