"""
Feature Engineering Module
Creates features for predictive modeling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def encode_categorical_features(df, columns):
    """Encode categorical features using label encoding."""
    df_encoded = df.copy()
    encoders = {}
    
    for col in columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le
    
    return df_encoded, encoders


def create_risk_features(mckinsey_df):
    """
    Create features for job displacement risk prediction.
    """
    df = mckinsey_df.copy()
    
    # Create composite risk score
    df['composite_risk'] = (
        df['automation_potential'] * 0.4 +
        df['ai_impact_score'] * 0.3 +
        df['ai_substitutability_index'] * 0.3
    )
    
    # Create skill-based features
    skill_map = {'Very High': 5, 'High': 4, 'Medium': 3, 'Low': 2, 'Very Low': 1}
    df['skill_level_numeric'] = df['skill_level'].map(skill_map)
    
    # Create education level numeric
    edu_map = {
        'Doctoral degree': 5,
        'Master\'s degree': 4,
        'Bachelor\'s degree': 3,
        'Associate degree': 2,
        'Some college': 1.5,
        'Postsecondary nondegree': 1.5,
        'High school diploma': 1,
        'No degree required': 0
    }
    df['education_numeric'] = df['education_required'].map(edu_map).fillna(1)
    
    # Create protection score (higher = more protected from automation)
    df['protection_score'] = (
        df['creativity_required'].map({'Very High': 5, 'High': 4, 'Medium': 3, 'Low': 2, 'Very Low': 1}) * 0.25 +
        df['social_interaction_required'].map({'Very High': 5, 'High': 4, 'Medium': 3, 'Low': 2, 'Very Low': 1}) * 0.25 +
        df['skill_level_numeric'] * 0.25 +
        df['education_numeric'] * 0.25
    )
    
    # Normalize protection score to 0-100
    df['protection_score'] = (df['protection_score'] - df['protection_score'].min()) / \
                              (df['protection_score'].max() - df['protection_score'].min()) * 100
    
    # Create vulnerability score
    df['vulnerability_score'] = 100 - df['protection_score']
    
    # Create employment size categories
    df['employment_category'] = pd.cut(
        df['current_employment_us_millions'],
        bins=[0, 0.01, 0.1, 0.5, 1.0, float('inf')],
        labels=['Micro', 'Small', 'Medium', 'Large', 'Very Large']
    )
    
    # Create wage categories
    df['wage_category'] = pd.cut(
        df['median_wage_usd'],
        bins=[0, 30000, 50000, 75000, 100000, float('inf')],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
    
    return df


def create_time_series_features(global_df):
    """
    Create time-series features for trend analysis.
    """
    df = global_df.copy()
    
    # Calculate growth rates
    df = df.sort_values(['country', 'year'])
    df['ai_adoption_growth'] = df.groupby('country')['ai_adoption_rate'].pct_change() * 100
    df['investment_growth'] = df.groupby('country')['investment_billions'].pct_change() * 100
    
    # Create lag features
    df['ai_adoption_lag1'] = df.groupby('country')['ai_adoption_rate'].shift(1)
    df['ai_adoption_lag2'] = df.groupby('country')['ai_adoption_rate'].shift(2)
    
    # Create moving averages
    df['ai_adoption_ma3'] = df.groupby('country')['ai_adoption_rate'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    
    return df


def prepare_ml_features(mckinsey_df):
    """
    Prepare features for machine learning models.
    """
    df = create_risk_features(mckinsey_df)
    
    # Select features for modeling
    feature_cols = [
        'automation_potential', 'ai_impact_score', 'ai_substitutability_index',
        'skill_level_numeric', 'education_numeric', 'tech_sophistication_required',
        'protection_score', 'vulnerability_score', 'median_wage_usd',
        'current_employment_us_millions'
    ]
    
    # Add sector encoding
    sector_encoded = pd.get_dummies(df['sector'], prefix='sector')
    
    # Create feature matrix
    X = pd.concat([df[feature_cols], sector_encoded], axis=1)
    
    # Create target variable (displacement risk score as numeric)
    risk_map = {'Very High': 4, 'High': 3, 'Medium': 2, 'Low': 1}
    y = df['displacement_risk_score'].map(risk_map)
    
    # Handle missing values
    X = X.fillna(X.median())
    
    return X, y, feature_cols


def create_interaction_features(df):
    """
    Create interaction features between key variables.
    """
    df = df.copy()
    
    # Interaction between automation potential and wage
    df['automation_wage_interaction'] = df['automation_potential'] * np.log(df['median_wage_usd'] + 1)
    
    # Interaction between skill and automation
    df['skill_automation_interaction'] = df['skill_level_numeric'] * df['automation_potential']
    
    # Employment size vs risk
    df['employment_risk'] = df['current_employment_us_millions'] * df['vulnerability_score']
    
    return df


def scale_features(X_train, X_test=None):
    """
    Scale features using StandardScaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    
    return X_train_scaled, scaler
