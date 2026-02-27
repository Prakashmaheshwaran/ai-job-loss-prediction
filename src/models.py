"""
Predictive Models Module
Machine learning models for predicting job displacement risk.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class JobDisplacementPredictor:
    """
    Predict job displacement risk using ensemble of ML models.
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_importance = None
        self.is_trained = False
    
    def prepare_data(self, X, y, test_size=0.2):
        """
        Prepare train/test splits with scaling.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train(self, X, y, test_size=0.2):
        """
        Train multiple models and select best performer.
        """
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(X, y, test_size)
        
        # Initialize models
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                class_weight='balanced'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )
        }
        
        # Train models
        results = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            results[name] = {
                'model': model,
                'train_score': train_score,
                'test_score': test_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
        
        # Select best model based on CV score
        best_model_name = max(results, key=lambda x: results[x]['cv_mean'])
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        self.results = results
        self.is_trained = True
        
        # Calculate feature importance if available
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = self.best_model.feature_importances_
        
        return results
    
    def predict(self, X):
        """
        Make predictions on new data.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.best_model.predict(X_scaled)
        probabilities = self.best_model.predict_proba(X_scaled)
        
        return predictions, probabilities
    
    def get_feature_importance(self, feature_names):
        """
        Get feature importance scores.
        """
        if self.feature_importance is None:
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        X_scaled = self.scaler.transform(X_test)
        y_pred = self.best_model.predict(X_scaled)
        
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        return {
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }


class TimeSeriesForecaster:
    """
    Forecast AI adoption and job displacement trends.
    """
    
    def __init__(self):
        self.models = {}
    
    def forecast_linear_trend(self, df, column, periods=5):
        """
        Forecast future values using linear regression on time.
        """
        from sklearn.linear_model import LinearRegression
        
        X = df['year'].values.reshape(-1, 1)
        y = df[column].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Forecast future years
        last_year = df['year'].max()
        future_years = np.arange(last_year + 1, last_year + periods + 1)
        future_X = future_years.reshape(-1, 1)
        predictions = model.predict(future_X)
        
        return {
            'years': future_years.tolist(),
            'predictions': predictions.tolist(),
            'model': model,
            'r2_score': model.score(X, y)
        }
    
    def forecast_by_country(self, df, country, column='ai_adoption_rate', periods=5):
        """
        Forecast for a specific country.
        """
        country_data = df[df['country'] == country].sort_values('year')
        return self.forecast_linear_trend(country_data, column, periods)


def predict_job_loss_risk(occupation_data):
    """
    Convenience function to predict job loss risk for occupations.
    """
    try:
        from feature_engineering import prepare_ml_features
    except ImportError:
        from .feature_engineering import prepare_ml_features
    
    X, y, feature_cols = prepare_ml_features(occupation_data)
    
    predictor = JobDisplacementPredictor()
    results = predictor.train(X, y)
    
    return predictor, results, feature_cols


def get_sector_risk_ranking(mckinsey_df):
    """
    Rank sectors by average displacement risk.
    """
    sector_risk = mckinsey_df.groupby('sector').agg({
        'automation_potential': 'mean',
        'ai_impact_score': 'mean',
        'current_employment_us_millions': 'sum'
    }).sort_values('automation_potential', ascending=False)
    
    return sector_risk
