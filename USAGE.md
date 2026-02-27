# Usage Guide

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Prakashmaheshwaran/ai-job-loss-prediction.git
cd ai-job-loss-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Analysis
```bash
python src/main_analysis.py
```

This will:
- Load all datasets
- Generate visualizations
- Train predictive models
- Create summary reports

### 4. Explore Jupyter Notebooks
```bash
jupyter notebook notebooks/
```

Available notebooks:
- `01_data_exploration.ipynb` - Explore datasets and trends
- `02_feature_engineering.ipynb` - Feature creation and analysis
- `03_predictive_modeling.ipynb` - ML model training and evaluation

## Using the Python API

### Load Data
```python
from src.data_loader import load_all_data

data = load_all_data()
wef = data['wef']
mckinsey = data['mckinsey']
bls = data['bls']
global_ai = data['global_ai']
```

### Get High-Risk Occupations
```python
from src.data_loader import get_high_risk_occupations

high_risk = get_high_risk_occupations()
print(high_risk[['occupation_name', 'automation_potential']])
```

### Train Predictive Model
```python
from src.models import predict_job_loss_risk
from src.data_loader import load_mckinsey_data

mckinsey = load_mckinsey_data()
predictor, results, feature_cols = predict_job_loss_risk(mckinsey)

# Make predictions
X, y, _ = prepare_ml_features(mckinsey)
predictions, probabilities = predictor.predict(X)
```

### Create Visualizations
```python
from src.visualizations import (
    plot_job_displacement_trends,
    plot_automation_risk_distribution,
    create_comprehensive_dashboard
)
from src.data_loader import load_wef_data, load_mckinsey_data

wef = load_wef_data()
mckinsey = load_mckinsey_data()

# Create specific plots
plot_job_displacement_trends(wef, save_path='reports/trends.png')
plot_automation_risk_distribution(mckinsey, save_path='reports/risk.png')

# Create full dashboard
figures = create_comprehensive_dashboard(wef, mckinsey, bls, global_ai)
```

## Output Files

After running the analysis, you'll find:

- `reports/figures/` - All visualizations
- `models/job_displacement_predictor.pkl` - Trained ML model
- `data/processed/predictions.csv` - Job risk predictions
- `reports/executive_summary.txt` - Text summary

## Customization

### Add New Data
1. Save your CSV file to `data/`
2. Add a loader function in `src/data_loader.py`
3. Update analysis scripts as needed

### Modify Models
Edit `src/models.py` to:
- Add new algorithms
- Change hyperparameters
- Adjust train/test splits

### Create New Visualizations
Add functions to `src/visualizations.py` and call them from analysis scripts.

## Troubleshooting

### Import Errors
Ensure you're running from the project root:
```bash
cd ai-job-loss-prediction
python -c "from src.data_loader import load_all_data; print('OK')"
```

### Missing Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Jupyter Not Found
```bash
pip install jupyter
jupyter notebook
```

## Citation

If you use this project in your research, please cite:
```
AI Job Loss Prediction Project
https://github.com/Prakashmaheshwaran/ai-job-loss-prediction
```
