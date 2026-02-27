# AI Job Loss Prediction Project

## Overview

This project analyzes AI adoption and its impact on job displacement using data from multiple authoritative sources including:
- **World Economic Forum (WEF)** - Future of Jobs Report 2025
- **McKinsey & Company** - Future of Work Research
- **Bureau of Labor Statistics (BLS)** - Employment Projections
- **Kaggle** - Global AI Impact Datasets

## Project Structure

```
ai-job-loss-prediction/
├── data/                    # Raw and processed datasets
│   ├── wef_job_displacement.csv
│   ├── mckinsey_automation_index.csv
│   ├── bls_employment_projections.csv
│   └── global_ai_adoption.csv
├── notebooks/              # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_predictive_modeling.ipynb
├── src/                  # Source code
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── models.py
│   └── visualizations.py
├── reports/              # Generated reports and visualizations
│   ├── figures/
│   └── final_report.pdf
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Key Features

- **Data Collection**: Aggregated data from WEF, McKinsey, BLS, and other sources
- **Exploratory Analysis**: Comprehensive analysis of AI adoption trends and job displacement patterns
- **Predictive Modeling**: Machine learning models to predict job displacement risk
- **Visualizations**: Interactive charts and dashboards
- **Insights**: Actionable recommendations for workforce planning

## Key Findings

Based on the analysis:
- **85 million jobs** may be displaced by AI by 2025 (WEF)
- **97 million new roles** may emerge that are more adapted to the new division of labor
- **40% of workers** will need reskilling within the next 3 years
- **High-risk occupations**: Administrative, manufacturing, and data entry roles
- **Low-risk occupations**: Healthcare, education, and creative roles

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
# Load and explore data
from src.data_loader import load_all_data
df = load_all_data()

# Run predictive models
from src.models import predict_job_loss_risk
predictions = predict_job_loss_risk(df)
```

## Data Sources

1. **World Economic Forum** - Future of Jobs Report 2025
2. **McKinsey Global Institute** - Future of Work research
3. **Bureau of Labor Statistics** - Occupational Employment Projections
4. **Kaggle** - Global AI Impact on Jobs dataset

## Author

AI Job Loss Prediction Research Project

## License

MIT License
