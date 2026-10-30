# Green Destinations Employee Attrition Analytics

An end-to-end HR analytics portfolio project using Python, SQL, scikit-learn, and Power BI.

## Dataset
The provided CSV has 1,470 rows and 35 columns. It contains employee characteristics, job context, satisfaction ratings, tenure, income, and an Attrition target. See the source file in `data/raw/`.

## Quick start
```bash
pip install -r requirements.txt
python analysis.py
```

## Deliverables
- `analysis.py`: data quality, EDA, three classifiers, model evaluation, permutation importance.
- `notebooks/hr_attrition_analysis.ipynb`: notebook runner.
- `sql/hr_analytics.sql`: SQL business questions.
- `data/processed/`: actual output tables generated from the dataset.
- `images/`: generated visualizations.
- `models/best_attrition_pipeline.joblib`: fitted pipeline chosen by holdout F1.
- `reports/final_report.md`: report draft with measured findings.
- `dashboard/POWER_BI_GUIDE.md`: dashboard build guide.

## Responsible use
This is an educational analysis of historical data. Observational patterns are not causal. Model predictions should not be used to make automated employment decisions. Consider fairness, privacy, and human review before any real-world use.
