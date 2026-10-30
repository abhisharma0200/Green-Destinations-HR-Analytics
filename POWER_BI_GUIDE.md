# Green Destinations HR Analytics — Implementation Guide

## Project objective
Analyze employee attrition patterns, quantify descriptive differences across workforce groups, compare classification models, and create an HR dashboard. This dataset is a commonly used historical HR attrition sample; it should not be treated as live company data or as evidence of causation.

## Run
1. Install Python 3.10+.
2. `pip install -r requirements.txt`
3. Run `python analysis.py` from this folder.
4. Review generated CSVs in `data/processed/`, charts in `images/`, and the model in `models/`.
5. Open `notebooks/hr_attrition_analysis.ipynb` and run the pipeline cell if you prefer Jupyter.

## Power BI data model
Import:
- `data/processed/attrition_summary.csv`
- `data/processed/attrition_by_department.csv`
- `data/processed/attrition_by_jobrole.csv`
- `data/processed/attrition_by_overtime.csv`
- `data/processed/attrition_by_businesstravel.csv`
- `data/processed/attrition_by_maritalstatus.csv`
- `data/processed/attrition_by_gender.csv`
- `data/processed/model_comparison.csv`
- `data/processed/permutation_feature_importance.csv`

You can also import the raw employees table for slicers, but restrict access and avoid exposing employee-level details in public reports.

## Suggested Power BI measures (if raw table is named employees)
Employee Count = COUNTROWS(employees)
Attrition Count = CALCULATE(COUNTROWS(employees), employees[Attrition] = "Yes")
Attrition Rate = DIVIDE([Attrition Count], [Employee Count], 0)
Active Employees = CALCULATE(COUNTROWS(employees), employees[Attrition] = "No")
Average Monthly Income = AVERAGE(employees[MonthlyIncome])

## Dashboard pages
1. Executive Overview: employee count, attrition count, attrition rate, active employees, attrition distribution.
2. Attrition Drivers (descriptive): attrition rate by overtime, department, job role, business travel, work-life balance.
3. Workforce Profile: age/income/tenure distributions; use only appropriate aggregated views.
4. Model Evaluation: model comparison table, ROC curves, confusion matrix, permutation importance; explain precision/recall trade-offs.

## Model notes
- The positive class is Attrition = Yes.
- A stratified 80/20 holdout with random_state=42 is used.
- One-hot encoding and numeric scaling are fit inside a Pipeline to reduce preprocessing leakage.
- EmployeeNumber, EmployeeCount, Over18, StandardHours are excluded as ID/administrative or constant-like fields.
- The data is imbalanced; accuracy alone is not enough. Report precision, recall, F1, ROC-AUC and average precision. Thresholds should be chosen based on the intended HR workflow.
- Feature importance is associational, not causal. Avoid claims that any feature causes attrition.
- Do not use a model score to make automated employment decisions. Treat predictions as exploratory and use human review, fairness checks, and data governance.

## Report
Use `reports/final_report.md` and update all results from generated outputs after running the script. Do not report metrics from a different run or imply the model is production-ready.
