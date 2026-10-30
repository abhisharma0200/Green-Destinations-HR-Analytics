# Green Destinations Employee Attrition Analytics

**Project type:** End-to-end HR Analytics and Machine Learning  
**Tools:** Python, Pandas, Matplotlib, scikit-learn, SQL, Power BI

## Executive summary
The uploaded dataset contains **1,470 records and 35 original columns**. The observed attrition rate is **16.12%** (237 records marked `Attrition = Yes`). Initial checks found **0 missing cells** and **0 exact duplicate rows**. These findings describe this file only and do not establish causes of employee turnover.

## 1. Business problem and objectives
The objective is to understand employee attrition patterns and create a reproducible analytics workflow that supports human-led review. The work includes data quality checks, descriptive analysis, SQL queries, model comparison, and dashboard planning.

## 2. Dataset and preparation
The file contains employee characteristics, job context, satisfaction ratings, tenure, income, and an `Attrition` target. `EmployeeNumber` is excluded as an identifier; `EmployeeCount`, `Over18`, and `StandardHours` are excluded as administrative or constant-like fields. The pipeline removes exact duplicates, encodes categorical variables, and standardizes numeric fields within a scikit-learn pipeline. The split is stratified (80/20, random state 42).

## 3. Descriptive findings
- **Overtime:** attrition is **30.5%** for employees with overtime versus **10.4%** for employees without overtime.
- **Job role:** the highest observed role-level attrition rate is **Sales Representative (39.8%)**.
- **Department:** the highest observed department-level attrition rate is **Sales (20.6%)**.
- Review group sizes and context before acting on subgroup differences. These are associations, not evidence of causation.

## 4. Machine learning methodology and results
Three classifiers were compared on the same stratified holdout: Logistic Regression, Random Forest, and Extra Trees. Accuracy is not sufficient by itself because attrition is the minority class. Precision, recall, F1, ROC-AUC, and average precision are also reported.

| Model               |   Accuracy |   Precision |   Recall |    F1 |   ROC_AUC |   PR_AUC_AveragePrecision |
|:--------------------|-----------:|------------:|---------:|------:|----------:|--------------------------:|
| Logistic Regression |      0.752 |       0.349 |    0.638 | 0.451 |     0.803 |                     0.561 |
| Extra Trees         |      0.85  |       0.556 |    0.319 | 0.405 |     0.805 |                     0.539 |
| Random Forest       |      0.837 |       0.471 |    0.17  | 0.25  |     0.797 |                     0.453 |

**Highest F1 in this run:** Logistic Regression (F1 **0.451**, recall **0.638**, precision **0.349**, ROC-AUC **0.803**, average precision **0.561**).

The top permutation-importance features in this run are **OverTime, NumCompaniesWorked, BusinessTravel, YearsSinceLastPromotion, MaritalStatus**. Permutation importance describes model reliance, not causal impact. The best model is an educational baseline, not a validated HR decision system. Threshold selection should depend on the relative cost of false positives and false negatives.

## 5. SQL analysis
`sql/hr_analytics.sql` includes queries for total headcount, attrition rate, department and job-role comparisons, overtime, work-life balance, income summaries, and tenure bands. Run these after importing the CSV as a SQLite table named `employees`.

## 6. Power BI dashboard plan
1. **Executive Overview:** headcount, attrition count/rate, active employees, attrition distribution.
2. **Attrition Patterns:** overtime, department, job role, business travel, work-life balance.
3. **Workforce Profile:** aggregated age, income, tenure, and satisfaction distributions.
4. **Model Evaluation:** model comparison, confusion matrix, ROC curves, and permutation importance.

Suggested measures (assuming raw table is named `employees`):
- `Employee Count = COUNTROWS(employees)`
- `Attrition Count = CALCULATE(COUNTROWS(employees), employees[Attrition] = "Yes")`
- `Attrition Rate = DIVIDE([Attrition Count], [Employee Count], 0)`
- `Active Employees = CALCULATE(COUNTROWS(employees), employees[Attrition] = "No")`
- `Average Monthly Income = AVERAGE(employees[MonthlyIncome])`

## 7. Recommendations for further investigation
1. Review workload and support practices for overtime groups with HR stakeholders.
2. Investigate job-role patterns after checking sample sizes and role context.
3. Consider pilots around early-tenure support, career progression, manager support, and work-life balance.
4. Track interventions prospectively before claiming they reduce attrition.
5. Do not use model scores for automated employment decisions. Require human review, privacy safeguards, and subgroup fairness checks.

## 8. Limitations
This is a historical sample dataset, not live workforce data. A single stratified holdout does not guarantee generalization. Attrition is imbalanced; the classification threshold affects precision and recall. Feature importance and subgroup comparisons do not establish causality. No intervention data is available to estimate the impact of retention actions.

## 9. Conclusion
This project creates a reproducible workflow for HR data quality checks, descriptive analytics, classification benchmarking, and Power BI reporting. It highlights patterns that may merit further investigation while documenting the limitations of observational data and a single holdout evaluation.

## References
- Uploaded Green Destinations employee attrition CSV (provided for this project).
- scikit-learn documentation: https://scikit-learn.org/stable/
- pandas documentation: https://pandas.pydata.org/docs/
