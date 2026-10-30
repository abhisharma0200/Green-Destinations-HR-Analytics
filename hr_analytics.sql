-- Green Destinations HR Analytics (SQLite)
-- Import data/raw/green_destinations_attrition.csv as employees.
-- Keep employee-level data internal; share only aggregated reporting outputs.

-- 1. Workforce and attrition KPI
SELECT COUNT(*) AS employee_count,
       SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS attrition_count,
       ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS attrition_rate_pct
FROM employees;

-- 2. Attrition by department
SELECT Department, COUNT(*) AS employees,
       SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS leavers,
       ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS attrition_rate_pct
FROM employees
GROUP BY Department
ORDER BY attrition_rate_pct DESC;

-- 3. Attrition by overtime status
SELECT OverTime, COUNT(*) AS employees,
       ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS attrition_rate_pct
FROM employees
GROUP BY OverTime
ORDER BY attrition_rate_pct DESC;

-- 4. Attrition by job role (interpret small groups cautiously)
SELECT JobRole, COUNT(*) AS employees,
       ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS attrition_rate_pct
FROM employees
GROUP BY JobRole
ORDER BY attrition_rate_pct DESC;

-- 5. Attrition by work-life balance
SELECT WorkLifeBalance, COUNT(*) AS employees,
       ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS attrition_rate_pct
FROM employees
GROUP BY WorkLifeBalance
ORDER BY WorkLifeBalance;

-- 6. Monthly income comparison (descriptive, not causal)
SELECT Attrition,
       ROUND(AVG(MonthlyIncome), 2) AS average_monthly_income,
       ROUND(MIN(MonthlyIncome), 2) AS min_income,
       ROUND(MAX(MonthlyIncome), 2) AS max_income
FROM employees
GROUP BY Attrition;

-- 7. Attrition by years at company
SELECT YearsAtCompany, COUNT(*) AS employees,
       ROUND(100.0 * SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS attrition_rate_pct
FROM employees
GROUP BY YearsAtCompany
HAVING COUNT(*) >= 10
ORDER BY YearsAtCompany;
