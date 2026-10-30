"""
Green Destinations HR Analytics pipeline
Run from the project root: python analysis.py

The target is Attrition (Yes/No). EmployeeNumber and constant/administrative
columns are excluded from predictors. The evaluation uses a stratified holdout.
This is an educational benchmark, not a validated employee-level risk system.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay, RocCurveDisplay,
    PrecisionRecallDisplay
)
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "raw" / "green_destinations_attrition.csv"
OUT = ROOT / "data" / "processed"
MODELS = ROOT / "models"
IMAGES = ROOT / "images"
for p in (OUT, MODELS, IMAGES):
    p.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
print("Dataset shape:", df.shape)
print("Missing cells:", int(df.isna().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))

# Data quality summary
quality = pd.DataFrame({
    "column": df.columns,
    "dtype": [str(df[c].dtype) for c in df.columns],
    "missing_count": [int(df[c].isna().sum()) for c in df.columns],
    "unique_values": [int(df[c].nunique(dropna=False)) for c in df.columns]
})
quality.to_csv(OUT / "data_quality_summary.csv", index=False)

# Exact duplicate removal; no missing values expected in the supplied file.
df = df.drop_duplicates().copy()
df["AttritionFlag"] = df["Attrition"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
if df["AttritionFlag"].isna().any():
    raise ValueError("Unexpected Attrition labels. Inspect target values before training.")

# EDA tables
attrition_counts = df["Attrition"].value_counts().rename_axis("Attrition").reset_index(name="Employees")
attrition_counts["Percentage"] = 100 * attrition_counts["Employees"] / len(df)
attrition_counts.to_csv(OUT / "attrition_summary.csv", index=False)

for col in ["Department", "JobRole", "OverTime", "BusinessTravel", "MaritalStatus", "Gender"]:
    table = df.groupby(col, dropna=False).agg(
        Employees=("AttritionFlag", "size"),
        AttritionRate=("AttritionFlag", "mean")
    ).reset_index()
    table["AttritionRatePct"] = 100 * table["AttritionRate"]
    table.to_csv(OUT / f"attrition_by_{col.lower()}.csv", index=False)

numeric_summary = df.select_dtypes(include=np.number).describe().T
numeric_summary.to_csv(OUT / "numeric_summary.csv")

# Visualizations
plt.figure(figsize=(7, 5))
counts = df["Attrition"].value_counts().reindex(["No", "Yes"], fill_value=0)
plt.bar(counts.index, counts.values)
plt.title("Employee Attrition Counts")
plt.xlabel("Attrition")
plt.ylabel("Employees")
plt.tight_layout()
plt.savefig(IMAGES / "attrition_counts.png", dpi=160)
plt.close()

ot = df.groupby("OverTime", as_index=False)["AttritionFlag"].mean()
ot["AttritionRatePct"] = 100 * ot["AttritionFlag"]
plt.figure(figsize=(7, 5))
plt.bar(ot["OverTime"], ot["AttritionRatePct"])
plt.title("Attrition Rate by Overtime Status")
plt.xlabel("Overtime")
plt.ylabel("Attrition rate (%)")
plt.tight_layout()
plt.savefig(IMAGES / "attrition_overtime.png", dpi=160)
plt.close()

role = df.groupby("JobRole", as_index=False)["AttritionFlag"].mean()
role["AttritionRatePct"] = 100 * role["AttritionFlag"]
role = role.sort_values("AttritionRatePct", ascending=False)
plt.figure(figsize=(10, 6))
plt.barh(role["JobRole"], role["AttritionRatePct"])
plt.title("Attrition Rate by Job Role")
plt.xlabel("Attrition rate (%)")
plt.ylabel("Job role")
plt.tight_layout()
plt.savefig(IMAGES / "attrition_by_job_role.png", dpi=160)
plt.close()

plt.figure(figsize=(8, 5))
plt.boxplot([df.loc[df['Attrition']=='No', 'MonthlyIncome'], df.loc[df['Attrition']=='Yes', 'MonthlyIncome']], tick_labels=['No', 'Yes'])
plt.title("Monthly Income by Attrition Status")
plt.xlabel("Attrition")
plt.ylabel("Monthly income (dataset units)")
plt.tight_layout()
plt.savefig(IMAGES / "income_by_attrition.png", dpi=160)
plt.close()

# Classification: exclude target, ID, and constant/administrative columns.
drop_cols = ["Attrition", "AttritionFlag", "EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]
X = df.drop(columns=drop_cols, errors="ignore")
y = df["AttritionFlag"]
categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric = X.select_dtypes(include=np.number).columns.tolist()

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

candidates = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=500, min_samples_leaf=2,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=500, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
}

rows = []
fitted = {}
for name, estimator in candidates.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, proba),
        "PR_AUC_AveragePrecision": average_precision_score(y_test, proba)
    })
    fitted[name] = (pipe, pred, proba)

metrics = pd.DataFrame(rows).sort_values(["F1", "Recall"], ascending=False)
metrics.to_csv(OUT / "model_comparison.csv", index=False)
print("\nModel comparison:\n", metrics.round(3).to_string(index=False))

# Save classification report and confusion matrix for the leading F1 model.
best_name = metrics.iloc[0]["Model"]
best_pipe, best_pred, best_proba = fitted[best_name]
report = classification_report(
    y_test, best_pred, target_names=["No Attrition", "Attrition"],
    output_dict=True, zero_division=0
)
with open(OUT / "best_model_classification_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
pd.DataFrame(
    confusion_matrix(y_test, best_pred),
    index=["Actual No", "Actual Yes"],
    columns=["Predicted No", "Predicted Yes"]
).to_csv(OUT / "best_model_confusion_matrix.csv")

ConfusionMatrixDisplay.from_predictions(
    y_test, best_pred, display_labels=["No Attrition", "Attrition"], cmap="Blues"
)
plt.title(f"Confusion Matrix — {best_name}")
plt.tight_layout()
plt.savefig(IMAGES / "confusion_matrix.png", dpi=160)
plt.close()

plt.figure(figsize=(7, 6))
for name, (pipe, pred, proba) in fitted.items():
    RocCurveDisplay.from_predictions(y_test, proba, name=name, ax=plt.gca())
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
plt.title("ROC Curves — Stratified Holdout")
plt.tight_layout()
plt.savefig(IMAGES / "roc_curves.png", dpi=160)
plt.close()

# Permutation importance is calculated on the original columns (more interpretable than one-hot columns).
perm = permutation_importance(
    best_pipe, X_test, y_test, n_repeats=8,
    random_state=42, scoring="roc_auc", n_jobs=-1
)
importance = pd.DataFrame({
    "Feature": X_test.columns,
    "MeanImportance": perm.importances_mean,
    "StdImportance": perm.importances_std
}).sort_values("MeanImportance", ascending=False)
importance.to_csv(OUT / "permutation_feature_importance.csv", index=False)

top = importance.head(15).sort_values("MeanImportance")
plt.figure(figsize=(9, 7))
plt.barh(top["Feature"], top["MeanImportance"], xerr=top["StdImportance"])
plt.title(f"Permutation Importance (ROC-AUC) — {best_name}")
plt.xlabel("Decrease in ROC-AUC after shuffling")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(IMAGES / "permutation_importance.png", dpi=160)
plt.close()

# Store trained model and the split/test predictions for reproducibility.
joblib.dump(best_pipe, MODELS / "best_attrition_pipeline.joblib")
predictions = X_test.copy()
predictions["ActualAttrition"] = y_test.map({0: "No", 1: "Yes"}).values
predictions["PredictedAttrition"] = pd.Series(best_pred, index=X_test.index).map({0: "No", 1: "Yes"}).values
predictions["PredictedAttritionProbability"] = best_proba
predictions.to_csv(OUT / "holdout_predictions.csv", index=False)

summary = {
    "rows": int(len(df)),
    "columns": int(df.shape[1] - 1),
    "missing_cells": int(df.isna().sum().sum()),
    "duplicate_rows": int(df.duplicated().sum()),
    "attrition_count": int(y.sum()),
    "attrition_rate_pct": float(y.mean() * 100),
    "best_model_by_f1": str(best_name),
    "best_model_metrics": metrics.iloc[0].to_dict(),
    "random_state": 42,
    "test_size": 0.2,
    "positive_class": "Yes"
}
with open(OUT / "project_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, default=float)
print("\nSummary:", json.dumps(summary, indent=2, default=float))
print(f"\nTables: {OUT}\nPlots: {IMAGES}\nModel: {MODELS}")
