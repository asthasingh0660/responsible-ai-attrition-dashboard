"""
train_model.py
==============
Fairness-Aware Comparative ML Pipeline for Employee Attrition Prediction

Trains multiple models, evaluates them with full metrics (Accuracy, F1, AUC),
and computes fairness metrics (Disparate Impact, Statistical Parity Difference)
per model — forming the experimental core of the research paper.

Models trained:
  - Logistic Regression   (interpretable baseline)
  - Random Forest         (ensemble baseline)
  - XGBoost               (gradient boosting — install xgboost to enable)
  - Neural Network / MLP  (deep learning baseline)

Outputs:
  - models/   → one .pkl per model
  - features.pkl
  - results/model_comparison.csv   → accuracy + fairness table
"""

import os
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report
)

# Try importing XGBoost — optional, skipped if not installed
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not installed. Skipping. Run: pip install xgboost")

# -----------------------
# Setup output dirs
# -----------------------
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# -----------------------
# Load & prepare data
# -----------------------
data = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

FEATURES = [
    "Age",
    "Gender",
    "Education",
    "JobLevel",
    "MonthlyIncome",
    "YearsAtCompany",
]

TARGET = "Attrition"
SENSITIVE_ATTR = "Gender"   # used for fairness metrics

df = data[FEATURES + [TARGET]].copy()

# Encode
le_gender    = LabelEncoder()
le_attrition = LabelEncoder()
df["Gender"]    = le_gender.fit_transform(df["Gender"])       # Female=0, Male=1
df["Attrition"] = le_attrition.fit_transform(df["Attrition"]) # No=0, Yes=1

X = df[FEATURES]
y = df[TARGET]

# Store raw gender column for fairness computation (matches X index)
gender_series = X["Gender"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

gender_test = gender_series.loc[X_test.index]

# -----------------------
# Define models
# -----------------------
model_definitions = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",   LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",   RandomForestClassifier(n_estimators=100, random_state=42))
    ]),
    "Neural Network (MLP)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",   MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=42
        ))
    ]),
}

if XGBOOST_AVAILABLE:
    model_definitions["XGBoost"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",   XGBClassifier(
            n_estimators=100,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        ))
    ])

# -----------------------
# Fairness metric helpers
# -----------------------
def disparate_impact(y_pred, gender):
    """
    Ratio of positive prediction rate: unprivileged / privileged.
    Unprivileged = Female (0), Privileged = Male (1).
    Ideal value = 1.0. Below 0.8 = fairness risk (80% rule).
    """
    female_mask = (gender == 0)
    male_mask   = (gender == 1)
    rate_female = y_pred[female_mask].mean()
    rate_male   = y_pred[male_mask].mean()
    if rate_male == 0:
        return float("nan")
    return rate_female / rate_male


def statistical_parity_diff(y_pred, gender):
    """
    Difference in positive prediction rates: female - male.
    Ideal value = 0.0. Negative = females predicted positive less often.
    """
    female_mask = (gender == 0)
    male_mask   = (gender == 1)
    rate_female = y_pred[female_mask].mean()
    rate_male   = y_pred[male_mask].mean()
    return rate_female - rate_male


# -----------------------
# Train, evaluate, log
# -----------------------
results = []
trained_models = {}

print("\n" + "="*60)
print("  Fairness-Aware Comparative ML Training Pipeline")
print("="*60)

for name, pipeline in model_definitions.items():
    print(f"\n▶ Training: {name}")

    pipeline.fit(X_train, y_train)

    y_pred      = pipeline.predict(X_test)
    y_prob      = pipeline.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)

    di   = disparate_impact(y_pred, gender_test.values)
    spd  = statistical_parity_diff(y_pred, gender_test.values)

    # Cross-validated AUC for robustness
    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc").mean()

    print(f"   Accuracy : {acc:.3f}")
    print(f"   F1 Score : {f1:.3f}")
    print(f"   AUC      : {auc:.3f}  (5-fold CV AUC: {cv_auc:.3f})")
    print(f"   Disparate Impact       : {di:.3f}  {'⚠️ Fairness risk' if di < 0.8 else '✅ OK'}")
    print(f"   Statistical Parity Diff: {spd:.3f}")

    results.append({
        "Model":                    name,
        "Accuracy":                 round(acc, 3),
        "F1 Score":                 round(f1, 3),
        "AUC":                      round(auc, 3),
        "CV AUC (5-fold)":          round(cv_auc, 3),
        "Disparate Impact":         round(di, 3),
        "Statistical Parity Diff":  round(spd, 3),
    })

    trained_models[name] = pipeline
    model_filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".pkl"
    joblib.dump(pipeline, f"models/{model_filename}")
    print(f"   ✅ Saved → models/{model_filename}")

# -----------------------
# Save comparison table
# -----------------------
results_df = pd.DataFrame(results)
results_df.to_csv("results/model_comparison.csv", index=False)

# Also save best model as the default (by AUC)
best_model_name = results_df.loc[results_df["AUC"].idxmax(), "Model"]
best_filename   = best_model_name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".pkl"
best_pipeline   = trained_models[best_model_name]
joblib.dump(best_pipeline, "attrition_model.pkl")
joblib.dump(FEATURES, "features.pkl")

# -----------------------
# Print summary table
# -----------------------
print("\n" + "="*60)
print("  MODEL COMPARISON RESULTS")
print("="*60)
print(results_df.to_string(index=False))
print(f"\n🏆 Best model by AUC: {best_model_name}")
print(f"   Saved as attrition_model.pkl (used by dashboard)")
print(f"\n📄 Full results saved → results/model_comparison.csv")
print("\n✅ Training pipeline complete.")
