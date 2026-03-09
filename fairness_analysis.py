"""
fairness_analysis.py
====================
Per-Model Fairness Evaluation for Employee Attrition Prediction

Loads all trained models and evaluates each against two established
Responsible AI fairness metrics:

  1. Disparate Impact (DI)
       = P(ŷ=1 | female) / P(ŷ=1 | male)
       Threshold: DI < 0.8 or DI > 1.25 indicates fairness risk
       (the "80% rule" from US employment law / EEOC guidelines)

  2. Statistical Parity Difference (SPD)
       = P(ŷ=1 | female) - P(ŷ=1 | male)
       Ideal = 0.0. Negative = females predicted as leaving less often.

  3. Equal Opportunity Difference (EOD)  [bonus metric]
       = TPR(female) - TPR(male)
       Measures whether the model catches actual attrition equally across groups.
       Ideal = 0.0.

Also flags the accuracy-fairness tradeoff — the core research insight.

Outputs:
  results/fairness_comparison.csv   → full per-model fairness table
  results/model_comparison_full.csv → merged performance + fairness table
"""

import os
import glob
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix
)

os.makedirs("results", exist_ok=True)

# ──────────────────────────────────────────────
# 1. Prepare data  (same split as train_model.py)
# ──────────────────────────────────────────────
data = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

FEATURES = ["Age", "Gender", "Education", "JobLevel",
            "MonthlyIncome", "YearsAtCompany"]
TARGET   = "Attrition"

df = data[FEATURES + [TARGET]].copy()

le_gender    = LabelEncoder()
le_attrition = LabelEncoder()
df["Gender"]    = le_gender.fit_transform(df["Gender"])       # Female=0, Male=1
df["Attrition"] = le_attrition.fit_transform(df["Attrition"]) # No=0, Yes=1

X = df[FEATURES]
y = df[TARGET]
gender_series = X["Gender"].copy()

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
gender_test = gender_series.loc[X_test.index].values

female_mask = (gender_test == 0)
male_mask   = (gender_test == 1)

print(f"Test set: {len(X_test)} employees")
print(f"  Female: {female_mask.sum()}  |  Male: {male_mask.sum()}")
print(f"  Actual attrition rate — Female: "
      f"{y_test[female_mask].mean():.3f}  "
      f"Male: {y_test[male_mask].mean():.3f}\n")

# ──────────────────────────────────────────────
# 2. Fairness metric functions
# ──────────────────────────────────────────────
def disparate_impact(y_pred, female_mask, male_mask):
    rate_f = y_pred[female_mask].mean()
    rate_m = y_pred[male_mask].mean()
    if rate_m == 0:
        return np.nan
    return round(rate_f / rate_m, 3)

def statistical_parity_diff(y_pred, female_mask, male_mask):
    rate_f = y_pred[female_mask].mean()
    rate_m = y_pred[male_mask].mean()
    return round(rate_f - rate_m, 3)

def equal_opportunity_diff(y_pred, y_true, female_mask, male_mask):
    """TPR (recall) difference: female - male, among actual attrition cases."""
    # female TPR
    f_actual_pos = (y_true[female_mask] == 1)
    tpr_f = y_pred[female_mask][f_actual_pos].mean() if f_actual_pos.sum() > 0 else np.nan
    # male TPR
    m_actual_pos = (y_true[male_mask] == 1)
    tpr_m = y_pred[male_mask][m_actual_pos].mean() if m_actual_pos.sum() > 0 else np.nan

    if np.isnan(tpr_f) or np.isnan(tpr_m):
        return np.nan
    return round(tpr_f - tpr_m, 3)

def fairness_verdict(di, spd):
    issues = []
    if not np.isnan(di):
        if di < 0.8:
            issues.append("DI below 0.8 (female under-predicted)")
        elif di > 1.25:
            issues.append("DI above 1.25 (male under-predicted)")
    if abs(spd) > 0.05:
        issues.append(f"SPD |{spd}| > 0.05")
    return " | ".join(issues) if issues else "✅ Fair"

# ──────────────────────────────────────────────
# 3. Load and evaluate all models
# ──────────────────────────────────────────────
model_files = sorted(glob.glob("models/*.pkl"))

if not model_files:
    print("❌ No models found in models/. Run train_model.py first.")
    exit(1)

print("="*65)
print("  PER-MODEL FAIRNESS EVALUATION")
print("="*65)

rows = []

for path in model_files:
    # Friendly name from filename
    raw = os.path.basename(path).replace(".pkl", "").replace("_", " ").title()
    raw = raw.replace("Mlp", "MLP").replace("Xgboost", "XGBoost")

    model = joblib.load(path)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc  = round(accuracy_score(y_test, y_pred), 3)
    f1   = round(f1_score(y_test, y_pred, zero_division=0), 3)
    auc  = round(roc_auc_score(y_test, y_prob), 3)

    di   = disparate_impact(y_pred, female_mask, male_mask)
    spd  = statistical_parity_diff(y_pred, female_mask, male_mask)
    eod  = equal_opportunity_diff(y_pred, y_test.values, female_mask, male_mask)

    # Attrition prediction rates by gender
    rate_f = round(y_pred[female_mask].mean(), 3)
    rate_m = round(y_pred[male_mask].mean(), 3)

    verdict = fairness_verdict(di, spd)

    rows.append({
        "Model":                      raw,
        "Accuracy":                   acc,
        "F1 Score":                   f1,
        "AUC":                        auc,
        "Pred Rate (Female)":         rate_f,
        "Pred Rate (Male)":           rate_m,
        "Disparate Impact":           di,
        "Stat Parity Diff":           spd,
        "Equal Opportunity Diff":     eod,
        "Fairness Verdict":           verdict,
    })

    print(f"\n▶ {raw}")
    print(f"   Performance  → Accuracy: {acc}  F1: {f1}  AUC: {auc}")
    print(f"   Pred rates   → Female: {rate_f}  Male: {rate_m}")
    print(f"   Disparate Impact       : {di}")
    print(f"   Stat Parity Diff       : {spd}")
    print(f"   Equal Opportunity Diff : {eod}")
    print(f"   Verdict: {verdict}")

results_df = pd.DataFrame(rows)
results_df.to_csv("results/fairness_comparison.csv", index=False)

# ──────────────────────────────────────────────
# 4. Research insight: accuracy vs fairness tradeoff
# ──────────────────────────────────────────────
print("\n" + "="*65)
print("  RESEARCH INSIGHT: ACCURACY vs FAIRNESS TRADEOFF")
print("="*65)

valid = results_df.dropna(subset=["Disparate Impact"])
if len(valid) > 1:
    corr = valid["AUC"].corr(valid["Disparate Impact"].apply(
        lambda x: abs(x - 1.0)  # distance from perfect fairness
    ))
    print(f"\n  Correlation (AUC vs |DI - 1.0|): {corr:.3f}")
    if corr > 0.3:
        print("  → Higher-AUC models tend to exhibit MORE fairness deviation.")
        print("    This is the core accuracy-fairness tradeoff finding.")
    elif corr < -0.3:
        print("  → Higher-AUC models tend to be MORE fair in this dataset.")
    else:
        print("  → No strong linear tradeoff observed. "
              "Both dimensions vary independently.")

best_acc = results_df.loc[results_df["Accuracy"].idxmax(), "Model"]
best_f1  = results_df.loc[results_df["F1 Score"].idxmax(), "Model"]
best_auc = results_df.loc[results_df["AUC"].idxmax(), "Model"]

print(f"\n  Best Accuracy : {best_acc}")
print(f"  Best F1 Score : {best_f1}")
print(f"  Best AUC      : {best_auc}")

fair_models = results_df[results_df["Fairness Verdict"] == "✅ Fair"]
print(f"\n  Models passing all fairness thresholds: "
      f"{len(fair_models)}/{len(results_df)}")
if len(fair_models) > 0:
    print(f"  → {', '.join(fair_models['Model'].tolist())}")

# ──────────────────────────────────────────────
# 5. Visualisations
# ──────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
MODELS = results_df["Model"].tolist()
x_pos  = np.arange(len(MODELS))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    "Model Performance vs Fairness — HR Attrition Prediction",
    fontsize=14, fontweight="bold", y=1.02
)

# — Plot 1: Performance metrics
ax1 = axes[0]
width = 0.25
ax1.bar(x_pos - width, results_df["Accuracy"], width, label="Accuracy", color="#4C72B0")
ax1.bar(x_pos,         results_df["F1 Score"], width, label="F1 Score",  color="#55A868")
ax1.bar(x_pos + width, results_df["AUC"],      width, label="AUC",       color="#C44E52")
ax1.set_xticks(x_pos)
ax1.set_xticklabels(MODELS, rotation=15, ha="right", fontsize=9)
ax1.set_ylim(0, 1.05)
ax1.set_title("Predictive Performance")
ax1.set_ylabel("Score")
ax1.legend(fontsize=8)
ax1.axhline(0.8, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

# — Plot 2: Disparate Impact per model
ax2 = axes[1]
colors = []
for di in results_df["Disparate Impact"]:
    if pd.isna(di):
        colors.append("#AAAAAA")
    elif di < 0.8 or di > 1.25:
        colors.append("#C44E52")
    else:
        colors.append("#55A868")

bars = ax2.bar(x_pos, results_df["Disparate Impact"].fillna(0),
               color=colors, edgecolor="white")
ax2.axhline(1.0, color="black",  linestyle="-",  linewidth=1,   label="Ideal (1.0)")
ax2.axhline(0.8, color="#C44E52", linestyle="--", linewidth=1.2, label="Lower bound (0.8)")
ax2.axhline(1.25,color="#C44E52", linestyle="--", linewidth=1.2, label="Upper bound (1.25)")
ax2.set_xticks(x_pos)
ax2.set_xticklabels(MODELS, rotation=15, ha="right", fontsize=9)
ax2.set_title("Disparate Impact (Gender)")
ax2.set_ylabel("DI ratio")
ax2.legend(fontsize=7)

green_patch = mpatches.Patch(color="#55A868", label="Fair")
red_patch   = mpatches.Patch(color="#C44E52", label="Fairness risk")
grey_patch  = mpatches.Patch(color="#AAAAAA", label="Undefined (no positives)")
ax2.legend(handles=[green_patch, red_patch, grey_patch], fontsize=7, loc="upper right")

# — Plot 3: Prediction rates by gender
ax3 = axes[2]
width = 0.35
ax3.bar(x_pos - width/2, results_df["Pred Rate (Female)"],
        width, label="Female", color="#DD8452")
ax3.bar(x_pos + width/2, results_df["Pred Rate (Male)"],
        width, label="Male",   color="#4C72B0")
ax3.set_xticks(x_pos)
ax3.set_xticklabels(MODELS, rotation=15, ha="right", fontsize=9)
ax3.set_title("Predicted Attrition Rate by Gender")
ax3.set_ylabel("Proportion predicted leaving")
ax3.legend(fontsize=8)

plt.tight_layout()
plt.savefig("results/fairness_comparison_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n📊 Plot saved → results/fairness_comparison_plot.png")

# ──────────────────────────────────────────────
# 6. Paper-ready summary table
# ──────────────────────────────────────────────
print("\n" + "="*65)
print("  PAPER-READY SUMMARY TABLE")
print("="*65)
paper_cols = ["Model", "Accuracy", "F1 Score", "AUC",
              "Disparate Impact", "Stat Parity Diff", "Fairness Verdict"]
print(results_df[paper_cols].to_string(index=False))

print("\n✅ Fairness analysis complete.")
print("   results/fairness_comparison.csv")
print("   results/fairness_comparison_plot.png")