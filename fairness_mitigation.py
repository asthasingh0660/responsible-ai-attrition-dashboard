"""
fairness_mitigation.py
======================
Bias Mitigation via Reweighing for HR Attrition Prediction

Implements the Reweighing pre-processing technique to reduce gender-based
bias in the Random Forest model (chosen as it has the best F1 and actually
predicts attrition cases, unlike Logistic Regression).

Reweighing (Kamiran & Calders, 2012):
  Assigns instance weights during training so that each
  (gender, attrition) combination is equally represented.
  No changes to the model architecture — only the training weights change.
  This makes it model-agnostic and easy to apply in practice.

Outputs:
  models/fair_random_forest.pkl          → mitigated model
  results/mitigation_comparison.csv      → before/after table (paper-ready)
  results/mitigation_comparison_plot.png → before/after visualisation
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score
)

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ──────────────────────────────────────────────
# 1. Prepare data  (same split as always)
# ──────────────────────────────────────────────
data = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

FEATURES = ["Age", "Gender", "Education", "JobLevel",
            "MonthlyIncome", "YearsAtCompany"]
TARGET   = "Attrition"

df = data[FEATURES + [TARGET]].copy()

le_gender    = LabelEncoder()
le_attrition = LabelEncoder()
df["Gender"]    = le_gender.fit_transform(df["Gender"])        # Female=0, Male=1
df["Attrition"] = le_attrition.fit_transform(df["Attrition"]) # No=0, Yes=1

X = df[FEATURES]
y = df[TARGET]
gender_series = X["Gender"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

gender_train = gender_series.loc[X_train.index].values
gender_test  = gender_series.loc[X_test.index].values

female_mask = (gender_test == 0)
male_mask   = (gender_test == 1)

# ──────────────────────────────────────────────
# 2. Fairness metric helpers
# ──────────────────────────────────────────────
def disparate_impact(y_pred, female_mask, male_mask):
    rate_f = y_pred[female_mask].mean()
    rate_m = y_pred[male_mask].mean()
    if rate_m == 0:
        return np.nan
    return round(rate_f / rate_m, 3)

def statistical_parity_diff(y_pred, female_mask, male_mask):
    return round(
        y_pred[female_mask].mean() - y_pred[male_mask].mean(), 3
    )

def equal_opportunity_diff(y_pred, y_true, female_mask, male_mask):
    f_pos  = (y_true[female_mask] == 1)
    m_pos  = (y_true[male_mask]   == 1)
    tpr_f  = y_pred[female_mask][f_pos].mean() if f_pos.sum()  > 0 else np.nan
    tpr_m  = y_pred[male_mask][m_pos].mean()   if m_pos.sum() > 0 else np.nan
    if np.isnan(tpr_f) or np.isnan(tpr_m):
        return np.nan
    return round(tpr_f - tpr_m, 3)

def evaluate(model, X_test, y_test, female_mask, male_mask, label):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "Model":                    label,
        "Accuracy":                 round(accuracy_score(y_test, y_pred), 3),
        "F1 Score":                 round(f1_score(y_test, y_pred, zero_division=0), 3),
        "AUC":                      round(roc_auc_score(y_test, y_prob), 3),
        "Pred Rate (Female)":       round(y_pred[female_mask].mean(), 3),
        "Pred Rate (Male)":         round(y_pred[male_mask].mean(), 3),
        "Disparate Impact":         disparate_impact(y_pred, female_mask, male_mask),
        "Stat Parity Diff":         statistical_parity_diff(y_pred, female_mask, male_mask),
        "Equal Opportunity Diff":   equal_opportunity_diff(
                                        y_pred, y_test.values,
                                        female_mask, male_mask
                                    ),
    }

# ──────────────────────────────────────────────
# 3. Baseline Random Forest  (no mitigation)
# ──────────────────────────────────────────────
print("="*60)
print("  BIAS MITIGATION VIA REWEIGHING")
print("="*60)

print("\n▶ Training baseline Random Forest (no mitigation)...")

baseline_rf = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    RandomForestClassifier(n_estimators=100, random_state=42))
])
baseline_rf.fit(X_train, y_train)
baseline_results = evaluate(
    baseline_rf, X_test, y_test, female_mask, male_mask,
    "Random Forest (Baseline)"
)

print(f"   Accuracy: {baseline_results['Accuracy']}  "
      f"F1: {baseline_results['F1 Score']}  "
      f"AUC: {baseline_results['AUC']}")
print(f"   Disparate Impact: {baseline_results['Disparate Impact']}  "
      f"SPD: {baseline_results['Stat Parity Diff']}")

# ──────────────────────────────────────────────
# 4. Compute Reweighing sample weights
# ──────────────────────────────────────────────
print("\n▶ Computing reweighing sample weights...")

"""
Reweighing formula (Kamiran & Calders, 2012):
  W(x) = P(sensitive) * P(label) / P(sensitive, label)

  For each combination of (gender, attrition_label):
    expected_count = (group_size / total) * (label_count / total) * total
    actual_count   = actual samples in that cell
    weight         = expected_count / actual_count
"""

train_df = X_train.copy()
train_df["Attrition"] = y_train.values
train_df["Gender_raw"] = gender_train

n_total  = len(train_df)
n_female = (gender_train == 0).sum()
n_male   = (gender_train == 1).sum()
n_yes    = (y_train == 1).sum()
n_no     = (y_train == 0).sum()

# Expected proportions under independence
p_female = n_female / n_total
p_male   = n_male   / n_total
p_yes    = n_yes    / n_total
p_no     = n_no     / n_total

# Actual counts per cell
cells = train_df.groupby(["Gender_raw", "Attrition"]).size().to_dict()

def get_weight(gender_val, attrition_val):
    p_group = p_female if gender_val == 0 else p_male
    p_label = p_yes    if attrition_val == 1 else p_no
    expected = p_group * p_label * n_total
    actual   = cells.get((gender_val, attrition_val), 1)
    return expected / actual

sample_weights = np.array([
    get_weight(g, a)
    for g, a in zip(gender_train, y_train.values)
])

# Print weights for transparency
print("\n   Sample weights per (gender, attrition) cell:")
for (g, a), cnt in sorted(cells.items()):
    w = get_weight(g, a)
    g_label = "Female" if g == 0 else "Male"
    a_label = "Yes"    if a == 1 else "No"
    print(f"   ({g_label}, Attrition={a_label}) — n={cnt:4d}  weight={w:.4f}")

# ──────────────────────────────────────────────
# 5. Train Fair Random Forest with reweighing
# ──────────────────────────────────────────────
print("\n▶ Training Fair Random Forest (with reweighing)...")

# Pipeline: scale first, then pass weights to classifier
scaler  = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

fair_clf = RandomForestClassifier(n_estimators=100, random_state=42)
fair_clf.fit(X_train_scaled, y_train, sample_weight=sample_weights)

# Wrap into a simple callable that mimics pipeline interface
class FairPipeline:
    """Thin wrapper so fair model works identically to sklearn Pipeline."""
    def __init__(self, scaler, clf):
        self.scaler = scaler
        self.clf    = clf

    def predict(self, X):
        return self.clf.predict(self.scaler.transform(X))

    def predict_proba(self, X):
        return self.clf.predict_proba(self.scaler.transform(X))

    def fit(self, X, y, **kwargs):
        Xs = self.scaler.fit_transform(X)
        self.clf.fit(Xs, y, **kwargs)
        return self

fair_pipeline = FairPipeline(scaler, fair_clf)

fair_results = evaluate(
    fair_pipeline, X_test, y_test, female_mask, male_mask,
    "Random Forest (Fair — Reweighing)"
)

print(f"   Accuracy: {fair_results['Accuracy']}  "
      f"F1: {fair_results['F1 Score']}  "
      f"AUC: {fair_results['AUC']}")
print(f"   Disparate Impact: {fair_results['Disparate Impact']}  "
      f"SPD: {fair_results['Stat Parity Diff']}")

# Save fair model
joblib.dump(fair_pipeline, "models/fair_random_forest.pkl")
print("   ✅ Saved → models/fair_random_forest.pkl")

# ──────────────────────────────────────────────
# 6. Before / After comparison
# ──────────────────────────────────────────────
results_df = pd.DataFrame([baseline_results, fair_results])
results_df.to_csv("results/mitigation_comparison.csv", index=False)

print("\n" + "="*60)
print("  BEFORE / AFTER MITIGATION — PAPER-READY TABLE")
print("="*60)
paper_cols = ["Model", "Accuracy", "F1 Score", "AUC",
              "Disparate Impact", "Stat Parity Diff", "Equal Opportunity Diff"]
print(results_df[paper_cols].to_string(index=False))

# Compute deltas
di_change  = fair_results["Disparate Impact"] - baseline_results["Disparate Impact"]
acc_change = fair_results["Accuracy"]          - baseline_results["Accuracy"]
f1_change  = fair_results["F1 Score"]          - baseline_results["F1 Score"]

print(f"\n  Δ Disparate Impact : {di_change:+.3f}  "
      f"({'closer to 1.0 = fairer ✅' if abs(fair_results['Disparate Impact'] - 1.0) < abs(baseline_results['Disparate Impact'] - 1.0) else 'further from 1.0 ⚠️'})")
print(f"  Δ Accuracy         : {acc_change:+.3f}")
print(f"  Δ F1 Score         : {f1_change:+.3f}")

print("\n  Research interpretation:")
print("  Reweighing adjusts training sample weights so each")
print("  (gender × attrition) combination is equally represented.")
print("  The fairness-accuracy tradeoff is quantified above.")

# ──────────────────────────────────────────────
# 7. Visualisation
# ──────────────────────────────────────────────
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "Bias Mitigation via Reweighing — Before vs After",
    fontsize=13, fontweight="bold", y=1.02
)

labels  = ["Baseline RF", "Fair RF\n(Reweighing)"]
colors  = ["#4C72B0", "#55A868"]
metrics = [
    ("Accuracy",         "Accuracy",         [baseline_results["Accuracy"],  fair_results["Accuracy"]],  (0.7, 1.0)),
    ("F1 Score",         "F1 Score",          [baseline_results["F1 Score"],  fair_results["F1 Score"]],  (0.0, 0.5)),
    ("Disparate Impact", "Disparate Impact",  [baseline_results["Disparate Impact"], fair_results["Disparate Impact"]], (0.5, 1.8)),
]

for ax, (title, ylabel, vals, ylim) in zip(axes, metrics):
    bars = ax.bar(labels, vals, color=colors, width=0.4, edgecolor="white")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)

    # Value labels on bars
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    # Fairness reference lines on DI plot
    if title == "Disparate Impact":
        ax.axhline(1.0,  color="black",   linestyle="-",  linewidth=1,   label="Ideal (1.0)")
        ax.axhline(0.8,  color="#C44E52", linestyle="--", linewidth=1.2, label="Min threshold (0.8)")
        ax.axhline(1.25, color="#C44E52", linestyle="--", linewidth=1.2, label="Max threshold (1.25)")
        ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig("results/mitigation_comparison_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n📊 Plot saved → results/mitigation_comparison_plot.png")
print("\n✅ Mitigation analysis complete.")