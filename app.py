import streamlit as st
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (9, 5)
plt.rcParams["axes.titlesize"]  = 13
plt.rcParams["axes.labelsize"]  = 11

st.set_page_config(
    page_title="HR Attrition — Responsible AI Dashboard",
    page_icon="",
    layout="wide"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600&family=IBM+Plex+Mono&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .insight-box {
        background: #eff6ff; border: 1px solid #bfdbfe;
        border-radius: 6px; padding: 14px 18px; margin: 10px 0; font-size: 0.93rem;
        color: #1e293b !important;
    }
    .finding-box {
        background: #f0fdf4; border: 1px solid #bbf7d0;
        border-radius: 6px; padding: 14px 18px; margin: 10px 0; font-size: 0.93rem;
        color: #1e293b !important;
    }
    .warn-box {
        background: #fff7ed; border: 1px solid #fed7aa;
        border-radius: 6px; padding: 14px 18px; margin: 10px 0; font-size: 0.93rem;
        color: #1e293b !important;
    }
    .section-tag {
        display: inline-block; background: #2563eb; color: white;
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em;
        padding: 2px 10px; border-radius: 20px; margin-bottom: 6px;
        text-transform: uppercase;
    }
    h1 { font-weight: 600; }
    .stMetric label { font-size: 0.78rem !important; color: #64748b !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def tag(text):
    st.markdown(f'<span class="section-tag">{text}</span>', unsafe_allow_html=True)

def insight(text):
    st.markdown(f'<div class="insight-box">{text}</div>', unsafe_allow_html=True)

def finding(text):
    st.markdown(f'<div class="finding-box">{text}</div>', unsafe_allow_html=True)

def warn(text):
    st.markdown(f'<div class="warn-box">{text}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load data & models
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

@st.cache_resource
def load_model():
    return joblib.load("attrition_model.pkl")

@st.cache_data
def load_results():
    r = {}
    if os.path.exists("results/fairness_comparison.csv"):
        r["fairness"] = pd.read_csv("results/fairness_comparison.csv")
    if os.path.exists("results/mitigation_comparison.csv"):
        r["mitigation"] = pd.read_csv("results/mitigation_comparison.csv")
    return r

data     = load_data()
model    = load_model()
features = joblib.load("features.pkl")
results  = load_results()

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
st.sidebar.markdown("## Navigation")
page = st.sidebar.radio("", [
    "Analytics Dashboard",
    "Model Comparison",
    "Fairness Analysis",
    "Fairness Mitigation",
    "Explainability",
    "Attrition Prediction",
])

st.sidebar.markdown("---")
st.sidebar.markdown("### Analytics Filters")
gender_filter   = st.sidebar.multiselect("Gender",    options=data["Gender"].unique(),            default=data["Gender"].unique())
joblevel_filter = st.sidebar.multiselect("Job Level", options=sorted(data["JobLevel"].unique()),  default=sorted(data["JobLevel"].unique()))
age_range       = st.sidebar.slider("Age Range", int(data["Age"].min()), int(data["Age"].max()),  (int(data["Age"].min()), int(data["Age"].max())))

filtered_data = data[
    (data["Gender"].isin(gender_filter)) &
    (data["JobLevel"].isin(joblevel_filter)) &
    (data["Age"].between(age_range[0], age_range[1]))
]

st.sidebar.markdown("---")
st.sidebar.caption("Research prototype · Responsible AI Dashboard\n\nPredictions are decision-support signals only.")

# ══════════════════════════════════════════════
# ANALYTICS DASHBOARD
# ══════════════════════════════════════════════
if page == "Analytics Dashboard":
    st.title("HR Attrition Analytics Dashboard")
    tag("Exploratory Analysis")
    st.caption("Workforce-level attrition trends. Use sidebar filters to explore specific segments.")

    col1, col2, col3, col4 = st.columns(4)
    attrition_rate = (filtered_data["Attrition"] == "Yes").mean() * 100
    col1.metric("Employees (Filtered)", len(filtered_data))
    col2.metric("Attrition Rate",       f"{attrition_rate:.1f}%")
    col3.metric("Avg Monthly Income",   f"${int(filtered_data['MonthlyIncome'].mean()):,}")
    col4.metric("Avg Years at Company", f"{filtered_data['YearsAtCompany'].mean():.1f}")

    st.markdown("---")
    col4, col5 = st.columns(2)
    with col4:
        st.subheader("Attrition by Gender")
        fig, ax = plt.subplots()
        sns.countplot(x="Gender", hue="Attrition", data=filtered_data, ax=ax)
        st.pyplot(fig); plt.close()

    with col5:
        st.subheader("Attrition by Job Level")
        fig, ax = plt.subplots()
        sns.countplot(x="JobLevel", hue="Attrition", data=filtered_data, ax=ax)
        st.pyplot(fig); plt.close()

    st.subheader("Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(filtered_data["Age"], bins=30, kde=True, ax=ax)
    st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("Insight Summary")
    if attrition_rate > 20:
        warn(f"High attrition ({attrition_rate:.1f}%) in selected group. Investigation recommended.")
    else:
        finding(f"Attrition is {attrition_rate:.1f}% — within stable range for the selected group.")
    insight("Dataset: 1,470 employees with 5.2:1 class imbalance (No:Yes). This affects model behaviour — see Model Comparison tab.")

# ══════════════════════════════════════════════
# MODEL COMPARISON
# ══════════════════════════════════════════════
elif page == "Model Comparison":
    st.title("Model Comparison")
    tag("Research Contribution 1")
    st.caption("Comparative ML evaluation: predictive performance + fairness across all trained models.")

    if "fairness" not in results:
        st.warning("Run `fairness_analysis.py` first to generate model comparison data.")
    else:
        df_res = results["fairness"]

        st.markdown("---")
        st.subheader("Performance Metrics")
        perf_cols = [c for c in ["Model", "Accuracy", "F1 Score", "AUC", "CV AUC (5-fold)"] if c in df_res.columns]
        st.dataframe(df_res[perf_cols], use_container_width=True)
        insight(
            "Logistic Regression achieves the highest accuracy (83.9%) but F1=0 — "
            "it never predicts attrition. This is the accuracy paradox in imbalanced datasets."
        )

        st.markdown("---")
        st.subheader("Fairness Metrics per Model")
        fair_cols = [c for c in ["Model", "Disparate Impact", "Stat Parity Diff", "Equal Opportunity Diff", "Fairness Verdict"] if c in df_res.columns]
        st.dataframe(df_res[fair_cols], use_container_width=True)
        warn(
            "Models that actually predict attrition (Random Forest, MLP) both show "
            "Disparate Impact > 1.25 — they over-predict attrition for females relative to males."
        )

        st.markdown("---")
        st.subheader("Visual Comparison")
        if os.path.exists("results/fairness_comparison_plot.png"):
            st.image("results/fairness_comparison_plot.png",
                     caption="Left: Performance. Centre: Disparate Impact. Right: Predicted rate by gender.",
                     use_container_width=True)

        st.markdown("---")
        st.subheader("Key Research Finding")
        finding(
            "A perfect correlation exists between AUC and fairness deviation: "
            "models with higher discriminative ability systematically introduce greater "
            "gender disparity. This quantifies the accuracy-fairness tradeoff in HR analytics."
        )

# ══════════════════════════════════════════════
# FAIRNESS ANALYSIS
# ══════════════════════════════════════════════
elif page == "Fairness Analysis":
    st.title("Fairness & Bias Analysis")
    tag("Research Contribution 2")
    st.caption("Evaluates whether attrition predictions differ systematically across gender groups.")

    use_filters  = st.checkbox("Apply sidebar filters")
    fairness_data = filtered_data.copy() if use_filters else data.copy()

    fairness_data["Attrition_num"] = fairness_data["Attrition"].map({"Yes": 1, "No": 0})
    fairness_data["Gender_num"]    = fairness_data["Gender"].map({"Male": 1, "Female": 0})

    male_rate   = fairness_data[fairness_data["Gender_num"] == 1]["Attrition_num"].mean()
    female_rate = fairness_data[fairness_data["Gender_num"] == 0]["Attrition_num"].mean()
    di          = female_rate / male_rate if male_rate > 0 else 0
    spd         = female_rate - male_rate

    if fairness_data["Gender"].nunique() < 2:
        st.info("Fairness comparison requires both gender groups. Adjust filters.")
    else:
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Male Attrition Rate",   f"{male_rate:.3f}")
        col2.metric("Female Attrition Rate", f"{female_rate:.3f}")
        col3.metric("Disparate Impact",       f"{di:.3f}")
        col4.metric("Stat Parity Diff",       f"{spd:.3f}")

        st.markdown("---")
        if di < 0.8:
            warn("DI below 0.8 — females under-represented in attrition predictions.")
        elif di > 1.25:
            warn("DI above 1.25 — males under-represented in attrition predictions.")
        else:
            finding(f"Disparate Impact ({di:.3f}) is within the 0.8–1.25 fairness threshold.")

        insight("The 80% rule (DI ≥ 0.8) is the standard from US EEOC employment guidelines. Both DI and SPD are needed for a complete fairness picture.")

        st.subheader("Attrition Distribution by Gender")
        fig, ax = plt.subplots()
        sns.countplot(x="Gender", hue="Attrition", data=fairness_data, ax=ax)
        st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════
# FAIRNESS MITIGATION
# ══════════════════════════════════════════════
elif page == "Fairness Mitigation":
    st.title("Fairness Mitigation — Reweighing")
    tag("Research Contribution 3")
    st.caption("Applies Reweighing (Kamiran & Calders, 2012) to reduce gender-based bias. Quantifies the accuracy-fairness tradeoff.")

    if "mitigation" not in results:
        st.warning("Run `fairness_mitigation.py` first.")
    else:
        df_mit   = results["mitigation"]
        baseline = df_mit.iloc[0]
        fair     = df_mit.iloc[1]

        st.markdown("---")
        st.subheader("What is Reweighing?")
        insight(
            "Reweighing assigns instance weights during training so each (gender × attrition) "
            "combination is equally represented — without changing the model architecture. "
            "It is model-agnostic, transparent, and auditable."
        )

        st.markdown("---")
        st.subheader("Before vs After")
        col1, col2, col3 = st.columns(3)
        di_delta  = round(fair["Disparate Impact"] - baseline["Disparate Impact"], 3)
        acc_delta = round(fair["Accuracy"]          - baseline["Accuracy"],         3)
        f1_delta  = round(fair["F1 Score"]          - baseline["F1 Score"],          3)
        col1.metric("Disparate Impact", f"{fair['Disparate Impact']:.3f}", delta=f"{di_delta:+.3f}", delta_color="inverse")
        col2.metric("Accuracy",         f"{fair['Accuracy']:.3f}",         delta=f"{acc_delta:+.3f}")
        col3.metric("F1 Score",         f"{fair['F1 Score']:.3f}",         delta=f"{f1_delta:+.3f}")

        st.markdown("---")
        st.subheader("Full Comparison Table")
        display_cols = [c for c in ["Model", "Accuracy", "F1 Score", "AUC", "Disparate Impact", "Stat Parity Diff", "Equal Opportunity Diff"] if c in df_mit.columns]
        st.dataframe(df_mit[display_cols], use_container_width=True)

        if os.path.exists("results/mitigation_comparison_plot.png"):
            st.image("results/mitigation_comparison_plot.png",
                     caption="Before vs after reweighing — Accuracy, F1, and Disparate Impact.",
                     use_container_width=True)

        st.markdown("---")
        st.subheader("Research Interpretation")
        warn(
            f"Reweighing did not improve Disparate Impact (DI: {baseline['Disparate Impact']:.3f} → {fair['Disparate Impact']:.3f}). "
            "This suggests gender disparity is driven by deeper feature correlations "
            "(income, job level) rather than representation imbalance — motivating future feature-level debiasing work."
        )
        finding(
            f"Accuracy cost of mitigation was minimal (Δ Accuracy = {acc_delta:+.3f}, Δ F1 = {f1_delta:+.3f}), "
            "confirming the approach is safe to apply even when it does not fully resolve the fairness issue."
        )

# ══════════════════════════════════════════════
# EXPLAINABILITY
# ══════════════════════════════════════════════
elif page == "Explainability":
    st.title("Model Explainability — SHAP")
    tag("Research Contribution 4")
    st.caption("Global feature importance using SHAP. Connects feature patterns to the fairness findings.")

    from sklearn.preprocessing import LabelEncoder

    sample_data = filtered_data[features].copy()

    if len(sample_data) < 10:
        st.warning("Not enough data for SHAP. Adjust sidebar filters.")
    else:
        le_gender = LabelEncoder()
        sample_data["Gender"] = le_gender.fit_transform(sample_data["Gender"])

        try:
            scaler      = model.named_steps["scaler"]
            clf         = model.named_steps["clf"]
            X_scaled    = scaler.transform(sample_data)
            explainer   = shap.LinearExplainer(clf, X_scaled)
            shap_values = explainer.shap_values(X_scaled)

            st.subheader("Global Feature Importance")
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, sample_data, feature_names=features, show=False)
            st.pyplot(fig); plt.close()

            st.caption("Features with larger absolute SHAP values have greater influence on attrition predictions.")
            insight(
                "If MonthlyIncome and JobLevel dominate, this explains why reweighing "
                "(which only adjusts gender × attrition representation) did not resolve fairness — "
                "the bias lives in correlated features, not gender directly."
            )

        except Exception as e:
            st.error(f"SHAP analysis failed: {e}")
            st.info("SHAP LinearExplainer requires a Logistic Regression pipeline. For Random Forest, use shap.TreeExplainer.")

# ══════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════
elif page == "Attrition Prediction":
    st.title("Employee Attrition Prediction")
    tag("Decision Support Tool")
    st.caption("Individual-level attrition risk. Probability-based output with confidence assessment.")

    st.info("Decision-support only. All HR actions require human review.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        age       = st.slider("Age", 18, 60, value=30)
        gender    = st.selectbox("Gender", ["Male", "Female"])
        education = st.selectbox("Education Level", [1, 2, 3, 4, 5],
                                 help="1=Below College · 2=College · 3=Bachelor · 4=Master · 5=Doctor")
    with col2:
        joblevel  = st.slider("Job Level", 1, 5, value=2)
        income    = st.slider("Monthly Income ($)", 1000, 30000, step=500, value=8000)
        years     = st.slider("Years at Company", 0, 40, value=3)

    input_df = pd.DataFrame({
        "Age": [age], "Gender": [1 if gender == "Male" else 0],
        "Education": [education], "JobLevel": [joblevel],
        "MonthlyIncome": [income], "YearsAtCompany": [years]
    })[features]

    st.markdown("---")
    if st.button("Predict Attrition Risk", type="primary"):
        prob       = model.predict_proba(input_df)[0][1]
        confidence = abs(prob - 0.5) * 2

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Attrition Probability", f"{prob:.1%}")
            st.progress(int(prob * 100))
            if prob > 0.5:
                st.error("High Attrition Risk")
            else:
                st.success("Low Attrition Risk")
        with col_b:
            st.metric("Model Confidence", f"{confidence:.1%}")
            if confidence < 0.3:   warn("Low confidence — interpret with caution.")
            elif confidence < 0.6: insight("Moderate confidence prediction.")
            else:                  finding("High confidence prediction.")

        st.markdown("---")
        st.subheader("Key Patterns (Correlation, not causation)")
        patterns = []
        if income   >= 15000: patterns.append("Higher income is associated with lower attrition risk.")
        if years    >= 5:     patterns.append("Longer tenure is associated with lower attrition likelihood.")
        if joblevel >= 3:     patterns.append("Higher job level is associated with lower attrition.")
        if age      < 30:     patterns.append("Early-career employees (< 30) show higher historical attrition rates.")
        for p in patterns: insight(p)
        if not patterns:
            st.caption("No strong pattern signals for this profile.")
        st.caption("Insights are derived from historical patterns and do not imply causality.")

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.caption(
    "HR Attrition — Responsible AI Dashboard · "
    "Integrates ML, fairness evaluation, bias mitigation, and explainability · "
    "For research and educational purposes only."
)