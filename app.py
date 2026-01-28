import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np

# -----------------------
# Page config
# -----------------------
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12

st.set_page_config(
    page_title="HR Attrition Analytics Dashboard",
    layout="wide"
)

# -----------------------
# Helper function
# -----------------------
def section_help(text):
    st.caption(f"ℹ️ {text}")

# -----------------------
# Load data & model
# -----------------------
data = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

model = joblib.load("attrition_model.pkl")
features = joblib.load("features.pkl")

# -----------------------
# Sidebar navigation
# -----------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["📊 Analytics Dashboard", "⚖️ Fairness View", "🧠 Explainability", "🤖 Attrition Prediction"]
)

# -----------------------
# Global Analytics Filters
# -----------------------
st.sidebar.subheader("Analytics Filters")

gender_filter = st.sidebar.multiselect(
    "Select Gender",
    options=data["Gender"].unique(),
    default=data["Gender"].unique()
)

joblevel_filter = st.sidebar.multiselect(
    "Select Job Level",
    options=sorted(data["JobLevel"].unique()),
    default=sorted(data["JobLevel"].unique())
)

age_range = st.sidebar.slider(
    "Select Age Range",
    min_value=int(data["Age"].min()),
    max_value=int(data["Age"].max()),
    value=(int(data["Age"].min()), int(data["Age"].max()))
)

# ALWAYS define filtered_data
filtered_data = data[
    (data["Gender"].isin(gender_filter)) &
    (data["JobLevel"].isin(joblevel_filter)) &
    (data["Age"].between(age_range[0], age_range[1]))
]

# =======================
# 📊 ANALYTICS DASHBOARD
# =======================
if page == "📊 Analytics Dashboard":
    st.title("📊 HR Attrition Analytics Dashboard")
    section_help(
        "This dashboard provides exploratory analysis of employee attrition trends "
        "across demographics, roles, and experience levels."
    )

    # -----------------------
    # KPIs
    # -----------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Employees (Filtered)", len(filtered_data))

    with col2:
        attrition_rate = (filtered_data["Attrition"] == "Yes").mean() * 100
        st.metric("Attrition Rate", f"{attrition_rate:.2f}%")

    with col3:
        st.metric(
            "Avg Monthly Income",
            f"{int(filtered_data['MonthlyIncome'].mean())}"
        )

    st.markdown("---")

    # -----------------------
    # Charts
    # -----------------------
    col4, col5 = st.columns(2)

    with col4:
        st.subheader("Attrition by Gender")
        fig, ax = plt.subplots()
        sns.countplot(x="Gender", hue="Attrition", data=filtered_data, ax=ax)
        st.pyplot(fig)

    with col5:
        st.subheader("Attrition by Job Level")
        fig, ax = plt.subplots()
        sns.countplot(x="JobLevel", hue="Attrition", data=filtered_data, ax=ax)
        st.pyplot(fig)

    st.subheader("Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(filtered_data["Age"], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

    # -----------------------
    # Insight Summary
    # -----------------------
    st.subheader("📌 Key Insight Summary")

    if attrition_rate > 20:
        st.warning(
            "High attrition observed in the selected group. "
            "Further investigation into compensation, role satisfaction, "
            "or career growth may be required."
        )
    else:
        st.success(
            "Attrition levels appear stable for the selected group."
        )

# =======================
# ⚖️ FAIRNESS VIEW
# =======================
elif page == "⚖️ Fairness View":
    st.title("⚖️ Fairness & Bias Analysis")
    section_help(
        "Fairness metrics evaluate whether attrition outcomes differ systematically "
        "across sensitive demographic groups. Metrics may be undefined when comparison "
        "groups are missing."
    )

    st.write(
        "This section evaluates whether attrition risk "
        "differs across sensitive demographic groups."
    )

    use_filters = st.checkbox("Use analytics filters")

    fairness_data = data.copy()   # ALWAYS defined

    if use_filters:
        fairness_data = filtered_data.copy()
    

    fairness_data["Attrition_num"] = fairness_data["Attrition"].map({"Yes": 1, "No": 0})
    fairness_data["Gender_num"] = fairness_data["Gender"].map({"Male": 1, "Female": 0})

    male_rate = fairness_data[fairness_data["Gender_num"] == 1]["Attrition_num"].mean()
    female_rate = fairness_data[fairness_data["Gender_num"] == 0]["Attrition_num"].mean()

    disparate_impact = female_rate / male_rate if male_rate > 0 else 0
    statistical_parity = female_rate - male_rate

    if fairness_data["Gender"].nunique() < 2:
        st.info(
            "Fairness comparison requires at least two demographic groups. "
            "Adjust filters to include multiple genders."
        )

    if fairness_data["Gender"].nunique() >= 2:
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Male Attrition Rate", f"{male_rate:.2f}")
        col2.metric("Female Attrition Rate", f"{female_rate:.2f}")
        col3.metric("Disparate Impact", f"{disparate_impact:.2f}")
        col4.metric("Statistical Parity Diff", f"{statistical_parity:.2f}")

        if disparate_impact < 0.8:
            st.warning(
                "⚠️ Potential fairness risk detected. "
                "Attrition impact differs significantly across gender groups."
            )
        elif disparate_impact > 1.25:
            st.warning(
                "⚠️ Reverse disparity detected. "
                "Privileged group may be adversely affected."
            )
        else:
            st.success("✅ Attrition impact appears balanced across gender groups.")

    st.subheader("Attrition Distribution by Gender")
    fig, ax = plt.subplots()
    sns.countplot(x="Gender", hue="Attrition", data=data, ax=ax)
    st.pyplot(fig)

# =======================
# 🤖 ATTRITION PREDICTION
# =======================
else:
    st.title("🤖 Employee Attrition Prediction")
    section_help(
        "This prediction is a decision-support signal, not an automated decision. "
        "Human judgment is required for any HR action."
    )

    st.info(
        "⚠️ This system is a decision-support tool. "
        "Final HR decisions should involve human review."
    )

    age = st.slider("Age", 18, 60, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    education = st.selectbox("Education Level", [1, 2, 3, 4, 5])
    joblevel = st.slider("Job Level", 1, 5, value=2)
    income = st.slider("Monthly Income", 1000, 30000, step=500, value=8000)
    years = st.slider("Years At Company", 0, 40, value=3)

    gender_val = 1 if gender == "Male" else 0

    input_df = pd.DataFrame({
        "Age": [age],
        "Gender": [gender_val],
        "Education": [education],
        "JobLevel": [joblevel],
        "MonthlyIncome": [income],
        "YearsAtCompany": [years]
    })

    input_df = input_df[features]

    if st.button("Predict Attrition"):
        prob = model.predict_proba(input_df)[0][1]
        st.progress(int(prob * 100))

        if prob > 0.5:
            st.error(f"High Attrition Risk (Probability: {prob:.2f})")
        else:
            st.success(f"Low Attrition Risk (Probability: {prob:.2f})")

        confidence = abs(prob - 0.5) * 2

        if confidence < 0.3:
            st.warning("⚠️ Low confidence prediction. Interpret with caution.")
        elif confidence < 0.6:
            st.info("ℹ️ Moderate confidence prediction.")
        else:
            st.success("✅ High confidence prediction.")

        st.caption(f"Model confidence score: {confidence:.2f}")

        st.subheader("Key Insights (Correlation, not causation)")
        if income >= 15000:
            st.write("• Higher income reduces attrition risk")
        if years >= 5:
            st.write("• Longer tenure reduces attrition likelihood")
        if joblevel >= 3:
            st.write("• Higher job level is associated with lower attrition")
        if age < 30:
            st.write("• Early-career employees show higher attrition trends")

        st.caption(
            "Insights are derived from historical patterns and do not imply causality. "
            "External factors may influence attrition decisions."
        )

# -----------------------
# Footer
# -----------------------
st.markdown("---")
# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.caption(
    "This dashboard integrates data analytics, machine learning, "
    "and responsible AI principles for HR decision support."
)

# Ensure the page navigation logic is correctly structured
if page == "🧠 Explainability":
    st.title("🧠 Model Explainability")

    st.write(
        "This section explains which features most influence "
        "employee attrition predictions."
    )

    import shap
    import numpy as np

    # Sample data for explanation
    sample_data = filtered_data[features].copy()

    if len(sample_data) < 10:
        st.warning("Not enough data to generate explanations.")
    else:
        # Encode categorical variables (Gender) as done in training
        from sklearn.preprocessing import LabelEncoder
        le_gender = LabelEncoder()
        sample_data["Gender"] = le_gender.fit_transform(sample_data["Gender"])
        
        # Extract model components
        scaler = model.named_steps["scaler"]
        clf = model.named_steps["clf"]

        X_scaled = scaler.transform(sample_data)

        explainer = shap.LinearExplainer(clf, X_scaled)
        shap_values = explainer.shap_values(X_scaled)

        st.subheader("Global Feature Importance")
        fig, ax = plt.subplots()
        shap.summary_plot(
            shap_values,
            sample_data,
            feature_names=features,
            show=False
        )
        st.pyplot(fig)

        st.caption(
            "Features with larger absolute SHAP values have "
            "greater influence on attrition predictions."
        )

elif page == "🧠 Explainability":
    st.title("🧠 Model Explainability")

    st.write(
        "This section explains which features most influence "
        "employee attrition predictions."
    )

    # Sample data for explanation
    sample_data = filtered_data[features].copy()

    if len(sample_data) < 10:
        st.warning("Not enough data to generate explanations.")
    else:
        # Extract model components
        scaler = model.named_steps["scaler"]
        clf = model.named_steps["clf"]

        X_scaled = scaler.transform(sample_data)

        explainer = shap.LinearExplainer(clf, X_scaled)
        shap_values = explainer.shap_values(X_scaled)

        st.subheader("Global Feature Importance")
        fig, ax = plt.subplots()
        shap.summary_plot(
            shap_values,
            sample_data,
            feature_names=features,
            show=False
        )
        st.pyplot(fig)

        st.caption(
            "Features with larger absolute SHAP values have "
            "greater influence on attrition predictions."
        )
