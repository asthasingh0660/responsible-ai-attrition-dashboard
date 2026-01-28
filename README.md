# HR Attrition Analytics and Responsible AI Decision Support Dashboard

## Live Demo

https://responsible-ai-attrition-dashboard.streamlit.app/

## Overview

Employee attrition is a critical challenge for organizations, impacting productivity, costs, and workforce stability.  
This project presents an **end-to-end HR analytics and decision support system** that combines **data analytics, machine learning, fairness evaluation, and explainability** to analyze and predict employee attrition in a responsible manner.

The system is designed not only to predict attrition risk, but also to **support informed human decision-making** by providing transparent insights, bias-aware analysis, and interpretable model outputs.

---

## Objectives

The key objectives of this project are:

- To analyze workforce attrition patterns using exploratory data analysis
- To build a predictive model for employee attrition using machine learning
- To evaluate fairness across sensitive demographic groups
- To provide explainable predictions for transparency and accountability
- To design an interactive dashboard for decision support rather than automation

---

## Dataset

The project uses the **IBM HR Analytics Employee Attrition Dataset**, which contains employee demographic, compensation, and job-related attributes.

**Key features used include:**

- Age  
- Gender  
- Education  
- Job Level  
- Monthly Income  
- Years at Company  
- Attrition (target variable)

The dataset is publicly available and widely used in HR analytics research.

**Dataset source:**  
IBM HR Analytics Employee Attrition & Performance Dataset  
https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

---

## System Architecture

The system follows a modular and interpretable architecture:

1. **Data Layer**
   - Structured HR dataset loaded and filtered dynamically
   - Interactive slicing based on demographic and role-based attributes

2. **Analytics Layer**
   - Workforce-level KPIs (attrition rate, income statistics)
   - Visual trend analysis across gender, job level, and age groups

3. **Machine Learning Layer**
   - Logistic Regression model implemented using a Scikit-learn pipeline
   - Feature scaling via StandardScaler to ensure numerical stability
   - Probabilistic predictions for attrition risk assessment

4. **Fairness Evaluation Layer**
   - Disparate Impact analysis
   - Statistical Parity Difference computation
   - Scenario-based fairness evaluation using user-selected filters
   - Explicit handling of undefined fairness scenarios

5. **Explainability Layer**
   - SHAP-based global feature importance analysis
   - Transparent interpretation of model behavior

6. **Presentation Layer**
   - Interactive dashboard built using Streamlit
   - Clear separation between analytics, fairness, explainability, and prediction views

---

## Key Features

### Analytics Dashboard

- Interactive filters for gender, job level, and age range
- Dynamic KPIs reflecting filtered workforce segments
- Visual analysis of attrition trends and distributions
- Insight summaries to support data-driven interpretation

### Fairness and Bias Analysis

- Measurement of gender-based attrition differences
- Disparate Impact and Statistical Parity metrics
- Filter-aware fairness evaluation
- Guardrails for scenarios where fairness comparison is not statistically valid

### Attrition Prediction

- Individual-level attrition risk estimation
- Probability-based output rather than binary automation
- Model confidence assessment
- Human-readable explanatory insights

### Explainability

- Global model interpretability using SHAP values
- Identification of features that increase or reduce attrition risk
- Transparent connection between data patterns and model behavior

---

## Responsible AI Considerations

This project is explicitly designed as a **decision-support system**, not an automated decision-maker.

Key responsible AI practices implemented include:

- Human-in-the-loop design with clear disclaimers
- Fairness evaluation across sensitive attributes
- Honest handling of undefined or insufficient data scenarios
- Explainability to improve transparency and trust
- Confidence-aware predictions to avoid over-reliance on model outputs

---

## Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Matplotlib, Seaborn
- SHAP

---

## How to Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/responsible-ai-attrition-dashboard.git
   cd responsible-ai-attrition-dashboard
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Train the model:

   ```bash
   python train_model.py
   ```

5. Run the dashboard:

   ```bash
   streamlit run app.py
   ```

---

## Research and Academic Relevance

This project can be used as a foundation for research work in:

- HR Analytics
- Responsible AI and Fairness in Machine Learning
- Explainable AI (XAI)
- Decision Support Systems
- Applied Data Analytics

The modular design allows extension toward longitudinal bias monitoring, alternative fairness definitions, or comparative model evaluation.

---

## Author

Astha Singh

---

## Disclaimer

This system is intended for educational and research purposes only.
Predictions and insights should not be used as the sole basis for real-world HR decisions.
