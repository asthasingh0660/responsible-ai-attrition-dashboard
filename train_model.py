import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# -----------------------
# Load data
# -----------------------
data = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# -----------------------
# Select relevant features
# -----------------------
cols = [
    "Age",
    "Gender",
    "Education",
    "JobLevel",
    "MonthlyIncome",
    "YearsAtCompany",
    "Attrition"
]
data = data[cols].copy()

# -----------------------
# Encode categorical variables
# -----------------------
le_gender = LabelEncoder()
le_attrition = LabelEncoder()

data["Gender"] = le_gender.fit_transform(data["Gender"])       # Male=1, Female=0
data["Attrition"] = le_attrition.fit_transform(data["Attrition"]) # Yes=1, No=0

# -----------------------
# Split data
# -----------------------
X = data.drop("Attrition", axis=1)
y = data["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# -----------------------
# Build pipeline (Scaling + Model)
# -----------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])

# -----------------------
# Train model
# -----------------------
pipeline.fit(X_train, y_train)

# -----------------------
# Evaluate (optional but good practice)
# -----------------------
train_acc = accuracy_score(y_train, pipeline.predict(X_train))
test_acc = accuracy_score(y_test, pipeline.predict(X_test))

print(f"Train Accuracy: {train_acc:.3f}")
print(f"Test Accuracy : {test_acc:.3f}")

# -----------------------
# Save model and feature list
# -----------------------
joblib.dump(pipeline, "attrition_model.pkl")
joblib.dump(X.columns.tolist(), "features.pkl")

print("✅ Model trained and saved successfully")

