import sys
sys.path.insert(
    0,
    "/Users/adarsh/Documents/internship_2026/sdk"
)
from experiment_sdk import ExperimentSession

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
# -------------------------------------------------------
# Load dataset
# -------------------------------------------------------
df = pd.read_csv("Chocolate_Sales.csv")
# -------------------------------------------------------
# Create a 3-class target from Boxes_Shipped
#
# LOW    : <= 80
# MEDIUM : 81 - 180
# HIGH   : > 180
# -------------------------------------------------------
def shipment_class(x):
    if x <= 80:
        return "LOW"
    elif x <= 180:
        return "MEDIUM"
    return "HIGH"
df["shipment_class"] = df["Boxes_Shipped"].apply(shipment_class)
# -------------------------------------------------------
# Features
#
# We deliberately exclude:
#   - Boxes_Shipped (target source)
#   - shipment_class (target)
#   - Amount (derived variable)
# -------------------------------------------------------
X = df.drop(
    columns=[
        "Boxes_Shipped",
        "shipment_class",
        "Amount",
    ]
)
y = df["shipment_class"]
sample_ids = df["Order_ID"]
# -------------------------------------------------------
# Train/Test split
# -------------------------------------------------------
X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X,
    y,
    sample_ids,
    test_size=0.25,
    random_state=42,
    stratify=y,
)
# -------------------------------------------------------
# Feature preprocessing
# -------------------------------------------------------
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = [
    c for c in X.columns if c not in numeric_features
]
numeric_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
    ]
)
categorical_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)
preprocessor = ColumnTransformer(
    [
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ]
)
# -------------------------------------------------------
# Model
# -------------------------------------------------------
model = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "classifier",
            DecisionTreeClassifier(
                max_depth=6,
                random_state=42,
            ),
        ),
    ]
)
print("Training Decision Tree...")
model.fit(X_train, y_train)
print("Training complete.\n")
# -------------------------------------------------------
# Predictions
# -------------------------------------------------------
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
# -------------------------------------------------------
# Report
# -------------------------------------------------------
print(classification_report(y_test, predictions))
print("Finished.")

# -------------------------------------------------------
# Experiment SDK integration
# -------------------------------------------------------
session = ExperimentSession(
    output_directory="/Users/adarsh/Documents/internship_2026/platform2/research-os/workspace/experiments/experiment_1/run_3"
)

session.publish_evaluation(
    task="classification",
    sample_ids=ids_test.tolist(),
    ground_truth=y_test.tolist(),
    predictions=predictions.tolist(),
    probabilities=probabilities.tolist(),
)

session.finish()