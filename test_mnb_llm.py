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
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# -------------------------------------------------------
# Load dataset
# -------------------------------------------------------

sales = pd.read_csv("Chocolate_Sales.csv")

# -------------------------------------------------------
# Build 3-class target
# -------------------------------------------------------

def shipping_band(value):
    if value <= 80:
        return "LOW"
    elif value <= 180:
        return "MEDIUM"
    return "HIGH"


sales["shipping_band"] = sales["Boxes_Shipped"].apply(shipping_band)

# -------------------------------------------------------
# Features
# -------------------------------------------------------

feature_table = sales.drop(
    columns=[
        "Boxes_Shipped",
        "shipping_band",
        "Amount",  # derived value
    ]
)

labels = sales["shipping_band"]
record_ids = sales["Order_ID"]

# -------------------------------------------------------
# Train/Test split
# -------------------------------------------------------

(
    train_features,
    validation_features,
    train_labels,
    validation_labels,
    train_ids,
    validation_ids,
) = train_test_split(
    feature_table,
    labels,
    record_ids,
    test_size=0.25,
    random_state=42,
    stratify=labels,
)

# -------------------------------------------------------
# Feature preprocessing
# -------------------------------------------------------

numeric_columns = feature_table.select_dtypes(include=["number"]).columns.tolist()

categorical_columns = [
    column
    for column in feature_table.columns
    if column not in numeric_columns
]

numeric_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", MinMaxScaler()),
    ]
)

categorical_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ]
)

transformer = ColumnTransformer(
    [
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ]
)

classifier = Pipeline(
    [
        ("preprocessor", transformer),
        ("model", MultinomialNB()),
    ]
)

print("=" * 60)
print("Training Multinomial Naive Bayes...")
print("=" * 60)

classifier.fit(train_features, train_labels)

print("\nTraining complete.\n")

# -------------------------------------------------------
# Inference
# -------------------------------------------------------

predicted_classes = classifier.predict(validation_features)

class_probabilities = classifier.predict_proba(validation_features)

print(classification_report(validation_labels, predicted_classes))

# -------------------------------------------------------
# Preview
# -------------------------------------------------------

inspection = pd.DataFrame(
    {
        "Order": validation_ids.values[:10],
        "Expected": validation_labels.values[:10],
        "Predicted": predicted_classes[:10],
        "Confidence": class_probabilities.max(axis=1)[:10],
    }
)

print("\nSample predictions\n")
print(inspection)

print("\nDone.")

# -------------------------------------------------------
# Experiment SDK integration
# -------------------------------------------------------

session = ExperimentSession(
    output_directory="/Users/adarsh/Documents/internship_2026/platform2/research-os/workspace/experiments/experiment_1/run_4"
)

session.publish_evaluation(
    task="classification",
    sample_ids=validation_ids.tolist(),
    ground_truth=validation_labels.tolist(),
    predictions=predicted_classes.tolist(),
    probabilities=class_probabilities.tolist(),
)

session.finish()