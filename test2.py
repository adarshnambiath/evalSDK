import sys
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# =====================================================
# Import your SDK
# =====================================================

sys.path.insert(0, "/Users/adarsh/Documents/internship_2026/sdk")

from experiment_sdk import ExperimentSession

# =====================================================
# Configuration
# =====================================================

CSV_PATH = "Chocolate_Sales.csv"
OUTPUT_DIR = r"/Users/adarsh/Documents/internship_2026/platform2/research-os/workspace/runs/run_2"

print("=" * 60)
print("Research OS SDK Demo")
print("=" * 60)

# =====================================================
# Load dataset
# =====================================================

print("[1/7] Loading dataset...")

df = pd.read_csv(CSV_PATH)

print(f"Loaded {len(df):,} rows.")

# =====================================================
# Feature Engineering
# =====================================================

print("[2/7] Preparing features...")

df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")

df["Order_Year"] = df["Order_Date"].dt.year
df["Order_Month"] = df["Order_Date"].dt.month
df["Order_Day"] = df["Order_Date"].dt.day

df = df.drop(columns=["Order_Date"])

# -----------------------------------------------------
# Create classification target
# -----------------------------------------------------

def shipping_category(boxes):
    if boxes < 75:
        return "LOW"
    elif boxes < 175:
        return "MEDIUM"
    return "HIGH"

df["Shipping_Category"] = df["Boxes_Shipped"].apply(shipping_category)

# Don't leak target information

X = df.drop(columns=[
    "Shipping_Category",
    "Boxes_Shipped",
    "Amount",
])

y = df["Shipping_Category"]

# =====================================================
# Split
# =====================================================

print("[3/7] Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# =====================================================
# Preprocessing
# =====================================================

categorical_features = X.select_dtypes(
    include=["object", "string"]
).columns

numeric_features = X.select_dtypes(
    include=["number"]
).columns

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

preprocessor = ColumnTransformer([
    ("cat", categorical_pipeline, categorical_features),
    ("num", numeric_pipeline, numeric_features),
])

# =====================================================
# Model
# =====================================================

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        max_depth=8,
        random_state=42,
    )),
])

print("[4/7] Training Decision Tree...")

model.fit(X_train, y_train)

print("✓ Training complete.")

# =====================================================
# Predictions
# =====================================================

print("[5/7] Generating predictions...")

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

print(f"Generated predictions for {len(predictions)} samples.")

# =====================================================
# SDK
# =====================================================

print("[6/7] Publishing evaluation to SDK...")

session = ExperimentSession(
    output_directory=OUTPUT_DIR,
)

session.publish_evaluation(
    task="classification",
    sample_ids=X_test["Order_ID"].tolist(),
    ground_truth=y_test.tolist(),
    predictions=predictions.tolist(),
    probabilities=probabilities.tolist(),
)


session.finish()

print("✓ SDK outputs written.")

# =====================================================
# Done
# =====================================================

print("[7/7] Complete!")

print()
print("Generated files:")

for file in Path(OUTPUT_DIR).iterdir():
    print("  •", file.name)

print("\nDone.")