import sys
sys.path.insert(0, '/Users/adarsh/Documents/internship_2026/sdk')

from experiment_sdk import ExperimentSession

# Create a session
session = ExperimentSession(output_directory="./results")

# Publish a classification evaluation
session.publish_evaluation(
    task="classification",
    sample_ids=["img_001", "img_002", "img_003"],
    ground_truth=[0, 1, 0],
    predictions=[0, 1, 1],
    probabilities=[
        [0.95, 0.05],
        [0.10, 0.90],
        [0.40, 0.60],
    ],
)

# Register artifacts
session.publish_artifact(
    type="MODEL_CHECKPOINT",
    path="./models/best_model.pt",
    name="Best Model",
)

# Finalize and write outputs
session.finish()

# Outputs created:
# - results/evaluation.parquet
# - results/metrics.json
# - results/artifacts.json