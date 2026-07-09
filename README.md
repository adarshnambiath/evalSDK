# experiment_sdk

A lightweight, platform-agnostic Python package for standardizing machine learning experiment outputs.

The SDK is **completely standalone**. It has no knowledge of any backend, frontend, database, or research platform. It exists solely to validate, standardize, serialize, and register experiment outputs.

## Installation

```bash
pip install experiment-sdk
```

## Quick Start

```python
from experiment_sdk import ExperimentSession

session = ExperimentSession(output_directory="./results")

session.publish_evaluation(
    task="classification",
    sample_ids=sample_ids,
    ground_truth=y_true,
    predictions=y_pred,
    probabilities=probabilities,
)

session.publish_artifact(
    type="MODEL_CHECKPOINT",
    path="checkpoint.pt",
)

session.finish()
```


## Per-Sample Columns

You can attach arbitrary per-sample columns to any evaluation. The SDK
validates, stores, and serializes them without interpretation.

```python
session.publish_evaluation(
    task="classification",
    sample_ids=sample_ids,
    ground_truth=ground_truth,
    predictions=predictions,
    columns={
        "window_start": [0, 10, 20, ...],
        "window_end":   [10, 20, 30, ...],
        "record_name":  ["rec_1", "rec_2", "rec_3", ...],
    },
)
```

**Requirements:**

- Each column value must be a list (or list-like) with the same length as
  ``sample_ids``.
- Column names must **not** collide with SDK-owned columns:
  ``sample_id``, ``ground_truth``, ``prediction``, ``confidence``,
  ``correct``, ``residual``, ``absolute_error``, ``evaluation_index``.
- Any other name is allowed.

These columns are written into ``evaluation.parquet`` automatically.

They are **ignored** by metrics computation and do not appear in
``metrics.json`` or ``artifacts.json``.

This feature is domain agnostic. Examples of use:

| Domain | Example columns |
|--------|----------------|
| ECG | ``window_start``, ``window_end``, ``record_name`` |
| Images | ``bounding_box``, ``image_path`` |
| NLP | ``token_span``, ``document_id`` |
| Audio | ``timestamp``, ``segment_id`` |

The SDK never interprets column values. They exist purely for downstream
investigation, visualization, or provenance. They are stored as-is in
the evaluation parquet table.

## Classification Example

```python
from experiment_sdk import ExperimentSession

session = ExperimentSession("./output")

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

session.finish()
```

The evaluation table will include:
- `sample_id`
- `ground_truth`
- `prediction`
- `confidence` (predicted class confidence when probabilities are provided)
- `correct`

Metrics are automatically computed and saved to `metrics.json`:
- Accuracy
- Precision
- Recall
- F1
- Confusion Matrix

## Regression Example

```python
session.publish_evaluation(
    task="regression",
    sample_ids=["house_1", "house_2", "house_3"],
    ground_truth=[300000.0, 450000.0, 200000.0],
    predictions=[310000.0, 440000.0, 210000.0],
)
```

The evaluation table will include:
- `sample_id`
- `ground_truth`
- `prediction`
- `residual`
- `absolute_error`

Metrics are automatically computed:
- MAE
- RMSE
- R²

## Output Files

After calling `session.finish()`, the SDK writes exactly three files to the chosen output directory:

| File | Format | Description |
|------|--------|-------------|
| `evaluation.parquet` | Parquet | Per-sample predictions, labels, and derived values |
| `metrics.json` | JSON | Computed evaluation metrics |
| `artifacts.json` | JSON | Registered artifact metadata |

## Public API

| Symbol | Description |
|--------|-------------|
| `ExperimentSession` | Primary entry point for an experiment run |
| `ExperimentSession(output_directory)` | Initialize a session targeting a directory |
| `session.publish_evaluation(...)` | Validate and register an evaluation |
| `session.publish_artifact(...)` | Register an artifact metadata record |
| `session.finish()` | Serialize all registered content to disk |

Everything else is **internal** and subject to change.

## Future Extensions

Additional task types (segmentation, object detection, forecasting, ranking, survival analysis) will fit into the same architecture without changing the public API.

## Philosophy

- **No training**
- **No inference**
- **No framework coupling**
- **No platform assumptions**

This package begins **after predictions already exist**. Its only job is to validate, standardize, serialize, and register experiment outputs so that downstream consumers can process them consistently.
