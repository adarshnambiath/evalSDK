# experiment_sdk

A lightweight, platform-agnostic Python package for validating, standardizing, serializing, and registering machine learning experiment outputs.

## 1. Overview

The Experiment SDK standardizes how ML experiments produce outputs. It exists solely to validate, standardize, serialize, and register experiment outputs so downstream consumers can process them consistently.

**What the SDK is**:
- A standalone Python package that validates, standardizes, serializes, and registers ML experiment outputs.
- It begins **after predictions already exist**.

**What the SDK is NOT**:
- It does not train models.
- It does not preprocess data.
- It does not perform inference.
- It does not replace ML frameworks.
- It has no knowledge of any backend, frontend, database, or research platform.

### Philosophy

- **No training** — the SDK begins after predictions already exist.
- **No inference** — it never runs a model.
- **No framework coupling** — it works with any ML framework.
- **No platform assumptions** — it writes files to disk and nothing else.

Its only job is to validate, standardize, serialize, and register experiment outputs.

## 2. Current Installation

The current development version is not available via `pip`. It is used by manually adding the SDK directory to `sys.path`:

```python
import sys

sys.path.insert(
    0,
    "/path/to/evalsdk"
)

from experiment_sdk import ExperimentSession
```

`pip install experiment-sdk` is planned in a future release.

## 3. Quick Start

```python
import sys

sys.path.insert(
    0,
    "/path/to/evalsdk"
)

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

## 4. Public API

### ExperimentSession

Primary entry point for an experiment run.

```python
session = ExperimentSession(output_directory="./results")
```

### publish_evaluation()

Validate inputs, compute metrics, and store an evaluation.

```python
session.publish_evaluation(
    task: str,
    sample_ids: list,
    ground_truth: list,
    predictions: list,
    probabilities: Optional[list] = None,
    metadata: Optional[Dict[str, Any]] = None,
    columns: Optional[Dict[str, list]] = None,
) -> Evaluation
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `task` | `str` | Yes | `"classification"` or `"regression"` |
| `sample_ids` | `list` | Yes | Unique identifiers for each sample |
| `ground_truth` | `list` | Yes | True labels or target values |
| `predictions` | `list` | Yes | Predicted labels or values |
| `probabilities` | `Optional[list]` | No | Predicted class probabilities (classification) |
| `metadata` | `Optional[Dict[str, Any]]` | No | Arbitrary metadata dictionary for the evaluation |
| `columns` | `Optional[Dict[str, list]]` | No | Arbitrary per-sample columns |

### publish_artifact()

Register an artifact metadata record without copying or uploading the file.

```python
session.publish_artifact(
    type: str,
    path: str,
    name: Optional[str] = None,
) -> None
```

### finish()

Serialize all registered content to disk.

```python
session.finish(confirm: bool = True) -> None
```

By default, prints a preview and waits for confirmation before writing files. Pass `confirm=False` to write immediately.

## 5. Classification Example

```python
import sys

sys.path.insert(
    0,
    "/path/to/evalsdk"
)

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

The evaluation table (`evaluation.parquet`) will include:
- `sample_id`
- `ground_truth`
- `prediction`
- `correct`
- `confidence` (maximum predicted class probability)

Metrics are automatically computed and saved to `metrics.json`:
- Accuracy
- Precision
- Recall
- F1
- Confusion Matrix
- `average_confidence` (when probabilities are provided)

## 6. Regression Example

```python
import sys

sys.path.insert(
    0,
    "/path/to/evalsdk"
)

from experiment_sdk import ExperimentSession

session = ExperimentSession("./output")

session.publish_evaluation(
    task="regression",
    sample_ids=["house_1", "house_2", "house_3"],
    ground_truth=[300000.0, 450000.0, 200000.0],
    predictions=[310000.0, 440000.0, 210000.0],
)

session.finish()
```

The evaluation table (`evaluation.parquet`) will include:
- `sample_id`
- `ground_truth`
- `prediction`
- `residual`
- `absolute_error`

Metrics are automatically computed and saved to `metrics.json`:
- MAE
- RMSE
- R²

## 7. Per-Sample Columns

You can attach **arbitrary** per-sample columns to any evaluation. The SDK validates, stores, and serializes them without interpretation.

### Arbitrary columns

Pass a `columns` dictionary mapping a column name to a list of per-sample values:

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

### Validation

- Each value must be list-like and have the same length as `sample_ids`.
- Column names must **not** collide with SDK-owned reserved names (see below).

### Reserved names

The following names are reserved by the SDK and must not be used as column names:
- `sample_id`
- `ground_truth`
- `prediction`
- `confidence`
- `correct`
- `residual`
- `absolute_error`
- `evaluation_index`

### Storage

Per-sample columns are written into `evaluation.parquet` alongside the standard columns. They are stored exactly as provided.

### Interpretation

The SDK never interprets these values. They exist purely for downstream investigation, visualization, or provenance.

### Domain examples

| Domain | Example columns |
|--------|-----------------|
| ECG | `window_start`, `window_end`, `record_name` |
| Images | `bounding_box`, `image_path` |
| NLP | `token_span`, `document_id` |
| Audio | `timestamp`, `segment_id` |
| Time Series | `window_start`, `window_end` |
| Medical Imaging | `study_id`, `series_uid` |
| Industrial Sensors | `sensor_id`, `machine_id` |

## 8. WFDB Provenance Example

When working with WFDB ECG data, per-sample columns can carry window provenance:

```python
session.publish_evaluation(
    task="classification",
    sample_ids=sample_ids,
    ground_truth=labels,
    predictions=preds,
    columns={
        "record_name": ["100", "100", "101", ...],
        "window_start": [0, 180, 360, ...],
        "window_end": [180, 360, 540, ...],
    },
)
```

- **record_name** — the WFDB record identifier.
- **window_start** — sample index where the window begins.
- **window_end** — sample index where the window ends.

These fields are **optional**. They should only be published if they are already available from your preprocessing pipeline. The SDK never infers provenance.

## 9. Artifact Registration

Register artifacts by path. The SDK records metadata only; it does not move, copy, or upload files.

```python
session.publish_artifact(
    type="MODEL_CHECKPOINT",
    path="./outputs/model.pt",
    name="best_model",
)
```

Supported artifact types include:
- `MODEL_CHECKPOINT`
- `CONFIG`
- `LOGS`
- `REPORT`
- `VISUALIZATION`
- `PREDICTIONS`

Artifacts are written to `artifacts.json`.

## 10. Output Files

After calling `session.finish()`, exactly three files are written to the output directory:

| File | Format | Description |
|------|--------|-------------|
| `evaluation.parquet` | Parquet | Per-sample predictions, labels, derived values, and any attached columns |
| `metrics.json` | JSON | Computed evaluation metrics |
| `artifacts.json` | JSON | Registered artifact metadata |

## 11. Philosophy

- **No training** — the SDK begins after predictions already exist.
- **No inference** — it never runs a model.
- **No framework coupling** — it works with any ML framework.
- **No platform assumptions** — it writes files to disk and nothing else.

Its only job is to validate, standardize, serialize, and register experiment outputs so that downstream consumers can process them consistently.
