# Experiment SDK Integration Guide

This document is intended for **developers, researchers, coding agents, and LLM assistants** (Cursor, Claude Code, Cline, Copilot, ChatGPT, etc.). It teaches how to integrate the Experiment SDK into an existing ML training or inference script.

## What the SDK Does

The SDK begins **after predictions already exist**. Its sole purpose is to validate, standardize, serialize, and register experiment outputs.

It does **not**:
- train models
- preprocess data
- perform inference
- replace ML frameworks (PyTorch, TensorFlow, scikit-learn, etc.)
- require changes to model architecture, loss functions, or training loops

It **does**:
- validate output arrays (lengths, uniqueness, types)
- compute standard metrics for classification and regression
- serialize per-sample evaluation tables to Parquet
- register artifact metadata to JSON
- enforce a structured, portable output schema

## Current Integration Method

The current development version of the SDK is not pip-installable. Import it by adding the SDK directory to `sys.path`:

```python
import sys

sys.path.insert(
    0,
    "/absolute/path/to/evalsdk"
)

from experiment_sdk import ExperimentSession
```

This is a temporary development pattern. A proper pip package is planned for the future.

## Where It Fits

Place the SDK at the **end** of your ML pipeline, immediately after predictions are produced.

```
Dataset
    ↓
Preprocessing
    ↓
Feature Engineering
    ↓
Training
    ↓
Inference
    ↓
Predictions
    ↓
Experiment SDK
    ↓
evaluation.parquet
metrics.json
artifacts.json
```

The SDK should be the **last step** before results are archived or transferred.

## Step-by-Step Integration

1. **Locate where predictions are produced.**
   Find the Python variable holding your model's outputs. Common names: `y_pred`, `preds`, `predictions`, `logits`.

2. **Create one `ExperimentSession` instance.**
   Do not create more than one session per experiment run unless running multiple independent evaluations.
   ```python
   session = ExperimentSession(output_directory="./experiment_outputs")
   ```

3. **Reuse existing variables.**
   Do not rename or reshape your existing tensors/lists unless necessary. The SDK accepts Python lists and NumPy arrays.
   ```python
   session.publish_evaluation(
       task="classification",
       sample_ids=sample_ids,
       ground_truth=y_true,
       predictions=y_pred,
   )
   ```

4. **Publish an evaluation.**
   Call `publish_evaluation()` at minimum once (or once per intended evaluation).

5. **Publish artifacts if they already exist.**
   If you have a checkpoint, config file, or report on disk, register it.
   ```python
   session.publish_artifact(
       type="MODEL_CHECKPOINT",
       path="./runs/run_42/model.pt",
   )
   ```
   Do not invent artifacts.

6. **Finish exactly once.**
   ```python
   session.finish()
   ```
   This writes the three output files.

## Required Inputs

The following arguments are **required** for every evaluation:

- `sample_ids` — unique string or numeric identifiers for each sample
- `ground_truth` — true labels or target values
- `predictions` — model outputs (class labels for classification, continuous values for regression)

### Obtaining `sample_ids`

If your dataset does not already have unique identifiers, generate them before calling `publish_evaluation()`. Ensure they are unique across the evaluation set.

```python
sample_ids = [f"Sample_{i:04d}" for i in range(len(y_true))]
```

## Optional Inputs

| Input | When to use | When to omit | Effect if omitted |
|-------|-------------|--------------|-------------------|
| `probabilities` | Classification tasks where the model outputs class probabilities or logits that have been converted to probabilities. | Regression tasks, or classification models that only output class labels. | `confidence` is set to `NaN`. `average_confidence` metric is excluded from `metrics.json`. |
| `metadata` | You want to attach arbitrary key-value metadata to an evaluation record (e.g., dataset version, model commit hash). | You have no evaluation-level metadata to record. | No effect on serialized files. |
| `columns` | You have domain-specific per-sample metadata (e.g., ECG window timing, image paths, audio timestamps). | You do not have extra per-sample data, or you do not need it for downstream analysis. | Only the standard columns are written to `evaluation.parquet`. |
| `artifacts` (via `publish_artifact`) | Files already exist on disk that should be linked to this experiment (checkpoints, configs, logs, reports, visualizations). | No additional files exist, or you cannot locate a file path. | `artifacts.json` will contain an empty list `[]`. |

## Per-Sample Columns

Arbitrary dictionaries of list-like values may be attached to any evaluation via `columns`.

### Domain examples

| Domain | Column example | Why useful |
|--------|----------------|--------------|
| ECG | `record_name`, `window_start`, `window_end` | Trace predictions back to raw signal segments |
| Images | `image_path`, `bounding_box` | Open raw images or link detection outputs |
| Audio | `timestamp`, `segment_id` | Map predictions to temporal spans |
| NLP | `document_id`, `token_span` | Reconstruct passages or phrases |
| Time Series | `window_start`, `window_end`, `source_sensor` | Re-run analysis on specific windows |
| Medical Imaging | `study_uid`, `series_uid` | Link to PACS or DICOM origins |
| Industrial Sensors | `machine_id`, `measurement_id` | Correlate failures with equipment units |

### Storage and behavior

- Columns are stored **exactly as provided** in `evaluation.parquet`.
- `pandas` handles type inference when reading Parquet back.
- The SDK **never interprets** column values for metrics computation.
- Metrics computation uses only `sample_ids`, `ground_truth`, `predictions`, and optionally `probabilities`.

### When to omit

If a column is expensive to compute, redundant with `sample_ids`, or not needed for downstream reporting, omit it. The output files will remain valid.

## WFDB Example

ECG pipelines that extract beats or fixed-length windows often already know the record name and window boundaries. Attach them as columns if they exist.

```python
# Assume your beat extraction loop produces these alongside predictions:
record_names = ["100", "100", "100", "101", "101"]
window_starts = [0, 180, 360, 0, 180]   # sample indices
window_ends = [180, 360, 540, 180, 360]

session.publish_evaluation(
    task="classification",
    sample_ids=sample_ids,
    ground_truth=true_labels,
    predictions=predicted_labels,
    columns={
        "record_name": record_names,
        "window_start": window_starts,
        "window_end": window_ends,
    },
)
```

- `record_name` — the WFDB record identifier.
- `window_start` — sample index where the window begins.
- `window_end` — sample index where the window ends.

These values are **optional**. They should **only** be published if your preprocessing pipeline already computes them. The SDK **never infers** provenance. Preprocessing should **never** be modified solely to satisfy the SDK.

## Artifact Registration

Use `publish_artifact()` only for files that **already exist** on disk at the time of integration.

```python
session.publish_artifact(
    type="CONFIG",
    path="configs/run_42.yaml",
    name="training_config",
)
```

**Behavior:**
- The SDK stores `{type, path, name, timestamp}` in `artifacts.json`.
- The file at `path` is **not copied, moved, or uploaded**.
- The SDK does **not** verify that the file exists at `path`.

**When to omit:**
- The file has not been written yet.
- The file is transient and will be cleaned up.
- You only need the evaluation numbers, not a provenance trail.

## Common Mistakes

| Mistake | What happens | Fix |
|---------|--------------|-----|
| Modifying preprocessing | Introduces SDLC risk and potential bugs. | Let the SDK read existing outputs only. |
| Inventing probabilities | Metrics become misleading (e.g., inflated `average_confidence`). | Omit `probabilities` if not available. |
| Inventing provenance columns | `window_start` / `window_end` may not match the true data window. | Only include provenance that is produced by your pipeline. |
| Calling `finish()` twice | Writes files twice (or prompts twice). | Call `finish()` **exactly once** at the end. |
| Creating fake artifact paths | `artifacts.json` points to non-existent files. | Register only real, stable files. |
| Changing model code for the SDK | Unnecessary coupling. | The SDK does not import model code. |
| Using reserved column names in `columns` | `ReservedColumnError` is raised. | Rename or omit the conflicting column. |

## Guidance for Coding Agents

When integrating the Experiment SDK into a user's codebase, follow these rules:

1. **Preserve existing code.** Do not rename variables, refactor training loops, or restructure inference blocks.
2. **Preserve preprocessing.** Do not alter dataset loading, augmentation, or feature engineering unless the user explicitly asks.
3. **Preserve inference.** Do not change how the model is called or how logits are processed.
4. **Preserve model architecture.** Do not modify layer definitions, forward passes, or initialization.
5. **Preserve hyperparameters.** Do not touch optimizer settings, schedulers, or training configs.

**Your job is to connect existing outputs to the SDK.**

### Integration protocol for agents

- Search for variables named `y_pred`, `preds`, `predictions`, `y_true`, `labels`, `ground_truth`.
- Insert the `sys.path` snippet at the top of the file if the SDK is not already importable.
- Add `ExperimentSession` initialization near the end of the script, after predictions are computed.
- Reuse existing lists/arrays directly as arguments to `publish_evaluation()`.
- If the variable exists, pass it to `columns`, `probabilities`, `metadata`, or `publish_artifact()`.
- If the variable does not exist, **omit the argument**.
- **Never fabricate data.** Missing probabilities, missing provenance, and missing artifact paths should all lead to omission, not invention.
- After adding the SDK calls, ensure `finish()` is called exactly once.
- Do not run the script unless asked; just provide the minimal diff.

## Evolving This Document

This structure is intended to be extensible. Future task types (segmentation, object detection, forecasting, survival analysis) and features (multi-session runners, remote storage backends) can be appended without reorganizing the entire file.
