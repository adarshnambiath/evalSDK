# Architecture

This document describes the long-term architecture and design decisions of the experiment SDK.

## 1. Purpose of the SDK

The SDK is NOT responsible for:

- Training models
- Running inference
- Experiment tracking
- Visualization

Its responsibility begins **after predictions already exist**. Its job is to:

1. **Validate** experiment outputs
2. **Standardize** them into known evaluation objects
3. **Serialize** them to disk
4. **Register** artifacts
5. Nothing more

## 2. Core Philosophy

The SDK should never depend on any platform or service.

It should produce standardized evaluation artifacts that any platform can consume. Today those artifacts are written to disk. In the future they may be published elsewhere. The SDK itself should remain platform-agnostic.

Key principles:

- **No heavy dependencies**: only pandas, numpy, pyarrow, scikit-learn.
- **No implicit mutations**: user data is never modified or silently adjusted.
- **Explicit failures**: validation errors are described clearly.
- **Minimal public surface**: users should primarily interact with `ExperimentSession`.

## 3. Public API Philosophy

The public API is intentionally very small.

Users should primarily interact with:

- `ExperimentSession`
- `session.publish_evaluation()`
- `session.publish_artifact()`
- `session.finish()`

Avoid growing the API unnecessarily. Every new public symbol increases maintenance burden and locks us into compatibility promises.

## 4. Supported Tasks

Today, the SDK supports:

- **Classification**
- **Regression**

### Classification

Requires: `sample_ids`, `ground_truth`, `predictions`

Optional: `probabilities`, `metadata`, `columns`

Produces columns: `sample_id`, `ground_truth`, `prediction`, `confidence`, `correct`.
Plus any user-supplied columns appended in the order they are provided.

### Regression

Requires: `sample_ids`, `ground_truth`, `predictions`

Optional: `metadata`, `columns`

Produces columns: `sample_id`, `ground_truth`, `prediction`, `residual`, `absolute_error`.
Plus any user-supplied columns appended in the order they are provided.

## 5. Per-Sample Columns

Users may attach arbitrary per-sample columns via the ``columns``
dictionary argument to ``publish_evaluation()``. This is a generic
mechanism — the SDK remains completely domain agnostic.

**Validation:**

- Every value must be a list-like with the same length as ``sample_ids``.
- Column names must not collide with the reserved SDK-owned columns
  (``sample_id``, ``ground_truth``, ``prediction``, ``confidence``,
  ``correct``, ``residual``, ``absolute_error``, ``evaluation_index``).

**Behavior:**

- Columns are appended to the evaluation ``DataFrame`` after the
  standard columns, preserving the dict iteration order.
- They are serialized into ``evaluation.parquet`` automatically.
- They are **ignored** by ``compute_metrics()``.
- They do **not** appear in ``metrics.json`` or ``artifacts.json``.
- The SDK never interprets column values.

**Architectural rule:**

Downstream platforms may use these columns for investigation,
visualization, or provenance, but the SDK must never read, interpret,
or depend on any user-supplied column name or value.

### Future Tasks

New task types (Segmentation, Object Detection, Forecasting, Ranking, Survival Analysis) will fit into the same architecture:

1. Define the task constant in `constants.py`.
2. Add validation rules in `validation.py`.
3. Add metric functions in `metrics.py`.
4. Add table construction in `evaluation.py`.

No changes to `session.py` or the public API surface are required beyond
the already-added ``columns`` parameter.
