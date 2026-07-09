# Future Roadmap

This document explains the intended evolution of the SDK. It is written for future contributors and AI coding agents.

## Phase 1 — Current (Outputs to Local Disk)

The SDK writes three files to a user-specified directory:

- `evaluation.parquet`
- `metrics.json`
- `artifacts.json`

The SDK owns nothing beyond serialization. It does not search, query, index, or visualize outputs. Another application consumes them later.

## Phase 2 — Configurable Output Providers

The writing mechanism should become an abstraction.

For example:

```python
EvaluationWriter
├── LocalFilesystemWriter
├── PlatformWriter
├── CloudWriter
└── CustomWriter
```

The SDK public API should **not** change. Only the output backend changes.

This allows:

```python
session = ExperimentSession(
    output_directory="./results",
    writer=PlatformWriter(workspace_id="..."),
)
```

User code remains identical.

## Phase 3 — Research Operating System Integration

The intended consumer is a Research Operating System.

The platform will maintain its own local workspace. Each completed run will publish standardized evaluation artifacts into that workspace.

The platform—not the SDK—will own indexing, discovery, querying, and investigation.

The SDK should never become tightly coupled to that platform.

## Phase 4 — Queryable Evaluation Artifacts

The platform will query evaluation artifacts rather than model code.

Future investigation examples include:

- False positives / false negatives
- Class A predicted as Class B
- Confidence filtering
- Calibration
- Residual analysis
- Embedding exploration
- Attention visualization
- Filtering by per-sample columns (e.g. ``window_start``, ``record_name``)

The SDK should continue to focus only on producing standardized evaluation artifacts.

## Per-Sample Columns (Implemented)

The ``columns`` parameter on ``publish_evaluation()`` allows users to
attach domain-specific per-sample data (e.g. ``window_start``,
``record_name``, ``bounding_box``, ``token_span``) to the evaluation
table.

These columns are:

- Validated (length, reserved name collision)
- Stored in the evaluation table
- Serialized into ``evaluation.parquet``
- Ignored by metrics computation

This feature is intentionally generic. It exists so that downstream
platforms can consume the extra columns for investigation and
provenance without the SDK needing to know what they mean.

## Future Architectural Goal

Eventually the SDK should support replacing:

```python
session.publish_evaluation(...)
    ↓
Local filesystem
```

with:

```python
session.publish_evaluation(...)
    ↓
Research Platform
```

without changing user code.

This should be accomplished through interchangeable output providers.

---

**Do not prematurely implement this yet.** This document exists to communicate architectural intent to future contributors and future AI coding agents.
