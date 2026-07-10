"""Heuristic Parquet size estimation for DataFrames.

This module estimates the compressed Parquet file size from an
in-memory pandas DataFrame without serializing it to disk.
It operates column-by-column and is designed to be extended
as new data types and compression heuristics are added.
"""

from typing import Dict

import pandas as pd


# Named compression-factor constants.
# Each factor is the expected ratio of compressed Parquet size to raw
# in-memory column size. These values are deliberately conservative.
# They can be refined with empirical benchmarks over time.
_COMPRESSION_FACTORS: Dict[str, float] = {
    # Boolean columns have extreme repetition (True/False only) and
    # compress extremely well under dictionary / RLE encoding.
    "bool": 0.15,
    # Integer columns compress well due to delta encoding and run-length
    # encoding, but are less uniform than booleans.
    "integer": 0.50,
    # Float columns have less repetition than integers because of
    # continuous values; compression is lower.
    "float": 0.65,
    # String / object columns use dictionary encoding when cardinality
    # is low; see _estimate_string_compression_factor for the dynamic
    # breakdown.
    "string_low_cardinality": 0.25,
    "string_moderate_cardinality": 0.50,
    "string_high_cardinality": 0.75,
}

# Cardinality thresholds for string/object columns,
# expressed as the ratio (unique_values / total_rows).
_STRING_CARDINALITY_THRESHOLDS: Dict[str, float] = {
    "low_max_ratio": 0.1,  # Very low cardinality: excellent compression.
    "moderate_max_ratio": 0.5,  # Moderate cardinality: moderate compression.
}

# Sample size used when evaluating string cardinality to avoid expensive
# full-length nunique() operations on very large columns.
_STRING_CARDINALITY_SAMPLE_SIZE: int = 10_000


def estimate_dataframe_parquet_size(df: pd.DataFrame) -> int:
    """Estimate the compressed Parquet size of an entire DataFrame.

    Operates entirely on the in-memory DataFrame. Does not serialize
    to disk and does not allocate large additional memory buffers.

    Args:
        df: The pandas DataFrame to estimate.

    Returns:
        Estimated file size in bytes. Returns 0 for empty DataFrames.
    """
    total: float = 0.0
    for column in df.columns:
        total += _estimate_column_parquet_size(df, column)
    return int(total)


def _estimate_column_parquet_size(df: pd.DataFrame, column: str) -> float:
    """Estimate the compressed Parquet size of a single column.

    Uses the DataFrame's memory_usage() so that column sizing matches
    the DataFrame's total and avoids Series object overhead.

    Args:
        df: The pandas DataFrame containing the column.
        column: The column name to estimate.

    Returns:
        Estimated compressed size in bytes.
    """
    raw_size = int(df.memory_usage(deep=True)[column])
    if raw_size == 0:
        return 0.0

    compression_factor = _get_compression_factor_for_column(df[column])
    return raw_size * compression_factor


def _get_compression_factor_for_column(series: pd.Series) -> float:
    """Return the expected Parquet compression factor for a column.

    Args:
        series: A pandas Series representing one DataFrame column.

    Returns:
        A multiplicative compression factor (0.0 < factor <= 1.0).
    """
    dtype = series.dtype

    if pd.api.types.is_bool_dtype(dtype):
        return _COMPRESSION_FACTORS["bool"]

    if pd.api.types.is_integer_dtype(dtype):
        return _COMPRESSION_FACTORS["integer"]

    if pd.api.types.is_float_dtype(dtype):
        return _COMPRESSION_FACTORS["float"]

    if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
        return _estimate_string_compression_factor(series)

    # Default fallback for unknown or complex dtypes.
    # Assumes moderate compression.
    return 0.70


def _estimate_string_compression_factor(series: pd.Series) -> float:
    """Estimate compression factor for a string/object column.

    Uses sample-based cardinality estimation to avoid expensive full-
    column nunique() calls on very large columns.

    Args:
        series: A pandas Series with object or string dtype.

    Returns:
        A multiplicative compression factor based on cardinality.
    """
    len_series = len(series)
    if len_series == 0:
        return _COMPRESSION_FACTORS["string_high_cardinality"]

    sample_size = min(len_series, _STRING_CARDINALITY_SAMPLE_SIZE)

    if sample_size < len_series:
        sample = series.sample(n=sample_size, random_state=42)
    else:
        sample = series

    unique_values = sample.nunique()
    cardinality_ratio = unique_values / sample_size

    if cardinality_ratio <= _STRING_CARDINALITY_THRESHOLDS["low_max_ratio"]:
        return _COMPRESSION_FACTORS["string_low_cardinality"]

    if cardinality_ratio <= _STRING_CARDINALITY_THRESHOLDS["moderate_max_ratio"]:
        return _COMPRESSION_FACTORS["string_moderate_cardinality"]

    return _COMPRESSION_FACTORS["string_high_cardinality"]
