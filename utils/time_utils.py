"""Timestamp handling and temporal feature engineering.

Reddit gives us `created_utc` as a Unix epoch integer. On its own that is a
meaningless magnitude for a model, so this module turns it into features that
carry the circadian signal the project cares about: what hour of the day a post
was written, whether it was a weekday, and whether it landed in the small hours.

Hour-of-day and day-of-week are cyclical -- hour 23 is adjacent to hour 0, but
the raw integers are 23 apart. Encoding each as a (sin, cos) pair puts them on a
circle so the model sees that adjacency.
"""

import numpy as np
import pandas as pd

# Posts made in [LATE_NIGHT_START, LATE_NIGHT_END) local hours are flagged as
# late-night. This window is the "revenge bedtime procrastination" band that
# shows up in self-reported ADHD sleep patterns.
LATE_NIGHT_START = 0
LATE_NIGHT_END = 5


def convert_to_datetime(data, column, unit="s"):
    """Convert an epoch column to pandas datetime, in place on a copy.

    Args:
        data (pd.DataFrame): The dataset containing the column.
        column (str): The column name to convert.
        unit (str): Epoch unit passed to pd.to_datetime. Defaults to seconds.
    Returns:
        pd.DataFrame: A copy with `column` converted to datetime64.
    Raises:
        KeyError: If the column is not present.
    """
    if column not in data.columns:
        raise KeyError(f"The column '{column}' does not exist in the dataset.")

    data = data.copy()
    values = data[column]

    # Already datetime -- nothing to do.
    if pd.api.types.is_datetime64_any_dtype(values):
        return data

    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().any():
        # Epoch numbers. Guard against millisecond timestamps, which are ~1000x
        # too large and would otherwise land in the year 50000+.
        finite = numeric.dropna()
        if unit == "s" and len(finite) and finite.abs().median() > 1e11:
            unit = "ms"
        data[column] = pd.to_datetime(numeric, unit=unit, errors="coerce", utc=True)
    else:
        # Fall back to string parsing for ISO-formatted dumps.
        data[column] = pd.to_datetime(values, errors="coerce", utc=True)

    # Strip the tz so downstream .dt accessors and comparisons stay simple.
    if isinstance(data[column].dtype, pd.DatetimeTZDtype):
        data[column] = data[column].dt.tz_localize(None)

    return data


def _cyclical(values, period):
    """Map integer values in [0, period) onto the unit circle."""
    radians = 2.0 * np.pi * (np.asarray(values, dtype="float64") / period)
    return np.sin(radians), np.cos(radians)


def add_temporal_features(data, column="created_utc"):
    """Derive the model's temporal features from a datetime column.

    Adds: hour, day_of_week, month, hour_sin/cos, dow_sin/cos, month_sin/cos,
    is_weekend, is_late_night.

    Args:
        data (pd.DataFrame): Dataset with `column` already converted to datetime
            (or convertible -- this calls convert_to_datetime defensively).
        column (str): Name of the datetime column.
    Returns:
        pd.DataFrame: A copy with the temporal feature columns appended.
    """
    data = convert_to_datetime(data, column)
    ts = data[column]

    if ts.isna().all():
        raise ValueError(
            f"Column '{column}' contains no parseable timestamps; "
            "cannot derive temporal features."
        )

    data = data.copy()
    data["hour"] = ts.dt.hour.astype("Int64")
    data["day_of_week"] = ts.dt.dayofweek.astype("Int64")  # Monday = 0
    data["month"] = ts.dt.month.astype("Int64")

    # Fill gaps before the trig so NaN does not propagate into the features.
    # A missing timestamp gets the dataset's modal hour/day rather than a
    # silently-zero feature, which would read as "posted at midnight on Monday".
    hour = data["hour"].fillna(_mode_or(data["hour"], 12)).astype("int64")
    dow = data["day_of_week"].fillna(_mode_or(data["day_of_week"], 0)).astype("int64")
    month = data["month"].fillna(_mode_or(data["month"], 1)).astype("int64")

    data["hour_sin"], data["hour_cos"] = _cyclical(hour, 24)
    data["dow_sin"], data["dow_cos"] = _cyclical(dow, 7)
    # Months are 1-12, so shift to 0-11 before placing them on the circle.
    data["month_sin"], data["month_cos"] = _cyclical(month - 1, 12)

    data["is_weekend"] = (dow >= 5).astype("float64")
    data["is_late_night"] = (
        (hour >= LATE_NIGHT_START) & (hour < LATE_NIGHT_END)
    ).astype("float64")

    return data


def _mode_or(series, default):
    """Most common value in a series, or `default` if the series is all-NA."""
    modes = series.dropna().mode()
    return default if modes.empty else modes.iloc[0]


def temporal_feature_matrix(data, feature_names):
    """Stack the named temporal columns into a float32 matrix for the model.

    Args:
        data (pd.DataFrame): Dataset that has been through add_temporal_features.
        feature_names (list[str]): Columns to stack, in model input order.
    Returns:
        np.ndarray: Shape (len(data), len(feature_names)), dtype float32.
    Raises:
        KeyError: If any requested feature is missing.
    """
    missing = [name for name in feature_names if name not in data.columns]
    if missing:
        raise KeyError(
            f"Missing temporal features {missing}. "
            "Call add_temporal_features() before building the matrix."
        )
    matrix = data.loc[:, list(feature_names)].to_numpy(dtype="float32", copy=True)
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def hourly_distribution(data, column="created_utc", normalize=True):
    """Posts per hour of day, as a 24-entry Series indexed 0-23.

    Used by the timestamp analysis to show when the subreddit is awake.
    """
    data = convert_to_datetime(data, column)
    counts = data[column].dt.hour.value_counts().reindex(range(24), fill_value=0)
    counts = counts.sort_index()
    if normalize and counts.sum() > 0:
        counts = counts / counts.sum()
    return counts


def late_night_share(data, column="created_utc"):
    """Fraction of posts written in the late-night window. Returns 0.0 if empty."""
    data = convert_to_datetime(data, column)
    hours = data[column].dt.hour.dropna()
    if len(hours) == 0:
        return 0.0
    return float(((hours >= LATE_NIGHT_START) & (hours < LATE_NIGHT_END)).mean())
