"""Smoothing helpers for loss curves."""

import numpy as np


def moving_average(data, window_size=10):
    """Trailing moving average over a sequence of values.

    np.convolve(a, v, mode="valid") swaps its arguments when the window is
    longer than the data, so the old one-liner returned
    `window_size - len(data) + 1` fabricated values instead of nothing --
    smoothing a 3-batch run with a window of 25 produced 23 invented points and
    plotted them as if they were real. Short inputs now return an empty array,
    which callers can check.

    Args:
        data (sequence): Numerical values.
        window_size (int): Number of values per average. Must be >= 1.
    Returns:
        np.ndarray: len(data) - window_size + 1 smoothed values, or an empty
        array when the data is shorter than the window.
    """
    values = np.asarray(data, dtype="float64")

    if window_size < 1:
        raise ValueError(f"window_size must be at least 1, got {window_size}")
    if values.ndim != 1:
        raise ValueError(f"Expected a 1-D sequence, got shape {values.shape}")
    if len(values) < window_size:
        return np.array([], dtype="float64")

    kernel = np.ones(window_size) / window_size
    return np.convolve(values, kernel, mode="valid")


def smooth_for_plot(data, target_points=40, max_window=50):
    """Pick a sensible smoothing window for a series and apply it.

    Saves every caller from hand-tuning a window that happens to exceed the
    number of batches they ran.

    Returns:
        tuple[np.ndarray, int]: the smoothed values and the window used.
    """
    values = np.asarray(data, dtype="float64")
    if len(values) < 3:
        return np.array([], dtype="float64"), 0

    window = int(np.clip(len(values) // target_points, 1, min(max_window, len(values))))
    return moving_average(values, window), window
