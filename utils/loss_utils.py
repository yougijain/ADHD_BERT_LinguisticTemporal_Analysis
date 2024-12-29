import numpy as np

def moving_average(data, window_size=10):
    """
    Calculates the moving average for a list of values.
    Args:
        data (list): List of numerical values.
        window_size (int): Number of values to include in the moving average.
    Returns:
        np.ndarray: Smoothed values.
    """
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')
