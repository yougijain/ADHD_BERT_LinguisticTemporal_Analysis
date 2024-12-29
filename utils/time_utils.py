# Functions to process and analyze timestamp data
import pandas as pd

def convert_to_datetime(data, column):
    """
    Converts a specified column in the dataset to datetime format.
    Args:
        data (pd.DataFrame): The dataset containing the column.
        column (str): The column name to convert.
    Returns:
        pd.DataFrame: The dataset with the converted datetime column.
    """
    try:
        data[column] = pd.to_datetime(data[column], unit='s', errors='coerce')
    except KeyError:
        raise KeyError(f"The column '{column}' does not exist in the dataset.")
    except Exception as e:
        raise RuntimeError(f"Error converting column '{column}' to datetime: {e}")
    return data
