import pandas as pd

def inspect_dataset(file_path):
    """
    Load and inspect the dataset.
    Args:
        file_path (str): Path to the dataset file.
    """
    data = pd.read_csv(file_path)
    print("First 5 rows of the dataset:")
    print(data.head())
    print("\nDataset Info:")
    print(data.info())

if __name__ == "__main__":
    # Example: Inspect ADHD dataset
    inspect_dataset("datasets/ADHD.csv")
