"""Quick look at a raw CSV before running anything expensive on it."""

import argparse

import pandas as pd

from training.config import REQUIRED_COLUMNS, TEXT_COLUMN


def inspect_dataset(file_path, sample_rows=5):
    """Print the shape, columns, dtypes, missing-value counts, and a few rows.

    Also checks up front for the columns the pipeline needs, so a missing one
    surfaces here rather than midway through tokenization.
    """
    data = pd.read_csv(file_path)

    print(f"{file_path}: {len(data)} rows x {len(data.columns)} columns\n")
    print("Columns:")
    for column in data.columns:
        missing = int(data[column].isna().sum())
        share = missing / len(data) if len(data) else 0.0
        print(f"  {column:<20} {str(data[column].dtype):<10} "
              f"missing {missing} ({share:.1%})")

    absent = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if absent:
        print(f"\nWARNING: missing required column(s) {absent}. "
              f"The pipeline needs {list(REQUIRED_COLUMNS)}.")
    else:
        print("\nAll required columns present.")

    if TEXT_COLUMN in data.columns:
        placeholders = data[TEXT_COLUMN].isin(["[removed]", "[deleted]"]).sum()
        print(f"Placeholder bodies ([removed]/[deleted]): {placeholders} "
              f"({placeholders / max(len(data), 1):.1%}) -- these get dropped.")

    print(f"\nFirst {sample_rows} rows:")
    print(data.head(sample_rows).to_string(max_colwidth=60))
    return data


def main():
    parser = argparse.ArgumentParser(description="Inspect a raw dataset CSV.")
    parser.add_argument("--dataset", required=True, help="Path to the CSV.")
    parser.add_argument("--rows", type=int, default=5, help="Sample rows to show.")
    args = parser.parse_args()
    inspect_dataset(args.dataset, args.rows)


if __name__ == "__main__":
    main()
