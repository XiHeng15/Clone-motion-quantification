import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate kinematic data from CSV files."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing the CSV files to aggregate.",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Path to the output CSV file.",
    )
    args = parser.parse_args()

    input_folder = Path(args.input_dir)
    output_file = Path(args.output_file).resolve()

    if not input_folder.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_folder}")

    # Use glob() for CSVs directly inside the folder.
    csv_files = sorted(
        path
        for path in input_folder.glob("*.csv")
        if path.resolve() != output_file
    )

    if not csv_files:
        raise SystemExit(f"No CSV files found in: {input_folder}")

    dataframes = []

    for file_number, csv_file in enumerate(csv_files, start=1):
        dataframe = pd.read_csv(csv_file)

        # Mark every row with its original file.
        dataframe.insert(0, "file_number", file_number)
        dataframe.insert(1, "file_name", csv_file.name)
        dataframe.insert(2, "row_in_file", range(1, len(dataframe) + 1))

        dataframes.append(dataframe)

    aggregated_df = pd.concat(dataframes, ignore_index=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    aggregated_df.to_csv(output_file, index=False)

    print(f"Combined {len(csv_files)} CSV files.")
    print(f"Combined CSV saved to: {output_file}")
    print("Original files were not modified.")


if __name__ == "__main__":
    main()