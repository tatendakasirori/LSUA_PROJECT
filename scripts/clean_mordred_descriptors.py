import pandas as pd
from src.logger import Logger
from src.data_cleaning import (
    check_missing_values,
    drop_missing_rows,
    drop_low_variance_features,
    remove_highly_correlated_features,
    coerce_numeric_descriptors,
    drop_duplicate_smiles_columns
)

logger = Logger(log_name="clean_mordred_descriptors").get_logger()

INPUT_PATH = "data/descriptors/merged_qm9_mordred.csv"
OUTPUT_PATH = "data/processed/qm9_mordred_clean.csv"


def main():
    logger.info("Starting Mordred descriptor cleaning pipeline.")

    # Load data
    logger.info(f"Reading input CSV: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)

    # Drop duplicate SMILES columns
    df = drop_duplicate_smiles_columns(df)

    # Coerce descriptors to numeric
    df = coerce_numeric_descriptors(df)

    # Check and log missing values
    check_missing_values(df)

    # Drop rows with missing values
    df = drop_missing_rows(df)

    # Drop low-variance features
    df = drop_low_variance_features(df)

    # Drop highly correlated features
    df = remove_highly_correlated_features(df)

    # Save cleaned data
    logger.info(f"Saving cleaned data to: {OUTPUT_PATH}")
    df.to_csv(OUTPUT_PATH, index=False)

    logger.info("Mordred descriptor cleaning pipeline complete.")


if __name__ == "__main__":
    main()
