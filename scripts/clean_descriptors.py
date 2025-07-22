import pandas as pd
from src.data_cleaning import (
    check_missing_values,
    drop_missing_rows,
    drop_low_variance_features,
    remove_highly_correlated_features
)
from src.logger import Logger

logger = Logger(log_name="clean_descriptors").get_logger()

def main():
    input_path = "data/descriptors/rdkit_descriptors.csv"
    output_path = "data/processed/cleaned_descriptors.csv"

    logger.info(f"Loading descriptors from {input_path}")
    df = pd.read_csv(input_path)

    check_missing_values(df)
    df = drop_missing_rows(df)

    df = drop_low_variance_features(df, threshold=1e-3, protect_cols=["HOMO", "LUMO", "gap", "SMILES", "Index"])
    df = remove_highly_correlated_features(df, correlation_threshold=0.95, protect_cols=["HOMO", "LUMO", "gap", "SMILES", "Index"])

    logger.info(f"Saving cleaned descriptors to {output_path}")
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()
