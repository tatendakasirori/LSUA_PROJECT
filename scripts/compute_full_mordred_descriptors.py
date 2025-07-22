import pandas as pd
import os
from src.feature_engineering import generate_mordred_descriptors, add_descriptors_to_df
from src.logger import Logger

logger = Logger(log_name="compute_mordred_all").get_logger()

def main():
    input_file = "data/descriptors/rdkit_descriptors.csv"
    output_file = "data/descriptors/mordred_full_descriptors.csv"
    merged_output = "data/descriptors/merged_qm9_mordred.csv"

    logger.info(f"Reading input file: {input_file}")
    df = pd.read_csv(input_file)

    smiles_list = df["smiles"].tolist()
    df_mordred = generate_mordred_descriptors(smiles_list)

    if df_mordred is None or df_mordred.empty:
        logger.error("Mordred descriptor computation failed or returned no data.")
        return

    # Ensure order and row alignment are valid
    df_filtered = df[df["smiles"].isin(df_mordred["SMILES"])]
    df_filtered = df_filtered.reset_index(drop=True)
    df_mordred = df_mordred.reset_index(drop=True)

    # Drop 'SMILES' from Mordred if already in df_filtered to avoid duplicate columns
    if "SMILES" in df_filtered.columns and "SMILES" in df_mordred.columns:
        df_mordred = df_mordred.drop(columns=["SMILES"])

    logger.info("Merging original dataset with Mordred descriptors...")
    df_merged = add_descriptors_to_df(df_filtered, df_mordred)

    # Save files
    os.makedirs("data/descriptors", exist_ok=True)
    df_mordred.to_csv(output_file, index=False)
    df_merged.to_csv(merged_output, index=False)
    logger.info(f"Saved Mordred descriptors to {output_file}")
    logger.info(f"Saved merged dataset to {merged_output}")
    logger.info(f"Merged dataset shape: {df_merged.shape}")

if __name__ == "__main__":
    main()
