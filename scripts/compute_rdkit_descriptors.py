from src.feature_engineering import compute_descriptors_parallel, add_descriptors_to_df
import pandas as pd
import os

if __name__ == "__main__":
    df = pd.read_csv("data/processed/qm9_clean.csv")
    descriptors_df = compute_descriptors_parallel(df, smiles_column="smiles")
    df_with_desc = add_descriptors_to_df(df.loc[descriptors_df.index], descriptors_df)
    
    os.makedirs("data/descriptors", exist_ok=True)
    df_with_desc.to_csv("data/descriptors/rdkit_descriptors.csv", index=False)
