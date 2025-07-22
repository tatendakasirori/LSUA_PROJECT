import pandas as pd
import numpy as np
from src.logger import Logger

logger = Logger(log_name="data_cleaning").get_logger()


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Logs and returns the count of missing (NaN) values per column.
    """
    missing_counts = df.isna().sum()
    total_missing = missing_counts.sum()

    if total_missing == 0:
        logger.info("No missing values found in dataset.")
    else:
        logger.warning(f"Total missing values found: {total_missing}")
        for col, count in missing_counts.items():
            if count > 0:
                logger.warning(f"Column '{col}' has {count} missing values.")
    return missing_counts


def drop_missing_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops all rows with any missing values.
    """
    before = len(df)
    df_clean = df.dropna()
    after = len(df_clean)
    dropped = before - after
    logger.info(f"Dropped {dropped} rows containing missing values.")
    return df_clean


def drop_low_variance_features(df: pd.DataFrame, threshold: float = 1e-3, protect_cols=None) -> pd.DataFrame:
    """
    Drops columns with variance less than or equal to the threshold.
    Does NOT drop protected columns.
    """
    if protect_cols is None:
        protect_cols = ["HOMO", "LUMO", "gap", "SMILES", "Index"]

    variances = df.var(numeric_only=True)
    low_var_cols = [col for col in variances.index if variances[col] <= threshold and col not in protect_cols]

    if low_var_cols:
        logger.info(f"Dropping {len(low_var_cols)} low-variance columns: {low_var_cols}")
        df = df.drop(columns=low_var_cols)
    else:
        logger.info("No low-variance columns found.")
    return df


def remove_highly_correlated_features(df: pd.DataFrame, correlation_threshold: float = 0.95, protect_cols=None) -> pd.DataFrame:
    """
    Removes highly correlated features above the given threshold.
    Does NOT drop any columns listed in `protect_cols`.
    Keeps the first occurrence of correlated columns.
    """
    if protect_cols is None:
        protect_cols = ["HOMO", "LUMO", "gap", "SMILES", "Index"]

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols_to_consider = [col for col in numeric_cols if col not in protect_cols]

    if not cols_to_consider:
        logger.warning("No numeric columns available for correlation check.")
        return df

    corr_matrix = df[cols_to_consider].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]

    if to_drop:
        logger.info(f"Dropping {len(to_drop)} highly correlated columns: {to_drop}")
        df = df.drop(columns=to_drop)
    else:
        logger.info("No highly correlated features found above threshold.")

    return df

def coerce_numeric_descriptors(df: pd.DataFrame, protect_cols=None) -> pd.DataFrame:
    """
    Coerces all non-protected columns to numeric, setting errors='coerce'.
    This avoids corrupting identifier or string-based columns like SMILES.
    """
    if protect_cols is None:
        protect_cols = ["HOMO", "LUMO", "gap", "SMILES", "smiles", "Index", "index"]

    for col in df.columns:
        if col not in protect_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df



def drop_duplicate_smiles_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops known duplicate SMILES columns only if both exist and one is empty.
    Keeps the valid one (usually uppercase).
    """
    if "SMILES" in df.columns and "smiles" in df.columns:
        return df.drop(columns="SMILES")
    return df
