import tensorflow_datasets as tfds
import pandas as pd
import numpy as np

def load_qm9_datasets():
    """
    Load the QM9 dataset from tensorflow_datasets.
    
    Returns:
        - clean_df: DataFrame with SMILES, mu, HOMO, LUMO, alpha, gap
        - full_df:  DataFrame with SMILES and all numeric properties (excludes InChI or non-numeric fields)
    """
    print("Loading QM9 dataset from tensorflow_datasets...")
    dataset = tfds.load('qm9', split='train', as_supervised=False)
    data = tfds.as_numpy(dataset)

    clean_records = []
    full_records = []

    for entry in data:
        # Build full record
        full_record = {}
        for k, v in entry.items():
            if k == 'SMILES':
                full_record['smiles'] = v.decode('utf-8')
            elif isinstance(v, (float, int, np.number, np.float32, np.float64)):
                full_record[k] = float(v)
        full_records.append(full_record)

        # Build clean record
        clean_record = {
            'smiles': entry['SMILES'].decode('utf-8'),
            'mu': float(entry['mu']),
            'HOMO': float(entry['homo']),
            'LUMO': float(entry['lumo']),
            'alpha': float(entry['alpha']),
            'gap': float(entry['gap']),
        }
        clean_records.append(clean_record)

    clean_df = pd.DataFrame(clean_records)
    full_df = pd.DataFrame(full_records)

    print(f"Loaded {len(full_df)} molecules")
    print(f"  - Clean features: {clean_df.shape[1]}")
    print(f"  - Full features:  {full_df.shape[1]}")
    
    return clean_df, full_df


def save_qm9_csvs():
    """
    Save both clean and full QM9 DataFrames to CSV files.
    """
    clean_df, full_df = load_qm9_datasets()
    output_dir='data/processed'
    clean_df.to_csv(f'{output_dir}/qm9_clean.csv', index=False)
    full_df.to_csv(f'{output_dir}/qm9_full.csv', index=False)
