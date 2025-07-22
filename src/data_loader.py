import tensorflow_datasets as tfds
import pandas as pd

def load_qm9():
    """
    Load the QM9 dataset from tensorflow_datasets and return as a pandas DataFrame.
    """
    print("Loading QM9 dataset from tensorflow_datasets...")
    dataset = tfds.load('qm9', split='train', as_supervised=False)
    data = tfds.as_numpy(dataset)

    records = []
    for entry in data:
        record = {
            'smiles': entry['SMILES'].decode('utf-8'),
            'mu': float(entry['mu']),              # Dipole moment
            'HOMO': float(entry['homo']),
            'LUMO': float(entry['lumo']),
            'alpha': float(entry['alpha']),        # Polarizability
            'gap': float(entry['gap']),            # HOMO-LUMO gap
            # Add more properties if needed
        }
        records.append(record)

    df = pd.DataFrame(records)
    return df

def save_qm9_csv(output_path='data/processed/qm9_clean.csv'):
    """
    Load and save the QM9 dataset as a CSV to a specified path.
    """
    df = load_qm9()
    df.to_csv(output_path, index=False)
    print(f"Saved QM9 data to {output_path}")
