import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from joblib import Parallel, delayed
from tqdm import tqdm
from mordred import Calculator, descriptors 
from src.logger import Logger
from tqdm.auto import tqdm
import logging


# -------------------------
# 2D Descriptor Functions
# -------------------------

def compute_2d_descriptors(mol):
    """
    Compute a dictionary of selected 2D descriptors for a single RDKit molecule.
    Returns dict or None if molecule is invalid.
    """
    if mol is None:
        return None
    try:
        return {
            "MolWt": Descriptors.MolWt(mol),
            "TPSA": Descriptors.TPSA(mol),
            "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
            "NumHAcceptors": Descriptors.NumHAcceptors(mol),
            "NumHDonors": Descriptors.NumHDonors(mol),
            "BalabanJ": rdMolDescriptors.CalcBalabanJ(mol),
            "BertzCT": rdMolDescriptors.CalcBertzCT(mol),
        }
    except Exception:
        return None

# -------------------------
# 3D Descriptor Functions
# -------------------------

def generate_3d_conformer(mol, max_attempts=10):
    """
    Generate a 3D conformer for the molecule if possible.
    Returns molecule with conformer or None.
    """
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    for _ in range(max_attempts):
        try:
            result = AllChem.EmbedMolecule(mol, params)
            if result == 0:
                AllChem.UFFOptimizeMolecule(mol)
                mol = Chem.RemoveHs(mol)
                return mol
        except Exception:
            continue
    return None

def compute_3d_descriptors(mol):
    """
    Compute 3D descriptors for a molecule with conformer.
    Returns dict or None if descriptors cannot be computed.
    """
    if mol is None or mol.GetNumConformers() == 0:
        return None
    try:
        return {
            "RadiusOfGyration": rdMolDescriptors.CalcRadiusOfGyration(mol),
            "InertialShapeFactor": rdMolDescriptors.CalcInertialShapeFactor(mol),
            "Asphericity": rdMolDescriptors.CalcAsphericity(mol),
            "Eccentricity": rdMolDescriptors.CalcEccentricity(mol),
            "PMI1": rdMolDescriptors.CalcPMI1(mol),
            "PMI2": rdMolDescriptors.CalcPMI2(mol),
            "PMI3": rdMolDescriptors.CalcPMI3(mol),
            "SpherocityIndex": rdMolDescriptors.CalcSpherocityIndex(mol),
            "NPR1": rdMolDescriptors.CalcNPR1(mol),
            "NPR2": rdMolDescriptors.CalcNPR2(mol),
        }
    except Exception:
        return None

# -------------------------
# Parallelized Descriptor Computation Wrappers
# -------------------------

def process_row(smiles):
    """
    Given a SMILES string:
    - Convert to RDKit mol
    - Compute 2D descriptors
    - Generate 3D conformer and compute 3D descriptors
    Returns combined descriptor dict or None.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    desc_2d = compute_2d_descriptors(mol)
    mol_3d = generate_3d_conformer(mol)
    desc_3d = compute_3d_descriptors(mol_3d) if mol_3d else None

    if desc_2d is None and desc_3d is None:
        return None

    combined = {}
    if desc_2d:
        combined.update(desc_2d)
    if desc_3d:
        combined.update(desc_3d)

    return combined

def compute_descriptors_parallel(df, smiles_column="smiles", n_jobs=-1):
    smiles_list = df[smiles_column].tolist()
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_row)(smiles) for smiles in tqdm(smiles_list, desc="Computing descriptors")
    )

    # Remove None values and adjust index accordingly
    valid_indices = [i for i, r in enumerate(results) if r is not None]
    results = [r for r in results if r is not None]
    index = df.index[valid_indices]

    desc_df = pd.DataFrame(results, index=index)
    return desc_df


# -------------------------
# Optional: Function to merge descriptors with original dataframe
# -------------------------

def add_descriptors_to_df(df, desc_df):
    """
    Safely merge the descriptor dataframe with the original dataframe.
    """
    return pd.concat([df, desc_df], axis=1)


logger = logging.getLogger(__name__)

from mordred import Calculator, descriptors
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def generate_mordred_descriptors(smiles_list):
    selected_descriptors = [
        'nHetero','nBonds', 'nBondsO', 'nBondsS', 'nBondsM', 'nBondsKS', 'nBondsKD',
        'C1SP1', 'C2SP1', 'C1SP2', 'C2SP2', 'C1SP3', 'C2SP3', 'HybRatio', 'FCSP3',
        'SpAbs_A', 'SpMax_A', 'SpDiam_A', 'SpAD_A', 'SpMAD_A',
        'VE1_A', 'VE2_A', 'VE3_A', 'VR1_A', 'VR2_A',
        'GATS1c', 'GATS2c'
    ]

    # Initialize calculator with all descriptors, then filter by name
    calc = Calculator(descriptors, ignore_3D=True)
    calc.descriptors = [d for d in calc.descriptors if str(d) in selected_descriptors]

    data = []
    valid_smiles = []

    for smiles in tqdm(smiles_list, desc="Calculating Mordred descriptors", ncols=100):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Skipping invalid SMILES: {smiles}")
            continue

        try:
            desc_values = calc(mol)
            desc_dict = {}
            for desc, value in desc_values.items():
                try:
                    if hasattr(value, 'fill_missing'):
                        value = value.fill_missing(None)
                    desc_dict[str(desc)] = value
                except Exception as e:
                    logger.warning(f"Descriptor error {desc} on {smiles}: {e}")
                    desc_dict[str(desc)] = None
            data.append(desc_dict)
            valid_smiles.append(smiles)
        except Exception as e:
            logger.error(f"Failed to process {smiles}: {e}")
            continue

    if not data:
        logger.error("Mordred descriptor computation failed or returned no data.")
        return None

    df = pd.DataFrame(data)
    df.insert(0, 'SMILES', valid_smiles)
    return df.dropna()
