import argparse
import pickle
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds

from propmolflow.sdf_data_fix.utils import (
    transfer_bonds_and_charges, get_inchi_atom_mapping_with_h,
    remove_stereo_from_inchi, adjust_formal_charges_neutralize,
    save_mol_to_sdf, zero_formal_charges,
    fix_carbon_valency, fix_nitrogen_valency, fix_valency
)

RDLogger.DisableLog('rdApp.*')

def fix_pipeline(mols, inchis):
    fixed_mols, err_inds = [], []
    valid_inds = []

    filtered_inchis = [remove_stereo_from_inchi(inchi) for inchi in inchis]

    # First pass: bond determination + charge adjustment
    for i, mol in enumerate(mols):
        try:
            mol = zero_formal_charges(mol)
            rdDetermineBonds.DetermineBonds(mol)
            fixed_mol, _ = adjust_formal_charges_neutralize(mol)
            fixed_mols.append(fixed_mol)
            valid_inds.append(i)
        except Exception:
            fixed_mols.append(-100)
            err_inds.append(i)

    # Retry: charge adjustment only
    for i in err_inds[:]:
        try:
            mol = zero_formal_charges(mols[i])
            fixed_mol, _ = adjust_formal_charges_neutralize(mol)
            fixed_mols[i] = fixed_mol
            valid_inds.append(i)
            err_inds.remove(i)
        except Exception:
            continue

    # Retry: valency fixes for N- and C- bonds
    for i in err_inds[:]:
        try:
            mol = zero_formal_charges(mols[i])
            mol = fix_nitrogen_valency(mol)
            mol = fix_carbon_valency(mol)
            fixed_mol, _ = adjust_formal_charges_neutralize(mol)
            fixed_mols[i] = fixed_mol
            valid_inds.append(i)
            err_inds.remove(i)
        except Exception:
            continue

    # Retry: InChI-based remapping
    for i in err_inds[:]:
        try:
            atom_map = get_inchi_atom_mapping_with_h(mols[i])
            qm9_mol = Chem.MolFromInchi(filtered_inchis[i])
            qm9_mol = Chem.AddHs(qm9_mol)
            Chem.Kekulize(qm9_mol, clearAromaticFlags=True)
            assert mols[i].GetNumAtoms() == qm9_mol.GetNumAtoms()
            _mol = fix_valency(mols[i])
            new_mol = transfer_bonds_and_charges(qm9_mol, _mol, atom_map)
            new_mol, _ = adjust_formal_charges_neutralize(new_mol)
            Chem.Kekulize(new_mol, clearAromaticFlags=True)
            Chem.SanitizeMol(new_mol)
            fixed_mols[i] = new_mol
            valid_inds.append(i)
            err_inds.remove(i)
        except Exception:
            continue

    return fixed_mols, err_inds

def main():
    parser = argparse.ArgumentParser(description="Fix molecules in SDF using InChI guidance")
    parser.add_argument("--inchi_pkl", required=True, help="Pickle file with list of InChI strings")
    parser.add_argument("--input_sdf", required=True, help="Input SDF file")
    parser.add_argument("--output_sdf", required=True, help="Final cleaned SDF output;Unfixed molecules are in the same order.")
    parser.add_argument("--bad_indices", required=True, help="Output .npy file for unfixed indices")
    args = parser.parse_args()

    with open(args.inchi_pkl, "rb") as f:
        inchis = pickle.load(f)

    mols = Chem.SDMolSupplier(args.input_sdf, removeHs=False, sanitize=False)
    mols = [m for m in mols]

    fixed_mols, err_inds = fix_pipeline(mols, inchis)

    cleaned_mols = []
    for i, _ in enumerate(fixed_mols):
        if _ != -100:
            cleaned_mols.append(_)
        else:
            cleaned_mols.append(mols[i])

    save_mol_to_sdf(cleaned_mols, args.output_sdf)
    np.save(args.bad_indices, np.array(err_inds, dtype=int))

if __name__ == "__main__":
    main()
