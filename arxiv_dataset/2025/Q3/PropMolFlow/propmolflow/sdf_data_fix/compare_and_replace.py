import argparse
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import inchi
from rdkit import RDLogger

from propmolflow.sdf_data_fix.utils import (
    remove_stereo_from_inchi,
    compute_mmff_energy,
    transfer_bonds_and_charges,
    get_inchi_atom_mapping_with_h,
    adjust_formal_charges_neutralize,
    save_mol_to_sdf,
    fix_valency
)

RDLogger.DisableLog("rdApp.*")


def compare_stability(sdf_file, inchi_pkl, plain_smiles, may_not_match_npy):
    """Return indices where the plain/original structure is more stable."""
    with open(inchi_pkl, "rb") as f:
        inchi_list = pickle.load(f)
    with open(plain_smiles, "rb") as f:
        plain_smiles_list = pickle.load(f)

    filtered_list = [remove_stereo_from_inchi(inchi_str) for inchi_str in inchi_list]
    smiles_unmatch = set(np.array(np.load(may_not_match_npy)) - 1)
    mols = Chem.SDMolSupplier(sdf_file, removeHs=False, sanitize=False)

    inchi_strs = []
    for mol in mols:
        try:
            inchi_strs.append(inchi.MolToInchi(mol, options="-SNon"))
        except Exception:
            inchi_strs.append(-100)

    not_match = [
        i for i, (xyz_inchi, sdf_inchi) in enumerate(zip(filtered_list, inchi_strs))
        if xyz_inchi != sdf_inchi
    ]
    not_match_2 = list(set(not_match) - set(smiles_unmatch))

    results = []
    for ind in not_match_2:
        try:
            mol = mols[int(ind)]
            if mol is None:
                continue

            gen_smiles = Chem.MolToSmiles(mol)
            e_gen = compute_mmff_energy(gen_smiles)

            plain_smiles = plain_smiles_list[ind]
            # plain_smiles = Chem.MolToSmiles(Chem.MolFromInchi(inchi_list[ind]))
            e_plain = compute_mmff_energy(plain_smiles)

            if e_gen is not None and e_plain is not None and e_gen >= e_plain:
                results.append(ind)
        except Exception as e:
            print(e)
            continue

    print(f"[compare] Found {len(results)} indices where plain is more stable")
    return results, inchi_list


def replace_structures(sdf_file, inchi_list, indices, out_full, log_file):
    """Replace unstable molecules, save final SDF, and log replaced indices."""
    filtered_list = [remove_stereo_from_inchi(inchi) for inchi in inchi_list]
    mols = Chem.SDMolSupplier(sdf_file, removeHs=False, sanitize=False)
    _mols = np.array(mols, dtype=object)

    replaced_indices = []

    for ind in indices:
        mol = _mols[ind]
        no_atoms_mol = mol.GetNumAtoms()
        atom_map = get_inchi_atom_mapping_with_h(mol)

        qm9_inchi = filtered_list[ind]
        qm9_mol = Chem.MolFromInchi(qm9_inchi)
        qm9_mol = Chem.AddHs(qm9_mol)
        Chem.Kekulize(qm9_mol, clearAromaticFlags=True)
        no_atoms_qm9 = qm9_mol.GetNumAtoms()

        try:
            assert no_atoms_mol == no_atoms_qm9

            mol = fix_valency(mol)

            new_mol = transfer_bonds_and_charges(qm9_mol, mol, atom_map)
            new_mol, _ = adjust_formal_charges_neutralize(new_mol)
            Chem.SanitizeMol(new_mol)

            _mols[ind] = new_mol
            replaced_indices.append(ind)

        except Exception:
            continue

    # Save final dataset
    save_mol_to_sdf(_mols, filename=out_full)

    # Save log
    with open(log_file, "w") as f:
        for idx in replaced_indices:
            f.write(f"{idx}\n")

    print(f"[replace] Replaced {len(replaced_indices)} molecules")
    print(f"[pipeline] Final dataset saved to {out_full}")
    print(f"[pipeline] Log of replaced indices saved to {log_file}")


def main():
    parser = argparse.ArgumentParser(description="Pipeline: compare + replace, outputs only final SDF + log.")
    parser.add_argument("--sdf", required=True, help="Input SDF file")
    parser.add_argument("--inchi", required=True, help="Pickle file with InChI strings")
    parser.add_argument("--plain_smiles", required=True, help="Pickle file with plain SMILES strings")
    parser.add_argument("--may_not_match", required=True, help="Numpy file with may not match indices")
    parser.add_argument("--out", default="rQM9.sdf", help="Output full SDF file with replacements")
    parser.add_argument("--log", default="replaced.log", help="Log file for replaced indices")
    args = parser.parse_args()

    indices, inchi_list = compare_stability(args.sdf, args.inchi, args.plain_smiles, args.may_not_match)
    replace_structures(args.sdf, inchi_list, indices, args.out, args.log)


if __name__ == "__main__":
    main()