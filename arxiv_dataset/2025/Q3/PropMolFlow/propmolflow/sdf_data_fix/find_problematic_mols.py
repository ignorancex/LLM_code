import argparse
from pathlib import Path
import numpy as np
from rdkit import Chem
from propmolflow.sdf_data_fix.utils import (
    count_multifragment_molecules,
    check_valency_charge_balance,
    get_rdkit_valid,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Detect problematic molecules (valency/charge / RDKit invalid / multi-fragment)"
    )
    p.add_argument("--input_sdf", required=True, help="Input (refined) SDF file, e.g. rQM9.sdf")
    p.add_argument(
        "--output_indices",
        required=True,
        help="Output file for problematic indices (supports .txt or .npy)",
    )
    return p.parse_args()


def load_total_count(sdf_path: str) -> int:
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
    return sum(1 for _ in supplier)


def main():
    args = parse_args()
    sdf_file = args.input_sdf

    # Valid according to custom valency/charge checks
    valid_mol_indices, mol_ratio, atom_ratio, _ = check_valency_charge_balance(sdf_file)

    # Valid according to RDKit sanitization (utility returns indices already)
    sdf_valid_index, _ = get_rdkit_valid(sdf_file)

    # Multi-fragment detection
    _, multifrag_indices = count_multifragment_molecules(sdf_file)

    total = load_total_count(sdf_file)

    not_fixed = set(range(total)) - set(valid_mol_indices)
    rdkit_invalid = set(range(total)) - set(sdf_valid_index)
    multifrag_set = set(multifrag_indices)

    all_invalid = sorted(not_fixed | rdkit_invalid | multifrag_set)

    out_path = Path(args.output_indices)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix == ".npy":
        np.save(out_path, np.array(all_invalid, dtype=int))
    else:
        with open(out_path, "w") as f:
            for idx in all_invalid:
                f.write(f"{idx}\n")

    print(
        f"Total molecules: {total}\n"
        f"Valency/charge invalid: {len(not_fixed)}\n"
        f"RDKit invalid: {len(rdkit_invalid)}\n"
        f"Multi-fragment: {len(multifrag_set)}\n"
        f"Unique problematic (union): {len(all_invalid)}\n"
        f"Saved indices to: {out_path}"
    )


if __name__ == "__main__":
    main()