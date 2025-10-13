import re
from collections import Counter
from rdkit import Chem
from rdkit.Chem import SDWriter

def summarize_bond_types_single(mol):
    if mol is None:
        return None
    bond_type_counts = Counter()
    for bond in mol.GetBonds():
        bond_type = str(bond.GetBondType())  # e.g., SINGLE, DOUBLE, TRIPLE, AROMATIC
        bond_type_counts[bond_type] += 1
    return dict(bond_type_counts)

def generate_mol_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    mol = Chem.AddHs(mol)  # optional: add explicit Hs
    return mol

def remove_stereo_from_inchi(inchi):
    # Remove /b (double bond stereo), /t (chiral centers), /m (isotopic stereo), /s (stereo type)
    cleaned = re.sub(r"/[btms][^/]*", "", inchi)
    return cleaned

expected_valences = {
    'H': 1,
    'C': 4,
    'N': 3,
    'O': 2,
    'F': 1,
    'P': 3,
    'S': 2,
    'Cl': 1,
    'Br': 1,
    'I': 1
}
def adjust_formal_charges_neutralize(mol):
    mol = Chem.RWMol(mol)
    changed = False
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        expected = expected_valences[sym]
        valence = atom.GetExplicitValence() 
        # sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
        charge = atom.GetFormalCharge()
        # print(sym, valence, expected, charge)
        if valence -charge == expected:
            continue
        # print(charge, valence)
        for delta in [-1, 1]:
            atom.SetFormalCharge(charge + delta)
            new_valence = atom.GetExplicitValence()
            new_charge = charge + delta
            # print(new_valence)
            if new_valence - new_charge == expected:
                # print(f"Adjusted charge on {sym} from {charge} to {new_charge}")
                changed = True
                break
            atom.SetFormalCharge(charge)  # revert

        else:
            raise ValueError(f"❌ Could not fix valency for atom {sym} (idx={atom.GetIdx()}) — valence={valence}, expected={expected}")

    net_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    if net_charge != 0:
        raise ValueError(f"❌ Molecule not charge neutral after adjustment (net charge = {net_charge})")
    return mol.GetMol(), changed

def save_mol_to_sdf(mols, filename="adjusted_molecule.sdf"):
    writer = SDWriter(filename)
    for mol in mols:
        writer.write(mol)
    writer.close()
    print(f"✅ Molecule saved to {filename}")
    
def get_inchi_atom_mapping_with_h(mol):
    """
    Returns a mapping from RDKit atom index to InChI-like index.
    Heavy atoms get InChI canonical indices; hydrogens follow RDKit order.
    """
    inchi, aux_info = Chem.MolToInchiAndAuxInfo(mol)
    # Parse heavy atom mapping from /N:
    for part in aux_info.split('/'):
        if part.startswith('N:'):
            orig_heavy_indices = list(map(int, part[2:].split(',')))
            break
    else:
        raise ValueError("No heavy atom mapping (/N:) found in AuxInfo.")

    # Create RDKit atom list
    atoms = list(mol.GetAtoms())

    # Build mapping for heavy atoms
    heavy_mapping = {orig-1: i for i, orig in enumerate(orig_heavy_indices)}

    # Initialize mapping with all atoms
    full_mapping = {}

    heavy_idx_set = set(heavy_mapping.keys())
    next_h_index = len(heavy_mapping)
    
    for i, atom in enumerate(atoms):
        if i in heavy_idx_set:
            full_mapping[i] = heavy_mapping[i]
        else:
            full_mapping[i] = next_h_index
            next_h_index += 1
    inv_map = {v:k for k, v in heavy_mapping.items()} 
    return inv_map

def transfer_bonds_and_charges(mol_from, mol_to, atom_map):
    """
    Transfer bond types and formal charges from mol_from to mol_to,
    but only for atoms present in atom_map (typically heavy atoms).
    
    Parameters:
        mol_from (Chem.Mol): source molecule
        mol_to (Chem.Mol): destination molecule
        atom_map (dict): mapping {from_idx: to_idx} for heavy atoms only

    Returns:
        Chem.Mol: modified mol_to with transferred bonds and charges
    """
    rw_mol_to = Chem.RWMol(mol_to)

    # Transfer formal charges for mapped (heavy) atoms
    for idx_from, idx_to in atom_map.items():
        atom_from = mol_from.GetAtomWithIdx(idx_from)
        if atom_from.GetAtomicNum() > 1:
            charge = atom_from.GetFormalCharge()
            rw_mol_to.GetAtomWithIdx(idx_to).SetFormalCharge(charge)

    # Remove bonds in mol_to only if both atoms are mapped heavy atoms
    for bond in list(mol_to.GetBonds()):
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        if a1 in atom_map.values() and a2 in atom_map.values():
            rw_mol_to.RemoveBond(a1, a2)

    # Add bonds from mol_from only if both atoms are heavy and in atom_map
    for bond in mol_from.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        if a1 in atom_map and a2 in atom_map:
            new_a1 = atom_map[a1]
            new_a2 = atom_map[a2]
            bond_type = bond.GetBondType()
            rw_mol_to.AddBond(new_a1, new_a2, bond_type)
    rw_mol_to.UpdatePropertyCache(strict=False)
    return rw_mol_to.GetMol()

def get_atom_symbols(mol):
    return [atom.GetSymbol() for atom in mol.GetAtoms()]

def print_valency_and_charges(mol):
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        symbol = atom.GetSymbol()
        valency = atom.GetExplicitValence()
        print(f"Atom {idx} ({symbol}): Valency = {valency} Formal Charge = {atom.GetFormalCharge()}")
        

def smiles_to_conformer(smiles: str, num_confs: int = 1, random_seed: int = 42):
    # Step 1: Convert InChI to Mol
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")

    # Step 2: Add Hydrogens (important for geometry)
    mol = Chem.AddHs(mol)

    # Step 3: Embed 3D conformer(s)
    success = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, randomSeed=random_seed)
    if not success:
        raise RuntimeError("Conformer embedding failed.")

    # Step 4: Optimize geometry
    for conf_id in range(mol.GetNumConformers()):
        AllChem.UFFOptimizeMolecule(mol, confId=conf_id)

    return mol

def check_mol_stable(mol):
    atoms = mol.GetAtoms()
    atom_validities = []
    for atom in atoms:
        symbol = atom.GetSymbol()
        valence = atom.GetExplicitValence()
        charge = atom.GetFormalCharge()
        expected = Chem.GetPeriodicTable().GetDefaultValence(symbol)
        deficit = valence - expected
        atom_valid = (deficit - charge == 0)
        atom_validities.append(atom_valid)
    if all(atom_validities):
        return True
    return False

def zero_formal_charges(mol):
    mol = Chem.RWMol(mol)  # Make editable copy
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0:
            atom.SetFormalCharge(0)
    # Chem.SanitizeMol(mol)  # Recompute properties
    return mol

def fix_carbon_valency(mol):
    mol = Chem.RWMol(mol)  # Make editable

    # Loop over all bonds to find C–C bonds
    bonds_to_remove = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()

        if a1.GetAtomicNum() == 6 and a2.GetAtomicNum() == 6:
            valence1 = a1.GetExplicitValence()
            valence2 = a2.GetExplicitValence()

            if valence1 > 4 and valence2 > 4:
                bonds_to_remove.append((a1.GetIdx(), a2.GetIdx()))

    # Remove problematic bonds
    for idx1, idx2 in bonds_to_remove:
        mol.RemoveBond(idx1, idx2)

    mol.UpdatePropertyCache(strict=False)
    return mol

# Fix the aromatic bonds autotection in RDKit
def fix_valency(mol):
    mol = Chem.RWMol(mol)  # Editable copy

    # Step 1: Clear all aromaticity flags and convert to single bonds
    for bond in mol.GetBonds():
        bond.SetIsAromatic(False)
        bond.SetBondType(Chem.BondType.SINGLE)

    for atom in mol.GetAtoms():
        atom.SetIsAromatic(False)
        atom.SetNoImplicit(True)  # Remove implicit H handling
        atom.SetNumExplicitHs(0)  # Remove any implicit hydrogens

    # Step 2: Manually adjust bond orders if needed
    # For example, check atom 3's neighbors and adjust
    atom = mol.GetAtomWithIdx(2)  # Atom index 2 is atom #3
    if atom.GetExplicitValence() > atom.GetTotalValence():
        # Loop through bonds and downgrade one bond
        for bond in atom.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                bond.SetBondType(Chem.BondType.SINGLE)
                break

    # Step 3: Sanitize molecule (skip aromaticity again)
    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)

    return mol

def fix_nitrogen_valency(mol):
    mol = Chem.RWMol(mol)  # Editable molecule
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 7:  # Skip if not nitrogen
            continue

        neighbors = atom.GetNeighbors()
        if len(neighbors) != 2:
            continue

        # Count nitrogen's explicit valence (sum bond orders)
        n_valence = sum(
            mol.GetBondBetweenAtoms(atom.GetIdx(), nbr.GetIdx()).GetBondTypeAsDouble()
            for nbr in neighbors
        )

        if n_valence != 2:
            continue  # Skip if nitrogen valence is not 2

        # Search for carbon neighbor with valence 3
        for neighbor in neighbors:
            if neighbor.GetAtomicNum() != 6:
                continue

            bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            if bond.GetBondType() != Chem.BondType.SINGLE:
                continue

            # Count carbon's explicit valence
            c_valence = sum(
                mol.GetBondBetweenAtoms(neighbor.GetIdx(), n2.GetIdx()).GetBondTypeAsDouble()
                for n2 in neighbor.GetNeighbors()
            )

            if c_valence == 3:
                # Replace N–C single bond with double bond
                mol.RemoveBond(atom.GetIdx(), neighbor.GetIdx())
                mol.AddBond(atom.GetIdx(), neighbor.GetIdx(), Chem.BondType.DOUBLE)
                break  # Fix only one such bond per N atom

    mol.UpdatePropertyCache(strict=False)
    return mol

def check_valency_charge_balance(file_path):
    suppl = Chem.SDMolSupplier(file_path, removeHs=False, sanitize=False)
    
    valid_mol_indices = []
    valid_mols = []
    valid_mol_count = 0
    valid_atom_count = 0
    total_mol_count = 0
    total_atom_count = 0

    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        total_mol_count += 1
        atoms = mol.GetAtoms()
        total_atom_count += len(atoms)

        atom_validities = []
        for atom in atoms:
            symbol = atom.GetSymbol()
            valence = atom.GetExplicitValence()
            charge = atom.GetFormalCharge()
            expected = Chem.GetPeriodicTable().GetDefaultValence(symbol)
            deficit = valence - expected
            atom_valid = (deficit - charge == 0)
            atom_validities.append(atom_valid)
        
        valid_atom_count += sum(atom_validities)
        if all(atom_validities):
            valid_mol_count += 1
            valid_mol_indices.append(i)
            valid_mols.append(mol)
    
    atom_ratio = valid_atom_count / total_atom_count if total_atom_count > 0 else 0.0
    mol_ratio = valid_mol_count / total_mol_count if total_mol_count > 0 else 0.0

    return valid_mol_indices, mol_ratio, atom_ratio, valid_mols

def get_rdkit_valid(sdf_file, n_mols=1000000):
    from rdkit import RDLogger, Chem
    # Suppress RDKit warnings
    RDLogger.DisableLog('rdApp.*')
    suppl = Chem.SDMolSupplier(sdf_file, removeHs=False, sanitize=True)
    suppl = [_ for i, _ in enumerate(suppl) if i < n_mols]
    sdf_valid_index = [i for i, mol in enumerate(suppl) if mol is not None]
    return sdf_valid_index, suppl

def count_multifragment_molecules(sdf_path):
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
    count = 0
    multi_frag_indices = []

    for i, mol in enumerate(supplier):
        if mol is None:
            continue
        frags = Chem.GetMolFrags(mol, asMols=False)
        if len(frags) > 1:
            count += 1
            multi_frag_indices.append(i)

    return count, multi_frag_indices

from rdkit.Chem import AllChem
def compute_mmff_energy(smiles: str):
    """Compute MMFF94 energy of a molecule given SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
        return None
    if AllChem.MMFFHasAllMoleculeParams(mol):
        ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol))
        return ff.CalcEnergy()
    return None