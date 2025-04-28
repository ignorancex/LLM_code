from typing import Any, Dict, List, Optional, Tuple, Union
from typing_extensions import Self

import copy
from datetime import datetime
import numpy as np
import os
import pickle
from rdkit import Chem, DataStructs, RDLogger
RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.AllChem import RWMol
import re

from open_biomed.core.tool import Tool
from open_biomed.data.text import Text
from open_biomed.utils.exception import MoleculeConstructError

class Molecule:
    def __init__(self) -> None:
        super().__init__()
        self.name = None

        # basic properties: 1D SMILES/SELFIES strings, RDKit mol object, 2D graphs, 3D coordinates
        self.smiles = None
        self.selfies = None
        self.rdmol = None
        self.graph = None
        self.conformer = None

        # other related properties: image, textual descriptions and identifier in knowledge graph
        self.img = None
        self.description = None
        self.kg_accession = None

    @classmethod
    def from_smiles(cls, smiles: str) -> Self:
        # initialize a molecule with a SMILES string
        molecule = cls()
        molecule.smiles = smiles
        # molecule._add_rdmol(base="smiles")
        return molecule

    @classmethod
    def from_selfies(cls, selfies: str) -> Self:
        import selfies as sf
        molecule = cls()
        molecule.selfies = selfies
        molecule.smiles = sf.decoder(selfies)
        return molecule

    @classmethod
    def from_rdmol(cls, rdmol: RWMol) -> Self:
        # initialize a molecule with a RDKit molecule
        molecule = cls()
        molecule.rdmol = rdmol
        molecule.smiles = Chem.MolToSmiles(rdmol)
        conformer = rdmol.GetConformer()
        if conformer is not None:
            molecule.conformer = np.array(conformer.GetPositions())
        return molecule

    @classmethod
    def from_pdb_file(cls, pdb_file: str) -> Self:
        # initialize a molecule with a pdb file
        pass

    @classmethod
    def from_sdf_file(cls, sdf_file: str) -> Self:
        # initialize a molecule with a sdf file
        loader = Chem.SDMolSupplier(sdf_file)
        for mol in loader:
            if mol is not None:
                molecule = Molecule.from_rdmol(mol)
                conformer = mol.GetConformer()
                molecule.conformer = np.array(conformer.GetPositions())
        molecule.name = sdf_file.split("/")[-1].strip(".sdf")
        return molecule

    @classmethod
    def from_image_file(cls, image_file: str) -> Self:
        # initialize a molecule with a image file
        pass

    @classmethod
    def from_binary_file(cls, file: str) -> Self:
        return pickle.load(open(file, "rb"))

    @staticmethod
    def convert_smiles_to_rdmol(smiles: str, canonical: bool=True) -> RWMol:
        # Convert the smiles string into rdkit mol
        # If the smiles is invalid, raise MolConstructError
        pass

    @staticmethod
    def generate_conformer(rdmol: RWMol, method: str='mmff', num_conformers: int=1) -> np.ndarray:
        # Generate 3D conformer with algorithms in RDKit
        # TODO: identify if ML-based conformer generation can be applied
        pass

    def _add_name(self) -> None:
        if self.name is None:
            self.name = "mol_" + re.sub(r"[-:.]", "_", datetime.now().isoformat(sep="_", timespec="milliseconds"))

    def _add_smiles(self, base: str='rdmol') -> None:
        # Add class property: smiles, based on selfies / rdmol / graph, default: rdmol
        pass

    def _add_selfies(self, base: str='smiles') -> None:
        import selfies as sf
        # Add class property: selfies, based on smiles / selfies / rdmol / graph, default: smiles
        if base == "smiles":
            self.selfies = sf.encoder(self.smiles, strict=False)
        else:
            raise NotImplementedError

    def _add_rdmol(self, base: str='smiles') -> None:
        # Add class property: rdmol, based on smiles / selfies / graph, default: smiles
        if self.rdmol is not None:
            return
        if base == 'smiles':
            self.rdmol = Chem.MolFromSmiles(self.smiles)
        if self.conformer is not None:
            conf = mol_array_to_conformer(self.conformer)
            self.rdmol.AddConformer(conf)

    def _add_conformer(self, mode: str='2D', base: str='rdmol') -> None:
        # Add class property: conformer, based on smiles / selfies / rdmol, default: rdmol
        if self.conformer is None:
            self._add_rdmol()
            if mode == '2D':
                AllChem.Compute2DCoords(self.rdmol)
            elif mode == '3D':
                self.rdmol = Chem.AddHs(self.rdmol)
                AllChem.EmbedMolecule(self.rdmol)
                AllChem.MMFFOptimizeMolecule(self.rdmol)
            conformer = self.rdmol.GetConformer()
            self.conformer = np.array(conformer.GetPositions())
    
    def _add_description(self, text_database: Dict[Any, Text], identifier_key: str='SMILES', base: str='smiles') -> None:
        # Add class property: description, based on smiles / selfies / rdmol, default: smiles
        pass

    def _add_kg_accession(self, kg_database: Dict[Any, str], identifier_key: str='SMILES', base: str='smiles') -> None:
        # Add class property: kg_accession, based on smiles / selfies / rdmol, default: smiles
        pass

    def save_sdf(self, file: Optional[str]=None, overwrite: bool=False) -> str:
        if file is None:
            self._add_name()
            file = f"./tmp/{self.name}.sdf"

        if not os.path.exists(file) or overwrite:
            writer = Chem.SDWriter(file)
            self._add_rdmol()
            self._add_conformer()
            writer.write(self.rdmol)
        return file

    def save_binary(self, file: Optional[str]=None, overwrite: bool=False) -> str:
        if file is None:
            self._add_name()
            file = f"./tmp/{self.name}.pkl"

        if not os.path.exists(file) or overwrite:
            pickle.dump(self, open(file, "wb"))
        return file

    def get_num_atoms(self) -> None:
        self._add_rdmol()
        return self.rdmol.GetNumAtoms()

    def __str__(self) -> str:
        return self.smiles

def molecule_fingerprint_similarity(mol1: Molecule, mol2: Molecule, fingerprint_type: str="morgan") -> float:
    # Calculate the fingerprint similarity of two molecules
    try:
        mol1._add_rdmol()
        mol2._add_rdmol()
        if fingerprint_type == "morgan":
            fp1 = AllChem.GetMorganFingerprint(mol1.rdmol, 2)
            fp2 = AllChem.GetMorganFingerprint(mol2.rdmol, 2)
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        if fingerprint_type == "rdkit":
            fp1 = Chem.RDKFingerprint(mol1.rdmol)
            fp2 = Chem.RDKFingerprint(mol2.rdmol)
        if fingerprint_type == "maccs":
            fp1 = MACCSkeys.GenMACCSKeys(mol1.rdmol)
            fp2 = MACCSkeys.GenMACCSKeys(mol2.rdmol)
        return DataStructs.FingerprintSimilarity(
            fp1, fp2,
            metric=DataStructs.TanimotoSimilarity
        )
    except Exception:
        return 0.0

def check_identical_molecules(mol1: Molecule, mol2: Molecule) -> bool:
    # Check if the two molecules are the same
    try:
        mol1._add_rdmol()
        mol2._add_rdmol()
        return Chem.MolToInchi(mol1.rdmol) == Chem.MolToInchi(mol2.rdmol)
    except Exception:
        return False

def fix_valence(mol: Chem.RWMol) -> Tuple[Chem.RWMol, bool]:
    # Fix valence erros in a molecule by adding electrons to N
    mol = copy.deepcopy(mol)
    fixed = False
    cnt_loop = 0
    while True:
        try:
            Chem.SanitizeMol(copy.deepcopy(mol))
            fixed = True
            Chem.SanitizeMol(mol)
            break
        except Chem.rdchem.AtomValenceException as e:
            err = e
        except Exception as e:
            return mol, False # from HERE: rerun sample
        cnt_loop += 1
        if cnt_loop > 100:
            break
        N4_valence = re.compile(u"Explicit valence for atom # ([0-9]{1,}) N, 4, is greater than permitted")
        index = N4_valence.findall(err.args[0])
        if len(index) > 0:
            mol.GetAtomWithIdx(int(index[0])).SetFormalCharge(1)
    return mol, fixed

def mol_array_to_conformer(conf: np.ndarray) -> Chem.Conformer:
    new_conf = Chem.Conformer(conf.shape[0])
    for i in range(conf.shape[0]):
        new_conf.SetAtomPosition(i, tuple(conf[i]))
    return new_conf
