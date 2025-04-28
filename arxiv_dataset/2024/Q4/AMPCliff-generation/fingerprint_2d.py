import json
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

class Fingerprint_Generation:
    def __init__(self, smiles_file, nbits, radius):
        self.nbits = nbits
        self.radius = radius
        self.smiles_to_mol = self._load_smiles(smiles_file)

    def _load_smiles(self, smiles_file):
        """load JSON of SMILESand transfer into RDKit mol dict"""
        with open(smiles_file, 'r') as f:
            smiles_dict = json.load(f)
        smiles_to_mol = {}
        for key, smi in smiles_dict.items():
            mol = Chem.MolFromSmiles(smi)
            if mol:  # ensure the availability of mol
                smiles_to_mol[key] = mol
        return smiles_to_mol

    def seq(self, sequence):
        """generate fingerprint according to the sequence"""
        # process every char
        fps = []
        for char in sequence:
            mol = self.smiles_to_mol.get(char)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.nbits)
                # transfer RDKit ExplicitBitVect into NumPy array
                arr = np.zeros((1,))
                DataStructs.ConvertToNumpyArray(fp, arr)
                fps.append(arr)
            else:
                # add zero vector if the key not exist
                fps.append(np.zeros((self.nbits,)))
        # concate all of the fingerprint
        fp_seq = np.array(fps)
        return fp_seq