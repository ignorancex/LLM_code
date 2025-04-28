from tdc.generation import MolGen 
from tdc.chem_utils import MolConvert
from tdc.chem_utils.oracle.oracle import smiles_to_rdkit_mol, qed, penalized_logp, jnk3, zaleplon_mpo, Vina_smiles, Vina_3d
from torch_geometric.data import Batch
import time
import pickle
from rdkit import Chem
converter = MolConvert(src = 'SMILES', dst = 'SELFIES')
inverter = MolConvert(src='SELFIES', dst = 'SMILES')

splits = {
    "train" : "/home/khangnn4/ligand_generation/data/timesplit_no_lig_or_rec_overlap_train", 
    "valid" : "/home/khangnn4/ligand_generation/data/timesplit_no_lig_or_rec_overlap_val", 
    "test" : "/home/khangnn4/ligand_generation/data/timesplit_test"
}

print("loading all data")
with open("complex_2_smiles.pkl", "rb") as f:
    smiles_dict = pickle.load(f)

invalid = {}

for idx, complex_name in enumerate(smiles_dict.keys()):
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles_dict[complex_name]))
    try:
        converter(smiles)
    except:
        invalid[complex_name] = smiles_dict[complex_name]
    print(idx)

with open("invalid.pkl", "wb") as file:
    pickle.dump(invalid, file)
print(len(invalid))
# for key in invalid.keys():
#     print(invalid[key])