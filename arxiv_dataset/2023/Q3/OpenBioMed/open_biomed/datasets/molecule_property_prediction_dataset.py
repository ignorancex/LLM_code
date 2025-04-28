from typing import Tuple, Union, Any, Dict, Optional, List
from typing_extensions import Self
from enum import Enum

import json
import os
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
from rdkit import Chem
from torch.utils.data import Dataset
from rdkit.Chem import AllChem, Descriptors

from open_biomed.data import Molecule, Text
from open_biomed.datasets.base_dataset import BaseDataset, assign_split, featurize
from open_biomed.utils.config import Config
from open_biomed.utils.featurizer import Featurizer, Featurized
from open_biomed.utils.split_utils import random_split, scaffold_split

class MoleculePropertyPredictionDataset(BaseDataset):
    def __init__(self, cfg: Config, featurizer: Featurizer) -> None:
        self.molecules, self.texts, self.labels = [], [], []
        super(MoleculePropertyPredictionDataset, self).__init__(cfg, featurizer)

    def __len__(self) -> int:
        return len(self.molecules)

    @featurize
    def __getitem__(self, index) -> Dict[str, Featurized[Any]]:
        return {
            "molecule": self.molecules[index], 
            "label": self.labels[index],
        }

class MoleculePropertyPredictionEvalDataset(Dataset):
    def __init__(self) -> None:
        super(MoleculePropertyPredictionEvalDataset, self).__init__()
        self.molecules, self.texts, self.labels = [], [], []

    @classmethod
    def from_train_set(cls, dataset: MoleculePropertyPredictionDataset) -> Self:
        # NOTE: 
        # Given the same original molecule and text, multiple results are acceptable
        # We combine these results for evaluation
        mol2label = dict()
        for i in range(len(dataset)):
            molecule = dataset.molecules[i]
            label = dataset.labels[i]
            dict_key = str(molecule)
            if dict_key not in mol2label:
                mol2label[dict_key] = []
            mol2label[dict_key].append((molecule, label))
        new_dataset = cls()
        for k, v in mol2label.items():
            new_dataset.molecules.append(v[0][0])
            new_dataset.labels.append([x[1] for x in v])
        new_dataset.featurizer = dataset.featurizer
        return new_dataset

    def __len__(self) -> int:
        return len(self.molecules)

    @featurize
    def __getitem__(self, index) -> Dict[str, Featurized[Any]]:
        return {
            "molecule": self.molecules[index], 
            "label": self.labels[index],
        }
    
    def get_labels(self) -> List[List[Text]]:
        return self.labels


def _load_bbbp_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    preprocessed_rdkit_mol_objs_list = [m if m is not None else None
                                        for m in rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m is not None else None
                                for m in preprocessed_rdkit_mol_objs_list]
    labels = input_df['p_np']
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, \
           preprocessed_rdkit_mol_objs_list, labels.values


def _load_clintox_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    preprocessed_rdkit_mol_objs_list = [m if m is not None else None
                                        for m in rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m is not None else None
                                for m in preprocessed_rdkit_mol_objs_list]
    tasks = ['FDA_APPROVED', 'CT_TOX']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, \
           preprocessed_rdkit_mol_objs_list, labels.values


# input_path = 'dataset/clintox/raw/clintox.csv'
# smiles_list, rdkit_mol_objs_list, labels = _load_clintox_dataset(input_path)

def _load_esol_dataset(input_path):
    # NB: some examples have multiple species
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['measured log solubility in mols per litre']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


# input_path = 'dataset/esol/raw/delaney-processed.csv'
# smiles_list, rdkit_mol_objs_list, labels = _load_esol_dataset(input_path)

def _load_freesolv_dataset(input_path):

    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['expt']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_lipophilicity_dataset(input_path):

    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['exp']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_malaria_dataset(input_path):

    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['activity']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_cep_dataset(input_path):

    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['PCE']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_muv_dataset(input_path):

    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689',
             'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810',
             'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def check_columns(df, tasks, N):
    bad_tasks = []
    total_missing_count = 0
    for task in tasks:
        value_list = df[task]
        pos_count = sum(value_list == 1)
        neg_count = sum(value_list == -1)
        missing_count = sum(value_list == 0)
        total_missing_count += missing_count
        pos_ratio = 100. * pos_count / (pos_count + neg_count)
        missing_ratio = 100. * missing_count / N
        assert pos_count + neg_count + missing_count == N
        if missing_ratio >= 50:
            bad_tasks.append(task)
        print('task {}\t\tpos_ratio: {:.5f}\tmissing ratio: {:.5f}'.format(task, pos_ratio, missing_ratio))
    print('total missing ratio: {:.5f}'.format(100. * total_missing_count / len(tasks) / N))
    return bad_tasks


def check_rows(labels, N):
    from collections import defaultdict
    p, n, m = defaultdict(int), defaultdict(int), defaultdict(int)
    bad_count = 0
    for i in range(N):
        value_list = labels[i]
        pos_count = sum(value_list == 1)
        neg_count = sum(value_list == -1)
        missing_count = sum(value_list == 0)
        p[pos_count] += 1
        n[neg_count] += 1
        m[missing_count] += 1
        if pos_count + neg_count == 0:
            bad_count += 1
    print('bad_count\t', bad_count)
    
    print('pos\t', p)
    print('neg\t', n)
    print('missing\t', m)
    return


def _load_pcba_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=',')
    tasks = list(input_df.columns)[:-2]

    N = input_df.shape[0]
    temp_df = input_df[tasks]
    temp_df = temp_df.replace(0, -1)
    temp_df = temp_df.fillna(0)

    bad_tasks = check_columns(temp_df, tasks, N)
    for task in bad_tasks:
        tasks.remove(task)
    print('good tasks\t', len(tasks))

    labels = input_df[tasks]
    labels = labels.replace(0, -1)
    labels = labels.fillna(0)
    labels = labels.values
    print(labels.shape)  # 439863, 92
    check_rows(labels, N)

    input_df.dropna(subset=tasks, how='all', inplace=True)
    # convert 0 to -1
    # input_df = input_df.replace(0, -1)
    # convert nan to 0
    input_df = input_df.fillna(0)
    labels = input_df[tasks].values
    print(input_df.shape)  # 435685, 92
    N = input_df.shape[0]
    check_rows(labels, N)

    smiles_list = input_df['smiles'].tolist()
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels


def _load_sider_dataset(input_path):

    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['Hepatobiliary disorders',
             'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders',
             'Investigations', 'Musculoskeletal and connective tissue disorders',
             'Gastrointestinal disorders', 'Social circumstances',
             'Immune system disorders', 'Reproductive system and breast disorders',
             'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
             'General disorders and administration site conditions',
             'Endocrine disorders', 'Surgical and medical procedures',
             'Vascular disorders', 'Blood and lymphatic system disorders',
             'Skin and subcutaneous tissue disorders',
             'Congenital, familial and genetic disorders',
             'Infections and infestations',
             'Respiratory, thoracic and mediastinal disorders',
             'Psychiatric disorders', 'Renal and urinary disorders',
             'Pregnancy, puerperium and perinatal conditions',
             'Ear and labyrinth disorders', 'Cardiac disorders',
             'Nervous system disorders',
             'Injury, poisoning and procedural complications']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_toxcast_dataset(input_path):

    # NB: some examples have multiple species, some example smiles are invalid
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    # Some smiles could not be successfully converted
    # to rdkit mol object so them to None
    preprocessed_rdkit_mol_objs_list = [m if m is not None else None
                                        for m in rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m is not None else None
                                for m in preprocessed_rdkit_mol_objs_list]
    tasks = list(input_df.columns)[1:]
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, \
           preprocessed_rdkit_mol_objs_list, labels.values

# root_path = 'dataset/chembl_with_labels'
def check_smiles_validity(smiles):
    try:
        m = Chem.MolFromSmiles(smiles)
        if m:
            return True
        else:
            return False
    except:
        return False


def split_rdkit_mol_obj(mol):
    """
    Split rdkit mol object containing multiple species or one species into a
    list of mol objects or a list containing a single object respectively """

    smiles = AllChem.MolToSmiles(mol, isomericSmiles=True)
    smiles_list = smiles.split('.')
    mol_species_list = []
    for s in smiles_list:
        if check_smiles_validity(s):
            mol_species_list.append(AllChem.MolFromSmiles(s))
    return mol_species_list

def get_largest_mol(mol_list):
    """
    Given a list of rdkit mol objects, returns mol object containing the
    largest num of atoms. If multiple containing largest num of atoms,
    picks the first one """

    num_atoms_list = [len(m.GetAtoms()) for m in mol_list]
    largest_mol_idx = num_atoms_list.index(max(num_atoms_list))
    return mol_list[largest_mol_idx]

def _load_tox21_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
             'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_hiv_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['HIV_active']
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values

def _load_bace_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['Class']
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    folds = input_df['Model']
    folds = folds.replace('Train', 0)  # 0 -> train
    folds = folds.replace('Valid', 1)  # 1 -> valid
    folds = folds.replace('Test', 2)  # 2 -> test
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    assert len(smiles_list) == len(folds)
    # return smiles_list, rdkit_mol_objs_list, folds.values, labels.values
    return smiles_list, rdkit_mol_objs_list, labels.values


datasetname2function = {
    "bbbp": _load_bbbp_dataset,
    "clintox": _load_clintox_dataset,
    "tox21": _load_tox21_dataset,
    "toxcast": _load_toxcast_dataset,
    "sider": _load_sider_dataset,
    "hiv": _load_hiv_dataset,
    "bace": _load_bace_dataset,
    "muv": _load_muv_dataset,
    "freesolv": _load_freesolv_dataset,
    "esol": _load_esol_dataset,
    "lipophilicity": _load_lipophilicity_dataset,
}

class Task(Enum):
    CLASSFICATION = 0
    REGRESSION = 1


class MoleculeNet(MoleculePropertyPredictionDataset):

    name2target = {
          "BBBP":     ["p_np"],
          "Tox21":    ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", 
                      "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"],
          "ClinTox":  ["CT_TOX", "FDA_APPROVED"],
          "HIV":      ["HIV_active"],
          "Bace":     ["class"],
          "SIDER":    ["Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues", 
                      "Eye disorders", "Investigations", "Musculoskeletal and connective tissue disorders", 
                      "Gastrointestinal disorders", "Social circumstances", "Immune system disorders", 
                      "Reproductive system and breast disorders", 
                      "Neoplasms benign, malignant and unspecified (incl cysts and polyps)", 
                      "General disorders and administration site conditions", "Endocrine disorders", 
                      "Surgical and medical procedures", "Vascular disorders", 
                      "Blood and lymphatic system disorders", "Skin and subcutaneous tissue disorders", 
                      "Congenital, familial and genetic disorders", "Infections and infestations", 
                      "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders", 
                      "Renal and urinary disorders", "Pregnancy, puerperium and perinatal conditions", 
                      "Ear and labyrinth disorders", "Cardiac disorders", 
                      "Nervous system disorders", "Injury, poisoning and procedural complications"],
          "MUV":      ['MUV-692', 'MUV-689', 'MUV-846', 'MUV-859', 'MUV-644', 'MUV-548', 'MUV-852',
                      'MUV-600', 'MUV-810', 'MUV-712', 'MUV-737', 'MUV-858', 'MUV-713', 'MUV-733',
                      'MUV-652', 'MUV-466', 'MUV-832'],
          "Toxcast":  [""], # 617
          "FreeSolv": ["expt"],
          "ESOL":     ["measured log solubility in mols per litre"],
          "Lipo":     ["exp"],
          "qm7":      ["u0_atom"],
          "qm8":      ["E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0", 
                      "f1-PBE0", "f2-PBE0", "E1-CAM", "E2-CAM", "f1-CAM","f2-CAM"],
          "qm9":      ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'cv']
      }
    name2task = {
          "BBBP":     Task.CLASSFICATION,
          "Tox21":    Task.CLASSFICATION,
          "ClinTox":  Task.CLASSFICATION,
          "HIV":      Task.CLASSFICATION,
          "Bace":     Task.CLASSFICATION,
          "SIDER":    Task.CLASSFICATION,
          "MUV":      Task.CLASSFICATION,
          "Toxcast":  Task.CLASSFICATION,
          "FreeSolv": Task.REGRESSION,
          "ESOL":     Task.REGRESSION,
          "Lipo":     Task.REGRESSION,
          "qm7":      Task.REGRESSION,
          "qm8":      Task.REGRESSION,
          "qm9":      Task.REGRESSION
      }

    name2text = {
      "bbbp": "Binary labels of blood-brain barrier penetration(permeability)",
      "clintox": "Qualitative data of drugs that failed clinical trials for toxicity reasons",
      "tox21": "Qualitative toxicity measurements including nuclear receptors and stress response pathways",
      "toxcast": "Compounds based on in vitro high-throughput screening",
      "sider": "marketed drugs and adverse drug reactions (ADR), grouped into 27 system organ classes",
      "hiv": "Experimentally measured abilities to inhibit HIV replication",
      "bace": "Quantitative (IC50) and qualitative (binary label) binding results for human β-secretase 1(BACE-1)",
      "muv": "Subset of PubChem BioAssay designed for validation of virtual screening techniques",
    }
    def __init__(self, cfg: Config, featurizer: Featurizer) -> None:
        self.name = cfg.name
        self.targets = self.name2target[cfg.name]
        # TODO: 看一下graphmvp这里是干什么用的，后续上regression任务的时候要考虑
        self.task = self.name2task[cfg.name]
        file_name = os.listdir(os.path.join(cfg.path, self.name.lower(), "raw"))[0]
        assert file_name[-4:] == ".csv"
        self.path = os.path.join(cfg.path, self.name.lower(), "raw", file_name)
        super(MoleculeNet, self).__init__(cfg, featurizer)

    def _train_test_split(self, strategy="scaffold"):
        if strategy == "random":
            self.train_index, self.validation_index, self.test_index = random_split(len(self), 0.1, 0.1)
        elif strategy == "scaffold":
            self.train_index, self.validation_index, self.test_index = scaffold_split(self, 0.1, 0.1, is_standard=True)

    def _normalize(self):
        if self.name in ["qm7", "qm9"]:
            self.normalizer = []
            for i in range(len(self.targets)):
                self.normalizer.append(Normalizer(self.labels[:, i]))
                self.labels[:, i] = self.normalizer[i].norm(self.labels[:, i])
        else:
            # TODO:
            self.normalizer = [None] * len(self.targets)

    def _load_data(self) -> None:
        smiles_list, rdkit_mol_objs, labels = datasetname2function[self.name.lower()](self.path)
        if labels.ndim == 1:
            labels = np.expand_dims(labels, axis=1)
        # TODO: 
        for i in range(len(smiles_list)):
            rdkit_mol = rdkit_mol_objs[i]
            if rdkit_mol is None:
                continue
            # TODO: drugs and smiles are all get from AllChem.MolFromSmiles()
            #self.smiles.append(smiles_list[i])
            self.molecules.append(Molecule.from_smiles(smiles_list[i]))
            self.labels.append(labels[i])
        self._train_test_split()
        self._normalize()
        self.split_indexes = {}
        for split in ["train", "validation", "test"]:
             self.split_indexes[split] = getattr(self, f"{split}_index")
        
    @assign_split
    def split(self, split_cfg: Optional[Config] = None) -> Tuple[Any, Any, Any]:
        attrs = ["molecules", "labels"]
        ret = (
            self.get_subset(self.split_indexes["train"], attrs), 
            MoleculePropertyPredictionEvalDataset.from_train_set(self.get_subset(self.split_indexes["validation"], attrs)),
            MoleculePropertyPredictionEvalDataset.from_train_set(self.get_subset(self.split_indexes["test"], attrs)),
        )
        del self
        return ret
