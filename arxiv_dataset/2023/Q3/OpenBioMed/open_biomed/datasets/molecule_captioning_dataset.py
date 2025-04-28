from typing import Tuple, Union, Any, Dict, Optional, List
from typing_extensions import Self

import json
import os
from torch.utils.data import Dataset

from open_biomed.data import Molecule, Text
from open_biomed.datasets.base_dataset import BaseDataset, assign_split, featurize
from open_biomed.utils.config import Config
from open_biomed.utils.featurizer import Featurizer, Featurized
import csv

class MoleculeCaptioningDataset(BaseDataset):
    def __init__(self, cfg: Config, featurizer: Featurizer) -> None:
        self.molecules, self.labels = [], []
        super(MoleculeCaptioningDataset, self).__init__(cfg, featurizer)

    def __len__(self) -> int:
        return len(self.molecules)

    @featurize
    def __getitem__(self, index) -> Dict[str, Featurized[Any]]:
        return {
            "molecule": self.molecules[index], 
            "label": self.labels[index],
        }

class MoleculeCaptioningEvalDataset(Dataset):
    def __init__(self) -> None:
        super(MoleculeCaptioningEvalDataset, self).__init__()
        self.molecules, self.labels = [], []

    @classmethod
    def from_train_set(cls, dataset: MoleculeCaptioningDataset) -> Self:
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
            new_dataset.labels.append(v[0][1])
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
    
    def get_labels(self) -> List[List[Molecule]]:
        return self.labels

class CheBI20ForMol2Text(MoleculeCaptioningDataset):
    def __init__(self, cfg: Config, featurizer: Featurizer) -> None:
        super(CheBI20ForMol2Text, self).__init__(cfg, featurizer)

    def _load_data(self) -> None:
        self.split_indexes = {}
        cnt = 0
        for split in ["train", "validation", "test"]:
            self.split_indexes[split] = []
            with open(os.path.join(self.cfg.path, f"{split}.txt"), "r") as f:
                reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
                cur = 0
                for line in reader:
                    # Getting the length of a list could be slow
                    self.split_indexes[split].append(cnt)
                    cnt += 1
                    cur += 1

                    self.molecules.append(Molecule.from_smiles(line["SMILES"]))
                    self.labels.append(Text.from_str(line["description"]))
                    if cur >= 5 and self.cfg.debug:
                        break
        
    @assign_split
    def split(self, split_cfg: Optional[Config] = None) -> Tuple[Any, Any, Any]:
        attrs = ["molecules", "labels"]
        ret = (
            self.get_subset(self.split_indexes["train"], attrs), 
            MoleculeCaptioningEvalDataset.from_train_set(self.get_subset(self.split_indexes["validation"], attrs)),
            MoleculeCaptioningEvalDataset.from_train_set(self.get_subset(self.split_indexes["test"], attrs)),
        )
        del self
        return ret
