from typing import Tuple, Union, Any, Dict, Optional, List
from typing_extensions import Self

import json
import os
from torch.utils.data import Dataset

from open_biomed.data import Molecule, Text
from open_biomed.datasets.base_dataset import BaseDataset, assign_split, featurize
from open_biomed.utils.config import Config
from open_biomed.utils.featurizer import Featurizer, Featurized

class TextBasedMoleculeEditingDataset(BaseDataset):
    def __init__(self, cfg: Config, featurizer: Featurizer) -> None:
        self.molecules, self.texts, self.labels = [], [], []
        super(TextBasedMoleculeEditingDataset, self).__init__(cfg, featurizer)

    def __len__(self) -> int:
        return len(self.molecules)

    @featurize
    def __getitem__(self, index) -> Dict[str, Featurized[Any]]:
        return {
            "molecule": self.molecules[index], 
            "text": self.texts[index], 
            "label": self.labels[index],
        }

class TextBasedMoleculeEditingEvalDataset(Dataset):
    def __init__(self) -> None:
        super(TextBasedMoleculeEditingEvalDataset, self).__init__()
        self.molecules, self.texts, self.labels = [], [], []

    @classmethod
    def from_train_set(cls, dataset: TextBasedMoleculeEditingDataset) -> Self:
        # NOTE: 
        # Given the same original molecule and text, multiple results are acceptable
        # We combine these results for evaluation
        mol2label = dict()
        for i in range(len(dataset)):
            molecule = dataset.molecules[i]
            text = dataset.texts[i]
            label = dataset.labels[i]
            dict_key = str(molecule) + "___" + str(text)
            if dict_key not in mol2label:
                mol2label[dict_key] = []
            mol2label[dict_key].append((molecule, text, label))
        new_dataset = cls()
        for k, v in mol2label.items():
            new_dataset.molecules.append(v[0][0])
            new_dataset.texts.append(v[0][1])
            new_dataset.labels.append([x[2] for x in v])
        new_dataset.featurizer = dataset.featurizer
        return new_dataset

    def __len__(self) -> int:
        return len(self.molecules)

    @featurize
    def __getitem__(self, index) -> Dict[str, Featurized[Any]]:
        return {
            "molecule": self.molecules[index], 
            "text": self.texts[index],
            "label": self.labels[index],
        }
    
    def get_labels(self) -> List[List[Text]]:
        return self.labels

class FSMolEdit(TextBasedMoleculeEditingDataset):
    def __init__(self, cfg: Config, featurizer: Featurizer) -> None:
        super(FSMolEdit, self).__init__(cfg, featurizer)

    def _load_data(self) -> None:
        self.split_indexes = {}
        cnt = 0
        for split in ["train", "valid", "test"]:
            self.split_indexes[split] = []
            with open(os.path.join(self.cfg.path, f"{split}.jsonl"), "r") as f:
                cur = 0
                for line in f.readlines():
                    # Getting the length of a list could be slow
                    self.split_indexes[split].append(cnt)
                    cnt += 1
                    cur += 1

                    sample = json.loads(line)
                    self.molecules.append(Molecule.from_smiles(sample["SMILES"]))
                    self.texts.append(Text.from_str(sample["text"]))
                    self.labels.append(Molecule.from_smiles(sample["SMILES2"]))
                    if cur >= 5 and self.cfg.debug:
                        break
        
    @assign_split
    def split(self, split_cfg: Optional[Config] = None) -> Tuple[Any, Any, Any]:
        attrs = ["molecules", "texts", "labels"]
        ret = (
            self.get_subset(self.split_indexes["train"], attrs), 
            TextBasedMoleculeEditingEvalDataset.from_train_set(self.get_subset(self.split_indexes["valid"], attrs)),
            TextBasedMoleculeEditingEvalDataset.from_train_set(self.get_subset(self.split_indexes["test"], attrs)),
        )
        del self
        return ret
