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

class TextGuidedMoleculeGenerationDataset(BaseDataset):
    def __init__(self, cfg: Config, featurizer: Featurizer) -> None:
        self.texts, self.labels = [], []
        super(TextGuidedMoleculeGenerationDataset, self).__init__(cfg, featurizer)

    def __len__(self) -> int:
        return len(self.texts)

    @featurize
    def __getitem__(self, index) -> Dict[str, Featurized[Any]]:
        return {
            "text": self.texts[index], 
            "label": self.labels[index],
        }

class TextGuidedMoleculeGenerationEvalDataset(Dataset):
    def __init__(self) -> None:
        super(TextGuidedMoleculeGenerationEvalDataset, self).__init__()
        self.texts, self.labels = [], []

    @classmethod
    def from_train_set(cls, dataset: TextGuidedMoleculeGenerationDataset) -> Self:
        mol2label = dict()
        for i in range(len(dataset)):
            text = dataset.texts[i]
            label = dataset.labels[i]
            dict_key = str(text)
            if dict_key not in mol2label:
                mol2label[dict_key] = []
            mol2label[dict_key].append((text, label))
        new_dataset = cls()
        for k, v in mol2label.items():
            new_dataset.texts.append(v[0][0])
            new_dataset.labels.append(v[0][1])
        new_dataset.featurizer = dataset.featurizer
        return new_dataset

    def __len__(self) -> int:
        return len(self.texts)

    @featurize
    def __getitem__(self, index) -> Dict[str, Featurized[Any]]:
        return {
            "text": self.texts[index], 
            "label": self.labels[index],
        }
    
    def get_labels(self) -> List[List[Molecule]]:
        return self.labels

class CheBI20ForText2Mol(TextGuidedMoleculeGenerationDataset):
    def __init__(self, cfg: Config, featurizer: Featurizer) -> None:
        super(CheBI20ForText2Mol, self).__init__(cfg, featurizer)

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

                    self.texts.append(Text.from_str(line["description"]))
                    self.labels.append(Molecule.from_smiles(line["SMILES"]))
                    if cur >= 5 and self.cfg.debug:
                        break
        
    @assign_split
    def split(self, split_cfg: Optional[Config] = None) -> Tuple[Any, Any, Any]:
        attrs = ["texts", "labels"]
        ret = (
            self.get_subset(self.split_indexes["train"], attrs), 
            TextGuidedMoleculeGenerationEvalDataset.from_train_set(self.get_subset(self.split_indexes["validation"], attrs)),
            TextGuidedMoleculeGenerationEvalDataset.from_train_set(self.get_subset(self.split_indexes["test"], attrs)),
        )
        del self
        return ret
