from typing import Tuple, Union, Any, Dict, Optional, List
from typing_extensions import Self

import json
import os
from torch.utils.data import Dataset

from open_biomed.data import Molecule, Text
from open_biomed.datasets.base_dataset import BaseDataset, assign_split, featurize
from open_biomed.utils.config import Config
from open_biomed.utils.featurizer import Featurizer, Featurized

class MoleculeQADataset(BaseDataset):
    def __init__(self, cfg: Config, featurizer: Featurizer) -> None:
        self.molecules, self.texts, self.labels = [], [], []
        super(MoleculeQADataset, self).__init__(cfg, featurizer)

    def __len__(self) -> int:
        return len(self.texts)

    @featurize
    def __getitem__(self, index) -> Dict[str, Featurized[Any]]:
        return {
            "molecule": self.molecules[index],
            "text": self.texts[index], 
            "label": self.labels[index],
        }

class MQA(MoleculeQADataset):
    def __init__(self, cfg: Config, featurizer: Featurizer) -> None:
        super(MQA, self).__init__(cfg, featurizer)

    def _load_data(self) -> None:
        self.split_indexes = {}
        cnt = 0
        for split in ["train", "val", "test"]:
            self.split_indexes[split] = []
            with open(os.path.join(self.cfg.path, f"{split}.json"), "r") as f:
                cur = 0
                sample_list = json.loads(f.readlines()[0])
                for sample in sample_list:
                    # Getting the length of a list could be slow
                    if len(sample["smiles"]) > 1:
                        continue
                    self.split_indexes[split].append(cnt)
                    cnt += 1
                    cur += 1

                    self.molecules.append(Molecule.from_smiles(sample["smiles"][0]))
                    self.texts.append(Text.from_str(sample["question"]))
                    self.labels.append(Text.from_str(sample["answer"]))
                    if (split != "train" and cur >= 50 or cur >= 5000) and self.cfg.debug:
                        break
        
    @assign_split
    def split(self, split_cfg: Optional[Config] = None) -> Tuple[Any, Any, Any]:
        attrs = ["molecules", "texts", "labels"]
        ret = (
            self.get_subset(self.split_indexes["train"], attrs), 
            self.get_subset(self.split_indexes["val"], attrs),
            self.get_subset(self.split_indexes["test"], attrs),
        )
        del self
        return ret
