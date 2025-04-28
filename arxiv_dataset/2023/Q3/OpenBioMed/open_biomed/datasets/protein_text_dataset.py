from typing import Any, Optional, Tuple
from typing_extensions import Self

import json
import os
import random

from open_biomed.data import Protein, Text
from open_biomed.datasets.base_dataset import BaseDataset, assign_split, featurize
from open_biomed.utils.config import Config
from open_biomed.utils.featurizer import Featurizer

class ProteinTextDataset(BaseDataset):
    def __init__(self, cfg: Config, featurizer: Featurizer) -> None:
        self.proteins, self.texts = [], []
        super().__init__(cfg, featurizer)

    def __len__(self) -> int:
        return len(self.proteins)

class TextBasedProteinGenerationDataset(ProteinTextDataset):
    def __init__(self, cfg: Config, featurizer: Featurizer) -> None:
        super().__init__(cfg, featurizer)

    def _load_data(self) -> None:
        self.labels = self.proteins

    @featurize
    def __getitem__(self, index) -> Any:
        return {
            "text": self.texts[index],
            "label": self.proteins[index],
        }

class MolInstructionsForProteinDesign(TextBasedProteinGenerationDataset):
    def __init__(self, cfg: Config, featurizer: Featurizer) -> None:
        super().__init__(cfg, featurizer)

    def _load_data(self) -> None:
        self.split_indexes = {"train": [], "valid": [], "test": []}
        data = json.load(open(os.path.join(self.cfg.path, "protein_design.json"), "r"))
        cnt = 0
        for i, sample in enumerate(data):
            seq = sample["output"].split("\n")[-2]
            if "X" in seq:
                continue
            self.texts.append(Text.from_str(sample["input"]))
            self.proteins.append(Protein.from_fasta(seq))
            if sample["metadata"]["split"] == "train":
                if self.cfg.debug:
                    self.split_indexes["train"].append(cnt)
                    if cnt >= 498:
                        self.split_indexes["valid"].append(cnt)
                elif random.randint(1, 100) > 95:
                    self.split_indexes["valid"].append(cnt)
                else:
                    self.split_indexes["train"].append(cnt)
            else:
                self.split_indexes[sample["metadata"]["split"]].append(cnt)
            cnt += 1
            if cnt >= 500 and self.cfg.debug:
                break
        # print(len(self.split_indexes["train"]), len(self.split_indexes["valid"]), len(self.split_indexes["test"]))
        super()._load_data()

    @assign_split
    def split(self, split_cfg: Optional[Config]=None) -> Tuple[Self, Self, Self]:
        attrs = ["proteins", "texts", "labels"]
        ret = (
            self.get_subset(self.split_indexes["train"], attrs),
            self.get_subset(self.split_indexes["valid"], attrs),
            self.get_subset(self.split_indexes["test"], attrs),
        )
        del self
        return ret