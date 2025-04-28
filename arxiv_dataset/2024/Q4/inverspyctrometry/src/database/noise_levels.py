import json
from typing import List, Sequence

from pydantic import RootModel


class NoiseLevelListSchema(Sequence, RootModel):
    root: List[float]

    def __getitem__(self, item: int) -> float:
        return self.root[item]

    def __len__(self):
        return len(self.root)
