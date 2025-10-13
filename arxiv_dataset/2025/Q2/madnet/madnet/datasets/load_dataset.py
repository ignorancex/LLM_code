import pathlib
from typing import Literal

from .bhp import load_bhp
from .data_model import SyntheticDataGenerator
from .ihdp import load_ihdp


def load_dataset(
    name: Literal["ihdp", "bhp"] = "ihdp", data_dir: str = "data"
) -> SyntheticDataGenerator:
    """Load a dataset as a SyntheticDataGenerator.

    Example
    -------
    ```python
    from datasets import load_dataset

    generator = load_dataset("ihdp")
    df = generator(seed=123)
    data_splits = df.split()
    ```
    """
    data_dir_path = pathlib.Path(data_dir)
    data_dir_path.mkdir(parents=True, exist_ok=True)

    match name:
        case "ihdp":
            return load_ihdp(data_dir)
        case "bhp":
            return load_bhp(data_dir)
        case _:
            raise NotImplementedError
