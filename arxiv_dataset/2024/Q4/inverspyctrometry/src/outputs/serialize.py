from __future__ import annotations

from pathlib import Path

import numpy as np


def numpy_save_list(
        filenames: list[str],
        arrays: list[np.ndarray],
        directories: list[Path],
        subdirectory: str = "",
):
    for directory in directories:
        save_dir = directory / subdirectory
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)
        for filename, array in zip(filenames, arrays):
            np.save(file=save_dir / filename, arr=array)


def numpy_load_list(
        filenames: list[str],
        directory: Path,
        subdirectory: str = "",
) -> list[np.ndarray]:
    load_dir = directory / subdirectory
    if not load_dir.is_dir():
        raise FileNotFoundError(f"The directory '{load_dir}' does not exist.")
    arrays = [np.load(file=load_dir / filename) for filename in filenames]
    return arrays
