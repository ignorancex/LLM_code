import json
from pathlib import Path
import os
import shutil

"""
__Make RGB Summary__

This script moves all the RGB summary images, `1_rgb_fit_summary.png`, from the dataset folders to a new folder called
`images/rgb_summary`. 

This makes all images accessible from a single location, which is useful for browsing the results of many datasets
quickly.
"""

dataset_folder_list = [
    "M25",
    "S12",
    "S11",
    "S10",
    "S09",
    "S08",
    "S07",
    "S06",
    "S05",
    "S04",
    "S03",
    "S02",
    "S01",
    "S00",
]

for dataset_folder in dataset_folder_list:
    for dataset_name in os.listdir(dataset_folder):
        dataset_path = os.path.join(dataset_folder, dataset_name)

        with open(Path(dataset_path) / "info.json") as json_file:
            info = json.load(json_file)
            json_file.close()

        score = info["score"]

        try:
            shutil.copy(
                Path(dataset_path) / "1_rgb_fit_summary.png",
                Path("images") / "rgb_summary" / f"{score}_{dataset_name}",
            )
        except FileNotFoundError:
            continue
