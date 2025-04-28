import os
from typing import List, Optional, Union

import pandas as pd


def read_result_csv(
        root_directory: Union[os.PathLike, List[os.PathLike]],
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        files: Optional[List[str]] = None
) -> pd.DataFrame:
    """This method reads all csv files in a given root directory, where the sub-path contains a specific pattern.

    Args:
        root_directory: the root directory from which to recursively search for csv files
        include: the patterns the sub-path must contain
        exclude: the patterns the sub-paths is not allowed to contain
        files: a number of files to include anyway

    Returns:
        pandas DataFrame of concatenated csv files

    """
    dataframe = pd.DataFrame()
    if not isinstance(root_directory, list):
        root_directory = [root_directory]
    for root in root_directory:
        if include is not None:
            for dir_path, _, filenames in os.walk(root):
                for filename in filenames:
                    if not filename.endswith(".csv"):
                        continue
                    path = os.path.join(dir_path, filename)
                    if any(x in path for x in include) and not any(x in path for x in exclude):
                        element = pd.read_csv(path)
                        dataframe = pd.concat([dataframe, element], ignore_index=True)
    if files is not None:
        for filename in files:
            element = pd.read_csv(os.path.join(filename))
            dataframe = pd.concat([dataframe, element], ignore_index=True)
    return dataframe
