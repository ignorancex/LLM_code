import csv
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def process_df(path_to_save_csv: str, path_chicago_df: str, train_fold: bool = True):
    """
    Processes a DataFrame of metadata to generate a CSV file for training or testing.

    Parameters
    ----------
    path_to_save_csv : str
        The directory path where the processed CSV file will be saved. The file name
        will be "train_fold.csv" or "test_fold.csv" depending on the `train_fold` flag.
    path_chicago_df : str
        The file path to the input CSV containing metadata (e.g., partition, filenames, labels).
    train_fold : bool, optional, default=True
        If `True`, processes data for training and development folds. If `False`, processes data for the test fold.
    """
    if train_fold:
        path_to_save_csv = str(Path(path_to_save_csv) / "train_fold.csv")
    else:
        path_to_save_csv = str(Path(path_to_save_csv) / "test_fold.csv")

    chicago_df = pd.read_csv(path_chicago_df)
    with open(path_to_save_csv, mode="w", newline="") as file:
        columns = [
            "path",
            "file_id",
            "sequence_id",
            "participant_id",
            "phrase",
            "fold",
            "seq_len",
        ]
        writer = csv.writer(file)
        writer.writerow(columns)
        info = "train" if train_fold else "test"
        for row_id in tqdm(range(len(chicago_df)), desc=f"processing {info}"):
            row = chicago_df.iloc[row_id]
            fold = row["partition"]
            filename = row["filename"]
            number_of_frames = row["number_of_frames"]
            label_proc = row["label_proc"]
            signer = row["signer"]
            file_id, sequence_id = filename.split("/")
            sequence_id = sequence_id.replace(".", "")

            if fold in ["train", "dev"] and train_fold:
                fold_num = 1 if fold == "dev" else 0
                row_to_write = [
                    f"{filename}.parquet",
                    file_id,
                    sequence_id,
                    int(signer),
                    label_proc,
                    fold_num,
                    number_of_frames,
                ]
                writer.writerow(row_to_write)
            elif fold == "test" and not train_fold:
                fold_num = 0
                row_to_write = [
                    f"{filename}.parquet",
                    file_id,
                    sequence_id,
                    int(signer),
                    label_proc,
                    fold_num,
                    number_of_frames,
                ]
                writer.writerow(row_to_write)


if __name__ == "__main__":
    path_to_save_csv = ""
    path_chicago_df = ""

    for train_fold in [True, False]:
        process_df(path_to_save_csv, path_chicago_df, train_fold)
