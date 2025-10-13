import argparse
from pathlib import Path
from typing import Dict, List

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import rootutils
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from app.RGB_KP.exctract_landmarks_from_frames import get_kept_ids, process_landmarks

mp_holistic = mp.solutions.holistic.Holistic(
    static_image_mode=True,
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def chunker(seq, size):
    """
    Yield successive n-sized chunks from the given sequence.

    Parameters
    ----------
    seq : iterable
        The sequence to be chunked.
    size : int
        The size of each chunk.

    Yields
    ------
    list
        A sublist of the given sequence, of length `size`.
    """
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


def split_filenames_on_chunks_and_start_end(
    all_names: List[str], chunck_size: int
) -> Dict[str, tuple]:
    """

    Split the given list of filenames into chunks of given size and associate each with its chunk index and some dummy values.

    Parameters
    ----------
    all_names : List[str]
        The list of filenames to be split.

    chunck_size : int
        The size of each chunk.

    Returns
    -------
    Dict[str, tuple]
        A dictionary with the filenames as keys and a tuple containing the chunk index, start and end as values.
    """
    dct = {}
    all_names = sorted(all_names)
    for ch, videos in enumerate(chunker(all_names, chunck_size)):
        for video in videos:
            dct[video] = (ch, 1, 1)
    return dct


def process_video_landmarks(path_to_video: str) -> List:
    """
    Process the given video file and extract the landmarks from each frame.

    Parameters
    ----------
    path_to_video : str
        The path to the video file to be processed.

    Returns
    -------
    List
        A list of arrays, each containing the extracted landmarks for a frame.
    """
    rows = []

    cap = cv2.VideoCapture(path_to_video)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in tqdm(
        range(frame_count),
        desc=f"extracting landmarks from {path_to_video.split('/')[-1]}",
        leave=False,
    ):
        ret, frame = cap.read()
        if not ret:
            break

        # to RGB
        frame = frame[:, :, ::-1]

        row = process_landmarks(frame)
        rows.append(row)
    return np.asarray(rows).reshape(len(rows), -1)


def process_videos_and_save_landmarks(
    frames_dir: Dict[str, List[str]],
    path_to_save_npy: str,
    path_to_videos: str,
    start=None,
    end=None,
):
    """
    Process the given videos and save the extracted landmarks as .npy files.

    Parameters
    ----------
    frames_dir : dict of str : list of str
        A dictionary where keys are directory names (relative paths), and values are lists
        of file names to be processed.
    path_to_save_npy : str
        The root path where the processed .npy files will be saved.
    path_to_videos : str
        The root path where the video files are located.
    start : int, optional
        The starting index for processing directories in `frames_dir`. Defaults to 0 if not provided.
    end : int, optional
        The ending index (exclusive) for processing directories in `frames_dir`.
        Defaults to the length of `frames_dir` if not provided.
    """

    if start is None and end is None:
        start = 0
        end = len(frames_dir)

    for file_name in tqdm(
        list(frames_dir.keys())[start:end], desc="processing videos", leave=False
    ):
        chunk = frames_dir[file_name][0]
        path_video = f"{path_to_videos}/{file_name}"

        sub_dir = f"{chunk}/{Path(file_name).stem}"
        full_dir_path = Path(path_to_save_npy) / sub_dir
        full_dir_path.parent.mkdir(parents=True, exist_ok=True)
        file_path = full_dir_path.with_suffix(".npy")

        row = process_video_landmarks(path_video)
        np.save(str(file_path), row)
        exit()


def add_path_npy_column(path_csv: str, path_to_save_new: str, path_to_npys: str):
    """
    Add a new column to a CSV file, 'path_npy', with the path to the corresponding .npy file containing the extracted landmarks.

    Parameters
    ----------
    path_csv : str
        The path to the input CSV file.
    path_to_save_new : str
        The path to the output CSV file.
    path_to_npys : str
        The root path where the .npy files are located.
    """

    df = pd.read_csv(path_csv, sep="\t")
    df["path_npy"] = np.nan

    root_path = Path(path_to_npys)
    stem_to_new_path = {}
    for jpg_file in tqdm(root_path.rglob("*.npy")):
        stem_to_new_path[Path(jpg_file).stem] = Path(jpg_file).parts[-2]

    for i in tqdm(range(len(df))):
        df.loc[i, "path_npy"] = (
            stem_to_new_path[Path(df.loc[i, "path"]).stem]
            + "/"
            + Path(df.loc[i, "path"]).stem
            + ".npy"
        )

    df.to_csv(path_to_save_new, index=False)


if __name__ == "__main__":
    # This code used for extracting landmarks from videos from Znaki dataset
    parser = argparse.ArgumentParser(description="Process video landmarks and save to CSV.")

    # Add start and end arguments
    parser.add_argument(
        "--start", type=int, default=0, help="Starting index for processing videos"
    )
    parser.add_argument("--end", type=int, default=None, help="Ending index for processing videos")

    # Parse the arguments
    args = parser.parse_args()

    # Get start and end values
    start = args.start
    end = args.end

    _, idx_to_save = get_kept_ids()
    # provide path to trimmed videos
    path_to_videos = "Path to videos"

    # provide path to annotations
    df = pd.read_csv(
        "Path to annotations",
        sep="\t",
    )

    all_names = sorted(df["path"].tolist())
    videos_to_chunk = split_filenames_on_chunks_and_start_end(all_names, 2500)
    print("DONE")

    # provide path to save npys
    process_videos_and_save_landmarks(
        videos_to_chunk,
        "Path to save npy",
        path_to_videos,
        start=start,
        end=end,
    )

    add_path_npy_column(path_csv="", path_to_save_new="", path_to_npys="")
