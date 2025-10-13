from pathlib import Path
from typing import List

import cv2
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


def extract_frames_trimmed_video(video_path, output_dir):
    """
    Extract frames from a video and save them to a directory.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    output_dir : str
        Directory where the frames will be saved.

    Returns
    -------
    None
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_number = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_filename = f"{output_dir}/frame_{frame_number:04d}.jpg"
        cv2.imwrite(frame_filename, frame)

        frame_number += 1

    cap.release()


def get_all_videos(path: str, return_names: bool = False) -> list:
    """
    Get all .mp4 videos in a directory.

    Parameters
    ----------
    path : str
        Path to the directory with the videos.
    return_names : bool, optional
        If True, return both the paths and the names of the videos.

    Returns
    -------
    list
        List of paths to the videos. If return_names is True, returns a tuple of
        two lists: the paths and the names of the videos.
    """

    path = Path(path)
    paths = [f for f in path.glob("*.mp4")]
    if return_names:
        return paths, [f.name for f in paths]
    else:
        return paths


def process_video(video_path, output_dir):
    """
    Process a video by extracting its frames and saving them to a directory.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    output_dir : str
        Directory where the frames will be saved.

    Returns
    -------
    None
    """
    current_folder_video = Path(output_dir) / Path(video_path).stem
    current_folder_video.mkdir(parents=True, exist_ok=True)
    extract_frames_trimmed_video(
        video_path=str(video_path),
        output_dir=str(current_folder_video),
    )


def main(
    path_to_save: str,
    path_to_videos: List[str],
    num_videos_split,
):
    """
    Process a list of videos and save their frames to a directory.

    Parameters
    ----------
    path_to_save : str
        Path to the directory where the frames will be saved.
    path_to_videos : List[str]
        List of paths to the video files to be processed.
    num_videos_split : int
        Number of videos to process in each chunk.

    Returns
    -------
    None
    """
    for chunk in tqdm(
        range(0, len(path_to_videos), num_videos_split),
        desc=f"processing videos {len(path_to_videos)}",
    ):
        current_folder = Path(path_to_save) / f"{chunk // num_videos_split}"
        current_folder.mkdir(parents=True, exist_ok=True)
        chunk_of_videos = path_to_videos[chunk : chunk + num_videos_split]

        Parallel(n_jobs=16, backend="threading")(
            delayed(process_video)(
                video_path,
                current_folder,
            )
            for video_path in tqdm(chunk_of_videos, desc="extracting frames", leave=False)
        )


if __name__ == "__main__":
    trimmed_videos_path = ""
    path_to_save = ""
    df = pd.read_csv(
        "",
        sep="\t",
    )
    path_to_videos = []
    for i in range(len(df)):
        path_to_videos.append(trimmed_videos_path + "/" + df.iloc[[i]]["path"].values[0])
    main(path_to_save, path_to_videos, num_videos_split=2500)
