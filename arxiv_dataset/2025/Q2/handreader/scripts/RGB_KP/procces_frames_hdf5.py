from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib


def process_video(path_to_frames: str, hdf5_file: str):
    """
    Processes video frames from a directory and stores them in an HDF5 file as JPEG binary data.

    Parameters
    ----------
    path_to_frames : str
        The path to the directory containing video frames in .jpg format.
    hdf5_file : str
        The path to the HDF5 file where the processed video frames will be stored.

    The function reads each frame from the specified directory, encodes it as JPEG binary data,
    and saves it into the specified HDF5 file under a group named after the directory. The group
    contains attributes for frame count, dimensions, and source path, and a dataset for the JPEG frames.
    """
    frames = sorted(list(Path(path_to_frames).glob("*.jpg")))
    frame_count = len(frames)
    frame_sample = cv2.imread(frames[0])
    h, w, c = frame_sample.shape
    frame_width = w
    frame_height = h

    with h5py.File(hdf5_file, "a") as h5f:
        video_group = h5f.create_group(Path(path_to_frames).stem)
        video_group.attrs["frame_count"] = len(frames)
        video_group.attrs["width"] = frame_width
        video_group.attrs["height"] = frame_height
        video_group.attrs["path_to_frames"] = str(path_to_frames)

        jpeg_dataset = video_group.create_dataset(
            "jpeg_frames", shape=(frame_count,), dtype=h5py.special_dtype(vlen=np.dtype("uint8"))
        )

        for i, path in enumerate(frames):
            frame = cv2.imread(str(path))

            ret, jpeg_frame = cv2.imencode(".jpg", frame)
            if not ret:
                continue

            jpeg_dataset[i] = np.frombuffer(jpeg_frame, dtype="uint8")


from joblib import Parallel, delayed


def main(hdf5_file: str, path_to_subdirs: list):
    """
    Processes multiple video directories in parallel and stores their frames in an HDF5 file.

    Parameters
    ----------
    hdf5_file : str
        The path to the HDF5 file where the processed video frames will be stored.
    path_to_subdirs : list
        A list of paths to the directories containing video frames in .jpg format.

    The function processes each directory in parallel using 16 threads, and stores the JPEG frames in the specified HDF5 file.
    """

    with tqdm_joblib(tqdm(desc="Processing videos", total=len(path_to_subdirs))):
        Parallel(n_jobs=16, backend="threading")(
            delayed(process_video)(path, hdf5_file) for path in path_to_subdirs
        )

    print(
        f"Successfully stored {len(path_to_subdirs)} videos with frames in binary format in {hdf5_file}"
    )


if __name__ == "__main__":
    hdf5_file = ""
    path_videos = ""
    p = Path(path_videos)
    dirs = []
    for f in p.iterdir():
        for p in f.iterdir():
            dirs.append(p)

    main(hdf5_file, dirs)
