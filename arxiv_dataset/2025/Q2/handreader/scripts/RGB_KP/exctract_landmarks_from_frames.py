import collections
import typing as tp
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

mp_holistic = mp.solutions.holistic.Holistic(
    static_image_mode=True,
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def get_kept_ids():
    """
    Generates a list of column names corresponding to selected landmark points
    and their x, y, z coordinates, and returns these along with the indices of the kept landmarks.

    The function defines groups of landmark points, including facial features, hands, pose,
    and other body parts, and combines them into a subset of interest. The corresponding column
    names for x, y, and z coordinates are also generated.

    Returns
    -------
    tuple of (numpy.ndarray, list)
        kept_cols_xyz : numpy.ndarray
            An array of strings representing the x, y, and z coordinate column names for the kept landmarks.
        POINT_LANDMARKS : list
            A list of integers representing the indices of the kept landmark points.
    """

    all_cols = [f"face_{i}" for i in range(468)]
    all_cols += [f"left_hand_{i}" for i in range(21)]
    all_cols += [f"pose_{i}" for i in range(33)]
    all_cols += [f"right_hand_{i}" for i in range(21)]
    all_cols = np.array(all_cols)
    NOSE = [1, 2, 98, 327]
    LIP = [
        0,
        61,
        185,
        40,
        39,
        37,
        267,
        269,
        270,
        409,
        291,
        146,
        91,
        181,
        84,
        17,
        314,
        405,
        321,
        375,
        78,
        191,
        80,
        81,
        82,
        13,
        312,
        311,
        310,
        415,
        95,
        88,
        178,
        87,
        14,
        317,
        402,
        318,
        324,
        308,
    ]
    LARMS = [501, 503, 505, 507, 509, 511]
    RARMS = [500, 502, 504, 506, 508, 510]

    REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
    LEYE = [
        263,
        249,
        390,
        373,
        374,
        380,
        381,
        382,
        362,
        466,
        388,
        387,
        386,
        385,
        384,
        398,
    ]

    LHAND = np.arange(468, 489).tolist()
    RHAND = np.arange(522, 543).tolist()

    POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE + LARMS + RARMS

    kept_cols = all_cols[POINT_LANDMARKS]

    kept_cols_xyz = np.array(
        ["x_" + c for c in kept_cols]
        + ["y_" + c for c in kept_cols]
        + ["z_" + c for c in kept_cols]
    )
    return kept_cols_xyz, POINT_LANDMARKS


def extract_landmarks(image_path: str) -> Dict[str, any]:
    """
    Extracts pose, face, and hand landmarks from an image if detected.

    This function uses a holistic model to process the input image and detect
    landmarks for the face, left hand, right hand, and pose. If detected,
    the landmarks are stored in a dictionary.

    Parameters
    ----------
    image_path : str
        The file path to the input image.

    Returns
    -------
    dict of str : any
        A dictionary containing detected landmarks, with the following possible keys:
        - "face": The facial landmarks.
        - "left_hand": The landmarks of the left hand.
        - "right_hand": The landmarks of the right hand.
        - "pose": The pose landmarks.
    """
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        results = mp_holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        results = mp_holistic.process(image_path)

    landmarks = {}
    if results.face_landmarks:
        landmarks["face"] = results.face_landmarks.landmark
    if results.left_hand_landmarks:
        landmarks["left_hand"] = results.left_hand_landmarks.landmark
    if results.right_hand_landmarks:
        landmarks["right_hand"] = results.right_hand_landmarks.landmark
    if results.pose_landmarks:
        landmarks["pose"] = results.pose_landmarks.landmark

    return landmarks


def get_jpg_files_and_dirs(root_dir: str) -> Dict[str, List[str]]:
    """
    Scans a directory recursively and creates a mapping of subdirectory paths to lists of .jpg image file paths.

    Parameters
    ----------
    root_dir : str
        The root directory to start the search for .jpg files.

    Returns
    -------
    dict of str : list of str
        A dictionary where:
        - Keys are relative paths of subdirectories (relative to `root_dir`).
        - Values are lists of full paths to .jpg image files within each subdirectory.
    """

    root_path = Path(root_dir)
    jpg_files = defaultdict(list)

    for jpg_file in tqdm(root_path.rglob("*.jpg")):
        relative_path = jpg_file.parent.relative_to(root_path)
        jpg_files[str(relative_path)].append(str(jpg_file))

    return jpg_files


_, IDX_TO_SAVE = get_kept_ids()


def process_landmarks(img: tp.Union[str, np.ndarray]) -> List:
    """
    Processes landmarks from an image and returns their x, y, and z coordinates.

    This function extracts the x, y, and z coordinates for specified types of landmarks
    ("face", "left_hand", "pose", "right_hand") from the input image. If a particular
    landmark type is not detected in the image, the coordinates for that landmark are
    filled with `NaN` values. Only selected indices from the coordinates are returned
    as a stacked NumPy array.

    Parameters
    ----------
    img : Union[str, np.ndarray]
        The input image, either as a file path (str) or a NumPy array.

    Returns
    -------
    List
        A NumPy array containing stacked x, y, and z coordinates for the landmarks
        corresponding to the specified types. The indices of the returned coordinates
        are filtered based on the `IDX_TO_SAVE` constant.
    """

    landmarks = extract_landmarks(img)
    row = []
    x_row = []
    y_row = []
    z_row = []
    for landmark_type in ["face", "left_hand", "pose", "right_hand"]:
        # for landmark_type in ['face', 'left_hand']:
        if landmark_type in landmarks:
            coords = landmarks[landmark_type]
            x_coords = [lm.x for lm in coords]
            y_coords = [lm.y for lm in coords]
            z_coords = [lm.z for lm in coords]
        else:
            # Determine the number of landmarks based on the landmark type
            num_landmarks = (
                468 if landmark_type == "face" else 33 if landmark_type == "pose" else 21
            )
            x_coords = y_coords = z_coords = [np.nan] * num_landmarks

        x_row.extend(x_coords)
        y_row.extend(y_coords)
        z_row.extend(z_coords)

    row = np.stack(
        [
            np.asarray(x_row)[IDX_TO_SAVE],
            np.asarray(y_row)[IDX_TO_SAVE],
            np.asarray(z_row)[IDX_TO_SAVE],
        ]
    )
    return row


def process_paths(file_paths: List[str], dir_name: str) -> List:
    files = sorted(file_paths)
    rows = []
    for file_path in tqdm(files, desc=f"files processing in dir {dir_name}", leave=False):
        row = process_landmarks(file_path)
        rows.append(row)

    return np.asarray(rows).reshape(len(rows), -1)


def process_and_save_landmarks(
    frames_dir: Dict[str, List[str]], path_to_save_npy: str, start=None, end=None
):
    """
    Processes landmarks from a directory of image frames and saves the results as .npy files.

    Parameters
    ----------
    frames_dir : dict of str : list of str
        A dictionary where keys are directory names (relative paths), and values are lists
        of image file paths to be processed.
    path_to_save_npy : str
        The root path where the processed .npy files will be saved.
    start : int, optional
        The starting index for processing directories in `frames_dir`. Defaults to 0 if not provided.
    end : int, optional
        The ending index (exclusive) for processing directories in `frames_dir`.
        Defaults to the length of `frames_dir` if not provided.
    """

    if start is None and end is None:
        start = 0
        end = len(frames_dir)

    for dir_name in tqdm(list(frames_dir.keys())[start:end], desc="processing dirs", leave=False):
        parquet_to_process = Path(f"{path_to_save_npy}/{dir_name}")
        sub_dir = f"{parquet_to_process.parent.stem}/{parquet_to_process.stem}"
        full_dir_path = Path(path_to_save_npy) / sub_dir
        full_dir_path.parent.mkdir(parents=True, exist_ok=True)
        file_path = full_dir_path.with_suffix(".npy")
        row = process_paths(frames_dir[dir_name], dir_name)
        np.save(str(file_path), row)


if __name__ == "__main__":
    # This code used for extracting landmarks from frames from ChicagoFSWild dataset
    frames_dir = "Path to frames"
    jpg_files_and_dirs = get_jpg_files_and_dirs(frames_dir)
    jpg_files_and_dirs = collections.OrderedDict(sorted(jpg_files_and_dirs.items()))
    print("DONE")
    # processing landmarks and save to csv
    process_and_save_landmarks(
        jpg_files_and_dirs,
        "Path to save npy",
        start=0,
        end=len(jpg_files_and_dirs),
    )
