from typing import Dict, List

import mediapipe as mp
import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils import LOGGER

if len(LOGGER.handlers) > 1:
    LOGGER.handlers = []

mp_holistic = mp.solutions.holistic.Holistic(
    static_image_mode=False,
    model_complexity=0,
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
    # 1st place kept landmarks
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


_, IDX_TO_SAVE = get_kept_ids()


def extract_landmarks(frame) -> Dict[str, any]:
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
    results = mp_holistic.process(frame)

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


def process_landmarks(frame) -> List:
    """
    Processes landmarks from an image and returns a list of rows.

    Parameters
    ----------
    frame : np.ndarray
        The input image.

    Returns
    -------
    np.ndarray
        A numpy array of shape (1, num_landmarks * 3) containing the processed landmarks.
    """

    rows = []
    # Extract x, y, z coordinates for each type of landmark
    landmarks = extract_landmarks(frame)
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
    rows.append(row)
    return np.asarray(rows).reshape(len(rows), -1)
