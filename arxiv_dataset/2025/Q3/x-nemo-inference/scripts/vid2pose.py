# *************************************************************************
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0 
# *************************************************************************
import os
# import ffmpeg
from PIL import Image
import cv2
from tqdm import tqdm

from src.utils.util import get_fps, read_frames, save_videos_from_pil
import numpy as np
from src.utils.draw_util import FaceMeshVisualizer
from src.utils.mp_utils  import LMKExtractor
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2


def get_lmks(face_landmarks):
    lmks = []
    for index in range(len(face_landmarks)):
        x = face_landmarks[index].x
        y = face_landmarks[index].y
        z = face_landmarks[index].z
        lmks.append([x, y, z])
    lmks = np.array(lmks)
    return lmks


def draw_landmarks_on_image(rgb_image, lmks, face_connection_spec=None):
    annotated_image = np.copy(rgb_image)

    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend(
        [
            landmark_pb2.NormalizedLandmark(
                x=landmark[0], y=landmark[1], z=landmark[2]
            )
            for landmark in lmks
        ]
    )

    if face_connection_spec is None:
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
        )
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
        )
    else:
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=face_connection_spec.keys(),
            landmark_drawing_spec=None,
            connection_drawing_spec=face_connection_spec,
        )

        iris_connection_spec = {
            (476, 477): mp.solutions.drawing_styles.DrawingSpec(
                color=(250, 200, 10), thickness=2, circle_radius=1
            ),
            (475, 476): mp.solutions.drawing_styles.DrawingSpec(
                color=(250, 200, 10), thickness=2, circle_radius=1
            ),
            (477, 474): mp.solutions.drawing_styles.DrawingSpec(
                color=(250, 200, 10), thickness=2, circle_radius=1
            ),
            (474, 475): mp.solutions.drawing_styles.DrawingSpec(
                color=(250, 200, 10), thickness=2, circle_radius=1
            ),
            (469, 470): mp.solutions.drawing_styles.DrawingSpec(
                color=(10, 200, 250), thickness=2, circle_radius=1
            ),
            (472, 469): mp.solutions.drawing_styles.DrawingSpec(
                color=(10, 200, 250), thickness=2, circle_radius=1
            ),
            (471, 472): mp.solutions.drawing_styles.DrawingSpec(
                color=(10, 200, 250), thickness=2, circle_radius=1
            ),
            (470, 471): mp.solutions.drawing_styles.DrawingSpec(
                color=(10, 200, 250), thickness=2, circle_radius=1
            ),
        }
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=iris_connection_spec,
        )

    return annotated_image


def extract_bbox_mp(frame, refbbox, detector, timestamp_ms=None):
    if refbbox is None:
        refbbox = (0, 0, frame.shape[1], frame.shape[0])
    # bboxes = fa.face_detector.detect_from_image(frame[..., ::-1])

    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(frame, dtype=np.uint8).copy())   # 如果frame在存储空间不连续则会报错

    if timestamp_ms is None:
        detection_result = detector.detect(image)
    else:
        detection_result = detector.detect_for_video(image, timestamp_ms)
    if len(detection_result.detections) > 0:  # 有多个人脸
        bbox = [
                (
                    detection.categories[0].score,
                    (
                        detection.bounding_box.origin_x,
                        detection.bounding_box.origin_y,
                        detection.bounding_box.origin_x + detection.bounding_box.width,
                        detection.bounding_box.origin_y + detection.bounding_box.height,
                    ),  # LEFT,TOP,RIGHT,BOT
                )
                for detection in detection_result.detections
            ]

        if len(bbox) > 1 and min(bbox)[0] > 0.8:
            return "Detect Multiple faces!"
        bbox = np.array(max(bbox)[1])   # 根据给定的画面范围，找出最显著的人脸
    else:
        return "Detect No face!"
    return np.maximum(bbox, 0)  # bbox 范围不能小于零


def extract_lmk_mp(frame, refbbox, detector, timestamp_ms=None):
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(frame, dtype=np.uint8).copy())   # 如果frame在存储空间不连续则会报错

    if timestamp_ms is None:
        detection_result = detector.detect(image)
    else:
        detection_result = detector.detect_for_video(image, timestamp_ms)
    if len(detection_result.face_landmarks) == 0:
        return "Detect No face!"
    face_lmks = get_lmks(detection_result.face_landmarks[0]).astype(np.float32)
    return face_lmks

def scale_bb(bbox, scale, size):
    left, top, right, bot = bbox[:4].tolist()
    width = right - left
    height = bot - top
    length = max(width, height) * scale
    center_X = (left + right) * 0.5
    center_Y = (top + bot) * 0.5
    left, top, right, bot = [
        center_X - length / 2,
        center_Y - length / 2,
        center_X + length / 2,
        center_Y + length / 2,
    ]
    if left < 0 or top < 0 or right > size[1] - 1 or bot > size[0] - 1:
        return bbox
    else:
        return np.array([left, top, right, bot])


def customize_face_connection_spec(draw_edge=True, mask_eye=False, mask_mouth=False):
    forehead_edge = False

    mp_face_mesh = mp.solutions.face_mesh
    DrawingSpec = mp.solutions.drawing_styles.DrawingSpec
    f_thick = 2
    f_rad = 1

    right_iris_draw = DrawingSpec(
        color=(10, 200, 250), thickness=f_thick, circle_radius=f_rad
    )
    right_eye_draw = DrawingSpec(
        color=(10, 200, 180), thickness=f_thick, circle_radius=f_rad
    )
    right_eyebrow_draw = DrawingSpec(
        color=(10, 220, 180), thickness=f_thick, circle_radius=f_rad
    )
    left_iris_draw = DrawingSpec(
        color=(250, 200, 10), thickness=f_thick, circle_radius=f_rad
    )
    left_eye_draw = DrawingSpec(
        color=(180, 200, 10), thickness=f_thick, circle_radius=f_rad
    )
    left_eyebrow_draw = DrawingSpec(
        color=(180, 220, 10), thickness=f_thick, circle_radius=f_rad
    )
    head_draw = DrawingSpec(color=(10, 200, 10), thickness=f_thick, circle_radius=f_rad)

    mouth_draw_obl = DrawingSpec(
        color=(10, 180, 20), thickness=f_thick, circle_radius=f_rad
    )
    mouth_draw_obr = DrawingSpec(
        color=(20, 10, 180), thickness=f_thick, circle_radius=f_rad
    )

    mouth_draw_ibl = DrawingSpec(
        color=(100, 100, 30), thickness=f_thick, circle_radius=f_rad
    )
    mouth_draw_ibr = DrawingSpec(
        color=(100, 150, 50), thickness=f_thick, circle_radius=f_rad
    )

    mouth_draw_otl = DrawingSpec(
        color=(20, 80, 100), thickness=f_thick, circle_radius=f_rad
    )
    mouth_draw_otr = DrawingSpec(
        color=(80, 100, 20), thickness=f_thick, circle_radius=f_rad
    )

    mouth_draw_itl = DrawingSpec(
        color=(120, 100, 200), thickness=f_thick, circle_radius=f_rad
    )
    mouth_draw_itr = DrawingSpec(
        color=(150, 120, 100), thickness=f_thick, circle_radius=f_rad
    )

    FACEMESH_LIPS_OUTER_BOTTOM_LEFT = [
        (61, 146),
        (146, 91),
        (91, 181),
        (181, 84),
        (84, 17),
    ]
    FACEMESH_LIPS_OUTER_BOTTOM_RIGHT = [
        (17, 314),
        (314, 405),
        (405, 321),
        (321, 375),
        (375, 291),
    ]

    FACEMESH_LIPS_INNER_BOTTOM_LEFT = [
        (78, 95),
        (95, 88),
        (88, 178),
        (178, 87),
        (87, 14),
    ]
    FACEMESH_LIPS_INNER_BOTTOM_RIGHT = [
        (14, 317),
        (317, 402),
        (402, 318),
        (318, 324),
        (324, 308),
    ]

    FACEMESH_LIPS_OUTER_TOP_LEFT = [(61, 185), (185, 40), (40, 39), (39, 37), (37, 0)]
    FACEMESH_LIPS_OUTER_TOP_RIGHT = [
        (0, 267),
        (267, 269),
        (269, 270),
        (270, 409),
        (409, 291),
    ]

    FACEMESH_LIPS_INNER_TOP_LEFT = [(78, 191), (191, 80), (80, 81), (81, 82), (82, 13)]
    FACEMESH_LIPS_INNER_TOP_RIGHT = [
        (13, 312),
        (312, 311),
        (311, 310),
        (310, 415),
        (415, 308),
    ]

    FACEMESH_CUSTOM_FACE_OVAL = [
        (176, 149),
        (150, 136),
        (356, 454),
        (58, 132),
        (152, 148),
        (361, 288),
        (251, 389),
        (132, 93),
        (389, 356),
        (400, 377),
        (136, 172),
        (377, 152),
        (323, 361),
        (172, 58),
        (454, 323),
        (365, 379),
        (379, 378),
        (148, 176),
        (93, 234),
        (397, 365),
        (149, 150),
        (288, 397),
        (234, 127),
        (378, 400),
        (127, 162),
        (162, 21),
    ]

    # mp_face_mesh.FACEMESH_CONTOURS has all the items we care about.
    face_connection_spec = {}
    if forehead_edge:
        for edge in mp_face_mesh.FACEMESH_FACE_OVAL:
            face_connection_spec[edge] = head_draw
    if draw_edge:
        for edge in FACEMESH_CUSTOM_FACE_OVAL:
            face_connection_spec[edge] = head_draw

    if not mask_eye:
        for edge in mp_face_mesh.FACEMESH_LEFT_EYE:
            face_connection_spec[edge] = left_eye_draw
        for edge in mp_face_mesh.FACEMESH_LEFT_EYEBROW:
            face_connection_spec[edge] = left_eyebrow_draw
        # for edge in mp_face_mesh.FACEMESH_LEFT_IRIS:
        #    face_connection_spec[edge] = left_iris_draw
        for edge in mp_face_mesh.FACEMESH_RIGHT_EYE:
            face_connection_spec[edge] = right_eye_draw
        for edge in mp_face_mesh.FACEMESH_RIGHT_EYEBROW:
            face_connection_spec[edge] = right_eyebrow_draw
        # for edge in mp_face_mesh.FACEMESH_RIGHT_IRIS:
        #    face_connection_spec[edge] = right_iris_draw

    if not mask_mouth:
        for edge in FACEMESH_LIPS_OUTER_BOTTOM_LEFT:
            face_connection_spec[edge] = mouth_draw_obl
        for edge in FACEMESH_LIPS_OUTER_BOTTOM_RIGHT:
            face_connection_spec[edge] = mouth_draw_obr
        for edge in FACEMESH_LIPS_INNER_BOTTOM_LEFT:
            face_connection_spec[edge] = mouth_draw_ibl
        for edge in FACEMESH_LIPS_INNER_BOTTOM_RIGHT:
            face_connection_spec[edge] = mouth_draw_ibr
        for edge in FACEMESH_LIPS_OUTER_TOP_LEFT:
            face_connection_spec[edge] = mouth_draw_otl
        for edge in FACEMESH_LIPS_OUTER_TOP_RIGHT:
            face_connection_spec[edge] = mouth_draw_otr
        for edge in FACEMESH_LIPS_INNER_TOP_LEFT:
            face_connection_spec[edge] = mouth_draw_itl
        for edge in FACEMESH_LIPS_INNER_TOP_RIGHT:
            face_connection_spec[edge] = mouth_draw_itr

    return face_connection_spec
