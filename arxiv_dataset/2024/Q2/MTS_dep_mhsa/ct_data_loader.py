import glob
import os
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from typing import List
from itertools import groupby

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import DualTransform


class CutOut(DualTransform):
    def __init__(self, always_apply: bool = False, p: float = 0.1, n_holes=10, length=28):
        super().__init__(always_apply, p)
        self.n_holes = np.random.choice(n_holes)
        self.length = np.random.choice(np.arange(10, length))
        
    def apply(self, img:np.ndarray, **params):
        h, w, c = img.shape
        mask = np.ones((h, w))
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1: y2, x1: x2] = 0.
        mask = mask.reshape(h,w,1)
        mask = mask.astype(np.uint8)
        img *= mask
        max_pixel = np.max(img)
        masked = np.random.random()
        if masked <= self.p:
            plane = np.zeros((h, w, c))
            plane = plane.astype(np.uint8)
            img = plane 
        return img
        


class CtImages(Dataset):
    """
    CtImages class.
    """

    def __init__(self,
                input_path: str = None,
                resample: bool = True,
                n_frames: int = 11, #48,
                origin_size_x: int = 512,
                origin_size_y: int = 512,
                x_image_size: int = 512,
                y_image_size: int = 512,
                pixels_crop_height: int = 100,
                pixels_crop_width: int = 100,
                max_width: int = 1680,
                max_height: int = 1260,
                skip_frames: int = 0,
                multiple_planes: bool = False,
                background_frames_filename: str = None, #"background_frames.csv",
                resize_only: bool = False,
                padding: bool = True,
                mode: str = None,
                diagnosis_labels_path=""
                ) -> None:
        """

        Args:
            input_path:
            resample:
            n_frames: if multiple_planes, n_frames must be divisible by 3
            origin_size_x: the input scans' width
            origin_size_y: the input scans' height
            x_image_size: the output scans' width
            y_image_size: the output scans' height
            pixels_crop_height: 
            pixels_crop_width:
            max_width:
            max_height:
            skip_frames:
            multiple_planes:
            background_frames_filename:
            resize_only:
            padding:
            mode:
            diagnosis_labels_path:  is a csv file which contains patient_id and its label for train,
                                    the label can be empty. 
        """
        self.input_path = input_path
        self.n_frames = n_frames
        self.x_image_size = x_image_size
        self.y_image_size = y_image_size
        self.max_width = max_width
        self.max_height = max_height
        self.pixels_crop_height = pixels_crop_height
        self.pixels_crop_width = pixels_crop_width
        self.multiple_planes = multiple_planes
        self.diagnosis_labels = pd.read_csv(diagnosis_labels_path, sep=",", usecols=["patient_name", "label"])
        self.resize_only = resize_only
        self.origin_size_x = origin_size_x
        self.origin_size_y = origin_size_y

        assert (self.resize_only and pixels_crop_width == 0) or (not self.resize_only)
        assert mode in ["train", "val", None]
        self.mode = mode
        self.patient_id = self.diagnosis_labels["patient_name"].tolist()
        if mode in ['train', 'val']:
            self.id_2_class = dict(zip(self.patient_id, self.diagnosis_labels["label"].tolist()))

        self.resample = resample
        self.chunk_frames = []
        self.patient_id_by_chunk = []
        self.frames_to_omit = []

        # whether some frames needs to skip/omit.
        if background_frames_filename:
            background_frames = pd.read_csv(os.path.join(input_path, background_frames_filename))
            background_frames = background_frames["background_frames"].tolist()
            background_frames = [f"{f}.png" for f in background_frames]
            self.frames_to_omit = background_frames

        for pid in self.patient_id:
            frames = glob.glob(f"{self.input_path}/{pid}/*.png")
            frames = sorted(frames, key=numericalSort)
            frames = [f for f in frames if self._get_filename(f) not in self.frames_to_omit]

            if multiple_planes:
                chunk_frames = self.chunk_frames_multiple_planes(frames, n_frames)
            else:
                chunk_frames = chunks(frames, n_frames, skip_frames) # sip_frames=0
            
            # padding the frames if any subject's scans is less than 12  
            for chunk in chunk_frames:
                if len(chunk) != n_frames and not padding:
                    continue
                chunk.extend(["padding" for _ in range(n_frames - len(chunk))])
                self.chunk_frames.append([chunk, pid])
                self.patient_id_by_chunk.append(pid)


    def __getitem__(self, x):
        """

        Args:
            x:

        Returns:
        """
        chunk_frames, patient_name = self.chunk_frames[x][0], self.chunk_frames[x][1]            
        patient_dir = f"{self.input_path}/{patient_name}/" 
        video_stack = self.load_video(patient_dir, chunk_frames, resample=True)
        label = self.diagnosis_labels.loc[self.diagnosis_labels["patient_name"] == patient_name, "label"]
        try:
            label = label.item()
        except:
            raise KeyError(patient_name)

        video_stack = torch.tensor(video_stack)
        first_frame = self._first_frame(chunk_frames)
        return video_stack, label, self.patient_id_by_chunk[x], first_frame

    def __len__(self) -> int:
        return len(self.chunk_frames)

    def _get_filename(self, full_path):
        return os.path.basename(full_path)


    def load_video(self, patient_dir: str = None,
                   chunk_frames: list = None,
                   resample: bool = True):
        """
        Args:
            resample:
            chunk_frames:
            patient_dir:

        Returns: transformed video
        """
        if not os.path.exists(patient_dir):
            raise FileNotFoundError(patient_dir)

        video = np.zeros(
            (1,  len(chunk_frames), self.y_image_size, self.x_image_size),
            np.float32)
        video_frames = {}
        for count, chunk in enumerate(chunk_frames):
            if resample:
                if "padding" in chunk:
                    continue
                frame = cv2.imread(chunk, cv2.IMREAD_GRAYSCALE) # H, W, C

                if self.resize_only:
                    transformations = A.Compose([
                        A.Resize(self.y_image_size, self.x_image_size),
                    ])
                else:
                    height, width = frame.shape[:2]  # H, W, C
                    transformations = A.Compose([
                        # A.Resize(transformed_height, transformed_width),
                        A.CenterCrop(self.origin_size_y - self.pixels_crop_height, self.origin_size_x - self.pixels_crop_width),
                        A.Resize(self.x_image_size, self.y_image_size),
                        A.PadIfNeeded(self.y_image_size, self.x_image_size, border_mode=cv2.BORDER_CONSTANT, value=0),
                    ])

                transformed_frame = transformations(image=frame)
                transformed_frame = transformed_frame["image"]
                transformed_frame = np.expand_dims(transformed_frame, axis=2)
                frame_name = f"image{count - 1}" if count != 0 else "image"
                video_frames[frame_name] = transformed_frame

        if self.mode == "val":
            augmentations = A.Compose([ToTensorV2(p=1)], 
                                      additional_targets={f"image{i}": "image" for i in range(len(video_frames) - 1)})
            transformed_video = augmentations(**video_frames)

        else:
            blackout = A.Compose([CutOut(p=0.15)]) # 0.15 for 0.82
            augmentations = A.Compose([A.Rotate(limit=(-35, 35)),
                                        A.HorizontalFlip(p=0.3),
                                        A.VerticalFlip(p=0.3),
                                        A.RandomBrightnessContrast(0.05, 0.05),
                                        A.ImageCompression(p=0.12),
                                        A.OneOf([
                                            A.MotionBlur(p=.2),
                                            A.MedianBlur(blur_limit=3, p=0.2),
                                            A.Blur(blur_limit=3, p=0.5),
                                            A.GaussianBlur(p=0.3)], 
                                            p=0.2),
                                            ToTensorV2(p=1)],
                                        additional_targets={f"image{i}": "image" for i in range(len(video_frames) - 1)}
                                        )
            blackout_res = blackout(**video_frames)
            transformed_video = augmentations(**blackout_res)

        for i, (frame_name, frame) in enumerate(video_frames.items()):
            augmented_frame = transformed_video[frame_name]
            video[0][i] = augmented_frame

        video = (video - 0)/255.  # img = (img - mean * max_pixel_value) / (std * max_pixel_value)
        return video

    def chunk_frames_multiple_planes(self, frames, n_frames):
        by_body_parts = groupby(frames, self._body_part_from_frame_path)
        body_parts_dict = defaultdict(list)
        max_length = 0
        for b_p, fr in by_body_parts:
            b_p_list = list(el for el in fr)
            if len(b_p_list) > max_length:
                max_length = len(b_p_list)
            body_parts_dict[b_p] = b_p_list
        max_length_with_padding = max_length + (n_frames // 3 - (max_length % (n_frames // 3)))
        abdomen_frames = self._pad(body_parts_dict["abdomen"], max_length_with_padding)
        femur_frames = self._pad(body_parts_dict["femur"], max_length_with_padding)
        head_frames = self._pad(body_parts_dict["head"], max_length_with_padding)
        chunk_frames = []
        body_part_frames_num = n_frames // 3
        for i in range(max_length_with_padding // (n_frames // 3)):
            new_chunk = []
            new_chunk.extend(abdomen_frames[i * body_part_frames_num: i * body_part_frames_num + body_part_frames_num])
            new_chunk.extend(femur_frames[i * body_part_frames_num: i * body_part_frames_num + body_part_frames_num])
            new_chunk.extend(head_frames[i * body_part_frames_num: i * body_part_frames_num + body_part_frames_num])
            all_padding = True
            for f in new_chunk:
                if "padding" not in f:
                    all_padding = False
            if all_padding:
                continue
            chunk_frames.append(new_chunk)

        return chunk_frames

    def _pad(self, l: list, pad_to: int):
        l.extend(["padding" for _ in range(pad_to - len(l))])
        return l

    def _first_frame(self, frames):
        for f in frames:
            if "padding" not in f:
                return "video" + f.split("video")[-1]


def chunks(L:List, n:int, skip:int):
    """

    Args:
        L:
        n:
        skip:

    Returns:

    """
    if skip > 0:
        return [L[x: x + (skip+1) * (n-1)+1: skip + 1] for x in range(0, len(L), (skip+1) * (n-1)+1)]
    else:
        return [L[x: x + n: skip + 1] for x in range(0, len(L), n)]


def numericalSort(img_name):
    # numbers = re.compile(r'(\d+)')
    # parts = numbers.split(value)
    # parts[1::2] = map(int, parts[1::2])
    try:
        number_part = re.findall(".*_C?T?.*?0(\d+).png$", img_name)
        number_part = int(number_part[0])
    except:
        print(img_name)
        raise IndexError
    return number_part

if __name__ == "__main__":
    print(numericalSort("Zhangxuhua2399_CT010"))


