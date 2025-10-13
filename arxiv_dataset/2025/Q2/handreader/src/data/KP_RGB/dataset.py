import json
import math
import random
from pathlib import Path
from typing import Any, Optional

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import rootutils
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch.nn as nn

from src.data.data_utils import Preprocessing, get_vocab, getChicagoTokens, numerize
from src.data.KP_RGB.augmentations import RandomRotation, TemporalRescale
from src.utils import LOGGER


class JointDataset(Dataset):
    def __init__(
        self,
        path_to_rgb: str,
        path_to_landmarks: str,
        df,
        transforms,
        partition: str,
        img_size: int,
        inference_args: str,
        symmetry_fp: str,
        horizontal_flip_prob: float = 0.5,
        temporal_resample_prob: float = 0.8,
        rotatation_prob: float = 0.5,
        inds_to_filter: list[str] = None,
    ):
        """
        JointDataset constructor

        Parameters
        ----------
        path_to_rgb : str
            Path to RGB video folder
        path_to_landmarks : str
            Path to RGB video folder
        df : pandas.DataFrame
            Dataframe with sample information
        transforms : albumentations.Compose
            Compose object for image augmentations
        partition : str
            Partition to load (train, val, test)
        img_size : int
            Input image size
        inference_args : str
            Path to JSON file with inference arguments
        symmetry_fp : str
            Path to CSV file with symmetry information
        horizontal_flip_prob : float
            Probability of applying random horizontal flip
        temporal_resample_prob : float
            Probability of applying random temporal resampling
        rotatation_prob : float
            Probability of applying random rotation
        inds_to_filter : list[str]
            List of strings to filter from the columns

        Returns
        -------
        None

        """
        super().__init__()
        self.img_size = img_size
        self.df = df
        self.path_to_rgb = path_to_rgb
        self.path_to_landmarks = path_to_landmarks

        self.partition = partition

        self.partition = partition
        self.horizontal_flip_prob = horizontal_flip_prob
        self.temporal_resample_prob = temporal_resample_prob
        self.rotation_prob = rotatation_prob
        self.temporal_resample = TemporalRescale((0.5, 1.5))
        self.random_rotation = RandomRotation()

        with open(inference_args) as f:
            columns = json.load(f)["selected_columns"]

        if inds_to_filter is not None:
            self.filtered_columns = [
                idx
                for idx, col in enumerate(columns)
                if any(substring in col for substring in inds_to_filter)
            ]

        self.xyz_landmarks = np.array(
            [col for col in columns if any(substring in col for substring in inds_to_filter)]
        )
        landmarks = np.array(
            [item[2:] for item in self.xyz_landmarks[: len(self.xyz_landmarks) // 3]]
        )

        symmetry = pd.read_csv(symmetry_fp).set_index("id")

        flipped_landmarks = symmetry.loc[landmarks]["corresponding_id"].values
        self.flip_array = np.where(landmarks[:, None] == flipped_landmarks[None, :])[1]

        self.labels = self.df["phrase"]

        chars = getChicagoTokens()

        self.vocab_map_ctc, self.inv_vocab_map_ctc, self.char_list_ctc = get_vocab(chars)

        self.targets_enc_ctc = numerize(self.labels, self.vocab_map_ctc, False)

        self.transforms = transforms
        self.proc = Preprocessing()

    def transform(self, img, transforms):
        """
        Applies augmentation to the input data.

        Parameters
        ----------
        x : torch.Tensor
            The input data to augment.

        Returns
        -------
        torch.Tensor
            Augmented data.
        """
        return transforms(image=img)["image"]

    def __len__(
        self,
    ):
        return len(self.df)

    def load_one(self, file_id, sequence_id):
        """
        Loads a single data sample from a specified file path.

        Parameters
        ----------
        file_id : str
            The file id of the data sample.
        sequence_id : int
            The sequence id of the data sample.

        Returns
        -------
        data : np.ndarray
            The loaded data sample in the format (seq_len, 3 * nlandmarks).
        """

        path = self.path_to_landmarks + f"/{file_id}/{sequence_id}.npy"
        data = np.load(path)  # seq_len, 3* nlandmarks
        return data

    def get_resampling_indices(self, original_len, new_len):
        """
        Gets a list of resampling indices for a sequence of length
        original_len to be resampled to a sequence of length new_len.

        Parameters
        ----------
        original_len : int
            The length of the original sequence.
        new_len : int
            The length of the resampled sequence.

        Returns
        -------
        list
            A list of indices to use for resampling the original sequence.
        """

        if new_len <= original_len:
            indices = sorted(random.sample(range(original_len), new_len))
        else:
            indices = []
            step = original_len / new_len
            for i in range(new_len):
                indices.append(int(i * step))
            indices = sorted(indices)
        return indices

    def __getitem__(self, index) -> Any:
        """
        Retrieves a data sample and applies specified transformations.

        Parameters
        ----------
        index : int
            Index of the data sample to retrieve.

        Returns
        -------
        tuple
            A tuple containing:
            - imgs : torch.Tensor
                Padded tensor of RGB video frames after transformations.
            - data : torch.Tensor
                Processed 3D skeleton data after transformations.
            - label_ctc : torch.Tensor
                Encoded target phrase.
            - len(imgs) : int
                Length of the RGB video frames.
            - len(label_ctc) : int
                Length of the target sequence.
        """
        apply_horizontal_flip = random.random() < self.horizontal_flip_prob
        if self.partition == "test" or self.partition == "dev":
            apply_horizontal_flip = False

        apply_resample = random.random() < self.temporal_resample_prob
        if self.partition == "test" or self.partition == "dev":
            apply_resample = False
        self.sample_rate = random.uniform(0.5, 1.5)

        apply_rotation = random.random() < self.rotation_prob
        angle = random.uniform(10, -10)

        if self.partition == "test" or self.partition == "dev":
            apply_rotation = False

        row = self.df.iloc[index]
        file_id, sequence_id, phrase = row[["file_id", "sequence_id", "phrase"]]

        data = self.load_one(file_id, sequence_id)

        T = len(data)
        new_len = max(int(T * self.sample_rate), 1)
        self.indices = self.get_resampling_indices(len(data), new_len)

        data = self.proc(torch.from_numpy(data), self.filtered_columns)
        # after normalization, data center is (0,0)
        if apply_horizontal_flip:
            data[:, :, 0] = -data[:, :, 0]
            data = data[:, self.flip_array]

        if apply_rotation:
            data_tmp = None
            data = data.to(torch.float)
            # if input is xyz, split off z and re-attach later
            if data.shape[-1] == 3:
                data_tmp = data[..., 2:]
                data = data[..., :2]

            center = (0, 0)
            center = torch.tensor(center)
            data -= center
            angle2 = -angle
            radian = angle2 / 180 * np.pi
            c = math.cos(radian)
            s = math.sin(radian)

            rotate_mat = torch.tensor([[c, s], [-s, c]])

            data = data @ rotate_mat
            data = data + center
            data = torch.cat([data, data_tmp], axis=-1)

        if apply_resample:
            data = data[self.indices]

        file_name = f"{row['file_id']}/{row['sequence_id']}"

        img_paths = (Path(self.path_to_rgb) / file_name).glob("*.jpg")
        img_names = sorted([f for f in img_paths])
        imgs = []

        for img_name in img_names:
            if apply_horizontal_flip:
                img = cv2.resize(cv2.imread(str(img_name)), (self.img_size, self.img_size))[
                    :, ::-1, ::-1
                ].copy()  # to rgb, using numpy faster
            else:
                img = cv2.resize(cv2.imread(str(img_name)), (self.img_size, self.img_size))[
                    :, :, ::-1
                ].copy()  # to rgb, using numpy faster

            imgs.append(img)

        if apply_rotation:
            imgs = self.random_rotation.rotate(imgs, angle)

        # apply resize, normalize, toTensor
        for i in range(len(imgs)):
            img = self.transforms(image=imgs[i])["image"]
            imgs[i] = img

        imgs = torch.stack(imgs)

        if apply_resample:
            imgs = imgs[self.indices]

        label_ctc = self.targets_enc_ctc[index]
        return imgs, data, torch.tensor(label_ctc), len(imgs), len(label_ctc)

    def collate_fn(self, batch):
        """
        Collates a batch of samples into tensors for model input.

        Parameters
        ----------
        batch : list
            A list of tuples, where each tuple contains:
            - imgs: torch.Tensor (input images)
            - landmarks: torch.Tensor (landmark data)
            - labels_ctc: torch.Tensor (CTC labels)
            - input_lengths: int (length of each input sequence)
            - target_lengths: int (length of each target sequence)

        Returns
        -------
        tuple
            A tuple containing:
            - imgs: torch.FloatTensor (concatenated input images)
            - landmarks: torch.FloatTensor (padded landmark data)
            - labels_ctc: torch.LongTensor (padded CTC labels)
            - input_lengths: torch.LongTensor (tensor of input sequence lengths)
            - target_lengths: torch.LongTensor (tensor of target sequence lengths)
        """
        imgs = []
        labels_ctc = []
        input_lengths = []
        target_lengths = []
        landmarks = []

        for sample in batch:
            imgs.append(sample[0])
            landmarks.append(sample[1])
            labels_ctc.append(sample[2])
            input_lengths.append(sample[3])  # Length of each input sequence
            target_lengths.append(sample[4])  # Length of each target sequence

        imgs = torch.concat(imgs, dim=0)  # bs collate here

        labels_ctc = pad_sequence(labels_ctc, True, len(self.char_list_ctc) + 1)

        landmarks = nn.utils.rnn.pad_sequence(landmarks, batch_first=True, padding_value=0.0)
        return (
            imgs.float(),
            landmarks.float(),
            labels_ctc.long(),
            torch.tensor(input_lengths).long(),
            torch.tensor(target_lengths).long(),
        )


class JointDatasetDataModule:

    """
    A data module for managing datasets and dataloaders for training, validation, and testing
    in a PyTorch-based pipeline.

    Parameters
    ----------
    test_df : str
        Path to the CSV file containing metadata for the test dataset.
    train_df : str
        Path to the CSV file containing metadata for the training and validation datasets.
    path_to_rgb : str
        Path to the directory containing RGB image files.
    path_to_landmarks : str
        Path to the directory containing landmark data files.
    batch_size : int
        Number of samples per batch for the dataloaders.
    num_workers : int
        Number of workers for loading data in parallel.
    img_size : int
        Image size for resizing input images.
    inference_args : str
        Additional arguments for inference configuration.
    symmetry_fp : str
        Path to the file containing symmetry information for landmarks.
    horizontal_flip_prob : float, optional
        Probability of applying horizontal flip augmentation. Default is 0.5.
    temporal_resample_prob : float, optional
        Probability of resampling temporal sequences. Default is 0.8.
    rotatation_prob : float, optional
        Probability of applying rotation augmentation. Default is 0.5.
    inds_to_filter : list[str], optional
        List of indices to filter out from the dataset. Default is None.

    Attributes
    ----------
    data_train : Optional[Dataset]
        Training dataset.
    data_val : Optional[Dataset]
        Validation dataset.
    data_test : Optional[Dataset]
        Test dataset.
    transfomrs_train : albumentations.core.composition.Compose
        Transformations applied to training data.
    transfomrs_val : albumentations.core.composition.Compose
        Transformations applied to validation data.

    Methods
    -------
    setup_data()
        Sets up datasets for training, validation, and testing.
    get_dataset(stage: str)
        Returns the dataset corresponding to the specified stage ('train', 'val', or 'test').
    get_dataloader(stage: str)
        Returns the dataloader for the specified stage ('train', 'val', or 'test').
    """

    def __init__(
        self,
        test_df: str,
        train_df: str,
        path_to_rgb: str,
        path_to_landmarks: str,
        batch_size: int,
        num_workers: int,
        img_size: int,
        inference_args: str,
        symmetry_fp: str,
        horizontal_flip_prob: float = 0.5,
        temporal_resample_prob: float = 0.8,
        rotatation_prob: float = 0.5,
        inds_to_filter: list[str] = None,
    ):
        """
        Initializes the JointDatasetDataModule.

        Parameters
        ----------
        test_df : str
            Path to the CSV file containing metadata for the test dataset.
        train_df : str
            Path to the CSV file containing metadata for the training and validation datasets.
        path_to_rgb : str
            Path to the directory containing RGB image files.
        path_to_landmarks : str
            Path to the directory containing landmark data files.
        batch_size : int
            Number of samples per batch for the dataloaders.
        num_workers : int
            Number of workers for loading data in parallel.
        img_size : int
            Image size for resizing input images.
        inference_args : str
            Additional arguments for inference configuration.
        symmetry_fp : str
            Path to the file containing symmetry information for landmarks.
        horizontal_flip_prob : float, optional
            Probability of applying horizontal flip augmentation. Default is 0.5.
        temporal_resample_prob : float, optional
            Probability of resampling temporal sequences. Default is 0.8.
        rotatation_prob : float, optional
            Probability of applying rotation augmentation. Default is 0.5.
        inds_to_filter : list[str], optional
            List of indices to filter out from the dataset. Default is None.

        Attributes
        ----------
        data_train : Optional[Dataset]
            Training dataset.
        data_val : Optional[Dataset]
            Validation dataset.
        data_test : Optional[Dataset]
            Test dataset.
        transfomrs_train : albumentations.core.composition.Compose
            Transformations applied to training data.
        transfomrs_val : albumentations.core.composition.Compose
            Transformations applied to validation data.
        """
        super().__init__()
        self.path_to_rgb = path_to_rgb
        self.path_to_landmarks = path_to_landmarks
        self.test_df = test_df
        self.train_df = train_df
        self.img_size = img_size

        self.inference_args = inference_args
        self.inds_to_filter = inds_to_filter

        self.symmetry_fp = symmetry_fp

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.horizontal_flip_prob = horizontal_flip_prob
        self.temporal_resample_prob = temporal_resample_prob
        self.rotatation_prob = rotatation_prob

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.transfomrs_train = A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.Resize(img_size, img_size),
                ToTensorV2(),
            ]
        )
        self.transfomrs_val = A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.Resize(img_size, img_size),
                ToTensorV2(),
            ]
        )

    def setup_data(
        self,
    ):
        """
        Initializes the datasets for training, validation, and test sets.

        This method initializes the datasets using the get_dataset method and
        assigns them to the respective attributes (data_train, data_val, data_test).

        Returns
        -------
        None
        """

        self.data_train = self.get_dataset("train")
        self.data_test = self.get_dataset("test")
        self.data_val = self.get_dataset("val")

    def get_dataset(self, stage: str):
        """
        Returns the dataset corresponding to the specified stage ('train', 'val', or 'test')


        Parameters
        ----------
        stage : str
            The stage of the dataset to return ('train', 'val', or 'test').


        Returns
        -------
        dataset : Dataset
            The dataset corresponding to the specified stage.
        """
        LOGGER.info(f"Ininting {stage} dataset")

        if stage == "train":
            df = pd.read_csv(self.train_df)
            train_df = df[df["fold"] != 1].copy()

            self.data_train = JointDataset(
                path_to_rgb=self.path_to_rgb,
                path_to_landmarks=self.path_to_landmarks,
                partition="train",
                df=train_df,
                transforms=self.transfomrs_train,
                img_size=self.img_size,
                horizontal_flip_prob=self.horizontal_flip_prob,
                temporal_resample_prob=self.temporal_resample_prob,
                rotatation_prob=self.rotatation_prob,
                inds_to_filter=self.inds_to_filter,
                inference_args=self.inference_args,
                symmetry_fp=self.symmetry_fp,
            )
            LOGGER.info(f"len is {len(self.data_train)}")
            return self.data_train

        elif stage == "val":
            df = pd.read_csv(self.train_df)
            val_df = df[df["fold"] == 1].copy()
            self.data_val = JointDataset(
                path_to_rgb=self.path_to_rgb,
                path_to_landmarks=self.path_to_landmarks,
                partition="dev",
                df=val_df,
                transforms=self.transfomrs_train,
                img_size=self.img_size,
                horizontal_flip_prob=self.horizontal_flip_prob,
                temporal_resample_prob=self.temporal_resample_prob,
                rotatation_prob=self.rotatation_prob,
                inds_to_filter=self.inds_to_filter,
                inference_args=self.inference_args,
                symmetry_fp=self.symmetry_fp,
            )
            LOGGER.info(f"len is {len(self.data_val)}")
            return self.data_val

        elif stage == "test":
            df = pd.read_csv(self.test_df)
            test_df = df[df["fold"] == 0].copy()

            self.data_test = JointDataset(
                path_to_rgb=self.path_to_rgb,
                path_to_landmarks=self.path_to_landmarks,
                partition="test",
                df=test_df,
                transforms=self.transfomrs_train,
                img_size=self.img_size,
                horizontal_flip_prob=self.horizontal_flip_prob,
                temporal_resample_prob=self.temporal_resample_prob,
                rotatation_prob=self.rotatation_prob,
                inds_to_filter=self.inds_to_filter,
                inference_args=self.inference_args,
                symmetry_fp=self.symmetry_fp,
            )
            LOGGER.info(f"len is {len(self.data_test)}")
            return self.data_test

    def get_dataloader(self, stage: str):
        LOGGER.info(f"Ininting {stage} dataloader")
        if stage == "train":
            self.data_train = self.get_dataset("train")
            sampler = None
            shuffle = True
            self.train_loader = DataLoader(
                self.data_train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self.data_train.collate_fn,
                shuffle=shuffle,
                sampler=sampler,
            )
            return self.train_loader

        elif stage == "val":
            self.data_val = self.get_dataset("val")
            sampler = None

            self.val_loader = DataLoader(
                self.data_val,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self.data_val.collate_fn,
                shuffle=False,
                sampler=sampler,
            )
            return self.val_loader

        elif stage == "test":
            self.data_test = self.get_dataset("test")
            self.test_loader = DataLoader(
                self.data_test,
                batch_size=1,
                num_workers=self.num_workers,
                collate_fn=self.data_test.collate_fn,
                shuffle=False,
            )
            return self.test_loader
