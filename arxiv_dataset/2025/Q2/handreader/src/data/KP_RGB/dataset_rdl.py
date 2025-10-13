import json
import math
import random
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

import h5py
import torch.nn as nn
from turbojpeg import TJCS_RGB, TurboJPEG

from src.data.data_utils import Preprocessing, get_vocab, getRuTokens, numerize
from src.data.KP_RGB.augmentations import RandomRotation, TemporalRescale
from src.utils import LOGGER


class JointDatasetRU(Dataset):
    """
    A PyTorch dataset class for joint RGB and landmarks-based sequence learning tasks.

    This class represents a dataset where each sample consists of RGB frames and corresponding landmarks
    (such as keypoints) that can be used for a variety of tasks like pose estimation, action recognition,
    or sequence classification. The dataset supports various transformations such as temporal rescaling,
    random rotations, and horizontal flips.

    Parameters
    ----------
    path_to_rgb : str
        Path to the directory containing RGB frame sequences in HDF5 format.

    path_to_landmarks : str
        Path to the directory containing landmarks (e.g., keypoints) in `.npy` format.

    df : pandas.DataFrame
        DataFrame containing metadata for the dataset, including file paths and corresponding processed labels.

    transforms : callable
        A transformation pipeline to apply to each image sample, typically from a library like `albumentations`.

    partition : str
        The dataset partition, either "train", "val", or "test".

    img_size : int
        The size of the images (height and width) after resizing.

    inference_args : str
        Path to a JSON file containing inference settings, including the selected columns for landmarks.

    symmetry_fp : str
        Path to a CSV file containing symmetry mapping for keypoints, which allows for flipping landmarks.

    horizontal_flip_prob : float, optional, default=0.5
        Probability of applying a horizontal flip to the data.

    temporal_resample_prob : float, optional, default=0.8
        Probability of applying temporal resampling to the input video sequence.

    rotatation_prob : float, optional, default=0.5
        Probability of applying random rotations to the data.


    inds_to_filter : list of str, optional, default=None
        List of substrings to filter the columns (landmarks) from the selected columns.

    Attributes
    ----------
    transforms : callable
        A transformation pipeline to apply to each image sample, typically from a library like `albumentations`.

    proc : Preprocessing
        An instance of the `Preprocessing` class used for additional pre-processing of data (e.g., normalization).

    targets_enc_ctc : list
        A list of numerized labels (processed text) corresponding to each sample in the dataset.

    vocab_map_ctc : dict
        A dictionary mapping characters (including special tokens like `<PAD>`, `<BOS>`, `<EOS>`) to integers for CTC.

    inv_vocab_map_ctc : dict
        A dictionary mapping integers to characters for CTC.

    char_list_ctc : list
        The list of characters (including special tokens) used in the CTC model.

    flip_array : numpy.ndarray
        Array containing indices for flipping landmarks based on symmetry.

    Methods
    -------
    transform(img, transforms)
        Apply the given transformation pipeline to the input image.

    fill_nans(x)
        Replace NaN values with zeros and reshape the input tensor.

    __len__()
        Return the number of samples in the dataset.

    load_one(path_to_file)
        Load the landmark data from a `.npy` file corresponding to the given file path.

    get_resampling_indices(original_len, new_len)
        Generate indices for resampling a sequence, either by sampling randomly or using a fixed step.

    rotate_image(img, angle)
        Rotate the input image by a specified angle.

    load_hdf5_frames(path_hdf5, video_name)
        Load and decode RGB frames from an HDF5 file, applying necessary transformations (flip, rotation).

    __getitem__(index)
        Return a sample at the specified index, including the RGB frames, landmarks, and labels.

    collate_fn(batch)
        Custom collate function to handle padding and batching for sequences of varying lengths.

    """

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
        Parameters
        ----------
        path_to_rgb : str
            Path to the directory containing RGB frame sequences in HDF5 format.

        path_to_landmarks : str
            Path to the directory containing landmarks (e.g., keypoints) in `.npy` format.

        df : pandas.DataFrame
            DataFrame containing metadata for the dataset, including file paths and corresponding processed labels.

        transforms : callable
            A transformation pipeline to apply to each image sample, typically from a library like `albumentations`.

        partition : str
            The dataset partition, either "train", "val", or "test".

        img_size : int
            The size of the images (height and width) after resizing.

        inference_args : str
            Path to a JSON file containing inference settings, including the selected columns for landmarks.

        symmetry_fp : str
            Path to a CSV file containing symmetry mapping for keypoints, which allows for flipping landmarks.

        horizontal_flip_prob : float, optional, default=0.5
            Probability of applying a horizontal flip to the data.

        temporal_resample_prob : float, optional, default=0.8
            Probability of applying temporal resampling to the input video sequence.

        rotatation_prob : float, optional, default=0.5
            Probability of applying random rotations to the data.

        inds_to_filter : list of str, optional, default=None
            List of substrings to filter the columns (landmarks) from the selected columns.

        Attributes
        ----------
        transforms : callable
            A transformation pipeline to apply to each image sample, typically from a library like `albumentations`.

        proc : Preprocessing
            An instance of the `Preprocessing` class used for additional pre-processing of data (e.g., normalization).

        targets_enc_ctc : list
            A list of numerized labels (processed text) corresponding to each sample in the dataset.

        vocab_map_ctc : dict
            A dictionary mapping characters (including special tokens like `<PAD>`, `<BOS>`, `<EOS>`) to integers for CTC.

        inv_vocab_map_ctc : dict
            A dictionary mapping integers to characters for CTC.

        char_list_ctc : list
            The list of characters (including special tokens) used in the CTC model.

        flip_array : numpy.ndarray
            Array containing indices for flipping landmarks based on symmetry.
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

        self.labels = self.df["processed_text"]

        chars = getRuTokens()

        self.vocab_map_ctc, self.inv_vocab_map_ctc, self.char_list_ctc = get_vocab(chars)

        self.targets_enc_ctc = numerize(self.labels, self.vocab_map_ctc, False)

        self.transforms = transforms
        self.proc = Preprocessing()

    def transform(self, img, transforms):
        """
        Applies the given transforms to the input image.

        Args:
            img (PIL.Image): Input image
            transforms (albumentations.Compose): Transforms to apply

        Returns:
            torch.Tensor: Transformed image
        """
        return transforms(image=img)["image"]

    def fill_nans(self, x):
        """
        Fill NaN values in the input tensor with 0, and reshape it
        from (seq_len, 3* n_landmarks) to (seq_len, n_landmarks, 3)

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor with NaN values filled
        """

        x[torch.isnan(x)] = 0
        x = x.reshape(x.shape[0], 3, -1).permute(0, 2, 1)
        return x

    def __len__(
        self,
    ):
        """
        Returns the number of samples in the dataset.

        Returns
        -------
        int
            The number of samples in the dataset.
        """
        return len(self.df)

    def load_one(self, path_to_file: str):
        """
        Loads a single data sample from a specified file path.

        Parameters
        ----------
        path_to_file : str
            The relative path to the file containing the data sample.

        Returns
        -------
        data : np.ndarray
            The loaded data sample in the format (seq_len, 3 * nlandmarks).
        """
        path = self.path_to_landmarks + f"/{path_to_file}"
        data = np.load(path)  # seq_len, 3* nlandmarks
        return data

    def get_resampling_indices(self, original_len, new_len):
        """
        Generates a list of indices for resampling a sequence to a new length.

        This function either selects a random subset of indices if the new length
        is less than or equal to the original length, or generates evenly spaced
        indices to interpolate if the new length is greater.

        Parameters
        ----------
        original_len : int
            The length of the original sequence.
        new_len : int
            The desired new length of the sequence.

        Returns
        -------
        list of int
            A sorted list of indices for resampling the sequence.
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

    def rotate_image(self, img, angle):
        """
        Rotates an image by a specified angle while maintaining its center.

        Parameters
        ----------
        img : numpy.ndarray
            The input image to be rotated, expected in NumPy array format.
        angle : float
            The angle in degrees by which the image should be rotated.

        Returns
        -------
        numpy.ndarray
            The rotated image as a NumPy array.
        """
        size_reverse = np.array(img.shape[1::-1])
        M = cv2.getRotationMatrix2D(tuple(size_reverse / 2.0), angle, 1.0)
        MM = np.absolute(M[:, :2])
        size_new = MM @ size_reverse + 1
        M[:, -1] += (size_new - size_reverse) / 2.0
        return cv2.warpAffine(img, M, tuple(size_new.astype(int)))

    def load_hdf5_frames(self, path_hdf5: str, video_name: str):
        """
        Loads video frames from an HDF5 file and applies optional augmentations.

        Parameters
        ----------
        path_hdf5 : str
            The path to the HDF5 file containing the video data.
        video_name : str
            The name of the video group within the HDF5 file to load frames from.

        Returns
        -------
        list of torch.Tensor
            A list of transformed video frames as PyTorch tensors.
        """
        imgs = []
        jpeg = TurboJPEG()
        with h5py.File(path_hdf5, "r") as h5f:
            video_group = h5f[video_name]
            jpeg_dataset = video_group["jpeg_frames"]
            frame_count = video_group.attrs["frame_count"]
            for i in range(frame_count):
                # Get the JPEG binary data
                jpeg_data = jpeg_dataset[i]

                frame = jpeg.decode(jpeg_data, pixel_format=TJCS_RGB)
                if self.apply_horizontal_flip:
                    frame = frame[:, ::-1]
                if self.apply_rotation:
                    frame = self.rotate_image(frame, self.angle)
                imgs.append(self.transforms(image=frame)["image"])
        return imgs

    def __getitem__(self, index) -> Any:
        """
        Gets a single data point from the dataset based on the given index.

        This function applies the following augmentations to the data:

        - Random horizontal flipping of the 3D skeleton data and RGB video frames.
        - Random rotation of the 3D skeleton data and RGB video frames.
        - Temporal resampling of the 3D skeleton data and RGB video frames.

        Parameters
        ----------
        index : int
            The index of the data point to retrieve.

        Returns
        -------
        tuple of torch.Tensor, torch.Tensor, torch.Tensor, int, int
            A tuple containing the following:

            - The RGB video frames as a PyTorch tensor.
            - The 3D skeleton data as a PyTorch tensor.
            - The encoded CTC label as a PyTorch tensor.
            - The length of the RGB video frames.
            - The length of the CTC label.
        """
        self.apply_horizontal_flip = random.random() < self.horizontal_flip_prob
        if self.partition == "test" or self.partition == "val":
            self.apply_horizontal_flip = False

        self.apply_resample = random.random() < self.temporal_resample_prob
        if self.partition == "test" or self.partition == "val":
            self.apply_resample = False
        self.sample_rate = random.uniform(0.5, 1.5)

        self.apply_rotation = random.random() < self.rotation_prob
        self.angle = random.uniform(10, -10)

        if self.partition == "test" or self.partition == "val":
            self.apply_rotation = False

        row = self.df.iloc[index]
        file_path, phrase = row[["path_npy", "processed_text"]]

        data = self.load_one(file_path)

        T = len(data)
        new_len = max(int(T * self.sample_rate), 1)
        self.indices = self.get_resampling_indices(len(data), new_len)

        data = self.proc(torch.from_numpy(data), self.filtered_columns)
        if self.apply_horizontal_flip:
            data[:, :, 0] = -data[:, :, 0]
            data = data[:, self.flip_array]

        if self.apply_rotation:
            data_tmp = None
            data = data.to(torch.float)
            # if input is xyz, split off z and re-attach later
            if data.shape[-1] == 3:
                data_tmp = data[..., 2:]
                data = data[..., :2]

            center = (0, 0)
            center = torch.tensor(center)
            data -= center
            angle2 = -self.angle
            radian = angle2 / 180 * np.pi
            c = math.cos(radian)
            s = math.sin(radian)

            rotate_mat = torch.tensor([[c, s], [-s, c]])

            data = data @ rotate_mat
            data = data + center
            data = torch.cat([data, data_tmp], axis=-1)

        if self.apply_resample:
            data = data[self.indices]

        file_name = row["attachment_id"]

        imgs = self.load_hdf5_frames(self.path_to_rgb, file_name)

        imgs = torch.stack(imgs)

        if self.apply_resample:
            imgs = imgs[self.indices]

        label_ctc = self.targets_enc_ctc[index]

        return imgs, data, torch.tensor(label_ctc), len(imgs), len(label_ctc)

    def collate_fn(self, batch):
        """
        Collates a batch of samples into padded tensors for model input.

        Parameters
        ----------
        batch : list[tuple]
            List of tuples where each tuple contains:
            - data : torch.Tensor
                The processed RGB video frames.
            - landmarks : torch.Tensor
                The processed 3D skeleton data.
            - labels_ctc : torch.Tensor
                Encoded target phrase.
            - input_lengths : int
                Length of the RGB video frames.
            - target_lengths : int
                Length of the target sequence.

        Returns
        -------
        tuple
            A tuple containing:
            - data : torch.Tensor
                Padded tensor of RGB video frames.
            - landmarks : torch.Tensor
                Padded tensor of 3D skeleton data.
            - labels : torch.Tensor
                Padded tensor of encoded target phrases.
            - input_lengths : torch.Tensor
                Tensor of input sequence lengths.
            - label_lengths : torch.Tensor
                Tensor of target sequence lengths.
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

        # padding with PAD token
        labels_ctc = pad_sequence(labels_ctc, True, len(self.char_list_ctc) + 1)

        landmarks = nn.utils.rnn.pad_sequence(landmarks, batch_first=True, padding_value=0.0)
        return (
            imgs.float(),
            landmarks.float(),
            labels_ctc.long(),
            torch.tensor(input_lengths).long(),
            torch.tensor(target_lengths).long(),
        )


class JointDatasetDataModuleRU:
    """
    A data module for managing datasets and dataloaders for training, validation, and testing
    in a PyTorch-based pipeline.

    Parameters
    ----------
    df : str
        Path to the CSV file containing metadata and split information.
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
        df: str,
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
        Initializes the data module.

        Parameters
        ----------
        df : str
            Path to the CSV file containing metadata and split information.
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
        self.symmetry_fp = symmetry_fp
        self.path_to_rgb = path_to_rgb
        self.path_to_landmarks = path_to_landmarks
        self.df = df
        self.img_size = img_size

        self.inference_args = inference_args
        self.inds_to_filter = inds_to_filter

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
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
        self.transfomrs_val = A.Compose(
            [
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
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
        Initializes the dataset for a given stage (train, val, or test) and returns it.

        Parameters
        ----------
        stage : str
            The stage of the dataset to initialize and return.

        Returns
        -------
        dataset : Dataset
            The initialized dataset for the given stage.
        """
        LOGGER.info(f"Ininting {stage} dataset")

        if stage == "train":
            df = pd.read_csv(self.df)
            train_df = df[df["split"] == "train"].copy()

            self.data_train = JointDatasetRU(
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
            df = pd.read_csv(self.df)
            val_df = df[df["split"] == "val"].copy()
            self.data_val = JointDatasetRU(
                path_to_rgb=self.path_to_rgb,
                path_to_landmarks=self.path_to_landmarks,
                partition="val",
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
            df = pd.read_csv(self.df)
            test_df = df[df["split"] == "test"].copy()

            self.data_test = JointDatasetRU(
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
