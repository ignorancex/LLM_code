import random
from pathlib import Path
from typing import Any, Optional

import albumentations as A
import cv2
import h5py
import numpy as np
import pandas as pd
import rootutils
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from turbojpeg import TJCS_RGB, TurboJPEG

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.data_utils import get_vocab, getChicagoTokens, getRuTokens, numerize
from src.data.KP_RGB.augmentations import RandomRotation, TemporalRescale
from src.utils import LOGGER


class ChicagoDataset(Dataset):
    """
    A PyTorch dataset for handling ChicagoCrop Wristle data with temporal, spatial, and
    augmentative preprocessing.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing dataset metadata.
    partition : str
        Dataset partition to load ('train', 'test', or 'dev').
    path_to_data : str
        Path to the directory containing the image data.
    transforms : callable
        Transformations to apply to the images, such as resizing and normalization.
    horizontal_flip_prob : float, optional
        Probability of applying horizontal flip augmentation. Default is 0.5.
    temporal_resample_prob : float, optional
        Probability of resampling temporal sequences. Default is 0.8.
    rotatation_prob : float, optional
        Probability of applying rotation augmentation. Default is 0.5.

    Attributes
    ----------
    data_partition : pandas.DataFrame
        Subset of the dataset corresponding to the specified partition.
    filenames : list[str]
        List of file names for the images in the dataset.
    labels : pandas.Series
        Processed labels for the dataset.
    vocab_map_ctc : dict
        Mapping from characters to numeric indices for CTC-based tokenization.
    inv_vocab_map_ctc : dict
        Reverse mapping from numeric indices to characters for CTC.
    char_list_ctc : list[str]
        List of characters in the CTC vocabulary.
    targets_enc_ctc : list
        List of tokenized labels for CTC tasks.
    num_frames : list[int]
        Number of frames for each sample in the dataset.

    Methods
    -------
    __len__()
        Returns the number of samples in the dataset.
    __getitem__(index) -> Tuple[torch.Tensor, torch.Tensor, int, int]
        Retrieves the sample at the specified index, including images, CTC labels,
        and sequence lengths.
    collate_fn(batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Custom collate function for batching, handling variable-length sequences.
    """

    def __init__(
        self,
        csv_path: str,
        partition: str,
        path_to_data: str,
        transforms,
        horizontal_flip_prob: float = 0.5,
        temporal_resample_prob: float = 0.8,
        rotatation_prob: float = 0.5,
    ):
        """
        Initializes the ChicagoDataset.

        Parameters
        ----------
        csv_path : str
            Path to the CSV file containing metadata and split information.
        partition : str
            Partition to load (train, dev, or test).
        path_to_data : str
            Path to the directory containing RGB image files.
        transforms
            Transforms to apply to the data.
        horizontal_flip_prob : float, optional
            Probability of applying horizontal flip augmentation. Default is 0.5.
        temporal_resample_prob : float, optional
            Probability of resampling temporal sequences. Default is 0.8.
        rotatation_prob : float, optional
            Probability of applying rotation augmentation. Default is 0.5.

        Attributes
        ----------
        data_partition : pandas.DataFrame
            Subset of the dataset corresponding to the specified partition.
        filenames : list[str]
            List of file names for the images in the dataset.
        labels : pandas.Series
            Processed labels for the dataset.
        vocab_map_ctc : dict
            Mapping from characters to numeric indices for CTC-based tokenization.
        inv_vocab_map_ctc : dict
            Reverse mapping from numeric indices to characters for CTC.
        char_list_ctc : list[str]
            List of characters in the CTC vocabulary.
        targets_enc_ctc : list
            List of tokenized labels for CTC tasks.
        num_frames : list[int]
            Number of frames for each sample in the dataset.
        """
        super().__init__()
        data = pd.read_csv(csv_path)

        self.data_partition = data[data["partition"] == partition]
        self.partition = partition
        self.horizontal_flip_prob = horizontal_flip_prob
        self.temporal_resample_prob = temporal_resample_prob
        self.rotation_prob = rotatation_prob
        self.temporal_resample = TemporalRescale((0.5, 1.5))
        self.random_rotation = RandomRotation()

        self.filenames = self.data_partition["filename"].to_list()
        self.labels = self.data_partition["label_proc"]

        chars = getChicagoTokens()

        self.vocab_map_ctc, self.inv_vocab_map_ctc, self.char_list_ctc = get_vocab(chars)

        self.targets_enc_ctc = numerize(self.labels, self.vocab_map_ctc, False)

        self.path_to_data = path_to_data
        self.transforms = transforms
        self.num_frames = self.data_partition["number_of_frames"].to_list()

    def __len__(
        self,
    ):
        """
        Returns the number of samples in the dataset.

        Returns
        -------
        int
            The length of the dataset, i.e., the number of filenames.
        """
        return len(self.filenames)

    def __getitem__(self, index) -> Any:
        """
        Retrieves the sample at the specified index, including images, CTC labels,
        and sequence lengths.

        Parameters
        ----------
        index : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple
            A tuple containing the sample, including:
            - imgs : torch.Tensor
                A tensor of shape (T, H, W, C) containing the RGB image frames.
            - label_ctc : torch.Tensor
                A tensor of shape (S,) containing the CTC labels.
            - num_frames : int
                The number of frames in the sample.
            - num_labels : int
                The number of labels in the sample.
        """
        apply_horizontal_flip = random.random() < self.horizontal_flip_prob
        if self.partition == "test" or self.partition == "dev":
            apply_horizontal_flip = False

        apply_resample = random.random() < self.temporal_resample_prob
        if self.partition == "test" or self.partition == "dev":
            apply_resample = False

        apply_rotation = random.random() < self.rotation_prob
        if self.partition == "test" or self.partition == "dev":
            apply_rotation = False

        img_paths = (Path(self.path_to_data) / self.filenames[index]).glob("*.jpg")
        img_names = sorted([f for f in img_paths])
        imgs = []

        for img_name in img_names:
            if apply_horizontal_flip:
                img = cv2.imread(str(img_name))[:, ::-1, ::-1].copy()  # to rgb, using numpy faster
            else:
                img = cv2.imread(str(img_name))[:, :, ::-1].copy()  # to rgb, using numpy faster

            imgs.append(img)

        if apply_rotation:
            imgs = self.random_rotation(imgs)

        # apply resize, normalize, toTensor
        for i in range(len(imgs)):
            img = self.transforms(image=imgs[i])["image"]
            imgs[i] = img

        imgs = torch.stack(imgs)

        if apply_resample:
            imgs = self.temporal_resample(imgs)

        label_ctc = self.targets_enc_ctc[index]

        return imgs, torch.tensor(label_ctc), len(imgs), len(label_ctc)

    def collate_fn(self, batch):
        """
        Collates a batch of samples into a format suitable for model input.

        Parameters
        ----------
        batch : list
            A list of tuples, where each tuple contains:
            - imgs: torch.Tensor (input images)
            - labels_ctc: torch.Tensor (CTC labels)
            - input_lengths: int (length of each input sequence)
            - target_lengths: int (length of each target sequence)

        Returns
        -------
        tuple
            A tuple containing:
            - imgs: torch.FloatTensor (concatenated input images)
            - labels_ctc: torch.LongTensor (padded CTC labels)
            - input_lengths: torch.LongTensor (tensor of input sequence lengths)
            - target_lengths: torch.LongTensor (tensor of target sequence lengths)
        """
        imgs = []
        labels_ctc = []
        input_lengths = []
        target_lengths = []

        for sample in batch:
            imgs.append(sample[0])
            labels_ctc.append(sample[1])
            input_lengths.append(sample[2])  # Length of each input sequence
            target_lengths.append(sample[3])  # Length of each target sequence

        imgs = torch.concat(imgs, dim=0)  # bs collate here

        labels_ctc = pad_sequence(labels_ctc, True, len(self.char_list_ctc) + 1)

        return (
            imgs.float(),
            labels_ctc.long(),
            torch.tensor(input_lengths).long(),
            torch.tensor(target_lengths).long(),
        )


class ChicagoDataModule:
    """
    Data module for the ChicagoCropWristleMediaPipe dataset. Handles dataset initialization and
    DataLoader creation for training, validation, and testing stages.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing dataset metadata.
    path_to_data : str
        Path to the directory containing image data.
    batch_size : int
        Batch size for DataLoader.
    num_workers : int
        Number of worker threads for DataLoader.
    img_size : int
        Target size for resizing images.
    horizontal_flip_prob : float, optional
        Probability of applying horizontal flip augmentation (default is 0.5).
    temporal_resample_prob : float, optional
        Probability of applying temporal resampling augmentation (default is 0.8).
    rotatation_prob : float, optional
        Probability of applying random rotation augmentation (default is 0.5).

    Attributes
    ----------
    data_train : Optional[Dataset]
        Training dataset instance.
    data_val : Optional[Dataset]
        Validation dataset instance.
    data_test : Optional[Dataset]
        Testing dataset instance.
    transfomrs_train : albumentations.Compose
        Transformation pipeline for training data.
    transfomrs_val : albumentations.Compose
        Transformation pipeline for validation and test data.

    Methods
    -------
    setup_data()
        Initializes datasets for training, validation, and testing stages.
    get_dataset(stage: str)
        Returns a dataset instance for the specified stage ('train', 'val', 'test').
    get_dataloader(stage: str)
        Returns a DataLoader for the specified stage ('train', 'val', 'test').
    """

    def __init__(
        self,
        csv_path: str,
        path_to_data: str,
        batch_size: int,
        num_workers: int,
        img_size: int,
        horizontal_flip_prob: float = 0.5,
        temporal_resample_prob: float = 0.8,
        rotatation_prob: float = 0.5,
    ):
        """
        Initializes the ChicagoDataModule with dataset metadata and augmentation settings.

        Parameters
        ----------
        csv_path : str
            Path to the CSV file containing dataset metadata.
        path_to_data : str
            Path to the directory containing image data.
        batch_size : int
            Batch size for DataLoader.
        num_workers : int
            Number of worker threads for DataLoader.
        img_size : int
            Target size for resizing images.
        horizontal_flip_prob : float, optional
            Probability of applying horizontal flip augmentation (default is 0.5).
        temporal_resample_prob : float, optional
            Probability of applying temporal resampling augmentation (default is 0.8).
        rotatation_prob : float, optional
            Probability of applying random rotation augmentation (default is 0.5).

        Attributes
        ----------
        data_train : Optional[Dataset]
            Training dataset instance.
        data_val : Optional[Dataset]
            Validation dataset instance.
        data_test : Optional[Dataset]
            Testing dataset instance.
        transfomrs_train : albumentations.Compose
            Transformation pipeline for training data.
        transfomrs_val : albumentations.Compose
            Transformation pipeline for validation and test data.
        """
        super().__init__()

        self.csv_path = csv_path
        self.path_to_data = path_to_data
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
        Returns the dataset corresponding to the specified stage ('train', 'val', or 'test').

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
            self.data_train = ChicagoDataset(
                csv_path=self.csv_path,
                partition="train",
                path_to_data=self.path_to_data,
                transforms=self.transfomrs_train,
                horizontal_flip_prob=self.horizontal_flip_prob,
                temporal_resample_prob=self.temporal_resample_prob,
                rotatation_prob=self.rotatation_prob,
            )
            return self.data_train

        elif stage == "val":
            self.data_val = ChicagoDataset(
                csv_path=self.csv_path,
                partition="dev",
                path_to_data=self.path_to_data,
                transforms=self.transfomrs_val,
                horizontal_flip_prob=self.horizontal_flip_prob,
                temporal_resample_prob=self.temporal_resample_prob,
                rotatation_prob=self.rotatation_prob,
            )
            return self.data_val

        elif stage == "test":
            self.data_test = ChicagoDataset(
                csv_path=self.csv_path,
                partition="test",
                path_to_data=self.path_to_data,
                transforms=self.transfomrs_val,
                horizontal_flip_prob=self.horizontal_flip_prob,
                temporal_resample_prob=self.temporal_resample_prob,
                rotatation_prob=self.rotatation_prob,
            )
            return self.data_test

    def get_dataloader(self, stage: str):
        """Gets a PyTorch DataLoader for the given stage.

        This method will return a different DataLoader depending on the stage provided.

        Parameters
        ----------
        stage : str
            The stage for which to get the DataLoader. Must be one of "train", "val", or "test".

        Returns
        -------
        torch.utils.data.DataLoader
            The DataLoader for the given stage.
        """

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
            LOGGER.info(f"train dataset len: {len(self.data_train)}")
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
            LOGGER.info(f"val dataset len: {len(self.data_val)}")
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
            LOGGER.info(f"test dataset len: {len(self.data_test)}")
            return self.test_loader


class DatasetRU(Dataset):
    """
    A dataset class for Russian sign language processing, supporting temporal and spatial augmentations,
    and handling HDF5-stored video frames.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing metadata for the dataset.
    partition : str
        The partition of the dataset ('train', 'val', or 'test').
    path_to_data : str
        Path to the directory or HDF5 file containing video data.
    transforms : callable
        A function or transform to apply to each frame (e.g., resizing, normalization).
    horizontal_flip_prob : float, optional
        Probability of applying horizontal flip augmentation (default is 0.5).
    temporal_resample_prob : float, optional
        Probability of applying temporal resampling augmentation (default is 0.8).
    rotatation_prob : float, optional
        Probability of applying random rotation augmentation (default is 0.5).

    Attributes
    ----------
    data_partition : pd.DataFrame
        Subset of the dataset corresponding to the specified partition.
    filenames : list
        List of video identifiers corresponding to the dataset.
    labels : pd.Series
        Preprocessed text labels for the dataset.
    vocab_map_ctc : dict
        Mapping of characters to indices for CTC (Connectionist Temporal Classification).
    inv_vocab_map_ctc : dict
        Mapping of indices back to characters for CTC.
    char_list_ctc : list
        List of all characters in the vocabulary.
    targets_enc_ctc : list
        Encoded labels for each sample in the dataset.
    transforms : callable
        Transformations to apply to each video frame.
    temporal_resample : callable
        Augmentation function for resampling temporal sequences.
    random_rotation : callable
        Augmentation function for applying random rotation to frames.

    Methods
    -------
    rotate_image(img, angle)
        Rotates a given image by a specified angle.
    load_hdf5_frames(path_hdf5: str, video_name: str)
        Loads frames for a specified video from an HDF5 file.
    __getitem__(index)
        Retrieves a single sample (video frames and labels) at the specified index.
    __len__()
        Returns the number of samples in the dataset.
    collate_fn(batch)
        Custom collation function for handling batches of varying sequence lengths.
    """

    def __init__(
        self,
        csv_path: str,
        partition: str,
        path_to_data: str,
        transforms,
        horizontal_flip_prob: float = 0.5,
        temporal_resample_prob: float = 0.8,
        rotatation_prob: float = 0.5,
    ) -> None:
        """
        Initializes the Russian sign language dataset.

        Parameters
        ----------
        csv_path : str
            Path to the CSV file containing metadata for the dataset.
        partition : str
            The partition of the dataset ('train', 'val', or 'test').
        path_to_data : str
            Path to the directory or HDF5 file containing video data.
        transforms : callable
            A function or transform to apply to each frame (e.g., resizing, normalization).
        horizontal_flip_prob : float, optional
            Probability of applying horizontal flip augmentation (default is 0.5).
        temporal_resample_prob : float, optional
            Probability of applying temporal resampling augmentation (default is 0.8).
        rotatation_prob : float, optional
            Probability of applying random rotation augmentation (default is 0.5).

        Attributes
        ----------
        data_partition : pd.DataFrame
            Subset of the dataset corresponding to the specified partition.
        filenames : list
            List of video identifiers corresponding to the dataset.
        labels : pd.Series
            Preprocessed text labels for the dataset.
        vocab_map_ctc : dict
            Mapping of characters to indices for CTC (Connectionist Temporal Classification).
        inv_vocab_map_ctc : dict
            Mapping of indices back to characters for CTC.
        char_list_ctc : list
            List of all characters in the vocabulary.
        targets_enc_ctc : list
            Encoded labels for each sample in the dataset.
        transforms : callable
            Transformations to apply to each video frame.
        temporal_resample : callable
            Augmentation function for resampling temporal sequences.
        random_rotation : callable
            Augmentation function for applying random rotation to frames.
        """
        super().__init__()

        data = pd.read_csv(csv_path)

        self.data_partition = data[data["split"] == partition]

        self.partition = partition
        self.horizontal_flip_prob = horizontal_flip_prob
        self.temporal_resample_prob = temporal_resample_prob
        self.rotation_prob = rotatation_prob
        self.temporal_resample = TemporalRescale((0.5, 1.5))
        self.random_rotation = RandomRotation()

        self.labels = self.data_partition["processed_text"]
        self.filenames = self.data_partition["attachment_id"].to_list()

        chars = getRuTokens()

        self.vocab_map_ctc, self.inv_vocab_map_ctc, self.char_list_ctc = get_vocab(chars)

        self.targets_enc_ctc = numerize(self.labels, self.vocab_map_ctc, False)

        self.path_to_data = path_to_data
        self.transforms = transforms

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns
        -------
        int
            The length of the dataset, i.e., the number of filenames.
        """
        return len(self.filenames)

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

                # Decode the JPEG binary data back to an image
                frame = jpeg.decode(jpeg_data, pixel_format=TJCS_RGB)
                if self.apply_horizontal_flip:
                    frame = frame[:, ::-1]
                if self.apply_rotation:
                    frame = self.rotate_image(frame, self.angle)
                imgs.append(self.transforms(image=frame)["image"])
        return imgs

    def __getitem__(self, index):
        """
        Retrieves a single data sample and its associated target from the dataset.

        Parameters
        ----------
        index : int
            Index of the data sample to retrieve.

        Returns
        -------
        tuple
            A tuple containing:
            - data : torch.Tensor
                The input video tensor.
            - token_ids : torch.Tensor
                The encoded target phrase as a tensor of integers.
            - input_length : int
                The length of the input sequence.
            - target_len : int
                The length of the target sequence.
        """

        self.apply_horizontal_flip = random.random() < self.horizontal_flip_prob
        if self.partition == "test" or self.partition == "val":
            self.apply_horizontal_flip = False

        self.apply_resample = random.random() < self.temporal_resample_prob
        if self.partition == "test" or self.partition == "val":
            self.apply_resample = False

        self.apply_rotation = random.random() < self.rotation_prob
        self.angle = random.uniform(-10, 10)
        if self.partition == "test" or self.partition == "val":
            self.apply_rotation = False

        imgs = []

        imgs = self.load_hdf5_frames(self.path_to_data, self.filenames[index])

        imgs = torch.stack(imgs)

        if self.apply_resample:
            imgs = self.temporal_resample(imgs)

        label_ctc = self.targets_enc_ctc[index]

        return imgs, torch.tensor(label_ctc), len(imgs), len(label_ctc)

    def collate_fn(self, batch):
        """
        Collates a batch of samples into a format suitable for model input.

        Parameters
        ----------
        batch : list
            A list of tuples, where each tuple contains:
            - imgs: torch.Tensor (input images)
            - labels_ctc: torch.Tensor (CTC labels)
            - input_lengths: int (length of each input sequence)
            - target_lengths: int (length of each target sequence)

        Returns
        -------
        tuple
            A tuple containing:
            - imgs: torch.FloatTensor (concatenated input images)
            - labels_ctc: torch.LongTensor (padded CTC labels)
            - input_lengths: torch.LongTensor (tensor of input sequence lengths)
            - target_lengths: torch.LongTensor (tensor of target sequence lengths)
        """
        imgs = []
        labels_ctc = []
        input_lengths = []
        target_lengths = []

        for sample in batch:
            imgs.append(sample[0])
            labels_ctc.append(sample[1])
            input_lengths.append(sample[2])  # Length of each input sequence
            target_lengths.append(sample[3])  # Length of each target sequence

        imgs = torch.concat(imgs, dim=0)  # bs collate here

        labels_ctc = pad_sequence(labels_ctc, True, len(self.char_list_ctc) + 1)

        return (
            imgs.float(),
            labels_ctc.long(),
            torch.tensor(input_lengths).long(),
            torch.tensor(target_lengths).long(),
        )


class DatsetRUDataModule:
    """
    A PyTorch data module for managing training, validation, and testing datasets,
    specifically for Russian sign language video data.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing dataset metadata.
    path_to_data : str
        Path to the directory or HDF5 file containing video data.
    batch_size : int
        Number of samples per batch for data loading.
    num_workers : int
        Number of subprocesses to use for data loading.
    img_size : int
        Target size to which video frames will be resized.
    horizontal_flip_prob : float, optional
        Probability of applying horizontal flip augmentation during training (default is 0.5).
    temporal_resample_prob : float, optional
        Probability of applying temporal resampling augmentation during training (default is 0.8).
    rotatation_prob : float, optional
        Probability of applying random rotation augmentation during training (default is 0.5).

    Attributes
    ----------
    csv_path : str
        Path to the CSV metadata file.
    path_to_data : str
        Path to the video data storage.
    batch_size : int
        Batch size for the DataLoader.
    num_workers : int
        Number of workers for DataLoader.
    horizontal_flip_prob : float
        Probability of horizontal flip during training.
    temporal_resample_prob : float
        Probability of temporal resampling during training.
    rotatation_prob : float
        Probability of rotation augmentation during training.
    data_train : Optional[Dataset]
        Training dataset instance.
    data_val : Optional[Dataset]
        Validation dataset instance.
    data_test : Optional[Dataset]
        Test dataset instance.
    transfomrs_train : callable
        Transformation pipeline for training data.
    transfomrs_val : callable
        Transformation pipeline for validation and test data.

    Methods
    -------
    setup_data()
        Sets up training, validation, and testing datasets.
    get_dataset(stage: str)
        Initializes and returns the dataset for the specified stage ('train', 'val', 'test').
    get_dataloader(stage: str)
        Initializes and returns the DataLoader for the specified stage ('train', 'val', 'test').

    """

    def __init__(
        self,
        csv_path: str,
        path_to_data: str,
        batch_size: int,
        num_workers: int,
        img_size: int,
        horizontal_flip_prob: float = 0.5,
        temporal_resample_prob: float = 0.8,
        rotatation_prob: float = 0.5,
    ):
        """
        Initializes a DatsetRUDataModule instance.

        Parameters
        ----------
        csv_path : str
            Path to the CSV file containing dataset metadata.
        path_to_data : str
            Path to the directory or HDF5 file containing video data.
        batch_size : int
            Batch size for the DataLoader.
        num_workers : int
            Number of worker threads to use for data loading.
        img_size : int
            Target size to which video frames will be resized.
        horizontal_flip_prob : float, optional
            Probability of applying horizontal flip augmentation during training (default is 0.5).
        temporal_resample_prob : float, optional
            Probability of applying temporal resampling augmentation during training (default is 0.8).
        rotatation_prob : float, optional
            Probability of applying random rotation augmentation during training (default is 0.5).

        Attributes
        ----------
        csv_path : str
            Path to the CSV metadata file.
        path_to_data : str
            Path to the video data storage.
        batch_size : int
            Batch size for the DataLoader.
        num_workers : int
            Number of workers for DataLoader.
        horizontal_flip_prob : float
            Probability of horizontal flip during training.
        temporal_resample_prob : float
            Probability of temporal resampling during training.
        rotatation_prob : float
            Probability of rotation augmentation during training.
        data_train : Optional[Dataset]
            Training dataset instance.
        data_val : Optional[Dataset]
            Validation dataset instance.
        data_test : Optional[Dataset]
            Test dataset instance.
        transfomrs_train : callable
            Transformation pipeline for training data.
        transfomrs_val : callable
            Transformation pipeline for validation and test data.
        """
        super().__init__()

        self.csv_path = csv_path
        self.path_to_data = path_to_data
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
        Initializes the datasets for training, validation, and testing.

        This method uses the `get_dataset` method to initialize and assign the
        datasets for the 'train', 'test', and 'val' stages to the respective
        attributes: `data_train`, `data_test`, and `data_val`.

        Returns
        -------
        None
        """
        self.data_train = self.get_dataset("train")
        self.data_test = self.get_dataset("test")
        self.data_val = self.get_dataset("val")

    def get_dataset(self, stage: str):
        """
        Initializes and returns the dataset for the specified stage ('train', 'val', 'test').


        Parameters
        ----------
        stage : str
            Stage for which to initialize the dataset ('train', 'val', 'test').


        Returns
        -------
        dataset : DatasetRU
            Dataset instance for the specified stage.

        Notes
        -----
        This method is used to initialize the datasets for training, validation, and testing.
        """
        if stage == "train":
            LOGGER.info(f"Ininting {stage} dataset")
            self.data_train = DatasetRU(
                csv_path=self.csv_path,
                partition="train",
                path_to_data=self.path_to_data,
                transforms=self.transfomrs_train,
                horizontal_flip_prob=self.horizontal_flip_prob,
                temporal_resample_prob=self.temporal_resample_prob,
                rotatation_prob=self.rotatation_prob,
            )
            return self.data_train

        elif stage == "val":
            LOGGER.info(f"Ininting {stage} dataset")
            self.data_val = DatasetRU(
                csv_path=self.csv_path,
                partition="val",
                path_to_data=self.path_to_data,
                transforms=self.transfomrs_val,
                horizontal_flip_prob=self.horizontal_flip_prob,
                temporal_resample_prob=self.temporal_resample_prob,
                rotatation_prob=self.rotatation_prob,
            )
            return self.data_val

        elif stage == "test":
            LOGGER.info(f"Ininting {stage} dataset")
            self.data_test = DatasetRU(
                csv_path=self.csv_path,
                partition="test",
                path_to_data=self.path_to_data,
                transforms=self.transfomrs_val,
                horizontal_flip_prob=self.horizontal_flip_prob,
                temporal_resample_prob=self.temporal_resample_prob,
                rotatation_prob=self.rotatation_prob,
            )
            return self.data_test

    def get_dataloader(self, stage: str):
        """
        Returns a PyTorch DataLoader for the specified stage ('train', 'val', or 'test').

        Parameters
        ----------
        stage : str
            The stage for which to get the DataLoader. Must be one of "train", "val", or "test".

        Returns
        -------
        torch.utils.data.DataLoader
            The DataLoader for the given stage with appropriate shuffling and batching settings.
        """
        if stage == "train":
            LOGGER.info(f"Ininting {stage} dataloader")
            self.data_train = self.get_dataset("train")
            sampler = None
            self.train_loader = DataLoader(
                self.data_train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self.data_train.collate_fn,
                shuffle=True,
                sampler=sampler,
            )
            return self.train_loader

        elif stage == "val":
            LOGGER.info(f"Ininting {stage} dataloader")
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
            LOGGER.info(f"Ininting {stage} dataloader")
            self.data_test = self.get_dataset("test")
            self.test_loader = DataLoader(
                self.data_test,
                batch_size=1,
                num_workers=self.num_workers,
                collate_fn=self.data_test.collate_fn,
                shuffle=False,
            )
            return self.test_loader
