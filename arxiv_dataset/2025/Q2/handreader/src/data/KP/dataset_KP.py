import json
from typing import List, Optional

import numpy as np
import pandas as pd
import rootutils
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from albumentations import Compose

import src.data.KP.augmentations as A
from src.data.data_utils import Preprocessing
from src.utils import LOGGER

BLANK_SYMBOL = "_"
BOS = "<BOS>"

chars_fsr = " &'.@abcdefghijklmnopqrstuvwxyz"


class Tokenizer:
    """
    Maps characters to integers and vice versa.

    This class handles the mapping between characters and their corresponding integer indices,
    and allows converting text to indices and vice versa. It supports two modes:
    CTC (Connectionist Temporal Classification) and autoregressive models. The class also handles
    padding, start-of-sequence (BOS), and end-of-sequence (EOS) tokens.

    Args:
    -----
    ctc : bool
        If True, includes a blank symbol for CTC training. Blank symbol is added at index 0.
    autoreg : bool
        If True, operates in autoregressive mode where each character is mapped to an index.
        If both `ctc` and `autoreg` are False, this defaults to a character-to-index mapping.

    Attributes:
    -----------
    ctc : bool
        Whether the tokenizer is operating in CTC mode.
    char_map : dict
        A dictionary mapping characters to integer indices.
    index_map : dict
        A dictionary mapping integer indices to characters.
    PAD_TOK : int
        The index of the padding token.
    EOS_TOK : int
        The index of the end-of-sequence token.
    BOS_TOK : int
        The index of the beginning-of-sequence token.

    Methods:
    --------
    text_to_indices(text):
        Maps a string to a list of integers (indices of characters).

    indices_to_text(labels):
        Maps a list of integers (indices) back to the corresponding string.

    get_symbol_index(sym):
        Returns the index for the specified symbol (character).
    """

    def __init__(self, ctc: bool, autoreg: bool):
        """
        Initializes the tokenizer with the specified settings.

        Parameters
        ----------
        ctc : bool
            If True, includes a blank symbol for CTC training. Blank symbol is added at index 0.
        autoreg : bool
            If True, operates in autoregressive mode where each character is mapped to an index.
            If both `ctc` and `autoreg` are False, this defaults to a character-to-index mapping.

        """
        self.ctc = ctc
        self.char_map = {}
        self.index_map = {}
        if self.ctc:
            for i, ch in enumerate([BLANK_SYMBOL] + list(chars_fsr)):
                self.char_map[ch] = i
                self.index_map[i] = ch
        elif autoreg:
            for i, ch in enumerate(list(chars_fsr)):
                self.char_map[ch] = i
                self.index_map[i] = ch
        self.PAD_TOK = len(self.char_map)
        self.EOS_TOK = len(self.char_map) + 1
        self.BOS_TOK = len(self.char_map) + 2

    def text_to_indices(self, text: str) -> List[int]:
        """
        Converts a string into a list of integer indices based on a character-to-index mapping.

        Parameters
        ----------
        text : str
            The input string to be converted.

        Returns
        -------
        List[int]
            A list of integers representing the indices of the characters in the input string.
        """
        return [self.char_map[ch] for ch in text]

    def indices_to_text(self, labels: List[int]) -> str:
        """
        Converts a list of integer indices back into a string based on an index-to-character mapping.

        Parameters
        ----------
        labels : List[int]
            A list of integer indices to be converted back to text.

        Returns
        -------
        str
            The resulting string corresponding to the input indices.
        """
        return "".join([self.index_map[i] for i in labels])

    def get_symbol_index(self, sym: str) -> int:
        """
        Retrieves the integer index for a specified symbol.

        Parameters
        ----------
        sym : str
            The symbol for which the index is to be retrieved.

        Returns
        -------
        int
            The index corresponding to the given symbol.
        """

        return self.char_map[sym]


def flip(data, flip_array):
    """
    Flips the input data along specified axes and reorders it based on the given flip array.

    Parameters
    ----------
    data : ndarray
        The input data array with shape (N, M, 3), where `N` and `M` are dimensions,
        and the last dimension corresponds to 3D coordinates (e.g., x, y, z).
    flip_array : array_like
        An array specifying the reordering of the data after flipping.

    Returns
    -------
    ndarray
        The modified data array with the first coordinate negated and reordered
        based on the `flip_array`.
    """
    data[:, :, 0] = -data[:, :, 0]
    data = data[:, flip_array]
    return data


class CustomDataset(Dataset):
    """
    A custom dataset for processing landmark data, with optional augmentation,
    flipping, and tokenization of labels.

    Arguments:
    ----------
    df: pd.DataFrame
        Contains the dataset information (e.g., file paths and processed text).
    data_folder: str
        The directory where the raw data files (e.g., .npy files) are stored.
    symmetry_fp: str
        Path to a CSV file containing symmetry mapping information for landmark flipping.
    flip_aug: float
        The probability of applying a flip augmentation.
    inference_args: str
        Path to a JSON file containing information about the selected columns for processing.
    tokenizer: Tokenizer
        A tokenizer object used to map labels (phrases) to integers and vice versa.
    aug: Optional[Callable]
        A function or transform to apply additional augmentations. Defaults to `None`.
    mode: Optional[str]
        The mode of the dataset ("train" or "test"). Defaults to `"train"`.
    inds_to_filter: Optional[list[str]]
        A list of substrings to filter the columns for landmark data. Defaults to `None`.


    Attributes:
    ----------
    df: pd.DataFrame
        Contains the dataset information (e.g., file paths and processed text).
    data_folder: str
        Directory path for data storage.
    mode: str
        The mode of operation, either "train" or "test".
    aug: Optional[Callable]
        The augmentation function, if provided.
    dec: Tokenizer
        The tokenizer used for label-to-index and index-to-label conversions.
    filtered_columns: Optional[list[int]]
        The list of column indices to filter based on `inds_to_filter`.
    flip_array: np.ndarray
        An array indicating the corresponding flipped landmarks for augmentation.
    processor: Preprocessing
        The preprocessing object used to normalize and fill NaNs in the data.
    flip_aug: float
        The probability of applying flip augmentation.
    xyz_landmarks: np.ndarray
        The landmarks to be used, based on column filtering.
    """

    def __init__(
        self,
        df,
        data_folder: str,
        symmetry_fp: str,
        flip_aug: float,
        inference_args: str,
        tokenizer: Tokenizer,
        aug=None,
        mode="train",
        inds_to_filter: list[str] = None,
    ):
        """
        Initializes the CustomDataset with data for landmarks, augmentation, and tokenization.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing dataset information such as file paths and processed text.
        data_folder : str
            The directory path where raw data files (e.g., .npy files) are stored.
        symmetry_fp : str
            Path to a CSV file containing symmetry mapping information for landmark flipping.
        flip_aug : float
            The probability of applying a flip augmentation.
        inference_args : str
            Path to a JSON file containing information about the selected columns for processing.
        tokenizer : Tokenizer
            A tokenizer object used to map labels (phrases) to integers and vice versa.
        aug : Optional[Callable]
            A function or transform to apply additional augmentations. Defaults to `None`.
        mode : str, optional
            The mode of operation for the dataset, either "train" or "test". Defaults to `"train"`.
        inds_to_filter : list[str], optional
            A list of substrings to filter the columns for landmark data. Defaults to `None`.
        """
        self.df = df.copy()
        self.data_folder = data_folder

        self.mode = mode
        self.aug = aug
        self.dec = tokenizer

        # input stuff
        with open(inference_args) as f:
            columns = json.load(f)["selected_columns"]

        self.filtered_columns = None
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

        self.processor = Preprocessing()

        self.flip_aug = flip_aug

    def __getitem__(self, idx):
        """
        Retrieves a single data sample and its associated target.

        Parameters
        ----------
        idx : int
            Index of the data sample to retrieve.

        Returns
        -------
        tuple
            A tuple containing:
            - data : torch.Tensor
                The processed data sample (float tensor).
            - token_ids : list[int]
                Encoded target phrase as a list of integers.
            - input_length : torch.Tensor
                Length of the input sequence.
            - target_len : torch.Tensor
                Length of the target sequence.
        """
        row = self.df.iloc[idx]
        file_id, sequence_id, phrase = row[["file_id", "sequence_id", "phrase"]]

        data = self.load_one(file_id, sequence_id)
        data = torch.from_numpy(data)
        data = self.processor(data, self.filtered_columns)

        if self.mode == "train":
            if np.random.rand() < self.flip_aug:
                data = flip(data, self.flip_array)

            if self.aug:
                data = self.augment(data)

        input_length = data.size(0)

        target_len = len(phrase)

        token_ids = self.dec.text_to_indices(phrase)
        return (
            data.float(),
            token_ids,
            torch.tensor(input_length),
            torch.tensor(target_len),
        )

    def collate_fn(self, batch):
        """
        Collates a batch of samples into padded tensors for model input.

        Parameters
        ----------
        batch : list[tuple]
            List of tuples where each tuple contains:
            - data : torch.Tensor
                The processed data sample.
            - label : list[int]
                Encoded target phrase.
            - input_length : torch.Tensor
                Length of the input sequence.
            - label_length : torch.Tensor
                Length of the target sequence.

        Returns
        -------
        tuple
            A tuple containing:
            - data : torch.Tensor
                Padded tensor of data samples.
            - labels : torch.Tensor
                Padded tensor of encoded target phrases.
            - input_lengths : torch.Tensor
                Tensor of input sequence lengths.
            - label_lengths : torch.Tensor
                Tensor of target sequence lengths.
        """
        data = []
        labels = []
        input_lengths = []
        label_lengths = []

        for d, label, in_len, t_len in batch:
            data.append(d)
            labels.append(torch.IntTensor(label))
            input_lengths.append(in_len)
            label_lengths.append(t_len)

        data = nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0.0)
        labels = nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=self.dec.PAD_TOK
        )

        return (
            data,
            torch.IntTensor(labels),
            torch.IntTensor(input_lengths),
            torch.IntTensor(label_lengths),
        )

    def augment(self, x):
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
        x_aug = self.aug(image=x)["image"]
        return x_aug

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return len(self.df)

    def load_one(self, file_id, sequence_id):
        """
        Loads a single data sample from the dataset.

        Parameters
        ----------
        file_id : str
            The file id of the data sample.
        sequence_id : int
            The sequence id of the data sample.

        Returns
        -------
        data : np.ndarray
            The loaded data sample (seq_len, 3* nlandmarks).
        """
        path = self.data_folder + f"/{file_id}/{sequence_id}.npy"
        data = np.load(path)  # seq_len, 3* nlandmarks
        return data


class KP_Datamodule:
    """
    A PyTorch DataModule for the ChicagoKPKaggle dataset.

    This class is responsible for managing the dataset, applying transformations,
    and setting up the data loaders for training, validation, and testing.

    Attributes
    ----------
    data_folder: str
        The directory where the raw data files (e.g., .npy files) are stored.
    symmetry_fp: str
        Path to a CSV file containing symmetry mapping information for landmark flipping.
    flip_aug: float
        The probability of applying a flip augmentation.
    inference_args: str
        Path to a JSON file containing information about the selected columns for processing.
    batch_size: int
        The batch size used for loading data during training and testing.
    num_workers: int
        The number of workers for data loading.
    test_df: str
        Path to the CSV file containing the test set metadata.
    train_df: str
        Path to the CSV file containing the training set metadata.
    tokenizer: Tokenizer
        A tokenizer object used to map labels (phrases) to integers and vice versa.
    inds_to_filter: list[str]
        A list of substrings to filter the columns for landmark data.
    data_train: Dataset
        The training dataset. Defaults to None.
    data_val: Dataset
        The validation dataset. Defaults to None.
    data_test: Dataset
        The test dataset. Defaults to None.
    transfomrs_train: Compose
        A collection of augmentations to apply during training.

    Methods
    -------
    setup_data()
        Initializes the datasets for training, validation, and test sets.
    get_dataset(stage: str)
        Returns the appropriate dataset (train, val, test) for the given stage.
    get_dataloader(stage: str)
        Returns the DataLoader for the given stage (train, val, test).
    """

    def __init__(
        self,
        data_folder: str,
        symmetry_fp: str,
        flip_aug: float,
        inference_args: str,
        batch_size: int,
        num_workers: int,
        test_df: str,
        train_df: str,
        tokenizer: Tokenizer,
        inds_to_filter: list[str],
    ):
        """
        Initializes the KP_Datamodule.

        Parameters
        ----------
        data_folder : str
            The directory where the raw data files (e.g., .npy files) are stored.
        symmetry_fp : str
            Path to a CSV file containing symmetry mapping information for landmark flipping.
        flip_aug : float
            The probability of applying a flip augmentation.
        inference_args : str
            Path to a JSON file containing information about the selected columns for processing.
        batch_size : int
            The batch size used for loading data during training and testing.
        num_workers : int
            The number of workers for data loading.
        test_df : str
            Path to the CSV file containing the test set metadata.
        train_df : str
            Path to the CSV file containing the training set metadata.
        tokenizer : Tokenizer
            A tokenizer object used to map labels (phrases) to integers and vice versa.
        inds_to_filter : list[str]
            A list of substrings to filter the columns for landmark data.
        """
        super().__init__()

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.data_folder = data_folder
        self.tokenizer = tokenizer
        self.symmetry_fp = symmetry_fp

        self.inference_args = inference_args
        self.batch_size = batch_size
        self.flip_aug = flip_aug
        self.inds_to_filter = inds_to_filter

        self.test_df = test_df
        self.train_df = train_df

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.transfomrs_train = Compose(
            [
                A.Resample(sample_rate=(0.5, 1.5), p=0.8),
                A.SpatialAffine(
                    scale=(0.8, 1.2),
                    shear=(-0.15, 0.15),
                    shift=(-0.1, 0.1),
                    degree=(-30, 30),
                    p=0.75,
                ),
                A.TemporalMask(
                    size=(0.2, 0.4), mask_value=0.0, p=0.5
                ),  # mask with 0 as it is post-normalization
                A.SpatialMask(
                    size=(0.05, 0.1), mask_value=0.0, mode="relative", p=0.5
                ),  # mask with 0 as it is post-normalization
            ]
        )
        self.transfomrs_train.disable_check_args_private()

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
        df = pd.read_csv(self.train_df)
        train_df = df[df["fold"] != 1].copy()
        if stage == "train":
            self.data_train = CustomDataset(
                train_df,
                data_folder=self.data_folder,
                symmetry_fp=self.symmetry_fp,
                flip_aug=self.flip_aug,
                inference_args=self.inference_args,
                aug=self.transfomrs_train,
                tokenizer=self.tokenizer,
                mode="train",
                inds_to_filter=self.inds_to_filter,
            )
            LOGGER.info(f"len is {len(self.data_train)}")
            return self.data_train

        elif stage == "val":
            df = pd.read_csv(self.train_df)
            val_df = df[df["fold"] == 1].copy()
            self.data_val = CustomDataset(
                val_df,
                data_folder=self.data_folder,
                symmetry_fp=self.symmetry_fp,
                flip_aug=self.flip_aug,
                inference_args=self.inference_args,
                aug=None,
                tokenizer=self.tokenizer,
                mode="val",
                inds_to_filter=self.inds_to_filter,
            )
            LOGGER.info(f"len is {len(self.data_val)}")
            return self.data_val

        elif stage == "test":
            df = pd.read_csv(self.test_df)
            test_df = df[df["fold"] == 0].copy()
            self.data_test = CustomDataset(
                test_df,
                data_folder=self.data_folder,
                symmetry_fp=self.symmetry_fp,
                flip_aug=self.flip_aug,
                inference_args=self.inference_args,
                aug=None,
                tokenizer=self.tokenizer,
                mode="test",
                inds_to_filter=self.inds_to_filter,
            )
            LOGGER.info(f"len is {len(test_df)}")
            return self.data_test

    def get_dataloader(self, stage: str):
        LOGGER.info(f"Ininting {stage} dataloader")
        if stage == "train":
            self.data_train = self.get_dataset("train")
            self.train_loader = DataLoader(
                self.data_train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self.data_train.collate_fn,
                shuffle=True,
            )
            return self.train_loader

        elif stage == "val":
            self.data_val = self.get_dataset("val")

            self.val_loader = DataLoader(
                self.data_val,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self.data_val.collate_fn,
                shuffle=False,
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
