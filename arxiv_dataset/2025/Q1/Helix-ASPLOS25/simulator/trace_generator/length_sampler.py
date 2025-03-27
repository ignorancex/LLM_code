# 2024.03.24 Yixuan Mei

import random
import pickle

from enum import Enum
from typing import List, Tuple
from pathlib import Path


class Dataset(Enum):
    # SharedGPT: https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
    # preprocessed based on vLLM paper
    # avg_input_length: 74.1, avg_output_length: 228.5
    # max_input_length: 1023, max_output_length: 1046
    SharedGPT = "Dataset.SharedGPT"
    # Alpaca: https://huggingface.co/datasets/tatsu-lab/alpaca/tree/main
    # preprocessed based on vLLM paper
    # avg_input_length: 13.9, avg_output_length: 48.3
    # max_input_length: 412, max_output_length: 717
    Alpaca = "Dataset.Alpaca"
    # Azure Code: https://github.com/Azure/AzurePublicDataset/blob/master/data/AzureLLMInferenceTrace_code.csv
    # no preprocessing, feature: very long input with short output
    # avg_input_length: 2047.8, avg_output_length: 27.8
    # max_input_length: 7437, max_output_length: 1899
    AzureCode = "Dataset.AzureCode"
    # Azure Conversation: https://github.com/Azure/AzurePublicDataset/blob/master/data/AzureLLMInferenceTrace_conv.csv
    # preprocessing: remove too long / short request (long: input > 2048, output > 1024; short: input < 4, output < 4)
    # feature: normal length input and output
    # avg_input_length: 763.1, avg_output_length: 232.4
    # max_input_length: 2047, max_output_length: 1000
    AzureConversation = "Dataset.AzureConversation"


class LengthSampler:
    def __init__(self, dataset: Dataset, seed: int) -> None:
        """
        Sample the length of input and output from the dataset.

        :param dataset: type of dataset
        :param seed: random seed
        :return: None
        """
        # save parameters
        self.dataset: Dataset = dataset
        self.seed: int = seed
        random.seed(seed)

        # load the dataset
        cur_abs_path = Path(__file__).parent.absolute()
        if self.dataset == Dataset.SharedGPT:
            with open(cur_abs_path / "length_data/shared_gpt_input.pkl", "rb") as file:
                input_length_list: List[int] = pickle.load(file)
            with open(cur_abs_path / "length_data/shared_gpt_output.pkl", "rb") as file:
                output_length_list: List[int] = pickle.load(file)
        elif self.dataset == Dataset.Alpaca:
            with open(cur_abs_path / "length_data/alpaca_input.pkl", "rb") as file:
                input_length_list: List[int] = pickle.load(file)
            with open(cur_abs_path / "length_data/alpaca_output.pkl", "rb") as file:
                output_length_list: List[int] = pickle.load(file)
        elif self.dataset == Dataset.AzureCode:
            with open(cur_abs_path / "length_data/azure_code_input.pkl", "rb") as file:
                input_length_list: List[int] = pickle.load(file)
            with open(cur_abs_path / "length_data/azure_code_output.pkl", "rb") as file:
                output_length_list: List[int] = pickle.load(file)
        elif self.dataset == Dataset.AzureConversation:
            with open(cur_abs_path / "length_data/azure_conv_input.pkl", "rb") as file:
                input_length_list: List[int] = pickle.load(file)
            with open(cur_abs_path / "length_data/azure_conv_output.pkl", "rb") as file:
                output_length_list: List[int] = pickle.load(file)
        else:
            assert False, "Found unknown dataset!"
        self.input_length_list: List[int] = input_length_list
        self.output_length_list: List[int] = output_length_list
        assert len(self.input_length_list) == len(self.output_length_list)

        # some statistics
        self.average_input_length: float = sum(self.input_length_list) / len(self.input_length_list)
        self.average_output_length: float = sum(self.output_length_list) / len(self.output_length_list)
        self.average_length: float = self.average_input_length + self.average_output_length

    def sample_length(self) -> Tuple[int, int]:
        """
        Sample a pair of input and output length.

        :return: input length, output length
        """
        index: int = random.randint(0, len(self.input_length_list) - 1)
        return self.input_length_list[index], self.output_length_list[index]

    def get_average_length(self) -> float:
        """
        Get the average length of input and output.

        :return: average length
        """
        return self.average_length
