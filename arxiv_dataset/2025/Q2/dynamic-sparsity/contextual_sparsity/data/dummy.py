# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import torch
from torch.utils.data import DataLoader, Dataset

N_FEATURES = 101
SEQUENCE_LENGTH = 50
N_SEQUENCES = 1000
BATCH_SIZE = 64


def compute_labels(x: torch.Tensor) -> torch.Tensor:
    """
    Make the label for each dummy input tensor
    """
    return (torch.floor(x).long() % 2).int()


class DummyDataset(Dataset):
    """
    Mock dataset for testing.
    The dataset consists of N_SEQUENCE sequences that differ by a small additive constant.
    Each sequence consists of SEQUENCE_LENGTH vectors of numbers 0 to DATA_SIZE shifted by i
    with i being the index withing the sequence.
    The labels corresponds to the parity of the closest integer for each sequence.
    """

    def __init__(self, n_features: int, sequence_length: int, n_sequences: int):
        dummy_sequence = torch.cat(
            [torch.roll(torch.arange(n_features), i).unsqueeze(0) for i in range(sequence_length)],
            0,
        )
        self.data = torch.cat(
            [
                dummy_sequence.unsqueeze(0) + float(i) / (n_sequences + 1)
                for i in range(1, n_sequences + 1)
            ],
            0,
        )

    def __getitem__(self, index):
        return {"x": self.data[index], "labels": compute_labels(self.data[index])}

    def __len__(self):
        return len(self.data)


def get_dummy_dataloader(
    n_features: int = N_FEATURES,
    sequence_length: int = SEQUENCE_LENGTH,
    n_sequences: int = N_SEQUENCES,
    batch_size: int = BATCH_SIZE,
) -> DataLoader:
    dataset = DummyDataset(
        n_features=n_features, sequence_length=sequence_length, n_sequences=n_sequences
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader
