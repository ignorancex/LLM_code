import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def pad_collate_fn_classification(batch):
    """
    The collate_fn that can add padding to the sequences so all can have
    the same length as the longest one.

    Args:
        batch (List[List, List]): The batch data, where the first element
        of the tuple are the word idx and the second element are the target
        label.

    Returns:
        A tuple (x, y). The element x is a tuple containing (1) a tensor of padded
        word vectors and (2) their respective lengths of the sequences. The element
        y is a tensor of padded tag indices. The word vectors are padded with vectors
        of 0s and the tag indices are padded with -100s. Padding with -100 is done
        because the cross-entropy loss, the accuracy metric and the F1 metric ignores
        the targets with values -100.
    """
    # This gets us two lists of tensors and a list of integer.
    # Each tensor in the first list is a sequence of word vectors.
    # Each tensor in the second list is a sequence of tag indices.
    # The list of integer consist of the lengths of the sequences in order.
    sequences_vectors, sequences_labels, lengths = [], [], []
    for (seq_vector, label) in sorted(batch, key=lambda x: len(x[0]), reverse=True):
        # print(seq_vector)
        # print(label)
        sequences_vectors.append(seq_vector)
        sequences_labels.append(label)
        lengths.append(len(seq_vector))
    # print("Lengths: ", lengths)

    padded_sequences_vectors = pad_sequence(sequences_vectors, batch_first=True, padding_value=-100)
    # 4print(padded_sequences_vectors)

    return padded_sequences_vectors, torch.from_numpy(np.array(sequences_labels, dtype=np.longlong)), lengths


class VaryingLengthCustomDataset(Dataset):
    def __init__(self, examples, labels):
        self._examples = []
        self._labels = []
        for example, label in zip(examples, labels):
            self._examples.append(torch.from_numpy(example).type(torch.FloatTensor))
            self._labels.append(label)

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, idx):
        return [self._examples[idx], self._labels[idx]]
