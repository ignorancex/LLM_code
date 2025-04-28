from torch.utils import data
import pandas as pd
import numpy as np
from Bio.Seq import translate
from Bio import SeqIO

import random

CHARSET = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, \
           'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, \
           'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20, \
           'O': 20, 'U': 20,
           'B': (2, 11),
           'Z': (3, 13),
           'J': (7, 9)}
CHARSET_SUB = {'B': 21, 'Z': 22, 'J': 23}
CHARLEN = 21

DNASET = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'K': (2, 3), 'M': (0, 1), 'R': (0, 2), 'Y': (1, 3), 'S': (1, 2), 'W': (0, 3),
          'B': (1, 2, 3), 'V': (0, 1, 2), 'H': (0, 1, 3), 'D': (0, 2, 3), 'X': (0, 1, 2, 3), 'N': (0, 1, 2, 3)
          }


def encoding_dna_seq_np(seq, arr, seq_len=1000):
    for frame_start in range(3):
        pep = translate(seq[frame_start:])
        for i, c in enumerate(pep):
            if i < seq_len:
                if c == "_" or c == "*":
                    # let them zero
                    continue
                elif isinstance(CHARSET[c], int):
                    idx = CHARSET[c]
                    arr[frame_start][i][idx] = 1
                else:
                    idx1 = CHARSET[c][0]
                    idx2 = CHARSET[c][1]
                    arr[frame_start][i][idx1] = 0.5
                    arr[frame_start][i][idx2] = 0.5


def encoding_dna_seq_np_aug(seq, arr, seq_len=1000, perc=0.0, type='train'):
    """

    :param seq:
    :param arr:
    :param seq_len:
    :param perc: kind like drop out, randomly replaced peptide as 0
    :param type:
    :return:
    """
    peps = []
    for frame_start in range(3):
        length = len(seq)
        frame_end = 3 * (length // 3) - 3 + frame_start
        pep = translate(seq[frame_start:frame_end])
        for i, c in enumerate(pep):
            if i < seq_len:
                if random.random() < perc and type == 'train':
                    continue
                if c == "_" or c == "*":
                    # let them zero
                    continue
                elif isinstance(CHARSET[c], int):
                    idx = CHARSET[c]
                    arr[frame_start][i][idx] = 1
                else:
                    idx1 = CHARSET[c][0]
                    idx2 = CHARSET[c][1]
                    arr[frame_start][i][idx1] = 0.5
                    arr[frame_start][i][idx2] = 0.5

        peps.append(pep)
    return peps


def encoding_dna_seq_idx_aug(seq, arr, seq_len=1000, type='train'):
    """

    :param seq:
    :param arr:
    :param seq_len:
    :param perc: kind like drop out, randomly replaced peptide as 0
    :param type:
    :return:
    """
    peps = []
    for frame_start in range(3):
        length = len(seq)
        frame_end = 3 * (length // 3) - 3 + frame_start
        pep = translate(seq[frame_start:frame_end])
        for i, c in enumerate(pep):
            if i < seq_len:
                if c == "_" or c == "*":
                    # let them zero
                    continue
                elif isinstance(CHARSET[c], int):
                    idx = CHARSET[c]
                    # we need add extra 1 to left 0 for *
                    arr[frame_start][i] = idx + 1
                else:
                    idx = CHARSET_SUB[c]
                    arr[frame_start][i] = idx + 1

        peps.append(pep)
    return peps


def encoding_dna_seq_frame1_np_aug(seq, arr, seq_len=1000, perc=0.7, type='train'):
    peps = []
    length = len(seq)
    frame_end = 3 * (length // 3)
    pep = translate(seq[:frame_end])
    for i, c in enumerate(pep):
        if i < seq_len:
            if random.random() < perc and type == 'train':
                continue
            if c == "_" or c == "*":
                # let them zero
                continue
            elif isinstance(CHARSET[c], int):
                idx = CHARSET[c]
                arr[0][i][idx] = 1
            else:
                idx1 = CHARSET[c][0]
                idx2 = CHARSET[c][1]
                arr[0][i][idx1] = 0.5
                arr[0][i][idx2] = 0.5

    peps.append(pep)
    return peps


def encoding_dna(seq, arr, seq_len=1000):
    for i, c in enumerate(seq):
        if i < seq_len:
            if c == "_" or c == "*":
                # let them zero
                continue
            elif isinstance(DNASET[c], int):
                idx = DNASET[c]
                arr[0][i][idx] = 1
            else:
                nums = len(DNASET[c])
                for idx in DNASET[c]:
                    arr[0][i][idx] = 1 / nums


class DnapepDataset(data.Dataset):
    # Input is DNA, then translate to 3 frame protein sequences
    def __init__(self, file_path, type='train', seq_len=1000, multilabel=False, num_classes=86, swap_dim=False,
                 use_embed=False, shuffle=True):
        self.type = type
        self.file = file_path
        self.seq_len = 1000
        # column 0 is label, column 1 is seq
        df = pd.read_csv(self.file, sep='\t', header=None)
        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
        self.labels = df[0]
        self.seqs = df[1]
        self.multilabel = multilabel
        self.num_classes = num_classes
        self.swap_dim = swap_dim
        self.use_embed = use_embed
        self.position = False

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        if self.use_embed:
            self.seq_dim = 1
            seq_np = np.zeros((3, self.seq_len, self.seq_dim), dtype=np.int)
            peps = encoding_dna_seq_idx_aug(self.seqs[index], seq_np, type=self.type)
        else:
            self.seq_dim = CHARLEN
            seq_np = np.zeros((3, self.seq_len, self.seq_dim), dtype=np.float32)
            # encoding_dna_seq_np(self.seqs[index], seq_np)
            peps = encoding_dna_seq_np_aug(self.seqs[index], seq_np, type=self.type)
        target = self.labels[index]
        if self.swap_dim:
            r = random.random()
            seq_np_aug = np.zeros((3, self.seq_len, self.seq_dim), dtype=np.float32)
            if r < (1 / 3):
                seq_np_aug[0, :, :] = seq_np[1, :, :]
                seq_np_aug[1, :, :] = seq_np[2, :, :]
                seq_np_aug[2, :, :] = seq_np[0, :, :]
                # peps[0, 1, 2] = peps[1, 2, 0]
            elif r < (2 / 3):
                seq_np_aug[0, :, :] = seq_np[2, :, :]
                seq_np_aug[1, :, :] = seq_np[0, :, :]
                seq_np_aug[2, :, :] = seq_np[1, :, :]
                # peps[0, 1, 2] = peps[2, 0, 1]
            else:
                seq_np_aug = seq_np

            seq_np = seq_np_aug

        if self.multilabel:
            target = np.zeros(self.num_classes, dtype=np.float32)
            target[self.labels[index]] = 1
        if self.position:
            pos_array = self.get_position(peps[0])
            seq_np = [seq_np, pos_array]
        return seq_np, target, peps

    def get_position(self, pep_seq):
        """
        generate position array
        :param pep_seq:
        :return:
        """
        inst_position = np.zeros(self.seq_len, dtype=np.int)
        for pos_i, w_i in enumerate(pep_seq):
            if pos_i < 1000:
                inst_position[pos_i] = pos_i + 1
        return inst_position


class DnaDataset(data.Dataset):
    # Input is DNA, then translate to 3 frame protein sequences
    def __init__(self, file_path, type='train', seq_len=1000):
        self.type = type
        self.file = file_path
        self.seq_len = 3000
        # column 0 is label, column 1 is seq
        df = pd.read_csv(self.file, sep='\t', header=None)
        self.labels = df[0]
        self.seqs = df[1]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        seq_np = np.zeros((1, self.seq_len, 4), dtype=np.float32)
        # encoding_dna_seq_np(self.seqs[index], seq_np)
        encoding_dna(self.seqs[index], seq_np)
        return seq_np, self.labels[index]


class SingleFrameDataset(data.Dataset):
    # Input is DNA, then translate to 3 frame protein sequences
    def __init__(self, file_path, type='train', seq_len=1000, multilabel=False, num_classes=86):
        self.type = type
        self.file = file_path
        self.seq_len = 1000
        # column 0 is label, column 1 is seq
        df = pd.read_csv(self.file, sep='\t', header=None)
        df = df.sample(frac=1).reset_index(drop=True)
        self.labels = df[0]
        self.seqs = df[1]
        self.multilabel = multilabel
        self.num_classes = num_classes

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        seq_np = np.zeros((1, self.seq_len, CHARLEN), dtype=np.float32)
        # encoding_dna_seq_np(self.seqs[index], seq_np)
        peps = encoding_dna_seq_frame1_np_aug(self.seqs[index], seq_np, type=self.type)
        target = self.labels[index]
        if self.multilabel:
            target = np.zeros(self.num_classes, dtype=np.float32)
            target[self.labels[index]] = 1
        return seq_np, target, peps


class PepFrameDataset(data.Dataset):
    # Input is DNA, then translate to 3 frame protein sequences
    def __init__(self, file_path, type='train', seq_len=1000, num_classes=86):
        self.type = type
        self.file = file_path
        self.seq_len = 1000
        # column 0 is label, column 1 is seq
        df = pd.read_csv(self.file, sep='\t', header=None)
        df = df.sample(frac=1).reset_index(drop=True)
        self.labels = df[0]
        self.seqs = df[1]
        self.num_classes = num_classes

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        peps = self.encoding(self.seqs[index])
        target = self.labels[index]
        return peps, target

    def encoding(self, seq):
        arr = [np.zeros((1, self.seq_len, CHARLEN), dtype=np.float32) for i in range(3)]
        for frame_start in range(3):
            length = len(seq)
            frame_end = 3 * (length // 3) - 3 + frame_start
            pep = translate(seq[frame_start:frame_end])
            for i, c in enumerate(pep):
                if i < self.seq_len:
                    if c == "_" or c == "*":
                        # let them zero
                        continue
                    elif isinstance(CHARSET[c], int):
                        idx = CHARSET[c]
                        arr[frame_start][0][i][idx] = 1
                    else:
                        idx1 = CHARSET[c][0]
                        idx2 = CHARSET[c][1]
                        arr[frame_start][0][i][idx1] = 0.5
                        arr[frame_start][0][i][idx2] = 0.5

        return arr


class OpenPepDataset(data.Dataset):
    # Input is DNA, then translate to 3 frame protein sequences
    def __init__(self, file_path, type='train', seq_len=1000, num_classes=86, use_embed=False, shuffle=True):
        self.type = type
        self.seq_len = 1000
        self.file = file_path
        # column 0 is label, column 1 is seq
        df = pd.read_csv(self.file, sep='\t', header=None)
        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
        self.labels = df[0]
        self.in_set = (self.labels >=0).astype('int32')  # 1: inset, 0: outset
        self.seqs = df[1]
        self.num_classes = num_classes
        self.use_embed = use_embed

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        if self.use_embed:
            raise NotImplementedError
        else:
            self.seq_dim = CHARLEN
            seq_np, peps = self.encoding(self.seqs[index])
        target = self.labels[index]
        inset = self.in_set[index]
        return seq_np, target, inset

    def encoding(self, seq):
        arr = np.zeros((3, self.seq_len, CHARLEN), dtype=np.float32)
        peps = []
        for frame_start in range(3):
            length = len(seq)
            frame_end = 3 * (length // 3) - 3 + frame_start
            pep = translate(seq[frame_start:frame_end])
            peps.append(pep)
            for i, c in enumerate(pep):
                if i < self.seq_len:
                    if c == "_" or c == "*":
                        # let them zero
                        continue
                    elif isinstance(CHARSET[c], int):
                        idx = CHARSET[c]
                        arr[frame_start][i][idx] = 1
                    else:
                        idx1 = CHARSET[c][0]
                        idx2 = CHARSET[c][1]
                        arr[frame_start][i][idx1] = 0.5
                        arr[frame_start][i][idx2] = 0.5

        return arr, peps


class DnaFastaDataset(data.Dataset):
    # Input is DNA, then translate to 3 frame protein sequences
    def __init__(self, file_path, labels=None, type='train', seq_len=1000, num_classes=86, use_embed=False,
                 shuffle=True):
        self.type = type
        self.seq_len = 1000
        # column 0 is label, column 1 is seq
        self.file = file_path

        records = list(SeqIO.parse(file_path, format='fasta'))
        self.ids = []
        for record in records:
            self.ids.append(record.id)
        if labels is None:
            labels = [-1] * len(records)
        self.labels = labels
        self.seqs = records
        self.num_classes = num_classes
        self.use_embed = use_embed

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        if self.use_embed:
            raise NotImplementedError
        else:
            self.seq_dim = CHARLEN
            seq_np, peps = self.encoding(self.seqs[index])
        target = self.labels[index]
        return seq_np, target, peps

    def encoding(self, record):
        arr = np.zeros((3, self.seq_len, CHARLEN), dtype=np.float32)
        peps = []
        for frame_start in range(3):
            length = len(record.seq)
            frame_end = 3 * (length // 3) - 3 + frame_start
            pep = translate(record.seq[frame_start:frame_end])
            peps.append(str(pep))
            for i, c in enumerate(pep):
                if i < self.seq_len:
                    if c == "_" or c == "*":
                        # let them zero
                        continue
                    elif isinstance(CHARSET[c], int):
                        idx = CHARSET[c]
                        arr[frame_start][i][idx] = 1
                    else:
                        idx1 = CHARSET[c][0]
                        idx2 = CHARSET[c][1]
                        arr[frame_start][i][idx1] = 0.5
                        arr[frame_start][i][idx2] = 0.5

        return arr, peps
