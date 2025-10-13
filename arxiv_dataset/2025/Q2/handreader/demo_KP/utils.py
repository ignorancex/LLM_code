import torch
import torch.nn as nn


class Preprocessing(nn.Module):
    """
    A preprocessing module that normalizes and fills NaN values in the input tensor.

    This class provides methods for preprocessing input data, including normalization
    and filling NaN values. It is commonly used for preprocessing sequences of landmark
    data in the context of time series or 2D/3D landmark-based models.

    Methods:
    -------
    normalize(x):
        Normalizes the input tensor by subtracting the mean and dividing by the standard deviation.
        NaN values are excluded in the calculation of mean and standard deviation.

    fill_nans(x):
        Fills NaN values in the input tensor with zero.

    forward(x, filtered_columns=None):
        The forward pass of the preprocessing module. It normalizes the input tensor and fills NaN values.
        It also reshapes the input if necessary and optionally filters the columns before processing.

    """

    def __init__(self):
        super().__init__()

    def normalize(self, x):
        nonan = x[~torch.isnan(x)].view(-1, x.shape[-1])
        x = x - nonan.mean(0)[None, None, :]
        x = x / nonan.std(0, unbiased=False)[None, None, :]
        return x

    def fill_nans(self, x):
        x[torch.isnan(x)] = 0
        return x

    def forward(self, x, filtered_columns: list[int] = None):
        # seq_len, 3* n_landmarks -> seq_len, n_landmarks, 3

        if filtered_columns is not None and len(filtered_columns) > 1:
            x = x[:, filtered_columns]
        x = x.reshape(x.shape[0], 3, -1).permute(0, 2, 1)

        # Normalize & fill nans
        x = self.normalize(x)
        x = self.fill_nans(x)

        return x


def getRuTokens() -> str:
    return " абвгдежзийклмнопрстуфхцчшщъыьэюяё"


def getChicagoTokens() -> str:
    return " &'.@abcdefghijklmnopqrstuvwxyz"


def get_vocab(char_list, joint_ctc_attention: bool = False):
    """
    Generate vocabulary mappings for character sequences with optional Joint CTC-Attention mechanism.

    This function creates a vocabulary for the given character list (`char_list`) that can be used for
    encoding and decoding sequences in a speech recognition or sequence-to-sequence model.
    Depending on the `joint_ctc_attention` flag, it generates two types of vocabularies: one for
    standard CTC and one for joint CTC-Attention.

    Parameters
    ----------
    char_list : list of str
        A list of characters (or tokens) that will form the vocabulary. These can be letters, symbols, or
        any other tokens relevant to the problem.

    joint_ctc_attention : bool, optional, default: False
        A flag to indicate whether to create the vocabulary for a joint CTC-Attention model.
        If set to `True`, special tokens (`<EOS>`, `<BOS>`, `<PAD>`) will be added, and the vocabulary
        will include both these special tokens and the provided `char_list`.
        If set to `False`, the vocabulary will be created for a standard CTC model, where a special
        `_` character is added to the front of the `char_list`.

    Returns
    -------
    vocab_map : dict
        A dictionary mapping characters or tokens to integer indices. The mapping depends on the value of
        `joint_ctc_attention`:
        - If `joint_ctc_attention=True`, the special tokens (`<EOS>`, `<BOS>`, `<PAD>`) will be included
          at the end of the vocabulary.
        - If `joint_ctc_attention=False`, a `_` character will be added as the first token for CTC.

    inv_vocab_map : dict
        A dictionary mapping integer indices to characters or tokens. This is the inverse of `vocab_map`.

    char_list : list of str
        The original list of characters, potentially modified depending on `joint_ctc_attention`. If `joint_ctc_attention`
        is `False`, it will prepend a `_` character to the original `char_list` for CTC models.
    """
    if joint_ctc_attention:
        vocab_map_ce, inv_vocab_map_ce = {}, {}

        for i, char in enumerate(char_list):
            vocab_map_ce[char] = i
            inv_vocab_map_ce[i] = char

        vocab_map_ce["<EOS>"] = len(char_list)
        inv_vocab_map_ce[len(char_list)] = "<EOS>"
        vocab_map_ce["<BOS>"] = len(char_list) + 1
        inv_vocab_map_ce[len(char_list) + 1] = "<BOS>"

        vocab_map_ce["<PAD>"] = len(char_list) + 2
        inv_vocab_map_ce[len(char_list) + 2] = "<PAD>"

        return vocab_map_ce, inv_vocab_map_ce, char_list

    else:
        vocab_map_ctc, inv_vocab_map_ctc = {}, {}
        char_list_ctc = "_" + char_list
        for i, char in enumerate(char_list_ctc):
            vocab_map_ctc[char] = i
            inv_vocab_map_ctc[i] = char

        return vocab_map_ctc, inv_vocab_map_ctc, char_list_ctc
