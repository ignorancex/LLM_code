import torch.nn as nn


class RNNHead(nn.Module):
    """
    A recurrent neural network (RNN) head for sequence classification.

    Parameters:
    -----------
    input_dim : int
        The dimension of the input feature vectors.
    hidden_dim : int
        The number of units in the RNN hidden layer.
    num_layers : int
        The number of recurrent layers.
    num_classes : int
        The number of output classes.
    bidirectional : bool
        Whether the RNN should be bidirectional. If True, the RNN will double the number of output dimensions.
    return_outs : bool, optional (default: False)
        Whether to return the outputs of the RNN or just the final logits. If True, returns a tuple (logits, outs).

    Attributes:
    -----------
    rnn : nn.GRU
        The recurrent neural network (GRU) layer.
    proj : nn.Linear
        A fully connected layer to map RNN outputs to class logits.

    Methods:
    --------
    forward(x):
        Performs a forward pass through the network. Given an input tensor `x`, it returns the predicted class logits.
        If `return_outs` is True, it also returns the RNN outputs.

    Returns:
    --------
    logits : torch.Tensor
        Predicted class logits.
    outs : torch.Tensor, optional
        Outputs from the RNN if `return_outs` is True.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        bidirectional: bool,
        return_outs: bool = False,
    ):
        """
        Initializes the RNNHead instance.

        Parameters:
        -----------
        input_dim : int
            The dimension of the input feature vectors.
        hidden_dim : int
            The number of units in the RNN hidden layer.
        num_layers : int
            The number of recurrent layers.
        num_classes : int
            The number of output classes.
        bidirectional : bool
            Whether the RNN should be bidirectional. If True, the RNN will double the number of output dimensions.
        return_outs : bool, optional (default: False)
            Whether to return the outputs of the RNN or just the final logits. If True, returns a tuple (logits, outs).
        """
        super().__init__()
        self.return_outs = return_outs
        self.rnn = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.proj = nn.Linear(2 * hidden_dim if bidirectional else hidden_dim, num_classes)

    def forward(self, x):
        """
        Performs a forward pass through the network.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
        --------
        logits : torch.Tensor
            Predicted class logits of shape (batch_size, seq_len, num_classes).
        outs : torch.Tensor, optional
            Outputs from the RNN if `return_outs` is True.
        """
        outs, _ = self.rnn(x)
        logits = self.proj(outs)
        if self.return_outs:
            return logits, outs
        return logits
