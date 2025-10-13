"""
Utils for module initialization.
"""

import math

from torch import nn


def init_linear(
    layer, init_gain_act="linear", kaiming_args=None, init_bias=0.0
):
    """
    Init parameters of linear layers.

    The initialization of 'weight' depends on the type of activation function
    following this layer.
    - relu, leaky_relu, gelu, mish, pmish:
        Kaiming method with normal distributions
    - others:
        Xavier method with normal distributions
    - float:
        Truncated normal distributions

    The initialization of 'bias' is constant, unless 'init_bias' is None, then
    it follows uniform distributions like PyTorch default.
    """
    # Weight
    if init_gain_act in ("relu", "leaky_relu", "gelu", "mish", "pmish"):
        if kaiming_args is None:
            kaiming_args = (0.0, "fan_in")
        nn.init.kaiming_normal_(
            layer.weight,
            a=kaiming_args[0],
            mode=kaiming_args[1],
            nonlinearity="leaky_relu",
        )
    elif isinstance(init_gain_act, float):
        nn.init.trunc_normal_(layer.weight, std=init_gain_act)
    else:
        nn.init.xavier_normal_(
            layer.weight, gain=nn.init.calculate_gain(init_gain_act)
        )
    # Bias
    if init_bias == "uniform" and layer.bias is not None:
        # Default PyTorch init for bias
        # pylint: disable=W0212
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(layer.bias, -bound, bound)
    elif layer.bias is not None:
        nn.init.constant_(layer.bias, init_bias)


def init_conv(layer, init_gain_act="linear", kaiming_args=None, init_bias=0.0):
    """
    Init parameters of convolution layers.

    The initialization of 'weight' depends on the type of activation function
    following this layer.
    - relu, leaky_relu, gelu, mish, pmish:
        Kaiming method with normal distributions
    - others:
        Xavier method with normal distributions
    - float:
        Truncated normal distributions

    The initialization of 'bias' is constant, unless 'init_bias' is None, then
    it follows uniform distributions like PyTorch default.
    """
    init_linear(layer, init_gain_act, kaiming_args, init_bias)


def init_lstm(lstm, forget_bias=1.0):
    """
    Init parameters of LSTM (Long Short-Term Memory) layers.

    This function comes from sketch_rnn of the magenta project
    [arxiv:1704.03477] (translation from TensorFlow).
    """
    forget_bias /= 2.0
    size_h = lstm.weight_hh_l0.size(1)
    for i in range(lstm.num_layers):
        nn.init.xavier_uniform_(getattr(lstm, f"weight_ih_l{i}"), gain=1.0)
        weight_hh = getattr(lstm, f"weight_hh_l{i}")
        nn.init.orthogonal_(weight_hh[:size_h], gain=1.0)
        nn.init.orthogonal_(weight_hh[size_h : size_h * 2], gain=1.0)
        nn.init.orthogonal_(weight_hh[size_h * 2 : size_h * 3], gain=1.0)
        nn.init.orthogonal_(weight_hh[size_h * 3 :], gain=1.0)
        bias_ih = getattr(lstm, f"bias_ih_l{i}")
        nn.init.constant_(bias_ih, 0)
        nn.init.constant_(bias_ih[size_h : size_h * 2], forget_bias)
        bias_hh = getattr(lstm, f"bias_hh_l{i}")
        nn.init.constant_(bias_hh, 0)
        nn.init.constant_(bias_hh[size_h : size_h * 2], forget_bias)
