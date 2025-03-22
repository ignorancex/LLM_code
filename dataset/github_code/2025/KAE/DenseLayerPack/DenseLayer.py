import torch
import torch.nn as nn
from .FourierKAN import NaiveFourierKANLayer as FourierKANLayer
from .KAN import KANLinear as KANLayer
from .MultiKAN import MultiKAN as MultiKANLayer
from .KAE import KAELayer as KAELayer
from .WavKAN import KANLinear as WavKANLayer
from .const import DENSE_LAYER_CONST


def set_params(layer_type, in_features, out_features, **kwargs) -> nn.Module:
    """
    Initialize a specific type of layer based on the layer_type argument.

    Args:
        layer_type (str): Type of the layer to initialize.
        **kwargs: Additional arguments specific to the layer type.

    Returns:
        torch.nn.Module: Initialized layer of the specified type.
    """
    conv_args = {
        "grid_size": kwargs.get("grid_size", 5),
        "spline_order": kwargs.get("spline_order", 3),
        "scale_noise": kwargs.get("scale_noise", 0.1),
        "scale_base": kwargs.get("scale_base", 1.0),
        "scale_spline": kwargs.get("scale_spline", 1.0),
        "enable_standalone_scale_spline": kwargs.get(
            "enable_standalone_scale_spline", True
        ),
        "base_activation": kwargs.get("base_activation", nn.SiLU),
        "grid_eps": kwargs.get("grid_eps", 0.02),
        "grid_range": kwargs.get("grid_range", [-1, 1]),
    }
    multi_args = {}
    taylor_args = {
        "order": kwargs.get("order", 2),
        "addbias": kwargs.get("addbias", True),
    }
    wavelet_args = {"wavelet_type": kwargs.get("wavelet_type", "mexican_hat")}
    fourier_args = {
        "gridsize": kwargs.get("gridsize", 5),
        "addbias": kwargs.get("addbias", True),
        "smooth_initialization": kwargs.get("smooth_initialization", False),
    }
    linear_args = {}

    if layer_type == DENSE_LAYER_CONST.KAN_LAYER:
        layer = KANLayer(in_features, out_features, **conv_args)
    elif layer_type == DENSE_LAYER_CONST.MULTI_KAN_LAYER:
        layer = MultiKANLayer(in_features, out_features, **multi_args)
    elif layer_type == DENSE_LAYER_CONST.KAE_LAYER:
        layer = KAELayer(in_features, out_features, **taylor_args)
    elif layer_type == DENSE_LAYER_CONST.FOURIER_KAN_LAYER:
        layer = FourierKANLayer(in_features, out_features, **fourier_args)
    elif layer_type == DENSE_LAYER_CONST.WAVELET_KAN_LAYER:
        layer = WavKANLayer(in_features, out_features, **wavelet_args)
    elif layer_type == DENSE_LAYER_CONST.LINEAR_LAYER:
        layer = nn.Linear(in_features, out_features, **linear_args)
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")

    return layer


class DenseLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        layer_type=DENSE_LAYER_CONST.LINEAR_LAYER,
        **kwargs,
    ):
        """
        Initialize a DenseLayer instance.

        Args:
            in_features (int): Input dimensionality.
            out_features (int): Output dimensionality.
            layer_type (str, optional): Type of the layer to initialize.
                Defaults to DENSE_LAYER_CONST.LINEAR_LAYER.
            **kwargs: Additional arguments specific to the layer type.
        """
        super(DenseLayer, self).__init__()
        self.layer = set_params(layer_type, in_features, out_features, **kwargs)

    def forward(self, x):
        return self.layer.forward(x)
