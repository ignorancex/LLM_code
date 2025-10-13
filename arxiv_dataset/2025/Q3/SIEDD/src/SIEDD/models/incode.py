from torchvision.models import resnet34
from torch import nn
import torch
import einops
from ..configs import EncoderConfig, MLPConfig, NoPosEncode

device = "cuda" if torch.cuda.is_available() else "cpu"


class INCODEHarmonizer(nn.Module):
    def __init__(self, data_shape):
        super().__init__()
        from .generic_mlp import get_mlp

        resnet = resnet34()
        self.feature_extractor = nn.Sequential(
            *list(resnet.children())[:5], nn.AdaptiveAvgPool2d(1)
        ).to(device)
        mlp_cfg = EncoderConfig(
            net=MLPConfig(
                num_layers=3,
                pos_encode_cfg=NoPosEncode(dim_in=64, dim_out=64),
                activation="selu",
            )
        )
        self.mlp = get_mlp(4, data_shape, mlp_cfg).to(device)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        # shared training: whole model incode, save abcd each iteration
        # shared eval: same, don't pass in abcd
        # normal training: decoder only incode, save abcds only in decoders
        # normal eval: don't pass in abcd
        features = self.feature_extractor(imgs)  # n 64 1 1
        features = einops.rearrange(features, "n c () () -> n c")
        incode_params = self.mlp(features, preprocess_output=False)
        return incode_params
