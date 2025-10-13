import torch.nn as nn
from ..utils import tcnn
from ..layers.positional_encoders import (
    PosEncodingCoordX,
    PosEncodingFourier,
    PosEncodingGaussian,
    PosEncodingNeRF,
    PosEncodingHashgrid,
)
from ..configs import (
    EncoderConfig,
    CoordXConfig,
    NeRFConfig,
    FourierConfig,
    GaussianConfig,
    CudaHashgridConfig,
    NoPosEncode,
)


class BaseModel(nn.Module):
    def __init__(self, cfg: EncoderConfig, data_shape: list[int]):
        super().__init__()
        self.data_shape = data_shape
        self.cfg = cfg
        self.positional_encoding = cfg.net.pos_encode_cfg
        self.dim_in = cfg.net.pos_encode_cfg.dim_in
        self.load_positional_encoder()

    def load_positional_encoder(self) -> None:
        self.positional_encoder: (
            nn.Identity
            | PosEncodingCoordX
            | PosEncodingFourier
            | PosEncodingGaussian
            | PosEncodingNeRF
            | tcnn.Encoding
        )
        if isinstance(self.positional_encoding, NoPosEncode):
            self.positional_encoder = nn.Identity()
        elif isinstance(self.positional_encoding, CoordXConfig):
            dims = self.data_shape
            self.positional_encoder = PosEncodingCoordX(
                dims=dims, cfg=self.positional_encoding
            )
        elif isinstance(self.positional_encoding, NeRFConfig):
            self.positional_encoder = PosEncodingNeRF(cfg=self.positional_encoding)

        elif isinstance(self.positional_encoding, FourierConfig):
            self.positional_encoder = PosEncodingFourier(cfg=self.positional_encoding)

        elif isinstance(self.positional_encoding, GaussianConfig):
            self.positional_encoder = PosEncodingGaussian(cfg=self.positional_encoding)

        elif isinstance(self.positional_encoding, CudaHashgridConfig):
            self.positional_encoder = PosEncodingHashgrid(
                dim_in=self.dim_in, cfg=self.positional_encoding
            )

        else:
            raise ValueError(
                f"Unsupported positional encoding type: {self.positional_encoding}"
            )
