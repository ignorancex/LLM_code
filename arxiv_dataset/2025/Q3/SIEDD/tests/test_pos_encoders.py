import pytest
from SIEDD.layers.positional_encoders import (
    PosEncodingNeRF,
    PosEncodingCoordX,
    PosEncodingFourier,
)
from SIEDD.configs import (
    NeRFConfig,
    CoordXConfig,
    FourierConfig,
    MLPConfig,
    NoPosEncode,
)
import torch


class TestNerf:
    def test_nerf_config(self):
        with pytest.raises(ValueError):
            NeRFConfig(dim_in=1, dim_out=16, include_coord=True)
        with pytest.raises(ValueError):
            NeRFConfig(dim_in=2, dim_out=17, include_coord=False)
        NeRFConfig(dim_in=1, dim_out=17, include_coord=True)
        NeRFConfig(dim_in=1, dim_out=16, include_coord=False)

    def test_nerf_1(self):
        cfg = NeRFConfig(dim_in=2, dim_out=16, include_coord=False)
        encoder = PosEncodingNeRF(cfg)
        inp = torch.randn(3, 2)
        out = encoder(inp)
        assert out.size(-1) == 16

    def test_nerf_2(self):
        cfg = NeRFConfig(dim_in=1, dim_out=17, include_coord=True)
        encoder = PosEncodingNeRF(cfg)
        inp = torch.randn(3, 1)
        out = encoder(inp)
        assert out.size(-1) == 17


class TestFourier:
    def test_fourier_config(self):
        FourierConfig(dim_out=256)
        with pytest.raises(ValueError):
            FourierConfig(dim_out=257)

    def test_fourier_1(self):
        cfg = FourierConfig(dim_in=2, dim_out=16)
        encoder = PosEncodingFourier(cfg)
        inp = torch.randn(3, 2)
        out = encoder(inp)
        assert out.size(-1) == 16


class TestCoordX:
    def test_coordx_1(self):
        cfg = CoordXConfig(
            dim_in=2,
            dim_out=16,
            net_cfg=MLPConfig(
                num_layers=2,
                dim_hidden=8,
                pos_encode_cfg=NoPosEncode(dim_in=1, dim_out=1),
            ),
        )
        encoder = PosEncodingCoordX(dims=[5, 6], cfg=cfg)
        inp = [torch.randn(5, 1), torch.randn(6, 1)]
        out = encoder(inp)
        assert out.size(-1) == 16

    def test_coordx_2(self):
        cfg = CoordXConfig(
            dim_in=2,
            dim_out=16,
            net_cfg=MLPConfig(
                num_layers=2,
                dim_hidden=8,
                pos_encode_cfg=FourierConfig(dim_in=1, dim_out=4),
            ),
        )
        encoder = PosEncodingCoordX(dims=[5, 6], cfg=cfg)
        inp = [torch.randn(5, 1), torch.randn(6, 1)]
        out = encoder(inp)
        assert out.size(-1) == 16

    def test_coordx_3(self):
        cfg = CoordXConfig(
            dim_in=2,
            dim_out=16,
            net_cfg=MLPConfig(
                num_layers=2,
                dim_hidden=8,
                pos_encode_cfg=NeRFConfig(dim_in=1, dim_out=4, include_coord=False),
            ),
        )
        encoder = PosEncodingCoordX(dims=[5, 6], cfg=cfg)
        inp = [torch.randn(5, 1), torch.randn(6, 1)]
        out = encoder(inp)
        assert out.size(-1) == 16

    def test_coordx_4(self):
        cfg = CoordXConfig(
            dim_in=2,
            dim_out=16,
            net_cfg=MLPConfig(
                num_layers=2,
                dim_hidden=8,
                pos_encode_cfg=NeRFConfig(dim_in=1, dim_out=5, include_coord=True),
            ),
        )
        encoder = PosEncodingCoordX(dims=[5, 6], cfg=cfg)
        inp = [torch.randn(5, 1), torch.randn(6, 1)]
        out = encoder(inp)
        assert out.size(-1) == 16
