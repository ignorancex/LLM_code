import pytest
from SIEDD.models import MLPBlock
from SIEDD.configs import MLPConfig, NoPosEncode, EncoderConfig


class TestMLPBlock:
    def test_empty(self):
        cfg = EncoderConfig(
            net=MLPConfig(num_layers=-1, pos_encode_cfg=NoPosEncode(dim_in=4))
        )
        a = MLPBlock(dim_out=4, data_shape=[3, 1920, 1080], cfg=cfg)
        n_params = sum([param.numel() for param in a.parameters()])
        assert n_params == 0

    def test_empty_2(self):
        cfg = EncoderConfig(
            net=MLPConfig(num_layers=-1, pos_encode_cfg=NoPosEncode(dim_in=3))
        )
        with pytest.raises(ValueError):
            MLPBlock(dim_out=4, data_shape=[3, 1920, 1080], cfg=cfg)

    def test_1(self):
        cfg = EncoderConfig(
            net=MLPConfig(
                num_layers=0, pos_encode_cfg=NoPosEncode(dim_in=2), use_bias=False
            )
        )
        a = MLPBlock(dim_out=4, data_shape=[3, 1920, 1080], cfg=cfg)
        n_params = sum([param.numel() for param in a.parameters()])
        assert n_params == 8

    def test_2(self):
        cfg = EncoderConfig(
            net=MLPConfig(
                dim_hidden=4,
                num_layers=1,
                pos_encode_cfg=NoPosEncode(dim_in=2),
                use_bias=False,
            )
        )
        a = MLPBlock(dim_out=5, data_shape=[3, 1920, 1080], cfg=cfg)
        n_params = sum([param.numel() for param in a.parameters()])
        assert n_params == 8 + 20

    def test_3(self):
        cfg = EncoderConfig(
            net=MLPConfig(
                dim_hidden=4,
                num_layers=2,
                pos_encode_cfg=NoPosEncode(dim_in=2),
                use_bias=False,
            )
        )
        a = MLPBlock(dim_out=4, data_shape=[3, 1920, 1080], cfg=cfg)
        n_params = sum([param.numel() for param in a.parameters()])
        assert n_params == 8 + 16 + 16
