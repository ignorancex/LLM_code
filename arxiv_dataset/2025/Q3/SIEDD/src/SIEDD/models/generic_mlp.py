from . import MLPBlock, CudaMLP
from ..configs import MLPConfig, CudaMLPConfig, EncoderConfig


def get_mlp(dim_out, data_shape, cfg: EncoderConfig, use_first_init: bool = True):
    if isinstance(cfg.net, MLPConfig):
        net = MLPBlock(
            dim_out=dim_out,
            data_shape=data_shape,
            cfg=cfg,
            use_first_init=use_first_init,
        )
    elif isinstance(cfg.net, CudaMLPConfig):
        net = CudaMLP(dim_out=dim_out, data_shape=data_shape, cfg=cfg)
    else:
        raise ValueError("Not supported")
    return net
