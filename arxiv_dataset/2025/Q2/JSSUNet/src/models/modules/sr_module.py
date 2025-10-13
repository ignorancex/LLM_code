from omegaconf import DictConfig

from torch import nn

from torch.nn.functional import interpolate
from src.models.operators.downsampling import Downsampling
from src.models.operators.proximity import ProxSR
from src.models.operators.upsampling import DBPUp



class SRModule(nn.Module):
    def __init__(
            self,
            channels: int,
            sampling: int,
            shared: bool,
            iter_stages: int,
            sr_cfg: DictConfig,
    ):
        super(SRModule, self).__init__()

        self.sampling = sampling
        self.channels = channels
        self.iter_stages = iter_stages
        self.shared = shared

        self.Down = nn.ModuleList([Downsampling(**sr_cfg.down.args,groups=channels)
                                    for i in range(self.iter_stages)])
        self.Up = nn.ModuleList([DBPUp(**sr_cfg.up.args)
                                    for i in range(self.iter_stages)])
        self.Prox = nn.ModuleList([ProxSR(**sr_cfg.prox.args)
                                    for i in range(self.iter_stages)])

    def forward(self, f):
        out = interpolate(f, scale_factor=self.sampling)
        out_list = []
        for i in range(self.iter_stages):
            out = self.Prox[i](out -self.Up[i](self.Down[i](out) - f))
            out_list.append(out)
        return out_list
