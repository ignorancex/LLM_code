import torch
from omegaconf import OmegaConf, DictConfig
from torch import nn

from src.models.operators.frequency_estimation import InjectHighFreq, GetLowFreq
from src.models.operators.proximity import ProxFus

class FusModule(nn.Module):
    def __init__(
            self,
            ms_channels: int,
            hs_channels: int,
            shared: bool,
            iter_stages: int,
            fusion_cfg: DictConfig,
    ):
        super(FusModule, self).__init__()

        self.ms_channels = ms_channels
        self.hs_channels = hs_channels
        self.iter_stages = iter_stages
        self.shared = shared
        features = fusion_cfg.features
        print(fusion_cfg)
        self.sampling = fusion_cfg.sampling
        self.alpha = nn.ModuleList(
            [nn.Conv2d(hs_channels, hs_channels, kernel_size=1, stride=1, padding=0, bias=False, groups=hs_channels)
             for i in range(self.iter_stages)])

        self.Prox = nn.ModuleList([ProxFus(**fusion_cfg.prox.args)
                                   for i in range(self.iter_stages)])

        self.get_lf_from_ms = GetLowFreq(in_channels=ms_channels, out_channels=hs_channels, features=features,
                                         kernel_size=3)
        self.get_lf_from_hs = GetLowFreq(in_channels=hs_channels, out_channels=hs_channels, features=features,
                                         kernel_size=3)
        self.get_hf_from_ms = InjectHighFreq(ms_channels=ms_channels, hs_channels=hs_channels, features=features,
                                             kernel_size=3)
        self.init = InjectHighFreq(ms_channels=ms_channels, hs_channels=hs_channels, features=features, kernel_size=3)
        self.relu = nn.ReLU()

    def forward(self, ms, hs):
        hs_up = nn.functional.interpolate(hs, scale_factor=self.sampling, mode='bicubic')
        ms_down_and_up = nn.functional.interpolate(
            nn.functional.interpolate(ms, scale_factor=1. / self.sampling, mode='bicubic'), scale_factor=self.sampling,
            mode='bicubic')

        ms_tilde = self.get_lf_from_ms(ms_down_and_up)
        hs_tilde = self.get_lf_from_hs(hs_up)
        ms_hf = self.get_hf_from_ms(ms_tilde, ms, hs_up)
        out = self.init(ms_tilde, ms, hs_up)
        fused_list = [ms_hf]
        fused_list.append(out)
        for i in range(self.iter_stages):
            Brovey = ms_tilde * ( ms_tilde * out - hs_tilde * ms_hf)
            out = self.Prox[i](out - self.alpha[i](Brovey),ms=ms)
            fused_list.append(out)
        return fused_list, ms_tilde, hs_tilde