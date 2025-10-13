import torch
from torch import nn
from omegaconf import DictConfig




from src.models.modules.fus_module import FusModule
from src.models.modules.sr_module import SRModule
from src.models.modules.ssr_module import SSRModule
from src.models.operators.postprocessing import NLPost


class JSSUNet(nn.Module):
    def __init__(
            self,
            ms_channels: int,
            hs_channels: int,
            sampling: int,
            iter_stages: int,
            shared: bool,
            sr_cfg: DictConfig,
            ssr_cfg: DictConfig,
            fusion_cfg: DictConfig,
            postprocessing: DictConfig,
    ):
        super(JSSUNet, self).__init__()

        self.sampling = sampling
        self.hs_channels = hs_channels
        self.ms_channels = ms_channels
        self.iter_stages = iter_stages


        self.SR = SRModule(channels=ms_channels, sampling=sampling, shared=shared, iter_stages=iter_stages, sr_cfg=sr_cfg)
        self.SSR = SSRModule(ms_channels=ms_channels, hs_channels=hs_channels, shared=shared, iter_stages=iter_stages, ssr_cfg=ssr_cfg)
        self.Fusion = FusModule(ms_channels=ms_channels, hs_channels=hs_channels, shared=shared, iter_stages=iter_stages, fusion_cfg=fusion_cfg)

        self.post = NLPost(**postprocessing)
        self.fine_tune = False



    def forward(self, ms_low):
        ms_high_list = self.SR(ms_low)
        hs_low_list = self.SSR(ms_low)
        fused_list, ms_lf, hs_lf = self.Fusion(ms=ms_high_list[-1], hs=hs_low_list[-1])
        if self.fine_tune:
            pred = self.post(fused_list[-1])
        else:
            pred = fused_list[-1]
        return {"pred": pred, "ms_list": ms_high_list, "hs_list": hs_low_list, "fused_list": fused_list[:-1]}

    def freeze_post(self ):
        for param in self.post.parameters():
            param.requires_grad = False

    def fine_tune_param_upd(self ):
        print("HOLA")
        for param in self.SR.parameters():
            param.requires_grad = False
        for param in self.SSR.parameters():
            param.requires_grad = False
        for param in self.Fusion.parameters():
            param.requires_grad = False
        for param in self.post.parameters():
            param.requires_grad = True

