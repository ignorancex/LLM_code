import torch
from omegaconf import OmegaConf, DictConfig

from torch import nn

from src.models.operators.proximity import ProxSSR
from src.models.operators.spectral_downsampling import ClusterDown
from src.models.operators.spectral_upsampling import ClustersUp


class SSRModule(nn.Module):
    def __init__(
            self,
            ms_channels: int,
            hs_channels: int,
            shared: bool,
            iter_stages: int,
            ssr_cfg: DictConfig,
    ):
        super(SSRModule, self).__init__()

        self.ms_channels = ms_channels
        self.hs_channels = hs_channels
        self.iter_stages = iter_stages
        self.shared = shared
        up = ssr_cfg.up
        prox = ssr_cfg.prox
        self.classes = ssr_cfg.up.args.classes
        self.Down = nn.ModuleList([ClusterDown(ms_channels=ms_channels, hs_channels=hs_channels)
                                   for i in range(self.iter_stages)])
        self.Up = nn.ModuleList([ClustersUp(**up.args)
                                 for i in range(self.iter_stages)])
        self.Prox = nn.ModuleList([ProxSSR(**prox.args)
                                   for i in range(self.iter_stages)])
        self.spectral_inter = ClustersUp(**up.args)
        self.clustering_cnn = nn.Sequential(
            nn.Conv2d(ms_channels, 32, kernel_size=3, stride=1, padding=1),  # Conv1
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Conv2
            nn.ReLU(),
            nn.Conv2d(64, self.classes, kernel_size=3, stride=1, padding=1),  # Conv3
            nn.Softmax(dim=1)  # Probabilidades de pertenencia a cada clúster
        )

    def forward(self, input):
        f = input
        clusters_probs = self.clustering_cnn(f)
        clusters = torch.argmax(clusters_probs, dim=1, keepdim=True)
        out = self.spectral_inter(f,clusters)
        out_list = [out]
        for i in range(self.iter_stages):
            out = self.Prox[i](out -self.Up[i](self.Down[i](out, clusters)-f,clusters))
            out_list.append(out)
        return out_list