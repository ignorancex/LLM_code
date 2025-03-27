import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmengine.registry import MODELS
from mmengine.model import BaseModel
from .map_converter import MapConverter
from .utils import visualize


@MODELS.register_module()
class BEV(BaseModel):
    def __init__(
        self,
        backbone,
        neck,
        n_class=14,
        dim=256,
        h_polar=256,
        w_polar=704,
        class_weights=[],
        stage=0,
        vis_list=[],
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck)
        self.pos_src = nn.Parameter(torch.randn(1, dim, 64, 1))
        self.pos_tgt = nn.Parameter(torch.randn(1, dim, 64, 1))
        self.transformer = nn.Transformer(
            d_model=dim,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=dim,
        )
        self.cnn = nn.Sequential(
            nn.Conv2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(n_class, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, n_class, 4, 2, 1),
            nn.Sigmoid(),
        )

        self.h_polar, self.w_polar = h_polar, w_polar
        self.map_converter_infos, self.map_converters = [], []
        self.class_weights = torch.tensor(class_weights)
        assert stage in [0, 1, 2, 3]
        self.stage = stage
        self.vis_list = vis_list
        self.epoch = None

    def init_map_converters(self, data_samples, device):
        map_converter_indices = []
        for data_sample in data_samples:
            for cam_item in data_sample.cams:
                map_converter_info = dict(
                    h_polar=self.h_polar,
                    w_polar=self.w_polar,
                    h_cart=cam_item["bev_shape"][1],
                    w_cart=cam_item["bev_shape"][2],
                    extents_polar=[0, 53],
                    extents_cart=cam_item["extents"],
                    cam2img=cam_item["cam2img"],
                    image_width=cam_item["image_width"],
                    colormap=cam_item["colormap"],
                    device=device,
                )
                if map_converter_info not in self.map_converter_infos:
                    self.map_converter_infos.append(map_converter_info)
                    self.map_converters.append(MapConverter(**map_converter_info))
                map_converter_indices.append(self.map_converter_infos.index(map_converter_info))
        return map_converter_indices

    def cart2polar(self, map_cart, map_converter_indices):
        """Convert map from cart to polar.

        Args:
            map_cart (torch.bool): (N, *, 200, 200)

        Returns:
            map_polar (torch.bool): (N, *, 256, 352)
        """
        map_polar = []
        for i, map_converter_index in enumerate(map_converter_indices):
            map_polar.append(self.map_converters[map_converter_index].cart2polar(map_cart[i]))
        return torch.stack(map_polar)

    def polar2cart(self, map_polar, map_converter_indices):
        """Convert map from polar to cart.

        Args:
            map_polar (torch.float): (N, *, 256, 352)

        Returns:
            map_cart (torch.float): (N, *, 200, 200)
        """
        map_cart = []
        for i, map_converter_index in enumerate(map_converter_indices):
            map_cart.append(self.map_converters[map_converter_index].polar2cart(map_polar[i]))
        return torch.stack(map_cart)

    def forward(self, data_samples, inputs, mode=None):
        image = torch.concat(inputs["image"])
        bev_map = torch.concat(inputs["bev_map"])
        map_converter_indices = self.init_map_converters(data_samples, bev_map.device)
        expand_map = self.cart2polar(bev_map, map_converter_indices)

        x = expand_map[:, :-1].float()

        if mode == "loss":
            if np.random.rand() < 0.5:
                image = torch.flip(image, [3])
                x = torch.flip(x, [3])
            if self.stage in [0, 2, 3]:
                feature = self.neck(self.backbone(image))
                feature = torch.flip(feature[0], [2])

                n, c, h, w = feature.size()
                src, tgt = feature + self.pos_src, torch.zeros_like(feature) + self.pos_tgt
                src, tgt = src.permute(2, 0, 3, 1).reshape(h, -1, c), tgt.permute(2, 0, 3, 1).reshape(h, -1, c)
                feature = self.transformer(src, tgt)
                feature = feature.reshape(h, n, w, c).permute(1, 3, 0, 2)

                feature = self.cnn(feature)
            if self.stage in [1, 2]:
                ze = self.encoder(x)  # (N, 14, 256, 704) -> (N, 256, 8, 22)
            if self.stage == 0:
                x_recon = self.decoder(feature)
                loss = F.binary_cross_entropy(x_recon, x, self.class_weights.to(x).view(1, -1, 1, 1))
            elif self.stage == 1:
                noise = torch.randn_like(ze)
                x_recon = self.decoder(np.sqrt(0.5) * ze + np.sqrt(0.5) * noise)
                loss = F.binary_cross_entropy(x_recon, x, self.class_weights.to(x).view(1, -1, 1, 1))
            elif self.stage == 2:
                loss = F.mse_loss(feature, ze.detach())
            elif self.stage == 3:
                x_recon = self.decoder(feature.detach())
                loss = F.binary_cross_entropy(x_recon, x, self.class_weights.to(x).view(1, -1, 1, 1))
            return dict(loss=loss)
        else:
            if self.stage in [0, 2, 3]:
                with torch.no_grad():
                    feature = self.neck(self.backbone(image))
                    feature = torch.flip(feature[0], [2])

                    n, c, h, w = feature.size()
                    src, tgt = feature + self.pos_src, torch.zeros_like(feature) + self.pos_tgt
                    src, tgt = src.permute(2, 0, 3, 1).reshape(h, -1, c), tgt.permute(2, 0, 3, 1).reshape(h, -1, c)
                    feature = self.transformer(src, tgt)
                    feature = feature.reshape(h, n, w, c).permute(1, 3, 0, 2)

                    feature = self.cnn(feature)
            if self.stage == 1:
                with torch.no_grad():
                    feature = self.encoder(x)  # (N, 14, 256, 704) -> (N, 256, 8, 22)
            with torch.no_grad():
                x_recon = self.decoder(feature)

            x_bev = self.polar2cart(x_recon, map_converter_indices)

            for i, data_sample in enumerate(data_samples):
                for j, cam in enumerate(data_sample.cams):
                    if cam["image_path"] in self.vis_list:
                        folder_name = cam["camera_type"] + "_" + cam["image_path"].split("/")[-1].split(".")[0]
                        visualize(
                            f"visualizations/stage_{self.stage}/epoch_{self.epoch}/{folder_name}",
                            bev_map=bev_map[i * len(data_sample.cams) + j],
                            expand_map=expand_map[i * len(data_sample.cams) + j],
                            x_recon=x_recon[i * len(data_sample.cams) + j] >= 0.5,
                            x_bev=x_bev[i * len(data_sample.cams) + j] >= 0.5,
                            image=cam["image_path"],
                        )

            return x_bev, bev_map
