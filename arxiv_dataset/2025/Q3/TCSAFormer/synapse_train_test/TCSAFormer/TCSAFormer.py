import copy
import math
import torch
import torch.nn as nn
from fairscale.nn.checkpoint import checkpoint_wrapper
from timm.models.layers import DropPath, trunc_normal_
from .cam import CAM
from ._common import DWConv


def get_pe_layer(name='none'):
    if name == 'none':
        return nn.Identity()
    else:
        raise ValueError(f'PE name {name} is not surpported!')


class DBFFN(nn.Module):
    def __init__(self, dim, bias, ffn_expansion_factor=2):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.norm_in = nn.BatchNorm2d(dim)
        self.norm_proj = nn.BatchNorm2d(hidden_features)
        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv3x3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1,
                                   groups=hidden_features, bias=bias)
        self.dwconv7x7 = nn.Conv2d(hidden_features, hidden_features, kernel_size=7, padding=3,
                                   groups=hidden_features, bias=bias)

        self.act3 = nn.GELU()
        self.act7 = nn.GELU()

        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.dwconv3x3_1 = nn.Conv2d(hidden_features, dim, kernel_size=3, padding=1, groups=dim,
                                     bias=bias)
        self.dwconv7x7_1 = nn.Conv2d(hidden_features, dim, kernel_size=7, padding=3, groups=dim,
                                     bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.norm_out = nn.BatchNorm2d(dim)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.norm_in(x)
        x = self.project_in(x)
        x = self.norm_proj(x)

        x3_1, x3_2 = self.act3(self.dwconv3x3(x)).chunk(2, dim=1)
        x7_1, x7_2 = self.act7(self.dwconv7x7(x)).chunk(2, dim=1)

        x1 = torch.cat([x3_1, x7_1], dim=1)
        x2 = torch.cat([x3_2, x7_2], dim=1)

        x1 = self.norm1(self.dwconv3x3_1(x1))
        x2 = self.norm2(self.dwconv7x7_1(x2))

        x = torch.cat([x1, x2], dim=1)
        x = self.project_out(x)
        x = self.norm_out(x)
        x = x.permute(0, 2, 3, 1)
        return x


class Skip(nn.Module):
    def __init__(self, in_channels, out_channels, rate=4):
        super(Skip, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        x = x * x_channel_att
        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att
        return out


class Block(nn.Module):
    def __init__(self,
                 dim,
                 H,
                 mlp_ratio,
                 stage,
                 keep_ratio,
                 merge_ratio,
                 drop_path=0.,
                 num_heads=8,
                 before_attn_dwconv=3,
                 mlp_dwconv=True,
                 ):
        super().__init__()
        if before_attn_dwconv > 0:
            self.pos_embed = nn.Conv2d(dim, dim, kernel_size=before_attn_dwconv, padding=1, groups=dim)  # DWConv
        else:
            self.pos_embed = lambda x: 0
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = CAM(dim=dim, num_heads=num_heads, keep_ratio=keep_ratio, H=H, merge_ratio=merge_ratio)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = DBFFN(dim=dim, bias=False, ffn_expansion_factor=2)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x.permute(0, 2, 3, 1)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.permute(0, 3, 1, 2)
        return x


class TCSA_Former(nn.Module):
    def __init__(self,
                 mlp_ratios,
                 depth=[3, 4, 8, 3],
                 H=[56, 28, 14, 7],
                 in_chans=3,
                 keep_ratio=[0.5, 0.6, 0.7, 0.9],
                 merge_ratio=[0.3, 0.2, 0.1, 0.1],
                 num_classes=100,
                 embed_dim=[64, 128, 320, 512],
                 head_dim=64,
                 drop_path_rate=0.,
                 use_checkpoint_stages=[],
                 qk_dims=[None, None, None, None],
                 pre_norm=True,
                 pe=None,
                 pe_stages=[0],
                 before_attn_dwconv=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        ################################################
        # downsample layers (patch embeddings)
        self.downsample_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.skip_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim[0] // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0] // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim[0] // 2, embed_dim[0], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0]),
        )

        if (pe is not None) and 0 in pe_stages:
            stem.append(get_pe_layer(emb_dim=embed_dim[0], name=pe))
        if use_checkpoint_stages:
            stem = checkpoint_wrapper(stem)
        self.downsample_layers.append(stem)
        nheads = [dim // head_dim for dim in qk_dims]
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.Conv2d(embed_dim[i], embed_dim[i + 1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.BatchNorm2d(embed_dim[i + 1])
            )
            if (pe is not None) and i + 1 in pe_stages:
                downsample_layer.append(get_pe_layer(emb_dim=embed_dim[i + 1], name=pe))
            if use_checkpoint_stages:
                downsample_layer = checkpoint_wrapper(downsample_layer)
            self.downsample_layers.append(downsample_layer)
        for i in range(3):
            upsample_layer = nn.Sequential(
                nn.Conv2d(embed_dim[3 - i], embed_dim[2 - i], kernel_size=(1, 1)),
                nn.BatchNorm2d(embed_dim[2 - i])
            )
            self.upsample_layers.append(upsample_layer)
        for i in range(3):
            skip = Skip(embed_dim[2 - i], embed_dim[2 - i])
            self.skip_layers.append(skip)
        ##########################################################################
        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stages_de = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=embed_dim[i],
                        H=H[i],
                        stage=i,
                        keep_ratio=keep_ratio[i],
                        merge_ratio=merge_ratio[i],
                        mlp_ratio=mlp_ratios[i],
                        drop_path=dp_rates[cur + j],
                        num_heads=nheads[i],
                        before_attn_dwconv=before_attn_dwconv) for j in range(depth[i])],
            )
            if i < 3:
                stages_de = nn.Sequential(
                    *[Block(dim=embed_dim[2 - i],
                            H=H[2 - i],
                            stage=4 + i,
                            keep_ratio=keep_ratio[2 - i],
                            merge_ratio=merge_ratio[2 - i],
                            mlp_ratio=mlp_ratios[2 - i],
                            drop_path=dp_rates[cur + j],
                            num_heads=nheads[2 - i],
                            before_attn_dwconv=before_attn_dwconv) for j in range(depth[2 - i])],
                )
                self.stages_de.append(stages_de)
            if i in use_checkpoint_stages:
                stage = checkpoint_wrapper(stage)
            self.stages.append(stage)
            cur += depth[i]

        ##########################################################################
        self.output = nn.Conv2d(in_channels=embed_dim[0], out_channels=self.num_classes, kernel_size=1, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x):
        out = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            out.append(x)
        for i in range(3):
            x = nn.functional.interpolate(x, size=(x.size(2) * 2, x.size(3) * 2), mode='bilinear', align_corners=True)
            x = self.upsample_layers[i](x)
            x = out[2 - i] + x
            x = self.skip_layers[i](x)
            x = self.stages_de[i](x)
        x = nn.functional.interpolate(x, size=(x.size(2) * 4, x.size(3) * 4), mode='bilinear', align_corners=True)
        x = self.output(x)
        return x


class TCSAFormer(nn.Module):
    def __init__(self, img_size=256, in_chans=3, num_classes=1, n_win=8):
        super(TCSAFormer, self).__init__()
        self.TCSAFormer = TCSA_Former(
            # --------------------------
            depth=[2, 2, 8, 2],
            # embed_dim=[96, 192, 384, 768],
            embed_dim=[64, 128, 256, 512],
            mlp_ratios=[4, 4, 4, 4],
            before_attn_dwconv=3,
            num_classes=9,
            qk_dims=[64, 128, 256, 512],
            # qk_dims=[96, 192, 384, 768],
            head_dim=32,
            pre_norm=True,
            pe=None,
            # use grad ckpt to save memory on old gpus
            use_checkpoint_stages=[],
            # drop_path_rate=0.3,
            drop_path_rate=0.2)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        out = self.TCSAFormer(x)
        return out

    def load_from(self):
        pretrained_path = '/home/xiazunhui/project/MedFormer/classification/TCSFormer_ImgNet/best.pth'
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            model_dict = self.TCSAFormer.state_dict()
            full_dict = copy.deepcopy(pretrained_dict['model'])
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, full_dict[k].shape,
                                                                                  model_dict[k].shape))
                        del full_dict[k]
                if k not in model_dict:
                    del full_dict[k]
            msg = self.TCSAFormer.load_state_dict(full_dict, strict=False)
            print(msg)
        else:
            print("none pretrain")
