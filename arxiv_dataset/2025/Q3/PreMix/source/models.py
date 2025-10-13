import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np

from pathlib import Path
from typing import Optional
from einops import rearrange
from omegaconf import DictConfig, OmegaConf

from source.vision_transformer import vit_small, vit4k_xs
from source.model_utils import (
    Attn_Net_Gated,
    PositionalEncoderFactory,
    masked_softmax,
    get_lambda,
    get_lambda_per_sample,
    to_one_hot,
    mixup_process,
    mixup_process_per_sample,
)
from source.utils import update_state_dict

class LogisticRegression(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
    ):
        super(LogisticRegression, self).__init__()
        self.backbone = backbone

        self.classifier = nn.Linear(192, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        embeddings = self.backbone(x, mask)
        zero_embeddings = torch.zeros_like(embeddings)
        logits = self.classifier(embeddings)
        return embeddings, zero_embeddings, logits

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = self.classifier.to(device)

class LogisticRegressionMixing(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
    ):
        super(LogisticRegressionMixing, self).__init__()
        self.backbone = backbone

        self.classifier = nn.Linear(192, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, target: torch.Tensor=None, mixup: bool=False, manifold_mixup: bool=False, mixup_type: str="random"):
        embeddings, reweighted_target = self.backbone(x, mask, target, mixup, manifold_mixup, mixup_type) # reweighted target is dummy var when validation or testing
        logits = self.classifier(embeddings)
        return embeddings, reweighted_target, logits

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = self.classifier.to(device)

class ModelFactory:
    def __init__(
        self,
        level: str,
        model_opt: Optional[DictConfig] = None,
    ):

        if level == "global":

            if model_opt.pretrained or getattr(model_opt, 'mixing', None) is None:
                self.model = GlobalHIPT(
                    embed_dim_region=model_opt.embed_dim_region,
                    dropout=model_opt.dropout,
                    slide_pos_embed=model_opt.slide_pos_embed,
                )
            else:
                self.model = GlobalHIPTMixing(
                    embed_dim_region=model_opt.embed_dim_region,
                    dropout=model_opt.dropout,
                    slide_pos_embed=model_opt.slide_pos_embed,
                    mixing=model_opt.mixing,
                )
        elif level == "local":
            self.model = LocalGlobalHIPT(
                img_size_4096=model_opt.img_size_4096,
                patch_size_4096=model_opt.patch_size_4096,
                pretrain_4096=model_opt.pretrain_4096,
                freeze_4096=model_opt.freeze_4096,
                freeze_4096_pos_embed=model_opt.freeze_4096_pos_embed,
                dropout=model_opt.dropout,
            )
        else:
            self.model = HIPT(
                pretrain_256=model_opt.pretrain_256,
                freeze_256=model_opt.freeze_256,
                freeze_256_pos_embed=model_opt.freeze_256_pos_embed,
                img_size_256=model_opt.img_size_256,
                patch_size_256=model_opt.patch_size_256,
                pretrain_4096=model_opt.pretrain_4096,
                freeze_4096=model_opt.freeze_4096,
                freeze_4096_pos_embed=model_opt.freeze_4096_pos_embed,
                img_size_4096=model_opt.img_size_4096,
                patch_size_4096=model_opt.patch_size_4096,
                dropout=model_opt.dropout,
            )

    def get_model(self):
        return self.model

class GlobalHIPT(nn.Module):
    def __init__(
        self,
        embed_dim_region: int = 192,
        d_model: int = 192,
        dropout: float = 0.25,
        slide_pos_embed: Optional[DictConfig] = None,
    ):

        super(GlobalHIPT, self).__init__()
        self.slide_pos_embed = slide_pos_embed

        # Global Aggregation
        self.global_phi = nn.Sequential(
            nn.Linear(embed_dim_region, 192), nn.ReLU(), nn.Dropout(dropout)
        )

        if self.slide_pos_embed.use:
            pos_encoding_options = OmegaConf.create(
                {
                    "agg_method": "concat",
                    "dim": d_model,
                    "dropout": dropout,
                    "max_seq_len": slide_pos_embed.max_seq_len,
                    "max_nslide": slide_pos_embed.max_nslide,
                    "tile_size": slide_pos_embed.tile_size,
                }
            )
            self.pos_encoder = PositionalEncoderFactory(
                slide_pos_embed.type, slide_pos_embed.learned, pos_encoding_options
            ).get_pos_encoder()

        self.global_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=192,
                nhead=3,
                dim_feedforward=192,
                dropout=dropout,
                activation="relu",
            ),
            num_layers=2,
        )
        self.global_attn_pool = Attn_Net_Gated(
            L=192, D=192, dropout=dropout, num_classes=1
        )
        self.global_rho = nn.Sequential(
            *[nn.Linear(192, 192), nn.ReLU(), nn.Dropout(dropout)]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):

        x = self.global_phi(x)  # x: [BS, R, 192]

        if self.slide_pos_embed.use:
            x = self.pos_encoder(x)

        # in nn.TransformerEncoderLayer, batch_first defaults to False
        # hence, input is expected to be of shape (seq_length, batch, emb_size)
        x = self.global_transformer(x.transpose(0, 1), src_key_padding_mask=mask)
        att, x = self.global_attn_pool(x)  # att: [R, BS, 1], x: [R, BS, 192]

        x = torch.transpose(x, 0, 1)  # x: [BS, R, 192]

        att = att.permute(1, 2, 0)  # att: [BS, 1, R]
        att = masked_softmax(att, mask.unsqueeze(1))  # att: [BS, 1, R] -> prevent attention mechanism assigns some attention to the padded regions

        # x = torch.transpose(x, 0, 1)  # x: [BS, R, 192]
        x_att = torch.bmm(att, x)  # x_att: [BS, 1, 192]
        x_wsi = self.global_rho(x_att.squeeze(1))  # x_wsi: [BS, 192]

        return x_wsi

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.global_phi = self.global_phi.to(device)
        if self.slide_pos_embed.use:
            self.pos_encoder = self.pos_encoder.to(device)
        self.global_transformer = self.global_transformer.to(device)
        self.global_attn_pool = self.global_attn_pool.to(device)
        self.global_rho = self.global_rho.to(device)

    def __repr__(self) -> str:
        num_params = 0
        num_params_train = 0
        for param in self.parameters():
            n = param.numel()
            num_params += n
            if param.requires_grad:
                num_params_train += n
        main_str = f"Total number of parameters: {num_params}\n"
        main_str += f"Total number of trainable parameters: {num_params_train}"
        return main_str

class ManifoldMixupTransformerEncoder(nn.TransformerEncoder):
    def __init__(self, encoder_layer, num_layers, num_classes, mixup_alpha=None, mixup_alpha_per_sample=None,norm=None):
        super().__init__(encoder_layer, num_layers, norm)
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha
        self.mixup_alpha_per_sample = mixup_alpha_per_sample

    def forward(self, src, src_key_padding_mask=None, target_reweighted=None, lam=None, target=None, mixup_type="random"):

        for mod in self.layers:
            src_transposed = src.transpose(0, 1)

            if target_reweighted is not None:
                if self.mixup_alpha_per_sample:
                    src_transposed, target_reweighted = mixup_process_per_sample(src_transposed, target_reweighted.cuda(), lam=lam, target=target, mixup_type=mixup_type)
                else:
                    src_transposed, target_reweighted = mixup_process(src_transposed, target_reweighted.cuda(), lam=lam, target=target, mixup_type=mixup_type)
                
            src = src_transposed.transpose(0, 1)
            src = mod(src, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            src = self.norm(src)
        
        if target_reweighted is not None:
            return src, target_reweighted
        
        return src

# Mixed approaches are used only when fine-tuning the encoder for downstream classification in PreMix -> not during pretraining or in the baseline
class GlobalHIPTMixing(nn.Module):
    def __init__(
        self,
        embed_dim_region: int = 192,
        d_model: int = 192,
        dropout: float = 0.25,
        slide_pos_embed: Optional[DictConfig] = None,
        mixing: Optional[DictConfig] = None,
    ):
        super(GlobalHIPTMixing, self).__init__()
        self.slide_pos_embed = slide_pos_embed

        self.manifold_mixup_transformer = mixing.manifold_mixup_transformer
        self.mixup_alpha = mixing.mixup_alpha
        self.mixup_alpha_per_sample = mixing.mixup_alpha_per_sample

        self.global_phi = nn.Sequential(
            nn.Linear(embed_dim_region, 192), nn.ReLU(), nn.Dropout(dropout)
        )

        if self.slide_pos_embed.use:
            pos_encoding_options = OmegaConf.create(
                {
                    "agg_method": "concat",
                    "dim": d_model,
                    "dropout": dropout,
                    "max_seq_len": slide_pos_embed.max_seq_len,
                    "max_nslide": slide_pos_embed.max_nslide,
                    "tile_size": slide_pos_embed.tile_size,
                }
            )
            self.pos_encoder = PositionalEncoderFactory(
                slide_pos_embed.type, slide_pos_embed.learned, pos_encoding_options
            ).get_pos_encoder()

        layer_config = nn.TransformerEncoderLayer(
            d_model=192,
            nhead=3,
            dim_feedforward=192,
            dropout=dropout,
            activation="relu",
        )

        if self.manifold_mixup_transformer:
            self.global_transformer = ManifoldMixupTransformerEncoder(
                layer_config, num_layers=2, num_classes=2, mixup_alpha=self.mixup_alpha, mixup_alpha_per_sample=self.mixup_alpha_per_sample
            )
        else:
            self.global_transformer = nn.TransformerEncoder(layer_config, num_layers=2)

        self.global_attn_pool = Attn_Net_Gated(
            L=192, D=192, dropout=dropout, num_classes=1
        )
        self.global_rho = nn.Sequential(
            *[nn.Linear(192, 192), nn.ReLU(), nn.Dropout(dropout)]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor, target: torch.Tensor=None, mixup: bool=False, manifold_mixup: bool=False, mixup_type="random"):
        if manifold_mixup:
            layer_mix = random.randint(0, 1)
        elif mixup:
            layer_mix = 0
        else:
            layer_mix = None

        if self.mixup_alpha is not None:
            if self.mixup_alpha_per_sample:
                lam = get_lambda_per_sample(x.size(0), self.mixup_alpha)
            else:
                lam = get_lambda(self.mixup_alpha)
            lam = torch.from_numpy(np.array([lam]).astype('float32')).cuda()
        
        if target is not None:
            target_reweighted = to_one_hot(target, num_classes=2)

        if layer_mix == 0:
            if self.mixup_alpha_per_sample:
                x, target_reweighted = mixup_process_per_sample(x, target_reweighted.cuda(), lam=lam, target=target, mixup_type=mixup_type)
            else:
                x, target_reweighted = mixup_process(x, target_reweighted.cuda(), lam=lam, target=target, mixup_type=mixup_type)

        x = self.global_phi(x)

        if self.slide_pos_embed.use:
            x = self.pos_encoder(x)

        x_transposed = x.transpose(0, 1)
        if self.manifold_mixup_transformer and target is not None:
            x_transposed, target_reweighted = self.global_transformer(x_transposed, src_key_padding_mask=mask, target_reweighted=target_reweighted, lam=lam, target=target, mixup_type=mixup_type)
        else:
            x_transposed = self.global_transformer(x_transposed, src_key_padding_mask=mask)

        x = x_transposed.transpose(0, 1)
        if layer_mix == 1:
            if self.mixup_alpha_per_sample:
                x, target_reweighted = mixup_process_per_sample(x, target_reweighted.cuda(), lam=lam, target=target, mixup_type=mixup_type)
            else:
                x, target_reweighted = mixup_process(x, target_reweighted.cuda(), lam=lam, target=target, mixup_type=mixup_type)

        x_transposed = x.transpose(0, 1)

        att, x = self.global_attn_pool(x_transposed)
        att = att.permute(1, 2, 0)
        att = masked_softmax(att, mask.unsqueeze(1))

        x_transposed = x.transpose(0, 1)
        x_att = torch.bmm(att, x_transposed)
        x_wsi = self.global_rho(x_att.squeeze(1))

        if target is not None:
            return x_wsi, target_reweighted
        else:
            # reweighted target is dummy var when validation or testing
            target_reweighted_dummy = torch.zeros_like(x_wsi)
            return x_wsi, target_reweighted_dummy
    
    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.global_phi = self.global_phi.to(device)
        if self.slide_pos_embed.use:
            self.pos_encoder = self.pos_encoder.to(device)
        self.global_transformer = self.global_transformer.to(device)
        self.global_attn_pool = self.global_attn_pool.to(device)
        self.global_rho = self.global_rho.to(device)

    def __repr__(self) -> str:
        num_params = 0
        num_params_train = 0
        for param in self.parameters():
            n = param.numel()
            num_params += n
            if param.requires_grad:
                num_params_train += n
        main_str = f"Total number of parameters: {num_params}\n"
        main_str += f"Total number of trainable parameters: {num_params_train}"
        return main_str
    
class LocalGlobalHIPT(nn.Module):
    def __init__(
        self,
        pretrain_4096: str = "path/to/pretrained/vit_4096/weights.pth",
        embed_dim_256: int = 384,
        embed_dim_region: int = 192,
        img_size_4096: int = 3584,
        patch_size_4096: int = 256,
        freeze_4096: bool = True,
        freeze_4096_pos_embed: bool = True,
        dropout: float = 0.25,
    ):

        super(LocalGlobalHIPT, self).__init__()

        checkpoint_key = "teacher"

        self.vit_4096 = vit4k_xs(
            img_size=img_size_4096,
            patch_size=patch_size_4096,
            input_embed_dim=embed_dim_256,
            output_embed_dim=embed_dim_region,
        )

        if Path(pretrain_4096).is_file():
            print("Loading pretrained weights for ViT_4096 model...")
            state_dict = torch.load(pretrain_4096, map_location="cpu")
            if checkpoint_key is not None and checkpoint_key in state_dict:
                print(f"Take key {checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            state_dict, msg = update_state_dict(self.vit_4096.state_dict(), state_dict)
            self.vit_4096.load_state_dict(state_dict, strict=False)
            print(f"Pretrained weights found at {pretrain_4096}")
            print(msg)

        else:
            print(
                f"{pretrain_4096} doesnt exist ; please provide path to existing file"
            )

        if freeze_4096:
            print("Freezing pretrained ViT_4096 model")
            for name, param in self.vit_4096.named_parameters():
                param.requires_grad = False
                if name == "pos_embed":
                    param.requires_grad = not (freeze_4096_pos_embed)
            print(
                f"ViT_4096 positional embedding layer frozen: {freeze_4096_pos_embed}"
            )
            print("Done")

        # Global Aggregation
        self.global_phi = nn.Sequential(
            nn.Linear(embed_dim_region, 192), nn.ReLU(), nn.Dropout(dropout)
        )
        self.global_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=192,
                nhead=3,
                dim_feedforward=192,
                dropout=dropout,
                activation="relu",
            ),
            num_layers=2,
        )
        self.global_attn_pool = Attn_Net_Gated(
            L=192, D=192, dropout=dropout, num_classes=1
        )
        self.global_rho = nn.Sequential(
            *[nn.Linear(192, 192), nn.ReLU(), nn.Dropout(dropout)]
        )

    def forward(self, x):

        # x = [M, 256, 384]
        x = self.vit_4096(x.unfold(1, 16, 16).transpose(1, 2))  # [M, 192]
        x = self.global_phi(x)  # [M, 192]

        # in nn.TransformerEncoderLayer, batch_first defaults to False
        # hence, input is expected to be of shape (seq_length, batch, emb_size)
        x = self.global_transformer(x.unsqueeze(1)).squeeze(1)
        att, x = self.global_attn_pool(x)
        att = torch.transpose(att, 1, 0)
        att = F.softmax(att, dim=1)
        x_att = torch.mm(att, x)
        x_wsi = self.global_rho(x_att)

        logits = self.classifier(x_wsi)

        return logits, x_wsi, att

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.vit_4096 = nn.DataParallel(self.vit_4096, device_ids=device_ids).to(
                "cuda:0"
            )

        self.global_phi = self.global_phi.to(device)
        self.global_transformer = self.global_transformer.to(device)
        self.global_attn_pool = self.global_attn_pool.to(device)
        self.global_rho = self.global_rho.to(device)

    def __repr__(self) -> str:
        num_params = 0
        num_params_train = 0
        for param in self.parameters():
            n = param.numel()
            num_params += n
            if param.requires_grad:
                num_params_train += n
        main_str = f"Total number of parameters: {num_params}\n"
        main_str += f"Total number of trainable parameters: {num_params_train}"
        return main_str


class HIPT(nn.Module):
    def __init__(
        self,
        pretrain_256: str = "path/to/pretrained/vit_256/weights.pth",
        freeze_256: bool = True,
        pretrain_4096: str = "path/to/pretrained/vit_4096/weights.pth",
        freeze_4096: bool = True,
        img_size_256: int = 224,
        patch_size_256: int = 16,
        embed_dim_256: int = 384,
        img_size_4096: int = 3584,
        patch_size_4096: int = 256,
        embed_dim_region: int = 192,
        freeze_256_pos_embed: bool = True,
        freeze_4096_pos_embed: bool = True,
        dropout: float = 0.25,
    ):

        super(HIPT, self).__init__()

        checkpoint_key = "teacher"

        self.vit_256 = vit_small(
            img_size=img_size_256,
            patch_size=patch_size_256,
            embed_dim=embed_dim_256,
        )

        if Path(pretrain_256).is_file():
            print("Loading pretrained weights for ViT_256 model...")
            state_dict = torch.load(pretrain_256, map_location="cpu")
            if checkpoint_key is not None and checkpoint_key in state_dict:
                print(f"Take key {checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            state_dict, msg = update_state_dict(self.vit_256.state_dict(), state_dict)
            self.vit_256.load_state_dict(state_dict, strict=False)
            print(f"Pretrained weights found at {pretrain_256}")
            print(msg)

        else:
            print(f"{pretrain_256} doesnt exist ; please provide path to existing file")

        if freeze_256:
            print("Freezing pretrained ViT_256 model")
            for name, param in self.vit_256.named_parameters():
                param.requires_grad = False
                if name == "pos_embed":
                    param.requires_grad = not (freeze_256_pos_embed)
            print("Done")

        self.vit_4096 = vit4k_xs(
            img_size=img_size_4096,
            patch_size=patch_size_4096,
            input_embed_dim=embed_dim_256,
            output_embed_dim=embed_dim_region,
        )

        if Path(pretrain_4096).is_file():
            print("Loading pretrained weights for ViT_4096 model...")
            state_dict = torch.load(pretrain_4096, map_location="cpu")
            if checkpoint_key is not None and checkpoint_key in state_dict:
                print(f"Take key {checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            state_dict, msg = update_state_dict(self.vit_4096.state_dict(), state_dict)
            self.vit_4096.load_state_dict(state_dict, strict=False)
            print(f"Pretrained weights found at {pretrain_4096}")
            print(msg)

        else:
            print(
                f"{pretrain_4096} doesnt exist ; please provide path to existing file"
            )

        if freeze_4096:
            print("Freezing pretrained ViT_4096 model")
            for name, param in self.vit_4096.named_parameters():
                param.requires_grad = False
                if name == "pos_embed":
                    param.requires_grad = not (freeze_4096_pos_embed)
            print("Done")

        # Global Aggregation
        self.global_phi = nn.Sequential(
            nn.Linear(embed_dim_region, 192), nn.ReLU(), nn.Dropout(dropout)
        )
        self.global_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=192,
                nhead=3,
                dim_feedforward=192,
                dropout=dropout,
                activation="relu",
            ),
            num_layers=2,
        )
        self.global_attn_pool = Attn_Net_Gated(
            L=192, D=192, dropout=dropout, num_classes=1
        )
        self.global_rho = nn.Sequential(
            *[nn.Linear(192, 192), nn.ReLU(), nn.Dropout(dropout)]
        )

    def forward(self, x):

        # x = [M, 3, 4096, 4096]
        # TODO: add prepare_img_tensor method
        x = x.unfold(2, 256, 256).unfold(3, 256, 256)  # [M, 3, 16, 16, 256, 256]
        x = rearrange(x, "b c p1 p2 w h -> (b p1 p2) c w h")  # [M*16*16, 3, 256, 256]
        x = x.to(self.device_256, non_blocking=True)  # [M*256, 3, 256, 256]

        # x = self.vit_256(x)                                     # [M, 256, 384]
        features_256 = []
        for mini_bs in range(0, x.shape[0], 256):
            minibatch = x[mini_bs : mini_bs + 256]
            f = self.vit_256(minibatch).detach()  # [256, 384]
            features_256.append(f.unsqueeze(0))

        x = torch.vstack(features_256)  # [M, 256, 384]
        x = x.to(self.device_4096, non_blocking=True)
        x = self.vit_4096(
            x.unfold(1, 16, 16).transpose(1, 2)
        )  # x = [M, 16, 16, 384] -> [M, 192]

        x = x.to(self.device_256, non_blocking=True)
        x = self.global_phi(x)

        # in nn.TransformerEncoderLayer, batch_first defaults to False
        # hence, input is expected to be of shape (seq_length, batch, emb_size)
        x = self.global_transformer(x.unsqueeze(1)).squeeze(1)
        att, x = self.global_attn_pool(x)
        att = torch.transpose(att, 1, 0)
        att = F.softmax(att, dim=1)
        x_att = torch.mm(att, x)
        x_wsi = self.global_rho(x_att)

        logits = self.classifier(x_wsi)

        return logits

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == torch.device("cuda"):
            assert torch.cuda.device_count() >= 2
            self.device_256 = torch.device("cuda:0")
            self.device_4096 = torch.device("cuda:1")
            device = self.device_256
        else:
            self.device_256 = device
            self.device_4096 = device

        self.vit_256 = self.vit_256.to(self.device_256)
        self.vit_4096 = self.vit_4096.to(self.device_4096)

        self.global_phi = self.global_phi.to(device)
        self.global_transformer = self.global_transformer.to(device)
        self.global_attn_pool = self.global_attn_pool.to(device)
        self.global_rho = self.global_rho.to(device)

    def __repr__(self) -> str:
        num_params = 0
        num_params_train = 0
        for param in self.parameters():
            n = param.numel()
            num_params += n
            if param.requires_grad:
                num_params_train += n
        main_str = f"Total number of parameters: {num_params}\n"
        main_str += f"Total number of trainable parameters: {num_params_train}"
        return main_str

class GlobalFeatureExtractor(nn.Module):
    def __init__(
        self,
        pretrain_256: str = "path/to/pretrained/vit_256/weights.pth",
        pretrain_4096: str = "path/to/pretrained/vit_4096/weights.pth",
        embed_dim_256: int = 384,
        embed_dim_region: int = 192,
        last_output_layer: int = 4,

    ):

        super(GlobalFeatureExtractor, self).__init__()
        checkpoint_key = "teacher"

        self.last_output_layer = last_output_layer

        self.device_256 = torch.device("cuda:0")
        self.device_4096 = torch.device("cuda:1")

        self.vit_256 = vit_small(
            img_size=224,
            patch_size=16,
            embed_dim=embed_dim_256,
        )

        if Path(pretrain_256).is_file():
            print("Loading pretrained weights for ViT_256 model...")
            state_dict = torch.load(pretrain_256, map_location="cpu")
            if checkpoint_key is not None and checkpoint_key in state_dict:
                print(f"Take key {checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            state_dict, msg = update_state_dict(self.vit_256.state_dict(), state_dict)
            self.vit_256.load_state_dict(state_dict, strict=False)
            print(f"Pretrained weights found at {pretrain_256}")
            print(msg)

        else:
            print(f"{pretrain_256} doesnt exist ; please provide path to existing file")

        print("Freezing pretrained ViT_256 model")
        for param in self.vit_256.parameters():
            param.requires_grad = False
        print("Done")

        self.vit_256.to(self.device_256)

        self.vit_4096 = vit4k_xs(
            img_size=3584,
            patch_size=256,
            input_embed_dim=embed_dim_256,
            output_embed_dim=embed_dim_region,
        )

        if Path(pretrain_4096).is_file():
            print("Loading pretrained weights for ViT_4096 model...")
            state_dict = torch.load(pretrain_4096, map_location="cpu")
            if checkpoint_key is not None and checkpoint_key in state_dict:
                print(f"Take key {checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            state_dict, msg = update_state_dict(self.vit_4096.state_dict(), state_dict)
            self.vit_4096.load_state_dict(state_dict, strict=False)
            print(f"Pretrained weights found at {pretrain_4096}")
            print(msg)

        else:
            print(
                f"{pretrain_4096} doesnt exist ; please provide path to existing file"
            )

        print("Freezing pretrained ViT_4096 model")
        for param in self.vit_4096.parameters():
            param.requires_grad = False
        print("Done")

        self.vit_4096.to(self.device_4096)

    def forward(self, x):

        # x = [1, 3, 4096, 4096]

        # TODO: add prepare_img_tensor method
        x = x.unfold(2, 256, 256).unfold(3, 256, 256)  # [1, 3, 16, 16, 256, 256]
        x = rearrange(x, "b c p1 p2 w h -> (b p1 p2) c w h")  # [1*16*16, 3, 256, 256]
        x = x.to(self.device_256, non_blocking=True)  # [256, 3, 256, 256]

        features_256 = self.vit_256(x)  # [256, 384]
        features_256 = features_256.unsqueeze(0)  # [1, 256, 384]
        features_256 = features_256.unfold(1, 16, 16).transpose(
            1, 2
        )  # [1, 384, 16, 16]
        features_256 = features_256.to(self.device_4096, non_blocking=True)

        # intermediate_output = self.vit_4096.get_intermediate_layers(features_256, self.last_output_layer)
        # output = [x[:, 0] for x in intermediate_output]
        # output.append(torch.mean(intermediate_output[-1][:, 1:], dim=1))
        # output = torch.cat(output, dim=-1).cpu() # [1, 192 * 4 + 192 * 1]

        feature_4096 = self.vit_4096(features_256).cpu()  # [1, 192]
        return feature_4096


class LocalFeatureExtractor(nn.Module):
    def __init__(
        self,
        pretrain_256: str = "path/to/pretrained/vit_256/weights.pth",
        embed_dim_256: int = 384,
    ):

        super(LocalFeatureExtractor, self).__init__()
        checkpoint_key = "teacher"

        self.device_256 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vit_256 = vit_small(
            img_size=224,
            patch_size=16,
            embed_dim=embed_dim_256,
        )

        if Path(pretrain_256).is_file():
            print("Loading pretrained weights for ViT_256 model...")
            state_dict = torch.load(pretrain_256, map_location="cpu")
            if checkpoint_key is not None and checkpoint_key in state_dict:
                print(f"Take key {checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            state_dict, msg = update_state_dict(self.vit_256.state_dict(), state_dict)
            self.vit_256.load_state_dict(state_dict, strict=False)
            print(f"Pretrained weights found at {pretrain_256}")
            print(msg)

        else:
            print(f"{pretrain_256} doesnt exist ; please provide path to existing file")

        print("Freezing pretrained ViT_256 model")
        for param in self.vit_256.parameters():
            param.requires_grad = False
        print("Done")

        self.vit_256.to(self.device_256)

    def forward(self, x):

        # x = [1, 3, 4096, 4096]
        # TODO: add prepare_img_tensor method
        x = x.unfold(2, 256, 256).unfold(
            3, 256, 256
        )  # [1, 3, 16, 4096, 256] -> [1, 3, 16, 16, 256, 256]
        x = rearrange(x, "b c p1 p2 w h -> (b p1 p2) c w h")  # [256, 3, 256, 256]
        x = x.to(self.device_256, non_blocking=True)

        feature_256 = self.vit_256(x).detach().cpu()  # [256, 384]

        return feature_256
