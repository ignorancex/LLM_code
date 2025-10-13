import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import numpy as np
import torch.nn as nn
from monai.networks.blocks import TransformerBlock
from monai.networks.layers import trunc_normal_
from torch.distributions import Dirichlet

    
class BMMAE(nn.Module):
    def __init__(
        self,
        modalities: Tuple[str, ...],
        tokenizers: Dict[str, nn.Module],
        decoder: nn.Module,
        masking_ratio: float = 0.75,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 1536,
        dropout_rate: float = 0.0,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        self.modalities = modalities
        self.tokenizers = nn.ModuleDict(tokenizers)
        self.decoder = decoder
        self.hidden_size = hidden_size
        self.norm = nn.LayerNorm(hidden_size)
        self.masking_ratio = masking_ratio
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    mlp_dim=mlp_dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    qkv_bias=qkv_bias,
                    save_attn=False,
                )
                for i in range(num_layers)
            ]
        )
        
    @classmethod
    def from_pretrained(model_filename: str = 'bmmae.pth') -> BMMAE:
        from huggingface_hub import hf_hub_download
        # Download weights from Hub
        filepath = hf_hub_download(
           repo_id="luklebigbosse/BM-MAE",
           filename=model_filename,
        )
   
        # Equip model with weights
        state_dict = torch.load(state_dict, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    def generate_random_masks(
        self,
        input_tokens: Dict[str, torch.Tensor],
        num_encoded_tokens: int,
        alphas: Union[float, List[float]] = 1.0,
    ):
        """
        adapted from https://github.com/EPFL-VILAB/MultiMAE/blob/main/multimae/multimae.py
        To handle the case of mask_ratio < 0.5 we use a rejection sampling method
        """
        bs = list(input_tokens.values())[0].shape[0]
        device = list(input_tokens.values())[0].device
        max_prop = list(input_tokens.values())[0].shape[1] / num_encoded_tokens
        
        alphas = [alphas] * len(input_tokens) 
        modality_sampling_dist = Dirichlet(torch.Tensor(alphas)).sample((bs,)).to(device)
        while True:
            invalid_idx = torch.argwhere(modality_sampling_dist >= max_prop)[:, 0]
        
            if not invalid_idx.any():
                break
            resample = Dirichlet(torch.Tensor(alphas)).sample((len(invalid_idx),)).to(device)
            modality_sampling_dist[invalid_idx] = resample
        samples_per_modality = (modality_sampling_dist *  num_encoded_tokens).round().long()
        modality_masks = []
        token_per_modality = [modality_tokens.shape[1] for modality_tokens in input_tokens.values()]
        for i, num_tokens in enumerate(token_per_modality):
            # Use noise to shuffle arange as in MAE
            noise = torch.rand(bs, num_tokens, device=device)  # noise in [0, 1]
            ids_arange_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            mask = torch.arange(num_tokens, device=device).unsqueeze(0).expand(bs, -1)
            mask = torch.gather(mask, dim=1, index=ids_arange_shuffle)
            mask = torch.where(mask < samples_per_modality[:, i].unsqueeze(1), 0, 1)
            modality_masks.append(mask)

        mask_all = torch.cat(modality_masks, dim=1)
        ids_shuffle = torch.argsort(mask_all + torch.rand_like(mask_all.float()), dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :num_encoded_tokens]
        
        mask_all = torch.ones_like(mask_all)
        mask_all[:, :num_encoded_tokens] = 0
        mask_all = torch.gather(mask_all, dim=1, index=ids_restore)
        modality_masks = torch.split(mask_all, token_per_modality, dim=1)
       
        modality_masks = {domain: mask for domain, mask in zip(input_tokens.keys(), modality_masks)}

        return modality_masks, ids_keep, ids_restore

    def forward(
        self,
        x: Union[Dict[str, torch.Tensor], torch.Tensor],
        modality_masks: Optional[Dict[str, torch.Tensor]] = None,
    ):
        input_tokens = {
            modality: self.tokenizers[modality](tensor) for modality, tensor in x.items() if modality in self.modalities
        }
   
        n_tokens = sum([modality_tokens.shape[1] for modality_tokens in input_tokens.values()])
        n_to_keep = int(n_tokens * (1 - self.masking_ratio))
        
        if modality_masks is None:
            modality_masks, ids_keep, ids_restore = self.generate_random_masks(
                input_tokens,
                n_to_keep,
                alphas=1.0,
            )
        else:
            mask_all = torch.cat([modality_masks[modality] for modality in input_tokens.keys()], dim=1)
            ids_shuffle = torch.argsort(mask_all, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            # mask must have the same shape
            ids_keep = ids_shuffle[:, : int((mask_all == 0).sum() / mask_all.shape[0])]
       
        input_tokens = torch.cat([modality_tokens for modality_tokens in input_tokens.values()], dim=1)
        
        input_tokens = torch.gather(
            input_tokens, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, input_tokens.shape[2])
        )
        cls_tokens = self.cls_token.expand(input_tokens.shape[0], -1, -1)
        input_tokens = torch.cat([cls_tokens, input_tokens], dim=1)
  
        
        for blk in self.blocks:
            input_tokens = blk(input_tokens)
        encoder_tokens = self.norm(input_tokens)
        outputs = self.decoder(encoder_tokens, ids_restore)

        return outputs, modality_masks


class BMMAEViT(nn.Module):
    def __init__(
        self,
        modalities: Tuple[str, ...],
        tokenizers: Dict[str, nn.Module],
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 1536,
        dropout_rate: float = 0.0,
        qkv_bias: bool = True,
        classification: bool = True,
        n_outputs: Optional[int] = None
    ) -> None:
        super().__init__()
        self.modalities = modalities
        self.tokenizers = nn.ModuleDict(tokenizers)
        self.hidden_size = hidden_size
        self.norm = nn.LayerNorm(hidden_size)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    mlp_dim=mlp_dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    qkv_bias=qkv_bias,
                    save_attn=False,
                )
                for i in range(num_layers)
            ]
        )

        if classification:
            if n_outputs is None:
                raise ValueError("if classification mode, provide a not None n_outputs")
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            self.output_layer = nn.Sequential(nn.Linear(hidden_size, 256), nn.ReLU(), nn.Linear(256, n_outputs))
            trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: Union[Dict[str, torch.Tensor], torch.Tensor]):
        # if x is not a dict convert it to a dict
        if not isinstance(x, dict):
            x = {m: x[:, i, :, :, :].unsqueeze(1) for i, m in enumerate(self.modalities)}

        input_tokens = {
            modality: self.tokenizers[modality](tensor) for modality, tensor in x.items() if modality in self.modalities
        }

        input_tokens = torch.cat([modality_tokens for modality_tokens in input_tokens.values()], dim=1)
        if hasattr(self, "cls_token"):
            cls_tokens = self.cls_token.expand(input_tokens.shape[0], -1, -1)
            input_tokens = torch.cat([cls_tokens, input_tokens], dim=1)

        if not hasattr(self, "cls_token"):
            hidden_states = []

        ## Transformer forward pass
        for blk in self.blocks:
            input_tokens = blk(input_tokens)
            if not hasattr(self, "cls_token"):
                if input_tokens.size(1) != 512:
                    hidden_states.append(
                        torch.stack(
                                [
                                    input_tokens[:, (i * 512) : ((i + 1) * 512), :]
                                    for i in range(input_tokens.size(1) // 512)
                                ],
                                dim=1,
                            ).mean(1)
                        )
                else:
                    hidden_states.append(input_tokens)

        input_tokens = self.norm(input_tokens)
        if hasattr(self, "cls_token"):
            return self.output_layer(input_tokens[:, 0])
        else:
            if input_tokens.size(1) != 512:
                return torch.stack(
                        [input_tokens[:, i * 512 : (i + 1) * 512, :] for i in range(input_tokens.size(1) // 512)],
                        dim=1,
                    ).mean(1), hidden_states
            else:
                return input_tokens, hidden_states
            

class ViTEncoder(nn.Module):
    def __init__(
        self,
        modalities: Tuple[str, ...],
        tokenizers: Dict[str, nn.Module],
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 1536,
        dropout_rate: float = 0.0,
        qkv_bias: bool = True,
        cls_token: bool = True,
    ) -> None:
        super().__init__()
        self.modalities = modalities
        self.tokenizers = nn.ModuleDict(tokenizers)
        self.hidden_size = hidden_size
        self.norm = nn.LayerNorm(hidden_size)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    mlp_dim=mlp_dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    qkv_bias=qkv_bias,
                    save_attn=False,
                )
                for i in range(num_layers)
            ]
        )

        if cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            trunc_normal_(self.cls_token, std=0.02)

    @classmethod
    def from_pretrained(model_filename: str = 'bmmae.pth') -> ViTEncoder:
        from huggingface_hub import hf_hub_download
        # Download weights from Hub
        filepath = hf_hub_download(
           repo_id="luklebigbosse/BM-MAE",
           filename=model_filename,
        )
       
        # Equip model with weights
        state_dict = torch.load(state_dict, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        
    def forward(self, x: Union[Dict[str, torch.Tensor], torch.Tensor]):
        # if x is not a dict convert it to a dict
        if not isinstance(x, dict):
            x = {m: x[:, i, :, :, :].unsqueeze(1) for i, m in enumerate(self.modalities)}

        input_tokens = {
            modality: self.tokenizers[modality](tensor) for modality, tensor in x.items() if modality in self.modalities
        }

        input_tokens = torch.cat([modality_tokens for modality_tokens in input_tokens.values()], dim=1)
        if hasattr(self, "cls_token"):
            cls_tokens = self.cls_token.expand(input_tokens.shape[0], -1, -1)
            input_tokens = torch.cat([cls_tokens, input_tokens], dim=1)


        ## Transformer forward pass
        for blk in self.blocks:
            input_tokens = blk(input_tokens)

        input_tokens = self.norm(input_tokens)
        return input_tokens
