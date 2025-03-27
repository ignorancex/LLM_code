import einops
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from pytorch_widedeep.models.tabular.embeddings_layers import \
    SameSizeCatAndContEmbeddings
from timm.models.layers import to_2tuple, trunc_normal_
from timm.models.vision_transformer import Block

from .modal_utils import TrajectoryEnc, MVS
from .utils import Encoder, Group, TransformerEncoder, get_3d_sincos_pos_embed


# PatchEmbed code from EVA: https://github.com/baaivision/EVA
class PatchEmbed2D(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
class Visual2DTokenizer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stop_grad_conv1=False, 
                 use_abs_pos_emb=True, drop_rate=0.):
        super().__init__()
        self.stop_grad_conv1 = stop_grad_conv1
        self.embed_dim = embed_dim
        
        self.patch_embed = PatchEmbed2D(img_size=img_size, 
                                        patch_size=patch_size, 
                                        in_chans=in_chans, 
                                        embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
    
    def forward(self, x):
        x = self.patch_embed(x)

        if self.stop_grad_conv1:
            x = x.detach()
        
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1) 
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        
        return x
    
    def get_embed_dim(self):
        return self.embed_dim
    

class TabTokenizer(nn.Module):
    
    def __init__(self, vocab, ncols, embed_dim=768):
        super().__init__()
        self.vocab_size = len(vocab)
        self.field_hidden_size = embed_dim
        self.nhead = 8 
        self.num_layers = 1
        self.word_embeddings = nn.Embedding(self.vocab_size, self.field_hidden_size,
                                            padding_idx=0, sparse=False)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.field_hidden_size, nhead=self.nhead,
                                                   dim_feedforward=self.field_hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.lin_proj = nn.Linear(self.field_hidden_size * ncols, self.field_hidden_size)
        
    def forward(self, input_ids):
        inputs_embeds = self.word_embeddings(input_ids)
        embeds_shape = list(inputs_embeds.size())

        inputs_embeds = inputs_embeds.view([-1] + embeds_shape[-2:])
        inputs_embeds = inputs_embeds.permute(1, 0, 2)
        inputs_embeds = self.transformer_encoder(inputs_embeds)
        inputs_embeds = inputs_embeds.permute(1, 0, 2)
        inputs_embeds = inputs_embeds.contiguous().view(embeds_shape[0:2]+[-1])

        inputs_embeds = self.lin_proj(inputs_embeds)

        return inputs_embeds
    

class HSItokenizer(nn.Module):
    def __init__(self, image_size, near_band, num_patches, dim, emb_dropout=0.1):
        super().__init__()
        
        patch_dim = image_size ** 2 * near_band
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
    
    def forward(self, x):
        x = self.patch_to_embedding(x) 
        b, n, _ = x.shape
        # add position embedding
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b) #[b,1,dim]
        x = torch.cat((cls_tokens, x), dim = 1) #[b,n+1,dim]
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        return x


class PointTokenizer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.group_size = cfg['group_size']
        self.num_group = cfg['num_group']
        self.encoder_dims = cfg['encoder_dims']
        self.trans_dim = cfg['trans_dim']
        self.depth = cfg['depth'] 
        self.drop_path_rate = cfg['drop_path_rate'] 
        self.num_heads = cfg['num_heads'] 
        
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        self.encoder = Encoder(encoder_channel = self.encoder_dims)
        self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        # self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )  
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads
        )
        
    def forward(self, pts):
        # divide the point cloud in the same form. This is important
        neighborhood, center = self.group_divider(pts)
        # encoder the input cloud blocks
        group_input_tokens = self.encoder(neighborhood)  #  B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)
        # # prepare cls
        # cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)  
        # cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)  
        # add pos embedding
        pos = self.pos_embed(center)
        # final input
        # x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        # pos = torch.cat((cls_pos, pos), dim=1)
        x = self.blocks(group_input_tokens, pos)
        return x
    

class TrajTokenizer(nn.Module):
    def __init__(self, obs_len, pred_len, embed_size):
        super().__init__()
        in_size = 2
        
        self.embedding = nn.Linear(in_size*(obs_len + pred_len), embed_size)
        
    def forward(self, ped_obs, motion_modes):
        # ped_obs [B obs_len 2]
        # motion_modes [K pred_len 2]

        ped_obs = ped_obs.unsqueeze(1).repeat(1, motion_modes.shape[0], 1, 1)  # [B K obs_len 2]
        motion_modes = motion_modes.unsqueeze(0).repeat(ped_obs.shape[0], 1, 1, 1)

        ped_seq = torch.cat((ped_obs, motion_modes), dim=-2)  # [B K seq_len 2] seq_len = obs_len + pred_len
        ped_seq = ped_seq.reshape(ped_seq.shape[0], ped_seq.shape[1], -1)  # [B K seq_len*2]
        ped_embedding = self.embedding(ped_seq) # [B K embed_size]
        
        return ped_embedding
    

class GraphTokenizer(nn.Module):
    def __init__(self, tod_embedding_dim=128, dow_embedding_dim=128, input_dim=3, input_embedding_dim=128,
                 steps_per_day=288, spatial_embedding_dim=0, num_nodes=207, adaptive_embedding_dim=256,
                 in_steps=12):
        super().__init__()
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.input_dim = input_dim
        self.input_embedding_dim = input_embedding_dim
        self.steps_per_day = steps_per_day
        self.spatial_embedding_dim = spatial_embedding_dim
        self.num_nodes = num_nodes
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.in_steps = in_steps
        
        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )
         
    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        batch_size = x.shape[0]

        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
        x = x[..., : self.input_dim]

        x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
        features = [x]
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)
        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(
                batch_size, self.in_steps, *self.node_emb.shape
            )
            features.append(spatial_emb)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)
        x = torch.cat(features, dim=-1)
        
        return rearrange(x, "B I N D -> B (I N) D")
    

class SARTokenizer(nn.Module):

    def __init__(self, inchannels=2, outchannels=1408):
        super().__init__()

        self._layer = nn.Sequential(
            # Block1
            nn.Conv2d(in_channels=inchannels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.GELU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            
            # Block2
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.GELU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            
            # Block3
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.GELU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            
            # out
            nn.Conv2d(in_channels=512, out_channels=outchannels, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x = self._layer(x)
        x = einops.rearrange(x, "B D h w -> B (h w) D")
        return x
    