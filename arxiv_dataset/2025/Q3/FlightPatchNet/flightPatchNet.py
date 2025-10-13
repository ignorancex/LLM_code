import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from einops.layers.torch import Rearrange
from layers.Embed import DataEmbedding_inverted
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer


def get_activation(activ):
    if activ == "relu":
        return nn.ReLU()
    elif activ == "gelu":
        return nn.GELU()
    elif activ == "leaky_relu":
        return nn.LeakyReLU()
    elif activ == "none":
        return nn.Identity()
    else:
        raise ValueError(f"activation:{activ}")

class FeedForward(nn.Module):

    def __init__(
        self,
        in_features: int,
        hid_features: int,
        activ="gelu",
        drop: float = 0.0
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features, hid_features),
            get_activation(activ),
            nn.Linear(hid_features, in_features),
            nn.Dropout(drop))
    def forward(self, x):
        x = self.net(x)
        return x

class MLPBlock(nn.Module):

    def __init__(
        self,
        dim,
        in_features: int,
        hid_features: int,
        out_features: int,
        activ="gelu",
        drop: float = 0.0,
        jump_conn='trunc',
    ):
        super().__init__()
        self.dim = dim
        self.out_features = out_features
        self.net = nn.Sequential(
            nn.Linear(in_features, hid_features),
            get_activation(activ),
            nn.Linear(hid_features, out_features),
            nn.Dropout(drop))
        if jump_conn == "trunc":
            self.jump_net = nn.Identity()
        elif jump_conn == 'proj':
            self.jump_net = nn.Linear(in_features, out_features)
        else:
            raise ValueError(f"jump_conn:{jump_conn}")

    def forward(self, x):
        x = torch.transpose(x, self.dim, -1)
        # print(x.shape)
        x = self.jump_net(x)[..., :self.out_features] + self.net(x)
        x = torch.transpose(x, self.dim, -1)
        return x

class PatchEncoder(nn.Module):

    def __init__(
            self,
            in_len: int,
            hid_len: int,
            patch_size: int,
            hid_pch: int,
            in_chn: int,
            norm=None,
            activ="gelu",
            drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential()
        patch_num = in_len // patch_size
        out_chn = in_chn     
        inter_patch_mlp = MLPBlock(2, patch_num, hid_len, patch_num, activ, drop)
        if norm == 'bn':
            norm_class = nn.BatchNorm2d
        elif norm == 'in':
            norm_class = nn.InstanceNorm2d
        else:
            norm_class = nn.Identity
        linear = nn.Linear(patch_size, 1)
        intra_patch_mlp = MLPBlock(3, patch_size, hid_pch, patch_size, activ, drop)
        self.net.append(Rearrange("b c (l1 l2) -> b c l1 l2", l2=patch_size))
        self.net.append(norm_class(in_chn))
        self.net.append(inter_patch_mlp)
        self.net.append(norm_class(out_chn))
        self.net.append(intra_patch_mlp)
        self.net.append(linear)
        self.net.append(Rearrange("b c l1 1 -> b c l1"))

    def forward(self, x):
        # b,c,l
        return self.net(x)

class PatchDecoder(nn.Module):

    def __init__(
            self,
            in_len: int,
            hid_len: int,
            in_chn: int,
            patch_size: int,
            hid_pch: int,
            norm=None,
            activ="gelu",
            drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential()
        patch_num = in_len // patch_size
        inter_patch_mlp = MLPBlock(2, patch_num, hid_len, patch_num, activ,
                                   drop)
        if norm == 'bn':
            norm_class = nn.BatchNorm2d
        elif norm == 'in':
            norm_class = nn.InstanceNorm2d
        else:
            norm_class = nn.Identity
        linear = nn.Linear(1, patch_size)
        intra_patch_mlp = MLPBlock(3, patch_size, hid_pch, patch_size, activ, drop)
        self.net.append(Rearrange("b c l1 -> b c l1 1"))
        self.net.append(linear)
        self.net.append(norm_class(in_chn))
        self.net.append(intra_patch_mlp)
        self.net.append(norm_class(in_chn))
        self.net.append(inter_patch_mlp)
        self.net.append(norm_class(in_chn))
        self.net.append(Rearrange("b c l1 l2 -> b c (l1 l2)"))

    def forward(self, x):
        # b,c,l
        return self.net(x)

class PredictionHead(nn.Module):
    def __init__(self,
                 in_len,
                 out_len,
                 hid_len,
                 in_chn,
                 out_chn,
                 hid_chn,
                 activ,
                 drop=0.0) -> None:
        super().__init__()
        self.net = nn.Sequential()
        if in_chn != out_chn:
            c_jump_conn = "proj"
        else:
            c_jump_conn = "trunc"
        self.net.append(
            MLPBlock(1,
                     in_chn,
                     hid_chn,
                     out_chn,
                     activ=activ,
                     drop=drop,
                     jump_conn=c_jump_conn))
        self.net.append(
            MLPBlock(2,
                     in_len,
                     hid_len,
                     out_len,
                     activ=activ,
                     drop=drop,
                     jump_conn='proj'))

    def forward(self, x):
        return self.net(x)
    
class FlightPatchNet(nn.Module):
    '''
    FlightPatchNet
    '''

    def __init__(self,
                 in_len,
                 out_len,
                 in_chn,
                 out_chn,
                 patch_sizes,
                 hid_len,
                 hid_pch,
                 hid_pred,
                 e_layers,
                 d_model,
                 norm,
                 last_norm,
                 activ,
                 drop,
                 reduction="sum") -> None:
        super().__init__()
        self.in_len = in_len
        self.out_len = out_len
        self.in_chn = in_chn
        self.out_chn = out_chn
        self.last_norm = last_norm
        self.reduction = reduction
        self.patch_encoders = nn.ModuleList()
        self.patch_decoders = nn.ModuleList()
        self.pred_heads = nn.ModuleList()
        self.patch_sizes = patch_sizes
        self.e_layers = e_layers
        self.d_model = d_model
        self.paddings = []
        self.pred_fusion = nn.Linear(len(patch_sizes), 1)
        self.scale_fusion_dim = in_chn * in_len
        self.channel_fusion_dim = in_len * len(patch_sizes)
        self.hid_chn = 16

        for i, patch_size in enumerate(patch_sizes):

            res = in_len % patch_size
            padding = (patch_size - res) % patch_size
            self.paddings.append(padding)
            padded_len = in_len + padding

            self.patch_encoders.append(
                PatchEncoder(padded_len, hid_len, patch_size, hid_pch, in_chn, norm, activ, drop)
            )

            self.patch_decoders.append(
                PatchDecoder(padded_len, hid_len, in_chn, patch_size, hid_pch, norm, activ, drop))

            if out_len != 0 and out_chn != 0:

                self.pred_heads.append(
                    PredictionHead(in_len, out_len, hid_pred,
                                   in_chn, out_chn, self.hid_chn, activ, drop))
            else:
                self.pred_heads.append(nn.Identity())

        self.time_embedding = DataEmbedding_inverted(self.in_chn, self.d_model)
        # temporal Encoder
        self.global_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor=3, attention_dropout=0.1), d_model, n_heads=8),
                    d_model,
                    d_ff=512,
                    dropout=drop,
                    activation=activ
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.channel_projection = nn.Linear(d_model, self.in_chn, bias=True)
          
        self.scale_fusion = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor=3, attention_dropout=0.1), self.scale_fusion_dim, n_heads=4),
                    self.scale_fusion_dim,
                    d_ff=512,
                    dropout=drop,
                    activation=activ
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.scale_fusion_dim)
        )
        self.channel_fusion = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor=3, attention_dropout=0.1), self.channel_fusion_dim, n_heads=4),
                    self.channel_fusion_dim,
                    d_ff=512,
                    dropout=drop,
                    activation=activ
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.channel_fusion_dim)
        )


    def forward(self, x):
        # B,C,L
        if self.last_norm:
            x_last = x[:, :, [-1]].detach()
            x = x - x_last
        preds = []
        channel_enc_out = self.time_embedding(x.permute(0, 2, 1))
        # [B L d_model]
        channel_enc_out, attns = self.global_encoder(channel_enc_out, attn_mask=None)
        # [B L d_model]
        x = self.channel_projection(channel_enc_out).permute(0, 2, 1)
        # [B C L]

        last_comp = torch.zeros_like(x)
        decoders = []
        # patch mixer blocks
        for i in range(len(self.patch_sizes)):
            x_in = x
            x_in = F.pad(x_in, (self.paddings[i], 0), "constant", 0)

            emb = self.patch_encoders[i](x_in)
            comp = self.patch_decoders[i](emb)[:, :, self.paddings[i]:]
            decoders.append(comp)
            x = x + comp

        multi_scale_dec = torch.stack(decoders, dim=0)
        K,B,C,L = multi_scale_dec.shape
       
       
        scale_in = multi_scale_dec.permute(1,0,2,3).reshape(B,K,C*L)   
       
        scale_fusion_out, scale_attn = self.scale_fusion(scale_in)
        scale_fusion_out = scale_fusion_out.reshape(B,K,C,L)
    

        # h,b,c*l
        channel_fusion_in = scale_fusion_out.permute(0,2,1,3).reshape(B,C,K*L)
       
        channel_fusion_out, chn_attn = self.channel_fusion(channel_fusion_in)
    

        channel_fusion_out =  channel_fusion_out.reshape(B,C,K,L).permute(2,0,1,3)
        
        # multiple predictors
        for i in range(len(self.patch_sizes)):
            last_comp = channel_fusion_out[i, ...].squeeze(0)
            pred = self.pred_heads[i](last_comp)

            preds.append(pred)
        if self.out_len != 0 and self.out_chn != 0:

            y_pred = reduce(preds, "k b c l -> b c l",self.reduction)
            if self.last_norm and self.out_chn == self.in_chn:
                y_pred += x_last
            return y_pred
        else:
            return None

