import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from timm.layers import PatchEmbed, Mlp, DropPath, to_2tuple, to_ntuple, trunc_normal_, _assert
from .FusionAgg import AggregatorLayer
class FusionUP(nn.Module):
    """"Upscaling using feat from dino and clip"""
    def __init__(self, in_channels, out_channels, clip_guidance_channels, dino_guidance_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels - clip_guidance_channels, kernel_size=2, stride=2)
        # if self.decoder_clip_guidance_dims[0] != 0 and self.decoder_clip_guidance_dims 
        self.conv = DoubleConv(in_channels+dino_guidance_channels, out_channels)

    def forward(self, x, clip_guidance,dino_guidance):
        x = self.up(x)
        if clip_guidance is not None:
            T = x.size(0) // clip_guidance.size(0)
            clip_guidance = repeat(clip_guidance, "B C H W -> (B T) C H W", T=T)
            # dino_guidance = repeat(dino_guidance, "B C H W -> (B T) C H W", T=T)
            x = torch.cat([x, clip_guidance], dim=1)
        if dino_guidance is not None: 
            T = x.size(0) // dino_guidance.size(0)
            # clip_guidance = repeat(clip_guidance, "B C H W -> (B T) C H W", T=T)
            dino_guidance = repeat(dino_guidance, "B C H W -> (B T) C H W", T=T)
            x = torch.cat([x,dino_guidance], dim=1)
        return self.conv(x)
class DoubleConv(nn.Module):
    """(convolution => [GN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(mid_channels // 16, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(mid_channels // 16, mid_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
class RIPD(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        decoder_dims = (64, 32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        decoder_clip_guidance_dims=(256, 128),
        decoder_clip_guidance_proj_dims=(32, 16),
        decoder_dino_guidance_dims=(256, 128),
        decoder_dino_guidance_proj_dims=(32, 16),
        feat_dim = 512,
        use_clip_corr = True,
        use_dino_corr = True,
        fusion_type = "query_guided",
        num_layers=4,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        prompt_channel=1,
        pad_len=256,
    ) -> None:

        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.fusion_type = fusion_type
        self.layers = nn.ModuleList([
            AggregatorLayer(
                hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim, 
                nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
            ) for _ in range(num_layers)
        ])

        # self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3)
        if fusion_type=='simple_concatenation':
            self.simple_concatenation_corr_embed = nn.Conv2d(3*feat_dim, hidden_dim, kernel_size=7, stride=1, padding=3)
        else:
            self.conv1 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3) 
            self.conv2 = nn.Conv2d(prompt_channel, hidden_dim, kernel_size=7, stride=1, padding=3) if (use_clip_corr and use_dino_corr) == True and self.fusion_type != 'fusion_query' else None
            self.fusion_corr = nn.Conv2d(2*hidden_dim, hidden_dim, kernel_size=7, stride=1, padding=3) if (use_clip_corr and use_dino_corr) == True and self.fusion_type != 'fusion_query'else None
            self.fusion_feats = nn.Conv2d(2*feat_dim, feat_dim,kernel_size=1, stride=1, padding=0) if fusion_type=='fusion_query'else None
        
        self.guidance_projection = nn.Sequential(
            nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None
        
        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None

        self.CLIP_decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_clip_guidance_proj_dims)
        ]) if decoder_clip_guidance_dims[0] > 0 else None
        
        self.DINO_decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_dino_guidance_proj_dims)
        ]) if decoder_dino_guidance_dims[0] > 0 else None
        
        
        self.Fusiondecoder1=FusionUP(hidden_dim, decoder_dims[0], decoder_clip_guidance_proj_dims[0],decoder_dino_guidance_proj_dims[0])
        self.Fusiondecoder2=FusionUP(decoder_dims[0], decoder_dims[1], decoder_clip_guidance_proj_dims[1], decoder_dino_guidance_proj_dims[1])
        self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)

        self.pad_len = pad_len

    def feature_map(self, img_feats, text_feats):
        # concatenated feature volume for feature aggregation baselines
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        img_feats = repeat(img_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = text_feats.mean(dim=-2) # average text features over different prompts
        text_feats = F.normalize(text_feats, dim=-1) # B T C
        text_feats = repeat(text_feats, "B T C -> B C T H W", H=img_feats.shape[-2], W=img_feats.shape[-1])
        return torch.cat((img_feats, text_feats), dim=1) # B 2C T H W
    
    def simple_concatenation_corr(self,clip_feats, dino_feats, text_feats):
        clip_feats = F.normalize(clip_feats, dim=1) # B C H W
        clip_feats = repeat(clip_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        dino_feats = F.normalize(dino_feats, dim=1) # B C H W
        dino_feats = repeat(dino_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = text_feats.mean(dim=-2) # average text features over different prompts
        text_feats = repeat(text_feats, "B T C -> B C T H W", H=clip_feats.shape[-2], W=clip_feats.shape[-1])
        cat_feats = torch.cat((clip_feats,dino_feats, text_feats), dim=1) # B 3C T H W
        return cat_feats 
    def correlation(self, img_feats, text_feats):
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bchw, btpc -> bpthw', img_feats, text_feats)
        return corr

    def corr_embed(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        corr_embed = self.conv1(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed
    
    def corr_fusion_embed_minimum(self,clip_corr,dino_corr):
        # this one does not import a 1*1 conv
        # instead we modify the original embedding layer to adapt to the concatenated corr volume.
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
        fused_corr = torch.cat([clip_corr,dino_corr],dim = 1)
        fused_corr = self.conv1_modified(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr

    def corr_fusion_embed_seperate(self,clip_corr,dino_corr):
        # this one does not import a 1*1 conv
        # instead we modify the original embedding layer to adapt to the concatenated corr volume.
        B = clip_corr.shape[0]
        self.sigmoid = nn.Sigmoid()
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
 
        clip_embed_corr = self.conv1(clip_corr)
        dino_embed_corr = self.conv2(dino_corr)
        clip_embed_corr = self.sigmoid(clip_embed_corr)
        dino_embed_corr = self.sigmoid(dino_embed_corr)
        fused_corr = torch.cat([clip_embed_corr,dino_embed_corr],dim = 1)
        fused_corr = self.fusion_corr(fused_corr)
        fused_corr = self.sigmoid(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        clip_embed_corr = rearrange(clip_embed_corr, '(B T) C H W -> B C T H W', B=B)
        dino_embed_corr = rearrange(dino_embed_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr, clip_embed_corr, dino_embed_corr
    
        
    def corr_fusion_embed(self,clip_corr,dino_corr):
        B = clip_corr.shape[0]
        clip_corr = rearrange(clip_corr, 'B P T H W -> (B T) P H W')
        dino_corr = rearrange(dino_corr, 'B P T H W -> (B T) P H W')
        fused_corr = torch.cat([clip_corr,dino_corr],dim = 1)
        fused_corr = self.fusion_corr(fused_corr)
        fused_corr = self.conv1(fused_corr)
        fused_corr = rearrange(fused_corr, '(B T) C H W -> B C T H W', B=B)
        return fused_corr
        # exit()
        
    def corr_projection(self, x, proj):
        corr_embed = rearrange(x, 'B C T H W -> B T H W C')
        corr_embed = proj(corr_embed)
        corr_embed = rearrange(corr_embed, 'B T H W C -> B C T H W')
        return corr_embed

    def upsample(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = F.interpolate(corr_embed, scale_factor=2, mode='bilinear', align_corners=True)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed

    def conv_decoder(self, x, guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, guidance[0])
        corr_embed = self.decoder2(corr_embed, guidance[1])
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    
    def Fusion_conv_decoer(self, x, clip_guidance,dino_guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.Fusiondecoder1(corr_embed, clip_guidance[0],dino_guidance[0])
        corr_embed = self.Fusiondecoder2(corr_embed, clip_guidance[1],dino_guidance[1])
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    def forward(self, img_feats,dino_feat, text_feats, appearance_guidance,dino_guidance):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (B, T, P, C) T是类别的个数
            apperance_guidance: tuple of (B, C, H, W)
        """

        classes = None

        if dino_feat is not None and img_feats is not None:
            if self.fusion_type == 'query_guided':
                corr = self.correlation(img_feats, text_feats)
                dino_corr = self.correlation(dino_feat,text_feats)
                fused_corr_embed,clip_embed_corr, dino_embed_corr  = self.corr_fusion_embed_seperate(clip_corr = corr,dino_corr=dino_corr)
                fused_corr_embed = fused_corr_embed+clip_embed_corr
            elif self.fusion_type == 'simple_concatenation':
                simple_concatenation_corr = self.simple_concatenation_corr(img_feats,dino_feat,text_feats)
                T = simple_concatenation_corr.shape[2]
                
                fused_corr_embed = self.simple_concatenation_corr_embed(rearrange(simple_concatenation_corr,"B C T H W -> (B T) C H W"))
                fused_corr_embed = rearrange(fused_corr_embed, "(B T) C H W -> B C T H W", T = T)
            elif self.fusion_type == 'fusion_query':
                fused_feat = self.fusion_feats(torch.cat([img_feats,dino_feat],dim=1))
                corr = self.correlation(fused_feat,text_feats)
                corr_embed = self.corr_embed(corr)
                fused_corr_embed = corr_embed 
        elif dino_feat is not None and img_feats is None:
            corr = self.correlation(dino_feat,text_feats)
            embed_corr = self.corr_embed(corr)
            fused_corr_embed = embed_corr
        
        elif dino_feat is None and img_feats is not None:
            corr = self.correlation(img_feats,text_feats)
            embed_corr = self.corr_embed(corr)
            fused_corr_embed = embed_corr

        projected_guidance, projected_text_guidance, CLIP_projected_decoder_guidance,DINO_projected_decoder_guidance  = None, None, [None, None], [None,None]
        if self.guidance_projection is not None and appearance_guidance is not None:
            projected_guidance = self.guidance_projection(appearance_guidance[0])
        if self.guidance_projection is not None and appearance_guidance is None:
            projected_guidance = self.guidance_projection(dino_feat)
        if self.CLIP_decoder_guidance_projection is not None:
            CLIP_projected_decoder_guidance = [proj(g) for proj, g in zip(self.CLIP_decoder_guidance_projection, appearance_guidance[1:])]
        if self.DINO_decoder_guidance_projection is not None:
            DINO_projected_decoder_guidance = [proj(g) for proj, g in zip(self.DINO_decoder_guidance_projection, dino_guidance)]
        if self.text_guidance_projection is not None:
            text_feats = text_feats.mean(dim=-2)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            projected_text_guidance = self.text_guidance_projection(text_feats)

        for layer in self.layers:
            fused_corr_embed = layer(fused_corr_embed, projected_guidance, projected_text_guidance)

        logit = self.Fusion_conv_decoer(fused_corr_embed, CLIP_projected_decoder_guidance,DINO_projected_decoder_guidance)

        

        return logit