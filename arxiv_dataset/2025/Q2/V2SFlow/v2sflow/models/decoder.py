import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp
from rotary_embedding_torch import RotaryEmbedding
from transformers import PretrainedConfig, PreTrainedModel

from third_party.opensora.acceleration.checkpoint import auto_grad_checkpoint
from third_party.opensora.models.layers.blocks import (
    TimestepEmbedder,
    approx_gelu,
    get_layernorm,
    t2i_modulate,
)
from third_party.opensora.utils.ckpt_utils import load_checkpoint

from third_party.cosyvoice.transformer.encoder import ConformerEncoder
from third_party.cosyvoice.flow.length_regulator import InterpolateRegulator

from v2sflow.registry import MODELS
from v2sflow.models.layers.blocks import Embedding, Linear, FirstBlock1D, FinalBlock1D, Attention_with_padding

class V2SDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        rope=None,
        qk_norm=False,
        dropout=0.0,
        temporal=False,
        enable_flash_attn=False,
        enable_layernorm_kernel=False,
    ):
        super().__init__()
        self.temporal = temporal
        self.hidden_size = hidden_size
        self.enable_flash_attn = enable_flash_attn

        self.norm1 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.attn = Attention_with_padding(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=qk_norm,
            rope=rope,
            enable_flash_attn=enable_flash_attn,
            attn_drop=dropout,
            proj_drop=dropout,
        )
        self.norm2 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

    def t_mask_select(self, x_mask, x, masked_x):
        # x: [B, T, C]
        # mased_x: [B, T, C]
        # x_mask: [B, T]
        x = torch.where(x_mask[:, :, None], x, masked_x)
        return x

    def forward(
        self,
        x,
        t,
        x_mask=None,  # temporal mask
        t0=None,  # t with timestamp=0
        x_mask_for_padding=None,
    ):
        # prepare modulate parameters
        B, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        if x_mask is not None:
            shift_msa_zero, scale_msa_zero, gate_msa_zero, shift_mlp_zero, scale_mlp_zero, gate_mlp_zero = (
                self.scale_shift_table[None] + t0.reshape(B, 6, -1)
            ).chunk(6, dim=1)

        # modulate (attention)
        x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)
        if x_mask is not None:
            x_m_zero = t2i_modulate(self.norm1(x), shift_msa_zero, scale_msa_zero)
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero)

        # attention
        x_m = self.attn(x_m, x_mask_for_padding)

        # modulate (attention)
        x_m_s = gate_msa * x_m
        if x_mask is not None:
            x_m_s_zero = gate_msa_zero * x_m
            x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero)

        # residual
        x = x + self.drop_path(x_m_s)

        # modulate (MLP)
        x_m = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)
        if x_mask is not None:
            x_m_zero = t2i_modulate(self.norm2(x), shift_mlp_zero, scale_mlp_zero)
            x_m = self.t_mask_select(x_mask, x_m, x_m_zero)

        # MLP
        x_m = self.mlp(x_m)

        # modulate (MLP)
        x_m_s = gate_mlp * x_m
        if x_mask is not None:
            x_m_s_zero = gate_mlp_zero * x_m
            x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero)

        # residual
        x = x + self.drop_path(x_m_s)

        return x

class V2SFlowDecoderConfig(PretrainedConfig):
    model_type = "V2SFlowDecoder"

    def __init__(
        self,
        content_vocab_size=None,
        pitch_vocab_size=None,
        speaker_embed_dim=None,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        pred_sigma=True,
        drop_path=0.0,
        dropout=0.0,
        qk_norm=True,
        enable_flash_attn=False,
        enable_layernorm_kernel=False,
        **kwargs,
    ):
        self.content_vocab_size = content_vocab_size
        self.pitch_vocab_size = pitch_vocab_size
        self.speaker_embed_dim = speaker_embed_dim
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.class_dropout_prob = class_dropout_prob
        self.pred_sigma = pred_sigma
        self.drop_path = drop_path
        self.dropout = dropout
        self.qk_norm = qk_norm
        self.enable_flash_attn = enable_flash_attn
        self.enable_layernorm_kernel = enable_layernorm_kernel
        super().__init__(**kwargs)

class V2SFlowDecoder(PreTrainedModel):
    config_class = V2SFlowDecoderConfig

    def __init__(self, config):
        super().__init__(config)
        self.pred_sigma = config.pred_sigma
        self.in_channels = config.in_channels
        self.out_channels = config.in_channels * 2 if config.pred_sigma else config.in_channels

        # model size related
        self.depth = config.depth
        self.mlp_ratio = config.mlp_ratio
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.dropout = config.dropout

        # computation related
        self.drop_path = config.drop_path
        self.enable_flash_attn = config.enable_flash_attn
        self.enable_layernorm_kernel = config.enable_layernorm_kernel

        self.class_dropout_prob = config.class_dropout_prob

        # input size related
        self.rope = RotaryEmbedding(dim=self.hidden_size // self.num_heads)

        # embedding
        self.t_embedder = TimestepEmbedder(config.hidden_size)
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True),
        )

        # temporal blocks
        drop_path = [x.item() for x in torch.linspace(0, self.drop_path, config.depth)]
        self.blocks = nn.ModuleList(
            [
                V2SDiTBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    drop_path=drop_path[i],
                    qk_norm=config.qk_norm,
                    enable_flash_attn=config.enable_flash_attn,
                    enable_layernorm_kernel=config.enable_layernorm_kernel,
                    rope=self.rope.rotate_queries_or_keys,
                    dropout=config.dropout,
                )
                for i in range(config.depth)
            ]
        )

        self.initialize_weights()

        self.use_content = config.content_vocab_size is not None
        self.use_pitch = config.pitch_vocab_size is not None
        self.use_speaker = config.speaker_embed_dim is not None

        num_feat = 1 + self.use_content + self.use_speaker + self.use_pitch
        self.first_layer = FirstBlock1D(config.in_channels * num_feat, config.hidden_size)
        self.final_layer = FinalBlock1D(config.hidden_size, self.out_channels)

        if self.use_content:
            self.padding_idx = config.content_vocab_size
            self.content_embedding = Embedding(self.padding_idx + 1, config.hidden_size, self.padding_idx)
            self.content_embedding_adapter = ConformerEncoder(
                input_size=config.hidden_size,
                output_size=config.hidden_size,
                attention_heads=4,
                linear_units=1024,
                num_blocks=2,
                dropout_rate=0.1,
                positional_dropout_rate=0.1,
                attention_dropout_rate=0.1,
                normalize_before=True,
                input_layer='linear',
                pos_enc_layer_type='rel_pos_espnet',
                selfattention_layer_type='rel_selfattn',
                use_cnn_module=False,
                macaron_style=False,
            )
            self.content_embedding_proj = torch.nn.Linear(config.hidden_size, config.in_channels)
            self.content_length_regulator = InterpolateRegulator(
                channels=config.in_channels,
                sampling_ratios=[1, 1],
            )

        if self.use_speaker:
            self.speaker_embedding_proj = Linear(config.speaker_embed_dim, config.in_channels)

        if self.use_pitch:
            self.pitch_padding_idx = config.pitch_vocab_size
            self.pitch_embedding = Embedding(self.pitch_padding_idx + 1, config.hidden_size, self.pitch_padding_idx)
            self.pitch_embedding_adapter = ConformerEncoder(
                input_size=config.hidden_size,
                output_size=config.hidden_size,
                attention_heads=4,
                linear_units=1024,
                num_blocks=2,
                dropout_rate=0.1,
                positional_dropout_rate=0.1,
                attention_dropout_rate=0.1,
                normalize_before=True,
                input_layer='linear',
                pos_enc_layer_type='rel_pos_espnet',
                selfattention_layer_type='rel_selfattn',
                use_cnn_module=False,
                macaron_style=False,
            )
            self.pitch_embedding_proj = torch.nn.Linear(config.hidden_size, config.in_channels)
            self.pitch_length_regulator = InterpolateRegulator(
                channels=config.in_channels,
                sampling_ratios=[1, 1],
            )

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)

    def forward(self, x, timestep, x_mask=None, **kwargs):
        dtype = self.t_embedder.mlp[0].weight.dtype
        x = x.to(dtype)
        timestep = timestep.to(dtype)

        x_mask_for_padding = kwargs["x_mask_for_padding"]

        cfg_mask = kwargs.get("cfg_mask", None)
        if cfg_mask is None:
            cfg_mask = torch.rand(x.shape[0]).cuda() > self.class_dropout_prob

        if self.use_content:
            content = kwargs["content"]
            content_mask_for_padding = kwargs["content_mask_for_padding"]
            content[~content_mask_for_padding] = self.padding_idx
            content = self.content_embedding(content)
            # content encode
            content, content_lengths = self.content_embedding_adapter(content, content_mask_for_padding.sum(dim=1))
            content = self.content_embedding_proj(content)
            content, content_lengths = self.content_length_regulator(content, x_mask_for_padding.sum(dim=1))
            if not cfg_mask.all():
                content = content * cfg_mask.view(-1, 1, 1)
            x = torch.cat([x, content], dim=-1)

        if self.use_speaker:
            speaker = kwargs["speaker"]
            speaker = F.normalize(speaker, dim=1)
            speaker = self.speaker_embedding_proj(speaker)
            if not cfg_mask.all():
                speaker = speaker * cfg_mask.view(-1, 1)
            x = torch.cat([x, speaker.unsqueeze(1).repeat(1,x.size(1),1)], dim=-1)

        if self.use_pitch:
            pitch = kwargs["pitch"]
            pitch_mask_for_padding = kwargs["pitch_mask_for_padding"]
            pitch[~pitch_mask_for_padding] = self.pitch_padding_idx
            pitch = self.pitch_embedding(pitch)
            # pitch encode
            pitch, pitch_lengths = self.pitch_embedding_adapter(pitch, pitch_mask_for_padding.sum(dim=1))
            pitch = self.pitch_embedding_proj(pitch)
            pitch, pitch_lengths = self.pitch_length_regulator(pitch, x_mask_for_padding.sum(dim=1))
            if not cfg_mask.all():
                pitch = pitch * cfg_mask.view(-1, 1, 1)
            x = torch.cat([x, pitch], dim=-1)

        # === get timestep embed ===
        t = self.t_embedder(timestep, dtype=x.dtype)  # [B, C]
        t_mlp = self.t_block(t)
        t0 = t0_mlp = None
        if x_mask is not None:
            t0_timestep = torch.zeros_like(timestep)
            t0 = self.t_embedder(t0_timestep, dtype=x.dtype)
            t0_mlp = self.t_block(t0)

        if x_mask_for_padding.size(0) != x.size(0):
            x_mask_for_padding = torch.cat([x_mask_for_padding, x_mask_for_padding], 0)

        # === first layer ===
        x = x.transpose(2,1)
        x = self.first_layer(x, x_mask_for_padding.unsqueeze(1))  # [B, N, C]
        x = x.transpose(2,1)

        # === blocks ===
        for block, in zip(self.blocks):
            x = auto_grad_checkpoint(block, x, t_mlp, x_mask, t0_mlp, x_mask_for_padding=x_mask_for_padding)

        # === final layer ===
        x = x.transpose(2,1)
        x = self.final_layer(x, x_mask_for_padding.unsqueeze(1))  # [B, N, C]
        x = x.transpose(2,1)

        # cast to float32 for better accuracy
        x = x.to(torch.float32)
        return x

@MODELS.register_module("V2SFlowDecoder-S")
def V2SFlowDecoder_S(from_pretrained=None, **kwargs):
    config = V2SFlowDecoderConfig(
        in_channels=80,
        depth=8,
        hidden_size=512,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.1,
        pred_sigma=False,
        **kwargs
    )
    model = V2SFlowDecoder(config)
    if from_pretrained is not None:
        load_checkpoint(model, from_pretrained)
    return model

@MODELS.register_module("V2SFlowDecoder-S/2")
def V2SFlowDecoder_S_2(from_pretrained=None, **kwargs):
    config = V2SFlowDecoderConfig(
        in_channels=160,
        depth=8,
        hidden_size=512,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.1,
        pred_sigma=False,
        **kwargs
    )
    model = V2SFlowDecoder(config)
    if from_pretrained is not None:
        load_checkpoint(model, from_pretrained)
    return model

@MODELS.register_module("V2SFlowDecoder-S/4")
def V2SFlowDecoder_S_4(from_pretrained=None, **kwargs):
    config = V2SFlowDecoderConfig(
        in_channels=320,
        depth=8,
        hidden_size=512,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.1,
        pred_sigma=False,
        **kwargs
    )
    model = V2SFlowDecoder(config)
    if from_pretrained is not None:
        load_checkpoint(model, from_pretrained)
    return model
