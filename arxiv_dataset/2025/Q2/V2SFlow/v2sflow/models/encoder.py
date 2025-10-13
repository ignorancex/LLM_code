import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import PretrainedConfig, PreTrainedModel

from third_party.opensora.utils.ckpt_utils import load_checkpoint

from third_party.cosyvoice.transformer.encoder import ConformerEncoder

from v2sflow.registry import MODELS
from v2sflow.models.layers.blocks import Linear

class ContentEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = 4 + config.content_vocab_size # 4 (special tokens)
        self.conformer = ConformerEncoder(
            input_size=config.encoder_embed_dim,
            output_size=config.conformer_embed_dim,
            attention_heads=config.conformer_attention_heads,
            linear_units=config.conformer_ffn_embed_dim,
            num_blocks=config.conformer_layers,
            dropout_rate=config.conformer_dropout,
            positional_dropout_rate=config.conformer_dropout,
            attention_dropout_rate=config.conformer_attention_dropout,
            normalize_before=config.conformer_layer_norm_first,
            input_layer='linear',
            pos_enc_layer_type='rel_pos_espnet',
            selfattention_layer_type='rel_selfattn',
            use_cnn_module=True,
            macaron_style=True,
            cnn_module_kernel=31,
        )
        if config.conformer_embed_dim != embed_dim:
            self.proj_out = Linear(config.conformer_embed_dim, embed_dim)
        else:
            self.proj_out = None

    def forward(self, **kwargs):
        x = kwargs['video_feat'].repeat_interleave(2, dim=1)
        padding_mask = kwargs['video_padding_mask'].repeat_interleave(2, dim=1)
        x, _ = self.conformer(x, (~padding_mask).sum(dim=1))
        x = self.proj_out(x)
        x[..., :4] = -math.inf
        x = x.argmax(dim=-1) - 4 # remove special tokens
        return x
    
class PitchEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = 4 + config.pitch_vocab_size # 4 (special tokens)
        self.downsample = nn.Conv1d(config.encoder_embed_dim, config.encoder_embed_dim, 3, 2, 1)
        self.conformer = ConformerEncoder(
            input_size=config.encoder_embed_dim,
            output_size=config.conformer_embed_dim,
            attention_heads=config.conformer_attention_heads,
            linear_units=config.conformer_ffn_embed_dim,
            num_blocks=config.conformer_layers,
            dropout_rate=config.conformer_dropout,
            positional_dropout_rate=config.conformer_dropout,
            attention_dropout_rate=config.conformer_attention_dropout,
            normalize_before=config.conformer_layer_norm_first,
            input_layer='linear',
            pos_enc_layer_type='rel_pos_espnet',
            selfattention_layer_type='rel_selfattn',
            use_cnn_module=True,
            macaron_style=True,
            cnn_module_kernel=31,
        )
        if config.conformer_embed_dim != embed_dim:
            self.proj_out = Linear(config.conformer_embed_dim, embed_dim)
        else:
            self.proj_out = None

    def forward(self, **kwargs):
        x = self.downsample(kwargs['video_feat'].transpose(1, 2)*(~kwargs['video_padding_mask'].unsqueeze(1))).transpose(1, 2)
        padding_mask = kwargs['video_padding_mask'][:, ::2]
        x, _ = self.conformer(x, (~padding_mask).sum(dim=1))
        x = self.proj_out(x)
        x[..., :4] = -math.inf
        x = x.argmax(dim=-1) - 4 # remove special tokens
        return x

class SpeakerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.speaker_embed_dim
        self.conformer = ConformerEncoder(
            input_size=config.encoder_embed_dim,
            output_size=config.conformer_embed_dim,
            attention_heads=config.conformer_attention_heads,
            linear_units=config.conformer_ffn_embed_dim,
            num_blocks=config.conformer_layers,
            dropout_rate=config.conformer_dropout,
            positional_dropout_rate=config.conformer_dropout,
            attention_dropout_rate=config.conformer_attention_dropout,
            normalize_before=config.conformer_layer_norm_first,
            input_layer='linear',
            pos_enc_layer_type='rel_pos_espnet',
            selfattention_layer_type='rel_selfattn',
            use_cnn_module=True,
            macaron_style=True,
            cnn_module_kernel=31,
        )
        if config.conformer_embed_dim != embed_dim:
            self.proj = Linear(config.conformer_embed_dim, embed_dim)
        else:
            self.proj = None
        self.proj_out = Linear(embed_dim, embed_dim)
        self.relu = nn.ReLU()

    def forward(self, **kwargs):
        x = kwargs['video_feat']
        padding_mask = kwargs['video_padding_mask']
        x, _ = self.conformer(x, (~padding_mask).sum(dim=1))
        x = self.proj(x)
        x = torch.nanmean(
            torch.where(
                padding_mask.unsqueeze(2), torch.tensor(float('nan')).to(x), x
            ),
            dim=1
        )
        x = self.proj_out(x)
        x = self.relu(x)
        x = x.float()
        x = F.normalize(x, dim=1, eps=1e-5)
        return x

class V2SFlowEncoderConfig(PretrainedConfig):
    model_type = "V2SFlowEncoder"

    def __init__(
        self,
        encoder_embed_dim=1024,
        conformer_layers=12,
        conformer_embed_dim=512,
        conformer_ffn_embed_dim=2048,
        conformer_attention_heads=8,
        conformer_dropout=0.1,
        conformer_attention_dropout=0.1,
        conformer_layer_norm_first=True,
        content_vocab_size=None,
        pitch_vocab_size=None,
        speaker_embed_dim=None,
        **kwargs,
    ):
        self.encoder_embed_dim = encoder_embed_dim
        self.conformer_layers = conformer_layers
        self.conformer_embed_dim = conformer_embed_dim
        self.conformer_ffn_embed_dim = conformer_ffn_embed_dim
        self.conformer_attention_heads = conformer_attention_heads
        self.conformer_dropout = conformer_dropout
        self.conformer_attention_dropout = conformer_attention_dropout
        self.conformer_layer_norm_first = conformer_layer_norm_first
        self.content_vocab_size = content_vocab_size
        self.pitch_vocab_size = pitch_vocab_size
        self.speaker_embed_dim = speaker_embed_dim
        super().__init__(**kwargs)

class V2SFlowEncoder(PreTrainedModel):
    config_class = V2SFlowEncoderConfig

    def __init__(self, config):
        super().__init__(config)
        self.content_encoder = ContentEncoder(config) if config.content_vocab_size is not None else None
        self.pitch_encoder = PitchEncoder(config) if config.pitch_vocab_size is not None else None
        self.speaker_encoder = SpeakerEncoder(config) if config.speaker_embed_dim is not None else None

    def forward(self, **kwargs):
        # kwargs['video_feat']: [B x T x C]
        # kwargs['video_padding_mask']: [B x T]
        ret = {
            'content': self.content_encoder(**kwargs) if self.content_encoder is not None else None,
            'pitch': self.pitch_encoder(**kwargs) if self.pitch_encoder is not None else None,
            'speaker': self.speaker_encoder(**kwargs) if self.speaker_encoder is not None else None,
        }
        return ret

@MODELS.register_module("V2SFlowEncoder-S")
def V2SFlowEncoder_S(from_pretrained=None, **kwargs):
    config = V2SFlowEncoderConfig(
        conformer_layers=12,
        conformer_embed_dim=512,
        conformer_ffn_embed_dim=2048,
        conformer_attention_heads=8,
        conformer_dropout=0.1,
        conformer_attention_dropout=0.1,
        **kwargs
    )
    model = V2SFlowEncoder(config)
    if from_pretrained is not None:
        load_checkpoint(model, from_pretrained)
    return model
