import torch
from torch import nn
from pathlib import Path
import copy
from ..configs import (
    EncoderConfig,
    STRAINERNetConfig,
    NoPosEncode,
    MLPConfig,
    QuantizationConfig,
)

from ..layers import BatchMLP, LoraLinear, BatchLinear
from .mlp_block import MLPBlock, MLPLayer
from .incode import INCODEHarmonizer


class IdentityDecoder(nn.Module):
    def __init__(self, num_decoders: int):
        super().__init__()
        self.num_decoders = num_decoders
        self.net: nn.ModuleList

    def forward(self, x: torch.Tensor, decoder_idx=None):
        return x.repeat((self.num_decoders, 1, 1))

    def forward_batched(self, x: torch.Tensor, batch_size: int, chunk_size: int):
        return self.forward(x)

    def __getitem__(self, item):
        return self


class Strainer(nn.Module):
    def __init__(
        self,
        dim_out: int,
        data_shape: list[int],
        num_decoders: int,
        enc_cfg: EncoderConfig,
        quant_config: QuantizationConfig,
        gt: torch.Tensor,
        shared: bool,
    ):
        from .generic_mlp import get_mlp

        super().__init__()
        self.qcfg = quant_config
        self.gt = gt
        self.shared = shared
        self.num_decoders = num_decoders
        enc_cfg = enc_cfg.model_copy(deep=True)
        assert isinstance(enc_cfg.net, STRAINERNetConfig)
        self.net_cfg = enc_cfg.net
        self.num_decoder_layers = enc_cfg.net.num_decoder_layers
        self.patch_size = enc_cfg.patch_size
        self.bottleneck_decoder = self.net_cfg.bottleneck_decoder
        self.bottleneck_decoder_override = self.net_cfg.bottleneck_decoder_override

        enc_cfg.net = enc_cfg.net.mlp_cfg.model_copy(deep=True)
        if self.num_decoder_layers == -1:
            # Whole model is shared
            self.encoderINR = get_mlp(dim_out, data_shape, enc_cfg)
            self.decoderINRs = IdentityDecoder(self.num_decoders)
        else:
            dec_cfg = enc_cfg.model_copy(deep=True)
            dec_cfg.net = enc_cfg.net.model_copy(deep=True)
            enc_cfg.net.final_activation = (
                enc_cfg.net.activation
            )  # final activation only for decoder
            if enc_cfg.net.num_layers == -1:
                enc_cfg.net.pos_encode_cfg.dim_out = enc_cfg.net.pos_encode_cfg.dim_in
                encoder_out_dim = enc_cfg.net.pos_encode_cfg.dim_out
            else:
                encoder_out_dim = enc_cfg.net.dim_hidden
            self.encoderINR = get_mlp(encoder_out_dim, data_shape, enc_cfg)
            dec_cfg.net.pos_encode_cfg = NoPosEncode(
                dim_in=encoder_out_dim, dim_out=encoder_out_dim
            )
            dec_cfg.net.num_layers = self.num_decoder_layers
            dec_cfg.net.bottleneck = self.bottleneck_decoder
            dec_cfg.net.bottleneck_override = self.bottleneck_decoder_override
            assert isinstance(dec_cfg.net, MLPConfig)
            self.decoderINRs = BatchMLP(
                dim_out=dim_out,
                num_decoders=num_decoders,
                cfg=dec_cfg.net,
                skip_connection=self.net_cfg.skip_connection,
                sep_patch_pix=self.net_cfg.sep_patch_pix,
                qcfg=self.qcfg,
            )
        if self.net_cfg.incode:
            self.incode_harmonizer = INCODEHarmonizer(data_shape)

    def forward(
        self,
        x: torch.Tensor | list[torch.Tensor],
        frame_idx=None,
        preprocess_output=True,
    ):
        abcd = None
        encoder_abcd = None
        if self.net_cfg.incode:
            abcd = self.incode_harmonizer(self.gt)
            if self.shared:
                encoder_abcd = abcd
        out: torch.Tensor = self.encoderINR(
            x, preprocess_output=False, abcd=encoder_abcd
        )
        out = self.decoderINRs(out, decoder_idx=frame_idx, abcd=abcd)
        if preprocess_output:
            out = self.process_output(out)
        return out

    def forward_batched(
        self,
        input: torch.Tensor | list[torch.Tensor],
        batch_size: int,
        chunk_size: int,
        preprocess_output=True,
    ):
        if isinstance(input, list):
            raise NotImplementedError("CoordX not implemented yet")
        input_chunked = torch.chunk(input, chunk_size)
        outs = []
        for input_chunk in input_chunked:
            chunk_out = self.encoderINR(input_chunk, preprocess_output=False)
            chunk_out = self.decoderINRs.forward_batched(chunk_out, batch_size, 1)
            outs.append(chunk_out)
        out = torch.concatenate(outs, dim=1)
        if preprocess_output:
            out = self.process_output(out)
        return out

    def process_output(self, out: torch.Tensor):
        if self.patch_size is not None:
            out = out.reshape(
                out.size(0), out.size(1), 3, self.patch_size, self.patch_size
            )
        return out

    def load_decoder_weights_from(
        self, fellow_model: "Strainer", self_decoder_idx, other_decoder_idx
    ):
        # Loads fellow_model's decoder to every decoder in self (broadcasts if needed)
        for this_layer, other_layer in zip(
            self.decoderINRs.net, fellow_model.decoderINRs.net
        ):
            assert isinstance(this_layer, BatchLinear)
            assert isinstance(other_layer, BatchLinear)
            idx = slice(None) if this_layer.using_lora else self_decoder_idx
            weight = other_layer.weight[other_decoder_idx]
            if (
                not other_layer.using_lora
                and this_layer.using_lora
                and this_layer.lora_type == "no_lora"
            ):
                weight = weight.mT
            this_layer.weight.data[idx].copy_(weight)
            this_layer.bias.data[idx].copy_(other_layer.bias[other_decoder_idx])

    def load_encoder_weights_from(self, fellow_model: "Strainer"):
        # Need to deepcopy state dict?
        self.encoderINR.load_state_dict(
            copy.deepcopy(fellow_model.encoderINR.state_dict()),
            strict=False,
        )

    def load_weights_from_file(self, file: Path):
        weights = torch.load(file)
        self.encoderINR.load_state_dict(weights["encoder_weights"], strict=False)

    def lora(self, encoder, decoder):
        assert isinstance(self.encoderINR, MLPBlock)
        if encoder:
            for i, layer in enumerate(self.encoderINR.net):
                assert isinstance(layer, MLPLayer)
                layer.linear = LoraLinear.from_linear(
                    layer.linear, self.net_cfg.lora_rank, self.net_cfg.lora_omega
                )

        if decoder:
            for layer in self.decoderINRs.net[:-1]:
                assert isinstance(layer, BatchLinear)
                layer.lora(
                    self.net_cfg.lora_type,
                    self.net_cfg.lora_rank,
                    self.net_cfg.lora_omega,
                )
