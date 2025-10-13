from itertools import batched
from typing import TYPE_CHECKING, Union

import einops
import torch
from bitsandbytes.nn import Linear4bit, Linear8bitLt
from hqq.core import quantize as hqq_quantize
from torch import nn
from torch.nn.functional import interpolate

from SIEDD.configs import (
    LoraType,
    QuantizationConfig,
    SkipConnectionType,
    STRAINERNetConfig,
)
from SIEDD.layers.batch import BatchLinear
from SIEDD.utils import generate_coordinates, helpers

if TYPE_CHECKING:
    from SIEDD.strainer import Strainer, StrainerNet


class InferenceBatchLinear(nn.Module):
    def __init__(self, linear: BatchLinear, is_last: bool):
        super().__init__()
        self.is_last = is_last
        self.register_buffer("bl_weight", linear.weight, persistent=False)
        self.register_buffer("bl_bias", linear.bias, persistent=False)
        self.bl_weight: nn.Buffer
        self.bl_bias: nn.Buffer
        self.dim_in = linear.in_features
        self.dim_out = linear.out_features
        self.num_decoders = linear.num_decoders
        self.skip_connection: SkipConnectionType = linear.skip_connection
        self.linears = nn.ModuleList(
            [
                nn.Linear(linear.in_features, linear.out_features, device="cuda")
                for _ in range(self.num_decoders)
            ]
        )
        with torch.no_grad():
            self.using_lora = linear.using_lora
            if linear.using_lora and linear.lora_type == "sinlora":
                self.lora_omega = linear.omega
                self.lora_rank = linear.lora_rank
                self.lora_type: LoraType = linear.lora_type
                self.register_buffer("bl_U", linear.U, persistent=False)
                self.register_buffer("bl_V", linear.V, persistent=False)
                self.bl_U: nn.Buffer
                self.bl_V: nn.Buffer
                adapter = torch.sin(linear.omega * self.bl_U @ self.bl_V) / linear.g
                W = self.bl_weight + adapter
            elif linear.using_lora and linear.lora_type == "lora":
                self.lora_rank = linear.lora_rank
                self.lora_type: LoraType = linear.lora_type
                self.register_buffer("bl_U", linear.U, persistent=False)
                self.register_buffer("bl_V", linear.V, persistent=False)
                self.bl_U: nn.Buffer
                self.bl_V: nn.Buffer
                adapter = self.bl_U @ self.bl_V
                W = self.bl_weight + adapter
            else:
                W = self.bl_weight
            for i in range(self.num_decoders):
                lin_i = self.linears[i]
                assert isinstance(lin_i, nn.Linear)
                lin_i.weight.copy_(W[i].T)
                if linear.use_bias:
                    lin_i.bias.copy_(self.bl_bias[i, 0])
        self.activation = linear.activation

    def forward(
        self, x: list[torch.Tensor | torch.Tensor], decoder_idx=None, abcd=None
    ):
        x_range = list(range(self.num_decoders))
        lin_range = list(range(self.num_decoders))
        idx = slice(None) if decoder_idx is None else decoder_idx
        lin_range = lin_range[idx]
        if isinstance(x, torch.Tensor):
            x = [x] * len(lin_range)
        x_range = list(range(len(x)))
        out = []
        for x_i, l_i in zip(x_range, lin_range):
            inp = x[x_i]
            output = self.activation(self.linears[l_i](inp))
            if output.shape[-1] == inp.shape[-1] and self.skip_connection == "+":
                output = output + inp
            elif output.shape[-1] == inp.shape[-1] and self.skip_connection == "*":
                output = output * inp
            out.append(output)
        if self.is_last:
            # for last layer
            out = torch.stack(out)
        return out

    @torch.no_grad()
    def to_batchlinear(self) -> BatchLinear:
        bl = BatchLinear(
            self.bl_weight.shape[-2],
            self.bl_weight.shape[-1],
            self.num_decoders,
            activation="none",
            skip_connection=self.skip_connection,
            sep_patch_pix=False,
            is_last=False,
            qcfg=QuantizationConfig(),
        )
        bl = bl.eval()
        bl.activation = self.activation
        bl.weight.copy_(self.bl_weight)
        bl.bias.copy_(self.bl_bias)
        if self.using_lora:
            bl.lora(self.lora_type, self.lora_rank, self.lora_omega)
            bl.U.copy_(self.bl_U)
            bl.V.copy_(self.bl_V)
        return bl


class NoLoraLinear(nn.Linear):
    def __init__(
        self, in_features, out_features, activation, bias=True, device=None, dtype=None
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.activation = activation

    def forward(self, input, decoder_idx, abcd: torch.Tensor | None):  # type: ignore
        return self.activation(super().forward(input))


class BnBLinear8bit(Linear8bitLt):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        bias=True,
        has_fp16_weights=False,
        threshold=0.0,
        index=None,
        device=None,
        activation=None,
    ):
        super().__init__(
            input_features,
            output_features,
            bias,
            has_fp16_weights,
            threshold,
            index,
            device,
        )
        self.activation = activation

    def forward(self, input, decoder_idx=None, abcd=None):  # type: ignore
        x = super().forward(input)
        if self.activation is not None:
            return self.activation(x)
        return x


class BnBLinear4bit(Linear4bit):
    def __init__(
        self,
        input_features,
        output_features,
        bias=True,
        compute_dtype=None,
        compress_statistics=True,
        quant_type="fp4",
        quant_storage=torch.uint8,
        device=None,
        activation=None,
    ):
        super().__init__(
            input_features,
            output_features,
            bias,
            compute_dtype,
            compress_statistics,
            quant_type,
            quant_storage,
            device,
        )
        self.activation = activation

    def forward(self, input, decoder_idx=None, abcd=None):  # type: ignore
        x = super().forward(input)
        if self.activation is not None:
            return self.activation(x)
        return x


# HQQ overwrites their HQQLinear.forward so we have to do this
normal_hqq_forward = getattr(
    hqq_quantize.HQQLinear, hqq_quantize.HQQLinear.backend.value
)


def hqq_linear_forward(instance, input, decoder_idx=None, abcd=None):
    x = normal_hqq_forward(instance, input)
    if instance.activation is not None:
        return instance.activation(x)


class HQQLinearLayer(hqq_quantize.HQQLinear):
    def __init__(
        self,
        linear_layer: Union[nn.Module, None],
        quant_config: dict,
        del_orig: bool = True,
        compute_dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        initialize: bool = True,
        activation=None,
    ):
        super().__init__(
            linear_layer, quant_config, del_orig, compute_dtype, device, initialize
        )
        self.activation = activation
        HQQLinearLayer.forward = hqq_linear_forward

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        state_dict = {k.removeprefix(prefix): v for k, v in state_dict.items()}
        return self.load_state_dict(state_dict, strict)


def replace(module: nn.Module):
    children = list(module.named_children())
    for i, (name, m) in enumerate(children):
        if isinstance(m, BatchLinear):
            is_last = i == len(children) - 1
            if m.using_lora and m.lora_type == "no_lora":
                linear = NoLoraLinear(
                    m.in_features, m.out_features, m.activation, m.use_bias
                )
                linear.weight = m.weight
                linear.bias = nn.Parameter(m.bias[0])
                setattr(module, name, linear)
            else:
                setattr(module, name, InferenceBatchLinear(m, is_last))
        else:
            replace(m)


@torch.no_grad()
def setup_dynamic_res_decoding(
    strainer: "Strainer", model: "StrainerNet", new_shape: tuple[int, int]
):
    # new_shape is (H, W)
    original_shape = strainer.data_pipeline.data_shape
    patch_size = strainer.enc_cfg.patch_size
    assert isinstance(patch_size, int)
    _, H_0, W_0 = original_shape
    H, W = new_shape
    smallest_h = H_0 // patch_size
    smallest_w = W_0 // patch_size
    if H % smallest_h != 0 or W % smallest_w != 0:
        raise ValueError(
            f"Invalid resolution {new_shape}, must be multiple of {smallest_h}x{smallest_w}"
        )
    new_patch_size = (patch_size * H // H_0, patch_size * W // W_0)

    # Update last decoder layer for dynamic resolution
    last_layer = model.decoderINRs.net[-1]
    assert isinstance(last_layer, InferenceBatchLinear)
    for linear in last_layer.linears:
        assert isinstance(linear, nn.Linear)
        w = linear.weight.clone()
        w_new = einops.rearrange(
            w, "(c ph pw) dim_in -> dim_in c ph pw", ph=patch_size, pw=patch_size
        )
        w_interp = interpolate(w_new, new_patch_size)
        w_interp = einops.rearrange(
            w_interp,
            "dim_in c ph pw -> (c ph pw) dim_in",
            ph=new_patch_size[0],
            pw=new_patch_size[1],
        )
        linear.weight = nn.Parameter(w_interp)
        b = linear.bias.clone()
        b_new = einops.rearrange(
            b, "(c ph pw) -> () c ph pw", ph=patch_size, pw=patch_size
        )
        b_interp = interpolate(b_new, new_patch_size)
        b_interp = einops.rearrange(
            b_interp,
            "() c ph pw -> (c ph pw)",
            ph=new_patch_size[0],
            pw=new_patch_size[1],
        )
        linear.bias = nn.Parameter(b_interp)
        linear.out_features = (
            linear.out_features
            * (new_patch_size[0] * new_patch_size[1])
            // (patch_size**2)
        )


def decode_image(strainer: "Strainer", frame: int, resolution: tuple[int, int]):
    """resolution should be W x H"""

    W, H = resolution
    orig_h, orig_w = strainer.data_pipeline.data_shape[-2:]
    if strainer.cfg.encoder_cfg.patch_size is not None and (H, W) != (orig_h, orig_w):
        coords_h, coords_w = orig_h, orig_w
    else:
        coords_w, coords_h = resolution

    # Set resolution
    if isinstance(strainer.enc_cfg.net, STRAINERNetConfig):
        correct_pos_enc_cfg = strainer.enc_cfg.net.mlp_cfg.pos_encode_cfg
    else:
        correct_pos_enc_cfg = strainer.enc_cfg.net.pos_encode_cfg
    strainer.coordinates = generate_coordinates(
        [coords_h, coords_w],
        strainer.enc_cfg.patch_size,
        strainer.enc_cfg.normalize_range,
        correct_pos_enc_cfg,
        None,
    ).cuda()  # type: ignore
    strainer.data_pipeline.data_set.load_state(strainer.save_path)
    num_frames = strainer.data_pipeline.num_frames
    group_size = strainer.train_cfg.group_size
    it = list(
        map(
            list,
            batched(range(num_frames), group_size),
        )
    )
    frame_group = [i for i in it if frame in i]
    if len(frame_group) != 1:
        raise ValueError("Should have 1 frame group identified, got", len(frame_group))
    frames = frame_group[0]

    # Create model and load shared encoder weights
    model = strainer.create_model(group_size, False)
    shared_state = strainer.load_artefacts(strainer.save_path / "shared_encoder.bin")
    shared_state = {
        k.removeprefix("_orig_mod.").removeprefix("encoderINR."): v
        for k, v in shared_state.items()
        if "encoderINR" in k
    }
    model.encoderINR.load_state_dict(shared_state, strict=False)
    model.encoderINR.requires_grad_(False)
    name = "_".join(map(str, frames))
    filename = f"model_{name}.bin"
    fn = strainer.save_path / filename
    decoder_state = strainer.load_artefacts(fn)
    if strainer.cfg.quant_cfg.quantize and strainer.cfg.quant_cfg.quant_method == "hqq":
        replace(model)
        model = helpers.convert_to_hqq(
            model, quant_bit=strainer.cfg.quant_cfg.quant_bit, group_size=32
        )
        model.load_state_dict(decoder_state, strict=False)
    elif (
        strainer.cfg.quant_cfg.quantize
        and strainer.cfg.quant_cfg.quant_method == "post"
    ):
        state_keys = [x for x in model.state_dict().keys() if "decoderINR" in x]
        recon_state = helpers.decompress_weights(decoder_state, state_keys)
        model.load_state_dict(recon_state, strict=False)
        replace(model)
    else:
        model.load_state_dict(decoder_state, strict=False)
        replace(model)
    model = model.bfloat16()
    if strainer.cfg.encoder_cfg.patch_size is not None and (H, W) != (orig_h, orig_w):
        setup_dynamic_res_decoding(strainer, model, (H, W))
        model.patch_size = strainer.cfg.encoder_cfg.patch_size * H // orig_h
        enc_cfg = strainer.enc_cfg.model_copy(deep=True)
        enc_cfg.patch_size = model.patch_size
    else:
        strainer.data_pipeline.data_shape = [H, W]
        enc_cfg = strainer.enc_cfg
    fps, model_output = strainer.get_fps(model, frames)
    processed_out = helpers.process_predictions(
        model_output, enc_cfg, input_data_shape=[H, W]
    )
    return processed_out, fps
