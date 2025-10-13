import torch
import torch.nn as nn
from torch.nn.utils.prune import L1Unstructured
import torchvision

import numpy as np
import os
import json
from pathlib import Path
import copy
from dataclasses import dataclass
import pickle
import compress_pickle
from dahuffman import HuffmanCodec
from typing import Dict, no_type_check, Any, Type, Optional
import constriction
from .state_tools import StateDictOperator
from . import coord_utils
from ..configs import EncoderConfig
from hqq.core import quantize as hqq_quantize
from ..decode import BnBLinear8bit, BnBLinear4bit, HQQLinearLayer, NoLoraLinear

if not hasattr(constriction, "stream"):
    raise Exception()


def trainable_state_dict(m: nn.Module):
    return {
        k: v.data
        for k, v in m.named_parameters()
        if v.requires_grad and "incode_harmonizer" not in k
    }


def get_padded_patch_size(tensor_shape: list[int], patch_size: int) -> list[int]:
    """
    Get the size of the padded tensor when patchified.

    Args:
        tensor_shape (tuple): Shape of the input tensor. C,H,W
        patch_size (int): Size of the patch.
    Returns:
        tuple: Shape of the padded tensor. C,H,W
    """
    channels, height, width = tensor_shape
    pad_height = (patch_size - height % patch_size) % patch_size
    pad_width = (patch_size - width % patch_size) % patch_size
    return [channels, height + pad_height, width + pad_width]


def save_tensor_img(tensor: torch.Tensor, filename: Path):
    """
    conver to image and save.
    """
    # convert tensor to int8 pytorch
    tensor = (tensor * 255).clamp(0, 255).to(torch.uint8).squeeze()
    tensor = tensor.cpu().detach()
    torchvision.io.write_png(tensor, str(filename))


def save_json(data: dict, filename: Path):
    if "quant_info" in data.keys() and "keys_to_ignore" in data["quant_info"].keys():
        data["quant_info"]["keys_to_ignore"] = list(
            data["quant_info"]["keys_to_ignore"]
        )
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def make_dir(path: Path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_pickle(data, filename, compressed=False, torch_save=False):
    if torch_save:
        torch.save(data, filename)
        return

    with open(filename, "wb") as f:
        if compressed:
            compress_pickle.dump(
                data,
                f,
                compression="lzma",
                set_default_extension=False,
                pickler_method="optimized_pickle",
            )
        else:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filename: Path, compressed=False):
    with open(filename, "rb") as f:
        if compressed:
            return compress_pickle.load(f, compression="lzma")
        return pickle.load(f)


DTYPE_BIT_SIZE: Dict[torch.dtype, int] = {
    torch.float32: 32,
    torch.float: 32,
    torch.float64: 64,
    torch.double: 64,
    torch.float16: 16,
    torch.half: 16,
    torch.bfloat16: 16,
    torch.complex32: 32,
    torch.complex64: 64,
    torch.complex128: 128,
    torch.cdouble: 128,
    torch.uint8: 8,
    torch.int8: 8,
    torch.int16: 16,
    torch.short: 16,
    torch.int32: 32,
    torch.int: 32,
    torch.int64: 64,
    torch.long: 64,
    torch.bool: 1,
}


def state_dict_size_in_bits(state_dict: dict[Any, torch.Tensor]):
    """Calculate total number of bits to store `state_dict`."""
    return sum(
        sum(t.nelement() * DTYPE_BIT_SIZE[t.dtype] for t in tensors)
        for tensors in state_dict.values()
    )


def model_size_in_bits(model: nn.Module) -> tuple[float, dict[str, float]]:
    """Calculate total number of bits to store `model` parameters that require gradients."""

    param_info: dict[str, float] = {}
    size = 0
    for name, tensors in model.named_parameters():
        if tensors.requires_grad:
            param_size = tensors.nelement() * DTYPE_BIT_SIZE[tensors.dtype]
            size += param_size
            param_info[name] = param_size
        else:
            param_size = 0
            param_info[name] = param_size

    return size, param_info


def model_size_in_bits_peft(model: nn.Module):
    """Calculate total number of bits to store `model` parameters and buffers in PEFT model."""
    num_bits = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_bits += param.nelement() * DTYPE_BIT_SIZE[param.dtype]
    return num_bits


@dataclass
class QuantizedTensor:
    quant_tensor: torch.Tensor | None
    bits: int
    scale: torch.Tensor
    offset: torch.Tensor
    shape: torch.Size
    sparse: bool


def sparsify_tensor(t: torch.Tensor, sparsity: float):
    prune = L1Unstructured(sparsity)
    t = prune.prune(t)
    return t


def quantize_tensor(
    t: torch.Tensor, bit: int, eps=1e-19, axis=(), sparsity: float | None = None
):
    axis = tuple(ax for ax in axis if -t.ndim < ax < t.ndim)
    if sparsity is not None:
        t = sparsify_tensor(t, sparsity)
    t = t.type(torch.float32)
    t_valid = t != 0
    if t_valid.sum() == 0:
        scale = torch.tensor(0, dtype=t.dtype).to(t.device)
        t_min = torch.tensor(0, dtype=t.dtype).to(t.device)
    else:
        t_min = t.amin(dim=axis, keepdim=True)
        t_max = t.amax(dim=axis, keepdim=True)
        scale = (t_max - t_min) / 2**bit

    max_val = min(torch.finfo(t.dtype).max, 2**bit - 1)
    quant_t = ((t - t_min) / (scale + eps)).clamp(0, max_val)
    quant_t = quant_t.type(torch.uint32).reshape(-1)
    return QuantizedTensor(quant_t.view(torch.uint8), bit, scale, t_min, t.shape, False)


def dequantize_tensor(t: QuantizedTensor, eps=1e-19):
    assert t.quant_tensor is not None
    if t.bits == 16:
        return t.quant_tensor.view(torch.float16).reshape(t.shape)

    return (
        t.quant_tensor.view(torch.uint32).reshape(t.shape) * (t.scale.detach() + eps)
        + t.offset.detach()
    )


def array_to_bits(a: np.ndarray):
    bits = a.itemsize * 8
    a = a.reshape((-1, 1))
    a = a & (2 ** np.arange(bits - 1, -1, -1, dtype=a.dtype)) != 0
    return a


def pack_bits(a: np.ndarray):
    # Expects array of zeros and ones
    # Packs bits into uint32 array
    rem = 0 if a.size % 32 == 0 else 32 - a.size % 32
    a = a.reshape((-1,))
    # Pad bits to make size divisible by 32
    a = np.concatenate((a, np.zeros(rem, dtype=a.dtype)))
    a = a.reshape((-1, 32))
    masks = 2 ** np.arange(31, -1, -1, dtype="uint32")[:, None]
    return (a @ masks).squeeze(-1)


def pack_tensor(t: torch.Tensor, bit: int):
    # expects int tensor
    arr = t.detach().cpu().numpy()
    arr = arr.reshape(-1)
    bits = array_to_bits(arr)
    # Cuts of extra bits beyond `bit`
    bits = bits.reshape((-1, t.itemsize * 8))[:, -bit:].reshape(-1)
    packed_data = pack_bits(bits)
    return torch.from_numpy(packed_data)


def unpack_tensor(t: torch.Tensor, shape, bit: int):
    assert t.dtype == torch.uint32
    arr = t.detach().cpu().numpy()
    bits = array_to_bits(arr)
    if arr.size != np.prod(shape):
        bits = bits.reshape(-1)
        bits = bits[: bit * np.prod(shape)]
    bits = bits.reshape((-1, bit))
    zeros = np.zeros((bits.shape[0], 32 - bit), dtype=np.uint32)
    bits = np.concatenate((zeros, bits), 1)
    arr = pack_bits(bits)
    return torch.from_numpy(arr)


@no_type_check
def ans_compress(input_array, use_quantized_gaussian=False):
    info = {}
    if use_quantized_gaussian:
        # Calculate mean and std of the quantized weights
        mean = np.mean(input_array)
        std = np.std(input_array)
        min_val, max_val = np.min(input_array), np.max(input_array)
        entropy_model = constriction.stream.model.QuantizedGaussian(
            min_val, max_val, mean, std
        )
        info["max"] = max_val
        info["min"] = min_val
        info["mean"] = mean
        info["std"] = std
        info["type"] = "QuantizedGaussian"
    else:
        # Calculate the frequency of each symbol
        unique, counts = np.unique(input_array, return_counts=True)
        probabilities = counts / len(input_array)
        entropy_model = constriction.stream.model.Categorical(
            probabilities, perfect=True
        )
        info["counts"] = counts
        info["length"] = len(input_array)
        info["type"] = "Categorical"

    # Perform ANS encoding
    encoder = constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(input_array.view("int32"), entropy_model)

    # Get the compressed data
    encoded = encoder.get_compressed()

    return encoded, entropy_model, info


@no_type_check
def ans_decompress(encoded, entropy_model_type, total_elements, model_info):
    if entropy_model_type == "QuantizedGaussian":
        min_val = model_info["min"]
        max_val = model_info["max"]
        mean = model_info["mean"]
        std = model_info["std"]
        entropy_model = constriction.stream.model.QuantizedGaussian(
            min_val, max_val, mean, std
        )

    elif entropy_model_type == "Categorical":
        counts = model_info["counts"]
        length = model_info["length"]
        probabilities = counts / length
        entropy_model = constriction.stream.model.Categorical(
            probabilities, perfect=True
        )

    decoder = constriction.stream.stack.AnsCoder(encoded)
    decoded = decoder.decode(entropy_model, total_elements)
    return decoded


def quantize_compress_weights(
    state,
    quant_bit: int,
    quant_axis: int | tuple,
    keys_to_ignore=[],
    entropy_coding="huffman",
    qfloat=False,
    use_quantized_gaussian=False,
    sparsity: float | None = None,
):
    if type(state) is not StateDictOperator:
        state = StateDictOperator(state)
    state = state.state_dict

    quant_weight_list: list[QuantizedTensor] = []
    n_bytes: list[int] = []

    for k, v in state.items():
        if isinstance(v, str):
            continue

        large_tensor = v.dim() in {2, 4} and "bias" not in k
        axis = quant_axis if large_tensor else ()

        if any(ignore_key in k for ignore_key in keys_to_ignore):
            quant_v = QuantizedTensor(
                v.reshape(-1).to(torch.float16).view(torch.uint8),
                16,
                torch.Tensor(0),
                torch.Tensor(0),
                v.shape,
                False,
            )
        else:
            if qfloat:
                orig_shape = v.shape
                v: torch.Tensor = v.to(torch.float8_e5m2).view(torch.uint8).reshape(-1)
                # pad_size = v.numel() % 4
                # if pad_size != 0:
                #     zeros = torch.zeros((pad_size,), dtype=torch.int8, device=v.device)
                #     v = torch.concat((v, zeros))
                # v = v.view(torch.uint32)

                quant_v = QuantizedTensor(
                    v, 8, torch.tensor(0), torch.tensor(0), orig_shape, False
                )
            else:
                quant_v = quantize_tensor(
                    v, bit=quant_bit, axis=axis, sparsity=sparsity
                )

        assert quant_v.quant_tensor is not None
        quant_weight_list.append(quant_v)
        n_bytes.append(quant_v.quant_tensor.numel())

    input_code_list = np.concatenate(
        [
            qv.quant_tensor.detach().cpu().numpy()
            for qv in quant_weight_list
            if qv.quant_tensor is not None
        ]
    )
    pad_size = 4 - input_code_list.size % 4
    zeros = np.zeros((pad_size,), dtype=np.uint8)
    input_code_list = np.concat((input_code_list, zeros))
    input_code_list = input_code_list.view(np.uint32)

    if entropy_coding == "ans":
        encoded, entropy_model, encoding_info = ans_compress(
            input_code_list, use_quantized_gaussian=use_quantized_gaussian
        )
        codec = None
    elif entropy_coding == "huffman":
        codec = HuffmanCodec.from_data(input_code_list)
        encoded = codec.encode(input_code_list)
        encoding_info = {}
    else:
        encoded = input_code_list
        encoding_info = {}
        codec = None

    for qv in quant_weight_list:
        qv.quant_tensor = None
    info: dict[str, "Any"] = {}
    info["metadata"] = quant_weight_list
    info["encoding"] = encoded
    info["entropy_model_info"] = encoding_info
    info["codec"] = codec
    info["quant_axis"] = quant_axis
    info["entropy_coding"] = entropy_coding
    info["use_quantized_gaussian"] = use_quantized_gaussian
    info["n_bytes"] = n_bytes
    info["qfloat"] = qfloat

    return info


def decompress_weights(compressed_dict, model_state_keys, device=None):
    """
    Decompress weights from the compressed dictionary.

    Args:
    compressed_dict: Dict returned by compress_weights function.
    state_keys: Keys found in the original model state_dict.
    device: torch device to put tensors on.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    compressed_dict = copy.deepcopy(compressed_dict)

    n_bytes = compressed_dict["n_bytes"]
    metadata = compressed_dict["metadata"]
    entropy_coding = compressed_dict.get("entropy_coding", "huffman")
    qfloat = compressed_dict["qfloat"]

    if entropy_coding == "huffman":
        codec = compressed_dict["codec"]
        encoding = compressed_dict["encoding"]
        decoded = np.array(codec.decode(encoding)).view(np.uint8)
    elif entropy_coding == "ans":
        model_info = compressed_dict["entropy_model_info"]
        entropy_model_type = model_info["type"]
        encoding = compressed_dict["encoding"]
        total_elements = sum(n_b for n_b in n_bytes)
        decoded = ans_decompress(
            encoding, entropy_model_type, total_elements, model_info
        )
    else:
        decoded = compressed_dict["encoding"].view(np.uint8)

    reconstructed_state = {}
    start = 0

    for k, key in enumerate(model_state_keys):
        tensor_metadata: QuantizedTensor = copy.deepcopy(metadata[k])
        end = start + n_bytes[k]
        temp = torch.tensor(decoded[start:end], dtype=torch.uint8, device=device)
        tensor_metadata.quant_tensor = temp
        if qfloat:
            tensor = tensor_metadata.quant_tensor.view(torch.float8_e5m2)
            numel = np.prod(tensor_metadata.shape)
            tensor = tensor[:numel]
            temp = tensor.float().reshape(tensor_metadata.shape)
        else:
            temp = dequantize_tensor(tensor_metadata)
        reconstructed_state[key] = temp
        start = end

    return reconstructed_state


def huffman_encode_weights(state_dict: dict[Any, torch.Tensor], int_only=True):
    """
    Store quantized weights.
    int_only: huffman encode only int weights.
    """

    shapes = []
    flattened_values = []
    not_included = {}
    for k, v in state_dict.items():
        if int_only and v.is_floating_point():
            not_included[k] = v
            continue
        shapes.append(tuple(v.shape))
        flattened_values.append(v.cpu().detach().flatten())

    cat_param = torch.cat(flattened_values)
    input_code_list = cat_param.tolist()
    input_code_list = [int(x) for x in input_code_list]
    codec = HuffmanCodec.from_data(input_code_list)
    encoded = codec.encode(input_code_list)

    info = {}
    info["encoding"] = encoded
    info["codec"] = codec
    info["shapes"] = shapes
    info["not_included"] = not_included
    return info


def huffman_decode_weights(compressed_dict, state_keys):
    codec = compressed_dict["codec"]
    encoding = compressed_dict["encoding"]
    shapes = compressed_dict["shapes"]
    not_included = compressed_dict["not_included"]
    decoded = codec.decode(encoding)

    reconstructed_state = {}
    start = 0
    for k, key in enumerate(state_keys):
        # temp = torch.tensor(decoded[start:np.prod(shapes[k])]).reshape(shapes[k])
        end = int(np.prod(shapes[k]) + start)
        temp = torch.tensor(decoded[start:end]).reshape(shapes[k])
        reconstructed_state[key] = temp
        start = end

    for k, v in not_included.items():
        reconstructed_state[k] = v

    return reconstructed_state


def get_auto_cast_dtype(precision="float32"):
    # Set the autocast dtype
    if precision in ["float16", "fp16"]:
        autocast_dtype = torch.float16
    elif precision in ["bfloat16", "bf16"]:
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = torch.float32

    return autocast_dtype


def process_predictions(output: torch.Tensor, cfg: EncoderConfig, input_data_shape):
    patch_size = cfg.patch_size
    if len(input_data_shape) == 2:
        H, W = input_data_shape
        C = 3
    elif len(input_data_shape) == 3:
        C, H, W = input_data_shape
    else:
        raise ValueError()

    # output = coord_utils.reverse_pixel_transform(output,pixel_perm_type,self.feature_transform)
    if patch_size is not None:
        output = coord_utils.unpatchify_images(
            output,
            original_height=H,
            original_width=W,
            patch_height=patch_size,
            patch_width=patch_size,
            strided=cfg.strided_patches,
        )
    else:
        N = output.shape[0]
        output = output.permute(0, 2, 1).reshape(N, C, H, W)

    return output


def _create_quantized_layer_from_linear(
    original_linear_layer: nn.Linear,
    quant_bit: int,
    quant_replacement_cls: Type[nn.Module],
    default_temp_dtype: torch.dtype,
    activation_fn: Optional[nn.Module] = None,
) -> nn.Module:
    """Helper function to quantize a given linear layer."""
    original_weight = original_linear_layer.weight
    original_bias = (
        original_linear_layer.bias
        if hasattr(original_linear_layer, "bias")
        and original_linear_layer.bias is not None
        else None
    )
    target_device = original_weight.device

    # 1. Create a temporary standard nn.Linear layer on CPU
    temp_linear_cpu = nn.Linear(
        original_weight.shape[1],  # in_features
        original_weight.shape[0],  # out_features
        bias=(original_bias is not None),
        device="cpu",
        dtype=default_temp_dtype,
    )
    temp_linear_cpu.weight.data = original_weight.to("cpu", dtype=default_temp_dtype)
    if original_bias is not None:
        temp_linear_cpu.bias.data = original_bias.to("cpu", dtype=default_temp_dtype)

    # 2. Create BnBLinear layer on CPU
    common_args = {
        "input_features": temp_linear_cpu.in_features,
        "output_features": temp_linear_cpu.out_features,
        "bias": (temp_linear_cpu.bias is not None),
        "device": "cpu",
        "activation": activation_fn,
    }

    layer_cpu: nn.Module
    if quant_bit == 8:
        # Assuming BnBLinear8bit takes has_fp16_weights
        layer_cpu = quant_replacement_cls(**common_args, has_fp16_weights=False)
    elif quant_bit == 4:
        # Assuming BnBLinear4bit might have other specific args or uses defaults
        layer_cpu = quant_replacement_cls(**common_args)
    else:
        raise ValueError(f"Unsupported quant_bit for helper: {quant_bit}")

    # 3. Convert temp_linear_cpu to half for loading (as per original logic)
    temp_linear_cpu = temp_linear_cpu.half()

    # 4. Load state_dict into the CPU layer
    layer_cpu.load_state_dict(temp_linear_cpu.state_dict())

    # 5. Move the layer to the target CUDA device to trigger quantization
    quantized_layer = layer_cpu.to(target_device)
    return quantized_layer


@no_type_check
def convert_to_bnb(model: nn.Module, layer_list="all", quant_bit=8):
    """
    Converts linear layers in the decoder part of a Strainer model to 8-bit or 4-bit.
    """
    print(f"Converting {layer_list} layers to {quant_bit}-bit.")

    quant_replacement_cls: Type[nn.Module]
    default_temp_dtype: torch.dtype

    if quant_bit == 8:
        quant_replacement_cls = BnBLinear8bit
        default_temp_dtype = torch.float32
    elif quant_bit == 4:
        quant_replacement_cls = BnBLinear4bit
        default_temp_dtype = torch.float16
    else:
        print(
            f"Warning: Unsupported quant_bit {quant_bit}. No conversion will be performed for model."
        )
        return model  # Early exit if quant_bit is not supported

    target_model = model
    if hasattr(
        model, "_orig_mod"
    ):  # Handle if model is wrapped (e.g., OptimizedModule)
        target_model = model._orig_mod

    if not hasattr(target_model, "decoderINRs"):
        print(
            "Warning: Model does not have 'decoderINRs' attribute. No conversion performed."
        )
        return model

    decoder = target_model.decoderINRs

    if not hasattr(decoder, "net") or not isinstance(decoder.net, nn.ModuleList):
        print(
            "Warning: Decoder does not have a 'net' ModuleList. No conversion performed."
        )
        return model

    for i in range(len(decoder.net)):
        module_in_net = decoder.net[i]
        module_type_name = str(type(module_in_net).__name__)

        if "NoLoraLinear" in module_type_name and (
            layer_list == "all" or layer_list == "middle"
        ):
            # Ensure module_in_net is compatible with nn.Linear for the helper
            if not isinstance(module_in_net, nn.Linear):
                print(
                    f"Skipping module {module_type_name} as it is not a nn.Linear subclass."
                )
                continue

            act = (
                module_in_net.activation
                if hasattr(module_in_net, "activation")
                else None
            )
            try:
                decoder.net[i] = _create_quantized_layer_from_linear(
                    original_linear_layer=module_in_net,
                    quant_bit=quant_bit,
                    quant_replacement_cls=quant_replacement_cls,
                    default_temp_dtype=default_temp_dtype,
                    activation_fn=act,
                )
            except Exception as e:
                print(f"Failed to convert NoLoraLinear layer at index {i}: {e}")

        elif "InferenceBatchLinear" in module_type_name and (
            layer_list == "all" or layer_list == "last"
        ):
            if hasattr(module_in_net, "linears") and isinstance(
                module_in_net.linears, nn.ModuleList
            ):
                for j in range(len(module_in_net.linears)):
                    sub_linear_layer = module_in_net.linears[j]
                    if isinstance(sub_linear_layer, nn.Linear):
                        try:
                            # Activation for sub-layers in InferenceBatchLinear is handled by InferenceBatchLinear itself
                            module_in_net.linears[j] = (
                                _create_quantized_layer_from_linear(
                                    original_linear_layer=sub_linear_layer,
                                    quant_bit=quant_bit,
                                    quant_replacement_cls=quant_replacement_cls,
                                    default_temp_dtype=default_temp_dtype,
                                    activation_fn=None,
                                )
                            )
                        except Exception as e:
                            print(
                                f"Failed to convert sub-linear layer in InferenceBatchLinear at main index {i}, sub-index {j}: {e}"
                            )
                    else:
                        print(
                            f"Skipping non-Linear layer {type(sub_linear_layer)} in InferenceBatchLinear.linears at index {j}."
                        )
            else:
                print(
                    f"Skipping InferenceBatchLinear at index {i}: 'linears' ModuleList not found."
                )

    # Final model move to CUDA if available (already handled by _create_quantized_layer_from_linear for individual layers)
    # However, this ensures the overall model is on the correct device if other parts were on CPU.
    if torch.cuda.is_available():
        try:
            # It's generally better to move to a specific device if known, e.g., model.parameters().__next__().device
            # but target_model.to(torch.cuda.current_device()) is a common pattern.
            # This step might be redundant if all layers were already moved by the helper.
            final_device = next(
                target_model.parameters(), torch.tensor(0)
            ).device  # Get device of first param as reference
            if final_device.type == "cpu":  # Only move if it's still on CPU
                target_model = target_model.to(torch.cuda.current_device())
                print(
                    f"Overall model moved to {torch.cuda.get_device_name(torch.cuda.current_device())}."
                )
            elif final_device.type == "cuda":
                print(f"Model already on CUDA device: {final_device}.")

        except Exception as e:
            print(f"Failed to move final model to CUDA or check device: {e}")
    else:
        print(
            "CUDA not available. Model not moved to GPU; quantization may not occur as expected."
        )
    return model


def convert_to_hqq(model, layer_list="middle", quant_bit=8, group_size=128):
    quant_config = hqq_quantize.BaseQuantizeConfig(
        nbits=quant_bit, group_size=group_size
    )

    target_model = model
    if hasattr(
        model, "_orig_mod"
    ):  # Handle if model is wrapped (e.g., OptimizedModule)
        target_model = model._orig_mod

    if not hasattr(target_model, "decoderINRs"):
        print(
            "Warning: Model does not have 'decoderINRs' attribute. No conversion performed."
        )
        return model

    decoder = model.decoderINRs

    for i in range(len(decoder.net)):
        module_in_net = decoder.net[i]
        module_type_name = str(type(module_in_net).__name__)

        if isinstance(module_in_net, NoLoraLinear) and (
            layer_list == "all" or layer_list == "middle"
        ):
            # Ensure module_in_net is compatible with nn.Linear for the helper
            if not isinstance(module_in_net, nn.Linear):
                print(
                    f"Skipping module {module_type_name} as it is not a nn.Linear subclass."
                )
                continue

            act = (
                module_in_net.activation
                if hasattr(module_in_net, "activation")
                else None
            )

            W = module_in_net.weight.data
            bias = module_in_net.bias.data if hasattr(module_in_net, "bias") else None
            temp_layer = nn.Linear(
                W.shape[1],
                W.shape[0],
                bias=bias is not None,
                device=W.device,
                dtype=W.dtype,
            )
            temp_layer.weight.data = W

            if bias is not None:
                temp_layer.bias.data = bias

            temp_layer.requires_grad_(False)

            hqq_layer = HQQLinearLayer(
                linear_layer=temp_layer,
                quant_config=quant_config,
                del_orig=True,
                compute_dtype=torch.bfloat16,
                device="cuda",
                initialize=True,
                activation=act,
            )
            hqq_layer.requires_grad_(False)
            decoder.net[i] = hqq_layer
    return model


def convert_hqq_linear(model):
    target_model = model
    if hasattr(
        model, "_orig_mod"
    ):  # Handle if model is wrapped (e.g., OptimizedModule)
        target_model = model._orig_mod

    if not hasattr(target_model, "decoderINRs"):
        print(
            "Warning: Model does not have 'decoderINRs' attribute. No conversion performed."
        )
        return model

    decoder = target_model.decoderINRs

    for i in range(len(decoder.net)):
        module_in_net = decoder.net[i]
        module_type_name = str(type(module_in_net).__name__)

        if module_type_name != "HQQLinearLayer":
            continue
        W = module_in_net.dequantize()
        B = module_in_net.bias.data
        out_feats, in_feats = W.shape
        linear = NoLoraLinear(
            in_features=in_feats,
            out_features=out_feats,
            activation=module_in_net.activation,
            bias=True,
        )
        linear.weight.data = W
        linear.bias.data = B
        linear.requires_grad_(False)
        decoder.net[i] = linear

    return model
