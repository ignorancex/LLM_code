import torch
import torch.nn as nn
from typing import no_type_check

# Handler registration decorator
from torchao.quantization.transform_module import register_quantize_module_handler

# Config and default transforms
from torchao.quantization.quant_api import (
    Int4WeightOnlyConfig,
    Int8WeightOnlyConfig,
    quantize_,
)

# Quant primitives and utils
from torchao.dtypes import to_affine_quantized_intx
from torchao.quantization.utils import _get_per_token_block_size
from torchao.quantization.quant_primitives import MappingType, ZeroPointDomain

# Your custom BatchLinear class
from SIEDD.layers.batch import BatchLinear


@register_quantize_module_handler(Int4WeightOnlyConfig)
@no_type_check
def _batchlinear_int4_transform(module: nn.Module, config: Int4WeightOnlyConfig):
    # Only handle 3-D BatchLinear, else fallback
    if not isinstance(module, BatchLinear):
        quantize_(module, config)
        return module
    w = module.weight.data
    if w.dim() != 3:
        quantize_(module, config)
        return module

    D, O, I = w.shape  # noqa: E741
    w2 = w.view(-1, I)
    block_size = _get_per_token_block_size(w2)
    q2 = to_affine_quantized_intx(
        w2,
        MappingType.ASYMMETRIC,
        block_size,
        torch.int32,  # packs int4 into int32 words
        quant_min=0,
        quant_max=15,
        eps=getattr(config, "eps", 1e-5),  # Default eps value if not in config
        scale_dtype=torch.float32,
        zero_point_dtype=torch.int32,
        zero_point_domain=ZeroPointDomain.FLOAT,
        preserve_zero=False,
        _layout=getattr(config, "layout", None),  # Default to None if not in config
        use_hqq=getattr(config, "use_hqq", False),  # Default to False if not in config
    )
    q3 = q2.view(D, O, I)
    module.weight = nn.Parameter(q3, requires_grad=False)
    return module


@register_quantize_module_handler(Int8WeightOnlyConfig)
@no_type_check
def _batchlinear_int8_transform(module: nn.Module, config: Int8WeightOnlyConfig):
    # Only handle 3-D BatchLinear, else fallback
    w = module.weight.data
    if isinstance(module, BatchLinear) and w.dim() == 3:
        # Print debug info
        print(f"Quantizing BatchLinear with shape {w.shape}")

        # Get dimensions
        D, O, I = w.shape  # noqa: E741

        # Process each batch dimension separately
        quantized_weights = []
        for d in range(D):
            # Extract the 2D slice for this batch
            w_slice = w[d]

            # Determine block size for this 2D slice
            block_size = _get_per_token_block_size(w_slice)

            # Use default values for missing attributes
            target_dtype = getattr(config, "target_dtype", torch.int8)
            quant_min = getattr(config, "quant_min", -128)
            quant_max = getattr(config, "quant_max", 127)
            eps = getattr(config, "eps", 1e-5)
            use_hqq = getattr(config, "use_hqq", False)

            # Quantize this 2D slice
            q_slice = to_affine_quantized_intx(
                w_slice,
                MappingType.SYMMETRIC,
                block_size,
                target_dtype,
                eps=eps,
                quant_min=quant_min,
                quant_max=quant_max,
                scale_dtype=torch.float16,
                zero_point_dtype=torch.int8,
                preserve_zero=True,
                zero_point_domain=ZeroPointDomain.INT,
                use_hqq=use_hqq,
            )

            # Add to our list
            quantized_weights.append(q_slice)

        # Keep original weight tensor for shape reference but mark as not requiring gradients
        module.weight.requires_grad_(False)

        # Store memory usage stats for comparison
        print(f"Original weight size: {w.nelement() * w.element_size()} bytes")
        quantized_size = sum(q.nelement() * q.element_size() for q in quantized_weights)
        print(f"Quantized weight size: {quantized_size} bytes")
        print(
            f"Compression ratio: {w.nelement() * w.element_size() / quantized_size:.2f}x"
        )

        # Register each quantized slice as a parameter so it's included in state_dict
        for i, q_weight in enumerate(quantized_weights):
            module.register_parameter(
                f"weight_quantized_{i}", nn.Parameter(q_weight, requires_grad=False)
            )

        # Store the batch shape for later use
        module.register_buffer(
            "_batch_shape", torch.tensor([D, O, I], dtype=torch.int64)
        )

        # Tag this module as quantized for easier identification
        module._is_quantized = True

        # Modify the forward method to use the quantized weights
        # original_forward = module.forward

        def new_forward(x):
            # Get batch size from input
            batch_size = x.size(0)

            # Check if we have enough quantized weights
            D = module._batch_shape[0].item()  # type: ignore
            if batch_size != D:
                raise ValueError(
                    f"Input batch size {batch_size} doesn't match weight batch size {D}"
                )

            # Process each batch item with its corresponding quantized weight
            batch_results = []
            for i in range(D):  # type: ignore
                q_weight = getattr(module, f"weight_quantized_{i}")
                result = torch.nn.functional.linear(x[i : i + 1], q_weight)
                batch_results.append(result)

            return torch.cat(batch_results, dim=0)

        # Replace the forward method
        module.forward = new_forward

        return module

    # fallback to default 2-D handler
    print(
        f"Falling back to standard quantization for module of type {type(module).__name__}"
    )
    quantize_(module, config)
    return module
