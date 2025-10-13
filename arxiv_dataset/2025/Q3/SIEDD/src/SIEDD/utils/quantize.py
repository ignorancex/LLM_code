from torch import nn
from . import helpers
from ..configs import QuantizationConfig, EncoderConfig
from typing import TypeVar
from torchao.quantization import (
    Int8WeightOnlyConfig,
    Int4WeightOnlyConfig,
    quantize_,
)
from optimum.quanto import quantize as quanto_quantize, freeze
from optimum import quanto
from SIEDD.decode import replace, InferenceBatchLinear

ModelT = TypeVar("ModelT", bound=nn.Module)


class Quantize:
    def __init__(self, cfg: QuantizationConfig, enc_cfg: EncoderConfig):
        self.cfg = cfg
        self.enc_cfg = enc_cfg
        self.method = self.cfg.quant_method

    def quantize_model(
        self, model: ModelT, best_model_state: dict, inp=None
    ) -> tuple[ModelT, dict]:
        """
        Quantizes the model based on params and returns the
        quantized version of model and the state dict.
        """
        if self.method == "post":
            keys_to_ignore = self.cfg.keys_to_ignore
            compressed_info = helpers.quantize_compress_weights(
                best_model_state,
                quant_axis=self.cfg.quant_axis,
                quant_bit=self.cfg.quant_bit,
                entropy_coding=self.cfg.entropy_coding,
                qfloat=self.cfg.qfloat,
                keys_to_ignore=keys_to_ignore,
                sparsity=self.cfg.sparsity,
            )

            state_keys = list(best_model_state.keys())
            recon_state = helpers.decompress_weights(compressed_info, state_keys)

            model.load_state_dict(recon_state, strict=False)
        elif self.method == "torchao":
            model.load_state_dict(best_model_state, strict=False)
            replace(model)

            quant_cfg = (
                Int4WeightOnlyConfig()
                if self.cfg.quant_bit == 4
                else Int8WeightOnlyConfig()
            )

            # Print all modules to debug the structure
            print("\nAll modules in model:")
            for name, module in model.named_modules():
                if isinstance(module, InferenceBatchLinear):
                    print(
                        f"  - {name}: BatchLinear with weight shape {module.weight.shape}"
                    )
                elif isinstance(module, nn.Linear):
                    print(f"  - {name}: Linear with weight shape {module.weight.shape}")
                elif isinstance(module, nn.ModuleList):
                    print(f"  - {name}: ModuleList with {len(module)} elements")

            def filter_fn(module: nn.Module, name: str):
                # only quantize decoderINRs, skip anything in keys_to_ignore,
                # and match both nn.Linear *and* BatchLinear
                include = False  # Default to not including

                # Only proceed if this is a quantizable module type
                if isinstance(module, (nn.Linear, InferenceBatchLinear)):
                    # Check if module is in decoderINRs section and not in keys_to_ignore
                    if "decoderINRs" in name and not any(
                        ignore in name for ignore in self.cfg.keys_to_ignore
                    ):
                        include = True

                # Report debug info about this decision
                print(
                    f"Module {name} of type {type(module).__name__} will{' ' if include else ' NOT '}be quantized"
                )
                if include and isinstance(module, InferenceBatchLinear):
                    print(f"  - weight shape: {module.weight.shape}")

                return include

            print(
                f"\nStarting quantization of model with bit width: {self.cfg.quant_bit}"
            )
            quantize_(
                model,
                quant_cfg,
                filter_fn,
                "cuda",
            )

            # Print which modules were actually quantized
            print("\nQuantized modules:")
            quantized_count = 0
            for n, m in model.named_modules():
                if hasattr(m, "_is_quantized"):
                    print(f"  - {n} (custom quantization)")
                    quantized_count += 1
                elif hasattr(m, "weight") and hasattr(m.weight, "tensor_impl"):
                    print(
                        f"  - {n} → {m.weight.tensor_impl_dtype}, {m.weight.tensor_impl.shape}"  # type: ignore
                    )
                    quantized_count += 1

            print(f"Total quantized modules: {quantized_count}")

            # Get the compressed state dict
            compressed_info = model.state_dict()
            print(f"State dict contains {len(compressed_info)} entries")
            # Return the quantized model and its state dict
            return model, compressed_info

        elif self.method == "hqq":
            model.load_state_dict(best_model_state, strict=False)
            replace(model)
            model = helpers.convert_to_hqq(
                model, quant_bit=self.cfg.quant_bit, group_size=32
            )
            compressed_info = {
                k: v for k, v in model.state_dict().items() if "encoderINR" not in k
            }

        elif self.method == "quanto":
            model.load_state_dict(best_model_state, strict=False)
            exclude_patterns = [f"*{k}*" for k in self.cfg.keys_to_ignore]
            weight_dtype = getattr(quanto, self.cfg.quanto_dtype)
            quanto_quantize(model, weights=weight_dtype, exclude=exclude_patterns)
            freeze(model)
            compressed_info = {
                k: v for k, v in model.state_dict().items() if "encoderINR" not in k
            }

        elif self.method == "bnb":
            model.load_state_dict(best_model_state, strict=False)
            replace(model)
            # first fp16
            model = model.half()
            model = helpers.convert_to_bnb(
                model, layer_list="middle", quant_bit=self.cfg.quant_bit
            )
            compressed_info = {
                k: v for k, v in model.state_dict().items() if "encoderINR" not in k
            }

        else:
            raise ValueError("Invalid Quantization Config")

        return model, compressed_info
