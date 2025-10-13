import os
import math
import json
from typing import List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoConfig

from loguru import logger

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    if is_causal:
        causal_mask = torch.triu(torch.ones(L, S, dtype=torch.bool, device=query.device), diagonal=1)
        attn_weight.masked_fill_(causal_mask, float('-inf'))
    if attn_mask is not None:
        attn_weight += attn_mask
    attn_weight = F.softmax(attn_weight, dim=-1)
    if dropout_p > 0.0:
        attn_weight = F.dropout(attn_weight, p=dropout_p)
    return attn_weight @ value

if not hasattr(F, 'scaled_dot_product_attention'):
    F.scaled_dot_product_attention = scaled_dot_product_attention

@dataclass
class AROMAConfig:
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: List[str]
    keep_original_weights: bool
    lora_only: bool = False
    trainable_scaling: bool = False
    quantize: str = None
    use_double_quant: bool = False
    convergence_threshold: float = 1e-4
    check_convergence: bool = False

def merge_and_reinit_functional(module):
    if not isinstance(module, AROMALinear):
        return

    _delta = module.lora_B.weight @ module.lora_A.weight
    _delta = _delta * module._post_lora_scale()
    module.weight.data += _delta # merge
    nn.init.kaiming_uniform_(module.lora_A.weight, a=math.sqrt(5))

    nn.init.zeros_(module.lora_B.weight)
    if module.trainable_scaling:
        nn.init.zeros_(module.scaling)

class AROMAModel(nn.Module): 
    def __init__(
        self,
        model,
        *,
        target_modules,
        r=128, # rank
        lora_alpha=32, # lora alpha
        lora_dropout=0.1,
        keep_original_weights=True,
        lora_only=False,
        trainable_scaling=False,
        quantize=None,
        use_double_quant=False,
        convergence_threshold=1e-4,
        check_convergence=False,
        convergence_window=3,
        lora_check_frequency=10,
        max_steps_before_reset=1000,
        lora_change_threshold=1e-4,
    ):
        if r <= 0:
            raise ValueError("r must be positive. If you want r == 0, use the original model.")
        super().__init__()
        self.wrapped_model: nn.Module = model
        self._original_config = model.config
        
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules
        self.keep_original_weights = keep_original_weights
        self.lora_only = lora_only
        self.trainable_scaling = trainable_scaling
        
        self.convergence_threshold = convergence_threshold
        self.check_convergence = check_convergence
        self.convergence_window = convergence_window
        
        self.lora_check_frequency = lora_check_frequency
        self.max_steps_before_reset = max_steps_before_reset
        self.lora_change_threshold = lora_change_threshold
        
        self._config = AROMAConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            keep_original_weights=keep_original_weights,
            quantize=quantize,
            use_double_quant=use_double_quant,
            convergence_threshold=convergence_threshold,
            check_convergence=check_convergence,
        )

        self.forward = self.wrapped_model.forward

        target_modules_list = target_modules
        if isinstance(target_modules_list, str):
            target_modules_list = [target_modules_list]

        for module_name, module in self.wrapped_model.named_modules():
            
            for param in module.parameters(recurse=False):
                param.requires_grad = False

            if not isinstance(module, nn.Linear): 
                continue

            if not any(target_key in module_name for target_key in target_modules_list): 
                continue

            weight_data = module.weight.data if keep_original_weights else None
            bias_data = module.bias.data if (module.bias is not None and keep_original_weights) else None

            new_module = AROMALinear(
                module.in_features,
                module.out_features,
                module_name=module_name,
                bias=module.bias is not None,
                r=self.r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                lora_only=self.lora_only,
                trainable_scaling=self.trainable_scaling,
                quantize=quantize,
                weight_data=weight_data,
                bias_data=bias_data,
                bnb_4bit_use_double_quant=use_double_quant,
                convergence_threshold=self._config.convergence_threshold,
                check_convergence=self._config.check_convergence,
                convergence_window=self.convergence_window,
                lora_check_frequency=self.lora_check_frequency,
                max_steps_before_reset=self.max_steps_before_reset,
                lora_change_threshold=self.lora_change_threshold,
            )

            if self.keep_original_weights:
                assert new_module.lora_A.bias is None
                assert new_module.lora_B.bias is None

            if self.lora_only:
                assert not self.keep_original_weights
                module.weight = None

            del module

            parent = self._get_parent(module_name)
            module_suffix = module_name.split(".")[-1]
            setattr(parent, module_suffix, new_module)
 
        torch.cuda.empty_cache()

    def _get_parent(self, module_name):
        module_names_list = module_name.split(".")
        parent_name = ".".join(module_names_list[:-1])
        parent = self.wrapped_model.get_submodule(parent_name)
        return parent

    def get_input_embeddings(self):
        return self.wrapped_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.wrapped_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.wrapped_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.wrapped_model.set_output_embeddings(new_embeddings)

    def merge_check_and_reinit(self):
        for module in self.modules():
            if isinstance(module, AROMALinear):
                module.merge_check_and_reinit()

    def get_convergence_status(self):
        status = {}
        for name, module in self.named_modules():
            if isinstance(module, AROMALinear):
                status[name] = module.weight_converged
        return status

    def check_all_converged(self):
        for module in self.modules():
            if isinstance(module, AROMALinear) and not module.weight_converged:
                return False
        return True

    def save_pretrained(self, path):
        self.wrapped_model.save_pretrained(path)
        with open(os.path.join(path, "AROMA_config.json"), "w") as f:
            json.dump(self._config.__dict__, f, indent=4)

    @classmethod
    def from_pretrained(cls, path):
        with open(os.path.join(path, "AROMA_config.json"), "r") as f:
            AROMA_config = json.load(f)

        config = AutoConfig.from_pretrained(path)
        base_model = AutoModelForCausalLM.from_config(config)
        
        if "keep_original" in AROMA_config:
            logger.warning("keep_original is deprecated. Use lora_only instead.")
            AROMA_config["lora_only"] = not AROMA_config.pop("keep_original")
            AROMA_config["keep_original_weights"] = not AROMA_config["lora_only"]

        if "trainable_scaling" not in AROMA_config:
            AROMA_config["trainable_scaling"] = False

        model = cls(base_model, **AROMA_config)
        
        with open(os.path.join(path, "pytorch_model.bin"), "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        model.wrapped_model.load_state_dict(state_dict, strict=True)
        
        return model

    @property
    def config(self):
        return self._original_config    

    def check_lora_changes(self):
        modules_to_reset = []
        for name, module in self.named_modules():
            if isinstance(module, AROMALinear):
                if module.check_lora_change():
                    modules_to_reset.append(name)
        return len(modules_to_reset), modules_to_reset

# The code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
class AROMALinear(nn.Module):
    def __init__(
        self,
        in_features: int, # size
        out_features: int, # size
        r: int, # rank
        *,
        lora_alpha: int = 1,
        lora_dropout: float = 0.1,
        lora_only: bool = False,
        weight_data=None,
        bias_data=None,
        trainable_scaling: bool = False,
        bias=True,
        device=None,
        dtype=None,
        quantize=None,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        convergence_threshold: float = 1e-4,
        check_convergence: bool = False,
        module_name: str = None,
        convergence_window: int = 3,
        lora_check_frequency=10,
        max_steps_before_reset=1000,
        lora_change_threshold=1e-4,
    ):
        """Wraps linear layer x W into x W + x W_a @ W_b * lora_alpha / r
        
        Notice that scale = lora_alpha / r.
        """
        nn.Module.__init__(self)
        if r <= 0:
            raise ValueError("r must be positive. If you want r == 0, use the original model.")

        if lora_only:
            self.weight = None
            self.bias = None
        else:
            if bias_data is None:
                bias_data = torch.zeros(out_features, device=device, dtype=dtype, requires_grad=True) if bias else None
            self.bias = nn.Parameter(bias_data, requires_grad=False) if bias else None

            if weight_data is None:
                # note that our trainable weight are W_a and W_b
                weight_data = torch.zeros(out_features, in_features, device=device, dtype=dtype, requires_grad=False)

            if quantize is None:
                self.weight = nn.Parameter(weight_data, requires_grad=False)
            elif quantize == "4bit":
                self.weight = bnb.nn.Params4bit(
                    weight_data,
                    requires_grad=False,
                    compress_statistics=bnb_4bit_use_double_quant,
                    quant_type=bnb_4bit_quant_type,
                )
            elif quantize == "8bit":
                self.weight = bnb.nn.Int8Params(
                self.weight = bnb.nn.Int8Params(
                    weight_data,
                    requires_grad=False,
                )
            else:
                raise ValueError(f"Unknown quantize type: {quantize}")


        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        self.lora_only = lora_only
        self.trainable_scaling = trainable_scaling
        self.quantize = quantize
        self.module_name = module_name
        self.previous_weight = None
        self.weight_converged = False
        self.lora_stabilized = False
        self.convergence_threshold = convergence_threshold
        self.check_convergence = check_convergence
        self.convergence_history = []
        self.convergence_window = convergence_window
        self.previous_lora_product = None
        self.lora_change_threshold = lora_change_threshold
        self.steps_since_last_check = 0
        self.check_frequency = lora_check_frequency
        self.max_steps_before_reset = max_steps_before_reset

        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            self.lora_B = nn.Linear(r, out_features, bias=False)
            nn.init.zeros_(self.lora_B.weight)
            if trainable_scaling:
                self.scaling = nn.Parameter(torch.tensor([1.]), requires_grad=True)
            else:
                self.scaling = self.lora_alpha / self.r

            # Freezing the pre-trained weight matrix
            if not self.lora_only:
                self.weight.requires_grad = False 

    def _post_lora_scale(self):
        if self.trainable_scaling:
            return self.scaling.tanh()

        return self.scaling


    @torch.no_grad()
    def merge_check_and_reinit(self):
        if self.lora_only or self.weight_converged:
            return

        if hasattr(self, 'lora_stabilized'):
            self.lora_stabilized = False

        module_info = f"Module {self.module_name}" if self.module_name else "Module"

        new_weight = self.weight.data.clone()
        if self.r > 0:
            new_weight += self.lora_B.weight @ self.lora_A.weight * self._post_lora_scale() # merge
    
        if self.previous_weight is not None:
            weight_change = torch.norm(new_weight - self.previous_weight) / torch.norm(self.previous_weight)
            self.convergence_history.append(weight_change.item())
            
            if len(self.convergence_history) > self.convergence_window:
                self.convergence_history.pop(0)
            
            if len(self.convergence_history) == self.convergence_window:
                avg_change = sum(self.convergence_history) / len(self.convergence_history)
                logger.info(f"{module_info} average weight change: {avg_change:.6f}")
                
                if avg_change < self.convergence_threshold:
                    logger.info(f"{module_info} converged")
                    self.weight_converged = True
                    del self.lora_A
                    del self.lora_B
                    if hasattr(self, 'scaling'):
                        del self.scaling
                    if hasattr(self, 'previous_weight'):
                        del self.previous_weight
                    if hasattr(self, 'convergence_history'):
                        del self.convergence_history
                    torch.cuda.empty_cache() 
                    return 
        
        self.weight.data = new_weight # merge
        self.previous_weight = new_weight.clone()
        
        if not self.weight_converged:
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5)) # reinit
            nn.init.zeros_(self.lora_B.weight)
            if self.trainable_scaling:
                nn.init.zeros_(self.scaling)
    
    def forward(self, x: torch.Tensor):
        if self.lora_only:
            return self.lora_B(self.lora_A(self.lora_dropout(x))) * self._post_lora_scale()
        
        if self.quantize == "4bit":
            result = bnb.matmul_4bit(x, self.weight.t(), bias=self.bias, quant_state=self.weight.quant_state)
        elif self.quantize == "8bit":
            result = bnb.matmul(x, self.weight.t(), bias=self.bias, quant_state=self.weight.quant_state)
        else:
            result = F.linear(x, self.weight, bias=self.bias)
        
        if self.r > 0 and not self.weight_converged:
            result += self.lora_B(self.lora_A(self.lora_dropout(x))) * self._post_lora_scale()
        
        return result

    def check_lora_change(self):
        if self.weight_converged or self.lora_stabilized:
            return True

        self.steps_since_last_check += 1
        
        if self.steps_since_last_check >= self.max_steps_before_reset:
            self.steps_since_last_check = 0
            return True
        if self.steps_since_last_check % self.check_frequency != 0:
            return False
            
        with torch.no_grad():
            current_product = torch.mm(self.lora_B.weight, self.lora_A.weight)
            
            if self.previous_lora_product is None:
                self.previous_lora_product = current_product.clone()
                return False
                
            diff = torch.norm(current_product.sub_(self.previous_lora_product))
            norm = torch.norm(self.previous_lora_product)
            relative_change = diff / (norm + 1e-9)
            
            self.previous_lora_product.copy_(current_product)
            
            if relative_change < self.lora_change_threshold:
                self.lora_stabilized = True
                self.steps_since_last_check = 0
                del current_product
                torch.cuda.empty_cache()
                return True
            
            del current_product
            return False
