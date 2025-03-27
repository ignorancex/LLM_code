"""This module defines the required functions to setup the optimizers for each
experiment type with the LM."""

from typing import Dict, Optional

import torch
from absl import flags
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer

FLAGS = flags.FLAGS
flags.DEFINE_float("learning_rate", 0.005, "The learning rate used in the optimizer", lower_bound=0.0)
flags.DEFINE_float("weight_decay_rate", 0.0, "The weight decay rate used in the optimizer.", lower_bound=0.0)

OPTIMIZER_ARGS_TYPE = Dict[str, torch.nn.Module]


def construct_optimizer(model: torch.nn.Module, second_model: Optional[torch.nn.Module] = None) -> Optimizer:
    """Define the AdamW optimizer over the parameters."""

    params = list(model.parameters())
    if second_model is not None:
        # concatinate the second module parameters and register in the optimizer.
        params += list(second_model.parameters())

    optimizer = AdamW(params, lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay_rate, amsgrad=True, fused=True)

    return optimizer


def all_weights_opt(opt_args: OPTIMIZER_ARGS_TYPE, lm_type: str) -> Optimizer:
    """Define the optimizer that fine-tunes all the weights in the model."""
    return construct_optimizer(model=opt_args[f"{lm_type}_model"])


def input_embeddings_opt(opt_args: OPTIMIZER_ARGS_TYPE, lm_type: str) -> Optimizer:
    """Define the optimizer that only fine-tunes the shared input embedding
    layer of the roberta model."""

    model: torch.nn.Module = None
    if lm_type == "roberta":
        model = opt_args["roberta_model"]
        for name, param in model.named_parameters():
            name = name.removeprefix("_orig_mod.")  # remove prefix of compiled model.
            if name in [
                "roberta.embeddings.word_embeddings.weight",
                "roberta.embeddings.LayerNorm.weight",
                "roberta.embeddings.LayerNorm.bias",
            ]:
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif lm_type == "t5":
        model = opt_args["t5_model"]
        for name, param in model.named_parameters():
            if name == "shared.weight":
                param.requires_grad = True
            else:
                param.requires_grad = False

    return construct_optimizer(model=model)


def output_embeddings_opt(opt_args: OPTIMIZER_ARGS_TYPE, lm_type: str) -> Optimizer:
    """Define the optimizer that only fine-tunes the output LM head of the
    roberta model."""

    model: torch.nn.Module = None
    if lm_type == "roberta":
        model = opt_args["roberta_model"]
        for name, param in model.named_parameters():
            name = name.removeprefix("_orig_mod.")  # remove prefix of compiled model.
            if name in [
                "lm_head.dense.weight",
                "lm_head.dense.bias",
                "lm_head.bias",
                "lm_head.layer_norm.weight",
                "lm_head.layer_norm.bias",
            ]:
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif lm_type == "t5":
        model = opt_args["t5_model"]
        for name, param in model.named_parameters():
            if name == "lm_head.weight":
                param.requires_grad = True
            else:
                param.requires_grad = False

    return construct_optimizer(model=model)


def no_weights_opt(opt_args: OPTIMIZER_ARGS_TYPE, lm_type: str) -> Optimizer:
    """Define the optimizer that does not fine-tune any weights."""

    model: torch.nn.Module = opt_args[f"{lm_type}_model"]
    # don't waste time storing grad data.
    for _, param in model.named_parameters():
        param.requires_grad = False

    return construct_optimizer(model=model)


def lora_opt(opt_args: OPTIMIZER_ARGS_TYPE, lm_type: str) -> Optimizer:
    """Define the optimizer that does not change the parameters defined by
    lora."""
    model: torch.nn.Module = opt_args[f"{lm_type}_model"]
    print(f"{lm_type}_model")
    return construct_optimizer(model=model)


def classifier_model_opt(opt_args: OPTIMIZER_ARGS_TYPE, lm_type: str) -> Optimizer:
    """Define the optimizer that only fine-tunes the classifier on top of the
    LM for the downstream task."""

    model: torch.nn.Module = opt_args[f"{lm_type}_model"]
    # don't waste time storing grad data.
    for _, param in model.named_parameters():
        param.requires_grad = False

    return construct_optimizer(model=model, second_model=opt_args["classifier_model"])


def prompt_model_opt(opt_args: OPTIMIZER_ARGS_TYPE, lm_type: str) -> Optimizer:
    """Define the optimizer that only fine-tunes the prompt vectors on the
    downstream task."""

    model: torch.nn.Module = None
    if lm_type == "roberta":
        model = opt_args["roberta_model"]
        for name, param in model.named_parameters():
            name = name.removeprefix("_orig_mod.")  # remove prefix of compiled model.
            if name == "roberta.embeddings.word_embeddings.prompt_embedder.weight":
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif lm_type == "t5":
        model = opt_args["t5_model"]
        for name, param in model.named_parameters():
            if name == "shared.prompt_embedder.weight":
                param.requires_grad = True
            else:
                param.requires_grad = False

    return construct_optimizer(model=model)


def define_optimizer(exp_type: str, model_pool: OPTIMIZER_ARGS_TYPE, lm_type: str) -> Optimizer:
    if exp_type == "all_finetune":
        return all_weights_opt(opt_args=model_pool, lm_type=lm_type)
    elif exp_type == "input_finetune":
        return input_embeddings_opt(opt_args=model_pool, lm_type=lm_type)
    elif exp_type == "output_finetune":
        return output_embeddings_opt(opt_args=model_pool, lm_type=lm_type)
    elif exp_type == "no_finetune":
        return no_weights_opt(opt_args=model_pool, lm_type=lm_type)
    elif exp_type == "classifier_finetune":
        return classifier_model_opt(opt_args=model_pool, lm_type=lm_type)
    elif exp_type == "soft_prompt_finetune":
        return prompt_model_opt(opt_args=model_pool, lm_type=lm_type)
    elif exp_type == "lora_finetune":
        return lora_opt(opt_args=model_pool, lm_type=lm_type)
    else:
        raise ValueError(f"Wrong {exp_type}.")
