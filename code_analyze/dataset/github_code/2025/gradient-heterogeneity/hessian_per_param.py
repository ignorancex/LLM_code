import json
import logging
import os
import sys
from contextlib import ExitStack, nullcontext
from functools import partialmethod

import numpy as np
import torch
import tqdm
import yaml

from pyhessian.hessian_nlp import hessian_nlp
from pyhessian.hessian_vision import hessian_vision
from utils_nlp.run_utils_nlp import get_my_last_checkpoint, prepare_args, setup_library
from utils_nlp.tasks.get_trainer import get_trainer
from utils_shared.run_utils_shared import parse_cli_args
from utils_vision.run_utils_vision import setup

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_TO_PARAM_PLOT = {
    "roberta-base": [
        "roberta.encoder.layer.0.attention.output.LayerNorm.weight",
        "roberta.encoder.layer.0.attention.self.query.weight",
        "roberta.encoder.layer.0.attention.self.key.weight",
        "roberta.encoder.layer.0.attention.self.value.weight",
        "roberta.encoder.layer.0.intermediate.dense.weight",
        "roberta.encoder.layer.0.output.LayerNorm.weight",
        "roberta.encoder.layer.0.output.dense.weight",
        "roberta.encoder.layer.6.attention.output.LayerNorm.weight",
        "roberta.encoder.layer.6.attention.self.query.weight",
        "roberta.encoder.layer.6.attention.self.key.weight",
        "roberta.encoder.layer.6.attention.self.value.weight",
        "roberta.encoder.layer.6.intermediate.dense.weight",
        "roberta.encoder.layer.6.output.LayerNorm.weight",
        "roberta.encoder.layer.6.output.dense.weight",
        "roberta.encoder.layer.11.attention.output.LayerNorm.weight",
        "roberta.encoder.layer.11.attention.self.query.weight",
        "roberta.encoder.layer.11.attention.self.key.weight",
        "roberta.encoder.layer.11.attention.self.value.weight",
        "roberta.encoder.layer.11.intermediate.dense.weight",
        "roberta.encoder.layer.11.output.LayerNorm.weight",
        "roberta.encoder.layer.11.output.dense.weight",
        "classifier.out_proj.weight",
    ],
    "resnet18": [
        "layer1.0.conv1.weight",
        "layer1.0.bn1.weight",
        "layer1.0.conv2.weight",
        "layer1.1.conv1.weight",
        "layer1.1.bn1.weight",
        "layer1.1.conv2.weight",
        "layer3.0.conv1.weight",
        "layer3.0.bn1.weight",
        "layer3.0.conv2.weight",
        "layer3.0.downsample.0.weight",
        "layer3.1.conv1.weight",
        "layer3.1.bn1.weight",
        "layer3.1.conv2.weight",
        "layer4.0.conv1.weight",
        "layer4.0.bn1.weight",
        "layer4.0.conv2.weight",
        "layer4.0.downsample.0.weight",
        "layer4.1.conv1.weight",
        "layer4.1.bn1.weight",
        "layer4.1.conv2.weight",
        "fc.weight",
    ],
    "vit-base": [
        "encoder.layers.encoder_layer_0.ln_1.weight",
        "encoder.layers.encoder_layer_0.ln_2.weight",
        "encoder.layers.encoder_layer_0.self_attention.in_proj_weight",
        "encoder.layers.encoder_layer_0.mlp.0.weight",
        "encoder.layers.encoder_layer_0.mlp.3.weight",
        "encoder.layers.encoder_layer_6.ln_1.weight",
        "encoder.layers.encoder_layer_6.ln_2.weight",
        "encoder.layers.encoder_layer_6.self_attention.in_proj_weight",
        "encoder.layers.encoder_layer_6.mlp.0.weight",
        "encoder.layers.encoder_layer_6.mlp.3.weight",
        "encoder.layers.encoder_layer_11.ln_1.weight",
        "encoder.layers.encoder_layer_11.ln_2.weight",
        "encoder.layers.encoder_layer_11.self_attention.in_proj_weight",
        "encoder.layers.encoder_layer_11.mlp.0.weight",
        "encoder.layers.encoder_layer_11.mlp.3.weight",
        "heads.head.weight",
    ],
}


def calculate_hessian_per_param(model, data_loader, model_name, hessian_class):
    criterion = torch.nn.CrossEntropyLoss()
    model = model.to(device)
    eigenvalues_and_grads = np.zeros(
        (len(data_loader), len(MODEL_TO_PARAM_PLOT[model_name]), 2)
    )
    ctx = (
        nullcontext()
        if model_name in ["resnet18"]
        else torch.nn.attention.sdpa_kernel(
            [
                torch.nn.attention.SDPBackend.MATH,
            ]
        )
    )
    for id_batch, batch in enumerate(data_loader):
        with ExitStack() as stack:
            if model_name in ["resnet18"]:
                stack.enter_context(nullcontext())
            else:
                stack.enter_context(
                    torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH])
                )
            for id_param, param_name in enumerate(MODEL_TO_PARAM_PLOT[model_name]):
                hessian_comp = hessian_class(
                    model,
                    criterion,
                    data=batch,
                    cuda=torch.cuda.is_available(),
                    param_name=param_name,
                )
                top_eigenvalue, _ = hessian_comp.eigenvalues(get_maximum=True, top_n=5)
                eigenvalues_and_grads[id_batch, id_param, 0] = top_eigenvalue
                eigenvalues_and_grads[id_batch, id_param, 1] = torch.linalg.norm(
                    hessian_comp.gradsH[0]
                ).item()

    return eigenvalues_and_grads  # shape: (num_batches, num_params, 2)


def main():
    cli_args = parse_cli_args()
    hessian_dir = os.path.join("results", "hessian_peer_param")
    model_dir_dict_path = os.path.join(
        hessian_dir, "model_dir_dict.json"
    )  # Path to dictionary of trained model directory to load
    if not os.path.exists(model_dir_dict_path):
        raise ValueError(f"{model_dir_dict_path} not found")
    with open(
        model_dir_dict_path,
        "r",
    ) as f:
        model_dir_dict = json.load(f)
    training_mode_or_pretrained = cli_args.training_mode
    if cli_args.training_mode != "pretrained":
        dict_key = (
            f"{cli_args.model_name}_{cli_args.dataset_name}_{cli_args.optimizer_name}"
        )
        if dict_key not in model_dir_dict:
            raise ValueError(f"Model dir not found for {dict_key}")
        model_dir_to_load = model_dir_dict[dict_key]
        seed = int(model_dir_to_load.split("_")[-1])
        cli_args.seed = seed
    else:
        cli_args.training_mode = "normal"
    dir_to_save = os.path.join(
        hessian_dir,
        cli_args.domain,
        cli_args.model_name,
        cli_args.dataset_name,
        cli_args.optimizer_name
        if training_mode_or_pretrained != "pretrained"
        else "pretrained",
    )
    os.makedirs(dir_to_save, exist_ok=True)
    if os.path.exists(os.path.join(dir_to_save, "hessian_gradient_per_param.npy")):
        logger.info(f"Already calculated for {dir_to_save}")
        return
    if cli_args.domain == "nlp":
        with open(cli_args.arguments, "r") as f:
            config = yaml.safe_load(f)
        model_args, data_args, training_args = prepare_args(config, cli_args)
        training_args = setup_library(training_args)
        trainer, _ = get_trainer((model_args, data_args, training_args))
        train_dataloader = trainer.get_train_dataloader()
        if training_mode_or_pretrained != "pretrained":
            last_checkpoint = get_my_last_checkpoint(model_dir_to_load)
            logger.info(f"Loding from {last_checkpoint}")
            trainer._load_from_checkpoint(last_checkpoint)
        model = trainer.model
        hessian_class = hessian_nlp
    elif cli_args.domain == "vision":
        model, _, data_loaders, _, _ = setup(cli_args)
        train_dataloader = data_loaders["train"]
        if training_mode_or_pretrained != "pretrained":
            checkpoint = torch.load(os.path.join(model_dir_to_load, "checkpoint.pth"))
            model.load_state_dict(checkpoint["state_dict_last"])
        hessian_class = hessian_vision
    else:
        raise ValueError(f"Domain {cli_args.domain} not supported")
    logger.info("Start Calculating Hessian")
    eigenvalues_and_grads = calculate_hessian_per_param(
        model, train_dataloader, cli_args.model_name, hessian_class
    )
    with open(os.path.join(dir_to_save, "hessian_gradient_per_param.npy"), "wb") as f:
        np.save(f, eigenvalues_and_grads)
    logger.info(f"Saved at {dir_to_save}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    _default_log_level = logging.INFO
    logger.setLevel(_default_log_level)

    main()
