CUDA_LAUNCH_BLOCKING = 1
import time

import torch
import random
import torch.nn as nn
from typing import List, Optional, Tuple, Union

# from vector_quantizer import *
import tqdm

# from quant import *
import random
import numpy as np
import os
import sys

if __name__ == "__main__":
    print(os.getcwd())
    sys.path.append(os.getcwd())


import yaml
import src.compression_parent as compression_parent
from src.utils.model_utils import find_layers, get_llama, inference_layer
import src.data as data
import src.utils.utils as utils


try:
    import wandb

    has_wandb = True
except:
    has_wandb = False


@torch.no_grad()
def generate_hessians(model, dataloader, dev):
    print("Starting...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=dev if not args.offload_activations else "cpu",
    )
    # ========== Preprocessing the data ==========
    train_cache = {"i": 0, "kwargs": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            # print(kwargs)
            # raise Exception("stop")
            inps[train_cache["i"]] = inp if not args.offload_activations else inp.cpu()
            train_cache["i"] += 1
            train_cache["kwargs"] = kwargs
            raise ValueError

    layers[0] = Catcher(layers[0])
    with torch.no_grad():
        for batch in tqdm.tqdm(
            dataloader, desc="getting inputs", miniters=len(dataloader) // 100
        ):
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)

    kwargs = train_cache["kwargs"]
    free, total = torch.cuda.mem_get_info(int(dev.split(":")[1]))
    print(free // 1024**2, "MiB free out of", total // 1024**2, "MiB total")

    # raise Exception("stop")

    for name in kwargs:
        if isinstance(kwargs[name], torch.Tensor):
            kwargs[name] = kwargs[name].to(dev)
            # print(name, kwargs[name].device, kwargs[name].dtype, kwargs[name].shape)

    import gc

    # data_ptrs = []
    # for obj in gc.get_objects():
    #     try:
    #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
    #             if obj.device == dev:
    #                 data_ptrs.append(obj.data_ptr())
    #     except:
    #         pass

    print("Ready.")

    total_bits = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        # print(layer)
        layer_dtype_orig = next(layer.parameters()).dtype
        print("layer original dtype", layer_dtype_orig)
        #     if i > 0:
        #         return

        full = find_layers(layer)

        sequential = [list(full.keys())]

        for l, names in enumerate(sequential):

            for name in names:
                parent_module = getattr(layer, name.split(".")[0])
                sublayer: nn.Linear = getattr(parent_module, name.split(".")[1])
                print("sublayer", sublayer)

                if args.weight_save_path:
                    weight_save_path = os.path.join(
                        args.weight_save_path, f"layer_{i}/{name}.pt"
                    )
                    os.makedirs(os.path.dirname(weight_save_path), exist_ok=True)
                    print("saving weights to", weight_save_path)
                    torch.save(
                        {"weight": sublayer.weight, "bias": sublayer.bias},
                        weight_save_path,
                    )
                    # continue

                new_layer = compression_parent.CompressedLinear(
                    weight=sublayer.weight, bias=sublayer.bias
                )
                if args.hessian_save_path:
                    new_layer.enable_hessian_logging()
                if args.hessianDiag_save_path:
                    new_layer.enable_hessianDiag_logging()
                new_layer.to(dev)
                new_layer.to(sublayer.weight.dtype)

                # delete the old layer
                delattr(parent_module, name.split(".")[1])
                setattr(parent_module, name.split(".")[1], new_layer)

            # garbage collect
            torch.cuda.empty_cache()

            if args.hessian_save_path or args.hessianDiag_save_path:

                # pass the inputs through the models:
                outs = inference_layer(
                    layer,
                    inps,
                    outs,
                    layer_kwargs=kwargs,
                    dev=dev,
                    offload_activations=args.offload_activations,
                    batch_size=args.forward_pass_batch_size,
                )

                for name in names:
                    new_layer = getattr(
                        getattr(layer, name.split(".")[0]), name.split(".")[1]
                    )

                    if args.hessian_save_path:
                        save_path = os.path.join(
                            args.hessian_save_path, f"layer_{i}/{name}.pt"
                        )
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        torch.save({"hessian": new_layer.hessian}, save_path)
                    if args.hessianDiag_save_path:
                        save_path = os.path.join(
                            args.hessianDiag_save_path, f"layer_{i}/{name}.pt"
                        )
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        torch.save({"hessianDiag": new_layer.hessianDiag}, save_path)

                    new_layer.clean()

        free, total = torch.cuda.mem_get_info(int(dev.split(":")[1]))
        print(free // 1024**2, "MiB free out of", total // 1024**2, "MiB total")

        layer.to(torch.device("cpu"))

        del layer
        torch.cuda.empty_cache()

        # print("stop after first layer", stop_after_first_layer)
        # if stop_after_first_layer:
        #     return

        print("after cleaning up", i)
        free, total = torch.cuda.mem_get_info(int(dev.split(":")[1]))
        print(free // 1024**2, "MiB free out of", total // 1024**2, "MiB total")

        # print(torch.cuda.memory_summary(device=args.device, abbreviated=False))

        # import gc
        # for obj in gc.get_objects():
        #     try:
        #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #             if obj.device == dev:
        #                 print(type(obj), obj.shape)
        #             # if obj.data_ptr() not in data_ptrs:
        #             #     print(type(obj), obj.shape)
        #             # print(type(obj), obj.shape)
        #     except:
        #         pass

        # raise Exception("stop")

        inps, outs = outs, inps
        # return
        # break
        print(
            "Done with layer",
            i,
            "total_time elapsed:",
            round(time.time() - tick),
            "estimated time left:",
            round((time.time() - tick) * (len(layers) - i - 1) / (i + 1)),
        )
    model.config.use_cache = use_cache

    # print("Total bits:", total_bits, "Total params:", total_params)
    # print("average bits per value:", total_bits / total_params)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str, help="LlaMA model to load")
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4", "pajama"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to run on."
    )
    parser.add_argument("--seqlen", type=int, default=-1, help="Sequence length.")
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--forward_pass_batch_size",
        type=int,
        default=4,
        help="Batch size for forward pass, parallel process these many sequences.",
    )
    parser.add_argument(
        "--hessian_save_path", type=str, default=None, help="Path to save hessians."
    )
    parser.add_argument(
        "--hessianDiag_save_path",
        type=str,
        default=None,
        help="Path to save hessian diagonals.",
    )
    parser.add_argument(
        "--weight_save_path", type=str, default=None, help="Path to save weights."
    )
    parser.add_argument(
        "--offload_activations",
        action="store_true",
        help="Offload activations to CPU to save memory.",
    )
    args = parser.parse_args()
    # init W&B logging

    model = get_llama(args.model)
    model.seqlen = (
        args.seqlen if args.seqlen > 0 else model.config.max_position_embeddings
    )
    # for either hessian_save_path, and hessianDiag_save_path, if the seqlen is left as {seqlen} fill with the seqlen
    if args.hessian_save_path is not None:
        args.hessian_save_path = args.hessian_save_path.format(seqlen=model.seqlen)
    if args.hessianDiag_save_path is not None:
        args.hessianDiag_save_path = args.hessianDiag_save_path.format(
            seqlen=model.seqlen
        )

    print("seqlen", model.seqlen)
    model.eval()
    model.to("cpu")
    # print("n samples val", args.nsamples_val)
    # raise Exception("stop")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.hessianDiag_save_path:
        os.makedirs(args.hessianDiag_save_path, exist_ok=True)
        # save the args as a yaml file
        with open(os.path.join(args.hessianDiag_save_path, "args.yaml"), "w") as f:
            yaml.dump(vars(args), f)
    if args.hessian_save_path:
        os.makedirs(args.hessian_save_path, exist_ok=True)
        # save the args as a yaml file
        with open(os.path.join(args.hessian_save_path, "args.yaml"), "w") as f:
            yaml.dump(vars(args), f)

    trainloader = data.get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
        train_test="train",
    )
    tick = time.time()
    n_params = sum(p.numel() for p in model.parameters())
    generate_hessians(model, trainloader, args.device)
    print("total time taken:", time.time() - tick)
