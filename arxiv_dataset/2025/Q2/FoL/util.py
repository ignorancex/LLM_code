# -*- coding: UTF-8 -*-
import torch
from os.path import join
import shutil

def resume_model(args, model):
    checkpoint = torch.load(args.resume, map_location=args.device)
    # If the checkpoint contains nested model state dictionary
    if 'model_state_dict' in checkpoint:
        # Load the nested model state dictionary
        model_state_dict = checkpoint['model_state_dict']
        # Remove "module." prefix from keys
        model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
    else:
        # If not nested, assume model state dict is directly in the checkpoint
        model_state_dict = checkpoint
        # Remove "module." prefix from keys
        model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
    # Load the model state dictionary
    model.load_state_dict(model_state_dict, strict=False)
    return model

def save_checkpoint(args, state, is_best, filename, recalls_rerank):
    model_path = join(args.save_dir, f"Rerank_R1_{recalls_rerank[0]:.2f}_R5_{recalls_rerank[1]:.2f}_{filename}")
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, join(args.save_dir, "best_R5_model.pth"))