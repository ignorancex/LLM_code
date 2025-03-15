from omegaconf import ListConfig, OmegaConf
import torch
import os
from . import gpu


def path_join(root, model_dir):
    if isinstance(model_dir, (list, ListConfig)):
        return [os.path.join(root, i) for i in model_dir]
    return os.path.join(root, model_dir)


OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("get_device_count", lambda: torch.cuda.device_count())
OmegaConf.register_new_resolver("path_join", path_join)
OmegaConf.register_new_resolver("gpu_total_memory", lambda: gpu.query_gpu_memory()[0])
