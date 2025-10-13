import time
import warnings
import torch
import hydra  # Add this import
import torch.nn as nn

import torch.distributed as dist
from pathlib import Path
from omegaconf import DictConfig
from pytorch_lightning import Callback
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only
from typing import List
import logging
log = logging.getLogger(__name__)
from os import listdir


class ContrastiveLearning(nn.Module):
    def __init__(self, feature_dim, queue_size=15360, momentum=0.999, temperature=0.02):
        super(ContrastiveLearning, self).__init__()
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature

        self.register_buffer("queue", torch.randn(feature_dim, queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)

        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels




class VisualPrompt(nn.Module):
    def __init__(self, sim_head, clip_state_dict, T):
        super().__init__()
        self.sim_header = sim_head
        self.T = T
        assert sim_head in ["meanP", "LSTM", "Transf", "Conv_1D", "Transf_cls"]

        if (
            self.sim_header == "LSTM"
            or self.sim_header == "Transf"
            or self.sim_header == "Transf_cls"
            or self.sim_header == "Conv_1D"
        ):
            embed_dim = clip_state_dict["text_projection"].shape[1]

            context_length = clip_state_dict["positional_embedding"].shape[0]
            vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
            transformer_width = clip_state_dict["ln_final.weight"].shape[0]
            transformer_heads = transformer_width // 64

            transformer_layers = len(
                set(
                    k.split(".")[2]
                    for k in clip_state_dict
                    if k.startswith(f"transformer.resblocks")
                )
            )

            self.frame_position_embeddings = nn.Embedding(context_length, embed_dim)
        if self.sim_header == "Transf":
            self.transformer = TemporalTransformer(
                width=embed_dim, layers=6, heads=transformer_heads
            )
        if self.sim_header == "LSTM":
            self.lstm_visual = nn.LSTM(
                input_size=embed_dim,
                hidden_size=embed_dim,
                batch_first=True,
                bidirectional=False,
                num_layers=1,
            )

        self.apply(self.init_weights)

        if self.sim_header == "Transf_cls":
            self.transformer = TAggregate(
                clip_length=self.T, embed_dim=embed_dim, n_layers=6
            )

        if self.sim_header == "Conv_1D":
            self.shift = nn.Conv1d(
                embed_dim, embed_dim, 3, padding=1, groups=embed_dim, bias=False
            )
            weight = torch.zeros(embed_dim, 1, 3)
            weight[: embed_dim // 4, 0, 0] = 1.0
            weight[embed_dim // 4 : embed_dim // 4 + embed_dim // 2, 0, 1] = 1.0
            weight[-embed_dim // 4 :, 0, 2] = 1.0
            self.shift.weight = nn.Parameter(weight)




def gen_label(labels):
    num = len(labels)
    gt = np.zeros(shape=(num, num))
    for i, label in enumerate(labels):
        for k in range(num):
            if labels[k] == label:
                gt[i, k] = 1
    return gt
# Logging setup: get_pylogger
def get_pylogger(name: str) -> logging.Logger:
    """
    Creates and returns a logger with the specified name.
    
    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # If the logger has not been configured before, configure it
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        
        # Create a console handler with a specific log format
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # Add the handler to the logger
        logger.addHandler(console_handler)
    
    return logger

# Utility function to instantiate callbacks from configuration
def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config."""
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("Callbacks config is empty.")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks

# Utility function to instantiate loggers from configuration
def instantiate_loggers(logger_cfg: DictConfig) -> List[LightningLoggerBase]:
    """Instantiates loggers from config."""
    loggers: List[LightningLoggerBase] = []

    if not logger_cfg:
        log.warning("Logger config is empty.")
        return loggers

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            loggers.append(hydra.utils.instantiate(lg_conf))

    return loggers






def convert_32(model: nn.Module):
    def _convert_weights_to_fp32(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.float()
            if l.bias is not None:
                l.bias.data = l.bias.data.float()

        if isinstance(l, nn.MultiheadAttention):
            for attr_name in [
                *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                "in_proj_bias",
                "bias_k",
                "bias_v",
            ]:
                if hasattr(l, attr_name):
                    attr = getattr(l, attr_name)
                    if attr is not None and isinstance(attr, torch.Tensor):
                        attr.data = attr.data.float()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None and isinstance(attr, torch.Tensor):
                    attr.data = attr.data.float()

    model.apply(_convert_weights_to_fp32)    




def create_logits(x1, x2, logit_scale):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_x1 = logit_scale * x1 @ x2.t()
    logits_per_x2 = logit_scale * x2 @ x1.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1, logits_per_x2



def compute_entropy(probs):

        
    return -torch.sum(probs * torch.log(probs + 1e-10), dim=1) 

def get_classes(data):
    ek_map = {
        "opening": 2,
        "taking": 0,
        "closing": 3,
        "putting": 1,
        "washing": 4,
        "pouring": 7,
        "mixing": 6,
        "cutting": 5,
    }

    hu_map = {
        "climb": 0,
        "fencing": 1,
        "golf": 2,
        "kick ball": 3,
        "pullup": 4,
        "punch": 5,
    }

    daily_da_map = {
    "drinking": 0,
    "jumping": 1,
    "picking": 2,
    "pouring": 3,
    "pushing": 4,
    "running": 5,
    "walking": 6,
    "waving": 7,
    }

    hmdb_ucf_map = {
        "climb": 0,
        "fencing": 1,
        "golf": 2,
        "kick ball": 3,
        "pullup": 4,
        "punch": 5,
        "pushup": 6,
        "ride bike": 7,
        "ride horse": 8,
        "shoot ball": 9,
        "shoot bow": 10,
        "walk": 11,
    }

    sports_da_map = {
        "archery": 0,
        "baseball": 1,
        "basketball": 2,
        "biking": 3,
        "bowling": 4,
        "breast stroke": 5,
        "diving": 6,
        "fencing": 7,
        "hockey": 8,
        "floor gymnastics": 9,
        "golfing": 10,
        "horseback riding": 11,
        "kayaking": 12,
        "rock climbing": 13,
        "rope climbing": 14,
        "skateboarding": 15,
        "skiing": 16,
        "sumo wrestling": 17,
        "surfing": 18,
        "taichi": 19,
        "tennis": 20,
        "trampoline jumping": 21,
        "volleyball": 22,
    }

    hmdb51_map = None
    if data["data_folder"] is not None:
        hmdb51_map = {
            c.replace("_", " "): i
            for i, c in enumerate(sorted(listdir(data["data_folder"])))
        }

    uo_map = {"basketball": 0, "clean and jerk": 1, "throw discus": 2}

    if "ek" in data["dataset"]:
        res = [[i, c] for c, i in sorted(ek_map.items(), key=lambda x: x[1])]
    elif data["dataset"] in [
        "hmdb_ucf",
        "ucf_hmdb",
        "hmdb_ucf_im2vid",
        "ucf_hmdb_im2vid",
    ]:
        res = [[i, c] for c, i in hmdb_ucf_map.items()]
    elif data["dataset"] in [
        "kinetics_hmdb",
        "hmdb_kinetics",
        "kinetics_arid",
        "arid_kinetics",
        "hmdb_arid",
        "arid_hmdb",
        "hmdb_mit",
        "mit_hmdb",
        "kinetics_mit",
        "arid_mit",
        "mit_kinetics",
        "mit_arid",
    ]:
        res = [[i, c] for c, i in daily_da_map.items()]
    elif data["dataset"] in [
        "ucf_kinetics",
        "ucf_sports",
        "kinetics_ucf",
        "kinetics_sports",
        "sports_kinetics",
        "sports_ucf",
    ]:
        res = [[i, c] for c, i in sports_da_map.items()]
    elif "olympic" in data["dataset"]:
        res = [[i, c] for c, i in uo_map.items()]
    elif data["dataset"] == "hmdb51":
        res = [[i, c] for c, i in sorted(hmdb51_map.items(), key=lambda x: x[1])]
    else:
        folder = data["train_file"]
        classes = sorted(listdir(folder))
        res = [[i, c] for i, c in enumerate(classes)]

    # if data["classes_limit"] > 0:
    #     res = res[: data["classes_limit"]]

 
    return res
# Save a file only on the main process
@rank_zero_only
def save_file(path: str, content: str) -> None:
    """Save file in rank zero mode (only on one process in multi-GPU setup)."""
    with open(path, "w+") as file:
        file.write(content)

# Function to get the rank of the current process
def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0

# A simple example for task wrapping with timing and exception logging
def task_wrapper(task_func):
    """Decorator for wrapping tasks with utilities like timing and exception logging."""
    def wrap(cfg: DictConfig):
        start_time = time.time()

        try:
            metric_dict, object_dict = task_func(cfg)
        except Exception as ex:
            log.exception("An error occurred during the task execution.")
            raise ex
        finally:
            exec_time = time.time() - start_time
            log.info(f"Task executed in {exec_time:.2f} seconds.")

        return metric_dict, object_dict

    return wrap

# Example utility function to log hyperparameters
@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Log hyperparameters for easy access and tracking."""
    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # Save the number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hparams["model/params/non_trainable"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    hparams["datamodule"] = cfg["datamodule"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # Send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)

# Extras function to apply optional utilities like warnings and config printing
def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started."""
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # Disable Python warnings if specified
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling Python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # Additional utilities can be added here
