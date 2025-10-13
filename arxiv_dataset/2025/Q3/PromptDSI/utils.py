import datetime
import logging
import math
import os
import random
import time
from collections import defaultdict, deque
from typing import Any, Dict

import numpy as np
import torch
import torch.distributed as dist
import yaml
from pynvml import *
from timm.optim import create_optimizer
from torch import nn
from torch.optim import SGD, AdamW
from torch.optim.optimizer import Optimizer
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)
BASE_KEY = "_BASE_"

def get_params_info(model):
    all_param = 0
    trainable_param = 0
    print("All trainable parameters:")
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_param += param.numel()
            print(name, param.numel())
    print_params_info(all_param, trainable_param)


def print_params_info(all_param, trainable_param):
    print(f" # all param       : {all_param}")
    print(f" # trainable param : {trainable_param}")
    print(f" # % trainable parameters: {trainable_param/all_param*100:.2f}%")


def get_scheduler(args, optimizer, K=None):
    if args.sched != "constant":
        print(f"Using linear warmup scheduler with {args.warmup_steps} warmup steps")
        return get_linear_schedule_with_warmup(optimizer, 100)


def get_optimizer(args, model):
    if args.opt == "adamw":
        optimizer = get_adamw(
            model,
            learning_rate=args.lr,
            adam_eps=args.opt_eps,
            weight_decay=args.weight_decay,
            key_lr=args.key_lr,
        )
    elif args.opt == "sgd":
        optimizer = SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = create_optimizer(args, model)
    return optimizer


def get_adamw(
    model: nn.Module,
    learning_rate: float = 1e-5,
    adam_eps: float = 1e-8,
    weight_decay: float = 0.0,
    key_lr: float = 1e-3,
) -> torch.optim.Optimizer:
    no_decay = ["bias", "LayerNorm.weight"]
    prompt_key = ["prompt_key", "prompt_keys"]  # larger learning rate
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
                and not any(
                    nd in n for nd in prompt_key
                )  # Enable to have different learning rate for prompt_key
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
        # Enable to have different learning rate for prompt_key
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in prompt_key)
            ],
            "weight_decay": 0.0,
            "lr": key_lr,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        eps=adam_eps,
        fused=True,
    )
    return optimizer


def get_model(
    args,
    device,
    original=False,
):
    logger.info(f"Loading model")

    # Original model for extracting CLS token
    if original:
        
        if args.sbert:
            from model.SbertModel import GeneralQueryClassifier

            logger.info("initializing all-mpnet-base-v2 model weight")
        elif "roberta" in args.model_encoder:
            from model.RobertaModel import GeneralQueryClassifier
            logger.info("initializing roberta-base model weight")
        else:
            from model.BertModel import GeneralQueryClassifier

            logger.info("initializing bert-base-uncased model weight")

        if args.sbert:
            original_model = GeneralQueryClassifier(args, question_model="incdsi_sbert")
        else:
            original_model = GeneralQueryClassifier(args, question_model="incdsi_bert")
        logger.info("original model loaded")

        # # Load the pre-trained bert model weights only
        # if args.load_weight_original:
        #     logger.info("initializing original model weight")
        #     load_saved_weights_original(original_model, args.original_model)
        # else:
        #     if args.sbert:
        #         logger.info("initializing all-mpnet-base-v2 model weight")
        #     else:
        #         logger.info("initializing bert-base-uncased model weight")

        for p in original_model.parameters():
            p.requires_grad = False
        original_model.to(device)

        return original_model


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


# def get_params_info(model):
#     all_param = 0
#     trainable_param = 0
#     print("Trainable parameters:")
#     for k, param in model.named_parameters():
#         all_param += param.numel()
#         if param.requires_grad:
#             trainable_param += param.numel()
#             print(k, param.numel())
#     return all_param, trainable_param


def load_config(config_path):
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config


def load_yaml_with_base(filename: str, allow_unsafe: bool = False) -> Dict[str, Any]:
    """
    Just like `yaml.load(open(filename))`, but inherit attributes from its
        `_BASE_`.

    Args:
        filename (str or file-like object): the file name or file of the current config.
            Will be used to find the base config file.
        allow_unsafe (bool): whether to allow loading the config file with
            `yaml.unsafe_load`.

    Returns:
        (dict): the loaded yaml
    """
    cfg = load_config(filename)

    def merge_a_into_b(a: Dict[str, Any], b: Dict[str, Any]) -> None:
        # merge dict a into dict b. values in a will overwrite b.
        for k, v in a.items():
            if isinstance(v, dict) and k in b:
                assert isinstance(
                    b[k], dict
                ), "Cannot inherit key '{}' from base!".format(k)
                merge_a_into_b(v, b[k])
            else:
                b[k] = v

    def _load_with_base(base_cfg_file: str) -> Dict[str, Any]:
        if base_cfg_file.startswith("~"):
            base_cfg_file = os.path.expanduser(base_cfg_file)
        if not any(map(base_cfg_file.startswith, ["/", "https://", "http://"])):
            # the path to base cfg is relative to the config file itself.
            base_cfg_file = os.path.join(os.path.dirname(filename), base_cfg_file)
        return load_yaml_with_base(base_cfg_file, allow_unsafe=allow_unsafe)

    if BASE_KEY in cfg:
        if isinstance(cfg[BASE_KEY], list):
            base_cfg: Dict[str, Any] = {}
            base_cfg_files = cfg[BASE_KEY]
            for base_cfg_file in base_cfg_files:
                merge_a_into_b(_load_with_base(base_cfg_file), base_cfg)
        else:
            base_cfg_file = cfg[BASE_KEY]
            base_cfg = _load_with_base(base_cfg_file)
        del cfg[BASE_KEY]

        merge_a_into_b(cfg, base_cfg)
        return base_cfg
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_saved_weights_original(model, model_path, load_classifier=True):
    # The current model and the model to initialize from can have different number of classes.
    state_dict = torch.load(model_path)
    del state_dict["classifier.weight"]
    model.load_state_dict(state_dict, strict=False)

    if load_classifier:
        state_dict = torch.load(model_path)
        model.classifier.weight.data[
            : len(state_dict["classifier.weight"])
        ] = state_dict["classifier.weight"]


def load_saved_weights(model, model_path):
    # The current model and the model to initialize from can have different number of classes.
    state_dict = torch.load(model_path, map_location="cpu")
    f = state_dict.get("f", 0)
    task_key_centroid = state_dict.get("task_key_centroid", None)
    if "model" in state_dict.keys():
        state_dict = state_dict["model"]

    # For torch.compile models
    # unwanted_prefix = "_orig_mod."
    # for k, v in list(state_dict.items()):
    #     if k.startswith(unwanted_prefix):
    #         state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    
    # For bert adapters
    # if "base_model_epoch" in model_path:
    #     for k, v in list(state_dict.items()):
    #         state_dict[k.replace("question_model.", "question_model.bert.")] = state_dict.pop(k)

    if state_dict["classifier.weight"].shape[0] > model.classifier.weight.data.shape[0]:
        print("state_dict has more classes than model")
        print(
            "Classifier weights of pre-trained model: ",
            state_dict["classifier.weight"].shape,
        )
        print(
            "Classifier weights of current model: ", model.classifier.weight.data.shape
        )
        state_dict["classifier.weight"] = state_dict["classifier.weight"][
            : model.classifier.weight.data.shape[0]
        ]
    else:
        print("state_dict has fewer classes than model")
        print(
            "Classifier weights of pre-trained model: ",
            state_dict["classifier.weight"].shape,
        )
        print(
            "Classifier weights of current model: ", model.classifier.weight.data.shape
        )
        model.classifier.weight.data[: len(state_dict["classifier.weight"])] = (
            state_dict["classifier.weight"]
        )
        del state_dict["classifier.weight"]

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print("missing:", missing)
    print("unexpected:", unexpected)
    print("Loaded model from checkpoint:", model_path)

    return f, task_key_centroid


def load_saved_weights_continue(model, model_path):
    # The current model and the model to initialize from can have different number of classes.
    state_dict = torch.load(model_path)
    if "model" in state_dict.keys():
        state_dict = state_dict["model"]

    # For torch.compile models
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    # For loading from a model trained with a different number of classes
    del state_dict["classifier.weight"]

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("missing:", missing)
    print("unexpected:", unexpected)

    # # For loading from a model trained with a different number of classes
    state_dict = torch.load(model_path)
    if "model" in state_dict.keys():
        state_dict = state_dict["model"]
        
    # For torch.compile models
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    model.classifier.weight.data[: len(state_dict["classifier.weight"])] = state_dict[
        "classifier.weight"
    ]
    print(
        "Classifier weights of pre-trained model: ",
        state_dict["classifier.weight"].shape,
    )
    print("Classifier weights of current model: ", model.classifier.weight.data.shape)
    
    epoch = 1
    optimizer = None
    lr_scheduler = None
    if "optimizer" in state_dict.keys():
        optimizer = state_dict["optimizer"]
    if "lr_scheduler" in state_dict.keys():
        lr_scheduler = state_dict["lr_scheduler"]
    if "epoch" in state_dict.keys():
        epoch = state_dict["epoch"]
    print("Loaded model from epoch: ", epoch)
    return epoch, optimizer, lr_scheduler


def save_checkpoint(output_dir, model, name="base_model_epoch", epoch=None):
    if epoch:
        cp = os.path.join(output_dir, name + str(epoch))
    else:
        cp = os.path.join(output_dir, name)

    torch.save(model.state_dict(), cp)
    return cp


def get_model_obj(model: nn.Module):
    return model.module if hasattr(model, "module") else model


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=100, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    # if is_main_process():
    torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if "initial_lr" not in group:
                    raise KeyError(
                        "param 'initial_lr' is not specified "
                        "in param_groups[{}] when resuming an optimizer".format(i)
                    )
        self.base_lrs = list(
            map(lambda group: group["initial_lr"], optimizer.param_groups)
        )
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class CosineSchedule(_LRScheduler):
    def __init__(self, optimizer, K):
        self.K = K
        super().__init__(optimizer, -1)

    def cosine(self, base_lr):
        return base_lr * math.cos(
            (99 * math.pi * (self.last_epoch)) / (200 * (self.K - 1))
        )

    def get_lr(self):
        return [self.cosine(base_lr) for base_lr in self.base_lrs]


def get_mixed_hard_negatives(
    query_embedding, doc_embeddings, current_doc_index, 
    sim=None, random_cands_pool=None, k=64, not_mask=None):

    """
    query_embedding: bs, dim
    doc_embeddings: doc_class, dim
    current_doc_index: bs (int)
    random_candidates: list(range(doc_class))
    
    By default the random rate is 0.5
    """

    if random_cands_pool is None:
        random_cands_pool = list(range(doc_embeddings.shape[0]))

    if sim is None:
        sim = torch.matmul(query_embedding, doc_embeddings.t())  # bs, doc_class

    current_sim = torch.cat(
        [sim[i, current_doc_index[i]].unsqueeze(0) for i in range(query_embedding.shape[0])]
    )

    ### Random negatives ###
    # with torch.no_grad():
    #     random_indices = []
    #     for i in range(query_embedding.shape[0]):
    #         rand_cands_indices = get_random_indices_v2(current_doc_index, random_cands_pool, k, i)
    #         random_indices.append(rand_cands_indices)

    # rand_sim = torch.gather(sim, 1, torch.tensor(random_indices, device=query_embedding.device))

    ### Hard negatives ###
    _sim = sim.clone()
    # Generate indices for diagonal elements
    diag_indices = torch.arange(query_embedding.shape[0]), current_doc_index

    # Set diagonal elements to negative infinity
    _sim[diag_indices] = float("-inf")

    # hard_sim = torch.topk(_sim, k // 2).values
    hard_sim = torch.topk(_sim, k).values
 
    mixed_hard_nce_loss = - torch.log(
        torch.exp(current_sim) / 
        (
            torch.exp(current_sim) + 
            # torch.exp(rand_sim).sum(dim=1) +
            torch.exp(hard_sim).sum(dim=1) +
            1.0e-6
        )
    ).mean()

    # Make sure logits are masked
    _sim = _sim.index_fill(dim=1, index=not_mask, value=float("-inf"))

    return mixed_hard_nce_loss


def get_random_indices_v2(current_doc_index, random_cands_pool, k, i):
    while True:
        rand_cands_indices = random.sample(random_cands_pool, k // 2)
        if current_doc_index[i].item() not in rand_cands_indices:
            return rand_cands_indices


def get_random_indices(current_doc_index, random_cands_pool, k, i):
    rand_cands_indices = random.sample(
        list(set(random_cands_pool) - set([current_doc_index[i].item()])), k // 2
    )

    return rand_cands_indices
