import json
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import ray
from omegaconf.listconfig import ListConfig
from ray.runtime_env import RuntimeEnv
from ray.util.placement_group import placement_group
from transformers import AutoTokenizer

from train_utils.dataset import PromptDataset
from train_utils.ppo import RayPPOTrainer
from train_utils.rl.actors import PolicyRayActor
from train_utils.utils import _validate_args, create_vllm_engines, get_strategy



import os
from abc import ABCMeta
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

try:
    from hydra._internal.utils import get_args_parser
    from omegaconf import OmegaConf

    HYDRA_INSTALLED = True
except Exception as e:
    print(f"Hydra is not installed, so override exp.xxx in command line is not supported: {e}")
    HYDRA_INSTALLED = False


__all__ = ["BaseExp", "BaseConfig"]


@dataclass
class BaseConfig:
    seed: Optional[int] = None


class BaseExp(metaclass=ABCMeta):
    """ """

    def __init__(self) -> None:
        pass

    def set_cfg(self, cfg):
        """
        Noticing that set-function should return self
        """

        override_cfg = {}
        if HYDRA_INSTALLED:
            from hydra._internal.config_loader_impl import ConfigLoaderImpl
            from hydra.core.override_parser.overrides_parser import OverridesParser

            hydra_cfg = OmegaConf.create()
            ConfigLoaderImpl._apply_overrides_to_config(
                OverridesParser.create().parse_overrides(overrides=self.args.overrides), hydra_cfg
            )

            if "exp" in hydra_cfg:
                for k, v in hydra_cfg.exp.items():
                    if hasattr(cfg, k):
                        if v != getattr(cfg, k):
                            print(f"Override {k} from {getattr(cfg, k)} to {v}")
                            cfg.__setattr__(k, v)
                            override_cfg[k] = v
                    else:
                        # only allow override exist attribute in exp
                        raise ValueError(f"Attribute {k} is not found in {cfg}")

        self.cfg = cfg
        self._override_cfg = override_cfg
        return self

    @cached_property
    def args(self):
        return self.get_args_parser().parse_args()

    @staticmethod
    def get_args_parser():
        parser = get_args_parser()
        return parser

    @staticmethod
    def get_cfg_as_str(dict_cfg) -> str:
        import numpy as np
        from tabulate import tabulate

        config_table = []

        def add_table_element(info_dict, config_table):
            for c, v in info_dict.items():
                if not isinstance(v, (int, float, str, list, tuple, dict, np.ndarray)):
                    if hasattr(v, "__name__"):
                        v = v.__name__
                    elif hasattr(v, "__class__"):
                        v = v.__class__
                if c[0] == "_" and c[1] == "_":
                    continue
                config_table.append((str(c), str(v)))

        add_table_element(dict_cfg.__dict__, config_table)
        headers = ["--------- config key ---------", "------ value ------ "]
        config_table = tabulate(config_table, headers, tablefmt="plain")
        return config_table

    @cached_property
    def exp_name(self):
        exp_class_name = self.__class__.__name__
        if hasattr(self, "_override_cfg"):
            for k, v in self._override_cfg.items():
                if isinstance(v, str):
                    exp_class_name += "_{}-{}".format(k, v.replace("/", "_"))
                else:
                    exp_class_name += "_{}-{}".format(k, v)
        return exp_class_name

    @cached_property
    def output_dir(self):
        output_root = getattr(self.cfg, "output_root", "./output")
        return os.path.join(output_root, self.exp_name)

    @cached_property
    def accelerator(self):
        return None

    def prepared_model_and_optimizer(self):
        return self.accelerator.prepare(self.model, self.optimizer)



@dataclass
class BasePPOExpConfig(BaseConfig):
    # resource related settings
    ref_num_nodes: int = 1
    ref_num_gpus_per_node: int = 2
    reward_num_nodes: int = 1
    reward_num_gpus_per_node: int = 2
    actor_num_nodes: int = 1
    actor_num_gpus_per_node: int = 2
    critic_num_nodes: int = 1
    critic_num_gpus_per_node: int = 2
    colocate_critic_reward: bool = True
    colocate_actor_ref: bool = True
    colocate_all: bool = False
    vllm_num_engines: int = 4
    vllm_tensor_parallel_size: int = 1
    vllm_sync_backend: str = "nccl"
    local_rank: int = -1

    # path related settings
    pretrain: Optional[str] = "example_path"
    critic_pretrain: Optional[str] = "example_path"
    reward_pretrain: Optional[str] = "example_path"
    ckpt_path: str = "example_path"
    save_path: str = "example_path"
    tensorboard_log_dir: str = "example_path"
    prompt_data: ListConfig = ListConfig([])

    # training related settings
    seed: int = 42
    load_checkpoint: bool = False
    zero_stage: int = 3

    bf16: bool = True
    zpg: int = 1
    adam_offload: bool = True
    flash_attn: bool = True
    grad_accum_dtype: Optional[str] = None
    disable_trace_cache: bool = False
    gradient_checkpointing: bool = True
    gradient_checkpointing_use_reentrant: bool = False
    disable_fast_tokenizer: bool = False
    target_modules: str = "all-linear"

    # inference realted settings
    enable_prefix_caching: bool = False
    enable_chunked_prefill: bool = False
    max_num_batched_tokens: int = 2048
    enforce_eager: bool = False
    gpu_memory_utilization: float = 0.85

    # logging related settings
    eval_steps: int = -1
    save_steps: int = -1
    save_interval: int = 100

    # ppo related settings
    actor_learning_rate: float = 5e-7
    critic_learning_rate: float = 9e-6
    num_episodes: int = 1
    max_epochs: int = 1
    prompt_max_len: int = 1024
    generate_max_len: int = 1024

    train_batch_size: int = 256
    micro_train_batch_size: int = 8
    rollout_batch_size: int = 256
    micro_rollout_batch_size: int = 32
    micro_forward_batch_size: int = 32
    policy_update_steps: int = 4
    critic_update_steps: int = 4
    max_len: Optional[int] = None
    max_norm: float = 1.0
    num_warmup_steps: int = 5

    l2: float = 0.0
    eps_clip: float = 0.2
    value_clip: float = 0.2
    lambd: float = 1
    gamma: float = 1

    normalize_reward: bool = True
    top_p: float = 1.0
    temperature: float = 1.0
    freezing_actor_steps: int = -1
    n_samples_per_prompt: int = 1

    kl_target: Optional[float] = None
    init_kl_coef: float = 0.01
    use_kl_estimator_k3: bool = False
    use_abs_kl: bool = False
    use_kl_loss: bool = False
    kl_loss_coef: float = 0.0
    adam_betas: tuple = (0.9, 0.95)
    reward_clip_range: tuple = (-10, 10)

    use_compute_reward_fn: bool = False
    advantage_normalize: bool = True

    value_head_prefix: str = "value_head"
    ref_reward_offload: bool = False

    enable_eval: bool = False
    eval_interval: int = -1
    update_ref_every_epoch: bool = False


class BasePPOExp(BaseExp):
    @cached_property
    def trainer(self):
        #vllm_engines = self.create_inference_engine()
        return RayPPOTrainer(
            cfg=self.cfg,
            strategy=self.strategy,
            tokenizer=self.tokenizer,
            #train_dataset=self.train_dataset,
            #eval_dataset=self.eval_dataset,
            #vllm_engines=vllm_engines,
        )

    @cached_property
    def tokenizer(self, padding_side="left"):
        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.pretrain, trust_remote_code=True, use_fast=not self.cfg.disable_fast_tokenizer
        )
        tokenizer.padding_side = padding_side
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    @cached_property
    def train_dataset(self):
        dialogues = []
        for file_path in self.cfg.prompt_data:
            with open(file_path, "r") as f:
                dialogues.extend(json.load(f))
        prompts_dataset = PromptDataset(
            dialogues, self.tokenizer, self.cfg.prompt_max_len, self.strategy, pretrain_mode=False
        )
        return prompts_dataset

    @cached_property
    def eval_dataset(self):
        return None

    @cached_property
    def strategy(self):
        return get_strategy(self.cfg)

    @cached_property
    def PolicyRayActor(self):
        return PolicyRayActor

    @cached_property
    def get_colocate_pg(self):
        if self.cfg.colocate_all:
            pg = placement_group([{"GPU": 1, "CPU": 1}] * self.cfg.vllm_num_engines, strategy="PACK")
            ray.get(pg.ready())
            return pg
        else:
            return None

    def create_inference_engine(self):
        return create_vllm_engines(
            self.cfg.vllm_num_engines,
            self.cfg.vllm_tensor_parallel_size,
            self.cfg.pretrain,
            self.cfg.seed,
            self.cfg.enable_prefix_caching,
            self.cfg.enforce_eager,
            self.cfg.max_len,
            self.cfg.colocate_all,
            self.cfg.enable_chunked_prefill,
            self.cfg.max_num_batched_tokens,
            self.cfg.gpu_memory_utilization,
            self.cfg.micro_rollout_batch_size,
            self.get_colocate_pg,
        )

    async def run(self):
        # validate the arguments
        _validate_args(self.cfg)

        # initialize the ray cluster
        ray.init(
            runtime_env=RuntimeEnv(
                env_vars={
                    "NCCL_DEBUG": "WARN",
                    "NCCL_PXN_DISABLE": "1",
                    "NCCL_ALGO": "^Ring",
                    "NCCL_NET_OVERHEAD": "1000000",
                    "CUDA_LAUNCH_BLOCKING": "1",
                }
            )
        )

        # build the models
        await self.trainer.build_models(self.PolicyRayActor)

        # initialize the trainer and enter the training loop
        await self.trainer.train()
