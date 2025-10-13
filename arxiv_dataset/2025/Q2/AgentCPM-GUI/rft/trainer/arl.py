import os
import io
import torch.distributed.tensor
import zmq
import torch
import datetime
import pickle
import uuid
import transformers
import copy

from typing import Any, Callable, Optional, Union, Sized, List, Dict
from collections import defaultdict
from multiprocessing import Process
from packaging import version
import contextlib
from types import MethodType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,StateDictType,FullStateDictConfig
from torch.distributed.device_mesh import DeviceMesh
import torch.distributed as dist

from datasets import Dataset, IterableDataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    Trainer
)
from transformers.trainer import seed_worker,DataLoader,is_datasets_available

from accelerate.utils.memory import clear_device_cache
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from accelerate.utils.other import is_compiled_module
from accelerate import PartialState,init_empty_weights

from trl.models import unwrap_model_for_generation,prepare_deepspeed
from trl.data_utils import is_conversational
from trl.trainer.grpo_trainer import GRPOConfig, RepeatRandomSampler
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.models import create_reference_model
from trl.import_utils import is_vllm_available

from transformers.trainer import (
    OptimizerNames,
    DistributedType,
    IS_SAGEMAKER_MP_POST_1_10,
    is_torch_xla_available,
    is_sagemaker_mp_enabled,
    Path,
    accelerate_version,
    remove_dummy_checkpoint,
    WEIGHTS_NAME,SAFE_WEIGHTS_NAME
)


if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams


from configs import GRPOTrainingConfig
from .utils import logger, Timer, _prepare_messages,_process_inputs,_create_inputs, no_sync, GlobalDistributed0MQDataLoader
from .zmq import global_sync_proc, local_balance_proc, TaskAndContent, TaskStatus

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

class AsyncRLGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """
    
    
    def __init__(
        self,
        model: PreTrainedModel,
        model_init_kwargs: dict[str, Any],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOTrainingConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        device_mesh: Optional[DeviceMesh] = None,
    ):
        def data_collator(inputs):
            assert "id" in inputs[0], "You must assign a unique `id` in your datasets to use ARL"
            prompt_inputs = _prepare_messages([inp["prompt"] for inp in inputs],processing_class,self.max_prompt_length)
            
            return prompt_inputs,inputs


        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            logger.warning("Using ARL with Deepspeed is only competible with DeepSeed Zero-1! May dramatically slow down with Zero-2 and Zero-3!")

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.clear_device = args.clear_device
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.ds3_gather_for_generation = args.ds3_gather_for_generation
        self.beta = args.beta
        self.epsilon = args.epsilon
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.max_items_to_cache = args.max_items_to_cache
        self.cached_data = []
        self.device_mesh = device_mesh
        self.rank = dist.get_rank()
        if self.device_mesh is not None:
            self.tp_size = self.device_mesh.shape[-1]
            self.tp_group_id = dist.get_rank() // self.tp_size
            self.tp_rank = dist.get_rank(self.device_mesh.get_group("tp"))
        else:
            self.tp_size = 1
            self.tp_group_id = dist.get_rank()
            self.tp_rank = 0
        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        
        
        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes * args.gradient_accumulation_steps // self.tp_size
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        # if self.args.eval_strategy != "no":
        #     global_batch_size = args.per_device_eval_batch_size * num_processes
        #     possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        #     if self.num_generations not in possible_values:
        #         raise ValueError(
        #             f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
        #             f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
        #             f"eval batch size, the valid values for the number of generations are: {possible_values}."
        #         )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)
        self.inference_model = None
        if self.accelerator.distributed_type == DistributedType.FSDP:
            # register_fsdp_forward_method(self.model_wrapped, "generate")
            # cache a model in cpu
            self.inference_model = create_reference_model(model).cpu()
            if self.accelerator.is_fsdp2:
                self.accelerator.no_sync = MethodType(no_sync,self.accelerator)


        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        
        # Enable gradient checkpointing if requested
        self.gradient_checkpointing = args.gradient_checkpointing
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Reference model
        self.beta = args.beta
        self.ref_model = None
        self.update_ref_model = False
        if self.beta == 0 or is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        elif self.is_deepspeed_enabled:
            self.ref_model = AutoModel.from_pretrained(model.name_or_path,**model_init_kwargs)
        else:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model).cpu()
            
        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_funcs = reward_funcs


        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.accelerator.is_fsdp2:
                self.update_ref_model = True
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
            
        # Setup 0MQ
        self.greedy_gather_wait_time = args.greedy_gather_wait_time
        self.main_process_ip = os.environ.get("MASTER_ADDR")
        self.global_sync_address = f"tcp://{self.main_process_ip}:{args.global_task_sync_port}"
        self.global_result_collect_address = f"tcp://{self.main_process_ip}:{args.global_result_collect_port}"
        self.global_data_dispatch_address =  f"tcp://{self.main_process_ip}:{args.global_data_dispatch_port}"
        
        if self.accelerator.is_main_process:
            # setup global sync thread
            self.global_sync_proc = Process(
                target=global_sync_proc,
                kwargs={
                    "sync_address": self.global_sync_address,
                    "collect_address": self.global_result_collect_address,
                    "num_generations": args.num_generations,
                    "num_to_sync": args.gradient_accumulation_steps * self.accelerator.num_processes * args.per_device_train_batch_size,
                    "num_nodes": self.accelerator.num_processes // torch.cuda.device_count(),
                    "tp_size": self.tp_size
                },
                daemon=True
            )
            self.global_sync_proc.start()
            
        self.accelerator.wait_for_everyone()
        
        if self.accelerator.is_local_main_process:
            self.local_balance_proc = Process(
                target=local_balance_proc,
                kwargs={
                    "local_collect_address": args.local_collect_address,
                    "local_provider_address": args.local_provider_address,
                    "local_steal_port": args.local_steal_port,
                    "global_sync_address": self.global_sync_address,
                    "global_result_collect_address":self.global_result_collect_address,
                    "global_data_dispatch_address":self.global_data_dispatch_address,
                    "chunk_size": args.per_device_train_batch_size,
                    "mt_max_beam_width": args.mt_max_beam_width,
                    "max_cache_size": args.gradient_accumulation_steps * args.per_device_train_batch_size * torch.cuda.device_count() * 8,
                    "processing_class_name_or_path": model.name_or_path,
                    "max_prompt_length": args.max_prompt_length,
                    "tp_size": self.tp_size
                }
            )
            self.local_balance_proc.start()
        
        self.chunk_size = args.per_device_train_batch_size
        
        self.accelerator.wait_for_everyone()
        
        self.zmqctx = zmq.Context(4)
        self.sync_signal = self.zmqctx.socket(zmq.SUB)
        self.sync_signal.setsockopt(zmq.TCP_KEEPALIVE, 1)
        self.sync_signal.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 60)
        self.sync_signal.setsockopt(zmq.TCP_KEEPALIVE_CNT, 5)
        self.sync_signal.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 10)
        self.sync_signal.setsockopt(zmq.SUBSCRIBE,b"SYNC_FOR_UPDATE")
        self.sync_signal.connect(self.global_sync_address)

        self.balance_send = self.zmqctx.socket(zmq.REQ)
        self.balance_send.connect(args.local_collect_address)

        self.balance_recv = self.zmqctx.socket(zmq.REQ)
        self.balance_recv.connect(args.local_provider_address)
        
        self.ack = self.zmqctx.socket(zmq.REQ)
        self.ack.connect(self.global_result_collect_address)
        
        self.poller = zmq.Poller()
        self.poller.register(self.sync_signal,zmq.POLLIN)
        self.poller.register(self.balance_recv,zmq.POLLIN)
        self.poller.register(self.ack, zmq.POLLIN)
        
        self.recv_idx = 0
        
        if self.device_mesh is not None:
            tp_group = dist.get_process_group_ranks(self.device_mesh.get_group("tp"))
            logger.info("Worker {}, TP rank {} is running in TP Groups {} and setup 0MQ.".format(self.rank,self.tp_rank,tp_group))
        else:
            logger.info("Worker {} is running on {} and setup 0MQ.".format(self.rank,os.environ.get("RANK",0)),)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _async_sampling(self, unwrapped_model, epoch_iterator, num_batches):
        # initial batch fill
        current_batch = [self.cached_data.pop() for _ in range(min(num_batches, len(self.cached_data)))]
        if len(current_batch) < num_batches:
            # first send sync request
            self.balance_recv.send_pyobj((self.tp_group_id,self.rank,self.recv_idx))

        waiting_for_ack = False
        # `wait_time_ms` will hold our dynamic timeout (in milliseconds)
        wait_time_ms = 0

        while True:
            # poll with the current wait_time
            socks = dict(self.poller.poll(timeout=wait_time_ms))

            # 1) handle ack-only sockets if we’re waiting for one
            if waiting_for_ack and self.ack in socks:
                self.ack.recv()
                waiting_for_ack = False

            # 2) if we successfully received backward data, reset wait_time and refill cache
            if self.balance_recv in socks and not waiting_for_ack:
                data: dict = self.balance_recv.recv_pyobj()                
                batch_samples = [data] * self.num_iterations
                self.recv_idx += 1
                self.ack.send_pyobj(self.chunk_size)
                waiting_for_ack = True
                logger.debug("Worker {} Received backward data".format(self.rank))
                
                wait_time_ms = self.greedy_gather_wait_time
                
                if len(current_batch) < num_batches:
                    needed = num_batches - len(current_batch)
                    current_batch.extend(batch_samples[:needed])
                    self.cached_data.extend(batch_samples[needed:])
                    # if still short, ask for more
                    if len(current_batch) < num_batches and len(self.cached_data) < self.max_items_to_cache:
                        self.balance_recv.send_pyobj((self.tp_group_id,self.rank,self.recv_idx))
                    else:
                        wait_time_ms = 0 # reset wait_time_ms to 0 to avoid waiting for more data
                else:
                    self.cached_data.extend(batch_samples)
                    wait_time_ms = 0 # reset wait_time_ms to 0 to avoid waiting for more data

                # immediately go back to polling (no sampling yet)
                continue

            # 3) if we get the sync signal, verify and break out
            if self.sync_signal in socks:
                try:
                    parts = self.sync_signal.recv_multipart()
                    wid, d = parts
                    sync_steps = pickle.loads(d)
                except:
                    logger.error(f"Receive Undcodeable Sync singal: {parts}")
                    sync_steps = "UNKNOWN"
                    wid = "UNKNOWN".encode()
                    
                assert len(current_batch) == num_batches, (
                    f"INVALID SYNC at step {sync_steps} with "
                    f"{len(current_batch)} + {len(self.cached_data)} backward data"
                )
                logger.debug(
                    f"Worker {self.rank} Received sync signal {wid.decode()} "
                    f"with {len(current_batch)} + {len(self.cached_data)} backward data"
                )
                break

            # 4) no backward data arrived within wait_time_ms
            if wait_time_ms > 0:
                # exponential back‐off: halve the wait time, but don’t go below zero
                wait_time_ms = int(max(0, wait_time_ms / 2))
                # skip sampling until wait_time_ms decays to 0
                continue

            # 5) wait_time_ms has decayed to zero → do a sample step
            try:
                inputs = next(epoch_iterator)
                self.sample_step(inputs, unwrapped_model)
            except StopIteration:
                # iterator is exhausted
                continue
                
        if waiting_for_ack:
            self.ack.recv()

        return current_batch

    def _iter_data_sampling(self, epoch_iterator, num_batches):
        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.
        
        # There are two possible strategies here:
        # 1. First complete all the tasks in the task pool and then return them interleaved
        # 2. Complete the mini-batch of tasks and return them one by one
        # These two strategies logically same when the num_batches == 1 (e.g. no gradiant accumulation), but they are different when num_batches > 1
        # The first strategy is more efficient when the task pool is large, with less context switching overhead
        # But does not a good idea when the memory is limited, since it will need to cache a large number of tasks in the task pool
        
        # Print Memory Usage
        # print(f"Memory Usage: {torch.cuda.memory_allocated() / 1024 / 1024} MB")
        
        # What we should do here is:
        # 1. listen to the sync signal and receive the inputs for the next training step
        # 2. keep sampling until signal received.
        # 3. return the batch_samples and num_items_in_batch when the global batch all ready 

        with torch.no_grad():
            with self.prepare_generation(self.model_wrapped) as unwrapped_model:
                batch_samples = self._async_sampling(
                    unwrapped_model,
                    epoch_iterator=epoch_iterator,
                    num_batches=num_batches
                )
    
    
            if self.beta > 0 and not self.control.should_evaluate:
                self.ref_model = self.ref_model.to(device=self.accelerator.device)
                for data in batch_samples:
                    if self.ref_model is not None:
                        ref_per_token_logps = self._get_per_token_logps(self.ref_model, data["prompt_inputs"])
                    else:
                        with self.accelerator.unwrap_model(self.model_wrapped).disable_adapter() as unwrapped_model:
                            ref_per_token_logps = self._get_per_token_logps(unwrapped_model, data["prompt_inputs"])
                    
                    data["ref_per_token_logps"] = ref_per_token_logps.detach().cpu()
                
                # if self.clear_device:
                #     self.ref_model.cpu()

        yield len(batch_samples)
        yield from batch_samples

    @contextlib.contextmanager
    def prepare_generation(self, model_wrapped, clear_device: bool = None, update_inference_model: bool = True):
        clear_device = clear_device if clear_device is not None else self.clear_device
        if clear_device:
            self.inference_model.cpu()
            clear_device_cache(True)
        device = self.accelerator.device

        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            
            with unwrap_model_for_generation(model_wrapped, self.accelerator, gather_deepspeed3_params=self.ds3_gather_for_generation) as unwrapped_model:
                if self.gradient_checkpointing:
                    unwrapped_model = self._disable_gradient_checkpointing(unwrapped_model, self.args)
                
                yield unwrapped_model
                
                
                if self.gradient_checkpointing:
                    unwrapped_model = self._enable_gradient_checkpointing(unwrapped_model, self.args)
        
        elif self.accelerator.distributed_type == DistributedType.FSDP:
            if self.accelerator.is_fsdp2:
                
                if update_inference_model:
                    state_dict = model_wrapped.state_dict()
                    class wrappeddict(dict):
                        def __getitem__(self, key):
                            d = super().__getitem__(key)
                            if isinstance(d,torch.distributed.tensor.DTensor):
                                return d.to(device=device).full_tensor().detach().to(dtype=torch.bfloat16,device=device)
                            else:
                                return d
                
                    state_dict = wrappeddict(state_dict)
                    self.inference_model.load_state_dict(state_dict,strict=True)
                    if self.update_ref_model:
                        self.ref_model.load_state_dict(state_dict,strict=True)
                        self.update_ref_model = False
                        
                self.inference_model.eval()
                
                yield self.inference_model.to(device=device)
                
                
            else:
                if update_inference_model:
                    cfg = FullStateDictConfig(offload_to_cpu=False, rank0_only=False)
                    with FSDP.state_dict_type(model_wrapped, StateDictType.FULL_STATE_DICT, cfg):
                        full_state = model_wrapped.state_dict()
                
                    self.inference_model.load_state_dict(full_state,strict=True)
                    del full_state

                yield self.inference_model.to(device=device)
                
        else:
            raise NotImplementedError(f"Unsupported distributed_type {self.accelerator.distributed_type}")
        
        if clear_device:
            self.inference_model.cpu()
            clear_device_cache(True)

    def get_batch_samples(self, epoch_iterator, num_batches):
        # TODO: Support num_items_in_batch
        gen = self._iter_data_sampling(epoch_iterator, num_batches)

        class _IterWithLen:
            def __init__(self, gen):
                self.gen = gen
                self.length = next(gen)
            
            def __iter__(self):
                return self.gen
            
            def __len__(self):
                return self.length

        # Assuming num_batches is defined somewhere
        wrapped_gen = _IterWithLen(gen)
        return wrapped_gen, len(wrapped_gen)
    
    def sample_step(self, tasks, model: PreTrainedModel):
        # TODO: this function should be rewrite for async sampling

        s_time = datetime.datetime.now()
        device = self.accelerator.device
        
        prompt_inputs,inputs = tasks
        prompt_inputs = self._prepare_inputs(prompt_inputs)
        prompt_inputs.pop('image_sizes',None)
        
        logger.debug(f"Worker {self.rank} Start Sampling {len(inputs)} Tasks.")
        # Start Generation
        completion_ids = model.generate(
            **prompt_inputs,
            tokenizer=self.processing_class.tokenizer,
            do_sample = True,
            temperature = 0.1 if self.control.should_evaluate else 1.0,
            repetition_penalty = 1.05,
            max_new_tokens = self.max_completion_length,
            use_cache=True,
            synced_gpus=False if self.ds3_gather_for_generation else True
        )
        
        logger.debug(f"Worker {self.rank} Sampling {len(inputs)} Tasks for time: {datetime.datetime.now() - s_time}")
        
        if isinstance(completion_ids, tuple):
            completion_ids = completion_ids[1].sequences

        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]
        
        # Compute the rewards
        rewards_per_func = torch.zeros(len(inputs), len(self.reward_funcs), device=device)
        # print(rewards_per_func.shape)
        prompts = [inp["prompt"] for inp in inputs]
        for i, reward_func in enumerate(self.reward_funcs):
            reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
            for key in reward_kwargs:
                for item in inputs:
                    reward_kwargs[key].append(item[key])
            output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
        rewards = rewards_per_func.mean(dim=1)
        
        # Log the metrics
        # mode = "eval" if self.control.should_evaluate else "train"
        
        # reward_per_func = self.accelerator.gather(rewards_per_func).mean(0)
        # for i, reward_func in enumerate(self.reward_funcs):
        #     reward_func_name = reward_func.__name__
        #     self._metrics[mode][f"rewards/{reward_func_name}"].append(reward_per_func[i].item())
            
        if self.control.should_evaluate:
            prompt_len = prompt_inputs["input_ids"].size(1)
            prompt_inputs["rewards"] = rewards.to(device=device)
            advantages = torch.zeros(len(inputs),device=device)
            step_ids = torch.tensor([inp.get("step_id",0) for inp in inputs],device=device)
            prompt_inputs,completion_mask = _create_inputs(self.processing_class,prompt_inputs,completion_ids)
            return {
                "prompt_inputs": prompt_inputs,
                "completion_mask": completion_mask,
                "advantages": advantages,
                "prompt_len": prompt_len,
                "step_ids": step_ids
            }
        
        rewards = rewards.cpu()
        # process and send them to local balance
        for idx,item in enumerate(inputs):
            tac = TaskAndContent(
                data={
                    **item,
                    "completion": completions[idx][0]['content'],
                    "completion_ids": completion_ids[idx].cpu(),
                    "reward": rewards[idx].item(),
                },
                status=TaskStatus(
                    task_id=item["id"],
                    completion_id=uuid.uuid4(),
                    score=rewards[idx].item()
                )
            )
            # with Timer("Sending Completions"):
            # send it
            self.balance_send.send_pyobj(tac)
            _ = self.balance_send.recv_string()
        del prompt_inputs,inputs

    def _get_per_token_logps(self, model, inputs):
        
        attention_mask = inputs["attention_mask"]
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        inputs = self._prepare_inputs({
            "input_ids": inputs["input_ids"],
            "image_bound": inputs["image_bound"],
            "tgt_sizes": inputs["tgt_sizes"],
            "pixel_values": inputs["pixel_values"],
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        })

        logits = model(data=inputs, use_cache=False).logits
        
        # if self.accelerator.is_main_process:
        #     import pdb; pdb.set_trace()
        # self.accelerator.wait_for_everyone()
        
        # assert not torch.any(torch.isnan(logits)), "{}".format(torch.isnan(logits).sum())
        
        logits = logits[:, :-1, :]
        input_ids = inputs["input_ids"][:, 1:]
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        del inputs
        return torch.stack(per_token_logps)

    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        mode = "eval" if self.control.should_evaluate else "train"
        # prompt_inputs, completion_mask, advantages, prompt_len, step_ids = inputs
        prompt_inputs = inputs["prompt_inputs"]
        prompt_len = inputs["prompt_len"]
        advantages = inputs["advantages"]
        completion_mask = inputs["completion_mask"]
        step_ids = inputs.get("step_ids",torch.zeros((1,1),device=self.accelerator.device))
        
        log_dict = {
            "rewards": prompt_inputs["rewards"],
            "length/prompt": torch.tensor(prompt_len,device=self.accelerator.device,dtype=torch.float),
            "length/completion": completion_mask.to(torch.float).sum(dim=1),
            "step": step_ids.to(torch.float),
            "advantages": advantages.to(dtype=torch.float)
        }
        
        per_token_logps = self._get_per_token_logps(model, prompt_inputs)
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        per_token_logps = per_token_logps[:, prompt_len - 1:]

        old_per_token_logps = per_token_logps.detach()
        
        # Compute the policy ratio and clipped version
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = - torch.min(per_token_loss1, per_token_loss2)

        # Add KL penalty if beta > 0, skip kl calculation during evaluation (it has no function)
        if self.beta > 0 and not self.control.should_evaluate:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            ref_per_token_logps = ref_per_token_logps[:, prompt_len - 1:].detach()

            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            per_token_loss = per_token_loss + self.beta * per_token_kl

            # Log KL divergence
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            log_dict["kl"] = mean_kl
            # self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        # Compute final loss
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log clip ratio
        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        log_dict["clip_ratio"] = clip_ratio
        log_dict = self.accelerator.gather_for_metrics(log_dict)
        for k in log_dict.keys():
            self._metrics[mode][k].append(log_dict[k].mean().item())
            self._metrics[mode][k+'/max'].append(log_dict[k].max().item())
            self._metrics[mode][k+'/min'].append(log_dict[k].min().item())
            self._metrics[mode][k+'/std'].append(log_dict[k].std().item())

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "eval" if self.control.should_evaluate else "train"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}
        
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics[mode].clear()

    def _get_train_sampler(self):
        """Returns a sampler that ensures proper data sampling for GRPO training."""
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        
        return RepeatRandomSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=1,
            seed=self.args.seed,
        )
        
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        
        # if self.device_mesh is not None:
        #     return TensorParallelDistributedZMQDataLoader(
        #         dataset=train_dataset,
        #         global_sync_address=self.global_data_dispatch_address,
        #         tp_group=self.device_mesh.get_group("tp"),
        #         **dataloader_params
        #     )
        
        return GlobalDistributed0MQDataLoader(
            dataset=train_dataset,
            global_sync_address=self.global_data_dispatch_address,
            world_size=self.accelerator.num_processes,
            **dataloader_params
        )
        
    def _get_eval_sampler(self, eval_dataset):
        """Returns a sampler for evaluation."""
        return RepeatRandomSampler(
            data_source=eval_dataset,
            mini_repeat_count=1,
            seed=self.args.seed,
        )

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        with torch.no_grad():
            with self.prepare_generation(self.model_wrapped,clear_device=False,update_inference_model=False) as unwrapped_model:
                batch_samples = self.sample_step(inputs,unwrapped_model)
            
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(unwrapped_model, batch_samples)
            loss = loss.mean().detach()
        return loss, None, None

        
        
    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable({"use_reentrant":use_reentrant})
        # Enable gradient checkpointing for non-PEFT models
        else:
            model.gradient_checkpointing_enable({"use_reentrant":use_reentrant})


        if use_reentrant:
            model.enable_input_require_grads()

        return model
    
    def _disable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Disables gradient checkpointing for the model."""
        
        # Disable gradient checkpointing for PEFT models
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_disable()
        # Disable gradient checkpointing for non-PEFT models
        else:
            model.gradient_checkpointing_disable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.disable_input_require_grads()
        
        # Ensure use_cache is enabled
        model.config.use_cache = True
        return model


    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Will save the model, so you can reload it using `from_pretrained()`.

        Will only save from the main process.
        """
        if self.ref_model is not None:
            self.ref_model.cpu()
        if self.inference_model is not None:
            self.inference_model.cpu()
        clear_device_cache(True)
        if output_dir is None:
            output_dir = self.args.output_dir

        if is_torch_xla_available():
            self._save_tpu(output_dir)
        elif is_sagemaker_mp_enabled():
            # Calling the state_dict needs to be done on the wrapped model and on all processes.
            os.makedirs(output_dir, exist_ok=True)
            state_dict = self.model_wrapped.state_dict()
            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
            if IS_SAGEMAKER_MP_POST_1_10:
                # 'user_content.pt' indicates model state_dict saved with smp >= 1.10
                Path(os.path.join(output_dir, "user_content.pt")).touch()
        elif self.is_fsdp_enabled:
            if self.accelerator.is_fsdp2 or ("FULL_STATE_DICT" in str(self.accelerator.state.fsdp_plugin.state_dict_type)) and (
                version.parse(accelerate_version) > version.parse("0.24.1")
            ):
                state_dict = self.accelerator.get_state_dict(self.model)
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
        elif self.is_deepspeed_enabled:
            try:
                state_dict = self.accelerator.get_state_dict(self.deepspeed)
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
            except ValueError:
                logger.warning(
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                    " zero_to_fp32.py to recover weights"
                )
                if self.args.should_save:
                    self._save(output_dir, state_dict={})
                # remove the dummy state_dict
                remove_dummy_checkpoint(self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
                self.model_wrapped.save_checkpoint(output_dir)

        elif self.args.should_save:
            self._save(output_dir)

        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")