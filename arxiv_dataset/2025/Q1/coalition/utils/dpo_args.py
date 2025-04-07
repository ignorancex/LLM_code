
from typing import Dict, Optional
from dataclasses import field, dataclass

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})
    dataset_path : Optional[str] = field(default = "")

    # training parameters
    model_name_or_path: Optional[str] = field(
        default="openlm-research/open_llama_7b_v2",
        metadata={"help": "the location of the SFT model name or path"},
    )
    learning_rate: Optional[float] = field(default=1e-6, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="adamw_torch", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    gradient_checkpointing_use_reentrant: Optional[bool] = field(
        default=False, metadata={"help": "whether to use reentrant for gradient checkpointing"}
    )

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    max_prompt_length: Optional[int] = field(default=1536, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=2048, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=20000, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=20000, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=10000, metadata={"help": "the evaluation frequency"})

    output_dir: Optional[str] = field(default="./guide2_stage3_dpo_open_llama_7b_v2_gsm8k_half_split", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "whether to load the model in 4bit"})
    model_dtype: Optional[str] = field(
        default="bfloat16", metadata={"help": "model_dtype[float16, bfloat16, float] for loading."}
    )

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    seed: Optional[int] = field(
        default=0, metadata={"help": "Random seed that will be set at the beginning of training."}
    )

    guide_ckpt: Optional[str] = field(default="./logs/master_warmup_ift_v2_140k_open_llama_7b_v2_bf16.pt", metadata={"help": "the output directory"})
    guide_training: Optional[bool] = field(default=False, metadata={"help": "the output directory"})
    max_epochs: Optional[int] = field(default=5, metadata={"help": "the output directory"})
    name: Optional[str] = field(default="guide1-stage1-open-llama-7b-v2-ift-140k-commonsense-dpo", metadata={"help": "the output directory"})
    guide_index: Optional[int] = field(default=1, metadata={"help": "guide index"})
    stage_index: Optional[int] = field(default=1, metadata={"help": "stage index"})
    hf_model_path: Optional[str] = field(default="", metadata={"help": "model path"})
    model_name: Optional[str] = field(default="", metadata={"help": "model name"})
    tokenizer_path: Optional[str] = field(default="", metadata={"help": "tokenizer path"})
    hf_token: Optional[str] = field(default="", metadata={"help": "huggingface token"})
    cache_dataset_path: Optional[str] = field(default="", metadata={"help": "cache dataset path"})
    sro: Optional[bool] = field(default=False, metadata={"help": "sro"})
    dataset_name: Optional[str] = field(default="", metadata={"help": "dataset name"})
