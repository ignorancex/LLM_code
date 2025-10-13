from dataclasses import dataclass, field
from typing import Optional
import transformers


@dataclass
class DeepspeedArguments:
    use_deepspeed: Optional[bool] = field(default=True)
    rank: Optional[int] = field(default=None)
    world_size: Optional[int] = field(default=None)
    deepspeed_config: Optional[str] = field(default=None)
    # wandb_run_name: Optional[str] = field(default=None)

## values specied by arguments will overwrite values in train_config.
@dataclass
class TrainerArguments(transformers.TrainingArguments):
    gradient_checkpointing: Optional[bool] = field(default=True)
    max_steps: Optional[int] = field(default=3000)
    eval_steps: Optional[int] = field(default=3000)    
    save_steps: Optional[int] = field(default=3000)
    learning_rate: Optional[float] = field(default=1e-5, metadata={"help": "learning rate"})
    inst_mode_doc: Optional[bool] = field(default=None, metadata={"help": "Use instruction tuning-style training for document data."})
    inst_mode_qa: Optional[bool] = field(default=None, metadata={"help": "Use <INST> </INST> tag for instruction tuning."})
    noise_ratio: Optional[float] = field(default=None, metadata={"help": "noise rate in denoising auto-regressive training"})
    dropout: Optional[float] = field(default=None, metadata={"help": "attention dropout rate"})
    shuffle: Optional[int] = field(default=None, metadata={"help": "If you try to use shuffled data, the value needs to be larger than training epochs."})
    eval_only: Optional[bool] = field(default=None)
    eval_qa: Optional[bool] = field(default=False)
    eval_perplex: Optional[bool] = field(default=None)
    max_seq_len: Optional[int] = field(default=None)
    per_device_train_batch_size: int = field(default=32)
    per_device_eval_batch_size: int = field(default=4)
    use_flash_attn: Optional[bool] = field(default=True, metadata={"help": "Activate use flash attention. Available for A100."})
    train_config: Optional[str] = field(default=None)
    init_ckpt: Optional[str] = field(default=None)
    task_type: Optional[str] = field(default=None)
    resume: Optional[bool] = field(default=None, metadata={"help": "Use if resume training from some checkpoints."})
    reload_data: Optional[bool] = field(default=None)
    report_to: Optional[str] = field(default=None)
    run_name: Optional[str] = field(default=None)
    model_name: Optional[str] = field(default=None)
    result_path: Optional[str] = field(default=None)
    doc_dataset: Optional[str] = field(default=None)
    qa_dataset: Optional[str] = field(default=None)
    path_doc_eval: Optional[str] = field(default=None)
    path_doc_train: Optional[str] = field(default=None)
    path_qa_train: Optional[str] = field(default=None)
    path_qa_eval: Optional[str] = field(default=None)
    data_output_path: Optional[str] = field(default=None)
    cache_path: Optional[str] = field(default=None)
    concat_dataset: Optional[bool] = field(default=None)
    seed: Optional[int] = field(default=None)
    wandb_run_name: Optional[str] = field(default=None)
    p_choose_qa: Optional[float] = field(default=None)
    logging_steps: Optional[int] = field(default=None)
    save_strategy: Optional[str] = field(default=None)
    deepspeed: Optional[str] = field(default=None)