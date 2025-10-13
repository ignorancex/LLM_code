import warnings
import torch
import transformers
import os
import json
from common.utils import jload
from transformers import (
    default_data_collator,
)

warnings.filterwarnings("ignore")
from utils.model_utils import build_model
from utils.create_dataset import create_dataset
from utils.custom_trainer import CustomTrainer
from arguments import TrainerArguments, DeepspeedArguments
from sconf import Config
import pdb
import transformers
import os
import json
from common.utils import jload

def read_ds_config(config_path):
    config = jload(config_path)
    return config

def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False

def save_arguments(output_dir, filename, target_dict):
    save_dict = vars(target_dict)
    write_dict = {k: v for k, v in save_dict.items() if is_jsonable(v)}
    with open(os.path.join(output_dir, filename), "w") as f:
        for k, v in write_dict.items():
            f.write(f"{k}:{v}\n")

def overwrite_base_config(train_config, hf_args):
    ## Overwrite train_config attribute by hf_args.
    for key in train_config.keys():
        if hasattr(hf_args, key) and getattr(hf_args, key) is None:
            setattr(hf_args, key, train_config[key])
        elif getattr(hf_args, key) != train_config[key]:            
            print(f"overwride {key} with {getattr(hf_args, key)}")
            train_config[key] = getattr(hf_args, key)
    for attr in dir(hf_args):
        if not attr.startswith("__") and not callable(getattr(hf_args, attr)):
            train_config[attr] = getattr(hf_args, attr)
    return hf_args, train_config

def merge_config(base_config, train_config):
    for key in base_config.keys():
        if key not in train_config:
            train_config[key] = base_config[key]
    return train_config

def setup_parser(base_config="./train_configs/base.yaml"):
    parser = transformers.HfArgumentParser((TrainerArguments, DeepspeedArguments))
    trainer_args, ds_args = parser.parse_args_into_dataclasses()
    assert ds_args.use_deepspeed
    ds_args.world_size = torch.distributed.get_world_size()
    base_config = Config(base_config)
    train_config = Config(trainer_args.train_config)
    train_config = merge_config(base_config, train_config)
    trainer_args, train_config = overwrite_base_config(train_config, trainer_args)
    os.makedirs(trainer_args.output_dir, exist_ok=True)
    save_arguments(trainer_args.output_dir, "train_args.txt", trainer_args)
    save_arguments(trainer_args.output_dir, "ds_args.txt", ds_args)
    return trainer_args, train_config


def main():
    global local_rank
    trainer_args, train_config = setup_parser()
    torch.cuda.set_device(trainer_args.local_rank)
    tokenizer, model = build_model(trainer_args)
    train_dataset, eval_dataset = create_dataset(tokenizer, train_config)
    model.config.use_cache = False
    trainer_class = CustomTrainer
    print(trainer_args)
    print("init_trainer")
    trainer = trainer_class(
        model=model,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )
    print("start_train")
    if trainer_args.eval_only:
        trainer.evaluate()
    else:
        trainer.train(resume_from_checkpoint=trainer_args.resume)


if __name__ == "__main__":
    main()
