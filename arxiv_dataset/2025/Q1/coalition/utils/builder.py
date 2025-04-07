
import json
from datasets import load_dataset
from torch.utils.data import DataLoader

from src import (
    WarmupInstructionTuningModule, 
    InferenceModule
)

from data import (
    CoTDataset,
    GSM8KRationaleGenerationDataset,
    GSM8KRationaleRefinementDataset,
    WinograndeRationaleGenerationDataset,
    WinograndeRationaleRefinementDataset,
    PIQARationaleGenerationDataset,
    PIQARationaleRefinementDataset,
    HellaswagRationaleGenerationDataset,
    HellaswagRationaleRefinementDataset
)

from datasets import Dataset
from typing import Dict

from trl import DPOConfig

# Factory method to build dataset of different types given the dataset name specified in args
# Return both for train and test split, if available, and if test split is not available, return validation split
def build_dataset(args, tokenizer=None, split="train"):

    if "json" in args.cache_dataset_path:
        with open(args.cache_dataset_path, "r") as f:
            dataset = json.load(f)
    else:
        if args.hf_dataset_config != "":
            dataset = load_dataset(args.hf_dataset_path, args.hf_dataset_config, trust_remote_code=True)
        else:
            dataset = load_dataset(args.hf_dataset_path, trust_remote_code=True)

    if args.dataset_name == "CoT":
        return CoTDataset(dataset, args, tokenizer, "train")
    
    elif args.dataset_name == "gsm8k":
        if args.generation_type == "rationale":
            return GSM8KRationaleGenerationDataset(dataset, args, tokenizer, split=split)
        elif args.generation_type == "rationale_refinement":
            return GSM8KRationaleRefinementDataset(dataset, args, tokenizer, split=split)

    elif args.dataset_name == "winogrande":
        if args.generation_type == "rationale":
            return WinograndeRationaleGenerationDataset(dataset, args, tokenizer, split=split)
        elif args.generation_type == "rationale_refinement":
            return WinograndeRationaleRefinementDataset(dataset, args, tokenizer, split=split)

    elif args.dataset_name == "piqa":
        if args.generation_type == "rationale":
            return PIQARationaleGenerationDataset(dataset, args, tokenizer, split=split)
        elif args.generation_type == "rationale_refinement":
            return PIQARationaleRefinementDataset(dataset, args, tokenizer, split=split)

    elif args.dataset_name == "hellaswag":
        if args.generation_type == "rationale":
            return HellaswagRationaleGenerationDataset(dataset, args, tokenizer, split=split)
        elif args.generation_type == "rationale_refinement":
            return HellaswagRationaleRefinementDataset(dataset, args, tokenizer, split=split)
        
    elif args.dataset_name == "sro":

        dataset = Dataset.from_dict(dataset)
        original_columns = dataset.column_names

        dataset = dataset.shuffle(seed = args.seed)
        
        if args.sanity_check:
            dataset = dataset.select(range(min(len(dataset), 1000)))

        def return_prompt_and_responses(samples) -> Dict[str, str]:

            return {
                "prompt": samples["prompt"],
                "chosen": samples["chosen"],
                "rejected": samples["rejected"],
            }

        return dataset.map(
            return_prompt_and_responses,
            batched = True,
            num_proc = args.num_proc,
            remove_columns = original_columns,
        )

# Function to build dataloader
def build_dataloader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=dataset.collate_fn)


def build_module(model, tokenizer, args, cache_path=None):
    if args.guide_training:
        pass
    elif args.guide_inference:
        return InferenceModule(model=model, tokenizer=tokenizer, generation_mode=args.generation_type, args=args, cache_path=cache_path)
    else:
        return WarmupInstructionTuningModule(model=model, tokenizer=tokenizer, args=args)


def build_dpo_training_args(args, script_args):

    training_args = DPOConfig(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        # max_steps=script_args.max_steps,
        num_train_epochs=args.max_epochs,
        logging_steps=script_args.logging_steps,
        # save_steps=script_args.save_steps,
        save_strategy = "epoch",
        save_total_limit = 1,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        eval_strategy="steps",
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name=args.name,
        gradient_checkpointing_kwargs=dict(use_reentrant=script_args.gradient_checkpointing_use_reentrant),
        seed=script_args.seed,
        save_only_model = True,
    )

    return training_args

def get_merged_list(args, device, data_dict):

    with open(f"./dataset/{args.dataset_save_name}_{device}.json", "r") as f:
        data = json.load(f)

    if data_dict is None:
        data_dict = {key: [] for key in list(data.keys())}

    for key in list(data.keys()):
        data_dict[key].extend(data[key])

    return data_dict