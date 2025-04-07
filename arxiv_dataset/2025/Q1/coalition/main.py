
import os
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from accelerate import Accelerator

from utils import (
    set_seed, 
    preprocess_data, 
    build_dataset, 
    build_dataloader, 
    build_module, 
    merge_data,
    ScriptArguments,
    build_dpo_training_args
)


from transformers import (
    AutoTokenizer, 
    HfArgumentParser, 
    TrainingArguments, 
    AutoModelForCausalLM, 
)

from trl import DPOTrainer


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Multi-turn LLM Agents Tuning Framework')

    # NOTE: Experiment/Run arguments
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of workers for data loader")
    parser.add_argument("--project", type = str, default = "multi-turn-llm-agents", help="specify name of the project or wandb")
    parser.add_argument("--name", type=str, default="multi-turn-llm-agents", help="Name of the experiment")
    parser.add_argument("--save_dir", type = str, default = "./logs/llama3-8b-warmup-instruction-tuning", help="specify save directory")
    parser.add_argument("--version", type = str, default = "master-ift-warmup-master-open-llama-7b-v2", help="specify name of experiment")

    # NOTE: Data arguments
    parser.add_argument("--dataset_name", type=str, default="CoT", help="Name of the dataset")
    parser.add_argument("--task_name", type=str, default="gsm8k", help="Name of the dataset")
    parser.add_argument("--preprocess_data", type=bool, default=False, help="specify whether to preprocess data")
    parser.add_argument("--filter_samples", type=bool, default=False, help="specify whether to filter samples")
    parser.add_argument("--hf_dataset_path", type=str, default="kaist-ai/CoT-Collection", help="HuggingFace path to the dataset")
    parser.add_argument("--hf_dataset_config", type=str, default="en", help="HuggingFace config to the dataset")
    parser.add_argument("--cache_dataset_path", type=str, default="./dataset/warmup_ift_llama3_8b_dataset.json", help="Cached path to the dataset")
    parser.add_argument("--save_dataset_path", type=str, default="./dataset/warmup_ift_llama3_8b_dataset.json", help="Save path to the dataset")
    parser.add_argument("--dataset_config", type=str, default="", help="Path to the dataset config file")
    parser.add_argument("--dpo_turn1_task_dataset1", type=str, default="./dataset/gsm8k_llama3_8b_multiturn_ift_split1_rationales_turn1.json", help="Path to the dataset config file")
    parser.add_argument("--dpo_turn2_task_dataset1", type=str, default="./dataset/gsm8k_llama3_8b_multiturn_ift_split1_refined_split1_rationales.json", help="Path to the dataset config file")
    parser.add_argument("--dpo_turn1_task_dataset2", type=str, default="./dataset/gsm8k_llama3_8b_multiturn_ift_split2_rationales_turn1.json", help="Path to the dataset config file")
    parser.add_argument("--dpo_turn2_task_dataset2", type=str, default="./dataset/gsm8k_llama3_8b_multiturn_ift_split1_refined_split2_rationales.json", help="Path to the dataset config file")

    # NOTE: Tokenizer arguments
    parser.add_argument("--tokenizer_type", type=str, default="llama3", help="Type of the model")
    parser.add_argument("--tokenizer_path", type=str, default="meta-llama/Meta-Llama-3-8B", help="Tokenizer path")
    parser.add_argument("--add_special_tokens", type=bool, help="specify whether to add special tokens", default=True)
    parser.add_argument("--padding", type=bool, help="specify whether to pad input", default=True)
    parser.add_argument("--truncation", type=bool, help="specify whether to truncate input", default=True)
    parser.add_argument("--return_attention_mask", type=bool, help="specify whether to return attention mask", default=True)
    parser.add_argument("--return_token_type_ids", type=bool, help="specify whether to return token type ids", default=True)
    parser.add_argument("--padding_side", default="left", type=str, help="specify padding side, left for decoder-only LLMs")
    parser.add_argument("--return_tensors", default="pt", type=str, help="specify whether to return torch tensor")
    parser.add_argument("--turn1_max_length", default=1536, type=int, help="specify maximum number of tokens LLM can process for first turn conversation")
    parser.add_argument("--turn2_max_length", default=1792, type=int, help="specify maximum number of tokens LLM can process for second turn conversation")
    parser.add_argument("--model_max_length", default=2048, type=int, help="specify the maximum length for the underlying LLM")

    # NOTE: Model arguments
    parser.add_argument("--model_name", type=str, default="llama3_8b", help="Model path")
    parser.add_argument("--hf_model_path", type=str, default="meta-llama/Meta-Llama-3-8B", help="Model path")
    parser.add_argument("--hf_token", type=str, default="", help="HuggingFace token")
    parser.add_argument("--ckpt_path", type=str, default="", help="Model path")
    parser.add_argument("--ckpt_path1", type=str, default="./logs/llama3_8b_warmup_ift_split1_epoch2.pt", help="Model path")
    parser.add_argument("--ckpt_path2", type=str, default="./logs/llama3_8b_warmup_ift_split2_epoch2.pt", help="Model path")
    parser.add_argument("--log_model", type=bool, default=True, help="specify whether to log model")

    # NOTE: Training arguments
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--max_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--multi_mode_training", type=bool, default=False, help="Specify whether to do multi-mode instruction tuning")
    parser.add_argument("--sro", type=bool, default=False, help="Specify whether to do Selective Rationale Optimization (SRO)")
    parser.add_argument("--guide_training", type=bool, default=False, help="Specify whether to guide the training")
    parser.add_argument("--guide_inference", type=bool, default=False, help="Specify whether to do inference using guide")
    parser.add_argument("--generation_type", type=str, default="rationale", help="Specify the generation type")

    # NOTE: DPO arguments
    parser.add_argument("--dpo_training", type=bool, default=False, help="Specify whether to do DPO training")
    parser.add_argument("--stage_index", type=int, default=1, help="Specify the stage index")
    parser.add_argument("--guide_index", type=int, default=1, help="Specify the guide index")
    parser.add_argument("--sanity_check", type=bool, default=False, help="Specify whether to do sanity check")
    parser.add_argument("--use_peft", type=bool, default=False, help="Specify whether to use PEFT")
    parser.add_argument("--sro_ckpt", type=str, default="", help="Specify the SRO checkpoint")
    parser.add_argument("--num_proc", type=int, default=64, help="Specify the number of processes")

    args = parser.parse_args()

    # Set the seed
    set_seed(args.seed)

    if args.preprocess_data:
        print("Preprocessing data")
        preprocess_data(args)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, token=args.hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = args.padding_side

    logger = pl.loggers.WandbLogger(project = args.project, name = args.name, version = args.version, mode = "disabled",
                                save_dir = args.save_dir, log_model = args.log_model)

    if args.guide_training:
        pass

    elif args.guide_inference:
        dataset = build_dataset(args, tokenizer)
        dataloader = build_dataloader(dataset, args.per_device_eval_batch_size, shuffle=False, num_workers=args.num_workers)

        devices = [i for i in range(torch.cuda.device_count())]

        model = AutoModelForCausalLM.from_pretrained(args.hf_model_path, token = args.hf_token, torch_dtype = torch.bfloat16)
        if args.ckpt_path != "":
            model.load_state_dict(torch.load(args.ckpt_path, map_location = "cpu"))
        
        module = build_module(model, tokenizer, args, cache_path = args.save_dataset_path)
        trainer = pl.Trainer(devices=devices, precision="bf16-true", strategy="deepspeed_stage_2", accelerator="gpu", logger=logger)
        trainer.predict(module, dataloaders = dataloader)

        merge_data(args, args.save_dataset_path)

    elif args.sro:
        parser = HfArgumentParser(ScriptArguments)
        script_args = parser.parse_args_into_dataclasses()[0]

        torch_dtype = torch.float
        if script_args.model_dtype == "float16":
            torch_dtype = torch.float16
        elif script_args.model_dtype == "bfloat16":
            torch_dtype = torch.bfloat16

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, token = args.hf_token)
        tokenizer.pad_token = tokenizer.eos_token

        train_dataset = build_dataset(args = args)
        train_dataset = train_dataset.filter(
            lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
            and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length
        )

        # 3. Load evaluation dataset
        eval_dataset = build_dataset(args = args)
        eval_dataset = eval_dataset.filter(
            lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
            and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length
        )

        model = AutoModelForCausalLM.from_pretrained(
            script_args.hf_model_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch_dtype,
            load_in_4bit=script_args.load_in_4bit,
            device_map={"": Accelerator().local_process_index},
            token = args.hf_token
        )

        if args.sro_ckpt != "":
            model.load_state_dict(torch.load(args.sro_ckpt, map_location = "cpu"))
        
        model.config.use_cache = False

        if script_args.ignore_bias_buffers:
            # torch distributed hack
            model._ddp_params_and_buffers_to_ignore = [
                name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
            ]

        training_args = build_dpo_training_args(args = args, script_args = script_args)

        if args.use_peft:
            peft_config = LoraConfig(
                r=script_args.lora_r,
                lora_alpha=script_args.lora_alpha,
                lora_dropout=script_args.lora_dropout,
                target_modules=[
                    "q_proj",
                    "v_proj",
                    "k_proj",
                    "out_proj",
                    "fc_in",
                    "fc_out",
                    "wte",
                ],
                bias="none",
                task_type="CAUSAL_LM",
            )
        
        dpo_trainer = DPOTrainer(
            model,
            ref_model=None,
            args=training_args,
            beta=script_args.beta,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            # peft_config=peft_config,
            max_prompt_length=script_args.max_prompt_length,
            max_length=script_args.max_length,
        )

        # 6. train
        dpo_trainer.train()
        dpo_trainer.save_model(script_args.output_dir)

        # 7. save
        output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
        dpo_trainer.model.save_pretrained(output_dir)        


    elif args.multi_mode_training:
        
        dataset = build_dataset(args, tokenizer)
        dataloader = build_dataloader(dataset, args.per_device_train_batch_size, shuffle=True, num_workers=args.num_workers)

        devices = [i for i in range(torch.cuda.device_count())]

        model = AutoModelForCausalLM.from_pretrained(args.hf_model_path, token = args.hf_token, torch_dtype = torch.bfloat16)
        if args.ckpt_path != "":
            model.load_state_dict(torch.load(args.ckpt_path, map_location = "cpu"))

        module = build_module(model, tokenizer, args)
        checkpoint_callback = ModelCheckpoint(dirpath = args.save_dir, every_n_epochs = 1, save_weights_only = True)

        trainer = pl.Trainer(max_epochs=args.max_epochs, devices=devices, precision="bf16-true", strategy="deepspeed_stage_2_offload",
                             accelerator="gpu", callbacks=[checkpoint_callback], logger=logger, gradient_clip_val=1.0, accumulate_grad_batches=4)

        trainer.fit(module, train_dataloaders = dataloader)