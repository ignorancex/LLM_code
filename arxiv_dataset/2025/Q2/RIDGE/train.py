import os
import sys
import argparse
import jsonlines
import warnings
import random
from datetime import datetime
warnings.filterwarnings("ignore")

import torch
from accelerate import PartialState
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TrainingArguments
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from trl import SFTTrainer


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable: {(100 * trainable_params / all_param):.4f} %"
    )


INSTRUCTION = """\
<s>[INST] <<SYS>>
You are a helpful assistant and good at considering and generating document layout.
<</SYS>>

Fill in the missing values in <FILL_i> with the correct coordinates. The "box" format is [x1,y1,x2,y2], representing the top-left corner (x1,y1) and the bottom-right corner (x2,y2). Ensure that x1 < x2 and y1 < y2.
Output in json format."""

def format_instruction(input: str, output: str, inference=False):
    if inference:
        prompt = f"""\
{INSTRUCTION}

{input}
[/INST]"""
        return prompt

    prompt = f"""\
{INSTRUCTION}

{input}
[/INST] {output}</s>"""
    return prompt


def generate_instruction_dataset(data_point, inference=False, testing=False):
    if not testing:
        return {"text": format_instruction(data_point["input"], data_point["output"], inference=inference)}
    else:
        return {"text": format_instruction(data_point["input"], None, inference=inference)}
     

def process_dataset(data: Dataset, inference=False, testing=False):
    if not inference and not testing:
        return (
            data.shuffle(seed=42)
            .map(lambda datapoint: generate_instruction_dataset(data_point=datapoint, inference=inference)).remove_columns(["input", "output"])
        )
    elif testing:
        return (
            data.map(lambda datapoint: generate_instruction_dataset(data_point=datapoint, inference=inference, testing=True)).remove_columns(["input"])
        )
    else:
        return (
            data.map(lambda datapoint: generate_instruction_dataset(data_point=datapoint, inference=inference)).remove_columns(["input", "output"])
        )


def main(args):
    # create the checkpoint directory
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    # load dataset
    data_files = {"train": "train.jsonl"}
    # data_files = {"train": "train.jsonl", "validation": "val.jsonl"}
    dataset = load_dataset("json", data_files=data_files, download_mode="force_redownload")
    # print(dataset)
    
    # process dataset
    dataset = process_dataset(dataset)
    # print(dataset)
    # print(dataset["train"]["text"][0])

    # quantization config
    if args.quantization:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.fp16_or_bf16 == "bf16" else torch.float16,
        )
    else:
        nf4_config = None

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        # cache_dir=cache_dir,
        quantization_config=nf4_config,
        device_map=args.device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    '''
    # observe max_seq_len
    max_seq_len = 0
    seq_len_list = []
    for text in dataset["train"]["text"]:
        # print("Text:", text, '\n')
        # for token_id in tokenizer(text).input_ids:
        #     print(token_id, "->", tokenizer.decode(token_id))
        length = len(tokenizer(text).input_ids)
        seq_len_list.append(length)
        max_seq_len = max(max_seq_len, length)
    print(f"max_seq_len: {max_seq_len}")
    # plot histogram
    import matplotlib.pyplot as plt
    plt.hist(seq_len_list, bins=100)
    plt.savefig("seq_len_hist.png")
    exit()
    '''

    # remove data that exceeds max_seq_len
    dataset["train"] = dataset["train"].filter(lambda x: len(tokenizer(x["text"]).input_ids) <= args.max_seq_len)

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # lora config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)

    training_arguments = TrainingArguments(
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing_kwargs={'use_reentrant':False} if args.ddp else None,
        optim="paged_adamw_32bit",
        weight_decay=args.weight_decay,
        max_grad_norm=0.3,
        logging_steps=args.logging_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        fp16=True if args.fp16_or_bf16 == "fp16" else False,
        bf16=True if args.fp16_or_bf16 == "bf16" else False,
        num_train_epochs=args.epoch,
        # evaluation_strategy="steps",
        # eval_steps=0.2,
        warmup_ratio=args.warmup_ratio,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        group_by_length=True,
        output_dir=args.ckpt_dir,
        report_to="tensorboard",
        save_safetensors=True,
        seed=42,
    )

    # enable model parallelism
    if args.world_size > 1:
        model.is_parallelizable = True

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        # eval_dataset=dataset["validation"],
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        tokenizer=tokenizer,
        args=training_arguments,
    )

    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != 'win32':
        model = torch.compile(model)

    # Resume training from checkpoint if specified
    if args.from_ckpt:
        trainer.train(resume_from_checkpoint=args.ckpt_name)
    else:
        trainer.train()

    # Save trained model
    trainer.model.save_pretrained(args.ckpt_dir)


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--ckpt_dir", type=str, default=f"./models/llama3.1-lora-{datetime.now().strftime('%Y%m%d-%H%M%S')}") # save checkpoint
    parser.add_argument("--dataset_path", type=str, default="datasets.jsonl")
    parser.add_argument("--epoch", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--fp16_or_bf16", type=str, choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--from_ckpt", action="store_true")
    parser.add_argument("--ckpt_name", type=str, default=None)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=8000)
    parser.add_argument("--quantization", type=bool, default=False)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--valset_size", type=int, default=0)
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--world_size", type=int, default=int(os.environ.get("WORLD_SIZE", 1)))
    args = parser.parse_args()

    # DistributedDataParallel
    print("world_size:", args.world_size)
    args.ddp = args.world_size != 1
    if args.ddp:
        device_string = PartialState().process_index
        args.device_map = {'': device_string}
        print("Use device map:", args.device_map)
        args.gradient_accumulation_steps = args.batch_size // args.micro_batch_size // args.world_size
    else:
        args.gradient_accumulation_steps = args.batch_size // args.micro_batch_size

    # set random seed
    seed = 42
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # split dataset
    train_ratio = 1
    train_writer = jsonlines.open('train.jsonl', mode='w')
    val_writer = jsonlines.open('val.jsonl', mode='w')
    train_data_list = []
    val_data_list = []

    filename = args.dataset_path
    with open(filename, 'r', encoding="utf-8") as f:
        num_sample = sum(1 for _ in f)
        train_num = int(num_sample * train_ratio)
        val_num = num_sample - train_num
        f.seek(0) # reset file pointer

        # shuffle and split dataset
        data = list(jsonlines.Reader(f))
        random.seed(seed)
        random.shuffle(data)
        train_data = data[:train_num]
        if val_num > 0:
            val_data = data[train_num:]

        train_data_list.extend(train_data)
        if val_num > 0:
            val_data_list.extend(val_data)

        f.close()
    
    # save train and val dataset
    train_writer.write_all(train_data_list)
    if len(val_data_list) > 0:
        val_writer.write_all(val_data_list)

    train_writer.close()
    val_writer.close()

    main(args)
