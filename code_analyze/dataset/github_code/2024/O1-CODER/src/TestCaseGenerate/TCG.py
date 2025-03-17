import argparse
import os
import torch
import transformers
from accelerate import PartialState
from datasets import load_dataset, Dataset
from peft import LoraConfig
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    logging,
    set_seed,
)
from trl import DPOTrainer, SFTTrainer
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
from peft import PeftModel, LoraConfig
import torch.distributed as dist
import json
from torch.utils.data import DataLoader, DistributedSampler
import tqdm
import re
import io, sys
import multiprocessing


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dataset_text_field", type=str, default="text")

    parser.add_argument("--max_seq_length", type=int, default=1024 * 4)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--bf16", type=bool, default=True)

    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="finetune_deepseek1.3_instruct_o1_format_SFT")
    parser.add_argument("--num_proc", type=int, default=None)

    parser.add_argument("--model_path", type=str, default="/data/FastSSD/LLM_Models/deepseek-coder-1.3b-instruct/")
    return parser.parse_args()

def build_test_part(A, B):
    test_part = """```case
# input:
{}
# output:
{}
```
""".format(A.strip(), B.strip())
    if len(test_part) > 100:
        raise ValueError
    return test_part

def build_TACO_SFT(item):
    return_item_lst = []
    for solve in eval(item['solutions']):
        in_out_case = eval(item['input_output'])
        test_case = [[x[0], x[1]] for x in zip(in_out_case['inputs'], in_out_case['outputs'])]
        test_case = random.choices(test_case, k=min(3, len(test_case)))
        test_part = "".join([build_test_part(item[0], item[1]) for item in test_case])
        templt = '''### Instruction
Please complete the task in the code part and generate some test case in the test part that can be used to test the quality of the generated code.
### Problem
{}
### Code Part
{}
```python
{}
```
### Test Part
[Generate 3 test cases here to validate the code.]
{}
<|EOT|>
'''.format(item['question'].strip(), ", ".join(eval(item['tags'])[:20]), solve.strip(), test_part.strip())
        return_item_lst.append(templt)
    return return_item_lst

def main(args):
    # config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    lora_config = LoraConfig(
        r=1,
        target_modules=[
            "q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        attention_dropout=args.attention_dropout,
        device_map={"": PartialState().process_index},
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    select_data = json.load(open("select_data_train_can_use.json"))
    SFT_dataset = []
    length = []
    too_long_count = 0
    for item in tqdm.tqdm(select_data):
        try:
            if eval(item['solutions']) != []:
                item_str_lst = build_TACO_SFT(item)
                if len(item_str_lst) >= 30:
                    item_str_lst = random.choices(item_str_lst, k=30)
                for item_str in item_str_lst:
                    if len(item_str) > 3000:
                        too_long_count += 1
                        continue
                    length.append(len(item_str))
                    SFT_dataset.append(item_str)
        except:
            pass
    print("Avg Length Len:", np.mean(length), len(SFT_dataset), too_long_count)
    SFT_dataset = Dataset.from_dict({"text": SFT_dataset})
    SFT_dataset = SFT_dataset.shuffle(seed=42)

    # setup the SFT trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=SFT_dataset,
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            max_steps=5000,
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler_type,
            weight_decay=args.weight_decay,
            bf16=args.bf16,
            fp16=False,
            logging_strategy="steps",
            logging_steps=1,
            output_dir=args.output_dir,
            optim="paged_adamw_8bit",
            seed=args.seed,
        ),
        peft_config=lora_config,
        dataset_text_field="text",
    )
    # launch
    print("Training SFT...")
    trainer.train()
    model.save_pretrained(os.path.join(args.output_dir, "SFT_final_checkpoint/"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "SFT_final_checkpoint/"))
    print("SFT Training Done!")


# accelerate launch TCG_SFT.py
if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    logging.set_verbosity_error()
    main(args)
