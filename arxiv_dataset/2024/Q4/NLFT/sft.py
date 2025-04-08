import json
import random
import sys
import numpy as np
import torch
import os

from datasets import Dataset
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
    MODEL_TYPE_TO_PEFT_MODEL_MAPPING,
    PeftModel,
    PeftModelForCausalLM,
)
from peft.utils import _prepare_prompt_learning_config
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainerCallback,
    TrainingArguments,
)
from trl import SFTTrainer

from finetune_unlikelihood_dynamic_correct import MyDataCollator, read_json, read_jsonl, SavePeftModelOnEpochEndCallback
from prompter import Prompter

torch.cuda.empty_cache()

def main():
    _name = 'NLFT800'
    base_model = '/root/autodl-tmp/cache_dir/LLM-Research/Meta-Llama-3-8B'
    data_path = './data/{}.json'.format(_name)
    output_dir = './saved_models/lora/sft_llama3_8b_{}'.format(_name)

    # 超参设置
    prompt_template_name = 'alpaca' # templates/alpaca.json
    batch_size = 4
    micro_batch_size = 1
    gradient_accumulation_steps = batch_size // micro_batch_size
    num_epochs = 10
    learning_rate = 5e-5
    cutoff_len = 8192
    lr_scheduler = 'cosine'
    lora_r = 16
    lora_alpha = 16
    lora_dropout = 0.05
    lora_target_modules = ['gate_proj', 'down_proj', 'up_proj']
    train_on_inputs = False
    add_eos_token = False
    group_by_length = False
    wandb_project = 'SFTv2'
    wandb_run_name = 'llama3_8b_sft_{}_epoch_{}'.format(_name, num_epochs)
    wandb_watch = ''
    wandb_log_model = ''
    resume_from_checkpoint = False
    seed = 42

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Params using prompt template {prompt_template_name}\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"lr_scheduler: {lr_scheduler}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"seed: {seed}\n"
        )

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    prompter = Prompter(prompt_template_name)
    device_map = 'auto'
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
        print("gradient_accumulation_steps: ", gradient_accumulation_steps)

    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # use_wandb = True
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    os.environ["WANDB_API_KEY"] = '61b242f346c474bf3bfa23802219c894e1d9aad4' # wy

    # bnb_config = BitsAndBytesConfig(
    #     load_in_8bit_fp32_cpu_offload=True
    # )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        # quantization_config=bnb_config,
        # attn_implementation="flash_attention_2",
        # attn_implementation="flash_attention_2", # FIXME: 本来都是注释的，试试 flash attn
        device_map=device_map,
        trust_remote_code=True,
        use_auth_token=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = prepare_model_for_int8_training(model)
    # model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # TODO: resume_from_checkpoint?
    
    def get_peft_model(model, peft_config, adapter_name: str = "default"):
        """
        Returns a Peft model object from a model and a config.

        Args:
            model ([`transformers.PreTrainedModel`]): Model to be wrapped.
            peft_config ([`PeftConfig`]): Configuration object containing the parameters of the Peft model.
        """
        model_config = getattr(model, "config", {"model_type": "custom"})
        if hasattr(model_config, "to_dict"):
            model_config = model_config.to_dict()

        peft_config.base_model_name_or_path = model.__dict__.get("name_or_path", None)

        if peft_config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys() and not peft_config.is_prompt_learning:
            return PeftModel(model, peft_config, adapter_name=adapter_name)
        if peft_config.is_prompt_learning:
            peft_config = _prepare_prompt_learning_config(peft_config, model_config)
        return PeftModelForCausalLM(model, peft_config, adapter_name=adapter_name)

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    print("pre-trained model's BOS EOS and PAD token id:", bos, eos, pad)
    # tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = read_json(data_path) if data_path.endswith(".json") else read_jsonl(data_path)
    else:
        raise NotImplementedError

    file_name = os.path.join("templates", f"{prompt_template_name}.json")
    with open(file_name) as fp:
        template = json.load(fp)
    
    SYSTEM_PROMPT = [
        {"role": "system", "content": "{instruction}"},
        {"role": "user", "content": "{input}"}
    ]
    template["prompt_no_input"] = tokenizer.apply_chat_template(SYSTEM_PROMPT, tokenize=False, add_generation_prompt=True).replace("<|begin_of_text|>", "")
    print(template["prompt_no_input"])
    train_processed = []
    for ix, x in enumerate(data):
        x_judgment = x['judgment']
        x_score = x['score'] if 'score' in x else None
        if x_score is not None and x_score >= 7:
            x_judgment = None
        x_input = x['input']
        x_instruction = x['instruction']
        x_out = x['output']
        x_i_ans = x['i_ans'] if "i_ans" in x else None
        x_answer = x['answer']
        # if x_input:
        #     x_instruction = f"{x_instruction}\n{x_input}"
        if x_judgment is not None:
            base = template["prompt_no_input"].format(instruction=x_instruction, input=x_input)
            x_new = {
                # 'output': x_out,
                'output': x_answer,
                'input': None,
                'instruction_list': [base],
                # 'i_ans': x_i_ans,
                # 'answer': x_answer,
                # 'score': x_score,
                # "polarity": 0,
            }
            train_processed.append(x_new)
        else:
            base = template["prompt_no_input"].format(instruction=x_instruction, input=x_input)
            x_new = {
                # 'output': x_out,
                'output': x_answer,
                'input': None,
                'instruction_list': [base],
                # 'i_ans': x_i_ans,
                # 'answer': x_answer,
                # 'score': x_score,
                # "polarity": 1,
            }
            train_processed.append(x_new)

    train_processed = Dataset.from_list(train_processed)
    print(f"num of training data: {len(train_processed)}")

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=False,
            padding=False,
            return_tensors=None,
        )
        if len(result["input_ids"]) > cutoff_len:  # truncate from left side to keep the response complete
            n_overflow = len(result["input_ids"]) - cutoff_len
            result["input_ids"] = result["input_ids"][-cutoff_len:]
            result["attention_mask"] = result["attention_mask"][-cutoff_len:]
        else:
            n_overflow = 0
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        result["n_overflow"] = n_overflow
        return result, n_overflow

    def generate_and_tokenize_prompt(data_point):
        instructions = data_point['instruction_list']
        tokenized_full_prompt_list = []
        for i_i, instruction in enumerate(instructions):
            data_point['instruction'] = instruction
            full_prompt = prompter.generate_prompt(
                data_point, output=True)

            tokenized_full_prompt, n_overflow_full = tokenize(full_prompt)
            if not train_on_inputs:
                user_prompt = prompter.generate_prompt(
                    data_point, output=False)

                tokenized_user_prompt, n_overflow_user = tokenize(
                    user_prompt, add_eos_token=add_eos_token)

                user_prompt_len = len(tokenized_user_prompt["input_ids"])
                offset = n_overflow_full - n_overflow_user
                user_prompt_len = user_prompt_len - offset
                if add_eos_token:
                    user_prompt_len -= 1
                if user_prompt_len > 0:
                    tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]  # TODO: Speed up?
                assert len(tokenized_full_prompt["labels"]) == len(tokenized_full_prompt["input_ids"])
                if i_i == 0:
                    answer_len = len(tokenized_full_prompt["labels"]) - user_prompt_len
                elif i_i == 1:
                    answer_len2 = len(tokenized_full_prompt["labels"]) - user_prompt_len
                    assert answer_len == answer_len2
                tokenized_full_prompt_list.append(tokenized_full_prompt)

        assert len(tokenized_full_prompt_list) == 1
        tokenized_full_prompt_base = tokenized_full_prompt_list[0]
        return tokenized_full_prompt_base

    train_data = train_processed.map(generate_and_tokenize_prompt)

    training_args = TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=0.1,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        bf16=True,
        logging_steps=1,
        optim="adamw_torch",
        evaluation_strategy="no",
        save_strategy="steps",
        eval_steps=None,
        save_steps=1000,
        lr_scheduler_type=lr_scheduler,
        output_dir=output_dir,
        save_total_limit=2,
        load_best_model_at_end=False,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=False,
        report_to="wandb" if use_wandb else None,
        run_name=wandb_run_name if use_wandb else None,
    )

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    def prepare_sample_text(example):
        """Prepare the text from a sample of the dataset."""
        text = f"Question: {example['instruction']}\n\nAnswer: {example['output']}"
        return text

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        tokenizer=tokenizer,
        max_seq_length=None, # 没问题
        args=training_args,
        data_collator=MyDataCollator(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding="max_length"
        ),
        packing=True, # 试试，好像等价于 ConstantLengthDataset
        formatting_func=prepare_sample_text,
        callbacks=[
            SavePeftModelOnEpochEndCallback(output_dir=output_dir),
        ],
    )
    model.config.use_cache = False
    
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    if local_rank == 0:
        trainer.save_model(output_dir)
        output_dir = os.path.join(output_dir, "final_checkpoint")
        os.makedirs(output_dir, exist_ok=True)
        pytorch_model_path = os.path.join(output_dir, "pytorch_model.bin")
        torch.save({}, pytorch_model_path)
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

if __name__ == '__main__':
    main()
