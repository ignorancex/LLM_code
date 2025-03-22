import argparse
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, Dataset
from modeling import OlmoeConfig, OlmoeForCausalLM


def transform_data(data):
    transformed_entries = []

    for entry in data:
        formatted_entry = {
            "instruction": entry["messages"][0]["content"],
            "output": entry["messages"][1]["content"]
        }
        transformed_entries.append(formatted_entry)
    
    return Dataset.from_list(transformed_entries)


def formatting_prompts_func(examples, instruction_key='instruction', input_key='input', output_key='output'):
    # alpaca style prompts
    # also works for gpteacher because gpteacher inherits alpaca prompt
    # https://github.com/huggingface/trl/pull/444#issue-1760952763
    instruction = examples[instruction_key]
    if 'input' in examples:
        input_text = examples[input_key]
    else:
        input_text = ''
    response = examples[output_key]

    if len(input_text) > 0:
        text = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
        
        ### Instruction:
        {instruction}
        
        ### Input:
        {input_text}
        
        ### Response:
        {response}
        '''
    else:
        text = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
        
        ### Instruction:
        {instruction}
        
        ### Response:
        {response}
        '''

    return text


def run_sft(model, sft_args, task):
    if task == "math":
        ds_name = "TIGER-Lab/MathInstruct"
        format_keys = {}
    elif task == "coding":
        ds_name = "ise-uiuc/Magicoder-Evol-Instruct-110K"
        format_keys = {"output_key": "response"}
    elif task == "general":
        ds_name = "allenai/tulu-3-sft-mixture"
        format_keys = {}
    train_dataset = load_dataset(ds_name, split='train')

    if task == "general":
        train_dataset = transform_data(train_dataset)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=sft_args,
        formatting_func=lambda examples: formatting_prompts_func(
                                    examples, **format_keys
                                ),
    )
    trainer.train()


def main(args):
    sft_args = SFTConfig(
                        per_device_train_batch_size=args.per_device_batch_size, 
                        packing=True,
                        learning_rate=args.lr,
                        gradient_checkpointing=True,
                        optim="adamw_torch_fused",
                        bf16=True,
                        lr_scheduler_type=args.lr_scheduler_type,
                        warmup_ratio=args.warmup_ratio,
                        num_train_epochs=args.num_epochs,
                        output_dir=args.output_dir,
                        report_to="none"
                    )
    config = OlmoeConfig.from_pretrained(args.model_config)
    model = OlmoeForCausalLM.from_pretrained('allenai/OLMoE-1B-7B-0924', 
                                        config=config, 
                                        device_map="auto", 
                                        attn_implementation="flash_attention_2",
                                        torch_dtype='bfloat16')
    run_sft(model, sft_args, args.task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="olmoe")
    parser.add_argument("--model-config", type=str, default="./model_configs/config_olmoe_ce.json")
    parser.add_argument("--task", type=str, default="math", choices=["general", "coding"])
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--per-device-batch-size", type=int, default=8)
    args = parser.parse_args()
    main(args)