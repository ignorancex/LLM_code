from unsloth import FastLanguageModel
import torch
import pandas as pd
import json
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported


max_seq_length = 2048 
dtype = None 
load_in_4bit = True 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",#"unsloth/Meta-Llama-3.1-405B-bnb-4bit", #"unsloth/Meta-Llama-3.1-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    
)


model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)




# Load the CSV file
file_path = "YOUR TRAINING DATASET FILE PATH"
df = pd.read_csv(file_path)

# Prepare the list to hold all formatted data
formatted_data = []
formatted_answers = []

# Iterate over each row in the dataframe
for index, row in df.iterrows():
    dna_sequence = row["DNA Sequence"].upper()
    drug_class = row["Drug Class"]

    # Create the formatted message as required
    formatted_message = {
        "messages": [
            {"role": "system", "content": "You are a DNA and antimicrobial resistance expert."},
            {"role": "user", "content": f"Tell me the resistance drug among drugs (Sulfonamides, Aminoglycosides, betalactams, Glycopeptides, Tetracyclines, Phenicol, Fluoroquinolones, MLS, Multi-drug_resistance) with DNA sequence ({dna_sequence})?"},
            #{"role": "assistant", "content": f"answer: {drug_class}"}
        ]
    }
    formatted_answer = {
        "messages": [
            {"role": "system", "content": "You are a DNA and antimicrobial resistance expert."},
            {"role": "user", "content": f"Tell me the resistance drug among drugs (Sulfonamides, Aminoglycosides, betalactams, Glycopeptides, Tetracyclines, Phenicol, Fluoroquinolones, MLS, Multi-drug_resistance) with DNA sequence ({dna_sequence})?"},
            {"role": "assistant", "content": f"answer: {drug_class}"}
        ]
    }

    # Append to the list
    formatted_data.append(formatted_message)
    formatted_answers.append(formatted_answer)





# Alpaca template
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

# Prepare the list to hold the formatted prompts
instructions = []
inputs = []
outputs = []

# Iterate over formatted_answers and split into instruction, input, and output
for formatted_answer in formatted_answers:
    instruction = "Tell me the resistance drug given a DNA sequence and a list of drug classes."
    user_input = formatted_answer["messages"][1]["content"]  # User's question
    assistant_response = formatted_answer["messages"][2]["content"]  # Assistant's answer

    instructions.append(instruction)
    inputs.append(user_input)
    outputs.append(assistant_response)

# Create the dataset in Alpaca prompt style
dataset = Dataset.from_dict({
    "instruction": instructions,
    "input": inputs,
    "output": outputs
})


# change with Alpaca template
def formatting_prompts_func(examples):
    texts = []
    for instruction, input, output in zip(examples["instruction"], examples["input"], examples["output"]):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

# change Dataset
dataset = dataset.map(formatting_prompts_func, batched=True)




trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# train
trainer_stats = trainer.train()


