from glob import glob
from tqdm import tqdm

import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq, DataCollatorForTokenClassification
from datasets import Dataset

from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model

from trl import SFTTrainer, SFTConfig

# if unsloth
from unsloth import is_bfloat16_supported, FastLanguageModel

CLASS_NAMES = ['4ASK', '4PAM', '8ASK', '16PAM', 'CPFSK', 'DQPSK', 'GFSK', 'GMSK', 'OQPSK', 'OOK']

def load_model_and_tokenizer(model_id, cache_dir="../../models", local_files_only=True):
    """Load model and tokenizer with error handling."""
    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            local_files_only=local_files_only
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            cache_dir=cache_dir,
            local_files_only=local_files_only
        )
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def load_npy_file(file_path):
    """Load the npy file from the given path."""
    return np.load(file_path)

def get_stats_context(file_path):
    data = load_npy_file(file_path)
    label = file_path.split('/')[-1].split('_')[0]
    des_stats = stats.describe(data)
    stats_summary = {
        'nobs': f'{des_stats.nobs}',
        'min': f"{des_stats.minmax[0]:.5f}",
        'max': f"{des_stats.minmax[1]:.5f}",
        'mean': f"{des_stats.mean:.5f}",
        'variance': f"{des_stats.variance:.5f}",
        'skewness': f"{des_stats.skewness:.5f}",
        'kurtosis': f"{des_stats.kurtosis:.5f}",
    }
    for i in range(10):
        stats_summary[f'moment_{i}'] = f'{stats.moment(data, order=i):.5f}'

    for i in range(1, 5):
        stats_summary[f'kstat_{i}'] = f'{stats.kstat(data, i):.5f}'

    for i in range(1, 3):
        stats_summary[f'kstatvar_{i}'] = f'{stats.kstatvar(data, i):.5f}'


    return str(stats_summary).replace("'", ''), label

def get_example_paths(noise_type='noisySignal'):
    example_paths = {
        '4ASK': f'../../data/unlabeled_10k/test/{noise_type}/4ASK_-0.17dB__081_20250127_164342.npy',
        '4PAM': f'../../data/unlabeled_10k/test/{noise_type}/4PAM_-0.00dB__031_20250127_164618.npy',
        '8ASK': f'../../data/unlabeled_10k/test/{noise_type}/8ASK_-0.11dB__016_20250127_164352.npy',
        '16PAM': f'../../data/unlabeled_10k/test/{noise_type}/16PAM_-0.08dB__058_20250127_145951.npy',
        'CPFSK': f'../../data/unlabeled_10k/test/{noise_type}/CPFSK_-0.03dB__088_20250127_164523.npy',
        'DQPSK': f'../../data/unlabeled_10k/test/{noise_type}/DQPSK_-0.01dB__036_20250127_164655.npy',
        'GFSK': f'../../data/unlabeled_10k/test/{noise_type}/GFSK_-0.05dB__042_20250127_164545.npy', 
        'GMSK': f'../../data/unlabeled_10k/test/{noise_type}/GMSK_-0.12dB__059_20250127_164925.npy',
        'OQPSK': f'../../data/unlabeled_10k/test/{noise_type}/OQPSK_-0.24dB__006_20250127_145655.npy',
        'OOK': f'../../data/unlabeled_10k/test/{noise_type}/OOK_-0.17dB__091_20250127_164311.npy'
    }
    # Ensure the example paths match the class names
    assert set(example_paths.keys()) == set(CLASS_NAMES), "Example paths do not match class names"

    return example_paths

def add_context_to_signal(file_path, noise_type='noiselessSignal'):
    stats_summary, label = get_stats_context(file_path)
    example_paths = get_example_paths(noise_type=noise_type)
    

    input_text = (
    f"Overall Signal Information: {stats_summary}"
    )
    
    prompt_prefix = f"""### Instructions
    You are an expert quantitative analyst in wireless communication modulation.
    Based on your knowledge in wireless communication modulation and the detailed signal statistics provided below, determine the modulation type."""

    prompt_examples = ""
    for idx, ex_path in enumerate(example_paths.values()):
        ex_summary, ex_label = get_stats_context(ex_path)
        prompt_examples += f"""
    ### Example {idx+1}: Overall Signal Information: {ex_summary} ### Answer {idx+1}: {ex_label}"""

    prompt_suffix = f"""
    YOUR ANSWER MUST BE STRICTLY ONE AND ONLY ONE OF {", ".join(CLASS_NAMES)} MODULATION TYPES AND NOTHING ELSE. DO NOT PROVIDE ANY ADDITIONAL INFORMATION OR CONTEXT. No OTHER TEXT, NO BLABBER. DO NOT PROVIDE ANY ADDITIONAL INFORMATION OR CONTEXT. No OTHER TEXT, NO BLABBER.
    ### Question: {input_text}
    ### Response: """

    prompt = prompt_prefix + prompt_examples + prompt_suffix

    return prompt, label

def get_signals_as_text(signal_paths):
    file_paths = glob(signal_paths)
    texts = []
    labels = []
    for file_path in tqdm(file_paths):
        prompt, label = add_context_to_signal(file_path)
        texts.append(prompt)
        labels.append(label)

    return texts, labels

class SignalTextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        
    def __getitem__(self, idx):
        return {
            "prompt": self.texts[idx],
            "completion": self.labels[idx]
        }

    def __len__(self):
        return len(self.labels)
    
    @property
    def column_names(self):
        return ["prompt", "completion"]

def formatting_func(example):
    """Format examples for training"""
    prompt = example["prompt"]
    completion = example["completion"]
    text = f"{prompt}{completion}{tokenizer.eos_token}"
    return {"text": text}

if __name__ == "__main__":

    unsloth_flag = True
    #%% load tokenizer and model
    # model_name = "DeepSeek-R1-Distill-Qwen-32B"
    model_name = "DeepSeek-R1-Distill-Qwen-7B"
    if unsloth_flag:
        model, tokenizer = load_model_and_tokenizer(model_id='deepseek-ai/'+model_name, local_files_only=True)
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        config = LoraConfig(
            r=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=16,
            lora_dropout=0, 
            bias="none", 
            task_type="SEQ_CLS",
        )

        model = get_peft_model(model, config)
        print_trainable_parameters(model)
    
    else:
        max_seq_length = 8000 # Choose any! We auto support RoPE Scaling internally!
        dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
        model, tokenizer = FastLanguageModel.from_pretrained(
                model_name = f"unsloth/{model_name}-unsloth-bnb-4bit",
                max_seq_length = max_seq_length,
                dtype = dtype,
                load_in_4bit = load_in_4bit,
                cache_dir = "../../models",
                # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
            )
        
        model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        task_type="SEQ_CLS",
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = True,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )


    #%% load training data
    EOS_TOKEN = tokenizer.eos_token
    def formatting_prompts_func(examples):
        prompt = examples["prompt"]
        outputs = examples["completion"]
        texts = []
        for prompt, output in zip(prompt, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = prompt + output + EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }

    signal_paths = "../../data/unlabeled_10k/test/noiselessSignal/*.npy"
    texts, labels = get_signals_as_text(signal_paths)
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1)
    
    train_dict = {
    "prompt": train_texts,
    "completion": train_labels
    }
    # Convert to a Hugging Face Dataset
    train_dataset = Dataset.from_dict(train_dict)
    train_dataset = train_dataset.map(formatting_func)

    val_dict = {
    "prompt": val_texts,
    "completion": val_labels
    }
    val_dataset = Dataset.from_dict(val_dict)
    val_dataset = val_dataset.map(formatting_func)
    
    # Create datasets directly with texts and labels
    # train_dataset = SignalTextDataset(train_texts, train_labels)
    # val_dataset = SignalTextDataset(val_texts, val_labels)
    
    #%% load test data
    # test_signal_paths = "../../data/unlabeled_10k/test/noisySignal/*.npy"
    # test_texts, test_labels = get_signals_as_text(test_signal_paths)
    # test_dict = {
    # "prompt": test_texts,
    # "completion": test_labels
    # }
    
    # # Convert to a Hugging Face Dataset
    # test_dataset = Dataset.from_dict(test_dict)
    # test_dataset = SignalTextDataset(test_texts, test_labels)

    # needed for gpt-neo-x tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    if unsloth_flag:
        response_template = " ### Response:"
        collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset = val_dataset,
            tokenizer=tokenizer,
            formatting_func=formatting_func,
            # data_collator=collator,
            # packing=False,
            args=SFTConfig(
                per_device_train_batch_size=4,
                gradient_accumulation_steps=4,
                warmup_steps=10,
                num_train_epochs=20,
                learning_rate=5e-4,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir="model_traning_outputs",
                report_to="none",
                dataset_num_proc=4,
                packing=False,
                save_strategy="epoch",
                save_total_limit=3,
            ),
        )

    else:
        trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        args = TrainingArguments(
            per_device_train_batch_size = 4,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs = 3, # Set this for 1 full training run.
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "unsloth_model_traning_outputs",
            save_strategy="epoch",
            save_total_limit=1,
            report_to = "none", # Use this for WandB etc
            # packing = False, 
            # dataset_text_field = "text",
            # dataset_num_proc = 2,
        ),
        )

    # trainer.args.logging_dir = "outputs"
    trainer.args.report_to = ["tensorboard"]

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

