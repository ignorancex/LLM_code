import numpy as np


DATA_DIRECTORY = "datasets_directory/MMLU/data"


def get_indices_for_choices(engine):
    if "Llama-2-7b-hf" in engine:
        INDICES_FOR_CHOICES = [319, 350, 315, 360]
    elif "Meta-Llama-3-8B" in engine:
        INDICES_FOR_CHOICES = [362, 426, 356, 423]
    elif "Meta-Llama-3" in engine:
        INDICES_FOR_CHOICES = [362, 426, 356, 423]
    elif "Qwen2.5" in engine:
        INDICES_FOR_CHOICES = [362, 425, 356, 422]
    return INDICES_FOR_CHOICES


def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


choices = ["A", "B", "C", "D"]


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


from torch.utils.data import Dataset as torchDataset
class MMLU_N_Shot_Dataset(torchDataset):
    def __init__(self, model_name, tokenizer, data_directory = DATA_DIRECTORY, use_train_split=False, \
                 verbose=False, ntrain=5, eval_start_p=.75, train_end_p=.75, subjects_to_use=None, \
                 make_data_wrong=False, device="cuda"):
        import pandas as pd
        import os
        self.prompts = []
        self.labels = []
        self.texts = []
        self.device = device
        self.verbose = verbose
        self.subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(data_directory, "test")) if "_test.csv" in f])
        if subjects_to_use == "all":
            subjects_to_use = self.subjects
        else:
            # subjects_to_use contains subject names, find their indices
            subjects_to_use = [s for s in self.subjects if s in subjects_to_use]
        print("subjects_to_use", subjects_to_use, self.subjects)
        self.tokenizer = tokenizer
        for subject in subjects_to_use:
            dev_df = pd.read_csv(os.path.join(data_directory, "dev", subject + "_dev.csv"), header=None)[:ntrain]
            test_df = pd.read_csv(os.path.join(data_directory, "test", subject + "_test.csv"), header=None)  # TEST CHANGE ONLY
            start = 0 if use_train_split else int(test_df.shape[0]*eval_start_p)
            stop = int(test_df.shape[0]*train_end_p) if use_train_split else test_df.shape[0]
            for i in range(start, stop):
                k = ntrain
                prompt_end = format_example(test_df, i, include_answer=False)
                train_prompt = gen_prompt(dev_df, subject, k)
                prompt = train_prompt + prompt_end
                label = test_df.iloc[i, test_df.shape[1]-1]
                text = prompt + " " + label
                self.prompts.append(prompt)
                self.labels.append(label)
                self.texts.append(text)
        print(f"Loaded a total of {len(self.texts)} examples from {subjects_to_use} prior to truncation")

    def shuffle(self, seed=42):
        import random
        combined_list = list(zip(self.texts, self.prompts, self.labels))
        random.seed(seed)
        random.shuffle(combined_list)
        self.texts, self.prompts, self.labels = zip(*combined_list)
        self.texts   = list(self.texts)
        self.prompts = list(self.prompts)
        self.labels  = list(self.labels)

    def truncate_to_seqlen_n_samples(self, seqlen, n_samples):
        tokens_to_give = seqlen*n_samples
        current_token_count = 0
        for i in range(len(self.texts)):
            i == 1 and print(f"Truncation: self.tokenizer(self.texts[i])['input_ids']: {self.tokenizer(self.texts[i])['input_ids']}")
            if current_token_count + len(self.tokenizer(self.texts[i])["input_ids"]) > tokens_to_give:
                break
            current_token_count += len(self.tokenizer(self.texts[i])["input_ids"])
        print(f"Truncated to {i} samples, current_token_count: {current_token_count}, tokens_to_give: {tokens_to_give}")
        self.texts = self.texts[:i]
        self.prompts = self.prompts[:i]
        self.labels = self.labels[:i]

    def to_hf_dataset(self):
        from datasets import Dataset as hfDataset
        dataset = hfDataset.from_dict({
            "text": self.texts,
        })
        return dataset
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        prompt = self.texts[idx]
        inputs = self.tokenizer(prompt, return_tensors="pt", padding="longest", truncation=False).to(self.device)
        inputs = {k:v.squeeze(0) for k, v in inputs.items()}
        idx <= 3 and self.verbose and print(f"DATALOADER INPUT IDS SHAPE {inputs["input_ids"].shape}") # DATALOADER INPUT IDS SHAPE torch.Size([2048])
        idx <= 3 and self.verbose and print(f"INPUT IDS {inputs["input_ids"]}")
        return inputs 


def get_mmlu_dataset(nsamples, seed, seqlen, model, data_directory = DATA_DIRECTORY, subjects_to_use = "all"):
    """Follows the GPTQ repo convention for preparing a calibration dataset for MMLU_MCQA"""
    import torch
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    calibration_dataset_no_padding = MMLU_N_Shot_Dataset(data_directory=data_directory, model_name=model, tokenizer=tokenizer, use_train_split=True, \
                 verbose=False, ntrain=5, eval_start_p=.75, train_end_p=.75, subjects_to_use=subjects_to_use, \
                 make_data_wrong=False)
    calibration_dataset_no_padding.shuffle(seed=seed) # Same shuffle function as used in measure_importance.py
    calibration_dataset_no_padding.truncate_to_seqlen_n_samples(seqlen, nsamples)
    calibration_dataset_no_padding = calibration_dataset_no_padding.to_hf_dataset()

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    def custom_tokenize_function_no_padding(examples):
        return tokenizer(examples["text"], truncation=False, padding=False, max_length=seqlen, return_tensors="pt")
    calibration_dataset_no_padding = calibration_dataset_no_padding.map(custom_tokenize_function_no_padding, batched=False, num_proc=1, remove_columns=["text"])

    trainloader = []
    tokens_to_give = seqlen*nsamples
    current_token_count = 0
    for i in range(len(calibration_dataset_no_padding)):
        current_token_count += len(calibration_dataset_no_padding[i]["input_ids"][0])
        inp = calibration_dataset_no_padding[i]["input_ids"]
        inp = torch.tensor(inp)  
        tar = inp.clone()  
        tar[:, :-1] = -100
        i == 0 and print(f"Calibration inp {inp}")
        trainloader.append((inp, tar))
    print(f"Examples used: {len(trainloader)}; current_token_count: {current_token_count}; tokens_to_give: {tokens_to_give}")
    return trainloader, None
    

