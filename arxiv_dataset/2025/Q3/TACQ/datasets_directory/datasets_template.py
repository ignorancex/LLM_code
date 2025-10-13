import os
import random
from torch.utils.data import Dataset as torchDataset

DATA_DIRECTORY = "datasets_directory/Template/data"

class TemplateDataset(torchDataset):
    """
    A template dataset class. Choose to subclass or edit to create your own class. 
    Replace 'TemplateDataset' with a relevant name and the code in __init__ and _load_data with logic specific to your dataset.
    """
    def __init__(
        self,
        model_name,
        tokenizer,
        data_filepath=DATA_DIRECTORY,
        use_train_split=False,
        verbose=False,
        device="cuda",
    ):
        """
        Args:
            model_name (str): huggingface model path to spawn tokenizer if tokenizer is not passed in.
            tokenizer: A tokenizer object, e.g. from AutoTokenizer.from_pretrained().
            data_filepath (str): Path to the dataset directory or file.
            use_train_split (bool): Whether to use training split or not.
            verbose (bool): Whether to print debug info.
            device (str): "cpu" or "cuda".
        """
        # Store parameters
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.data_filepath = data_filepath
        self.use_train_split = use_train_split
        self.verbose = verbose
        self.device = device
        
        # Placeholder for examples. Typically you would load from JSON, text files, etc.
        # self.texts will be a list of strings, one per data example
        self.texts = []
        
        # Example: Load data (implementation depends on your format).
        # Replace this with your dataset's loading logic:
        if use_train_split:
            # Load training set
            train_data_file = os.path.join(self.data_filepath, "train.txt")
            self.texts = self._load_data(train_data_file)
        else:
            # Load test/validation set
            test_data_file = os.path.join(self.data_filepath, "test.txt")
            self.texts = self._load_data(test_data_file)

    def _load_data(self, filepath):
        """
        Replace this with your dataset-specific loading.
        Returns a list of strings, each representing one data sample.
        """
        if self.verbose:
            print(f"Loading data from: {filepath}")
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} does not exist. Returning empty list.")
            return []
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n")
        return lines

    def shuffle(self, seed=42):
        """
        Shuffle the list of strings self.texts in-place.
        """
        random.seed(seed)
        random.shuffle(self.texts)

    def truncate_to_seqlen_n_samples(self, seqlen, n_samples):
        """
        Truncate this dataset so that the total number of tokens across the
        selected samples is <= seqlen * n_samples.

        seqlen: max sequence length
        n_samples: number of examples
        """
        tokens_to_give = seqlen*n_samples
        current_token_count = 0
        for i in range(len(self.texts)):
            if current_token_count + len(self.tokenizer(self.texts[i])["input_ids"]) > tokens_to_give:
                break
            current_token_count += len(self.tokenizer(self.texts[i])["input_ids"])
        print(f"Truncated to {i} samples, current_token_count: {current_token_count}, tokens_to_give: {tokens_to_give}")
        self.texts = self.texts[:i]

    def to_hf_dataset(self):
        """
        Convert text list into a Hugging Face Dataset object, required for current pipeline.
        """
        from datasets import Dataset as hfDataset
        dataset = hfDataset.from_dict({"text": self.texts})
        return dataset

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Returns a dict with keys 'input_ids', 'attention_mask', etc.
        following the typical tokenizer output. Make sure it's consistent
        with how you want to feed data to your model.
        """
        prompt = self.texts[idx]
        inputs = self.tokenizer(prompt, return_tensors="pt", padding="longest", truncation=False).to(self.device)
        # Optionally squeeze to remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs


def get_template_data(nsamples, seed, seqlen, model, data_filepath=DATA_DIRECTORY, verbose=True):
    """
    Creates a 'trainloader' using a TemplateDataset instance, replicating the
    convention used in GPTQ and many other repositories. Since we use our own evaluation scripts, 
    we only define behavior for gathering the training dataset. 

    Args:
        nsamples (int): Number of samples to target in calibration.
        seed (int): Random seed for any shuffle.
        seqlen (int): Max sequence length.
        model (str): Model name or path for the tokenizer.
        data_filepath (str): Directory path for the dataset files.

    Returns:
        trainloader (list): A list of (inp, tar) tensors, 
        inp should be tokenized instances of your dataset, 
        tar is not used but is by convention the same as inp 
        but where all except for the last token is masked. 
        None: Placeholder to conform to GPTQ / other baseline conventions
    """
    from transformers import AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    dataset = TemplateDataset(
        model_name=model,
        tokenizer=tokenizer,
        data_filepath=data_filepath,
        use_train_split=True,  # Use training split
        verbose=verbose,  # Optionally verbose
        device="cuda"
    )

    # Shuffle
    dataset.shuffle(seed=seed)

    # Truncate
    dataset.truncate_to_seqlen_n_samples(seqlen, nsamples)

    # Convert to Hugging Face dataset
    hf_dataset = dataset.to_hf_dataset()

    # Tokenize dataset
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    def custom_tokenize_function_no_padding(examples):
        return tokenizer(examples["text"], truncation=False, padding=False, max_length=seqlen, return_tensors="pt")
    hf_dataset = hf_dataset.map(custom_tokenize_function_no_padding, batched=False, num_proc=1, remove_columns=["text"])

    # Generate trainloader, which is a list containing tensors of tokens
    trainloader = []
    tokens_to_give = seqlen*nsamples
    current_token_count = 0
    for i in range(len(hf_dataset)):
        current_token_count += len(hf_dataset[i]["input_ids"][0])
        inp = hf_dataset[i]["input_ids"]
        inp = torch.tensor(inp)  
        tar = inp.clone()  
        tar[:-1] = -100
        i == 0 and print(f"Calibration inp {inp}")
        trainloader.append((inp, tar))
    print(f"Examples used: {len(trainloader)}; current_token_count: {current_token_count}; tokens_to_give: {tokens_to_give}")
    return trainloader, None
