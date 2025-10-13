import gzip
import json
import os
import os.path as osp
import ssl
import urllib.request
import re
import random


DATA_DIRECTORY = "datasets_directory/GSM8k/data"


def download_url(url: str, folder="folder"):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition("/")[2]
    file = file if file[0] == "?" else file.split("?")[0]
    path = osp.join(folder, file)
    if osp.exists(path):
        print(f"File {file} exists, use existing file.")
        return path

    print(f"Downloading {url}")
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, "wb") as f:
        f.write(data.read())

    return path


def load_jsonl(
    file_path,
    instruction="instruction",
    input="input",
    output="output",
    category="category",
    is_gzip=False,
):
    # Format of each line:
    # {'instruction': ..., 'input': ..., 'output':...}
    list_data_dict = []
    open_func = open if not is_gzip else gzip.open
    with open_func(file_path, "r") as f:
        for line in f:
            item = json.loads(line)
            new_item = dict(
                instruction=item[instruction] if instruction in item else None,
                input=item[input] if input in item else None,
                output=item[output] if output in item else None,
                category=item[category] if category in item else None,
            )
            item = new_item
            list_data_dict.append(item)
    return list_data_dict


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_answer, answer):
    gt_answer = extract_answer_from_output(answer)
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer


ANSWER_TRIGGER = "The answer is"


def create_demo_text(n_shot=8, cot_flag=True):
    question, chain, answer = [], [], []
    question.append(
        "There are 15 trees in the grove. "
        "Grove workers will plant trees in the grove today. "
        "After they are done, there will be 21 trees. "
        "How many trees did the grove workers plant today?"
    )
    chain.append(
        "There are 15 trees originally. "
        "Then there were 21 trees after some more were planted. "
        "So there must have been 21 - 15 = 6."
    )
    answer.append("6")

    question.append(
        "If there are 3 cars in the parking lot and 2 more cars arrive, "
        "how many cars are in the parking lot?"
    )
    chain.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
    answer.append("5")

    question.append(
        "Leah had 32 chocolates and her sister had 42. If they ate 35, "
        "how many pieces do they have left in total?"
    )
    chain.append(
        "Originally, Leah had 32 chocolates. "
        "Her sister had 42. So in total they had 32 + 42 = 74. "
        "After eating 35, they had 74 - 35 = 39."
    )
    answer.append("39")

    question.append(
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason "
        "has 12 lollipops. How many lollipops did Jason give to Denny?"
    )
    chain.append(
        "Jason started with 20 lollipops. Then he had 12 after giving some "
        "to Denny. So he gave Denny 20 - 12 = 8."
    )
    answer.append("8")

    question.append(
        "Shawn has five toys. For Christmas, he got two toys each from his "
        "mom and dad. How many toys does he have now?"
    )
    chain.append(
        "Shawn started with 5 toys. If he got 2 toys each from his mom and "
        "dad, then that is 4 more toys. 5 + 4 = 9."
    )
    answer.append("9")

    question.append(
        "There were nine computers in the server room. Five more computers "
        "were installed each day, from monday to thursday. "
        "How many computers are now in the server room?"
    )
    chain.append(
        "There were originally 9 computers. For each of 4 days, 5 more "
        "computers were added. So 5 * 4 = 20 computers were added. "
        "9 + 20 is 29."
    )
    answer.append("29")

    question.append(
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
        "wednesday, he lost 2 more. "
        "How many golf balls did he have at the end of wednesday?"
    )
    chain.append(
        "Michael started with 58 golf balls. After losing 23 on tuesday, "
        "he had 58 - 23 = 35. After losing 2 more, "
        "he had 35 - 2 = 33 golf balls."
    )
    answer.append("33")

    question.append(
        "Olivia has $23. She bought five bagels for $3 each. "
        "How much money does she have left?"
    )
    chain.append(
        "Olivia had 23 dollars. "
        "5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
        "So she has 23 - 15 dollars left. 23 - 15 is 8."
    )
    answer.append("8")

    # randomize order of the examples ...
    index_list = list(range(len(question)))
    random.shuffle(index_list)

    # Concatenate demonstration examples ...
    demo_text = ""
    for i in index_list[:n_shot]:
        if cot_flag:
            demo_text += (
                "Q: "
                + question[i]
                + "\nA: "
                + chain[i]
                + " "
                + ANSWER_TRIGGER
                + " "
                + answer[i]
                + ".\n\n"
            )
        else:
            demo_text += (
                "Question: "
                + question[i]
                + "\nAnswer: "
                + ANSWER_TRIGGER
                + " "
                + answer[i]
                + ".\n\n"
            )
    return demo_text


def build_prompt(input_text, n_shot, cot_flag):
    demo = create_demo_text(n_shot, cot_flag)
    input_text_prompt = demo + "Q: " + input_text + "\n" + "A:"
    return input_text_prompt


def clean_answer(model_pred):
    model_pred = model_pred.lower()
    preds = model_pred.split(ANSWER_TRIGGER.lower())
    answer_flag = True if len(preds) > 1 else False
    if answer_flag:
        # Pick first answer with flag
        pred = preds[1]
    else:
        # Pick last number without flag
        pred = preds[-1]

    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", pred)]

    if len(pred) == 0:
        return INVALID_ANS

    if answer_flag:
        # choose the first element in list
        pred = pred[0]
    else:
        # choose the last element in list
        pred = pred[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred[-1] == ".":
        pred = pred[:-1]

    return pred


def seed_everything(seed: int):
    import random
    import os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

from torch.utils.data import Dataset as torchDataset
class GSM8k_N_Shot_Dataset(torchDataset):
    def __init__(self, model_name, tokenizer, data_filepath = DATA_DIRECTORY, use_train_split=False, \
                 verbose=False, n_shot=8, cot_flag=True, \
                 make_data_wrong=False, device="cuda"):
        import pandas as pd
        import os
        self.texts = []
        self.tokenizer = tokenizer
        self.verbose = verbose
        self.device = device
        train_data = load_jsonl(os.path.join(data_filepath, "gsm8k_train.jsonl"), instruction="question", output="answer")
        test_data = load_jsonl(os.path.join(data_filepath, "gsm8k_test.jsonl"), instruction="question", output="answer")
        train_data_text = []
        for sample in train_data:
            input_text = build_prompt(sample["instruction"], n_shot, cot_flag)
            if cot_flag:
                input_text += (
                    " "
                    + sample["output"]
                    + " "
                    + ANSWER_TRIGGER
                    + " "
                    + extract_answer_from_output(sample["output"])
                    + ".\n\n"
                )
            else:
                raise Exception("None chain of thought GSM8k Eval is not supported")
            verbose and print(input_text)
            train_data_text.append(input_text)

        test_data_text = []
        for sample in test_data:
            input_text = build_prompt(sample["instruction"], n_shot, cot_flag)
            if cot_flag:
                input_text += (
                    " "
                    + sample["output"]
                    + " "
                    + ANSWER_TRIGGER
                    + " "
                    + extract_answer_from_output(sample["output"])
                    + ".\n\n"
                )
            else:
                raise Exception("None chain of thought GSM8k Eval is not supported")
            test_data_text.append(input_text)

        self.texts = train_data_text if use_train_split else test_data_text
    
    def shuffle(self, seed=42):
        import random
        random.seed(seed)
        random.shuffle(self.texts)

    def truncate_to_seqlen_n_samples(self, seqlen, n_samples):
        tokens_to_give = seqlen*n_samples
        current_token_count = 0
        for i in range(len(self.texts)):
            if current_token_count + len(self.tokenizer(self.texts[i])["input_ids"]) > tokens_to_give:
                break
            current_token_count += len(self.tokenizer(self.texts[i])["input_ids"])
        print(f"Truncated to {i} samples, current_token_count: {current_token_count}, tokens_to_give: {tokens_to_give}")
        self.texts = self.texts[:i]

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
        return inputs  # a dict with input_ids, labels, attention masks, etc

def get_gsm8k(nsamples, seed, seqlen, model, data_filepath=DATA_DIRECTORY, n_shot = 8, cot_flag = True):
    """Follows the GPTQ repo convention for preparing a calibration dataset for GSM8k"""
    from transformers import AutoTokenizer
    import torch
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    calibration_dataset_no_padding = GSM8k_N_Shot_Dataset(data_filepath=data_filepath, model_name=model, tokenizer=tokenizer, \
                use_train_split=True, n_shot=n_shot, cot_flag=cot_flag, make_data_wrong=False
                )
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
        tar[:-1] = -100
        i == 0 and print(f"Calibration inp {inp}")
        trainloader.append((inp, tar))
    print(f"Examples used: {len(trainloader)}; current_token_count: {current_token_count}; tokens_to_give: {tokens_to_give}")
    return trainloader, None
