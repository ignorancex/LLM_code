"""Module for text generation."""
# https://github.com/huggingface/notebooks/blob/main/examples/language_modeling.ipynb
# from datasets import ClassLabel
import random
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
import torch
import math
from tqdm import tqdm
import time
import evaluate
from collections import defaultdict
from jarvis.db.jsonutils import dumpjson

random_seed = 123
torch.manual_seed(random_seed)
random.seed(0)
np.random.seed(random_seed)
torch.backends.cudnn.deterministic = True

rouge_score = evaluate.load("rouge")

# import torch
# torch.cuda.is_available = lambda : False

tqdm.pandas()
device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")

# wget https://figshare.com/ndownloader/files/39768544 -O cond_mat.zip


def train_generator(
    csv_file="cond_mat.zip",
    train_dataset=[],
    val_dataset=[],
    train_size=0.8,
    block_size=64,
    model_checkpoint="gpt2-medium",
    SEED=42,
    batch_size=2,
    num_train_epochs=5,
    num_proc=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    calculate_baseline=True,
    max_new_tokens=200,
):
    "Train LLM generator." ""
    t1 = time.time()
    if not train_dataset:
        # Just working on supercon category
        df = pd.read_csv(csv_file, dtype="str")
        df = df[df["categories"] == "cond-mat.supr-con"][0:500]
        df = df.drop_duplicates(subset="id")
        n_train = int(len(df) * train_size)
        n_val = len(df) - n_train

        train_dataset = df[:n_train].reset_index(drop=True)
        val_dataset = df[-n_val:].reset_index(drop=True)
        xtrain = defaultdict()
        xtest = defaultdict()
        for i, ii in train_dataset.iterrows():
            xtrain[ii["id"]] = ii["abstract"]
        for i, ii in val_dataset.iterrows():
            xtest[ii["id"]] = ii["abstract"]
        m = {}
        m["train"] = xtrain
        m["test"] = xtest
        dumpjson(data=m, filename="arxiv_gen_text.json")

    train_dataset["text"] = (
        train_dataset["title"]
        + " can be described as "
        + train_dataset["abstract"]
    )
    train_dataset = train_dataset["text"]
    train_dataset.to_csv("train_dataset.csv", index=False)

    val_dataset["text"] = (
        val_dataset["title"]
        + " can be described as "
        + val_dataset["abstract"]
    )
    val_dataset.to_csv("val_dataset_full.csv", index=False)
    val_dataset = val_dataset["text"]
    val_dataset.to_csv("val_dataset.csv", index=False)

    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(val_dataset.shape))

    datasets = load_dataset(
        "csv",
        data_files={
            "train": "train_dataset.csv",
            "validation": "val_dataset.csv",
            "test": "val_dataset.csv",
        },
    )
    print('datasets["validation"]', datasets["validation"])
    # model_checkpoint = "chavinlo/alpaca-native" #"distilgpt2"
    # block_size = tokenizer.model_max_length
    # model_checkpoint = "HuggingFaceM4/tiny-random-LlamaForCausalLM"
    # model_checkpoint = "gpt2-medium"
    # model_checkpoint = "EleutherAI/gpt-neo-1.3B"

    # block_size = 512
    # block_size = 32
    # block_size = 128
    # block_size = 64

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: sum(examples[k], []) for k in examples.keys()
        }
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding
        # if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + block_size]
                for i in range(0, total_length, block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized_datasets = datasets.map(
        tokenize_function, batched=True, num_proc=1, remove_columns=["text"]
    )
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=batch_size,
        # batch_size=1000,
        # num_proc=1,
        num_proc=num_proc,
    )
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint).to(device)

    # from transformers import AutoModelForCausalLM, AutoTokenizer
    def generate_text(
        prompt="Define superconductors",
        checkpoint="gpt2-medium",
        max_new_tokens=200,
        model="",
        tokenizer="",
    ):
        # https://towardsdatascience.com/text-generation-with-python-and-gpt-2-1fecbff1635b
        # tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        inputs = tokenizer(prompt, return_tensors="pt")
        # model = AutoModelForCausalLM.from_pretrained(checkpoint)
        # outputs = model.generate(**inputs, do_sample=True)
        outputs = model.generate(
            **inputs, do_sample=True, max_new_tokens=max_new_tokens
        )
        out = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return out[0]

    tmp = []
    """
    if calculate_baseline:
        tmp = pd.read_csv("val_dataset.csv")
        print(tmp)

        tmp["title"] = tmp["text"].apply(
            lambda x: x.split(" can be described as ")[0]
        )
        tmp["abstract"] = tmp["text"].apply(
            lambda x: x.split(" can be described as ")[1]
        )
        tmp["pred"] = tmp["title"].progress_apply(
            lambda x: generate_text(prompt=x, model=model, tokenizer=tokenizer)
        )

        # tmp['pred'] = tmp['text'].progress_apply
        # (lambda x: generate_text(prompt=x,model=model,tokenizer=tokenizer))

        metric = evaluate.load("rouge")
        print(
            "Baseline",
            metric.compute(
                references=tmp["abstract"], predictions=tmp["pred"]
            ),
        )
    """
    # """

    # model_name = model_checkpoint.split("/")[-1]
    training_args = TrainingArguments(
        output_dir="./results",
        logging_dir="./logs",
        # f"{model_name}-finetuned-wikitext2",
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        push_to_hub=False,
        # overwrite_output_dir =True,
        num_train_epochs=num_train_epochs,
        do_eval=True,
        report_to="none",
        # dataloader_pin_memory=False,
        # gradient_accumulation_steps=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True
        # output_dir='results',
    )

    print("training_args", training_args)

    def compute_metrics(eval_preds):
        # metric = evaluate.load("glue","mrpc")
        metric = evaluate.load("rouge")
        # metric = evaluate.load("glue", "mrpc","rouge")
        logits, labels = eval_preds
        # predictions = np.argmax(logits, axis=-1)
        logits = np.argmax(logits, axis=-1)
        # print ('logits', logits,logits.shape)
        # print ()
        # print ('labels', labels,labels.shape)
        x1 = tokenizer.batch_decode(logits, skip_special_tokens=True)
        x2 = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # print ('x1',x1)
        # print ('x2',x2)
        return metric.compute(predictions=x1, references=x2)
        # return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        compute_metrics=compute_metrics,
    )
    trainer.train()
    eval_results = trainer.evaluate()
    print("eval_results", eval_results)
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    # return trainer,eval_results,lm_datasets["validation"]
    predictions = trainer.predict(lm_datasets["validation"])
    preds = np.argmax(predictions.predictions, axis=-1)
    out = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # predictions = trainer.predict(tokenized_datasets["validation"])
    # print(predictions.predictions.shape, predictions.label_ids.shape)
    model_preds = []
    base_preds = []
    val_full_df = pd.read_csv("val_dataset_full.csv", dtype="str")
    results = []
    #     f=open('predictions.csv','w')
    #     f.write('id,target,prediction,baseline\n')
    for i, v in val_full_df.iterrows():
        id = str(v["id"])
        info = {}
        info["id"] = id
        prompt = v["title"] + " can be described as "
        target = v["abstract"]
        info["target"] = target

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        # model = AutoModelForCausalLM.from_pretrained(checkpoint)
        # outputs = model.generate(**inputs, do_sample=True)
        outputs = trainer.model.generate(
            **inputs, do_sample=True, max_new_tokens=max_new_tokens
        )
        out = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        model_preds.append(out)
        info["prediction"] = out
        base_out = ""
        if calculate_baseline:
            outputs = model.generate(
                **inputs, do_sample=True, max_new_tokens=max_new_tokens
            )
            base_out = tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )[0]
            base_preds.append(base_out)
            info["baseline"] = base_out
        results.append(info)
    results = pd.DataFrame(results)
    results.to_csv("predictions.csv", index=False)
    metric = evaluate.load("rouge")
    print(
        "Baseline",
        metric.compute(
            references=results["target"], predictions=results["baseline"]
        ),
    )
    print(
        "Model",
        metric.compute(
            references=results["target"], predictions=results["prediction"]
        ),
    )

    #     f.close()
    t2 = time.time()
    print("Time taken", t2 - t1)
    return trainer, predictions, tokenizer, results, tmp


if __name__ == "__main__":

    trainer, predictions, tokenizer, results, tmp = train_generator()
