"""Module to summarize documents."""
# https://colab.research.google.com/github/abhimishra91/
# transformers-tutorials/blob/master/transformers_summarization_wandb.ipynb
import numpy as np
import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
    #    RandomSampler,
    #    SequentialSampler,
)
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch import cuda
from tqdm import tqdm
from transformers import pipeline
import pandas as pd
import pprint
from jarvis.db.jsonutils import dumpjson
from collections import defaultdict
import time
import evaluate
# from datasets import load_metric

tqdm.pandas()

device = "cuda" if cuda.is_available() else "cpu"
# metric = load_metric("rouge")


# def calc_rouge_scores(candidates, references):
#    result = metric.compute(
#        predictions=candidates, references=references, use_stemmer=True
#    )
#    result = {
#        key: round(value.mid.fmeasure * 100, 1)
#        for key, value in result.items()
#    }
#    return result


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.text = self.data.text  # title
        self.ctext = self.data.ctext  # abstract

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        ctext = str(self.ctext[index])
        ctext = " ".join(ctext.split())

        text = str(self.text[index])
        text = " ".join(text.split())
        source = self.tokenizer.batch_encode_plus(
            [ctext],
            max_length=self.source_len,
            pad_to_max_length=True,
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        # target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }


def train(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    for _, data in enumerate(loader, 0):
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        labels = y[:, 1:].clone().detach()
        labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=labels,
        )
        loss = outputs[0]

        print(f"Epoch: {epoch}, Loss:  {loss.item()}")
        # if _%10 == 0:
        #    print({"Training Loss": loss.item()})

        # if _%500==0:
        #    print(f'Epoch: {epoch}, Loss:  {loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # xm.optimizer_step(optimizer)
        # xm.mark_step()


def validate(epoch, tokenizer, model, device, loader, val_max_length=25):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data["target_ids"].to(device, dtype=torch.long)
            ids = data["source_ids"].to(device, dtype=torch.long)
            mask = data["source_mask"].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=val_max_length,
                # max_length=50,
                # max_length=150,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True,
            )
            preds = [
                tokenizer.decode(
                    g,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                for g in generated_ids
            ]
            target = [
                tokenizer.decode(
                    t,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                for t in y
            ]

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals


# wget https://figshare.com/ndownloader/files/39768544 -O cond_mat.zip


def main(
    csv_file="cond_mat.zip",
    train_size=0.8,
    train_dataset=[],
    val_dataset=[],
    shuffle=False,
    short_key="title",
    long_key="abstract",
    # train_batch_size=128,
    train_batch_size=32,
    val_batch_size=16,
    # train_epochs=10,
    train_epochs=10,
    val_epochs=1,
    lr=1e-4,
    seed=42,
    max_abstract_len=150,
    # max_abstract_len=100,
    max_val_summary_len=25,
    tokenizer=None,
    model=None,
):
    t1 = time.time()
    config = defaultdict()
    config["train_batch_size"] = train_batch_size
    config["val_batch_size"] = val_batch_size
    config["train_epochs"] = train_epochs
    config["val_epochs"] = val_epochs
    config["learning_rate"] = lr
    config["seed"] = seed
    config["max_abstract_len"] = max_abstract_len
    config["max_summary_len"] = max_abstract_len
    config["max_val_summary_len"] = max_val_summary_len
    print("Config:")
    pprint.pprint(config)
    TRAIN_BATCH_SIZE = config[
        "train_batch_size"
    ]  # input batch size for training
    VALID_BATCH_SIZE = config["val_batch_size"]  # input batch size for testing
    TRAIN_EPOCHS = config[
        "train_epochs"
    ]  # 50        # number of epochs to train
    VAL_EPOCHS = config["val_epochs"]
    LEARNING_RATE = config["learning_rate"]  # learning rate (default: 0.01)
    SEED = config["seed"]  # random seed (default: 42)
    # MAX_LEN: maximum length of abstracts
    # SUMMARY_LEN: maximum summarization length
    MAX_LEN = config["max_abstract_len"]  # 512
    # MAX_LEN = 25
    # SUMMARY_LEN = 10
    SUMMARY_LEN = config["max_summary_len"]
    # SUMMARY_LEN = 150
    val_max_length = config["max_val_summary_len"]

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(SEED)  # pytorch random seed
    np.random.seed(SEED)  # numpy random seed
    torch.backends.cudnn.deterministic = True
    if tokenizer is None:
        # tokenzier for encoding the text
        tokenizer = T5Tokenizer.from_pretrained("t5-base")

    # Importing and Pre-Processing the domain data
    # Selecting the needed columns only.
    # Adding the summarzie text in front of the text.
    # This is to format the dataset similar to
    # how T5 model was trained for summarization task.
    # df = pd.read_csv('news_summary.csv',encoding='latin-1')

    # Creation of Dataset and Dataloader
    # Defining the train size.
    # So 80% of the data will be used for
    # training and the rest will be used for validation.
    if not train_dataset:
        df = pd.read_csv(csv_file)
        df = df.drop_duplicates(subset="id")
        n_train = int(len(df) * train_size)
        n_val = len(df) - n_train
        df["text"] = df[short_key]
        df["ctext"] = df[long_key]
        df.ctext = "summarize: " + df.ctext
        train_dataset = df[:n_train].reset_index(
            drop=True
        )  # .sample(frac=train_size, random_state=SEED)
        val_dataset = df[-n_val:].reset_index(
            drop=True
        )  # .drop(train_dataset.index).reset_index(drop=True)
        print("n_train", n_train)
        print("n_val", n_val)
        print("train_dataset", train_dataset)
        print("val_dataset", val_dataset)
    train_dataset.to_csv("train_dataset.csv")
    val_dataset.to_csv("val_dataset.csv")
    mem = []
    xtrain = defaultdict()
    xtest = defaultdict()
    for i, ii in train_dataset.iterrows():
        info = {}
        info["id"] = ii["id"]
        info["text"] = ii["text"]
        info["ctext"] = ii["ctext"]
        xtrain[ii["id"]] = ii["text"]
        mem.append(info)
    for i, ii in val_dataset.iterrows():
        info = {}
        info["id"] = ii["id"]
        info["text"] = ii["text"]
        info["ctext"] = ii["ctext"]
        xtest[ii["id"]] = ii["text"]
        mem.append(info)
    m = {}
    m["train"] = xtrain
    m["test"] = xtest
    dumpjson(data=m, filename="arxiv_summary_text.json")
    dumpjson(data=mem, filename="arxiv_summary.json")
    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(val_dataset.shape))

    # Creating the Training and Validation
    # dataset for further creation of Dataloader
    training_set = CustomDataset(
        train_dataset, tokenizer, MAX_LEN, SUMMARY_LEN
    )
    val_set = CustomDataset(val_dataset, tokenizer, MAX_LEN, SUMMARY_LEN)

    # Defining the parameters for creation of dataloaders
    train_params = {
        "batch_size": TRAIN_BATCH_SIZE,
        "shuffle": True,
        "num_workers": 0,
    }

    val_params = {
        "batch_size": VALID_BATCH_SIZE,
        "shuffle": False,
        "num_workers": 0,
    }

    # Creation of Dataloaders for testing and validation.
    # This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)
    if model is None:
        # Defining the model. We are using t5-base model and
        # added a Language model layer on top for generation of Summary.
        # Further this model is sent to device(GPU/TPU)for using the hardware.
        model = T5ForConditionalGeneration.from_pretrained("t5-base")
        model = model.to(device)

    # Defining the optimizer that will be used to tune the
    # weights of the network in the training session.
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    # Log metrics with wandb
    # wandb.watch(model, log="all")
    # Training loop
    print("Initiating Fine-Tuning for the model on our dataset")

    for epoch in range(TRAIN_EPOCHS):
        train(epoch, tokenizer, model, device, training_loader, optimizer)

    torch.save(model.state_dict(), "model.pt")
    # Validation loop and saving the resulting
    # file with predictions and acutals in a dataframe.
    # Saving the dataframe as predictions.csv
    print(
        "Now generating summaries on our fine tuned model "
        + "for the validation dataset and saving it in a dataframe"
    )
    for epoch in range(VAL_EPOCHS):
        predictions, actuals = validate(
            epoch,
            tokenizer,
            model,
            device,
            val_loader,
            val_max_length=val_max_length,
        )
        final_df = pd.DataFrame(
            {"Generated Text": predictions, "Actual Text": actuals}
        )
        final_df.to_csv("predictions.csv")
        print("Output Files generated for review")
    df = pd.read_csv("val_dataset.csv")
    df2 = pd.read_csv("predictions.csv")
    df2["text"] = df2["Actual Text"]
    df3 = df2  # pd.merge(df, df2, on="text")
    df3["id"] = df["id"]
    df3["target"] = df3["text"]
    df3["prediction"] = df3["Generated Text"]
    df4 = df3[["id", "target", "prediction"]]

    print("predictions")
    print(df4)
    df4.to_csv("AI-TextSummary-text-arxiv_summary-test-rouge.csv", index=False)
    # df4.to_csv("predictions_id_target_pred.csv", index=False)

    rouge_score = evaluate.load("rouge")
    scores = rouge_score.compute(
        predictions=df4["prediction"], references=df4["target"]
    )
    rouge = scores["rouge1"]
    print("scores", scores, rouge)
    t2 = time.time()
    print("Time taken", t2 - t1)
    print("Untrained baseline")

    # metric = load_metric("rouge")
    # Load the summarization pipeline
    summarizer = pipeline("summarization")

    # Define the text to be summarized

    def summarz(text="", max_length=25, min_length=10):
        summary = summarizer(
            text, max_length=max_length, min_length=min_length, do_sample=False
        )[0]["summary_text"]
        return summary

    df = pd.read_csv("val_dataset.csv")
    df["title_split"] = df["title"].apply(lambda x: len(x.split()))
    print(df.describe())
    df["summ"] = df["abstract"].progress_apply(
        lambda x: summarz(x, max_length=val_max_length)
    )
    print(
        "Rougue baseline",
        rouge_score.compute(references=df["title"], predictions=df["summ"]),
    )
    # print("Rougue baseline", calc_rouge_scores(df["title"], df["summ"]))


if __name__ == "__main__":
    main()
