import audeer
import audmetric
import audobject
import numpy as np
import os
import pandas as pd
import torch

METRICS = {
    "ACC": audmetric.accuracy,
    "UAR": audmetric.unweighted_average_recall,
}


class LabelEncoder(audobject.Object):
    def __init__(self, labels):
        self.labels = sorted(labels)
        codes = range(len(self.labels))
        self.inverse_map = {code: label for code,
                            label in zip(codes, self.labels)}
        self.map = {label: code for code,
                    label in zip(codes, self.labels)}

    def encode(self, x):
        return self.map[x]

    def decode(self, x):
        return self.inverse_map[x]
    
    def __call__(self, x):
        return self.encode(x)

    def get_predictions(self, x):
        if torch.is_tensor(x):
            x = x.numpy()
        return x.argmax(-1).squeeze()


def create_data(labels):
    df = pd.read_csv(labels)
    df = df.loc[df["EmoClass"].isin(["N", "H", "A", "S"])]
    encoder = LabelEncoder(sorted(df["EmoClass"].unique().tolist()))
    df_train = df.loc[df["Split_Set"] == "Train"].reset_index(drop=True)
    df_dev = df.loc[df["Split_Set"] == "Development"].reset_index(drop=True)
    df_test = df.loc[df["Split_Set"] == "Test1"].reset_index(drop=True)
    return df_train, df_dev, df_test, encoder


def create_aibo(labels):
    df = pd.read_csv(
        labels,
        header=None,
        sep=" ",
    )
    df = df.rename(columns={0: "id", 1: "class", 2: "conf"})
    df["file"] = df["id"].apply(lambda x: x + ".wav")
    df["school"] = df["id"].apply(lambda x: x.split("_")[0])
    df["speaker"] = df["id"].apply(lambda x: x.split("_")[1])
    encoder = LabelEncoder(sorted(df["class"].unique().tolist()))
    df_test = df.loc[df["school"] == "Mont"]
    df_train_dev = df.loc[df["school"] == "Ohm"]
    speakers = sorted(df_train_dev["speaker"].unique())
    df_train = df_train_dev.loc[
        df_train_dev["speaker"].isin(speakers[:-2])
    ]
    df_dev = df_train_dev.loc[
        df_train_dev["speaker"].isin(speakers[-2:])
    ]
    return df_train, df_dev, df_test, encoder


def evaluate(
    model, 
    device, 
    loader, 
    tokenizer,
    encoder
):
    
    model.to(device)
    model.eval()
    outputs = torch.zeros(len(loader.dataset), len(encoder.labels))
    targets = torch.zeros(len(loader.dataset))
    assert loader.batch_size == 1
    with torch.no_grad():
        for index, (features, target) in audeer.progress_bar(
            enumerate(loader),
            desc='Batch',
            total=len(loader),
            disable=True
        ):
            features = tokenizer.encode_plus(features[0], return_tensors='pt').to(device)
            outputs[index, :] = model(**features).logits.squeeze().cpu()
            targets[index] = target
    targets = targets.numpy()
    outputs = outputs.numpy()
    predictions = encoder.get_predictions(outputs)
    return {
        key: METRICS[key](targets, predictions)
        for key in METRICS.keys()
    }, targets, predictions