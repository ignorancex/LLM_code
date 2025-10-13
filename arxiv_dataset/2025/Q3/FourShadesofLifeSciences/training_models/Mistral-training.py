#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__description__ = "train FSoLS datatset with Mistral 7B change amount of epochs, tokens and model manually."
__author__ = "Eva Seidlmayer <seidlmayer@zbmed.de>"
__copyright__ = "2024-2025 by Eva Seidlmayer"
__license__ = "MIT license"
__email__ = "seidlmayer@zbmed.de"
__version__ = "1.0.1 "


MODEL_IDENTIFIER = "mistralai/Mistral-7B-v0.3"
EPOCH_AMOUNT = 4
TOKEN_AMOUNT = 512



import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
import numpy as np
import argparse
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model


def load_dataset(input_file_csv):
    # Load dataset
    df = pd.read_csv(input_file_csv, sep=",")
    df = df.replace(r"^\s*$", np.nan, regex=True)
    df = df.dropna()
    df = df.sample(frac=1)
    df = df.astype(str)
    texts = df["text"].to_list()
    labels = df["category_id"].to_list()
    print("data input lists created")
    return texts, labels


def tokenize(texts, tokenizer):
    # set max_length
    max_length = 512
    # max_length = 1000
    # max_length = 15000

    # Tokenize the text data
    tokens = tokenizer(
        texts, max_length=max_length, padding="max_length", truncation=True
    )

    print("text is tokenized")
    return tokens


def convert_labels(labels):
    # Convert labels to numerical values
    label_map = {"scientific": 0, "popular": 1, "disinfo": 2, "alternative_science": 3}
    labels_conv = [label_map[label] for label in labels]
    labels_conv = torch.tensor(labels_conv, dtype=torch.long)
    labels_onehot = torch.nn.functional.one_hot(labels_conv, num_classes=4).float()
    print("labels converted")
    return labels_onehot


def calc_split_ratio(labels_onehot):
    # 80% training 20% validation
    split_ratio = int(len(labels_onehot) * 0.2)
    print("split_ratio defined")
    return split_ratio


def split_train_val_data(tokens, split_ratio, labels_onehot):
    # Split the data into training and validation sets
    train_inputs, val_inputs = np.split(tokens["input_ids"], [split_ratio])
    train_masks, val_masks = np.split(tokens["attention_mask"], [split_ratio])
    train_labels, val_labels = np.split(labels_onehot, [split_ratio])
    print("train/val -inputs, -masks, -labels created")
    return train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels


# quantization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # enable 4-bit quantization
    bnb_4bit_quant_type="nf4",  # information theoretically optimal dtype for normally distributed weights
    bnb_4bit_use_double_quant=True,  # quantize quantized weights //insert xzibit meme
    bnb_4bit_compute_dtype=torch.bfloat16,  # optimized fp format for ML
)

# lora config
lora_config = LoraConfig(
    r=16,  # the dimension of the low-rank matrices
    lora_alpha=8,  # scaling factor for LoRA activations vs pre-trained weight activations
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,  # dropout probability of the LoRA layers
    bias="none",  # wether to train bias weights, set to 'none' for attention layers
    task_type="SEQ_CLS",
)


def train_epoch(model, optimizer, train_inputs, train_labels, train_masks):

    train_loader = torch.utils.data.DataLoader(
        list(zip(train_inputs, train_labels, train_masks)), batch_size=1, shuffle=True
    )
    # batch_size mal mit 512

    # training for one epoch
    for batch in train_loader:
        optimizer.zero_grad()

        # batch auseinanerfriemeln
        batch_inputs, batch_labels, batch_masks = batch
        output = model(
            input_ids=batch_inputs, labels=batch_labels, attention_mask=batch_masks
        )

        # output: SequenceClassifierOutput
        # loss = output["loss"]  # oder output.loss
        loss = output.loss

        loss.backward()
        optimizer.step()

    # model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    print("epoch trained")
    return model


def evaluate_model(model, val_inputs, val_masks, val_labels):
    model.eval()

    val_labels = val_labels.to(model.device)

    # check shape of val_inputs and val_masks
    print(f"val_inputs shape: {val_inputs.shape}")
    print(f"val_masks shape: {val_masks.shape}")

    # assert val_inputs.dim() == 2, "val_inputs should be 2-dimensional"
    # assert val_masks.dim() == 2, "val_masks should be 2-dimensional"
    # assert val_labels.dim() == 2, "val_labels should be 2-dimensional"

    # Evaluate  model
    val_loader = torch.utils.data.DataLoader(
        list(zip(val_inputs, val_masks)),
        batch_size=1,
        shuffle=False,  # Never change to True, else all will break
    )

    all_logits = []
    with torch.no_grad():
        for i, (batch_input, batch_mask) in enumerate(val_loader):
            print(f"Processing batch {i+1} /{len(val_loader)}")
            if batch_input.size(0) == 0:
                continue

            batch_input = batch_input.to(model.device)
            batch_mask = batch_mask.to(model.device)

            outputs = model(input_ids=batch_input, attention_mask=batch_mask)
            logits = outputs.logits

            # assert logits.size(1) == 4, "Something went terribly wrong"
            all_logits.append(logits)

    if len(all_logits) == 0:
        raise ValueError(
            "No logits were generated. Please check the input data and model."
        )
    all_logits = torch.cat(all_logits, dim=0)

    # Calculate accuracy
    multiclass_accuracy = (
        (all_logits.argmax(dim=-1) == val_labels.argmax(dim=-1)).float().mean()
    ).item()  # .item() creates a python float

    # Turn logits into binary Yes/No decision per class with threshold 0.5
    # for multi-label classification (instead of taken just the maximum)
    # The multi-label classification uses thresholding (via torch.sigmoid and > 0.5), suitable for scenarios where each class is independent, and multiple classes can be positive.
    predictions = (torch.sigmoid(all_logits) > 0.5).long()  # produces yes/no decision

    # move prediction nd labls to CPU for metric calculation
    predictions = predictions.cpu().numpy()
    val_labels = val_labels.cpu().numpy()

    print(f"val_labels type: {type(val_labels)}, shape: {val_labels.shape}")
    print(f"predictions type: {type(predictions)}, shape: {predictions.shape}")

    # calculate f1-score (multi-label, weighted)
    f1 = f1_score(val_labels, predictions, average="weighted")

    # calculate accuracy per class
    target_class = [
        "class scientific",
        "class popular scientific",
        "class disinformative",
        "class alternative scientific",
    ]

    # classification report
    class_rep = classification_report(
        val_labels, predictions, target_names=target_class
    )
    return multiclass_accuracy, f1, class_rep


def main():
    torch.cuda.is_available()
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file_csv")
    parser.add_argument("output_file_csv")
    args = parser.parse_args()

    learning_rate = 3e-5
    epochs = EPOCH_AMOUNT

    output_file = args.output_file_csv


    # load dataset
    texts, labels = load_dataset(args.input_file_csv)

    # Initiate tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDENTIFIER, use_auth_token=token)

    # Pad token to be the same as the eos token
    tokenizer.pad_token = tokenizer.eos_token

    # tokenize the data set
    tokens = tokenize(texts, tokenizer)

    # labels
    labels_onehot = convert_labels(labels)

    # set sprlit ratio
    split_ratio = calc_split_ratio(labels_onehot)
    train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels = (
        split_train_val_data(tokens, split_ratio, labels_onehot)
    )

    # Check if GPU is available and move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # convert tokens to tensors
    train_inputs, val_inputs = torch.tensor(train_inputs).to(device), torch.tensor(
        val_inputs
    ).to(device)
    train_masks, val_masks = torch.tensor(train_masks).to(device), torch.tensor(
        val_masks
    ).to(device)
    train_labels, val_labels = torch.tensor(train_labels).to(device), torch.tensor(
        val_labels
    ).to(device)

    # Mistral 7b INIT
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_IDENTIFIER,
        num_labels=4,
        problem_type="multi_label_classification",
        use_auth_token=token,
        quantization_config=quantization_config,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    # initiate optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    # each loop is one epoch
    for epoch in range(epochs):
        print("start new epoch")

        # train model
        train_epoch(model, optimizer, train_inputs, train_labels, train_masks)

        acc, f1, class_rep = evaluate_model(model, val_inputs, val_masks, val_labels)

        # report results
        class_rep = str(class_rep)

        print(
            f"[{epoch+1}] Accuracy: {acc:.4f}, F1-score: {f1:.4f}, Classification_report:{class_rep}"
        )
        filename = f"Epoch{epoch+1}"+ output_file
        model.save_pretrained(filename)

    model.save_pretrained(output_file)
    print("done")


if __name__ == "__main__":
    main()
