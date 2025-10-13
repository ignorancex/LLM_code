#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__description__ = "train datatset with BERT using sliding windows approach; change amount of epochs, tokens and model manually."
__author__ = "Eva Seidlmayer <seidlmayer@zbmed.de>, Lukas Galke"
__copyright__ = "2023-2025 by Eva Seidlmayer"
__license__ = "MIT license"
__email__ = "seidlmayer@zbmed.de"
__version__ = "1.0.1 "


BERT_MODEL_IDENTIFIER = "dmis-lab/biobert-v1.1"
EPOCH_AMOUNT = 4
TOKEN_AMOUNT = 512

import pandas as pd
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    BertForPreTraining,
    AutoConfig,
    AutoTokenizer,
)
import numpy as np
import argparse
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from typing import Optional, Union, Tuple
from transformers.modeling_outputs import SequenceClassifierOutput



def load_dataset(input_file_csv):
    # Load dataset
    df = pd.read_csv(input_file_csv, sep=",")
    df = df.replace(r"^\s*$", np.nan, regex=True)
    df = df.sample(frac=1)
    df = df.astype(str)
    texts = df["text"].to_list()
    print(len(texts))
    labels = df["category_id"].to_list()
    print(len(labels))
    print("data input lists created")

    return texts, labels


def tokenize(texts):
    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_IDENTIFIER)

    # set max_length
    max_length = TOKEN_AMOUNT


    # Tokenize the text data
    tokens = tokenizer(
        texts, max_length=max_length, padding="max_length", truncation=True
    )
    print("text is tokenized")
    return tokens


def convert_labels(labels):
    # Convert labels to numerical values
    # label_map = {'scientific': 0, 'disinfo':1, 'alternative_science':2}
    label_map = {"scientific": 0, "popular": 1, "disinfo": 2, "alternative_science": 3}
    labels_conv = [label_map[label] for label in labels]
    labels_conv = torch.tensor(labels_conv, dtype=torch.long)
    labels_onehot = torch.nn.functional.one_hot(labels_conv, num_classes=4).float()
    # labels_onehot = torch.nn.functional.one_hot(labels_conv, num_classes=4).float()
    print("labels converted")
    return labels_onehot


def calc_split_ratio(labels_onehot):
    # 80% training 20% validation
    split_ratio = int(len(labels_onehot) * 0.8)
    print("split_ratio defined")
    return split_ratio


def split_train_val_data(tokens, split_ratio, labels_onehot):
    # Split the data into training and validation sets
    train_inputs, val_inputs = np.split(tokens["input_ids"], [split_ratio])
    train_masks, val_masks = np.split(tokens["attention_mask"], [split_ratio])
    train_labels, val_labels = np.split(labels_onehot, [split_ratio])
    print("train/val -inputs, -masks, -labels created")
    return train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels


# used in class AQUAS_slidingwindow
def sliding_window(item):
    window_size = 512
    stride = 256
    windows = [
        item[i : i + window_size] for i in range(0, len(item) - window_size + 1, stride)
    ]
    return windows


class AQUASSlidingBERT(BertForSequenceClassification):
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        AQUASwindowsvectors = []
        AQUASnumberwindows = 0


        batch_size = input_ids.size(0)

        assert batch_size == 1, "Please use batch size = 1"

        length = attention_mask.sum(1)
        length = int(length.item())
        if length > 512:
            window_tokens = sliding_window(input_ids.squeeze(0))
            window_attn_masks = sliding_window(attention_mask.squeeze(0))

            for tokens, attn_mask in zip(window_tokens, window_attn_masks):
                tokens = tokens.unsqueeze(0)
                attn_mask = attn_mask.unsqueeze(0)

                outputs = self.bert(
                    tokens,
                    attention_mask=attn_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                pooled_output = outputs[1]
                AQUASwindowsvectors.append(pooled_output)
                AQUASnumberwindows += 1
            # sum and mean
            AQUASpooled_output = torch.stack(AQUASwindowsvectors, dim=0).mean(dim=0)

        else:
            assert input_ids.dim() == 2, "input_ids should be 2-dimensional: [bsz,seq]"
            assert (
                attention_mask.dim() == 2
            ), "attention_mask should be 2-dimensional: [bsz,seq]"

            # Trim to 512 tokens.
            # because BERT can only process 512, and original maxlen was 2048.
            # We know they are shorter than 512 anyways, so nothing is lost.
            input_ids = input_ids[:, :512]
            attention_mask = attention_mask[:, :512]

            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            AQUASpooled_output = outputs[1]

        # AQUASpooled_output defined
        pooled_output = self.dropout(AQUASpooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # softmax crossentropie: CrossEntropyLoss()
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        print("sliding window: check!")


def train_epoch(model, optimizer, train_inputs, train_labels, train_masks):
    train_loader = torch.utils.data.DataLoader(
        list(zip(train_inputs, train_labels, train_masks)), batch_size=1, shuffle=True
    )

    # training for one epoch
    for batch in train_loader:
        optimizer.zero_grad()

        # batch auseinanerfriemeln
        batch_inputs, batch_labels, batch_masks = batch
        output = model(
            input_ids=batch_inputs, labels=batch_labels, attention_mask=batch_masks
        )

        # output: SequenceClassifierOutput
        loss = output.loss

        loss.backward()
        optimizer.step()

    print("epoch trained")
    return model


def evaluate_model(model, val_inputs, val_masks, val_labels):
    assert val_inputs.dim() == 2, "val_inputs should be 2-dimensional"
    assert val_masks.dim() == 2, "val_masks should be 2-dimensional"
    assert val_labels.dim() == 2, "val_labels should be 2-dimensional"
    # Evaluate  model
    val_loader = torch.utils.data.DataLoader(
        list(zip(val_inputs, val_masks)),
        batch_size=1,
        shuffle=False,  # Never change to True, else all will break
    )

    all_logits = []
    with torch.no_grad():
        model.eval()
        for batch_input, batch_mask in val_loader:
            outputs = model(input_ids=batch_input, attention_mask=batch_mask)
            logits = outputs.logits
            assert logits.size(1) == 4, "Something went terribly wrong"
            all_logits.append(logits)

    all_logits = torch.cat(all_logits, dim=0)

    # Calculate accuracy
    multiclass_accuracy = (
        (all_logits.argmax(dim=-1) == val_labels.argmax(dim=-1)).float().mean()
    )

    # Turn logits into binary Yes/No decision per class with threshold 0.5
    # for multi-label classification (instead of taken just the maximum)
    predictions = torch.sigmoid(all_logits) > 0.5

    # calculate f1 score
    val_labels = val_labels.long().numpy()
    predictions = predictions.long().numpy()
    f1 = f1_score(val_labels, predictions, average="weighted")

    # calculate accuracy per class
    target_class = [
        "class scientific",
        "class popular scientific",
        "class disinformation",
        "class alternative science",
    ]

    # classification report
    class_rep = classification_report(
        val_labels, predictions, target_names=target_class
    )

    return multiclass_accuracy, f1, class_rep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file_csv")
    parser.add_argument("output_file_csv")
    args = parser.parse_args()

    learning_rate = 3e-5
    epochs = EPOCH_AMOUNT
    output_file = args.output_file_csv

    texts, labels = load_dataset(args.input_file_csv)
    tokens = tokenize(texts)
    labels_onehot = convert_labels(labels)
    split_ratio = calc_split_ratio(labels_onehot)
    (
        train_inputs,
        val_inputs,
        train_masks,
        val_masks,
        train_labels,
        val_labels,
    ) = split_train_val_data(tokens, split_ratio, labels_onehot)

    train_inputs = torch.tensor(train_inputs)
    val_inputs = torch.tensor(val_inputs)
    train_masks = torch.tensor(train_masks)
    val_masks = torch.tensor(val_masks)
    train_labels = torch.tensor(train_labels)
    val_labels = torch.tensor(val_labels)


    model = AQUASSlidingBERT.from_pretrained(
        BERT_MODEL_IDENTIFIER,
        num_labels=4,
        problem_type="multi_label_classification",
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    # each loop is one epoch
    for epoch in range(epochs):
        print("start new epoch")
        train_epoch(model, optimizer, train_inputs, train_labels, train_masks)
        acc_val, f1, class_rep = evaluate_model(
            model, val_inputs, val_masks, val_labels
        )
        acc_train, f1_, class_rep_ = evaluate_model(
            model, train_inputs, train_masks, train_labels
        )

        class_rep = f"Accuracy_val: {acc_val:.4f}, Accuracy_train: {acc_train:.4f}  F1-score: {f1:.4f}, Classification_report:{class_rep}"

        print(
            f"[{epoch+1}] Accuracy_val: {acc_val:.4f}, Accuracy_train: {acc_train:.4f}  F1-score: {f1:.4f}, Classification_report:{class_rep}"
        )
        filename = f"{epoch+1}"+ output_file
        model.save_pretrained(filename)

    model.save_pretrained(output_file)
    print("done")


if __name__ == "__main__":
    main()
