"""
Defines a wrapper class of RobertaPreTrainedModel model to do regression on text data.
Based on: https://www.kaggle.com/code/sumantindurkhya/bert-for-regression
"""

from typing import Optional, Tuple, Union
from tqdm import tqdm, trange

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import BertModel, BertPreTrainedModel, RobertaPreTrainedModel, RobertaModel

class RobertaRegressor(RobertaPreTrainedModel):

    def __init__(self, config, num_outputs=1, dropout=0.1, freeze_bert=False):
        super().__init__(config)

        self.num_outputs = num_outputs

        self.roberta = RobertaModel(config)
        if freeze_bert:
            # freeze the roberta parameters
            for param in self.roberta.parameters():
                param.requires_grad = False
        self.classifier = nn.Linear(config.hidden_size, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.regressor = nn.Linear(128, num_outputs)
        

    def forward(self, input_ids, attention_mask):
        # forward pass of the model
        base_out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        logits = base_out.pooler_output
        out = self.classifier(logits)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.tanh(out)
        out = self.dropout(out)
        out = self.regressor(out)
        return out
    
    def predict(self, text:str, tokenizer, device, numpy=True) -> Tuple[float, float, float, float]:
        input_ids, attention_mask = tokenizer.encode_plus(text, padding=True, truncation=True, return_tensors='pt').values()
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        output = self(input_ids, attention_mask).squeeze()
        # free up memory
        del input_ids, attention_mask
        out = output.detach()
        if numpy:
            return out.cpu().numpy()
        return out
    

class RobertaSeqMultiRegressor(RobertaPreTrainedModel):
    """
    A wrapper class of RobertaPreTrainedModel model to do multi-output regression on text data.
    This models the task of predicting multiple outputs from a single text input.
    The problem is formulated in a sequential manner, where the model predicts the next output
    conditioned on the previous outputs.

    This approach is ideal for modeling problems where the outputs are correlated
    such as probability distributions, where the sum of the outputs must be 1.
    Or, for example, in the case of predicting the next word in a sentence, where the
    model must predict the next word conditioned on the previous words.

    The model is similar to the one described in the RobertaRegressor class, with the
    exception that the head of the model is a sequential model, where the output of the
    previous layer is fed as input to the next layer similar to how a RNN works.
    """

    def __init__(self, config, num_outputs=1, dropout=0.1, freeze_bert=False):
        super().__init__(config)

        self.num_outputs = num_outputs

        self.roberta = RobertaModel(config)
        if freeze_bert:
            # freeze the roberta parameters
            for param in self.roberta.parameters():
                param.requires_grad = False
        # head of the model is a model that takes the output of the previous layer as input
        # and outputs a single value until the number of outputs is reached
        for i in range(num_outputs):
            setattr(self, f"regressor_{i}", nn.Linear(config.hidden_size, 128))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()

    def forward(self, input_ids, attention_mask):
        # forward pass of the model
        base_out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        logits = base_out.pooler_output
        outputs = []
        for i in range(self.num_outputs):
            out = getattr(self, f"regressor_{i}")(logits)
            out = self.dropout(out)
            out = self.relu(out)
            out = self.tanh(out)
            outputs.append(out)
        return outputs


def sum_diff_loss(output, target):
    return torch.sum(torch.abs(output - target))

def evaluate(model, criterion, dataloader, device, sum_diff_penalty=False):
    model.eval()
    mean_acc, mean_loss, count = 0, 0, 0

    with torch.no_grad():
        for input_ids, attention_mask, target in (dataloader):
            
            input_ids, attention_mask, target = input_ids.to(device), attention_mask.to(device), target.to(device)
            output = model(input_ids, attention_mask)
            
            mean_loss += criterion(output.squeeze(), target.type_as(output)).item()
            count += 1
            
    return mean_loss/count    

# def predict(model, dataloader, device):
#     predicted_label = []
#     actual_label = []
#     with torch.no_grad():
#         for input_ids, attention_mask, target in (dataloader):
            
#             input_ids, attention_mask, target = input_ids.to(device), attention_mask.to(device), target.to(device)
#             output = model(input_ids, attention_mask)
                        
#             predicted_label += output
#             actual_label += target
            
#     return predicted_label

def train(model, criterion, optimizer, train_loader, val_loader, epochs, device):
    best_acc = 0
    for epoch in trange(epochs, desc="Epoch"):
        model.train()
        train_loss = 0
        for i, (input_ids, attention_mask, target) in enumerate(iterable=train_loader):
            optimizer.zero_grad()  
            
            input_ids, attention_mask, target = input_ids.to(device), attention_mask.to(device), target.to(device)
            
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            # out = model.classifier(output)
            loss = criterion(output.squeeze(), target.type_as(output))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        print(f"Training loss is {train_loss/len(train_loader)}")
        val_loss = evaluate(model=model, criterion=criterion, dataloader=val_loader, device=device)
        print("Epoch {} complete! Validation Loss : {}".format(epoch, val_loss))

def multi_reg_loss(loss='mse', sum_diff_penalty:float=0.0):
    """
    A custom loss function that penalizes the sum of differences
    between the predicted and actual values for multi-output regression.
    This is done to guide the model to predict outputs where 
    sum(y_hat1, y_hat2, ...) = sum(y1, y2, ...)

    e.g: in task d, we have that sum(label1, label2, label3, label4) = 1
    since its a probability distribution. 

    Parameters
    ----------
    loss : str, optional
        The loss function to be used, by default 'mse'
        Available options: 'mse' and 'cross_entropy'
        for mean squared error and cross entropy loss respectively
    sum_diff_penalty : float, optional
        The penalty to be applied to the sum of differences between the predicted and actual values, by default 0.0 (no penalty)
    """
    if loss == 'mse':
        loss_func = F.mse_loss
    elif loss == 'cross_entropy':
        loss_func = F.cross_entropy
    else:
        raise ValueError("Invalid loss function. Available options: 'mse' and 'cross_entropy'")
    def reg_loss(input, target):
        # first compute the normal MSE loss
        mse = loss_func(input, target)
        # then penalize the sum of differences between the predicted and actual values
        sum_diff = torch.square(torch.sum(input) - torch.sum(target))
        return mse + sum_diff_penalty*sum_diff
    return reg_loss

