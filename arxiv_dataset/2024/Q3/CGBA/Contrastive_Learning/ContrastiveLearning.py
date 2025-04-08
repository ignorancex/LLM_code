#!/usr/bin/env python
# coding: utf-8

import os
gpu_avail = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_avail

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import random
from torch.optim.lr_scheduler import OneCycleLR
import json
from tqdm import tqdm

path = '../COVID_pub'
name = 'COVID'
random_seed = 123
torch.manual_seed(random_seed)
random.seed(random_seed)
batch_size = 32 ## total batch size

### hyperparamerters margin & alpha used for contrastive loss and final loss, respectively
margin = 0.2
alpha = 0.1
###

# Pre-Tokenization of the Sentences
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

with open(os.path.join(path, 'Sentences_for_contrastive.json'), 'r') as file:
    sentences = json.load(file)
with open(os.path.join(path, 'Clusters_for_contrastive.json'), 'r') as file:
    clusters = json.load(file)  
with open(os.path.join(path, 'Claims_for_contrastive.json'), 'r') as file:
    claims = json.load(file)  
tokenized_sentences = [tokenizer(sentence, return_tensors='pt', padding='max_length', truncation=True, max_length=128) for sentence in sentences]
tokenized_claims = [tokenizer(claim, return_tensors='pt', padding='max_length', truncation=True, max_length=128) for claim in claims]
## clusters and tokenized_sentences are global object (not shuffled) ##


# Dataset Preparation
class MyDataset(Dataset):
    def __init__(self, tokenized_sentences, clusters, tokenized_claims):
        self.tokenized_sentences = tokenized_sentences
        self.clusters = clusters
        self.indices = list(range(len(clusters)))  # Original indices
        self.tokenized_claims = tokenized_claims

    def __len__(self):
        return len(self.tokenized_sentences)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        return self.tokenized_sentences[original_idx], original_idx, self.tokenized_claims[original_idx]

# Model Definition - Using CLS Token Embedding
class BertForContrastiveLearning(nn.Module):
    def __init__(self):
        super(BertForContrastiveLearning, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token embedding
        return cls_embedding

# Contrastive Loss Function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, anchor, positive, negative):
        distance_positive = (1 - self.cos(anchor, positive))/2 # set 0 to 1
        distance_negative = (1 - self.cos(anchor, negative))/2
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

# Claim Distance Loss Function
class ClaimDistanceLoss(nn.Module):
    def __init__(self):
        super(ClaimDistanceLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, anchor_batch, claim_batch):
        # anchor_batch and claim_batch are batches of embeddings
        # The cosine similarity is computed for each pair in the batch
        return ((1 - self.cos(anchor_batch, claim_batch))/2).mean()


# Initialize Dataset and DataLoader
dataset = MyDataset(tokenized_sentences, clusters, tokenized_claims)

train_size = int(0.8 * len(tokenized_sentences))
val_size = len(tokenized_sentences) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

def select_dynamic_samples(clusters, batch_indices):
    positive_indices = []
    negative_indices_lists = []
    batch_clusters = [clusters[idx] for idx in batch_indices]  # Clusters corresponding to the current batch

    for idx, cluster in zip(batch_indices, batch_clusters):
        # Choose a positive sample (different index but same cluster)
        # global index
        positive_options = [i for i, c in enumerate(clusters) if c == cluster and i != idx]
        positive_indices.append(random.choice(positive_options))

        # Choose all negative samples (same batch, different cluster)
        # batch index, not global index for reusing all_embeddings
        negative_options = [i for i, c in enumerate(batch_clusters) if c != cluster]
        negative_indices_lists.append(negative_options if negative_options else [i])

    return positive_indices, negative_indices_lists

def embedding_extract(model, all_embeddings, positive_indices, negative_indices_lists):
    # Gather all positive samples
    positive_input_ids = torch.stack([tokenized_sentences[idx]['input_ids'].to('cuda') for idx in positive_indices]).squeeze(1)
    positive_attention_masks = torch.stack([tokenized_sentences[idx]['attention_mask'].to('cuda') for idx in positive_indices]).squeeze(1)

    # Forward pass for all positive samples
    positive_embeddings = model(positive_input_ids, positive_attention_masks)

    # Initialize tensors for batch loss calculation
    anchor_embs = []
    positive_embs = []
    negative_embs = []

    for i, negative_indices in enumerate(negative_indices_lists):
        anchor_embedding = all_embeddings[i]  
        positive_embedding = positive_embeddings[i]

        for negative_idx in negative_indices:
            negative_embedding = all_embeddings[negative_idx]

            # Append embeddings to respective lists
            anchor_embs.append(anchor_embedding)
            positive_embs.append(positive_embedding)
            negative_embs.append(negative_embedding)
            
    return anchor_embs, positive_embs, negative_embs

def validate(model, dataloader, loss_fn, dist_loss_fn):
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Unpack the batch data
            batch_data, batch_indices, claim_data = batch
            input_ids, attention_masks = batch_data['input_ids'].squeeze(1).to('cuda'), batch_data['attention_mask'].squeeze(1).to('cuda')
            input_ids_claim, attention_masks_claim = claim_data['input_ids'].squeeze(1).to('cuda'), claim_data['attention_mask'].squeeze(1).to('cuda')
    
            # Forward pass for the entire batch (anchors and negatives)
            all_embeddings = model(input_ids, attention_masks)
    
            positive_indices, negative_indices_lists = select_dynamic_samples(clusters, batch_indices)
    
            anchor_embs, positive_embs, negative_embs = embedding_extract(model, all_embeddings, positive_indices, negative_indices_lists)
                        
            # Convert lists to tensors
            anchor_embs_tensor = torch.stack(anchor_embs)
            positive_embs_tensor = torch.stack(positive_embs)
            negative_embs_tensor = torch.stack(negative_embs)
    
            # Calculate triplet loss
            triplet_loss = loss_fn(anchor_embs_tensor, positive_embs_tensor, negative_embs_tensor)
    
            # Calculate distance loss
            claim_embeddings = model(input_ids_claim, attention_masks_claim)
            dist_loss = dist_loss_fn(all_embeddings, claim_embeddings)
    
            # Ensembled loss
            batch_loss = triplet_loss+alpha*dist_loss

            # Calculate batch loss
            val_losses.append(batch_loss.item())

    return sum(val_losses) / len(val_losses)

# Initialize Model, Loss Function, and Optimizer
model = BertForContrastiveLearning().to('cuda')
loss_fn = ContrastiveLoss()
dist_loss_fn = ClaimDistanceLoss()
num_epochs = 50
save_path = f"./Models/Models_dist_scale_margin{margin}_alpha{alpha}"
os.makedirs(save_path, exist_ok=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
scheduler = OneCycleLR(optimizer, max_lr=2e-5, pct_start=0.15, steps_per_epoch=len(train_dataloader), epochs=num_epochs)

# Check if multiple GPUs are available and wrap the model using DataParallel
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

# Early Stopping Setup
patience = 5
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    train_losses = []
    for batch in tqdm(train_dataloader):
        optimizer.zero_grad()

        # Unpack the batch data
        batch_data, batch_indices, claim_data = batch
        input_ids, attention_masks = batch_data['input_ids'].squeeze(1).to('cuda'), batch_data['attention_mask'].squeeze(1).to('cuda')
        input_ids_claim, attention_masks_claim = claim_data['input_ids'].squeeze(1).to('cuda'), claim_data['attention_mask'].squeeze(1).to('cuda')

        # Forward pass for the entire batch (anchors and negatives)
        all_embeddings = model(input_ids, attention_masks)

        positive_indices, negative_indices_lists = select_dynamic_samples(clusters, batch_indices)

        anchor_embs, positive_embs, negative_embs = embedding_extract(model, all_embeddings, positive_indices, negative_indices_lists)
                    
        # Convert lists to tensors
        anchor_embs_tensor = torch.stack(anchor_embs)
        positive_embs_tensor = torch.stack(positive_embs)
        negative_embs_tensor = torch.stack(negative_embs)

        # Calculate triplet loss
        triplet_loss = loss_fn(anchor_embs_tensor, positive_embs_tensor, negative_embs_tensor)

        # Calculate distance loss
        claim_embeddings = model(input_ids_claim, attention_masks_claim)
        dist_loss = dist_loss_fn(all_embeddings, claim_embeddings)

        # Ensembled loss
        batch_loss = triplet_loss+alpha*dist_loss
        
        batch_loss.backward()
        optimizer.step()
        scheduler.step()

        train_losses.append(batch_loss.item())

    train_loss = sum(train_losses) / len(train_losses)

    # Validation Step
    val_loss = validate(model, val_dataloader, loss_fn, dist_loss_fn)
    
    print(f"Epoch {epoch + 1}, Training Loss: {train_loss} / Validation Loss: {val_loss}")

    # Early Stopping Check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save the best model
        model_save_name = f'{name}_best.pt'
        torch.save(model.state_dict(), os.path.join(save_path, model_save_name))
        print(f"Best model saved to {os.path.join(save_path, model_save_name)}")
    else:
        patience_counter += 1
        print(f"Stopping count: {patience_counter}")
        if patience_counter >= patience:
            print(f"Stopping early at epoch {epoch + 1}")
            break
    
    # Save the model
    model_save_name = f'{name}_epoch_{epoch + 1}.pt'
    torch.save(model.state_dict(), os.path.join(save_path, model_save_name))
    print(f"Model saved to {os.path.join(save_path, model_save_name)}")
