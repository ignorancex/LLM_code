import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
import time
from tqdm import tqdm

# class Attention(nn.Module):
#     def __init__(self, dim, max_seq_len=197, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0., dp_rank=8, mustInPos=None):
#         super().__init__()
        
#         assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.head_dim = head_dim
#         self.scale = head_dim ** -0.5

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
        
#         self.seq_len = max_seq_len
#         self.dp_rank = dp_rank
#         self.fold = 8
        
#         # Simple linear layer for combining features
#         self.combine_features = nn.Linear(dim * 2, dim)
        
#         self.ln_1 = nn.LayerNorm(dim)
#         self.ln_2 = nn.LayerNorm(dim)
#         self.ln_post = nn.LayerNorm(dim)
#         self.ln_pre = nn.LayerNorm(dim)
        
#     def combine_heads(self, X):
#         X = X.transpose(1, 2)
#         X = X.reshape(X.size(0), X.size(1), self.num_heads * self.head_dim)
#         return X

#     def split_heads(self, X):
#         X = X.reshape(X.size(0), X.size(1), self.num_heads, self.head_dim)
#         X = X.transpose(1, 2)
#         return X
        
#     def forward(self, x):
#         B, N, C = x.shape
        
#         # Simple averaging for feature combination
#         if C // self.fold > 0:
#             u = x.reshape(B, N, self.fold, C // self.fold)
#             u = torch.mean(u, dim=-1)
#         else:
#             u = x
            
#         # Simple concatenation with original input
#         combined = torch.cat([u, x], dim=-1)
        
#         # Project back to original dimension
#         h = self.combine_features(combined)
        
#         # Apply dropout and normalization
#         x = self.attn_drop(F.gelu(h))
        
#         # Standard attention mechanism
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv.unbind(0)
        
#         # Compute attention scores
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
        
#         # Apply attention to values
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
#         # Final projection and dropout
#         x = self.proj(x)
#         x = self.proj_drop(x)
        
#         return x

class Attention(nn.Module):
    def __init__(self, dim, max_seq_len=197, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0., dp_rank=8, mustInPos=None):
        super().__init__()
        
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.seq_len = max_seq_len
        self.dp_rank = dp_rank
        self.fold = 8
        
        # Define projection layers
        self.pre_proj = nn.Linear(dim // self.fold, dim)
        self.combine_proj = nn.Linear(2 * dim, dim)
        
        # Layer norms
        self.ln_1 = nn.LayerNorm(dim)
        self.ln_2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Initial layer norm
        x = self.ln_1(x)
        
        # Compute averaged features
        if C // self.fold > 0:
            u = x.reshape(B, N, self.fold, C // self.fold)
            u = torch.mean(u, dim=2)  # [B, N, C//fold]
            u = self.pre_proj(u)  # [B, N, C]
        else:
            u = x
        
        # Concatenate along feature dimension and project
        h = torch.cat([u, x], dim=-1)  # [B, N, 2C]
        h = self.combine_proj(h)  # [B, N, C]
        h = self.ln_2(h)
        
        # Standard attention mechanism
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # Final projection
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None
    
    def get_emb(self, sin_inp):
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return torch.flatten(emb, -2, -1)

    def forward(self, tensor):
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = self.get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, :self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc

class DataEmbedding(nn.Module):
    def __init__(self, enc_in, d_model, embed_type='fixed', freq='h', dropout=0.1, seq_len=96):
        super(DataEmbedding, self).__init__()
        self.tiny_conv_linear = torch.nn.Conv1d(in_channels=enc_in, 
                                              out_channels=d_model*1, 
                                              kernel_size=1,
                                              padding=0, 
                                              groups=1)
        self.dropout = nn.Dropout(p=dropout)        
        self.posembeding = PositionalEncoding1D(channels=d_model)

    def forward(self, x, x_mark=None):
        device = x.device
        x1 = self.dropout(self.tiny_conv_linear(x.permute(0,2,1)).permute(0,2,1))
        pos_encoding = self.posembeding(x1).to(device)
        return pos_encoding + x1

class CustomS3AttentionModel(nn.Module):
    def __init__(self, configs):
        super(CustomS3AttentionModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.enc_in = configs.enc_in
        
        # Create all components
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, 
                                         configs.freq, configs.dropout).to(self.device)
        
        self.encoder = nn.ModuleList([
            Attention(dim=configs.d_model, 
                     max_seq_len=configs.seq_len, 
                     num_heads=configs.n_heads,
                     qkv_bias=False,
                     attn_drop=configs.dropout,
                     proj_drop=configs.dropout,
                     dp_rank=8).to(self.device)
            for _ in range(configs.e_layers)
        ])
        
        # Output layers
        self.projection = nn.Linear(configs.d_model, configs.enc_in).to(self.device)
        self.group_emotion_projection = nn.Linear(configs.d_model, 9).to(self.device)
        self.group_emotion_category_projection = nn.Linear(configs.d_model, 3).to(self.device)
        self.class_emotion_projection = nn.Linear(configs.d_model, 9).to(self.device)
        self.class_valence_projection = nn.Linear(configs.d_model, 1).to(self.device)
        self.class_arousal_projection = nn.Linear(configs.d_model, 1).to(self.device)
        self.event_emotion_projection = nn.Linear(configs.d_model, 10).to(self.device)
        self.event_projection = nn.Linear(configs.d_model, 10).to(self.device)

    def forward(self, x_enc, x_mark_enc=None):
        x_enc = x_enc.to(self.device)
        if x_mark_enc is not None:
            x_mark_enc = x_mark_enc.to(self.device)
            
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        
        # Encoder
        for attn_layer in self.encoder:
            enc_out = attn_layer(enc_out) + enc_out
            
        # Predictions
        time_series_pred = self.projection(enc_out)
        
        last_hidden = enc_out[:, -1, :]
        group_emotion = self.group_emotion_projection(last_hidden)
        group_emotion_category = self.group_emotion_category_projection(last_hidden)
        class_emotion = self.class_emotion_projection(last_hidden)
        class_valence = self.class_valence_projection(last_hidden)
        class_arousal = self.class_arousal_projection(last_hidden)
        event_emotion = self.event_emotion_projection(last_hidden)
        event = self.event_projection(last_hidden)
        
        return time_series_pred, group_emotion, group_emotion_category, class_emotion, class_valence, class_arousal, event_emotion, event

class CustomEmotionDataset(Dataset):
    def __init__(self, face_features_path, text_latent_path1, text_latent_path2, text_latent_path3, labels_path, seq_len, pred_len, root_path):
        self.root_path = root_path
        self.face_features_path = os.path.join(root_path, face_features_path)
        self.text_latent1_path = os.path.join(root_path, text_latent_path1)
        self.text_latent2_path = os.path.join(root_path, text_latent_path2)
        self.text_latent3_path = os.path.join(root_path, text_latent_path3)
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Load labels
        self.labels_df = pd.read_csv(os.path.join(root_path, labels_path))
        
        # Get valid subfolders
        self.valid_subfolders = self._get_valid_subfolders()

    def _get_valid_subfolders(self):
        face_subfolders = set(os.listdir(self.face_features_path))
        text1_files = set(f.split('.')[0] for f in os.listdir(self.text_latent1_path))
        text2_files = set(f.split('.')[0] for f in os.listdir(self.text_latent2_path))
        text3_files = set(f.split('.')[0] for f in os.listdir(self.text_latent3_path))
        label_files = set(self.labels_df['video_file_name'])
        
        return list(face_subfolders & text1_files & text2_files & text3_files & label_files)

    def __len__(self):
        return len(self.valid_subfolders)

    def __getitem__(self, idx):
        subfolder = self.valid_subfolders[idx]
        
        # Load face features
        face_features = []
        face_subfolder_path = os.path.join(self.face_features_path, subfolder)
        face_files = sorted(os.listdir(face_subfolder_path))
        for file in face_files[:self.seq_len + self.pred_len]:
            feature = np.load(os.path.join(face_subfolder_path, file))
            face_features.append(feature)
        
        # Pad or truncate face features if necessary
        if len(face_features) < self.seq_len + self.pred_len:
            pad_length = self.seq_len + self.pred_len - len(face_features)
            face_features.extend([np.zeros((1024,))] * pad_length)
        elif len(face_features) > self.seq_len + self.pred_len:
            face_features = face_features[:self.seq_len + self.pred_len]
        
        face_features = np.array(face_features)
        
        # Load text latent features
        text_latent1 = np.load(os.path.join(self.text_latent1_path, f"{subfolder}.npy"))
        text_latent2 = np.load(os.path.join(self.text_latent2_path, f"{subfolder}.npy"))
        text_latent3 = np.load(os.path.join(self.text_latent3_path, f"{subfolder}.npy"))
        
        # Reshape face features and text latent features
        face_features = face_features.reshape(self.seq_len + self.pred_len, -1)
        text_latent1 = np.tile(text_latent1, (self.seq_len + self.pred_len, 1))
        text_latent2 = np.tile(text_latent2, (self.seq_len + self.pred_len, 1))
        text_latent3 = np.tile(text_latent3, (self.seq_len + self.pred_len, 1))
        
        # Combine all features
        combined_features = np.concatenate([face_features, text_latent1, text_latent2, text_latent3], axis=1)
        
        # Split into input and target sequences
        x = combined_features[:self.seq_len]
        y = combined_features[self.seq_len:self.seq_len + self.pred_len]
        
        # Get labels
        labels = self.labels_df[self.labels_df['video_file_name'] == subfolder].iloc[0]
        
        return (
            torch.FloatTensor(x),
            torch.FloatTensor(y),
            torch.LongTensor([labels['group_emotion']]),
            torch.LongTensor([labels['group_emotion_category']]),
            torch.LongTensor([labels['class_emotion']]),
            torch.FloatTensor([labels['class_valence']]),
            torch.FloatTensor([labels['class_arousal']]),
            torch.LongTensor([labels['event_emotion']]),
            torch.LongTensor([labels['event']])
        )

def custom_collate_fn(batch):
    x = torch.stack([item[0] for item in batch])
    y = torch.stack([item[1] for item in batch])
    group_emotions = torch.cat([item[2] for item in batch])
    group_emotion_categories = torch.cat([item[3] for item in batch])
    class_emotions = torch.cat([item[4] for item in batch])
    class_valences = torch.cat([item[5] for item in batch])
    class_arousals = torch.cat([item[6] for item in batch])
    event_emotions = torch.cat([item[7] for item in batch])
    events = torch.cat([item[8] for item in batch])
    return x, y, group_emotions, group_emotion_categories, class_emotions, class_valences, class_arousals, event_emotions, events

def get_data_loaders(args):
    train_dataset = CustomEmotionDataset(
        face_features_path=os.path.join(args.root_path, 'train', 'face_features'),
        text_latent_path1=os.path.join(args.root_path, 'train', 'text_latent1'),
        text_latent_path2=os.path.join(args.root_path, 'train', 'text_latent2'),
        text_latent_path3=os.path.join(args.root_path, 'train', 'text_latent3'),
        labels_path=os.path.join(args.root_path, 'train', 'labels.csv'),
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        root_path=args.root_path
    )
    
    val_dataset = CustomEmotionDataset(
        face_features_path=os.path.join(args.root_path, 'val', 'face_features'),
        text_latent_path1=os.path.join(args.root_path, 'val', 'text_latent1'),
        text_latent_path2=os.path.join(args.root_path, 'val', 'text_latent2'),
        text_latent_path3=os.path.join(args.root_path, 'val', 'text_latent3'),
        labels_path=os.path.join(args.root_path, 'val', 'labels.csv'),
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        root_path=args.root_path
    )
    
    test_dataset = CustomEmotionDataset(
        face_features_path=os.path.join(args.root_path, 'test', 'face_features'),
        text_latent_path1=os.path.join(args.root_path, 'test', 'text_latent1'),
        text_latent_path2=os.path.join(args.root_path, 'test', 'text_latent2'),
        text_latent_path3=os.path.join(args.root_path, 'test', 'text_latent3'),
        labels_path=os.path.join(args.root_path, 'test', 'labels.csv'),
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        root_path=args.root_path
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

def normalize_to_range(values, new_min=0.35, new_max=0.675):
    min_val = np.min(values)
    max_val = np.max(values)
    normalized = (values - min_val) / (max_val - min_val)
    return normalized * (new_max - new_min) + new_min

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, f'checkpoint_{val_loss:.6f}.pth')
        torch.save(model.state_dict(), save_path)
        self.val_loss_min = val_loss

class CustomExpMain:
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            print(f'Using device: {device}')
            return device
        return torch.device('cpu')

    def _build_model(self):
        model = CustomS3AttentionModel(self.args)
        return model

    def _get_data(self, flag):
        try:
            data_loaders = get_data_loaders(self.args)
            return None, data_loaders[flag]
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def _select_optimizer(self):
        model_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion_reg = nn.MSELoss()
        criterion_cls = nn.CrossEntropyLoss()
        return criterion_reg, criterion_cls

    def train_step(self, batch_x, batch_y, group_emotions, group_emotion_categories, 
                class_emotions, class_valences, class_arousals, 
                event_emotions, events, criterion_reg, criterion_cls):
        
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        group_emotions = group_emotions.long().to(self.device)
        group_emotion_categories = group_emotion_categories.long().to(self.device)
        class_emotions = class_emotions.long().to(self.device)
        class_valences = class_valences.float().to(self.device)
        class_arousals = class_arousals.float().to(self.device)
        event_emotions = event_emotions.long().to(self.device)
        events = events.long().to(self.device)

        outputs = self.model(batch_x)
        time_series_pred, group_emotion_pred, group_emotion_category_pred, \
        class_emotion_pred, class_valence_pred, class_arousal_pred, \
        event_emotion_pred, event_pred = outputs

        loss_ts = criterion_reg(time_series_pred, batch_y)
        loss_group = criterion_cls(group_emotion_pred, group_emotions.squeeze())
        loss_group_category = criterion_cls(group_emotion_category_pred, group_emotion_categories.squeeze())
        loss_class = criterion_cls(class_emotion_pred, class_emotions.squeeze())
        loss_valence = criterion_reg(class_valence_pred.squeeze(), class_valences.squeeze())
        loss_arousal = criterion_reg(class_arousal_pred.squeeze(), class_arousals.squeeze())
        loss_event_emotion = criterion_cls(event_emotion_pred, event_emotions.squeeze())
        loss_event = criterion_cls(event_pred, events.squeeze())

        loss = (loss_ts + loss_group + loss_group_category + loss_class + 
                loss_valence + loss_arousal + loss_event_emotion + loss_event)
        
        return loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion_reg, criterion_cls = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            
            for i, (batch_x, batch_y, group_emotions, group_emotion_categories, 
                    class_emotions, class_valences, class_arousals, 
                    event_emotions, events) in enumerate(train_loader):
                
                model_optim.zero_grad()
                
                loss = self.train_step(batch_x, batch_y, group_emotions, 
                                    group_emotion_categories, class_emotions,
                                    class_valences, class_arousals, 
                                    event_emotions, events,
                                    criterion_reg, criterion_cls)
                
                loss.backward()
                model_optim.step()
                train_loss.append(loss.item())

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion_reg, criterion_cls)

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f}")
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

        return self.model

    def vali(self, vali_data, vali_loader, criterion_reg, criterion_cls):
        self.model.eval()
        total_loss = []
        
        with torch.no_grad():
            for batch_x, batch_y, group_emotions, group_emotion_categories, \
                class_emotions, class_valences, class_arousals, \
                event_emotions, events in vali_loader:
                
                loss = self.train_step(batch_x, batch_y, group_emotions,
                                    group_emotion_categories, class_emotions,
                                    class_valences, class_arousals,
                                    event_emotions, events,
                                    criterion_reg, criterion_cls)
                
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

import torch
import argparse
import os
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

def concordance_correlation_coefficient(y_true, y_pred):
    """Calculate concordance correlation coefficient."""
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    
    covariance = np.cov(y_true, y_pred)[0,1]
    
    numerator = 2 * covariance
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    
    return numerator / denominator if denominator != 0 else 0

def test_model(args, model_path):
    # Initialize the model and load weights
    model = CustomS3AttentionModel(args).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    model.eval()
    
    # Get test data loader
    data_loaders = get_data_loaders(args)
    test_loader = data_loaders['test']
    
    # Initialize metrics
    total_mse = 0
    num_samples = 0
    group_emotion_preds = []
    group_emotion_trues = []
    group_category_preds = []
    group_category_trues = []
    class_emotion_preds = []
    class_emotion_trues = []
    valence_preds = []
    valence_trues = []
    arousal_preds = []
    arousal_trues = []
    event_emotion_preds = []
    event_emotion_trues = []
    event_preds = []
    event_trues = []
    
    with torch.no_grad():
        for batch_x, batch_y, group_emotions, group_emotion_categories, \
            class_emotions, class_valences, class_arousals, \
            event_emotions, events in test_loader:
            
            # Move data to device
            batch_x = batch_x.float().to(model.device)
            batch_y = batch_y.float().to(model.device)
            
            # Get predictions
            time_series_pred, group_emotion_pred, group_emotion_category_pred, \
            class_emotion_pred, class_valence_pred, class_arousal_pred, \
            event_emotion_pred, event_pred = model(batch_x)
            
            # Calculate MSE batch-wise
            batch_mse = F.mse_loss(time_series_pred, batch_y)
            total_mse += batch_mse.item() * batch_x.size(0)
            num_samples += batch_x.size(0)
            
            # Store predictions and true values
            group_emotion_preds.extend(torch.argmax(group_emotion_pred, dim=1).cpu().numpy())
            group_emotion_trues.extend(group_emotions.numpy().squeeze())
            
            group_category_preds.extend(torch.argmax(group_emotion_category_pred, dim=1).cpu().numpy())
            group_category_trues.extend(group_emotion_categories.numpy().squeeze())
            
            class_emotion_preds.extend(torch.argmax(class_emotion_pred, dim=1).cpu().numpy())
            class_emotion_trues.extend(class_emotions.numpy().squeeze())
            
            valence_preds.extend(class_valence_pred.cpu().numpy().squeeze())
            valence_trues.extend(class_valences.numpy().squeeze())
            
            arousal_preds.extend(class_arousal_pred.cpu().numpy().squeeze())
            arousal_trues.extend(class_arousals.numpy().squeeze())
            
            event_emotion_preds.extend(torch.argmax(event_emotion_pred, dim=1).cpu().numpy())
            event_emotion_trues.extend(event_emotions.numpy().squeeze())
            
            event_preds.extend(torch.argmax(event_pred, dim=1).cpu().numpy())
            event_trues.extend(events.numpy().squeeze())
    
    # Calculate CCC for valence and arousal
    valence_ccc = concordance_correlation_coefficient(
        np.array(valence_trues),
        np.array(valence_preds)
    )
    
    arousal_ccc = concordance_correlation_coefficient(
        np.array(arousal_trues),
        np.array(arousal_preds)
    )
    
    # Calculate final metrics
    results = {
        'time_series_mse': total_mse / num_samples,
        'group_emotion_accuracy': accuracy_score(group_emotion_trues, group_emotion_preds),
        'group_category_accuracy': accuracy_score(group_category_trues, group_category_preds),
        'class_emotion_accuracy': accuracy_score(class_emotion_trues, class_emotion_preds),
        'valence_mae': mean_absolute_error(valence_trues, valence_preds),
        'valence_ccc': valence_ccc,
        'arousal_mae': mean_absolute_error(arousal_trues, arousal_preds),
        'arousal_ccc': arousal_ccc,
        'event_emotion_accuracy': accuracy_score(event_emotion_trues, event_emotion_preds),
        'event_accuracy': accuracy_score(event_trues, event_preds)
    }
    
    return results
# def main():
#     fix_seed = 2021
#     random.seed(fix_seed)
#     torch.manual_seed(fix_seed)
#     np.random.seed(fix_seed)

#     parser = argparse.ArgumentParser(description='Emotion Prediction Model')
    
#     # basic config
#     parser.add_argument('--is_training', type=int, default=0, help='status')
#     parser.add_argument('--task_id', type=str, default='emotion_prediction', help='task id')
#     parser.add_argument('--model', type=str, default='SimplifiedAttention', help='model name')
    
#     # data loader
#     parser.add_argument('--data', type=str, default='custom', help='dataset type')
#     parser.add_argument('--root_path', type=str, default='../Dataset', help='root path of the data file')
#     parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]')
#     parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')
#     parser.add_argument('--checkpoints', type=str, default='/media/main/Data/Abhinav_gems/S3Attention/saved_models_final_without_s3', help='location of model checkpoints')

#     # forecasting task
#     parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
#     parser.add_argument('--label_len', type=int, default=48, help='start token length')
#     parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')

#     # model define
#     parser.add_argument('--enc_in', type=int, default=3328, help='encoder input size')
#     parser.add_argument('--dec_in', type=int, default=3328, help='decoder input size')
#     parser.add_argument('--c_out', type=int, default=3328, help='output size')
#     parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
#     parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
#     parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
#     parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
#     parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
#     parser.add_argument('--factor', type=int, default=1, help='attn factor')
#     parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder')
#     parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
#     parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
#     parser.add_argument('--activation', type=str, default='gelu', help='activation')
#     parser.add_argument('--output_attention', action='store_true', help='whether to output attention')

#     # optimization
#     parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
#     parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
#     parser.add_argument('--batch_size', type=int, default=32, help='batch size')
#     parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
#     parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
#     parser.add_argument('--des', type=str, default='test', help='exp description')
#     parser.add_argument('--loss', type=str, default='mse', help='loss function')
#     parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

#     # GPU
#     parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
#     parser.add_argument('--gpu', type=int, default=0, help='gpu id')
#     parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus')
#     parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids')

#     args = parser.parse_args()

#     # Print args
#     print('Arguments in experiment:')
#     print(args)

#     # Check CUDA availability
#     if torch.cuda.is_available():
#         print(f"CUDA is available")
#         print(f"CUDA device count: {torch.cuda.device_count()}")
#         print(f"Current CUDA device: {torch.cuda.current_device()}")
#         print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
#         torch.cuda.empty_cache()

#     Exp = CustomExpMain

#     if args.is_training:
#         setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_{}_{}'.format(
#             args.task_id,
#             args.model,
#             args.data,
#             args.features,
#             args.seq_len,
#             args.label_len,
#             args.pred_len,
#             args.d_model,
#             args.n_heads,
#             args.e_layers,
#             args.d_layers,
#             args.d_ff,
#             args.des,
#             time.strftime("%Y%m%d%H%M%S", time.localtime())
#         )

#         exp = Exp(args)
#         print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        
#         # Train and get best model
#         best_model = exp.train(setting)
        
#         # Save the final model
#         torch.save(best_model.state_dict(), os.path.join(args.checkpoints, setting, 'final_model.pth'))
        
#         print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
#         # Load the best model for testing
#         exp.model.load_state_dict(torch.load(os.path.join(args.checkpoints, setting, 'checkpoint_best.pth')))
#         exp.model.eval()
        
#         # Perform validation on test set
#         _, test_loader = exp._get_data(flag='test')
#         criterion_reg, criterion_cls = exp._select_criterion()
#         test_loss = exp.vali(None, test_loader, criterion_reg, criterion_cls)
#         print(f"Test Loss: {test_loss:.7f}")

#     else:
#         setting = args.model_dir
#         exp = Exp(args)
#         print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        
#         # Load the model
#         exp.model.load_state_dict(torch.load(os.path.join(args.checkpoints, setting, 'checkpoint_best.pth')))
#         exp.model.eval()
        
#         # Perform validation on test set
#         _, test_loader = exp._get_data(flag='test')
#         criterion_reg, criterion_cls = exp._select_criterion()
#         test_loss = exp.vali(None, test_loader, criterion_reg, criterion_cls)
#         print(f"Test Loss: {test_loss:.7f}")

# if __name__ == "__main__":
#     main()

def main():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Emotion Prediction Model')
    
    # basic config
    parser.add_argument('--is_training', type=int, default=0, help='status')
    parser.add_argument('--model_path', type=str, default='../saved_models_final_without_s3/emotion_prediction_SimplifiedAttention_custom_ftM_sl96_ll48_pl1_dm512_nh8_el2_dl1_df2048_test_20250128070424/checkpoint_7.265444.pth', help='path to the trained model checkpoint')
    parser.add_argument('--task_id', type=str, default='emotion_prediction', help='task id')
    parser.add_argument('--model', type=str, default='SimplifiedAttention', help='model name')
    
    # data loader
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='../Dataset', help='root path of the data file')
    parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=3328, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=3328, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=3328, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus')
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids')

    args = parser.parse_args()

    # Print args
    print('Arguments in experiment:')
    print(args)

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA is available")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()

    # Training mode
    if args.is_training:
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_{}_{}'.format(
            args.task_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.des,
            time.strftime("%Y%m%d%H%M%S", time.localtime())
        )

        exp = CustomExpMain(args)
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        
        # Train and get best model
        best_model = exp.train(setting)
        
        # Save the final model
        final_model_path = os.path.join(args.checkpoints, setting, 'final_model.pth')
        torch.save(best_model.state_dict(), final_model_path)
        print(f'Final model saved to {final_model_path}')
        
        # Test using best model from early stopping
        print('>>>>>>>testing with best model : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        best_model_path = os.path.join(args.checkpoints, setting, 'checkpoint_best.pth')
        print(f'Loading best model from {best_model_path}')
        
        try:
            test_results = test_model(args, best_model_path)
            
            print("\nTest Results with Best Model:")
            print("-" * 50)
            for metric, value in test_results.items():
                print(f"{metric}: {value:.4f}")
            print("-" * 50)
            
            # Save test results
            results_path = os.path.join(args.checkpoints, setting, 'test_results.txt')
            with open(results_path, 'w') as f:
                f.write("Test Results:\n")
                f.write("-" * 50 + "\n")
                for metric, value in test_results.items():
                    f.write(f"{metric}: {value:.4f}\n")
                f.write("-" * 50 + "\n")
            print(f'Test results saved to {results_path}')
            
        except Exception as e:
            print(f"Error during testing: {str(e)}")
            raise

    # Testing mode
    else:
        if args.model_path is None:
            raise ValueError("Please provide --model_path when testing")
        
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model file not found: {args.model_path}")
            
        print('>>>>>>>testing with provided model<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        print(f'Loading model from: {args.model_path}')
        
        try:
            # Run testing
            test_results = test_model(args, args.model_path)
            
            print("\nTest Results:")
            print("-" * 50)
            for metric, value in test_results.items():
                print(f"{metric}: {value:.4f}")
            print("-" * 50)
            
            # Save test results
            results_dir = os.path.dirname(args.model_path)
            results_path = os.path.join(results_dir, 'test_results.txt')
            with open(results_path, 'w') as f:
                f.write("Test Results:\n")
                f.write("-" * 50 + "\n")
                for metric, value in test_results.items():
                    f.write(f"{metric}: {value:.4f}\n")
                f.write("-" * 50 + "\n")
            print(f'Test results saved to {results_path}')
            
        except Exception as e:
            print(f"Error during testing: {str(e)}")
            raise

if __name__ == "__main__":
    main()