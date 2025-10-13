import chardet
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from Hydra import Hydra
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, precision_score
import torch.nn.functional as F
import warnings
import sys

warnings.filterwarnings("ignore")


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# Feature Enhancement Module
class FeatureEnhancement(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.enhance = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.shortcut = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

    def forward(self, x):
        return self.enhance(x) + self.shortcut(x)


# Versus Learning Function
class VersusLearning(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, p1,  p2):
        """
        p1: [B, D]
        p2: [B, D]
        """
        p1 = F.normalize(p1, p=2, dim=-1)
        p2 = F.normalize(p2, p=2, dim=-1)

        opponent_distance = self.cosine_similarity(p1, p2)  # [B]
        ver_loss = torch.mean(F.relu(self.margin + opponent_distance))
        return ver_loss

class CAAM(nn.Module):
    def __init__(self, input_dim, num_heads=8):
        super().__init__()
        self.MHA = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)

        # used to learn weights for each dimension (serve, return, psychology, fatigue)
        self.attention_weights = nn.Parameter(torch.ones(4))  # Initially equal weights

    def forward(self, p1_aspects, p2_aspects):
        """
        p1_aspects: [B, 4, D]
        p2_aspects: [B, 4, D]
        """
        # Concatenate p1 and p2 aspects
        combined = torch.cat([p1_aspects, p2_aspects], dim=1)  # [B, 8, D]

        # Apply cross attention
        attended, _ = self.MHA(combined, combined, combined)  # [B, 8, D]

        # Split back into p1 and p2
        p1_attended = attended[:, :4, :]  # [B, 4, D]
        p2_attended = attended[:, 4:, :]  # [B, 4, D]

        # Calculate attention weights
        weights = torch.softmax(self.attention_weights, dim=0)  # [4], normalized weights

        # Weighted sum, reduce to [B, 1, D]
        p1_fused = torch.sum(p1_attended * weights.view(1, 4, 1), dim=1, keepdim=True)  # [B, 1, D]
        p2_fused = torch.sum(p2_attended * weights.view(1, 4, 1), dim=1, keepdim=True)  # [B, 1, D]

        return p1_fused, p2_fused  # [B, 1, D], [B, 1, D]

# HydraNet
class HydraNet(nn.Module):
    def __init__(self, num_matches, num_sets, num_games, game_offset, feature_dim=16, mamba_dim=16, hidden_dim=64,
                 num_heads=8, memory_dim=16):
        super().__init__()

        # Store game_offset
        self.game_offset = game_offset

        self.match_embedding = nn.Embedding(num_matches, 16)
        self.set_embedding = nn.Embedding(num_sets, 8)
        self.game_embedding = nn.Embedding(num_games, 8)

        # Feature interaction with normalization
        self.input_norm = nn.LayerNorm(feature_dim)

        # Hydra backbone for p1 and p2
        self.Hydra_p1 = Hydra(
            d_model=mamba_dim,
            n_layer=1,
            d_state=16,
            expand=2,
            headdim=32
        )
        self.Hydra_p2 = Hydra(
            d_model=mamba_dim,
            n_layer=1,
            d_state=16,
            expand=2,
            headdim=32
        )

        self.feature_combiner_p1 = nn.Sequential(
            nn.Linear(feature_dim, mamba_dim),  # p1_features: [L_g, 16]
            nn.LayerNorm(mamba_dim),
            nn.Dropout(0.1)
        )
        self.feature_combiner_p2 = nn.Sequential(
            nn.Linear(feature_dim, mamba_dim),  # p2_features: [L_g, 16]
            nn.LayerNorm(mamba_dim),
            nn.Dropout(0.1)
        )

        # Feature grouping networks with residual connections for p1 and p2
        self.serve_net_p1 = FeatureEnhancement(6, hidden_dim)
        self.return_net_p1 = FeatureEnhancement(2, hidden_dim)
        self.action_net_p1 = FeatureEnhancement(7, hidden_dim)
        self.pressure_net_p1 = FeatureEnhancement(1, hidden_dim)

        self.serve_net_p2 = FeatureEnhancement(6, hidden_dim)
        self.return_net_p2 = FeatureEnhancement(2, hidden_dim)
        self.action_net_p2 = FeatureEnhancement(7, hidden_dim)
        self.pressure_net_p2 = FeatureEnhancement(1, hidden_dim)

        self.game_linear_p1 = nn.Linear(in_features=16, out_features=16)
        self.game_linear_p2 = nn.Linear(in_features=16, out_features=16)
        self.set_linear_p1 = nn.Linear(in_features=16, out_features=16)
        self.set_linear_p2 = nn.Linear(in_features=16, out_features=16)

        self.CAAM = CAAM(hidden_dim, num_heads)

        # Contrastive Learning Loss
        self.VersusLearning = VersusLearning(margin=1.0)

        # Classifier
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

        # Memory management
        self.register_buffer('global_memory_p1', torch.randn(memory_dim) * 0.01)  # [memory_dim], small random values
        self.register_buffer('global_memory_p2', torch.randn(memory_dim) * 0.01)  # [memory_dim], small random values


        self.global_memory_p1.requires_grad_(True)
        self.global_memory_p2.requires_grad_(True)

        self.memory_dim = memory_dim
        self.match_memory = {}  # {match_id: {'p1': momentum_vector, 'p2': momentum_vector}}

        # Ensure all model parameters require gradients
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, batch_dict):
        features = batch_dict['features']  # [B, L, 32]  # p1 and p2 concatenated

        labels = batch_dict.get('label', None)  # List of [num_games -1]
        match_ids = batch_dict['match_id']  # [B]
        sets = batch_dict['set']  # [B, L]
        games = batch_dict['game']  # [B, L]

        B = features.shape[0]

        print(f"Model is running on device: {next(self.parameters()).device}")

        # Initialize lists to collect outputs and losses
        all_outputs = []
        all_versus_losses = []

        # Initialize list to collect predictions corresponding to labels
        match_predictions = []

        for b in range(B):
            match_id = match_ids[b].item()
            feature = features[b]  # [L, 32]
            set_no = sets[b]  # [L]
            game_no = games[b]  # [L]

            # Initialize momentum for this match
            if match_id in self.match_memory:
                momentum_p1 = self.match_memory[match_id]['p1'].clone()  # [memory_dim]
                momentum_p2 = self.match_memory[match_id]['p2'].clone()  # [memory_dim]
                print(f"Existing momentum found for Match ID: {match_id}")
            else:
                momentum_p1 = self.global_memory_p1.clone()  # [memory_dim]
                momentum_p2 = self.global_memory_p2.clone()  # [memory_dim]
                print(f"No existing momentum for Match ID: {match_id}, using global memory.")

            # Identify unique sets in the match, sorted
            unique_sets = torch.unique(set_no).tolist()
            unique_sets = sorted(unique_sets)
            half_sets = len(unique_sets) // 2 + 1  # Half the sets in the match
            last_set = unique_sets[-1] if len(unique_sets) > 0 else None

            # Identify unique games in the last set
            if last_set is not None:
                unique_games_last_set = sorted(torch.unique(game_no[set_no == last_set]).tolist())
                last_game_no = unique_games_last_set[-1] if len(unique_games_last_set) > 0 else None
            else:
                last_game_no = None

            print(
                f"Match {match_id}: last_set={last_set}, last_game_no={last_game_no}, unique_games_last_set={unique_games_last_set if last_set is not None else 'N/A'}")

            for s in unique_sets:

                if s > half_sets:
                    break  # Exit the loop early as we only need to process the first half

                # Create mask for the current set
                set_mask = (set_no == s)
                set_features = feature[set_mask]  # [L_s, 32]

                # Identify unique games in the current set, sorted
                unique_games = sorted(torch.unique(game_no[set_mask]).tolist())

                momentum_p1 = self.set_linear_p1(momentum_p1).detach()
                momentum_p2 = self.set_linear_p2(momentum_p2).detach()

                for g in unique_games:
                    # Create mask for the current game
                    game_mask = (set_no == s) & (game_no == g)
                    game_features = feature[game_mask]  # [L_g, 32]
                    num_points = game_features.shape[0]

                    # Split p1 and p2 features
                    p1_game = game_features[:, :16]  # [L_g, 16]
                    p2_game = game_features[:, 16:]  # [L_g, 16]

                    # Apply input normalization
                    x_p1 = self.input_norm(p1_game)  # [L_g, 16]
                    x_p2 = self.input_norm(p2_game)  # [L_g, 16]

                    # Prepare momentum tensors
                    momentum_p1_tensor = momentum_p1.unsqueeze(0).unsqueeze(1)  # [1, 1, 16]
                    momentum_p2_tensor = momentum_p2.unsqueeze(0).unsqueeze(1)  # [1, 1, 16]

                    # Concatenate momentum with current game data
                    # Current game data: [1, L_g, 16]
                    x_p1_with_momentum = torch.cat([momentum_p1_tensor, x_p1.unsqueeze(0)], dim=1)  # [1, 1 + L_g, 16]
                    # print(f"x_p1_with_momentum.requires_grad: {x_p1_with_momentum.requires_grad}")
                    x_p2_with_momentum = torch.cat([momentum_p2_tensor, x_p2.unsqueeze(0)], dim=1)  # [1, 1 + L_g, 16]
                    # print(f"x_p2_with_momentum.requires_grad: {x_p2_with_momentum.requires_grad}")

                    # Hydra processing
                    y_p1, last_ball_p1 = self.Hydra_p1(
                        x_p1_with_momentum)  # y_p1: [1, 1 + L_g, 16], last_ball_p1: [1, 16]
                    # print(f"y_p1.requires_grad: {y_p1.requires_grad}")
                    y_p2, last_ball_p2 = self.Hydra_p2(
                        x_p2_with_momentum)  # y_p2: [1, 1 + L_g, 16], last_ball_p2: [1, 16]
                    # print(f"y_p2.requires_grad: {y_p2.requires_grad}")

                    y_p1 = y_p1[:, 1:, :]  # remove p1 momentum
                    y_p2 = y_p2[:, 1:, :]  # remove p2 momentum

                    # Update momentum with detached tensors to prevent backprop through history
                    momentum_p1 = last_ball_p1.squeeze(0)  # [16], detached from the computation graph
                    momentum_p2 = last_ball_p2.squeeze(0)  # [16], detached from the computation graph

                    momentum_p1 = self.game_linear_p1(momentum_p1)
                    momentum_p2 = self.game_linear_p2(momentum_p2)

                    # Extract original features for grouping
                    # For p1
                    serve_features_p1 = y_p1[:, :, :6]  # [1, L_g, 6]
                    return_features_p1 = y_p1[:, :, 6:8]  # [1, L_g, 2]
                    action_features_p1 = y_p1[:, :, 8:15]  # [1, L_g, 7]
                    pressure_features_p1 = y_p1[:, :, 15:16]  # [1, L_g, 1]

                    # For p2
                    serve_features_p2 = y_p2[:, :, :6]  # [1, L_g, 6]
                    return_features_p2 = y_p2[:, :, 6:8]  # [1, L_g, 2]
                    action_features_p2 = y_p2[:, :, 8:15]  # [1, L_g, 7]
                    pressure_features_p2 = y_p2[:, :, 15:16]  # [1, L_g, 1]

                    # Process feature groups with residual connections for p1
                    F1_p1 = self.serve_net_p1(serve_features_p1)  # [1, L_g, 64]
                    F2_p1 = self.return_net_p1(return_features_p1)  # [1, L_g, 64]
                    F3_p1 = self.action_net_p1(action_features_p1)  # [1, L_g, 64]
                    F4_p1 = self.pressure_net_p1(pressure_features_p1)  # [1, L_g, 64]

                    # Process feature groups with residual connections for p2
                    F1_p2 = self.serve_net_p2(serve_features_p2)  # [1, L_g, 64]
                    F2_p2 = self.return_net_p2(return_features_p2)  # [1, L_g, 64]
                    F3_p2 = self.action_net_p2(action_features_p2)  # [1, L_g, 64]
                    F4_p2 = self.pressure_net_p2(pressure_features_p2)  # [1, L_g, 64]

                    # Stack features for p1 and p2
                    grouped_features_p1 = torch.stack([F1_p1, F2_p1, F3_p1, F4_p1], dim=2)  # [1, L_g, 4, 64]
                    grouped_features_p2 = torch.stack([F1_p2, F2_p2, F3_p2, F4_p2], dim=2)  # [1, L_g, 4, 64]

                    features_p1_list = torch.unbind(grouped_features_p1, dim=1)  # List with L_g [1, 4, 64] tensors
                    features_p2_list = torch.unbind(grouped_features_p2, dim=1)  # List with L_g [1, 4, 64] tensors

                    features_p1_pre_list = torch.unbind(y_p1, dim=1)  # List with L [1, 16] tensors
                    features_p2_pre_list = torch.unbind(y_p2, dim=1)  # List with L [1, 16] tensors

                    L = len(features_p1_list)
                    for t in range(L):  # go through L time steps
                        # Get the tensor for the current time step
                        y_pt1 = features_p1_pre_list[t]
                        y_pt2 = features_p2_pre_list[t]
                        # Contrastive Learning
                        ver_loss = self.VersusLearning(
                            p1=y_pt1,
                            p2=y_pt2
                        )  # scalar

                        feature_p1_t = features_p1_list[t]  # [1, 4, 64]
                        feature_p2_t = features_p2_list[t]  # [1, 4, 64]

                        # Cross-Attention between p1 and p2
                        attended_p1, attended_p2 = self.CAAM(feature_p1_t,feature_p2_t)  # [1, 1, D], [1, 1, D]

                        final_p1 = attended_p1.squeeze(1)  # [1, D]
                        final_p2 = attended_p2.squeeze(1)  # [1, D]

                        # Combine p1 and p2 features for classification
                        combined_features = torch.cat([final_p1, final_p2], dim=-1)  # [1, 2*D]

                        output = self.classifier(combined_features)  # [1, 1]

                        # Check if current point is the last point of the game
                        if s == half_sets and g == unique_games[-1] and t == L - 1:  # Last game, last point of the half
                            # Collect the prediction for this game's last point
                            print(f"Match {match_id}: Target Set={half_sets}, Last Game={unique_games[-1]}")
                            match_predictions.append(output)
                            all_versus_losses.append(ver_loss)

        #         # After processing all matches in the batch, align predictions with labels
        if match_predictions and labels is not None and len(all_versus_losses) > 0:
            # Convert list of tensors to a single tensor
            predictions_tensor = torch.cat(match_predictions, dim=0)  # [num_predictions, 1]
            predictions_tensor = predictions_tensor.view(-1)  # [num_predictions]

            # Convert labels list to a tensor
            labels_tensor = torch.cat(labels, dim=0).view(-1)  # [num_labels]

            # Debugging: Check for NaN in predictions and labels
            if torch.isnan(predictions_tensor).any():
                print("Warning: NaN detected in predictions_tensor")
            if torch.isnan(labels_tensor).any():
                print("Warning: NaN detected in labels_tensor")

            # Calculate average contrastive loss
            total_ver_loss = torch.stack(all_versus_losses).mean()  # scalar

            # Combine predictions and labels
            final_output = predictions_tensor.unsqueeze(0)  # [1, num_predictions]
            final_labels = labels_tensor.unsqueeze(0)  # [1, num_labels]
        else:
            dummy = torch.zeros(1, self.hidden_dim * 2, device=features.device)
            final_output = self.classifier(dummy)
            final_labels = torch.zeros(1, device=features.device)
            total_ver_loss = torch.tensor(0.0, device=features.device)

        return final_output, final_labels, total_ver_loss


class TennisDataset(Dataset):
    def __init__(self, df, match_to_idx, set_offset, game_offset):
        self.matches = {}
        self.labels = {}
        self.match_ids = []
        self.sets = {}
        self.games = {}

        grouped = df.groupby('match_id')
        total_matches = len(grouped)
        print(f"Total matches in dataset: {total_matches}")

        for match_id, group in grouped:
            group_sorted = group.sort_values(['set_no', 'game_no']).reset_index(drop=True)

            group_sorted['ball_order'] = group_sorted.groupby(['set_no', 'game_no']).cumcount() + 1

            # match winner
            # last set's last game
            last_set = group_sorted['set_no'].max()
            last_game = group_sorted[group_sorted['set_no'] == last_set]['game_no'].max()

            # last point
            last_game_row = \
            group_sorted[(group_sorted['set_no'] == last_set) & (group_sorted['game_no'] == last_game)].iloc[-1]

            # get p1_points_sum and p2_points_sum
            p1_points_sum = last_game_row['p1_points_sum']
            p2_points_sum = last_game_row['p2_points_sum']

            if p1_points_sum > p2_points_sum:
                self.labels[match_id] = torch.FloatTensor([1.0])
            elif p1_points_sum < p2_points_sum:
                self.labels[match_id] = torch.FloatTensor([0.0])
            else:
                if last_game_row['set_victor'] == 1:
                    self.labels[match_id] = torch.FloatTensor([1.0])
                else:
                    self.labels[match_id] = torch.FloatTensor([0.0])

            self.matches[match_id] = group_sorted[['p1_serve', 'p1_double_fault', 'p1_break_pt_missed', 'p1_ace',
                       'p1_serve_speed', 'p1_serve_depth',
                       'p1_break_pt_won', 'p1_return_depth',
                       'p1_distance_run',
                       'p1_unf_err', 'p1_net_pt', 'p1_net_pt_won', 'p1_winner',
                       'p1_points_diff', 'p1_game_diff', 'p1_set_diff',
                       'p2_serve', 'p2_double_fault', 'p2_break_pt_missed', 'p2_ace',
                       'p2_serve_speed', 'p2_serve_depth',
                       'p2_break_pt_won', 'p2_return_depth',
                       'p2_unf_err', 'p2_net_pt', 'p2_net_pt_won', 'p2_winner',
                       'p2_points_diff', 'p2_game_diff', 'p2_set_diff',
                       'p2_distance_run']].values  # [L, 32]

            self.match_ids.append(match_id)

            self.sets[match_id] = group_sorted['set_no'].values

            self.games[match_id] = group_sorted['game_no'].values

        print(f"Total matches stored: {len(self.match_ids)}")

        self.match_to_idx = match_to_idx
        self.set_offset = set_offset
        self.game_offset = game_offset

    def __len__(self):
        return len(self.match_ids)

    def __getitem__(self, idx):
        match_id = self.match_ids[idx]

        sequence = torch.FloatTensor(self.matches[match_id])  # [L, 32]

        label = self.labels[match_id]

        match_id_idx = torch.LongTensor([self.match_to_idx[match_id]])

        set_num = torch.LongTensor(self.sets[match_id])
        game_num = torch.LongTensor(self.games[match_id])

        return {
            'features': sequence,
            'label': label,
            'match_id': match_id_idx,
            'set': set_num,
            'game': game_num
        }

    @property
    def num_matches(self):
        return len(set(self.match_ids))

    @property
    def num_sets(self):
        return int(max([x.max() for x in self.sets.values()]) - self.set_offset + 1) if self.sets else 0

    @property
    def num_games(self):
        return int(max([x.max() for x in self.games.values()]) - self.game_offset + 1) if self.games else 0


# Collate Function for DataLoader
def collate_fn(batch):
    """
    Each batch contains one entire match.
    """
    batch_size = len(batch)
    features = [item['features'] for item in batch]
    labels = [item['label'] for item in batch]  # List of [num_games -1]
    match_ids = torch.cat([item['match_id'] for item in batch], dim=0)  # [B]
    sets = [item['set'] for item in batch]  # List of [L]
    games = [item['game'] for item in batch]  # List of [L]

    # Since labels are of variable lengths, we keep them as a list
    return {
        'features': torch.stack(features, dim=0),  # [B, L, 32]
        'label': labels,  # List of [num_games -1]
        'match_id': match_ids,  # [B]
        'set': torch.stack(sets, dim=0),  # [B, L]
        'game': torch.stack(games, dim=0)  # [B, L]
    }


# Training Function
def train_model(train_loader, model, optimizer, device):
    model.train()
    total_loss = 0
    predictions = []
    labels = []
    ver_losses = []

    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
        optimizer.zero_grad()

        # Move all batch data to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        print(f"Batch {batch_idx + 1}:")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f" - {k} tensor is on device: {v.device}")

        outputs, target_labels, ver_loss = model(
            batch)  # outputs: [B, num_predictions], target_labels: [B, num_predictions], ver_loss: scalar

        # Debugging: Check for NaN in outputs and labels
        if torch.isnan(outputs).any():
            print(f"Warning: NaN detected in outputs at batch {batch_idx + 1}")
        if torch.isnan(target_labels).any():
            print(f"Warning: NaN detected in target_labels at batch {batch_idx + 1}")
        if torch.isnan(ver_loss):
            print(f"Warning: NaN detected in ver_loss at batch {batch_idx + 1}")

        # main loss
        outputs = outputs.to("cuda")
        target_labels = target_labels.to("cuda")
        cla_loss = nn.BCEWithLogitsLoss()(outputs,
                                          target_labels.view(-1, 1))  # make sure [B, num_predictions] and target_labels' shape are [B, 1]

        # all loss
        loss = cla_loss + ver_loss

        # Debugging: Check for NaN in loss
        if torch.isnan(loss):
            print(f"Warning: NaN detected in loss at batch {batch_idx + 1}")

        loss.backward()

        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        print(f"Gradient Norm: {total_norm.item()}")

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

        optimizer.step()

        total_loss += loss.item()

        predictions.extend(outputs.detach().cpu().numpy().flatten())  # List of scalars
        labels.extend(target_labels.detach().cpu().numpy().flatten())  # List of scalars
        ver_losses.append(ver_loss.item())

    if predictions and labels:
        predictions = np.array(predictions)  # [Total Games]
        labels = np.array(labels)  # [Total Games]
    else:
        predictions = np.array([])
        labels = np.array([])

    predictions_binary = (predictions > 0.0).astype(int)

    if len(labels) > 0:
        accuracy = (predictions_binary == labels).mean()
    else:
        accuracy = 0.0

    try:
        auc = roc_auc_score(labels, predictions) if len(np.unique(labels)) > 1 else 0.0
    except ValueError:
        auc = 0.0

    try:
        auprc = average_precision_score(labels, predictions) if len(np.unique(labels)) > 1 else 0.0
    except ValueError:
        auprc = 0.0

    try:
        f1 = f1_score(labels, predictions_binary) if len(np.unique(labels)) > 1 else 0.0
    except ValueError:
        f1 = 0.0

    try:
        recall = recall_score(labels, predictions_binary) if len(np.unique(labels)) > 1 else 0.0
    except ValueError:
        recall = 0.0

    try:
        precision = precision_score(labels, predictions_binary) if len(np.unique(labels)) > 1 else 0.0
    except ValueError:
        precision = 0.0

    return (total_loss / len(train_loader), accuracy, auc, auprc, f1, recall, precision)


# Evaluation Function
def evaluate_model(val_loader, model, device):
    model.eval()
    total_loss = 0
    predictions = []
    labels = []
    ver_losses = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
            # Move all batch data to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            print(f"Validation Batch {batch_idx + 1}:")
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print(f" - {k} tensor is on device: {v.device}")

            outputs, target_labels, ver_loss = model(
                batch)  # outputs: [B, num_predictions], target_labels: [B, num_predictions], ver_loss: scalar

            # Debugging: Check for NaN in outputs and labels
            if torch.isnan(outputs).any():
                print(f"Warning: NaN detected in outputs at validation batch {batch_idx + 1}")
            if torch.isnan(target_labels).any():
                print(f"Warning: NaN detected in target_labels at validation batch {batch_idx + 1}")
            if torch.isnan(ver_loss):
                print(f"Warning: NaN detected in ver_loss at validation batch {batch_idx + 1}")

            # main loss
            outputs = outputs.to("cuda")
            target_labels = target_labels.to("cuda")
            cla_loss = nn.BCEWithLogitsLoss()(outputs, target_labels.view(-1,
                                                                          1))  # make sure [B, num_predictions] and target_labels' shape are [B, 1]

            # all loss
            loss = cla_loss + ver_loss

            # Debugging: Check for NaN in loss
            if torch.isnan(loss):
                print(f"Warning: NaN detected in loss at validation batch {batch_idx + 1}")

            total_loss += loss.item()

            predictions.extend(outputs.cpu().numpy().flatten())  # List of scalars
            labels.extend(target_labels.cpu().numpy().flatten())  # List of scalars
            ver_losses.append(ver_loss.item())

    if predictions and labels:
        predictions = np.array(predictions)  # [Total Games]
        labels = np.array(labels)  # [Total Games]
    else:
        predictions = np.array([])
        labels = np.array([])

    predictions_binary = (predictions > 0.0).astype(int)

    if len(labels) > 0:
        accuracy = (predictions_binary == labels).mean()
    else:
        accuracy = 0.0

    try:
        auc = roc_auc_score(labels, predictions) if len(np.unique(labels)) > 1 else 0.0
    except ValueError:
        auc = 0.0

    try:
        auprc = average_precision_score(labels, predictions) if len(np.unique(labels)) > 1 else 0.0
    except ValueError:
        auprc = 0.0

    try:
        f1 = f1_score(labels, predictions_binary) if len(np.unique(labels)) > 1 else 0.0
    except ValueError:
        f1 = 0.0

    try:
        recall = recall_score(labels, predictions_binary) if len(np.unique(labels)) > 1 else 0.0
    except ValueError:
        recall = 0.0

    try:
        precision = precision_score(labels, predictions_binary) if len(np.unique(labels)) > 1 else 0.0
    except ValueError:
        precision = 0.0

    return (total_loss / len(val_loader), accuracy, auc, auprc, f1, recall, precision)


# Save Model Information
def save_model_info(model_path, train_dataset, fold):
    """Save model-related information for later loading"""
    info = {
        'num_matches': train_dataset.num_matches,
        'num_sets': train_dataset.num_sets,
        'num_games': train_dataset.num_games,
        'match_to_idx': train_dataset.match_to_idx,
        'set_offset': train_dataset.set_offset,
        'game_offset': train_dataset.game_offset
    }
    info_path = f'model_fold_{fold}_info.pth'
    torch.save(info, info_path)
    return info_path


# Load Model Information
def load_model_info(fold):
    """Load model-related information"""
    info_path = f'model_fold_{fold}_info.pth'
    if os.path.exists(info_path):
        return torch.load(info_path)
    return None


# Main Function
def main():
    log_file = open('HydraNet-match.txt', 'w')  # Open the log file for writing
    sys.stdout = log_file  # Redirect stdout to the file
    # Enable anomaly detection for debugging
    torch.autograd.set_detect_anomaly(True)

    # Configuration
    TRAIN_MODE = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    BATCH_SIZE = 1  # Each batch contains one entire match
    EPOCHS = 8
    LR = 0.001  # Lower learning rate for stability
    MODEL_PATH = 'HydraNet-match'
    memory_dim = 16  # Define memory_dim

    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    print("Loading data...")
    with open('./WID.csv', 'rb') as f:
        result = chardet.detect(f.read())

    # Load CSV with detected encoding
    df = pd.read_csv("./WID.csv", encoding=result['encoding'], low_memory=False)
    print(f"Total samples: {len(df)}")

    # Calculate positive and negative samples
    positive_samples = (df['game_victor'] == 1).sum()
    negative_samples = (df['game_victor'] == 2).sum()

    if positive_samples == 0 or negative_samples == 0:
        raise ValueError("Dataset must contain both positive and negative samples.")

    # Standardize input features (Z-score normalization)
    feature_columns = ['p1_serve', 'p1_double_fault', 'p1_break_pt_missed', 'p1_ace',
                         'p1_serve_speed', 'p1_serve_depth',
                         'p1_break_pt_won', 'p1_return_depth',
                         'p1_distance_run',
                         'p1_unf_err', 'p1_net_pt', 'p1_net_pt_won', 'p1_winner',
                         'p1_points_diff', 'p1_game_diff', 'p1_set_diff',
                         'p2_serve', 'p2_double_fault', 'p2_break_pt_missed', 'p2_ace',
                         'p2_serve_speed', 'p2_serve_depth',
                         'p2_break_pt_won', 'p2_return_depth',
                         'p2_unf_err', 'p2_net_pt', 'p2_net_pt_won', 'p2_winner',
                         'p2_points_diff', 'p2_game_diff', 'p2_set_diff',
                         'p2_distance_run']

    for col in feature_columns:
        if df[col].std() == 0:
            df[col] = 0.0
        else:
            df[col] = (df[col] - df[col].mean()) / df[col].std()

    # Check for NaN or Inf after standardization
    if df[feature_columns].isnull().values.any() or np.isinf(df[feature_columns].values).any():
        raise ValueError("NaN or Inf detected in input features after standardization.")

    # Create global mappings based on the entire dataset
    unique_match_ids = df['match_id'].unique()
    match_to_idx = {match: idx for idx, match in enumerate(unique_match_ids)}
    set_offset = df['set_no'].min()
    game_offset = int(df['game_no'].min())  # Ensure game_offset is integer

    max_set_num = df['set_no'].max()
    max_game_num = df['game_no'].max()

    num_matches = len(unique_match_ids)
    num_sets = int(max_set_num - set_offset + 1)
    num_games = int(max_game_num - game_offset + 1)

    print(f"Global Mappings: num_matches={num_matches}, num_sets={num_sets}, num_games={num_games}")

    # Create KFold object ensuring no data leakage by splitting on match_id
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    match_folds = list(kfold.split(unique_match_ids))

    # Training loop
    best_metrics = []  # Keep as list

    for fold, (train_match_idx, val_match_idx) in enumerate(match_folds, 1):
        print(f"\nTraining Fold {fold}")

        # Get train and validation match_ids
        train_match_ids = unique_match_ids[train_match_idx]
        val_match_ids = unique_match_ids[val_match_idx]

        print(f"Number of training matches: {len(train_match_ids)}, Number of validation matches: {len(val_match_ids)}")

        # Split data
        train_data = df[df['match_id'].isin(train_match_ids)]
        val_data = df[df['match_id'].isin(val_match_ids)]

        # Create datasets with global mappings
        train_dataset = TennisDataset(train_data, match_to_idx, set_offset, game_offset)
        val_dataset = TennisDataset(val_data, match_to_idx, set_offset, game_offset)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        print(f"Fold {fold}: Number of training batches: {len(train_loader)}")
        print(f"Fold {fold}: Number of validation batches: {len(val_loader)}")

        if TRAIN_MODE:
            # Initialize model with global mappings and memory_dim
            model = HydraNet(
                num_matches=num_matches,
                num_sets=num_sets,
                num_games=num_games,
                game_offset=game_offset,
                memory_dim=memory_dim
            ).to(DEVICE)

            print("Model parameters are on device:", next(model.parameters()).device)

            # Initialize weights for better stability
            def initialize_weights(m):
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

            model.apply(initialize_weights)

            # Training setup
            optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0001)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

            best_val_loss = float('inf')
            best_val_metrics = None
            not_improved = 0

            for epoch in range(EPOCHS):
                print(f"\nEpoch {epoch + 1}/{EPOCHS}")

                # Train
                train_metrics = train_model(train_loader, model, optimizer, DEVICE)
                train_loss, train_acc, train_auc, train_auprc, train_f1, train_recall, train_precision = train_metrics

                # Validate
                val_metrics = evaluate_model(val_loader, model, DEVICE)
                val_loss, val_acc, val_auc, val_auprc, val_f1, val_recall, val_precision = val_metrics

                print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, AUC: {train_auc:.4f}, "
                      f"AUPRC: {train_auprc:.4f}, F1: {train_f1:.4f}, Recall: {train_recall:.4f}, "
                      f"Precision: {train_precision:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, AUC: {val_auc:.4f}, "
                      f"AUPRC: {val_auprc:.4f}, F1: {val_f1:.4f}, Recall: {val_recall:.4f}, "
                      f"Precision: {val_precision:.4f}")

                # Learning rate scheduling
                scheduler.step(val_loss)

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_metrics = val_metrics
                    model_path = os.path.join(MODEL_PATH, f'model_fold_{fold}.pth')
                    torch.save(model.state_dict(), model_path)
                    info_path = save_model_info(model_path, train_dataset, fold)
                    not_improved = 0
                    print(f"Saved best model for Fold {fold} at Epoch {epoch + 1}")
                else:
                    not_improved += 1
                    print(f"No improvement for Fold {fold} in Epoch {epoch + 1}")

                # Early stopping
                if not_improved >= 10:
                    print("Early stopping!")
                    break

            best_metrics.append(best_val_metrics)
            print(f"Fold {fold} best validation metrics:")
            print(f"Loss: {best_val_metrics[0]:.4f}, Accuracy: {best_val_metrics[1]:.4f}, "
                  f"AUC: {best_val_metrics[2]:.4f}, AUPRC: {best_val_metrics[3]:.4f}, "
                  f"F1: {best_val_metrics[4]:.4f}, Recall: {best_val_metrics[5]:.4f}, "
                  f"Precision: {best_val_metrics[6]:.4f}")

    # After all folds
    if TRAIN_MODE:
        print("\nTraining completed!")
        best_metrics = np.array(best_metrics)
        mean_metrics = best_metrics.mean(axis=0)
        std_metrics = best_metrics.std(axis=0)
        print("\nAverage best validation metrics across folds:")
        print(f"Loss: {mean_metrics[0]:.4f} ± {std_metrics[0]:.4f}")
        print(f"Accuracy: {mean_metrics[1]:.4f} ± {std_metrics[1]:.4f}")
        print(f"AUC: {mean_metrics[2]:.4f} ± {std_metrics[2]:.4f}")
        print(f"AUPRC: {mean_metrics[3]:.4f} ± {std_metrics[3]:.4f}")
        print(f"F1: {mean_metrics[4]:.4f} ± {std_metrics[4]:.4f}")
        print(f"Recall: {mean_metrics[5]:.4f} ± {std_metrics[5]:.4f}")
        print(f"Precision: {mean_metrics[6]:.4f} ± {std_metrics[6]:.4f}")


if __name__ == "__main__":
    main()
