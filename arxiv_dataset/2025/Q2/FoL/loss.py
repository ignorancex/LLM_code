# -*- coding: UTF-8 -*-
import os
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


loss_fn = losses.MultiSimilarityLoss(alpha=1.0, beta=50, base=0.0, distance=DotProductSimilarity())
miner = miners.MultiSimilarityMiner(epsilon=0.1, distance=CosineSimilarity())
BatchHardMiner = miners.BatchHardMiner()


def loss_function(descriptors, labels):
    # we mine the pairs/triplets if there is an online mining strategy
    if miner is not None:
        miner_outputs = miner(descriptors, labels)
        loss = loss_fn(descriptors, labels, miner_outputs)
        # calculate the % of trivial pairs/triplets 
        # which do not contribute in the loss value
        nb_samples = descriptors.shape[0]
        nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
        batch_acc = 1.0 - (nb_mined/nb_samples)

    else: # no online mining
        loss = loss_fn(descriptors, labels)
        batch_acc = 0.0
    return loss


def index_to_coords(index, grid_size=23):
    row = index // grid_size
    col = index % grid_size
    return row, col


def get_patch_center_mapping(index, l,  img_size=322, patch_size=14, feature_map_size=89):
    patch_i, patch_j = index_to_coords(index)
    scale = img_size / feature_map_size

    # Calculate the center of the patch
    center_x = patch_i * patch_size + patch_size // 2
    center_y = patch_j * patch_size + patch_size // 2

    new_i = int(round(center_x / scale))
    new_j = int(round(center_y / scale))

    return ( l[new_i,new_j,:] + l[new_i-1,new_j-1,:] + l[new_i-1,new_j,:] + l[new_i-1,new_j+1,:] +\
        l[new_i,new_j-1,:] + l[new_i,new_j+1,:] + l[new_i+1,new_j-1,:] + l[new_i+1,new_j,:] + l[new_i+1,new_j+1,:] ) / 9



def loss_function_weak_supervision(descriptors, weak_supervision_info, labels, masks, image_names, max_triplets_per_image=5):
    """
    优化后的弱监督损失函数
    
    参数:
    descriptors: 图像描述符张量
    weak_supervision_info: 包含三个元素的元组 (f, p, l)
        f: 特征张量 [bs, C, H*W]
        p: 类别概率张量 [bs, num_classes, H*W]
        l: 位置映射信息
    labels: 图像标签 [bs]
    masks: 分割掩码 [bs, 1, H, W]
    image_names: 图像名称列表 [bs]
    max_triplets_per_image: 每张图像最大三元组数量
    
    返回:
    计算得到的弱监督损失值
    """
    miner_outputs = BatchHardMiner(descriptors, labels)
    bs = len(image_names)
    
    all_f = weak_supervision_info[0]  # [bs, C, H*W]
    all_p = weak_supervision_info[1]  # [bs, num_classes, H*W]
    all_l = weak_supervision_info[2]  # 位置映射信息

    label_to_indices = defaultdict(list)
    for i, label in enumerate(labels):
        label_to_indices[int(label)].append(i)
    
    total_loss = 0.0
    total_triplets = 0
    
    for i in range(bs):
        current_label = int(labels[i])
        group_indices = label_to_indices[current_label]
        
        if len(group_indices) < 2:
            continue

        cur_idx = i % len(group_indices)
        next_idx = (cur_idx + 1) % len(group_indices)
        j = group_indices[next_idx]

        f1 = all_f[i]
        p1 = all_p[i]
        l1 = all_l[i]

        f2 = all_f[j]
        p2 = all_p[j]
        l2 = all_l[j]

        triplet_loc = get_weak_supervision_triplets(
            f1, f2, p1, p2, masks[i].squeeze(0), max_triplets_per_image
        )
        
        if not triplet_loc:
            continue

        a_ids, p_ids, simps = zip(*triplet_loc)
        n_triplets = len(a_ids)

        anchors = get_patch_center_mapping_batch(a_ids, l1)
        positives = get_patch_center_mapping_batch(p_ids, l2)
        
        losses = triplet_cosine_loss_vectorized(anchors, positives)
        batch_loss = torch.sum(losses * torch.tensor(simps, device=losses.device)) * 1e-2
        
        total_loss += batch_loss
        total_triplets += n_triplets
    
    print(f"Found {total_triplets} good triplets.")
    
    if total_triplets > 0:
        return total_loss / total_triplets
    return torch.tensor(0.0, device=descriptors.device)


def get_weak_supervision_triplets(f1, f2, p1, p2, mask1, max_triplets=4):
    """
    向量化版本的弱监督三元组选择
    
    参数:
    f1, f2: 特征张量 [C, H*W]
    p1, p2: 类别概率张量 [num_classes, H*W]
    mask1: 分割掩码 [H*W]
    max_triplets: 最大三元组数量
    
    返回:
    三元组列表 [[a_id, p_id, simp], ...]
    """
    valid_mask = mask1 > 0.1
    valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
    
    if valid_indices.numel() < 2:
        return []

    valid_indices = valid_indices[1:]

    assigned_classes = torch.argmax(p1[:, valid_indices], dim=0)  # [num_valid]

    candidate_mask = p2[assigned_classes] > 0.1  # [num_valid, H*W]

    anchors = f1[:, valid_indices].t()  # [num_valid, C]

    sim_matrix = F.cosine_similarity(
        anchors.unsqueeze(1), 
        f2.t().unsqueeze(0), 
        dim=-1
    )  # [num_valid, H*W]
    
    sim_matrix_masked = sim_matrix.clone()
    sim_matrix_masked[~candidate_mask] = -10.0
    
    top2_vals, top2_idxs = torch.topk(sim_matrix_masked, k=2, dim=1)

    condition1 = top2_vals[:, 0] > 0.8
    condition2 = top2_vals[:, 0] / top2_vals[:, 1] > 2
    valid_mask = condition1 & condition2

    valid_indices_filtered = valid_indices[valid_mask]
    p_ids = top2_idxs[valid_mask, 0]
    sim_vals = top2_vals[valid_mask, 0]
    
    triplets = [
        [a_id.item(), p_id.item(), sim_val.item()]
        for a_id, p_id, sim_val in zip(valid_indices_filtered, p_ids, sim_vals)
    ]
    return triplets[:max_triplets]

def get_patch_center_mapping_batch(indices, mapping_info):
    if isinstance(mapping_info, torch.Tensor) and mapping_info.dim() == 2:
        return mapping_info[indices]

    return torch.stack([get_patch_center_mapping(i, mapping_info) for i in indices])


def triplet_cosine_loss_vectorized(anchors, positives):
    return 1 - F.cosine_similarity(anchors, positives, dim=1)


def match_batch_tensor(fm1, fm2, trainflag, grid_size, T2=0.7):
    '''
    fm1: (l,D) 529,768
    fm2: (N,l,D) 100,529,768
    mask1: (l)
    mask2: (N,l)
    '''
    M = torch.matmul(fm2, fm1.T)  # (N,l,l) 100,529,529

    max1 = torch.argmax(M, dim=1)  # (N,l) 100,529
    max2 = torch.argmax(M, dim=2)  # (N,l) 100,529
    m = max2[torch.arange(M.shape[0]).reshape((-1, 1)), max1]  # (N, l) 100,529
    valid = torch.arange(M.shape[-1]).repeat((M.shape[0], 1)).cuda() == m  # (N, l) bool 100,529

    scores = torch.zeros(fm2.shape[0]).cuda()  # 100,

    # kps = get_keypoints(grid_size) # 529,2
    for i in range(fm2.shape[0]):  # fm2.shape[0]:2
        idx1 = torch.nonzero(valid[i, :]).squeeze()  # 832...会变
        idx2 = max1[i, :][idx1]  # 832
        assert idx1.shape == idx2.shape

        if len(idx1.shape) > 0:
            # Calculate cosine similarity and apply threshold
            cos_similarity = torch.sum(fm1[idx1] * fm2[i][idx2], dim=1)
            valid_pairs = cos_similarity > T2
            #print(len(cos_similarity))
            #print(valid_pairs.sum())
            idx1 = idx1[valid_pairs]
            idx2 = idx2[valid_pairs]

        if trainflag:
            if len(idx1.shape) > 0:
                similarity = torch.mean(torch.sum(fm1[idx1] * fm2[i][idx2], dim=1), dim=0)
            else:
                print("No mutual nearest neighbors!")
                similarity = torch.mean(torch.sum(fm1 * fm2[i], dim=1), dim=0)
            return similarity

        else:
            if len(idx1.shape) < 1:
                scores[i] = 0
            else:
                scores[i] = len(idx1)  # len(idx1)=832
    return scores  # 100,

# code from https://github.com/Lu-Feng/SelaVPR
def local_sim(features_1, features_2, trainflag=False):
    # B, H, W, C = features_2.shape
    B, Num, C = features_2.shape
    if trainflag:
        queries = features_1
        preds = features_2
        # queries,preds = queries.view(B, H*W, C),preds.view(B, H*W, C)
        similarity = torch.zeros(B).cuda()
        for i in range(B):
            query,pred = queries[i],preds[i].unsqueeze(0)
            similarity[i] = match_batch_tensor(query, pred, trainflag, grid_size=(61,61))
        return similarity
    else:
        query = features_1 # 61,61,128
        preds = features_2 # 100,61,61,128
        scores = match_batch_tensor(query, preds,trainflag, grid_size=(61,61))
        return scores


# code from https://github.com/Lu-Feng/SelaVPR
def calculate(local_features, triplets):
    total_loss = 0.0

    for anchor_idx, positive_idx, negative_idx in zip(triplets[0], triplets[1], triplets[2]):
        anchor = local_features[anchor_idx].unsqueeze(0)
        positive = local_features[positive_idx].unsqueeze(0)
        negative = local_features[negative_idx].unsqueeze(0)

        simP = local_sim(anchor, positive, trainflag=True)
        simN = local_sim(anchor, negative, trainflag=True)
        loss = torch.sum(torch.clamp(-simP + simN + 0., min=0.))
        total_loss += loss

    return total_loss

def loss_function_local(descriptors, local_f, labels):
    miner_outputs = BatchHardMiner(descriptors, labels)
    loss_local = calculate(local_f, miner_outputs)
    return loss_local


def weighted_cosine_similarity_loss(fg1, fg2, bg1, bg2, fg_weight=0.7):
    fg_similarity = F.cosine_similarity(fg1.unsqueeze(0), fg2.unsqueeze(0))
    bg_similarity = F.cosine_similarity(bg1.unsqueeze(0), bg2.unsqueeze(0))
    loss = fg_weight * (1 - fg_similarity) + (1 - fg_weight) * (1 - bg_similarity)
    return loss.mean()


def compute_foreground_background_features(local_feature, mask):
    foreground = local_feature * mask.unsqueeze(-1)
    background = local_feature * (1 - mask).unsqueeze(-1)
    summed_foreground = torch.sum(foreground, dim=(0, 1))
    summed_background = torch.sum(background, dim=(0, 1))
    return summed_foreground, summed_background


def loss_sep(seps, labels):
    bs = len(labels)
    total_loss = 0.0

    from collections import defaultdict
    import random
    grouped_samples = defaultdict(list)
    for i in range(bs):
        grouped_samples[int(labels[i])].append([seps[0][i], seps[1][i]])

    for i in range(bs):
        cur_id = i % 4
        current_label = labels[i]
        possible_samples = grouped_samples[int(current_label)].copy()
        current_sample = possible_samples[cur_id]
        del possible_samples[cur_id]

        selected_sample = random.choice(possible_samples)
        local_feature_current = current_sample[0]  # Tensor of shape (89, 89, 128)
        mask_current = current_sample[1]  # Tensor of shape (89, 89)
        local_feature_selected = selected_sample[0]  # Tensor of shape (89, 89, 128)
        mask_selected = selected_sample[1]  # Tensor of shape (89, 89)
        fg_current, bg_current = compute_foreground_background_features(local_feature_current, mask_current)
        fg_selected, bg_selected = compute_foreground_background_features(local_feature_selected, mask_selected)
        fg_current = F.normalize(fg_current, p=2, dim=-1)
        bg_current = F.normalize(bg_current, p=2, dim=-1)
        fg_selected = F.normalize(fg_selected, p=2, dim=-1)
        bg_selected = F.normalize(bg_selected, p=2, dim=-1)

        loss = weighted_cosine_similarity_loss(fg_current, fg_selected, bg_current, bg_selected)
        total_loss += loss / bs

    return total_loss