import argparse
import time
import numpy as np
import os
from datetime import datetime
import utils
import warnings


from train_models import get_model, evaluate_calibration, create_temp_scale_model, create_splines_model, create_AAA_model, create_calibration_attack_temp_scale_model, create_compressed_model, create_random_scaling_model
from utils import ECE, get_top_results, calculate_ks_error, one_hot_encode 
from get_data import get_dataset, get_attack_dataset

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from models import ModelPT
import sys

import random

from transformers import ViTFeatureExtractor, ViTForImageClassification
from torch.nn import functional as F

import math 

# Credit to https://github.com/max-andr/square-attack for basic Square Attack implementation


def p_selection(p_init, it, n_iters):
    """ Piece-wise constant schedule for p (the fraction of pixels changed on every iteration). """
    it = int(it / n_iters * 10000)

    if 10 < it <= 50:
        p = p_init / 2
    elif 50 < it <= 200:
        p = p_init / 4
    elif 200 < it <= 500:
        p = p_init / 8
    elif 500 < it <= 1000:
        p = p_init / 16
    elif 1000 < it <= 2000:
        p = p_init / 32
    elif 2000 < it <= 4000:
        p = p_init / 64
    elif 4000 < it <= 6000:
        p = p_init / 128
    elif 6000 < it <= 8000:
        p = p_init / 256
    elif 8000 < it <= 10000:
        p = p_init / 512
    else:
        p = p_init

    return p


def pseudo_gaussian_pert_rectangles(x, y):
    delta = np.zeros([x, y])
    x_c, y_c = x // 2 + 1, y // 2 + 1

    counter2 = [x_c - 1, y_c - 1]
    for counter in range(0, max(x_c, y_c)):
        delta[max(counter2[0], 0):min(counter2[0] + (2 * counter + 1), x),
              max(0, counter2[1]):min(counter2[1] + (2 * counter + 1), y)] += 1.0 / (counter + 1) ** 2

        counter2[0] -= 1
        counter2[1] -= 1

    delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))

    return delta


def meta_pseudo_gaussian_pert(s):
    delta = np.zeros([s, s])
    n_subsquares = 2
    if n_subsquares == 2:
        delta[:s // 2] = pseudo_gaussian_pert_rectangles(s // 2, s)
        delta[s // 2:] = pseudo_gaussian_pert_rectangles(s - s // 2, s) * (-1)
        delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))
        if np.random.rand(1) > 0.5: delta = np.transpose(delta)

    elif n_subsquares == 4:
        delta[:s // 2, :s // 2] = pseudo_gaussian_pert_rectangles(s // 2, s // 2) * np.random.choice([-1, 1])
        delta[s // 2:, :s // 2] = pseudo_gaussian_pert_rectangles(s - s // 2, s // 2) * np.random.choice([-1, 1])
        delta[:s // 2, s // 2:] = pseudo_gaussian_pert_rectangles(s // 2, s - s // 2) * np.random.choice([-1, 1])
        delta[s // 2:, s // 2:] = pseudo_gaussian_pert_rectangles(s - s // 2, s - s // 2) * np.random.choice([-1, 1])
        delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))

    return delta



def PGD_Calibration_Attack(model, x, y, eps, n_iters, attack_type, alpha=2/255, model_type='resnet'): #alpha = 2/255

    mean = np.reshape([0.485, 0.456, 0.406], [1, 3, 1, 1])
    std = np.reshape([0.229, 0.224, 0.225], [1, 3, 1, 1])
    
    mean, std = torch.from_numpy(mean.astype(np.float32)).cuda(), torch.from_numpy(std.astype(np.float32)).cuda()

    ori_images = torch.from_numpy(x).cuda()
    images = torch.from_numpy(x).cuda()
    labels = torch.from_numpy(y).cuda()
    loss = nn.CrossEntropyLoss()
    m = torch.nn.Softmax(dim=-1)



    vit_extractor = None
    if model_type == "vit":
        vit_extractor =  ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k", do_resize=False, do_normalize=False)
    elif model_type == "vitL":
        vit_extractor =  ViTFeatureExtractor.from_pretrained("google/vit-large-patch16-224-in21k", do_resize=False, do_normalize=False)

  
    with torch.no_grad():
        if model_type=="vit":
            temp_output = model(vit_extractor((images - mean) / std, return_tensors="pt")["pixel_values"].cuda(non_blocking=True)).logits
        else:
            temp_output = model((images - mean) / std)

    _, clean_predictions = torch.max(F.softmax(temp_output, dim=1), dim=1)



    for i in range(n_iters) :    
        
        if model_type=="vit":
            bb = [((t - mean) / std).cpu().numpy() for t in images]
            temp_images = vit_extractor(bb, return_tensors="pt")["pixel_values"].cuda(non_blocking=True)
            temp_images.requires_grad = True
            outputs = model(temp_images.squeeze()).logits
        else:
            images.requires_grad = True
            outputs = model((images - mean) / std)
        model.zero_grad()
        cost = loss(outputs, labels).cuda()


        cost.backward()

        if attack_type == 'underconf':
            if model_type=="vit":
                adv_images = images + alpha*temp_images.grad.sign().squeeze()
            else:
                adv_images = images + alpha*images.grad.sign()
        elif attack_type == 'overconf':
            if model_type=="vit":
                adv_images = images - alpha*temp_images.grad.sign().squeeze()
            else:
                adv_images = images - alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)

        if attack_type == 'underconf':
            eta =  F.dropout(eta, p=0.95)
        elif attack_type == 'overconf':
            eta =  F.dropout(eta, p=0.1)

        images_temp = torch.clamp(ori_images + eta, min=0, max=1).detach_()

        with torch.no_grad():

            if model_type=="vit":
                bb = [((t - mean) / std).cpu().numpy() for t in images_temp]
                temp_images = vit_extractor(bb, return_tensors="pt")["pixel_values"].cuda(non_blocking=True)
                outputs2 = model(temp_images.squeeze()).logits
            else:
                outputs2 = model((images_temp - mean) / std)
        softmaxes_ece = F.softmax(outputs2, dim=1)
        _, predictions_ece = torch.max(softmaxes_ece, dim=1)

        label_unchanged = torch.eq(predictions_ece , clean_predictions)

        if model_type=="vit":
            temp_images = temp_images.detach_()

        images = images.detach_()

        images[label_unchanged] = images_temp[label_unchanged]

    return images.cpu().numpy()



def calibration_attack_linf(model, x, y, eps, n_iters, p_init, attack_type, y_test, max_margin=None):
    """ The Linf calibration attack """
    np.random.seed(0)  # important to leave it here as well
    targeted = False
    
    log = utils.Logger('')
    min_val, max_val = 0, 1 if x.max() <= 1 else 255
    c, h, w = x.shape[1:]
    n_features = c*h*w
    n_ex_total = x.shape[0]

    init_delta = np.random.choice([-eps, eps], size=[x.shape[0], c, 1, w])
    
    x_best = np.clip(x + 0*init_delta, min_val, max_val) # no itialization
    
    
    if attack_type=='overconf':
        loss_type='margin_loss_overconf'
    
    elif (attack_type=='underconf') and (max_margin is not None):
        loss_type='margin_loss_overconf'
        
    logits = model.predict(x_best)
    
    initial_logits = torch.from_numpy(logits)
    _, initial_predictions = torch.max(F.softmax(initial_logits, dim=1), dim=1)
    initial_predictions = initial_predictions.numpy()
    
    
    loss_min = model.loss(y, logits, targeted, loss_type=loss_type)
    margin_min = model.loss(y, logits, targeted, loss_type=loss_type)
    n_queries = np.ones(x.shape[0])  # ones because we have already used 1 query
    
    if max_margin is not None:
        goal_margin = np.copy(margin_min)
        for i, margin in np.ndenumerate(margin_min):
            if attack_type=='underconf':
                goal_margin[i] = max_margin - 0.09
            elif attack_type=='overconf':
                goal_margin[i] = max_margin + 0.09
    
    complete_attacks = 0


    time_start = time.time()
    metrics = np.zeros([n_iters, 7])
    for i_iter in range(n_iters - 1):
        if max_margin is not None:
            if attack_type=='underconf':
                idx_to_fool = margin_min > goal_margin
            elif attack_type=='overconf':
                idx_to_fool = margin_min < goal_margin
        
        else:
            if attack_type=='underconf':
                idx_to_fool = margin_min > 0.1
            elif attack_type=='overconf':
                idx_to_fool = margin_min < 0.99
            
        x_curr, x_best_curr, y_curr = x[idx_to_fool], x_best[idx_to_fool], y[idx_to_fool]
        loss_min_curr, margin_min_curr = loss_min[idx_to_fool], margin_min[idx_to_fool]
        deltas = x_best_curr - x_curr

        p = p_selection(p_init, i_iter, n_iters)
        for i_img in range(x_best_curr.shape[0]):
            s = int(round(np.sqrt(p * n_features / c)))
            s = min(max(s, 1), h-1)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
            center_h = np.random.randint(0, h - s)
            center_w = np.random.randint(0, w - s)

            x_curr_window = x_curr[i_img, :, center_h:center_h+s, center_w:center_w+s]
            x_best_curr_window = x_best_curr[i_img, :, center_h:center_h+s, center_w:center_w+s]
            # prevent trying out a delta if it doesn't change x_curr (e.g. an overlapping patch)
            while np.sum(np.abs(np.clip(x_curr_window + deltas[i_img, :, center_h:center_h+s, center_w:center_w+s], min_val, max_val) - x_best_curr_window) < 10**-7) == c*s*s:
                deltas[i_img, :, center_h:center_h+s, center_w:center_w+s] = np.random.choice([-eps, eps], size=[c, 1, 1])


        
        x_new = np.clip(x_curr + deltas, min_val, max_val)

        logits = model.predict(x_new)
        loss = model.loss(y_curr, logits, targeted, loss_type=loss_type)
        margin = model.loss(y_curr, logits, targeted, loss_type=loss_type)
        class_loss = model.loss(y_curr, logits, targeted, loss_type='margin_loss')

        
        if attack_type=='underconf':
            new_logits = torch.from_numpy(logits)
            _, new_predictions = torch.max(F.softmax(new_logits, dim=1), dim=1)
            new_predictions = new_predictions.numpy()
            
            idx_improved = loss < loss_min_curr
            class_unchanged = class_loss > 0.0
           
            idx_improved = np.logical_and(idx_improved, class_unchanged)
            
            same_predictions = initial_predictions[idx_to_fool] == new_predictions
            idx_improved = np.logical_and(idx_improved, same_predictions)
            
        elif attack_type=='overconf':
            new_logits = torch.from_numpy(logits)
            _, new_predictions = torch.max(F.softmax(new_logits, dim=1), dim=1)
            new_predictions = new_predictions.numpy()

            idx_improved = loss > loss_min_curr
            class_unchanged = class_loss > 0.0
            idx_improved = np.logical_and(idx_improved, class_unchanged)
            
            same_predictions = initial_predictions[idx_to_fool] == new_predictions
            idx_improved = np.logical_and(idx_improved, same_predictions)

        loss_min[idx_to_fool] = idx_improved * loss + ~idx_improved * loss_min_curr
        margin_min[idx_to_fool] = idx_improved * margin + ~idx_improved * margin_min_curr
        idx_improved = np.reshape(idx_improved, [-1, *[1]*len(x.shape[:-1])])
        x_best[idx_to_fool] = idx_improved * x_new + ~idx_improved * x_best_curr
        n_queries[idx_to_fool] += 1

        acc = (margin_min > 0.0).sum() / n_ex_total


        if max_margin is not None:
            if attack_type=='underconf':
                complete_attacks = (margin_min > goal_margin).mean()
            elif attack_type=='overconf':
                complete_attacks = (margin_min < goal_margin).mean()        
        
        else: 
            if attack_type=='underconf':
                complete_attacks = (margin_min > 0.1).mean()
            elif attack_type=='overconf':
                complete_attacks = (margin_min < 0.99).mean()
            
        warnings.filterwarnings(action='ignore', message='Mean of empty slice')
        warnings.filterwarnings(action='ignore', message='invalid value encountered in double_scalars')
        
        acc_corr = (margin_min > 0.0).mean()
 
        if max_margin is not None:
            if attack_type=='underconf':
                mean_nq, mean_nq_ae, median_nq_ae = np.mean(n_queries), np.mean(n_queries[margin_min <= goal_margin]), np.median(n_queries[margin_min <= goal_margin])
            elif attack_type=='overconf':
                mean_nq, mean_nq_ae, median_nq_ae = np.mean(n_queries), np.mean(n_queries[margin_min >= goal_margin]), np.median(n_queries[margin_min >= goal_margin])
        else:
            if attack_type=='underconf':
                mean_nq, mean_nq_ae, median_nq_ae = np.mean(n_queries), np.mean(n_queries[margin_min <= 0.1]), np.median(n_queries[margin_min <= 0.1])
            elif attack_type=='overconf':
                mean_nq, mean_nq_ae, median_nq_ae = np.mean(n_queries), np.mean(n_queries[margin_min >= 0.99]), np.median(n_queries[margin_min >= 0.99])
            
        avg_margin_min = np.mean(margin_min)
        time_total = time.time() - time_start    
        metrics[i_iter] = [acc, acc_corr, mean_nq, mean_nq_ae, median_nq_ae, margin_min.mean(), time_total]

        if (i_iter % 100) == 0:
            evaluate_calibration(model, x_best, y_test, adv = "adv_" + attack_type + str(i_iter))
        if complete_attacks == 0:
            break
    print("Number of complete attacks: " + str(complete_attacks))
    final_metrics = [acc, acc_corr, mean_nq_ae, median_nq_ae, avg_margin_min, x.shape[0], eps, time_total]


    if attack_type=='underconf':
        resulting_queries = np.sum(n_queries[margin_min <= 0.1])
        resulting_queries2 = n_queries[margin_min <= 0.1]
    elif attack_type=='overconf':
        resulting_queries = np.sum(n_queries[margin_min >= 0.99])
        resulting_queries2 = n_queries[margin_min >= 0.99]


    return final_metrics, n_queries, x_best, resulting_queries, resulting_queries2
    
    
def calibration_attack_l2(model, x, y, eps, n_iters, p_init, attack_type, y_test, max_margin=None):
    """ The L2 calibration attack """
    np.random.seed(0)
    
    targeted = False
    min_val, max_val = 0, 1 if x.max() <= 1 else 255
    
    c, h, w = x.shape[1:]
    n_features = c * h * w
    n_ex_total = x.shape[0]

    if attack_type=='overconf':
        loss_type='margin_loss_overconf'
    elif (attack_type=='underconf') and (max_margin is not None):
        loss_type='margin_loss_rand_underconf'

    x_best = np.copy(x)

    logits = model.predict(x_best)
    
    initial_logits = torch.from_numpy(logits)
    _, initial_predictions = torch.max(F.softmax(initial_logits, dim=1), dim=1)
    initial_predictions = initial_predictions.numpy()    
    
    loss_min = model.loss(y, logits, targeted, loss_type=loss_type)
    margin_min = model.loss(y, logits, targeted, loss_type=loss_type)
    
    n_queries = np.ones(x.shape[0])  # ones because we have already used 1 query

    if max_margin is not None:
        goal_margin = np.copy(margin_min)
        for i, margin in np.ndenumerate(margin_min):
            if attack_type=='underconf':
                goal_margin[i] = max(0.1, margin - max_margin)
            elif attack_type=='overconf':
                goal_margin[i] = min(0.99, margin + max_margin)


    time_start = time.time()
    s_init = int(np.sqrt(p_init * n_features / c))
    metrics = np.zeros([n_iters, 7])
    for i_iter in range(n_iters):
        if max_margin is not None:
            if attack_type=='underconf':
                idx_to_fool = margin_min > goal_margin
            elif attack_type=='overconf':
                idx_to_fool = margin_min < goal_margin
        
        else:    
            if attack_type=='underconf':
                idx_to_fool = margin_min > 0.1
            elif attack_type=='overconf':
                idx_to_fool = margin_min < 0.99

        
        x_curr, x_best_curr = x[idx_to_fool], x_best[idx_to_fool]
        y_curr, margin_min_curr = y[idx_to_fool], margin_min[idx_to_fool]
        loss_min_curr = loss_min[idx_to_fool]
        delta_curr = x_best_curr - x_curr

        p = p_selection(p_init, i_iter, n_iters)
        s = max(int(round(np.sqrt(p * n_features / c))), 3)

        if s % 2 == 0:
            s += 1
        s2 = s + 0
        ### window_1
        center_h = np.random.randint(0, h - s)
        center_w = np.random.randint(0, w - s)
        new_deltas_mask = np.zeros(x_curr.shape)
        new_deltas_mask[:, :, center_h:center_h + s, center_w:center_w + s] = 1.0

        ### window_2
        center_h_2 = np.random.randint(0, h - s2)
        center_w_2 = np.random.randint(0, w - s2)
        new_deltas_mask_2 = np.zeros(x_curr.shape)
        new_deltas_mask_2[:, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] = 1.0
        norms_window_2 = np.sqrt(
            np.sum(delta_curr[:, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] ** 2, axis=(-2, -1),
                   keepdims=True))
        ### compute total norm available
        curr_norms_window = np.sqrt(
            np.sum(((x_best_curr - x_curr) * new_deltas_mask) ** 2, axis=(2, 3), keepdims=True))
        curr_norms_image = np.sqrt(np.sum((x_best_curr - x_curr) ** 2, axis=(1, 2, 3), keepdims=True))
        mask_2 = np.maximum(new_deltas_mask, new_deltas_mask_2)
        norms_windows = np.sqrt(np.sum((delta_curr * mask_2) ** 2, axis=(2, 3), keepdims=True))
        ### create the updates
        new_deltas = np.ones([x_curr.shape[0], c, s, s])
        new_deltas = new_deltas * meta_pseudo_gaussian_pert(s).reshape([1, 1, s, s])
        new_deltas *= np.random.choice([-1, 1], size=[x_curr.shape[0], c, 1, 1])
        old_deltas = delta_curr[:, :, center_h:center_h + s, center_w:center_w + s] / (1e-10 + curr_norms_window)
        new_deltas += old_deltas
        new_deltas = new_deltas / np.sqrt(np.sum(new_deltas ** 2, axis=(2, 3), keepdims=True)) * (
            np.maximum(eps ** 2 - curr_norms_image ** 2, 0) / c + norms_windows ** 2) ** 0.5
        delta_curr[:, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] = 0.0  # set window_2 to 0
        delta_curr[:, :, center_h:center_h + s, center_w:center_w + s] = new_deltas + 0  # update window_1

        hps_str = 's={}->{}'.format(s_init, s)
        x_new = x_curr + delta_curr / np.sqrt(np.sum(delta_curr ** 2, axis=(1, 2, 3), keepdims=True)) * eps
        x_new = np.clip(x_new, min_val, max_val)
        curr_norms_image = np.sqrt(np.sum((x_new - x_curr) ** 2, axis=(1, 2, 3), keepdims=True))

        logits = model.predict(x_new)
        loss = model.loss(y_curr, logits, targeted, loss_type=loss_type)
        margin = model.loss(y_curr, logits, targeted, loss_type=loss_type)
        
        class_loss = model.loss(y_curr, logits, targeted, loss_type='margin_loss')
        
        if attack_type=='underconf':
            idx_improved = loss < loss_min_curr
            class_unchanged = class_loss > 0.0
            
            idx_improved = np.logical_and(idx_improved, class_unchanged)
            
            new_logits = torch.from_numpy(logits)
            _, new_predictions = torch.max(F.softmax(new_logits, dim=1), dim=1)
            new_predictions = new_predictions.numpy()
            
            same_predictions = initial_predictions[idx_to_fool] == new_predictions
            idx_improved = np.logical_and(idx_improved, same_predictions)
            
        elif attack_type=='overconf':
            idx_improved = loss > loss_min_curr
        
    
        loss_min[idx_to_fool] = idx_improved * loss + ~idx_improved * loss_min_curr
        margin_min[idx_to_fool] = idx_improved * margin + ~idx_improved * margin_min_curr

        idx_improved = np.reshape(idx_improved, [-1, *[1] * len(x.shape[:-1])])
        x_best[idx_to_fool] = idx_improved * x_new + ~idx_improved * x_best_curr
        n_queries[idx_to_fool] += 1

        acc = (margin_min > 0.0).sum() / n_ex_total
        
        warnings.filterwarnings(action='ignore', message='Mean of empty slice')
        warnings.filterwarnings(action='ignore', message='invalid value encountered in double_scalars')

        acc_corr = (margin_min > 0.0).mean()

        if max_margin is not None:
            if attack_type=='underconf':
                complete_attacks = (margin_min > goal_margin).mean()
            elif attack_type=='overconf':
                complete_attacks = (margin_min < goal_margin).mean()        
        
        else:         
            if attack_type=='underconf':
                complete_attacks = (margin_min > 0.1).mean()
            elif attack_type=='overconf':
                complete_attacks = (margin_min < 0.99).mean()
                    
        if max_margin is not None:
            if attack_type=='underconf':
                mean_nq, mean_nq_ae, median_nq_ae = np.mean(n_queries), np.mean(n_queries[margin_min <= goal_margin]), np.median(n_queries[margin_min <= goal_margin])
            elif attack_type=='overconf':
                mean_nq, mean_nq_ae, median_nq_ae = np.mean(n_queries), np.mean(n_queries[margin_min >= goal_margin]), np.median(n_queries[margin_min >= goal_margin])
        else:        
            if attack_type=='underconf':
                mean_nq, mean_nq_ae, median_nq_ae = np.mean(n_queries), np.mean(n_queries[margin_min <= 0.1]), np.median(n_queries[margin_min <= 0.1])
            elif attack_type=='overconf':
                mean_nq, mean_nq_ae, median_nq_ae = np.mean(n_queries), np.mean(n_queries[margin_min >= 0.99]), np.median(n_queries[margin_min >= 0.99])

        time_total = time.time() - time_start
        metrics[i_iter] = [acc, acc_corr, mean_nq, mean_nq_ae, median_nq_ae, margin_min.mean(), time_total]

        if complete_attacks == 0:
            curr_norms_image = np.sqrt(np.sum((x_best - x) ** 2, axis=(1, 2, 3), keepdims=True))
            print('Maximal norm of the perturbations: {:.5f}'.format(np.amax(curr_norms_image)))
            break

    curr_norms_image = np.sqrt(np.sum((x_best - x) ** 2, axis=(1, 2, 3), keepdims=True))
    print('Maximal norm of the perturbations: {:.5f}'.format(np.amax(curr_norms_image)))
    print(complete_attacks)
    final_metrics = [acc, acc_corr, mean_nq_ae, median_nq_ae, margin_min.mean(), x.shape[0], eps, time_total]


    if attack_type=='underconf':
        resulting_queries = np.sum(n_queries[margin_min <= 0.1])
        resulting_queries2 = n_queries[margin_min <= 0.1]
    elif attack_type=='overconf':
        resulting_queries = np.sum(n_queries[margin_min >= 0.99])
        resulting_queries2 = n_queries[margin_min >= 0.99]


    return final_metrics, n_queries, x_best, resulting_queries, resulting_queries2



def maximum_calibration_attack(model, x, y_onehot, eps, n_iters, p_init, attack, y_test):
    
    logits_clean = model.predict(x)
    corr_classified = logits_clean.argmax(1) == y_test
    wrong_classified = logits_clean.argmax(1) != y_test
    
    x_corr, y_onehot_corr, y_test_corr = x[corr_classified], y_onehot[corr_classified], y_test[corr_classified] 
    
    x_misclassified, y_onehot_misclassified, y_test_misclassified = x[wrong_classified], y_onehot[wrong_classified], y_test[wrong_classified] 
    
    metrics_1, n_queries_1, x_adv_1, success_q_1, med1 = attack(model, x_corr, y_onehot_corr, eps, n_iters, p_init, "underconf", y_test_corr)
    
    metrics_2, n_queries_2, x_adv_2, success_q_2, med2 = attack(model, x_misclassified, y_onehot_misclassified, eps, n_iters, p_init, "overconf", y_test_misclassified)
    
    new_x = np.concatenate((x_adv_1, x_adv_2), axis=0)
    
    new_y = np.concatenate((y_test_corr, y_test_misclassified), axis=0)
    
    
    n_queries = np.concatenate((n_queries_1, n_queries_2))
    
    total_queries = n_queries_1.size + n_queries_2.size

    new_mean = (success_q_1 + success_q_2)/total_queries

    new_median = np.median(np.concatenate((med1, med2)))

    return metrics_1, metrics_2, n_queries, new_x, new_y, new_mean, new_median


def random_calibration_attack(model, x, y_onehot, eps, n_iters, p_init, attack, y_test, num_classes):
    
    logits_clean = model.predict(x)
    
    logits_pt = torch.from_numpy(logits_clean)
    softmaxes = F.softmax(logits_pt, dim=1)
    clean_predicted_prob, _ = torch.max(softmaxes, dim=1)

    target_random_confidences = np.random.default_rng().uniform(1.0/num_classes + 0.05,0.99, x.shape[0])

    overconf = clean_predicted_prob.numpy() <= target_random_confidences

    underconf = clean_predicted_prob.numpy() > target_random_confidences

    
    x_1, y_onehot_1, y_test_1 = x[underconf], y_onehot[underconf], y_test[underconf] 
    
    x_2, y_onehot_2, y_test_2 = x[overconf], y_onehot[overconf], y_test[overconf] 
    
    metrics_1, n_queries_1, x_adv_1, success_q_1, med1 = attack(model, x_1, y_onehot_1, eps, n_iters, p_init, "underconf", y_test_1)

    metrics_2, n_queries_2, x_adv_2, success_q_2, med2 = attack(model, x_2, y_onehot_2, eps, n_iters, p_init, "overconf", y_test_2)
    
    
    new_x = np.concatenate((x_adv_1, x_adv_2), axis=0)
    
    new_y = np.concatenate((y_test_1, y_test_2), axis=0)
      
    n_queries = np.concatenate((n_queries_1, n_queries_2), axis=0)
    
    total_queries = n_queries_1.size + n_queries_2.size

    new_mean = (success_q_1 + success_q_2)/total_queries

    new_median = np.median(np.concatenate((med1, med2)))


    return metrics_1, metrics_2, n_queries, new_x, new_y, new_mean, new_median




def PGD_maximum_calibration_attack(model, old_model, x, y_onehot, eps, n_iters, y_test, dataset='caltech', model_type="resnet"):
    
    logits_clean = model.predict(x)
    corr_classified = logits_clean.argmax(1) == y_test
    wrong_classified = logits_clean.argmax(1) != y_test
    
    x_corr, y_onehot_corr, y_test_corr = x[corr_classified], y_onehot[corr_classified], y_test[corr_classified] 
    
    x_misclassified, y_onehot_misclassified, y_test_misclassified = x[wrong_classified], y_onehot[wrong_classified], y_test[wrong_classified] 


    if model_type=="vitL":
        batch_size = 16
    else:
        batch_size = 64

    n_batches = math.ceil(x_corr.shape[0] / batch_size)

    x_adv_list = []

    for i in range(n_batches):
            
        x_batch = x_corr[i*batch_size:(i+1)*batch_size]
        x_batch_np = np.array(x_batch)

        y_batch = y_test_corr[i*batch_size:(i+1)*batch_size]
        y_batch_np = np.array(y_batch)

        x_adv_i = PGD_Calibration_Attack(old_model, x_batch_np, y_batch_np, eps, n_iters ,"underconf", alpha=5/255, model_type=model_type)
        x_adv_list.append(x_adv_i)

    x_adv_1 = np.vstack(x_adv_list)


    n_batches = math.ceil(x_misclassified.shape[0] / batch_size)

    x_adv_list = []

    for i in range(n_batches):
            
        x_batch = x_misclassified[i*batch_size:(i+1)*batch_size]
        x_batch_np = np.array(x_batch)

        y_batch = y_test_misclassified[i*batch_size:(i+1)*batch_size]
        y_batch_np = np.array(y_batch)

        x_adv_i = PGD_Calibration_Attack(old_model, x_batch_np, y_batch_np, eps, n_iters ,"overconf", alpha=5/255, model_type=model_type)
        x_adv_list.append(x_adv_i)

    x_adv_2 = np.vstack(x_adv_list)

    
    new_x = np.concatenate((x_adv_1, x_adv_2), axis=0)
    
    new_y = np.concatenate((y_test_corr, y_test_misclassified), axis=0)



    return new_x, new_y



def random_calibration_attack_pgd_version(model, old_model, x, y_onehot, eps, n_iters, y_test, num_classes, model_type, dataset='gtsrb'):
    
    logits_clean = model.predict(x)
    
    logits_pt = torch.from_numpy(logits_clean)
    softmaxes = F.softmax(logits_pt, dim=1)
    clean_predicted_prob, _ = torch.max(softmaxes, dim=1)

    target_random_confidences = np.random.default_rng().uniform(1.0/num_classes + 0.05,0.99, x.shape[0])

    overconf = clean_predicted_prob.numpy() <= target_random_confidences

    underconf = clean_predicted_prob.numpy() > target_random_confidences
    
    x_1, y_onehot_1, y_test_1 = x[underconf], y_onehot[underconf], y_test[underconf] 
    
    x_2, y_onehot_2, y_test_2 = x[overconf], y_onehot[overconf], y_test[overconf] 
    

    if model_type=="vitL":
        batch_size = 16
    else:
        batch_size = 64

    n_batches = math.ceil(x_1.shape[0] / batch_size)

    x_adv_list = []

    for i in range(n_batches):
            
        x_batch = x_1[i*batch_size:(i+1)*batch_size]
        x_batch_np = np.array(x_batch)

        y_batch = y_test_1[i*batch_size:(i+1)*batch_size]
        y_batch_np = np.array(y_batch)

        x_adv_i = PGD_Calibration_Attack(old_model, x_batch_np, y_batch_np, eps, n_iters ,"underconf", alpha=5/255, model_type=model_type)
        x_adv_list.append(x_adv_i)

    x_adv_1 = np.vstack(x_adv_list)    
    
    
    n_batches = math.ceil(x_2.shape[0] / batch_size)

    x_adv_list = []

    for i in range(n_batches):
            
        x_batch = x_2[i*batch_size:(i+1)*batch_size]
        x_batch_np = np.array(x_batch)

        y_batch = y_test_2[i*batch_size:(i+1)*batch_size]
        y_batch_np = np.array(y_batch)

        x_adv_i = PGD_Calibration_Attack(old_model, x_batch_np, y_batch_np, eps, n_iters ,"overconf", alpha=5/255, model_type=model_type)
        x_adv_list.append(x_adv_i)

    x_adv_2 = np.vstack(x_adv_list)
    
    new_x = np.concatenate((x_adv_1, x_adv_2), axis=0)
    
    new_y = np.concatenate((y_test_1, y_test_2), axis=0)
      
    return new_x, new_y




    
import os
import torch
from torch import nn

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Define hyperparameters.')
    parser.add_argument('--model', type=str, default='pt_resnet', help='Use pytorch version of the model? Tensorflow not implemented.')
    parser.add_argument('--attack', type=str, default='square_linf', choices=['square_linf', 'square_l2','pgd'], help='Attack.')
    parser.add_argument('--n_ex', type=int, default=500, help='Number of test ex to test on.') #10
    parser.add_argument('--p', type=float, default=0.05,
                        help='Probability of changing a coordinate. Note: check the paper for the best values. '
                             'Linf standard: 0.05, L2 standard: 0.1. But robust models require higher p.')
    parser.add_argument('--eps', type=float, default=0.05, help='Radius of the Lp ball.') #0.05 12.75
    parser.add_argument('--n_iter', type=int, default=200) #originally 10000
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for model.')
    parser.add_argument('--model_type', type=str, default='resnet', help='Model name.') # resnet or vit
    parser.add_argument('--dataset', type=str, default='caltech', help='dataset name.')
    parser.add_argument('--version', type=str, default='a', help='which of three models to test on')
    parser.add_argument('--calibration', type=str, default='underconf', help='which type of calibration attack to attempt.') #overconf, underconf, maximum, random  

    args = parser.parse_args()
    
    
    batch_size = args.batch_size

    if not (args.model_type=='resnet' or args.model_type=='resnet18' or args.model_type=='resnet152' or args.model_type=='mlp_mixer' or args.model_type=='vit' or args.model_type == "vitL" or args.model_type == "SwinTiny" or args.model_type == "SwinBase" ):
        print("ERROR: improper model type")
        sys.exit()

    if args.dataset == "caltech":
        n_cls = 101
    elif args.dataset == "cifar100":
        n_cls = 100
    elif args.dataset == "gtsrb":
        n_cls = 43
       

    model = get_model(nclasses=n_cls, model_type=args.model_type) 
    saved_model_name = "./models/trained_models/" + args.dataset + "_" + args.model_type + "_" + args.version +"_imbalanced_" + str(int(1/args.ratio)) + ".pt"
    print(saved_model_name)
    model.load_state_dict(torch.load(saved_model_name))                    

    log = utils.Logger('')
    
    old_model = model
    x_test, y_test = get_attack_dataset(args.dataset,args.n_ex)
    model = ModelPT(model, batch_size, model_type=args.model_type)

    logits_clean = model.predict(x_test)

    pre_accuracy, pre_f1, pre_avg_predicted_conf, pre_ece, pre_brs, pre_ks, logits, pre_conf_list, pre_acc_list = evaluate_calibration(model, x_test, y_test)


    if args.attack ==  'pgd':
        square_attack = PGD_Calibration_Attack
    else:        
        square_attack = calibration_attack_linf if args.attack == 'square_linf' else calibration_attack_l2
    
    corr_classified = logits_clean.argmax(1) == y_test
    
    y_target = y_test
    
    y_target_onehot = utils.dense_to_onehot(y_target, n_cls=n_cls)

    print("The current model is: " + args.model_type + " and its version is: " + args.version)
    print("The current dataset is: " + args.dataset)
    print("The current adversarial attack is: " + args.attack)
    print("The current calibration attack is: " + args.calibration)
    print("The current epsilon is: " + str(args.eps) + " and the number of iterations is: " + str(args.n_iter))


    if args.attack !=  'pgd':
        if args.calibration == "maximum":
            metrics_1, metrics_2, n_queries, x_adv, y_target, avq_q, med_q = maximum_calibration_attack(model, x_test, y_target_onehot, args.eps, args.n_iter, args.p, square_attack, y_target)
            print(metrics_1)
            print(metrics_2)
            print("true average queries: " + str(avq_q))
            print("true median queries: " + str(med_q))

        elif args.calibration == "random":
            metrics_1, metrics_2, n_queries, x_adv, y_target, avq_q, med_q = random_calibration_attack(model, x_test, y_target_onehot, args.eps, args.n_iter, args.p, square_attack, y_target, n_cls)
            print(metrics_1)
            print(metrics_2)
            print("true average queries: " + str(avq_q))
            print("true median queries: " + str(med_q))

        else:
            metrics, n_queries, x_adv,_, _ = square_attack(model, x_test, y_target_onehot, args.eps, args.n_iter, args.p, args.calibration, y_target)
            print(metrics)
            print("true average queries: " + str(metrics[2]))
            print("true median queries: " + str(metrics[3])) 


        post_accuracy, post_f1, post_avg_predicted_conf, post_ece, post_brs, post_ks, attacked_logits, post_conf_list, post_acc_list = evaluate_calibration(model, x_adv, y_target)
    
        print("pre attack accuracy: " + str(pre_accuracy))
        print("pre attack avg. confidence: " + str(pre_avg_predicted_conf))
        print("pre attack ece: " + str(pre_ece))
        print("pre attack ks error: " + str(pre_ks))

        print("post attack accuracy: " + str(post_accuracy))
        print("post attack avg. confidence: " + str(post_avg_predicted_conf))
        print("post attack ece: " + str(post_ece))
        print("post attack ks error: " + str(post_ks))    
        print(post_conf_list)
        print(post_acc_list)    

    else:
        if args.attack ==  'pgd':

            if args.calibration != "maximum" and args.calibration != "random": 

                if args.model_type=="vitL":
                    args.batch_size = 16
                else:
                    args.batch_size = 64
                
                n_batches = math.ceil(x_test.shape[0] / args.batch_size)
                x_adv_list = []

                for i in range(n_batches):
            
                    x_batch = x_test[i*args.batch_size:(i+1)*args.batch_size]
                    x_batch_np = np.array(x_batch)

                    y_batch = y_test[i*args.batch_size:(i+1)*args.batch_size]
                    y_batch_np = np.array(y_batch)

                    x_adv_i = square_attack(old_model, x_batch_np, y_batch_np, args.eps, args.n_iter, args.calibration, alpha=5/255, model_type=args.model_type)
                    x_adv_list.append(x_adv_i)

                x_adv = np.vstack(x_adv_list)

                post_accuracy, post_f1, post_avg_predicted_conf, post_ece, post_brs, post_ks, attacked_logits, post_conf_list, post_acc_list = evaluate_calibration(model, x_adv, y_target)
        
                print("pre attack accuracy: " + str(pre_accuracy))
                print("pre attack avg. confidence: " + str(pre_avg_predicted_conf))
                print("pre attack ece: " + str(pre_ece))
                print("pre attack ks error: " + str(pre_ks))
        
                print("post attack accuracy: " + str(post_accuracy))
                print("post attack avg. confidence: " + str(post_avg_predicted_conf))
                print("post attack ece: " + str(post_ece))
                print("post attack ks error: " + str(post_ks))    

            elif args.calibration == "maximum":
                x_adv, y_target = PGD_maximum_calibration_attack(model, old_model, x_test, y_target_onehot, args.eps, args.n_iter,  y_target, args.model_type, dataset=args.dataset)

                post_accuracy, post_f1, post_avg_predicted_conf, post_ece, post_brs, post_ks, attacked_logits, post_conf_list, post_acc_list = evaluate_calibration(model, x_adv, y_target)
        
                print("pre attack accuracy: " + str(pre_accuracy))
                print("pre attack avg. confidence: " + str(pre_avg_predicted_conf))
                print("pre attack ece: " + str(pre_ece))
                print("pre attack ks error: " + str(pre_ks))
        
                print("post attack accuracy: " + str(post_accuracy))
                print("post attack avg. confidence: " + str(post_avg_predicted_conf))
                print("post attack ece: " + str(post_ece))
                print("post attack ks error: " + str(post_ks))    

            elif args.calibration == "random":
                x_adv, y_target = random_calibration_attack_pgd_version(model, old_model, x_test, y_target_onehot, args.eps, args.n_iter,  y_target, n_cls, args.model_type, dataset=args.dataset)

                post_accuracy, post_f1, post_avg_predicted_conf, post_ece, post_brs, post_ks, attacked_logits, post_conf_list, post_acc_list = evaluate_calibration(model, x_adv, y_target)
        
                print("pre attack accuracy: " + str(pre_accuracy))
                print("pre attack avg. confidence: " + str(pre_avg_predicted_conf))
                print("pre attack ece: " + str(pre_ece))
                print("pre attack ks error: " + str(pre_ks))
        
                print("post attack accuracy: " + str(post_accuracy))
                print("post attack avg. confidence: " + str(post_avg_predicted_conf))
                print("post attack ece: " + str(post_ece))
                print("post attack ks error: " + str(post_ks))    


    
       
       
       
       
        

