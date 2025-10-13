import os
import time
from typing import Tuple
import numpy as np
from sklearn.metrics import f1_score, average_precision_score
from statistics import mean
import ivtmetrics
from datetime import datetime
import json
import random

import torch
from torch.cuda.amp import GradScaler, autocast  # type: ignore
import torch.nn.functional
from torch.optim import lr_scheduler

from log import logger
from loss import SPLC
from hspnet import HSPNetTrainer
from utils import AverageMeter, add_weight_decay, mAP
import warnings

from config import cfg  # isort:skip

def process_cholec80(true,pred,pred_ema,video_ids,test):

    unique_ids = np.unique(video_ids)
    video_f1s = {}
    video_f1s_ema = {}

    for vid in unique_ids:
        indices = np.where(video_ids==vid)[0]
        y_true_video = true[indices]  
        y_pred_video = pred[indices]
        y_pred_video_ema = pred_ema[indices]

        y_pred_video = np.argmax(y_pred_video,axis=1)
        y_pred_video_ema = np.argmax(y_pred_video_ema,axis=1)
        y_true_video = np.argmax(y_true_video,axis=1)

        f1 = f1_score(y_true_video, y_pred_video, average="macro",labels=np.unique(y_true_video))*100
        f1_ema = f1_score(y_true_video, y_pred_video_ema, average="macro",labels=np.unique(y_true_video))*100

        video_f1s[vid] = f1
        video_f1s_ema[vid] = f1_ema

    f1_score_reg = mean(video_f1s.values())
    f1_score_ema = mean(video_f1s_ema.values())

    if test == True:
        return f1_score_reg, None, video_f1s, None

    return f1_score_reg, f1_score_ema, video_f1s, video_f1s_ema

def resolve_nan(classwise):
        classwise[classwise==-0.0] = np.nan
        return classwise

def process_cholect50(true,pred,pred_ema,video_ids,test):

    unique_vids = np.unique(video_ids)

    ap_i_list = []
    ap_v_list = []
    ap_t_list = []
    ap_iv_list = []
    ap_it_list = []
    ap_ivt_list = []

    ap_i_list_ema = []
    ap_v_list_ema = []
    ap_t_list_ema = []
    ap_iv_list_ema = []
    ap_it_list_ema = []
    ap_ivt_list_ema = []

    for vid in unique_vids:
            indices = np.where(video_ids == vid)[0]
            ivt_labels = true[indices]
            ivt_preds = pred[indices]
            ivt_preds_ema = pred_ema[indices]

            filter = ivtmetrics.Disentangle()

            i_labels = filter.extract(inputs=ivt_labels, component="i")
            v_labels = filter.extract(inputs=ivt_labels, component="v")
            t_labels = filter.extract(inputs=ivt_labels, component="t")
            iv_labels = filter.extract(inputs=ivt_labels, component="iv")
            it_labels = filter.extract(inputs=ivt_labels, component="it")

            i_preds = filter.extract(inputs=ivt_preds, component="i")
            v_preds = filter.extract(inputs=ivt_preds, component="v")
            t_preds = filter.extract(inputs=ivt_preds, component="t")
            iv_preds = filter.extract(inputs=ivt_preds, component="iv")
            it_preds = filter.extract(inputs=ivt_preds, component="it")

            i_preds_ema = filter.extract(inputs=ivt_preds_ema, component="i")
            v_preds_ema = filter.extract(inputs=ivt_preds_ema, component="v")
            t_preds_ema = filter.extract(inputs=ivt_preds_ema, component="t")
            iv_preds_ema = filter.extract(inputs=ivt_preds_ema, component="iv")
            it_preds_ema = filter.extract(inputs=ivt_preds_ema, component="it")

            ap_i = average_precision_score(i_labels,i_preds,average=None)*100
            ap_v = average_precision_score(v_labels,v_preds,average=None)*100
            ap_t = average_precision_score(t_labels,t_preds,average=None)*100
            ap_iv = average_precision_score(iv_labels,iv_preds,average=None)*100
            ap_it = average_precision_score(it_labels,it_preds,average=None)*100
            ap_ivt = average_precision_score(ivt_labels,ivt_preds,average=None)*100

            ap_i_ema = average_precision_score(i_labels,i_preds_ema,average=None)*100
            ap_v_ema = average_precision_score(v_labels,v_preds_ema,average=None)*100
            ap_t_ema = average_precision_score(t_labels,t_preds_ema,average=None)*100
            ap_iv_ema = average_precision_score(iv_labels,iv_preds_ema,average=None)*100
            ap_it_ema = average_precision_score(it_labels,it_preds_ema,average=None)*100
            ap_ivt_ema = average_precision_score(ivt_labels,ivt_preds_ema,average=None)*100

            ap_i = resolve_nan(ap_i)
            ap_v = resolve_nan(ap_v)
            ap_t = resolve_nan(ap_t)
            ap_iv = resolve_nan(ap_iv)
            ap_it = resolve_nan(ap_it)
            ap_ivt = resolve_nan(ap_ivt)

            ap_i_ema = resolve_nan(ap_i_ema)
            ap_v_ema = resolve_nan(ap_v_ema)
            ap_t_ema = resolve_nan(ap_t_ema)
            ap_iv_ema = resolve_nan(ap_iv_ema)
            ap_it_ema = resolve_nan(ap_it_ema)
            ap_ivt_ema = resolve_nan(ap_ivt_ema)

            ap_i_list.append(ap_i.reshape([1,-1]))
            ap_v_list.append(ap_v.reshape([1,-1]))
            ap_t_list.append(ap_t.reshape([1,-1]))
            ap_iv_list.append(ap_iv.reshape([1,-1]))
            ap_it_list.append(ap_it.reshape([1,-1]))
            ap_ivt_list.append(ap_ivt.reshape([1,-1]))

            ap_i_list_ema.append(ap_i_ema.reshape([1,-1]))
            ap_v_list_ema.append(ap_v_ema.reshape([1,-1]))
            ap_t_list_ema.append(ap_t_ema.reshape([1,-1]))
            ap_iv_list_ema.append(ap_iv_ema.reshape([1,-1]))
            ap_it_list_ema.append(ap_it_ema.reshape([1,-1]))
            ap_ivt_list_ema.append(ap_ivt_ema.reshape([1,-1]))
    
    ap_i_list = np.concatenate(ap_i_list,axis=0)
    ap_i_list = np.nanmean(ap_i_list,axis=0)
    map_i = np.nanmean(ap_i_list)
    ap_v_list = np.concatenate(ap_v_list,axis=0)
    ap_v_list = np.nanmean(ap_v_list,axis=0)
    map_v = np.nanmean(ap_v_list)
    ap_t_list = np.concatenate(ap_t_list,axis=0)
    ap_t_list = np.nanmean(ap_t_list,axis=0)
    map_t = np.nanmean(ap_t_list)
    ap_iv_list = np.concatenate(ap_iv_list,axis=0)
    ap_iv_list = np.nanmean(ap_iv_list,axis=0)
    map_iv = np.nanmean(ap_iv_list)
    ap_it_list = np.concatenate(ap_it_list,axis=0)
    ap_it_list = np.nanmean(ap_it_list,axis=0)
    map_it = np.nanmean(ap_it_list)
    ap_ivt_list = np.concatenate(ap_ivt_list,axis=0)
    ap_ivt_list = np.nanmean(ap_ivt_list,axis=0)
    map_ivt = np.nanmean(ap_ivt_list)

    ap_i_list_ema = np.concatenate(ap_i_list_ema,axis=0)
    ap_i_list_ema = np.nanmean(ap_i_list_ema,axis=0)
    map_i_ema = np.nanmean(ap_i_list_ema)
    ap_v_list_ema = np.concatenate(ap_v_list_ema,axis=0)
    ap_v_list_ema = np.nanmean(ap_v_list_ema,axis=0)
    map_v_ema = np.nanmean(ap_v_list_ema)
    ap_t_list_ema = np.concatenate(ap_t_list_ema,axis=0)
    ap_t_list_ema = np.nanmean(ap_t_list_ema,axis=0)
    map_t_ema = np.nanmean(ap_t_list_ema)
    ap_iv_list_ema = np.concatenate(ap_iv_list_ema,axis=0)
    ap_iv_list_ema = np.nanmean(ap_iv_list_ema,axis=0)
    map_iv_ema = np.nanmean(ap_iv_list_ema)
    ap_it_list_ema = np.concatenate(ap_it_list_ema,axis=0)
    ap_it_list_ema = np.nanmean(ap_it_list_ema,axis=0)
    map_it_ema = np.nanmean(ap_it_list_ema)
    ap_ivt_list_ema = np.concatenate(ap_ivt_list_ema,axis=0)
    ap_ivt_list_ema = np.nanmean(ap_ivt_list_ema,axis=0)
    map_ivt_ema = np.nanmean(ap_ivt_list_ema)

    aps = {
        "AP_i" : map_i,
        "AP_v" : map_v,
        "AP_t" : map_t,
        "AP_iv" : map_iv,
        "AP_it" : map_it,
        "AP_ivt" : map_ivt
    }

    aps_ema = {
        "AP_i" : map_i_ema,
        "AP_v" : map_v_ema,
        "AP_t" : map_t_ema,
        "AP_iv" : map_iv_ema,
        "AP_it" : map_it_ema,
        "AP_ivt" : map_ivt_ema
    }

    if test == True:
        return aps, None

    return aps, aps_ema

def process_endo(true,pred,pred_ema,video_ids, test):

    num_classes = true.shape[1]

    per_class_map = {}
    per_class_map_ema = {}

    for label in range(num_classes):
        avg_precision = average_precision_score(
            true[:, label], pred[:, label]
        )
        avg_precision_ema  = average_precision_score(
            true[:, label], pred_ema[:, label]
        )
        per_class_map[f"C{label}"] = avg_precision * 100.0
        per_class_map_ema[f"C{label}"] = avg_precision_ema * 100
    
    mean_ap = mean(per_class_map.values())
    mean_ap_ema = mean(per_class_map_ema.values())

    if test == True:
        return mean_ap, None, per_class_map, None

    return mean_ap, mean_ap_ema, per_class_map, per_class_map_ema


def save_results(a,test,dir):

    if 'cholec80' in cfg.checkpoint:
        dataset = "Cholec80"
    elif 'endo' in cfg.checkpoint:
        dataset = "Endoscapes"
    else:
        dataset = "CholecT50"

    data = {
        "Test" : test,
        "Dataset" : dataset,
        "Results" : a
    }

    folder_name = f"results/{dir}"
    os.makedirs(f"results/{dir}",exist_ok=True)

    current_time = datetime.now()
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
    if test == True:
        file_name = f"{folder_name}/result_{timestamp}_test.json"
    else:
        file_name = f"{folder_name}/result_{timestamp}.json"

    try:
        with open(file_name, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print("Results saved successfully")
    except Exception as e:
        print(f"Error in saving results : {e}")


def calculate_metrics(labels,preds,preds_ema,video_ids,test,dir):

    labels = np.round(labels)

    if 'cholec80' in cfg.checkpoint:
        dataset = 'cholec80'
    elif 'endo' in cfg.checkpoint:
        dataset = 'endoscapes'
    else:
        dataset = 'cholect50'

    print(f"Processing dataset: {dataset}")

    if dataset == "cholec80":
        a = process_cholec80(labels,preds,preds_ema,video_ids,test)
    if dataset == 'endoscapes':
        a = process_endo(labels,preds,preds_ema,video_ids,test)
    if dataset == "cholect50":
        a = process_cholect50(labels,preds,preds_ema,video_ids,test)

    save_results(a,test,dir)

    print("Calculating metrics done")

def save_best(trainer, if_ema_better: bool) -> None:
    if if_ema_better:
        torch.save(trainer.ema.module.state_dict(),
                    os.path.join(cfg.checkpoint, 'model-highest.ckpt'))
    else:
        torch.save(trainer.model.state_dict(),
                    os.path.join(cfg.checkpoint, 'model-highest.ckpt'))
    torch.save(trainer.model.state_dict(),
                os.path.join(cfg.checkpoint, 'model-highest-regular.ckpt'))
    torch.save(trainer.ema.module.state_dict(),
                os.path.join(cfg.checkpoint, 'model-highest-ema.ckpt'))

def validate(trainer, epoch: int,dir) -> Tuple[float, bool]:

    trainer.model.eval()
    logger.info("Start validation...")
    sigmoid = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    all_vids = []
    targets = []
    for _, (input, target, vid) in enumerate(trainer.val_loader):
        target = target
        # compute output
        with torch.no_grad():
            with autocast():
                output_logits = trainer.model(input.cuda())
                output_ema_logits = trainer.ema.module(input.cuda())
                output_regular = sigmoid(output_logits)
                output_ema = sigmoid(output_ema_logits)

                # for mAP calculation
        preds_regular.append(output_regular.cpu().detach().numpy())
        preds_ema.append(output_ema.cpu().detach().numpy())
        targets.append(target.cpu().detach().numpy())
        all_vids.append(vid)

    all_labels = np.concatenate(targets, axis=0)
    all_predictions_reg = np.concatenate(preds_regular, axis=0)
    all_predictions_ema = np.concatenate(preds_ema, axis=0)
    all_vids = np.concatenate([np.array(sublist) for sublist in all_vids])
    calculate_metrics(all_labels,all_predictions_reg,all_predictions_ema,all_vids,False,dir)

    mAP_score_regular = mAP(all_labels,all_predictions_reg)
    mAP_score_ema = mAP(all_labels,all_predictions_ema)
    logger.info("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(
        mAP_score_regular, mAP_score_ema))
    mAP_max = max(mAP_score_regular, mAP_score_ema)
    if mAP_score_ema >= mAP_score_regular:
        if_ema_better_mAP = True
    else:
        if_ema_better_mAP = False

    return mAP_max, if_ema_better_mAP

def train(trainer,dir) -> None:
    # set optimizer
    criterion = SPLC()
    parameters = add_weight_decay(trainer.model, cfg.weight_decay)
    max_lr = [cfg.lr, cfg.lr, cfg.gcn_lr, cfg.gcn_lr]
    optimizer = torch.optim.Adam(
        params=parameters, lr=cfg.lr,
        weight_decay=0)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(trainer.train_loader)
    scheduler = lr_scheduler.OneCycleLR(  # type: ignore
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=steps_per_epoch,
        epochs=cfg.total_epochs,  # type: ignore
        pct_start=0.2)

    highest_mAP = 0
    scaler = GradScaler()
    best_epoch = 0
    trainer.model.train()
    for epoch in range(cfg.epochs):
        for i, (input, target,_) in enumerate(trainer.train_loader):
            target = target.cuda()  # (batch,3,num_classes)
            # target = target.max(dim=1)[0]
            if 'cholec80' in cfg.checkpoint:
                target = torch.argmax(target, dim=1)
            loss = trainer.train(input, target, criterion, epoch, i)

            trainer.model.zero_grad()
            scaler.scale(loss).backward()  # type: ignore
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            trainer.ema.update(trainer.model)
            if i % 100 == 0:
                logger.info('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                    .format(epoch, cfg.epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3), # noqa
                            scheduler.get_last_lr()[0], \
                            loss.item()))

        mAP_score, if_ema_better = validate(trainer, epoch,dir)

        if mAP_score > highest_mAP:
            highest_mAP = mAP_score
            best_epoch = epoch
            save_best(trainer, if_ema_better)
        logger.info(
            'current_mAP = {:.2f}, highest_mAP = {:.2f}, best_epoch={}\n'.
            format(mAP_score, highest_mAP, best_epoch))
        logger.info("Save text embeddings done")
        trainer.model.train()

def test(trainer,dir) -> None:
    # get model-highest.ckpt
    trainer.model.load_state_dict(
        torch.load(f"{cfg.checkpoint}/model-highest.ckpt"), strict=True)
    trainer.model.eval()

    logger.info("Start test...")

    sigmoid = torch.sigmoid

    preds = []
    targets = []
    all_vids = []
    for i, (input, target, vid) in enumerate(trainer.test_loader):
        target = target.cuda()
        # compute output
        with torch.no_grad():
            output = sigmoid(trainer.model(input.cuda()))

        # for mAP calculation
        preds.append(output.cpu().detach().numpy())
        targets.append(target.cpu().detach().numpy())
        all_vids.append(vid)

    all_labels = np.concatenate(targets, axis=0)
    all_predictions_reg = np.concatenate(preds, axis=0)
    all_predictions_ema = np.zeros_like(all_predictions_reg)
    all_vids = np.concatenate([np.array(sublist) for sublist in all_vids])
    calculate_metrics(all_labels,all_predictions_reg,all_predictions_ema,all_vids,True,dir)

    logger.info("Testing done...")


def main():
    warnings.filterwarnings("ignore", category=UserWarning) 
    warnings.filterwarnings("ignore", category=RuntimeWarning) 
    # torch.manual_seed(0)
    # random.seed(0)
    # np.random.seed(0)
    #torch.cuda.set_device(1)
    trainer = HSPNetTrainer()
    dir = cfg.dir
    print(dir)
    train(trainer,dir)
    test(trainer,dir)

if __name__ == '__main__':
    main()