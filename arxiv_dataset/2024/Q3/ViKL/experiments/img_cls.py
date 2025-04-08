# -*- coding: utf-8 -*-
import os
import argparse
import time
import datetime
import yaml
import tqdm
import numpy as np
from PIL import Image
from pathlib import Path
from dataset import create_dataset, create_sampler, create_loader
from sklearn.metrics import roc_curve, accuracy_score, f1_score, roc_auc_score, auc, confusion_matrix
import torch
from torch.utils.data import ConcatDataset, random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.resnet import resnet50, resnet18

from torchvision import transforms
import torchvision.transforms as T

# from HMBM.dataset.HMBM_dataset import HMBMDownstreamDataset
# from ALBEF.dataset.hmbm_dataset import HMBMDownstreamDataset
# from ALBEF import utils
import logging


class MetricCal:
    def __init__(self):
        pass

    def __call__(self, y, pred):
        fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
        auc_score = auc(fpr, tpr)

        r_pred = np.round(pred)
        tn, fp, fn, tp = confusion_matrix(y, r_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        acc = accuracy_score(y, r_pred)
        f1 = f1_score(y, r_pred)

        ret_dict = {'spe': specificity, 'sen': sensitivity, 'auc': auc_score, 'f1': f1, 'acc': acc}
        return ret_dict


class Trainer:
    def __init__(self, tag, main_metric, metric_direction, patient, epoch_num,
                 model_save_path, tb_save_path, best_init, **kwargs):
        self.config = kwargs
        self.tag = tag
        self.metric_direction = metric_direction
        if self.metric_direction == 'high':
            self.best_metric = best_init
            self.is_better = lambda new, old: new > old

        elif self.metric_direction == 'low':
            self.best_metric = best_init
            self.is_better = lambda new, old: new < old
        self.main_metric = main_metric
        self.model_save_path = os.path.join(model_save_path, tag)
        self.tb_save_path = tb_save_path
        os.makedirs(self.model_save_path, exist_ok=True)
        self.patient = patient
        self.patient_count = 0
        self.epoch_num = epoch_num
        self.tb_writer = SummaryWriter(log_dir=os.path.join(self.tb_save_path, f"{tag}"))
        self.val_tb_writer = SummaryWriter(log_dir=os.path.join(self.tb_save_path, f"{tag}_val"))
        self.metric_cal = MetricCal()
        # self.batch_size = batch_size
        # self.image_res = image_res
        self.logger = None
        self.init_logger()
        self.log_config()

    def train(self, model, train_loader, valid_loader, optimizer, criterion, scheduler):

        for epoch in range(self.epoch_num):
            all_labels, all_preds = None, None
            all_losses = []
            model.train()
            for d in tqdm.tqdm(train_loader):
                img = d['view1'].float().cuda()
                label = d['label'].cuda()
                output = model(img)
                loss = criterion(output, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                # for metrics
                all_losses.append(loss.item())
                preds = torch.softmax(output, dim=1).cpu().detach().numpy()[:, 1]
                if all_labels is None and all_preds is None:
                    all_preds = preds
                    all_labels = label.cpu().detach().numpy()
                else:
                    all_preds = np.concatenate([all_preds, preds], axis=0)
                    all_labels = np.concatenate([all_labels, label.cpu().detach().numpy()], axis=0)
            if epoch == 0:  # one time, for image preview
                self.tb_writer.add_images("img", img)

            met_dict = self.metric_cal(all_labels, all_preds)
            if epoch % 10 == 0:  # fixme: fixed for now
                self.tb_writer.add_scalar("loss", np.average(all_losses), epoch)
                self.tb_writer.add_scalar("lr", optimizer.param_groups[-1]['lr'], epoch)
                for i in met_dict:
                    self.tb_writer.add_scalar(i, met_dict[i], epoch)

            self.logger.info(f"epoch: {epoch}")
            #
            # torch.save(model.state_dict(), os.path.join(self.model_save_path, f"latest.pt")) # img_cls不需要latest
            met_ret = self.validation(model, valid_loader, criterion, epoch)
            if self.is_better(met_ret[self.main_metric], self.best_metric):
                torch.save(model.state_dict(), os.path.join(self.model_save_path, f"best.pt"))
                self.patient_count = 0
                self.best_metric = met_ret[self.main_metric]
                self.logger.info("new best")
            else:
                self.patient_count += 1
                self.logger.info(f"patient = {self.patient_count}")
                if self.patient_count == self.patient:
                    self.logger.info("early stop")
                    break

    def validation(self, model, valid_loader, criterion, epoch):
        # model = torch.load(os.path.join(self.model_save_path, f"latest.pth"))

        all_labels, all_preds = None, None
        all_losses = []
        with torch.no_grad():
            model.eval()
            for d in valid_loader:
                img = d['view1'].float().cuda()
                label = d['label'].cuda()

                logits = model(img)

                loss = criterion(logits, label)
                all_losses.append(loss.item())
                preds = torch.softmax(logits, dim=1).cpu().detach().numpy()[:, 1]  # preds[1]=0:良性, preds[1]=1:恶性

                if all_labels is None and all_preds is None:
                    all_preds = preds
                    all_labels = label.cpu().detach().numpy()
                else:
                    all_preds = np.concatenate([all_preds, preds], axis=0)
                    all_labels = np.concatenate([all_labels, label.cpu().detach().numpy()], axis=0)
        if epoch == 0:  # one time, for image preview
            self.tb_writer.add_images("img", img)

        met_dict = self.metric_cal(all_labels, all_preds)
        if epoch % 10 == 0:  # fixme: fixed for now
            self.val_tb_writer.add_scalar("loss", np.average(all_losses), epoch)
            for i in met_dict:
                self.val_tb_writer.add_scalar(i, met_dict[i], epoch)

        met_log = {k: "{:.5f}".format(met_dict[k]) for k in met_dict}
        self.logger.info(f"Evaluation {epoch}-epoch ==> {met_log}")
        return met_dict

    def test(self, model, test_loader):

        # model = torch.load(os.path.join(self.model_save_path, f"{config['test_model']}.pth"))
        state_dict = torch.load(os.path.join(self.model_save_path, f"{config['test_model']}.pt")) # load best weight
        model.load_state_dict(state_dict)
        start_time = time.time()
        all_labels, all_preds = None, None
        with torch.no_grad():
            model.eval()
            for d in test_loader:
                img = d['view1'].cuda()
                label = d['label'].cuda()

                with torch.no_grad():
                    logits = model(img)

                    preds = torch.softmax(logits, dim=1).cpu().detach().numpy()[:, 1]

                    if all_labels is None and all_preds is None:
                        all_preds = preds
                        all_labels = label.cpu().detach().numpy()
                    else:
                        all_preds = np.concatenate([all_preds, preds], axis=0)
                        all_labels = np.concatenate([all_labels, label.cpu().detach().numpy()], axis=0)

        met_dict = self.metric_cal(all_labels, all_preds)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info('Evaluation time {}'.format(total_time_str))
        med_log = {k: "{:.5f}".format(met_dict[k]) for k in met_dict}
        self.logger.info(f"Testing: {med_log}")

    def main(self):
        self.logger.info("======= ResNet-based image classification ========")
        # config = {'batch_size': self.batch_size, 'image_res': self.image_res,'modal':'v'}
        datasets = create_dataset(self.config['dataset'], self.config)
        train_set, val_set, test_set = datasets[0], datasets[1], datasets[2]
        train_loader, val_loader, test_loader = create_loader(
            [train_set, val_set, test_set],
            samplers=[None, None, None],
            batch_size=[self.config['batch_size'], self.config['batch_size'], self.config['batch_size']],
            num_workers=[self.config['batch_size'] // 2, self.config['batch_size'] // 2,
                         self.config['batch_size'] // 2],
            is_trains=[True, False, False],
            collate_fns=[None, None, None],
            drop_last=[False, False, False]
        )

        model = resnet50(pretrained=True, num_classes=1000)
        if config['pretrain_path'] is not None:
            self.logger.info(f"Load Pretrain Path: {config['pretrain_path']}")
            pretrain_state_dict = torch.load(self.config['pretrain_path'],
                                             map_location="cpu")
            new_s_dict = {'.'.join(k.split('.')[2:]): v for k, v in pretrain_state_dict.items() if
                          k.startswith("visual_encoder.vision_encoder.")}
            try:
                model.load_state_dict(new_s_dict, strict=True)
                # to check whether the new_s_dict is right
            except RuntimeError as re:
                self.logger.info(f"Pretrain Dict Information:{re}")

            model.load_state_dict(new_s_dict, strict=False)
        else:
            self.logger.info("Imnet pretrain")
        if config['l_eval']:
            for p in model.parameters():
                p.requires_grad = False

        model.fc = torch.nn.Linear(2048, 2)
        # self.logger.info(model) Do not need

        # model = vit_b_16(num_classes=2)
        model = model.cuda()

        criterion = torch.nn.CrossEntropyLoss()
        if not config["backbone_reduce_lr"]:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.config['lr'],
                weight_decay=self.config['weight_decay']
            )
        else:
            group_bb = {'params': [], 'lr': self.config["backbone_reduce_ratio"] * self.config['lr']}
            group_fc = {'params': [], 'lr': self.config['lr']}
            list_bb = []
            list_fc = []
            for n, p in model.named_parameters():
                if 'fc' not in n:
                    group_bb['params'].append(p)
                    list_bb.append(n)
                else:
                    group_fc['params'].append(p)
                    list_fc.append(n)
            self.logger.info("backbone_reduce_lr")
            self.logger.info(f"bb:{list_bb}")
            self.logger.info(f"fc:{list_fc}")
            optimizer = torch.optim.Adam(
                [group_bb, group_fc],
                lr=self.config['lr'],
                weight_decay=self.config['weight_decay']
            )
        scheduler_cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   len(train_loader) * self.epoch_num,
                                                                   eta_min=self.config['lr_min'])
        # =============== Train ==================
        self.train(model, train_loader, val_loader, optimizer, criterion, scheduler_cos)

        # =============== Test ===================

        self.test(model, test_loader)

    def init_logger(self):
        logger = logging.getLogger(self.tag)

        logger.setLevel(logging.INFO)
        sh = logging.StreamHandler()
        fh = logging.FileHandler(os.path.join(self.model_save_path, "train.log"), encoding='utf-8')
        formatter = logging.Formatter(fmt='[%(asctime)s:%(levelname)s:%(name)s] %(message)s', datefmt='%m/%d %H:%M:%S')
        sh.setFormatter(formatter)
        fh.setFormatter(formatter)
        logger.addHandler(sh)
        logger.addHandler(fh)
        self.logger = logger

    def log_config(self):
        if hasattr(self, "config"):
            for k in self.config:
                self.logger.info(f"config:{k}={self.config[k]}")
        for k in self.__dict__:
            if k[0] != '_' and (
                    type(self.__dict__[k]) == str or type(self.__dict__[k]) == int or type(self.__dict__[k]) == float):
                self.logger.info(f"config:{k}={self.__dict__[k]}")


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # pretrain_folder = "new_vikl_pt_ls_imnet"
    # pretrain_folder = "new_vikl_pt_hn_tadv2_imnet"
    # pretrain_folder = "new_vikl_pt_ls_projfix_attruni_imnet"
    # pretrain_folder = "new_vikl_pt_ls_fusenorm_attruni_imnet"
    # pretrain_folder = "new_vikl_pt_hn_fusenorm_imnet"
    # pretrain_folder = "new_vikl_pt_ls_projfix_attruni_random"
    pretrain_folder = [
        # "new_vikl_pt_f1_t0.7_imnet_fixed",
        # "new_vikl_pt_f2_t0.7_imnet_fixed",
        # "new_vikl_pt_f3_t0.7_imnet_fixed",
        # "new_vikl_pt_f4_t0.7_imnet_fixed",
        "new_vikl_pt_ls_imnet"
        # "new_vikl_pt_fdebug2_t0.7_imnet_fixed"
    ]
    dataset = "hmbm"
    for p in pretrain_folder:
        # for part in [0.2, 0.4, 0.6, 0.8, 1.0]:
        # L-EVAL Setting
        # config = {"dataset": dataset, "binary_label": True,
        #           'main_metric': 'auc', 'metric_direction': "high", 'patient': 100, 'epoch_num': 1000,
        #           'model_save_path': "/data2/zhai/HMBM/output/model_fixed",
        #           'tb_save_path': "/data2/zhai/HMBM/output/tb_fixed",
        #           'batch_size': 48, 'image_res': 256, 'best_init': -1, 'modal': 'v', 'pre_downsample': True,
        #           't_backbone': None, "backbone_reduce_lr": False, "backbone_reduce_ratio": 0.05,
        #           # 'pretrain_path': None,
        #           'pretrain_path': f'/data2/zhai/HMBM/output/model_pt_fixed/{p}/latest.pth',#TODO:Change here for best or latest
        #           'test_model': 'best','partial_data':None
        #           'crop_min_scale': 0.5, "padding_square": True, 'attr_noise': None,
        #           'lr': 1e-3, 'weight_decay': 1e-6, 'lr_min': 1e-6, 'l_eval': True}
        # FT Setting
        config = {"dataset": dataset, "binary_label": True,
                  'main_metric': 'auc', 'metric_direction': "high", 'patient': 100, 'epoch_num': 1000,
                  'model_save_path': "/data2/zhai/HMBM/output/model_fixed",
                  'tb_save_path': "/data2/zhai/HMBM/output/tb_fixed",
                  'batch_size': 48, 'image_res': 256, 'best_init': -1, 'modal': 'v', 'pre_downsample': True,
                  't_backbone': None, "backbone_reduce_ratio": 0.1,
                  # 'pretrain_path': None, "backbone_reduce_lr": False,
                  'pretrain_path': f'/data2/zhai/HMBM/output/model_pt_fixed/{p}/latest.pt' if p is not None else None,
                  "backbone_reduce_lr": True if p is not None else False,
                  # "backbone_reduce_lr": False,
                  'test_model': 'best', 'attr_noise': None,
                  'crop_min_scale': 0.5, "padding_square": True, 'partial_data': None,
                  'lr': 5e-5, 'weight_decay': 5e-5, 'lr_min': 5e-7, 'l_eval': False}
        for i in range(5):
            trainer = Trainer(
                tag=f"{p}_{dataset}_{'l_eval' if config['l_eval'] else 'ft'}_torch2_fixed_{i}",
                **config)
            # noes: no early stop
            # ps: padding square
            trainer.main()
