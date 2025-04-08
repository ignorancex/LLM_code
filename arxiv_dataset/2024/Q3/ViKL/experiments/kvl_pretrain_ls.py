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
# from HMBM.dataset import create_dataset, create_sampler, create_loader
from models.vikl_new import ViKLNet_mixProj
# from HMBM.models.vikl_new import ViKLNet_mixProj_2d
# from HMBM.models.vikl_fuse_norm import ViKLNet_mixProj
from dataset import create_dataset, generate_attr_unique, create_loader
from sklearn.metrics import roc_curve, accuracy_score, f1_score, roc_auc_score, auc, confusion_matrix
import torch
from torch.utils.data import ConcatDataset, random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.resnet import resnet50, resnet18

from torchvision import transforms
import torchvision.transforms as T

# from ALBEF.dataset.hmbm_dataset import HMBMDownstreamDataset
# from ALBEF import utils
import logging
from torch import nn
# from HMBM.models.loss_function import NT_Xent, SampleNT_Xent, Vikl_Loss
from models.loss_function import Vikl_Loss_AttrUni, Vikl_Fusion


class Trainer:
    def __init__(self, tag, main_metric, metric_direction, patient, epoch_num, warm_up_epoch,
                 model_save_path, tb_save_path, best_init, device_id, **kwargs):
        self.config = kwargs
        self.tag = tag
        self.metric_direction = metric_direction
        self.device = torch.device("cuda", device_id)
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
        self.warm_up_epoch = warm_up_epoch
        # self.drop_text = config['drop_text']
        # self.drop_attr = config['drop_attr']
        # FIXME: world size is now fixed to 1, which means the code is not support multi card now
        # fix later
        # self.nt_xent = NT_Xent(config['batch_size'], config['temperature'], self.device, 1)
        self.loss = Vikl_Fusion(self.device)
        # Vikl_Loss_AttrUni(config['temperature'], self.device, 0.2, 0.3, 0.2, 0.0)  # todo:fixed for now
        self.logger = None
        self.init_logger()
        self.log_config()

    def train(self, model, train_loader, valid_loader, optimizer, criterion, scheduler_main, scheduler_warmup):
        for epoch in range(self.epoch_num):
            if epoch < self.warm_up_epoch:
                warm_up = True
            else:
                warm_up = False

            all_losses = None
            k_list = None
            model.train()
            for d in tqdm.tqdm(train_loader):
                d = {k: v.to(self.device) for k, v in d.items()}

                _, img_z1, _, img_z2, _, text_z, _, attr_z = model(d)
                attr_uni = generate_attr_unique(d['attr']).to(self.device)

                loss, met_ret = criterion(img_z1, img_z2, text_z, attr_z, attr_uni)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if warm_up and scheduler_warmup is not None:
                    scheduler_warmup.step()
                if not warm_up and scheduler_main is not None:
                    scheduler_main.step()
                loss_list = np.array([v for k, v in met_ret.items()])[None, :]
                if k_list is None:
                    k_list = [k for k in met_ret.keys()]
                if all_losses is None:
                    all_losses = loss_list
                else:
                    all_losses = np.concatenate([all_losses, loss_list])

            # fixme: naive img show for image show

            self.tb_writer.add_images(f"view1", d['view1'], epoch)
            self.tb_writer.add_images(f"view2", d['view2'], epoch)

            self.tb_writer.add_scalar("lr", optimizer.param_groups[-1]['lr'], epoch)
            all_losses = np.average(all_losses, axis=0)
            for i in range(all_losses.shape[0]):
                self.tb_writer.add_scalar(k_list[i], all_losses[i], epoch)

            self.logger.info(f"epoch: {epoch}")
            torch.save(model.state_dict(), os.path.join(self.model_save_path, f"latest.pt"))
            met_dict = self.validation(model,valid_loader, criterion, epoch)
            # fixme:add later
            if self.is_better(met_dict[self.main_metric], self.best_metric):
                torch.save(model.state_dict(), os.path.join(self.model_save_path, f"best.pt"))
                self.patient_count = 0
                self.best_metric = met_dict[self.main_metric]
                self.logger.info("new best")
            else:
                self.patient_count += 1
                self.logger.info(f"patient = {self.patient_count}")
                if self.patient_count == self.patient:
                    self.logger.info("early stop")
                    break

    def validation(self,model ,valid_loader, criterion, epoch):
        # model = torch.load(os.path.join(self.model_save_path, f"latest.pth"))
        all_losses = None
        k_list = None
        with torch.no_grad():
            model.eval()
            for d in valid_loader:
                d = {k: v.to(self.device) for k, v in d.items()}
                _, img_z1, _, img_z2, _, text_z, _, attr_z = model(d)

                attr_uni = generate_attr_unique(d['attr']).to(self.device)
                loss, met_ret = criterion(img_z1, img_z2, text_z, attr_z, attr_uni)

                loss_list = np.array([v for k, v in met_ret.items()])[None, :]
                if k_list is None:
                    k_list = [k for k in met_ret.keys()]
                if all_losses is None:
                    all_losses = loss_list
                else:
                    all_losses = np.concatenate([all_losses, loss_list])
            self.val_tb_writer.add_images(f"view1", d['view1'], epoch)
            self.val_tb_writer.add_images(f"view2", d['view2'], epoch)

            all_losses = np.average(all_losses, axis=0)
            met_dict = {k_list[i]: all_losses[i] for i in range(all_losses.shape[0])}
            for i in met_dict:
                self.val_tb_writer.add_scalar(i, met_dict[i], epoch)
            met_log = {k: "{:.5f}".format(met_dict[k]) for k in met_dict}
            self.logger.info(f"Evaluation {epoch}-epoch ==> {met_log}")
            return met_dict

    def test(self,model ,test_loader, criterion):
        # model = torch.load(os.path.join(self.model_save_path, f"latest.pth"))
        state_dict = torch.load(os.path.join(self.model_save_path, f"latest.pt")) # load best weight
        model.load_state_dict(state_dict)
        k_list = None
        all_losses = None
        with torch.no_grad():
            model.eval()
            for d in test_loader:
                d = {k: v.to(self.device) for k, v in d.items()}
                _, img_z1, _, img_z2, _, text_z, _, attr_z = model(d)

                attr_uni = generate_attr_unique(d['attr']).to(self.device)
                loss, met_ret = criterion(img_z1, img_z2, text_z, attr_z, attr_uni)

                loss_list = np.array([v for k, v in met_ret.items()])[None, :]
                if k_list is None:
                    k_list = [k for k in met_ret.keys()]
                if all_losses is None:
                    all_losses = loss_list
                else:
                    all_losses = np.concatenate([all_losses, loss_list])

            all_losses = np.average(all_losses, axis=0)
            met_dict = {k_list[i]: all_losses[i] for i in range(all_losses.shape[0])}
            met_log = {k: "{:.5f}".format(met_dict[k]) for k in met_dict}
            self.logger.info(f"Testing: {met_log}")

    def main(self):
        self.logger.info("======= PT ========")
        # config = {'batch_size': self.batch_size, 'image_res': self.image_res, 'modal':'a'}
        datasets = create_dataset('hmbm', self.config)
        train_set, val_set, test_set = datasets[0], datasets[1], datasets[2]
        train_loader, val_loader, test_loader = create_loader(
            [train_set, val_set, test_set],
            samplers=[None, None, None],
            batch_size=[self.config['batch_size'], self.config['batch_size'], self.config['batch_size']],
            num_workers=[self.config['batch_size'] // 2, self.config['batch_size'] // 2,
                         self.config['batch_size'] // 2],
            is_trains=[True, False, False],
            collate_fns=[None, None, None],
            drop_last=[True, True, True]
        )

        model = ViKLNet_mixProj(**self.config).to(self.device)

        criterion = self.loss

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-4
        )
        scheduler_cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   len(train_loader) * (
                                                                           self.epoch_num - self.warm_up_epoch),
                                                                   eta_min=1e-7)
        scheduler_linear = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, end_factor=1,
                                                             total_iters=len(train_loader) * self.warm_up_epoch)
        # =============== Train ==================
        self.train(model, train_loader, val_loader, optimizer, criterion, scheduler_cos, scheduler_linear)

        # =============== Test ===================

        self.test(model,test_loader, criterion)

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
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    config = {'main_metric': 'loss_sum', 'metric_direction': "low", 'patient': -1, 'epoch_num': 300,
              'warm_up_epoch': 10,
              'model_save_path': "/data2/zhai/HMBM/output/model_pt_fixed",
              'tb_save_path': "/data2/zhai/HMBM/output/tb_pt_fixed",
              'batch_size': 64, 'image_res': 256, 'best_init': 10000, 'modal': 'dal', 'device_id': 0,
              'v_backbone': "resnet-50", 't_backbone': 'bert-base-chinese', 'pre_downsample': True, 'temperature': 0.7,
              'crop_min_scale': 0.5, 'drop_text': 0.5, 'drop_attr': 0.5, 'vision_pretrained': True, 'attr_noise': None,'partial_data': None}

    # trainer = Trainer(tag="new_vikl_pt_dp0.5_w_vpt", **config)
    trainer = Trainer(tag="new_vikl_pt_f2_t0.7_imnet_fixed", **config)
    trainer.main()
