# -*- coding: utf-8 -*-
import os
import tqdm
import numpy as np
from models.manineg_models import ManinegIM, ManinegII
from dataset import create_dataset, generate_attr_unique, create_loader
import torch
from torch.utils.tensorboard import SummaryWriter
import logging
from models.loss_function import Vikl_Loss_Adaptive, Vikl_Loss_Adaptive_HRW
from dataset.manifest_sampler import LinearMuSchedular, TruncatedGaussianSampler, NegManiBatchSampler
from torch.utils.data import DataLoader


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
        if config["reweight_hardneg_enable"]:
            self.loss = Vikl_Loss_Adaptive_HRW(self.device, config['ii_enable'], config['im_enable'],
                                               config['it_enable'],
                                               config['tm_enable'], config['temperature'],
                                               config['trainable_temperature'])
        else:
            self.loss = Vikl_Loss_Adaptive(self.device, config['ii_enable'], config['im_enable'], config['it_enable'],
                                           config['tm_enable'], config['temperature'], config['trainable_temperature'])
        self.mu_schedular = LinearMuSchedular(config['max_mu'], config['min_mu'], config['epoch_min_mu'])
        self.gaussian_sampler = TruncatedGaussianSampler(config['max_mu'], config['sigma'],
                                                         config['truncated_gaussian_left'],
                                                         config['truncated_gaussian_right'], config['batch_size'] - 1)

        self.logger = None
        self.init_logger()
        self.log_config()

    def train(self, model, train_loader, valid_loader, optimizer, scheduler_main, scheduler_warmup):
        for epoch in range(self.epoch_num):
            if epoch < self.warm_up_epoch:
                warm_up = True
            else:
                warm_up = False
            self.gaussian_sampler.set_param(self.mu_schedular.get_mu(epoch))
            all_losses = None
            k_list = None
            model.train()
            for d in tqdm.tqdm(train_loader):
                d = {k: v.to(self.device) for k, v in d.items()}

                _, img_z1, _, img_z2, _, text_z, _, attr_z = model(d)

                attr_uni = generate_attr_unique(d['attr']).to(self.device)
                loss, met_ret = self.loss(img_z1, img_z2, text_z, attr_z, attr_uni)

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
            self.tb_writer.add_scalar("mu", self.gaussian_sampler.mu, epoch)
            all_losses = np.average(all_losses, axis=0)
            for i in range(all_losses.shape[0]):
                self.tb_writer.add_scalar(k_list[i], all_losses[i], epoch)

            self.logger.info(f"epoch: {epoch}")
            torch.save(model.state_dict(), os.path.join(self.model_save_path, f"latest.pt"))
            met_dict = self.validation(model, valid_loader, epoch)
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

    def validation(self, model, valid_loader, epoch):
        # model = torch.load(os.path.join(self.model_save_path, f"latest.pth"))
        all_losses = None
        k_list = None
        with torch.no_grad():
            model.eval()
            for d in valid_loader:
                d = {k: v.to(self.device) for k, v in d.items()}
                _, img_z1, _, img_z2, _, text_z, _, attr_z = model(d)

                attr_uni = generate_attr_unique(d['attr']).to(self.device)
                loss, met_ret = self.loss(img_z1, img_z2, text_z, attr_z, attr_uni)

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

    def test(self, model, test_loader):
        # model = torch.load(os.path.join(self.model_save_path, f"latest.pth"))
        state_dict = torch.load(os.path.join(self.model_save_path, f"latest.pt"))
        model.load_state_dict(state_dict)
        k_list = None
        all_losses = None
        with torch.no_grad():
            model.eval()
            for d in test_loader:
                d = {k: v.to(self.device) for k, v in d.items()}
                _, img_z1, _, img_z2, _, text_z, _, attr_z = model(d)

                attr_uni = generate_attr_unique(d['attr']).to(self.device)
                loss, met_ret = self.loss(img_z1, img_z2, text_z, attr_z, attr_uni)

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
        datasets = create_dataset('hmbm', self.config)
        train_set, val_set, test_set = datasets[0], datasets[1], datasets[2]

        val_loader, test_loader = create_loader(
            [val_set, test_set],
            samplers=[None, None],
            batch_size=[self.config['batch_size'], self.config['batch_size']],
            num_workers=[self.config['batch_size'] // 2, self.config['batch_size'] // 2],
            is_trains=[False, False],
            collate_fns=[None, None],
            drop_last=[True, True]
        )

        self.neg_mani_batchsampler = NegManiBatchSampler(train_set, self.gaussian_sampler,
                                                         max_length=self.config['sampler_max_length'],
                                                         deduplicate=self.config['sampler_dedup'])
        if self.config['manineg_enable']:
            train_loader = DataLoader(
                train_set,
                num_workers=self.config['batch_size'] // 2,
                pin_memory=True,
                batch_sampler=self.neg_mani_batchsampler,
                persistent_workers=True,
            )
        else:
            train_loader = create_loader(
                [train_set],
                samplers=[None],
                batch_size=[self.config['batch_size']],
                num_workers=[self.config['batch_size'] // 2],
                is_trains=[False],
                collate_fns=[None],
                drop_last=[False]
            )[0]

        if self.config['model'] == "IM":
            model = ManinegIM(**self.config).to(self.device)
        elif self.config['model'] == "II":
            model = ManinegII(**self.config).to(self.device)
        else:
            raise NotImplementedError(f"Model {self.config['model']} is not implemented.")

        # First version of trainable temperature. For simplicity, now loss parameters also use adam optimizer.
        # Fixed lr and wd, can be changed later.
        if self.config['trainable_temperature']:
            group_model = {'params': model.parameters(), 'lr': 1e-4, "weight_decay": 1e-4}
            group_loss = {'params': self.loss.parameters(), 'lr': 1e-2, 'weight_decay': 0.0}
            optimizer = torch.optim.Adam(
                [group_model, group_loss],
            )
        else:
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
        self.train(model, train_loader, val_loader, optimizer, scheduler_cos, scheduler_linear)

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
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    os.environ["https_proxy"] = "http://172.17.146.34:8891"
    os.environ["http_proxy"] = "http://172.17.146.34:8891"
    config = {'main_metric': 'loss_sum', 'metric_direction': "low", 'patient': -1, 'epoch_num': 300,
              'warm_up_epoch': 10,
              'model_save_path': "/data2/zhai/HMBM/output/model_pt_fixed",
              'tb_save_path': "/data2/zhai/HMBM/output/tb_pt_fixed",
              'batch_size': 64, 'image_res': 256, 'best_init': 10000, 'modal': 'dal', 'device_id': 0,
              'v_backbone': "resnet-50", 't_backbone': 'bert-base-chinese', 'pre_downsample': True,
              'crop_min_scale': 0.5, 'drop_text': 0.5, 'drop_attr': 0.5, 'vision_pretrained': True, 'attr_noise': None,
              'partial_data': None,

              'ii_enable': True, 'im_enable': False, 'it_enable': False, 'tm_enable': False,
              'temperature': 0.7, 'trainable_temperature': True,

              'manineg_enable': True, 'reweight_hardneg_enable': False, 'model': "II",
              'max_mu': 11, 'min_mu': 0, 'epoch_min_mu': 3, 'sigma': 3, "truncated_gaussian_left": 1,
              "truncated_gaussian_right": 18, 'sampler_max_length': 100, 'sampler_dedup': True}

    trainer = Trainer(tag=f"manineg_{config['model']}_{'tt_' if config['trainable_temperature'] else ''}"
                          f"emm3_ms_pt_{'imnet' if config['vision_pretrained'] else 'random'}",
                      **config)
    # trainer = Trainer(tag=f"debug", **config)
    trainer.main()
