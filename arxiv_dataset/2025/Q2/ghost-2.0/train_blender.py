import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision
import lightning.pytorch as pl
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
import wandb
from lightning.pytorch.loggers import WandbLogger
import argparse
from omegaconf import OmegaConf

from src.data import Voxceleb2H5Dataset
from src.data import CustomBatchSampler
from src.losses import PerceptualLoss, DiscriminatorLoss
from src.aligner import Discriminator
from src.utils.color_conversion import *
import src.utils.kornia_morphology
from src.data import BlenderDataset
from src.blender.generator import BlenderGenerator
from src.losses import ReferenceRegularizationLoss



class BlenderLoss(nn.Module):
    def __init__(self, disc, w_perc_vgg=1e-2, w_rec=30, w_cycle=1, w_adv=1, w_reg=1):
        super(BlenderLoss, self).__init__()
        self.perc_vgg_loss = PerceptualLoss(1, './weights')
        self.disc = disc
        self.disc_loss = DiscriminatorLoss()
        self.reg_loss = ReferenceRegularizationLoss()
        
        self.weights = [w_perc_vgg, w_rec, w_cycle, w_adv, w_reg]
    
    def forward(self, forward_batch, train_batch):
        
        L_perc_vgg = self.perc_vgg_loss(forward_batch['fake'], train_batch['face_target'])
        
        L_rec = F.l1_loss(forward_batch['fake'], train_batch['face_target'])
        
        L_cycle = F.l1_loss(forward_batch['fake_pair'], forward_batch['label_pair'].detach()) + \
            F.l1_loss(forward_batch['fake_nopair'], forward_batch['label_nopair'].detach())
        
        disc_outputs = self.disc({
            'fake_rgbs': torch.cat((forward_batch['fake'], forward_batch['M_Ah'], forward_batch['M_Ai']), dim=1),
            'target_rgbs': torch.cat((train_batch['face_target'], forward_batch['M_Ah'], forward_batch['M_Ai']), dim=1)
        })
        L_disc_G_losses = self.disc_loss.forward_gen(disc_outputs)
        _, L_adv_G = L_disc_G_losses['L_perc_disc'], L_disc_G_losses['L_adv_G']
        L_adv_D = self.disc_loss.forward_disc(disc_outputs)
        L_reg = self.reg_loss(forward_batch['gen_total'], forward_batch['I_gd'])
        
        L_G = sum(
            L * w for L, w
            in zip((L_perc_vgg, L_rec, L_cycle, L_adv_G, L_reg), self.weights)
        )

        L_D = L_adv_D
        
        return {
            'L_perc_vgg': L_perc_vgg,
            'L_rec': L_rec,
            'L_cycle': L_cycle,
            'L_adv_G': L_adv_G,
            'L_adv_D': L_adv_D,
            'L_G': L_G,
            'L_D': L_D,
            'L_reg': L_reg
        }
    
class BlenderModule(pl.LightningModule):
    def __init__(self, cfg):
        super(BlenderModule, self).__init__()
        self.gen = BlenderGenerator()
        self.disc = Discriminator(in_channels=5)
        self.blender_loss = BlenderLoss(
            self.disc,
            w_perc_vgg=cfg.train_options.weights.w_perc_vgg,
            w_rec=cfg.train_options.weights.w_rec,
            w_cycle=cfg.train_options.weights.w_cycle,
            w_adv=cfg.train_options.weights.w_adv,
            w_reg=cfg.train_options.weights.w_reg
        )
        self.g_lr = cfg.train_options.optim.g_lr
        self.d_lr = cfg.train_options.optim.d_lr
        self.g_clip = cfg.train_options.optim.g_clip
        self.d_clip = cfg.train_options.optim.d_clip
        self.betas = (cfg.train_options.optim.beta1, cfg.train_options.optim.beta2)
        self.automatic_optimization = False
        self.save_hyperparameters()
        
    def forward(self, batch, old_version=False, copy_source_attrb=False, inpainter=None):
        oup, gen_h, gen_i, M_Ah, I_tb, M_Ai, I_ag = self.gen(
            batch['face_source'], batch['gray_source'], batch['face_target'],
            batch['mask_source'], batch['mask_target'],
            gt=batch['face_target'],
            M_a_noise=batch['mask_source_noise'], M_t_noise=batch['mask_target_noise'],
            cycle=False, train=False,
            return_inputs=True,
            old_version = old_version,
            copy_source_attrb = copy_source_attrb,
            inpainter=inpainter
        )
        
        return {
            'oup': oup,
            'gen_h': gen_h,
            'gen_i': gen_i,
            'M_Ah': M_Ah,
            'I_tb': I_tb,
            'M_Ai': M_Ai,
            'I_ag': I_ag
        }
    
    def forward_train(self, batch):
        
        fake, M_Ah, M_Ai, gen_total, I_gd = self.gen(
            batch['face_source'], batch['gray_source'], batch['face_target'],
            batch['mask_source'], batch['mask_target'],
            gt=batch['face_target'],
            M_a_noise=batch['mask_source_noise'], M_t_noise=batch['mask_target_noise'],
            cycle=False, train=True, old_version=True
        )
        
        fake_pair, label_pair = self.gen(
            batch['face_source'], batch['gray_source'], batch['face_target'],
            batch['mask_source'], batch['mask_target'],
            gt=None,
            cycle=True, train=False, old_version=True
        )
        
        fake_nopair, label_nopair = self.gen(
            batch['face_source'], batch['gray_source'], batch['face_side'],
            batch['mask_source'], batch['mask_side'],
            gt=None,
            cycle=True, train=False, old_version=True
        )

        return {
            'fake': fake,
            'M_Ah': M_Ah,
            'M_Ai': M_Ai,
            'fake_pair': fake_pair,
            'label_pair': label_pair,
            'fake_nopair': fake_nopair,
            'label_nopair': label_nopair,
            'gen_total': gen_total,
            'I_gd': I_gd
        }
        

    def configure_optimizers(self):
        opt_G = torch.optim.Adam(self.gen.parameters(), lr=self.g_lr, betas=self.betas, eps=1e-5)
        opt_D = torch.optim.Adam(self.disc.parameters(), lr=self.d_lr, betas=self.betas, eps=1e-5)
        return opt_G, opt_D

    def training_step(self, train_batch, batch_idx):
        opt_G, opt_D = self.optimizers()
        
        forward_dict = self.forward_train(train_batch)
        
        losses = self.blender_loss(forward_dict, train_batch)
        
        def closure_G():
            opt_G.zero_grad()
            self.manual_backward(losses['L_G'], retain_graph=True)
            self.clip_gradients(opt_G, gradient_clip_val=self.g_clip)
            return losses['L_G']
        opt_G.step(closure=closure_G)
        
        def closure_D():
            opt_D.zero_grad()
            self.manual_backward(losses['L_D'])
            self.clip_gradients(opt_D, gradient_clip_val=self.d_clip)
            return losses['L_D']
        opt_D.step(closure=closure_D)
        
        logs = dict((k, v.item()) for k, v in losses.items())
        self.log_dict(logs)

        return logs

    def validation_step(self, val_batch, batch_idx, old_version=True, copy_source_attrb=False):
        with torch.no_grad():
            return dict(self.forward(val_batch, old_version=old_version, copy_source_attrb=copy_source_attrb), **val_batch)


class BlenderLogPredictionSamplesCallback(pl.Callback):
    def __init__(self, wandb_logger, n=2):
        super(BlenderLogPredictionSamplesCallback, self).__init__()
        self.wandb_logger = wandb_logger
        self.n = n
    
    @staticmethod
    def put_text(img, text):
        return cv2.putText(np.ascontiguousarray(img), text, (64, 64), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    @staticmethod
    def img_to_bgr(dict, key, i):
        img = dict[key][i]
        img_shape = list(img.shape)
        img_shape[0] = 3
        img = (img.expand(*img_shape).cpu().numpy().transpose((1, 2, 0)) / 2 + 0.5).astype(np.float32)
        
        img = BlenderLogPredictionSamplesCallback.put_text(img, key)
        return img
    
    @staticmethod
    def mask_to_bgr(dict, key, i, scale=18, cmap='viridis'):
        mask = dict[key][i]
        mask = plt.get_cmap(cmap)(mask[0].cpu().numpy() / scale)[:, :, [2, 1, 0]].astype(np.float32)
        mask = BlenderLogPredictionSamplesCallback.put_text(mask, key)
        return mask
    
    @staticmethod
    def create_template():
        return np.full((512, 512, 3), 0, dtype=np.float32)
        
    def create_grids(self, outputs):
        """
            layout:
            'face_orig'   | 'face_source' | 'face_target' | 'gen_h' | 'M_Ah' | 'I_ag' | 'oup'
            'mask_source' | 'gray_source' | 'mask_target' | 'gen_i' | 'M_Ai' | 'I_tb' | <black>
        """
        
        samples = []

        batch_size = outputs['face_orig'].shape[0]
        for i in range(min(batch_size, self.n)):
            sample = [
                [
                    self.img_to_bgr(outputs, 'face_orig', i), self.img_to_bgr(outputs, 'face_source', i),
                    self.img_to_bgr(outputs, 'face_target', i), self.img_to_bgr(outputs, 'gen_h', i),
                    self.mask_to_bgr(outputs, 'M_Ah', i, scale=1., cmap='gray'), self.img_to_bgr(outputs, 'I_ag', i),
                    self.img_to_bgr(outputs, 'oup', i)
                ],


                [
                    self.mask_to_bgr(outputs, 'mask_source', i), self.img_to_bgr(outputs, 'gray_source', i),
                    self.mask_to_bgr(outputs, 'mask_target', i), self.img_to_bgr(outputs, 'gen_i', i),
                    self.mask_to_bgr(outputs, 'M_Ai', i, scale=1., cmap='gray'), self.img_to_bgr(outputs, 'I_tb', i),
                    self.create_template()
                ]
            ]
                
            sample = np.concatenate([np.concatenate(row, axis=1) for row in sample], axis=0)
            samples.append(sample)
        
        sample_shape = samples[0].shape
        border = np.full((16, sample_shape[1], 3), 1).astype(np.float32)
        samples_with_borders = []
        for i, sample in enumerate(samples):
            samples_with_borders.append(sample)
            if i != len(samples) - 1:
                samples_with_borders.append(border)
        
        samples_with_borders = np.concatenate(samples_with_borders, axis=0)
        samples_rgb = np.clip(np.nan_to_num(samples_with_borders[:, :, ::-1]), 0, 1)
        
        return samples_rgb
        
    
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Called when the validation batch ends."""
        
        if batch_idx == 0:
            samples_rgb = self.create_grids(outputs)
            wandb_logger.log_image(key="samples", images=[samples_rgb], step=trainer.global_step)

def create_dataset(cfg, source_transform=None):
    train_dataset = BlenderDataset(
        cfg.data_path,
        source_transform=source_transform,
        shuffle=cfg.shuffle,
        flip_target=cfg.flip_target,
        affine_source=cfg.affine_source,
        make_noise=cfg.make_noise,
        subset_size=cfg.subset_size
    )
    sampler = CustomBatchSampler(train_dataset)
    dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=sampler, num_workers=cfg.num_workers)
    return dataloader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/blender.yaml")
    args = parser.parse_args()
    
    with open(args.config, "r") as stream:
        cfg = OmegaConf.load(stream)
    
    model = BlenderModule(
        cfg
    )

    source_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.ColorJitter(
            *([cfg.train_options.jitter_value] * 4)
        ),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataloader = create_dataset(cfg.train_options, source_transform=source_transform)
    val_dataloader = create_dataset(cfg.inference_options)

    wandb_logger = WandbLogger(project='Blender', name=cfg.experiment_name, reinit=True, settings=wandb.Settings(code_dir="."))
    
    if wandb.run is not None:
        wandb.run.log_code('.')
    
    wandb_logger.watch(model, log_freq=cfg.train_options.wandb_log_freq)
    log_pred_callback = BlenderLogPredictionSamplesCallback(wandb_logger)
    trainer = pl.Trainer(
        max_epochs=cfg.train_options.max_epochs,
        accelerator='gpu', devices=cfg.num_gpus,
        log_every_n_steps=cfg.train_options.log_train_freq,
        val_check_interval=cfg.train_options.log_interval,
        logger=wandb_logger, callbacks=[
            log_pred_callback, pl.callbacks.ModelCheckpoint(
                save_last=cfg.train_options.save_last,
                every_n_epochs=cfg.train_options.save_every_n_epochs,
                save_top_k=cfg.train_options.save_top_k
            )
        ],
        precision=16,
        strategy='ddp_find_unused_parameters_true',
    )
    torch.set_float32_matmul_precision('medium')

    trainer.fit(model, train_dataloader, val_dataloader)
    wandb_logger.experiment.unwatch(model)