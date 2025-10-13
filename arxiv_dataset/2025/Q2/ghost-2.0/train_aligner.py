import os
import torch
import argparse
import numpy as np
import torchvision
import torch.nn as nn
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf


from repos.stylematte.stylematte.models import StyleMatte
from src.data import Voxceleb2H5Dataset, CustomBatchSampler
from src.utils.logging import LogPredictionSamplesCallback, PeriodicCheckpoint
from src.utils.preprocess import make_X_dict, blend_alpha
from src.losses import *
from src.aligner import *

    
class AlignerLoss(nn.Module):
    def __init__(self, id_encoder, gaze_start, disc, w_rec=30, w_perc_vgg=1e-2, w_perc_id=2e-3, w_perc_disc=10, w_id=1e-2, w_adv=0.1, w_mask=1, w_emotion=1, w_kpt=30, w_gaze=1):
        super(AlignerLoss, self).__init__()
        self.perc_vgg_loss = PerceptualLoss(1, './weights/')
        self.id_loss = ArcFaceLoss(id_encoder)
        self.disc = disc
        self.disc_loss = DiscriminatorLoss()
        self.mask_loss = DiceLoss()
        self.emotion_loss = EmotionLoss()
        self.keypoint_loss = KeypointLoss(mode='expr')
        self.gaze_start = gaze_start
        self.gaze = GazeLossRTGene(device='cuda', gaze_model_types=['vgg16'])

        self.weights = [w_rec, w_perc_vgg, w_perc_id, w_perc_disc, w_id, w_adv, w_mask, w_emotion, w_kpt, w_gaze]

        
    def forward(self, data_dict, X_dict, epoch):
        masked_fake = data_dict['fake_rgbs']
        masked_target = X_dict['target']['face_wide']
        
        if 'face_wide_mask' in X_dict['target']:
            mask = X_dict['target']['face_wide_mask']
            
            masked_fake = blend_alpha(masked_fake, mask)
            masked_target = blend_alpha(masked_target, mask) 
        
        
        L_rec = F.l1_loss(masked_fake, masked_target)
            
        L_perc_vgg = self.perc_vgg_loss(masked_fake, masked_target)

        id_loss_outputs = self.id_loss(data_dict, X_dict)
        
        L_perc_id, L_id = id_loss_outputs['L_perc_id'], id_loss_outputs['L_id']
        disc_outputs = self.disc({
            'fake_rgbs': masked_fake,
            'target_rgbs': masked_target
        })
        
        L_disc_G_losses = self.disc_loss.forward_gen(disc_outputs)
        L_perc_disc, L_adv_G = L_disc_G_losses['L_perc_disc'], L_disc_G_losses['L_adv_G']
        L_adv_D = self.disc_loss.forward_disc(disc_outputs)
        
        L_mask = self.mask_loss.forward({
            'fake_segm': data_dict['fake_segm'],
            'real_segm': X_dict['target']['face_wide_mask']
        })

        emotion_gt = X_dict['target']['face_emoca'][:, [2, 1, 0], :, :] / 2 + 0.5
        emotion_pred = masked_fake[:, [2, 1, 0], :, :] / 2 + 0.5
        L_emotion = self.emotion_loss(emotion_pred, emotion_gt, X_dict['target']['keypoints'])
        
        if epoch >= 0:
            input_kpts = (masked_fake[:, [2, 1, 0], :, :] / 2 + 0.5) * 255
            gt_kpts = (masked_fake[:, [2, 1, 0], :, :] / 2 + 0.5) * 255
            L_kpt = self.keypoint_loss(input_kpts, gt_kpts)

            


        if epoch >= self.gaze_start:
            try:
                L_gaze = self.gaze(masked_fake[:, [2, 1, 0], :, :], masked_target[:, [2, 1, 0], :, :],
                                                            X_dict['target']['keypoints'])
                if L_gaze.shape != torch.Size([]):
                    print('gaze_loss returned list: ', L_gaze)
                    L_gaze = L_adv_G * 0
                    
            except Exception as e:
                print(e)
                print('error in gaze')
                L_gaze = L_adv_G * 0


        if epoch == 0:
            L_G = sum(
                        L * w for L, w
                        in zip((L_rec, L_perc_vgg, L_perc_id, L_perc_disc, L_id, L_adv_G, L_mask, L_emotion), self.weights[:-2]))
        elif epoch > 0 and epoch < self.gaze_start:
            L_G = sum(
                    L * w for L, w
                    in zip((L_rec, L_perc_vgg, L_perc_id, L_perc_disc, L_id, L_adv_G, L_mask, L_emotion, L_kpt), self.weights[:-1]))
        else:
            L_G = sum(
                    L * w for L, w
                    in zip((L_rec, L_perc_vgg, L_perc_id, L_perc_disc, L_id, L_adv_G, L_mask, L_emotion, L_kpt, L_gaze), self.weights))
            
            
        L_D = L_adv_D 

        
        loss_dict = {
            'L_rec': L_rec,
            'L_perc_vgg': L_perc_vgg,
            'L_perc_id': L_perc_id,
            'L_perc_disc': L_perc_disc,
            'L_id': L_id,
            'L_adv_G': L_adv_G,
            'L_adv_D': L_adv_D,
            'L_mask': L_mask,
            'L_emotion': L_emotion,
            'L_G': L_G,
            'L_D': L_D
        }
        if epoch > 0:
            loss_dict['L_kpt'] = L_kpt

        if epoch >= self.gaze_start:
            loss_dict['L_gaze'] = L_gaze
            
        return loss_dict
    


class AlignerModule(pl.LightningModule):
    def __init__(self, cfg):
        super(AlignerModule, self).__init__()
        
        self.embedder = Embedder(**cfg.model.embed)
        self.gen = Generator(d_por=cfg.model.embed.d_por,
                             d_id=cfg.model.embed.d_id,
                             d_pose=cfg.model.embed.d_pose,
                             d_exp=cfg.model.embed.d_exp,
                             **cfg.model.gen)
        self.disc = Discriminator(**cfg.model.discr)
        self.aligner_loss = AlignerLoss(
            id_encoder=self.embedder.id_encoder,
            disc=self.disc,
            gaze_start = cfg.train_options.gaze_start,
            **cfg.train_options.weights
        )
        optim_options = cfg.train_options.optim
        self.g_lr = optim_options.g_lr
        self.d_lr = optim_options.d_lr
        self.g_clip = optim_options.g_clip
        self.d_clip = optim_options.d_clip
        self.betas = (optim_options.beta1, optim_options.beta2)
        self.segment_model = None
        self.automatic_optimization = False
        
        self.save_hyperparameters()
        
        if cfg.model.segment:
            self.segment_model = StyleMatte()
            self.segment_model.load_state_dict(
                torch.load( './repos/stylematte/stylematte/checkpoints/stylematte_synth.pth',
                    map_location='cpu'
                ))
            
            self.segment_model.eval()
            
        self.lpips = LPIPS()
        self.psnr = PSNR()
        self.ssim = SSIM()
        self.mssim = MS_SSIM()
        self.val_outputs = [[], []]

    def forward(self, X_dict, use_geometric_augmentations=False):
        return self.gen(self.embedder(X_dict, use_geometric_augmentations=use_geometric_augmentations))

    def configure_optimizers(self):
        
        opt_G = torch.optim.Adam(list(self.embedder.parameters()) + list(self.gen.parameters()), lr=self.g_lr, betas=self.betas, eps=1e-5)
        opt_D = torch.optim.Adam(self.disc.parameters(), lr=self.d_lr, betas=self.betas, eps=1e-5)
        return opt_G, opt_D
    
    def calc_mask(self, batch):
        batch = dict(list(batch.items()))
        batch_shape = list(batch['face_wide'].shape)
        batch_shape[2] = 1
        device = batch['face_wide'].device
        dtype = batch['face_wide'].dtype
        mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype)[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype)[None, :, None, None]
        
        with torch.no_grad():
            normalized = (batch['face_wide'] + 1) / 2
            normalized = (normalized - mean) / std
            
            mask = self.segment_model(
                torch.flatten(normalized, start_dim=0, end_dim=1)
            ).reshape(*batch_shape)
        batch['face_wide_mask'] = mask
        
        return batch

    def training_step(self, train_batch, batch_idx):

        X_dict = make_X_dict(
            X_arc=train_batch['face_arc'],
            X_wide=train_batch['face_wide'],
            X_mask=train_batch['face_wide_mask'], # if self.segment_model is not None else None,
            X_emotion=train_batch['face_emoca'],
            X_keypoints=train_batch['keypoints'],
            segmentation=train_batch['segmentation']
        )
        
        opt_G, opt_D = self.optimizers()
        
        data_dict = self.forward(X_dict, use_geometric_augmentations=True)

        losses = self.aligner_loss(data_dict, X_dict, epoch=self.current_epoch)
        
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

        return data_dict

    def validation_step(self, val_batch, batch_idx, dataloader_idx):

        X_dict = make_X_dict(val_batch['face_arc'], val_batch['face_wide'],  val_batch['face_wide_mask'])
        
        with torch.no_grad():
            outputs = self.forward(X_dict)
        
        
        if dataloader_idx == 0:
            masked_output = blend_alpha(outputs['fake_rgbs'], X_dict['target']['face_wide_mask'])
            
            lpips_val = self.lpips(masked_output,  X_dict['target']['face_wide'])
            psnr_val = self.psnr(masked_output,  X_dict['target']['face_wide'])
            ssim_val = self.ssim(masked_output,  X_dict['target']['face_wide'])
            mssim_val = self.mssim(masked_output,  X_dict['target']['face_wide'])

            id_dict = self.aligner_loss.id_loss(outputs, X_dict, return_embeds=True)
            id_metric = F.cosine_similarity(id_dict['fake_embeds'], id_dict['real_embeds']).mean()

            metrics = {'LPIPS': lpips_val,
                           'PSNR': psnr_val,
                           'SSIM': ssim_val,
                           'MS_SSIM': mssim_val,
                           'ID self': id_metric}
            
        if dataloader_idx == 1:
            id_dict = self.aligner_loss.id_loss(outputs, X_dict, return_embeds=True)
            id_score = F.cosine_similarity(id_dict['fake_embeds'], id_dict['real_embeds']).mean()
            metrics = {'ID cross': id_score}
        
        out_dict =  {'fake_rgbs': outputs['fake_rgbs'],
                     'fake_segm': outputs['fake_segm'],
                     'metrics': metrics}
        
        if dataloader_idx == 0:
            self.val_outputs[0].append(out_dict)
        else:
            self.val_outputs[1].append(out_dict)
            
        return out_dict
        
    def on_validation_epoch_end(self):
        outputs = self.val_outputs
        self_metics = outputs[0] #self reenacment dataloader, list of dicts for epoch
        cross_metics = outputs[1] # cross reenacment
        
        keys_self = list(self_metics[0]['metrics'].keys())
        keys_cross = list(cross_metics[0]['metrics'].keys())
        
        losses_self = {key:[] for key in keys_self}
        losses_cross = {key:[] for key in keys_cross}
        
        for batch_n in range(len(outputs[0])):
            
            for key in keys_self:
                losses_self[key].append(self_metics[batch_n]['metrics'][key].item())
                
        for batch_n in range(len(outputs[1])):                              
            for key in keys_cross:
                losses_cross[key].append(cross_metics[batch_n]['metrics'][key].item())
                
        for key, val in losses_self.items():
            self.log(key, np.mean(val), sync_dist=True)
            
        for key, val in losses_cross.items():
            self.log(key, np.mean(val), sync_dist=True)

        self.val_outputs[0].clear()
        self.val_outputs[1].clear()
        


def create_dataset(cfg, train_transform=None, flip_transform=None, cross=False):
    dataset = Voxceleb2H5Dataset(root_path=cfg.data_path, source_len=cfg.source_len, transform=train_transform, flip_transform=flip_transform, shuffle=cfg.shuffle, cross=cross)
    sampler = CustomBatchSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler, num_workers=cfg.num_workers)
    return dataloader
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/aligner.yaml")
    args = parser.parse_args()
    
    with open(args.config, "r") as stream:
        cfg = OmegaConf.load(stream)
    
    model = AlignerModule(cfg)

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataloader = create_dataset(cfg.train_options, train_transform=train_transform, flip_transform=True, cross=False)
    val_dataloader_self = create_dataset(cfg.inference_options, cross=False)
    val_dataloader_cross = create_dataset(cfg.inference_options, cross=True)
        
    ts_logger = TensorBoardLogger('ts_logs_aligner/', name=cfg.experiment_name)
    log_pred_callback = LogPredictionSamplesCallback(ts_logger, n=2, log_train_freq=cfg.train_options.log_train_freq)
    checkpoint_callback = PeriodicCheckpoint(cfg.train_options.ckpt_interval, dir='{}/aligner_checkpoints/{}/checkpoints'.format(cfg.home_dir, cfg.experiment_name))


    trainer = pl.Trainer(
        max_epochs=cfg.train_options.max_epochs,
        accelerator='gpu', devices=cfg.num_gpus,
        log_every_n_steps=cfg.train_options.log_interval,
        logger=ts_logger, callbacks=[checkpoint_callback, log_pred_callback],
        strategy='ddp_find_unused_parameters_true',
        precision=16,
        num_sanity_val_steps=0
        )
    torch.set_float32_matmul_precision('medium')
    
    trainer.fit(model, train_dataloader, [val_dataloader_self, val_dataloader_cross])
