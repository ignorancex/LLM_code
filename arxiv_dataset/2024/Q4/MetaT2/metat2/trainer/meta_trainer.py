import torch
import time
import os
import nibabel as nib
import numpy as np
from monai import transforms

# DMT Model
from metat2.model.dmt.models.dmt_model import DMTModel
from metat2.model.dmt.dmt_utils.script_util import create_gaussian_diffusion
from metat2.model.ddpm.script_util import create_model_and_diffusion
from functools import partial

# UNet Model
from metat2.model.unet.unet import UNet
from monai import losses
import torch.optim as optim

class MetaTrainer:
    def __init__(self, config):

        self.config = config
        self.device = self.set_device()

        self.trans_model = DMTModel(self.config).to(self.device)
        self.trans_optimizer = self.trans_model.create_optimizers(self.config)
        self.denoiser = self.init_denoiser()
        self.old_lr = self.config.lr

        self.seg_model = UNet(ch_in=2).to(self.device)
        self.seg_criterion = losses.DiceFocalLoss(include_background = False)
        self.seg_optimizer = optim.Adam(self.seg_model.parameters(), lr=self.config.seg_lr)

        # init model
        if self.config.isTrain and self.config.pretrained_netG:
            self.trans_model.netG.load_state_dict(torch.load(self.config.pretrained_netG))
        if self.config.isTrain and self.config.pretrained_unet:
            self.seg_model.load_state_dict(torch.load(self.config.pretrained_unet))

    def set_device(self):
        if torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_id = self.config.gpu_ids
            gpu_type = torch.cuda.get_device_name(gpu_id)
            print(f'>>> Using GPU {gpu_type}')
        else:
            device = torch.device('cpu')
            print('>>> Using CPU')
        return device

    def init_denoiser(self):

        denoise_step = self.config.denoise_step
        timestep_t = self.config.timestep_t
        num_timesteps = self.config.num_timesteps
        assert timestep_t % denoise_step == 0
        skip = timestep_t // denoise_step
        assert num_timesteps % skip == 0
        timestep_respacing = f'ddim{num_timesteps // skip}'
        respaced_timestep_t = timestep_t // skip
        diffuser = create_gaussian_diffusion(timestep_respacing=timestep_respacing)
        unet, _ = create_model_and_diffusion()
        state = torch.load(self.config.diffusion_path, map_location='cpu')
        unet.load_state_dict(state)
        if self.config.use_fp16:
            unet.convert_to_fp16()
        unet.cuda().eval()
        denoiser = partial(diffuser.ddim_sample_from_t_loop,
                           model=unet,
                           timestep_t=respaced_timestep_t)
        return denoiser
    
    def train_seg(self, data):
        # translation frozen, train segmentation
        self.trans_model.eval()
        self.seg_model.train()
        
        t2 = data['label'].to(self.device) #B,C,H,W
        dwi = data['image'].to(self.device)
        lesion = data['lesion'].to(self.device)
            
        noisy_syn_dwi = self.trans_model(data, mode='inference') # data intensity [-1,1]
        syn_dwi = self.denoiser(x_t=noisy_syn_dwi) # syn_dwi intensity [-1,1]

        train_images = torch.cat((t2, syn_dwi), dim=1) # train_images intensity [-1,1]
        train_images = transforms.ScaleIntensity(minv=0.0, maxv=1.0, channel_wise=True)(train_images) # train_images intensity [0,1]
        
        self.seg_optimizer.zero_grad()
        preds = self.seg_model(train_images)
        loss = self.seg_criterion(preds, lesion)
        loss.backward()
        self.seg_optimizer.step()

        return loss.item()

    def train_trans(self, data):
        # translation training, segmentation frozen
        self.trans_model.train()
        self.seg_model.eval()

        t2 = data['label'].to(self.device) #B,C,H,W
        dwi = data['image'].to(self.device)
        lesion = data['lesion'].to(self.device)

        self.trans_optimizer.zero_grad()
        trans_losses, syn_dwi = self.trans_model(data, mode='generator')

        test_images = torch.cat((t2, syn_dwi), dim=1)
        test_images = transforms.ScaleIntensity(minv=0.0, maxv=1.0, channel_wise=True)(test_images) # train_images intensity [0,1]

        with torch.no_grad():
            preds = self.seg_model(test_images)
        
        trans_losses = sum(trans_losses.values()).mean()
        seg_losses = self.seg_criterion(preds, lesion)
        loss = 0.5*seg_losses + 0.5*trans_losses

        loss.backward()
        self.trans_optimizer.step()

        return loss.item()

    def train_one_step(self, train_data, vali_data):

        seg_loss = self.train_seg(vali_data)
        trans_loss = self.train_trans(train_data)

        return seg_loss, trans_loss

    def inference(self, data, epoch):

        self.trans_model.eval()
        self.seg_model.eval()

        t2 = data['label'].to(self.device) #B,C,H,W
        dwi = data['image'].to(self.device)
        lesion = data['lesion'].to(self.device)

        noisy_syn_dwi = self.trans_model(data, mode='inference')
        syn_dwi = self.denoiser(x_t=noisy_syn_dwi)

        test_images = torch.cat((t2, syn_dwi), dim=1)
        test_images = transforms.ScaleIntensity(minv=0.0, maxv=1.0, channel_wise=True)(test_images) # train_images intensity [0,1]

        with torch.no_grad():
            preds = self.seg_model(test_images)
            binary_preds = (preds > 0.5).float()

            dsc = self.calculate_dsc(preds, lesion).tolist()
            # dhd = self.calculate_dhd(binary_preds, lesion).tolist()

            if self.config.save_inference_results:
                self.save_inference_images(binary_preds, data, epoch)

        return dsc

    def update_trans_learning_rate(self, epoch):
        if epoch > self.config.niter:
            lrd = self.config.lr / self.config.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.config.no_TTUR:
                new_lr_G = new_lr
            else:
                new_lr_G = new_lr / 2

            for param_group in self.trans_optimizer.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr

    def save_checkpoint(self, name, epoch):
        print(f'>>> Saving model {name} at epoch {epoch}')
        t = time.localtime()
        timestamp = time.strftime('%b-%d-%Y-%H%M', t)
        checkpoint = {
            'epoch': epoch,
            'trans': self.trans_model.netG.state_dict(),
            'seg': self.seg_model.state_dict(),
        }
        file_name = os.path.join(self.config.meta_model_save_path,f'{timestamp}_{epoch}_{name}.pth')
        torch.save(checkpoint, file_name)
        return file_name
    
    def calculate_dsc(self, inputs, targets, smooth=1e-6):
        '''
        y_pred, y_true -> [B, C, H, W]
        '''
        intersection = (inputs * targets).sum(dim=(2, 3))
        dice_score = (2. * intersection + smooth) / (inputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + smooth)

        return dice_score.squeeze(1)

    def save_inference_images(self, images, data, epoch):
        paths = [os.path.basename(path) for path in data['path']]
        os.makedirs(os.path.join(self.config.inference_save_path, str(epoch)), exist_ok=True)
        for i in range(images.shape[0]):
            sample = images[i]
            sample = sample.unsqueeze(-1)  # [H, W, 1]            
            sample_np = sample.cpu().numpy()            
            nifti_img = nib.Nifti1Image(sample_np, affine=np.eye(4))            
            nib.save(nifti_img, os.path.join(self.config.inference_save_path, str(epoch), paths[i]))
