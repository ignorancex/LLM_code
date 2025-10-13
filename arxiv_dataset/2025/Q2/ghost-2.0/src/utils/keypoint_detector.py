import os
import torch
import torchvision
import numpy as np
import torch.nn as nn

import repos.DECA.decalib.utils.config as config
from repos.DECA.decalib.deca import DECA
from repos.DECA.decalib.utils import util
from repos.BlazeFace_PyTorch.blazeface import BlazeFace

class DECAKeypoints(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        cfg = config.cfg
        config.cfg.deca_dir = './repos/DECA/'
        
        cfg.model.topology_path = os.path.join(config.cfg.deca_dir, 'data', 'head_template.obj')
        cfg.model.addfiles_path = 'data'
        cfg.model.flame_model_path = os.path.join(cfg.deca_dir, 'data', 'generic_model.pkl')
        cfg.model.flame_lmk_embedding_path = os.path.join(cfg.deca_dir, 'data', 'landmark_embedding.npy')
        cfg.model.face_mask_path = os.path.join(cfg.deca_dir, 'data', 'uv_face_mask.png')
        cfg.model.face_eye_mask_path = os.path.join(cfg.deca_dir, 'data', 'uv_face_eye_mask.png')
        cfg.pretrained_modelpath = os.path.join(cfg.deca_dir, 'data', 'deca_model.tar')
        cfg.model.use_tex = False
        # cfg.dataset.image_size = 224
        # cfg.model.uv_size = 224
        cfg.rasterizer_type = 'pytorch3d'
        cfg.model.extract_tex = False
    
        self.deca = DECA(cfg, device)
        self.deca.eval()

        self.front_net = BlazeFace().to(device)
        self.front_net.load_weights("./repos/BlazeFace_PyTorch/blazeface.pth")
        self.front_net.load_anchors("./repos/BlazeFace_PyTorch/anchors.npy")
        self.front_net.eval()


    def forward(self, imgs):
        '''imgs is RGB image in range[0, 255]'''
        if not isinstance(imgs, torch.Tensor):
            imgs = torch.tensor(imgs.copy()).permute(0, 3, 1, 2).to(self.device)

        img_batch_128 = torch.nn.functional.interpolate(imgs, size=(128, 128))
        with torch.no_grad():
            front_detections = self.front_net.predict_on_batch(img_batch_128)
            
        front_detections = [det.to(self.device) for det in front_detections]    
        front_detections = torch.cat(front_detections, 0)
        front_detections = torch.stack([front_detections[:, 0] - 0.1, front_detections[:, 1] - 0.1, front_detections[:, 2] + 0.1, front_detections[:, 3] + 0.1], 1) 

        b, c, h, w = imgs.shape

        top = (front_detections[:, 0] * h).int()
        left = (front_detections[:, 1]  * w).int()
        height = ((front_detections[:, 2] - front_detections[:, 0]) * h).int()
        width = ((front_detections[:, 3] - front_detections[:, 1]) * w).int()

        cropped_imgs = []
        
        for i in range(imgs.size(0)):
            try:
                img_cropped = torchvision.transforms.functional.crop(imgs[i], top[i], left[i], height[i], width[i])
            except IndexError:
                img_cropped = imgs[i]
                
            img_cropped = torch.nn.functional.interpolate(img_cropped.unsqueeze(0) / img_cropped.max(), size=(224, 224))
            cropped_imgs.append(img_cropped)
            
        cropped_imgs = torch.cat(cropped_imgs)
        
        
        codedict = self.deca.encode(cropped_imgs)
        _, landmarks2d_pred, _ = self.deca.flame(shape_params=codedict['shape'], expression_params=codedict['exp'], pose_params=codedict['pose'])
        landmarks2d_pred = util.batch_orth_proj(landmarks2d_pred, codedict['cam'])[:,:,:2]; landmarks2d_pred[:,:,1:] = -landmarks2d_pred[:,:,1:]
        
        return landmarks2d_pred  / 2 + 0.5 #normalize to [0, 1]