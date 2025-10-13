import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.preprocess import calc_arcface_borders

class ArcFaceLoss(nn.Module):
    def __init__(self, id_encoder, image_size=512):
        super(ArcFaceLoss, self).__init__()
        self.id_encoder = id_encoder
        self.crop_corners = calc_arcface_borders(image_size)
    
    def forward(self, data_dict, X_dict, return_embeds=False):

        cropped_fake_rgbs = data_dict['fake_rgbs']
        cropped_fake_rgbs = (
            cropped_fake_rgbs[:, :, self.crop_corners[0, 1]:self.crop_corners[1, 1],
                                    self.crop_corners[0, 0]:self.crop_corners[1, 0]]
        )
        cropped_fake_rgbs = F.interpolate(cropped_fake_rgbs, size=(112, 112))
        cropped_real_rgbs = X_dict['target']['face_arc']

        cropped_fake_rgbs = cropped_fake_rgbs[:, [2, 1, 0], :, :]
        cropped_real_rgbs = cropped_real_rgbs[:, [2, 1, 0], :, :]
        
        fake_embeds, fake_features = self.id_encoder(cropped_fake_rgbs, return_features=True)
        with torch.no_grad():
            real_embeds, real_features = self.id_encoder(cropped_real_rgbs.detach(), return_features=True)
        
        if return_embeds:
            return {
            'fake_embeds': fake_embeds,
            'real_embeds': real_embeds
        }
        
        L_perc_id = sum(F.l1_loss(fake_x, real_x) for fake_x, real_x in zip(fake_embeds, real_embeds))
        L_id = 1 - F.cosine_similarity(fake_embeds, real_embeds).mean()
        return {
            'L_perc_id': L_perc_id,
            'L_id': L_id
        }