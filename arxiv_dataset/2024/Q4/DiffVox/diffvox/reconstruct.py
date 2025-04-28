import torch
from diffdrr.drr import DRR
import torch.nn.functional as F

class Reconstruction(torch.nn.Module):
    """
    Main reconstruction module
    """
    def __init__(self, subject, device, initialize_vol, drr_params: dict, shift=0, density_regulator='sigmoid', beta=20):
        super().__init__()
        self.drr_params = drr_params
        self.shift = shift
        self.beta = beta
        if initialize_vol is None:
            self._density = torch.nn.Parameter(torch.zeros(*subject.volume.shape, device=device)[0])
        else:
            self._density = torch.nn.Parameter(torch.tensor(softplus_inv(initialize_vol.detach()), device=device))
            del initialize_vol
            
        self.drr = DRR(
            subject,
            sdd=drr_params['sdd'],
            height=drr_params['height'],
            width=drr_params['width'],
            delx=drr_params['delx'],
            renderer=drr_params['renderer'],
            patch_size=drr_params['patch_size'],
        ).to(device)
        
        if density_regulator == 'None':
            self.density_regulator = lambda x: x
        elif density_regulator == 'softplus':
            self.density_regulator = lambda x: torch.nn.functional.softplus(x - shift, beta=self.beta, threshold=20)
        elif density_regulator == 'sigmoid':
            self.density_regulator = lambda x: torch.sigmoid(x - shift)

    def forward(self, source, target, **kwargs):
        img = (target - source).norm(dim=-1).unsqueeze(1)
        source = self.drr.affine_inverse(source)
        target = self.drr.affine_inverse(target)
        if self.drr_params['renderer'] == 'trilinear':
            kwargs['n_points'] = self.drr_params['n_points']
        img = self.drr.renderer(
            self.density,
            source,
            target,
            img,
            **kwargs,
        )

        return img
    
    @property
    def density(self):
        return self.density_regulator(self._density - 0.3)
    
    @density.setter
    def density(self, value):
        # inverting the softplus function
        self._density.data = softplus_inv(value, self.shift)


def softplus_inv(x, shift, eps=1e-6):
    return (x * shift).expm1().clamp_min(eps).log() / shift


class TVLoss3D(torch.nn.Module):
    """
    Total variation loss for 3D data
    """
    def __init__(self, TVLoss_weight=1, norm='l2'):
        super(TVLoss3D, self).__init__()
        self.TVLoss_weight = TVLoss_weight
        self.norm = norm

    def forward(self, x):
        if self.norm == 'None':
            return 0
        batch_size = x.size()[0]
        d_x = x.size()[2]
        h_x = x.size()[3]
        w_x = x.size()[4]
        
        count_d = self._tensor_size(x[:,:,1:,:,:])
        count_h = self._tensor_size(x[:,:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,:,1:])
        
        if self.norm == 'l2':
            d_tv = torch.pow((x[:,:,1:,:,:] - x[:,:,:d_x-1,:,:]), 2).sum()
            h_tv = torch.pow((x[:,:,:,1:,:] - x[:,:,:,:h_x-1,:]), 2).sum()
            w_tv = torch.pow((x[:,:,:,:,1:] - x[:,:,:,:,:w_x-1]), 2).sum()
            return self.TVLoss_weight * 3 * (d_tv/count_d + h_tv/count_h + w_tv/count_w) / batch_size


        elif self.norm == 'l1':
            d_tv = torch.sum(torch.abs(x[:,:,1:,:,:] - x[:,:,:d_x-1,:,:]))
            h_tv = torch.sum(torch.abs(x[:,:,:,1:,:] - x[:,:,:,:h_x-1,:]))
            w_tv = torch.sum(torch.abs(x[:,:,:,:,1:] - x[:,:,:,:,:w_x-1]))
            return self.TVLoss_weight * 3 * (d_tv/count_d + h_tv/count_h + w_tv/count_w) / batch_size
            
        elif self.norm == 'nl1':
            loss = F.smooth_l1_loss(x[:,:,1:,:,:], x[:,:,:-1,:,:], reduction='mean').double() +\
                   F.smooth_l1_loss(x[:,:,:,1:,:], x[:,:,:,:-1,:], reduction='mean').double() +\
                   F.smooth_l1_loss(x[:,:,:,:,1:], x[:,:,:,:,:-1], reduction='mean').double()
            loss /= 3
            return self.TVLoss_weight * loss / batch_size
        
        elif self.norm == 'vl1': # same as l1, but consistent with vivek's implementation
            delx = x.diff(dim=-3).abs().mean()
            dely = x.diff(dim=-2).abs().mean()
            delz = x.diff(dim=-1).abs().mean()
            return self.TVLoss_weight * (delx + dely + delz) / 3 
        
        elif self.norm == 'vl2':  # Same as L2, but consistent with Vivek's implementation
            delx = x.diff(dim=-3).pow(2).mean()
            dely = x.diff(dim=-2).pow(2).mean()
            delz = x.diff(dim=-1).pow(2).mean()
            return self.TVLoss_weight * (delx + dely + delz) / 3

    
    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3] * t.size()[4] 

