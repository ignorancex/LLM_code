import torch.nn as nn
import torch
import numpy as np
import healpy as hp

from .spherical_harmonic import _sh_matrix_sh2s
from .spherical_harmonic import _sh_matrix


class Losses(nn.Module):
    def __init__(self, loss_config, sh_degree, has_equi, has_inva, has_fodf, writer, compute_extra_loss, n_epoch, n_batch, prefix_dataset='train', verbose=False):
        super(Losses, self).__init__()
        if verbose:
            print('-'*50)
            print('-'*10, ' Create Losses ', '-'*10)
            print(f'Compute extra loss: {compute_extra_loss}')
            print(f'Number of epochs: {n_epoch}')
            print(f'Number of batches: {n_batch}')
        self.writer = writer
        self.writer_step = 0
        self.compute_extra_loss = compute_extra_loss
        self.n_epoch = n_epoch
        self.epoch = 0
        self.n_batch = n_batch
        self.batch = 0
        self.has_equi = has_equi
        self.has_inva = has_inva
        self.has_fodf = has_fodf
        ignore_losses = ['fodf_reconstruction' if not has_fodf else '']
        self.prefix_dataset = prefix_dataset
        # Reconstruction related losses
        self.losses = [Loss(loss_name=loss_name, verbose=verbose, prefix_dataset=self.prefix_dataset, **loss_config['reconstruction'][loss_name]) for loss_name in loss_config['reconstruction']]
        # Equivariant decomposition related losses
        if has_equi:
            self.equi_fodf_interpolator = fODFInterpolator(n_side=loss_config['equi']['non_negativity']['n_side_fodf_interpolation'], sh_degree=sh_degree, use_hemisphere=loss_config['equi']['non_negativity']['use_hemisphere'], use_legacy=False)
            self.equi_rf_interpolator = RFInterpolator(sh_degree=sh_degree)
            equivariant_losses = [Loss(loss_name=loss_name, prefix='equi', verbose=verbose, prefix_dataset=self.prefix_dataset, **loss_config['equi'][loss_name]) for loss_name in loss_config['equi'] if not loss_name in ignore_losses]
            self.losses += equivariant_losses
        # Invariant decomposition related losses
        if has_inva:
            invariant_losses = [Loss(loss_name=loss_name, prefix='inva', verbose=verbose, prefix_dataset=self.prefix_dataset, **loss_config['inva'][loss_name]) for loss_name in loss_config['inva'] if not loss_name in ignore_losses]
            self.losses += invariant_losses
        self.losses = nn.ModuleList(self.losses)
        if verbose:
            print('-'*50)

    def forward(self, **params):
        to_print = ""
        total_loss = 0
        if self.has_equi:
            params['equi_deconvolved_shc_normed'] = params['equi_deconvolved_shc'] / (torch.linalg.norm(params['equi_deconvolved_shc'], dim=2, keepdim=True) + 1e-16)
            params['equi_deconvolved'] = self.equi_fodf_interpolator(params['equi_deconvolved_shc'])
            params['equi_deconvolved_normed'] = params['equi_deconvolved'] / (torch.linalg.norm(params['equi_deconvolved'], dim=2, keepdim=True) + 1e-16)
            params['equi_polar_filter'] = self.equi_rf_interpolator(params['equi_polar_filter_shc'])
        if self.has_inva:
            params['inva_deconvolved'] = params['inva_deconvolved_shc'] / np.sqrt(4*np.pi)
        for loss in self.losses:
            if loss.weight>0 or self.compute_extra_loss:
                loss_value, add_to_print = loss(self.writer, **params)
                if loss.weight>0:
                    total_loss += loss_value * loss.weight
                to_print += ', ' + add_to_print
        to_print = f"Epoch [{self.epoch + 1}/{self.n_epoch}], Iter [{self.batch + 1}/{self.n_batch}]: Loss: {total_loss.item():.10f}" + to_print
        self.writer.add_scalar(f'Total/{self.prefix_dataset}/Batch', total_loss.item(), self.writer_step)
        self.writer_step += 1
        self.batch += 1
        return total_loss, to_print

    def end_epoch(self):
        to_print = ""
        total_loss = 0
        for loss in self.losses:
            if loss.weight>0 or self.compute_extra_loss:
                loss_value, add_to_print = loss.end_epoch(self.writer)
                if loss.weight>0:
                    total_loss += loss_value * loss.weight
                to_print += ', ' + add_to_print
        to_print = f"Epoch [{self.epoch + 1}/{self.n_epoch}]: Loss: {total_loss:.10f}" + to_print
        self.writer.add_scalar(f'Total/{self.prefix_dataset}/Epoch', total_loss, self.epoch)
        self.epoch += 1
        self.batch = 0
        return total_loss, to_print


class Loss(nn.Module):
    def __init__(self, loss_name, prefix=None, prefix_dataset='train', verbose=False, **params):
        super(Loss, self).__init__()
        # List of implemented losses
        available_losses = {'intensity': ReconstructionLoss,
                            'non_negativity': NonNegativityLoss,
                            'sparsity': SparsityLoss,
                            'total_variation': TotalVariationLoss,
                            'gfa': GFALoss,
                            'pve': PVENormLoss,
                            'prior_rf': PriorRFLoss,
                            'nn_rf': NNRFLoss,
                            'fodf_reconstruction': FodfReconstructionLoss}
        # Check if loss is implemented
        if loss_name not in available_losses.keys():
            raise NotImplementedError(f'Expected one of {available_losses.keys()} but got {loss_name}')
        if verbose:
            print('-'*1, f"Using {f'{prefix}_'*(not prefix is None)}{loss_name} loss")
            print(f"Parameters: {params}")
        # Get loss function
        self.loss_function = available_losses[loss_name](prefix=prefix, **params)
        # Get weight of the loss
        self.weight = params['weight']
        # Get saving name
        self.loss_name = f"{f'{prefix}_'*(not prefix is None)}{loss_name}"
        # Initialize loss memory
        self.loss_memory = 0
        self.n_batch = 0
        self.epoch = 0
        self.writer_step = 0
        self.prefix_dataset = prefix_dataset

    def forward(self, writer, **params):
        loss = self.loss_function(**params)
        self.loss_memory += loss.item()
        self.n_batch += 1
        to_print = f"{self.loss_name}: {loss.item():.10f}"
        writer.add_scalar(f'{self.loss_name}/{self.prefix_dataset}/Batch', loss.item(), self.writer_step)
        self.writer_step += 1
        return loss, to_print

    def end_epoch(self, writer):
        loss = self.loss_memory / self.n_batch
        self.loss_memory = 0
        self.n_batch = 0
        to_print = f'{self.loss_name}: {loss:.10f}'
        writer.add_scalar(f'{self.loss_name}/{self.prefix_dataset}/Epoch', loss, self.epoch)
        self.epoch += 1
        return loss, to_print

class ReconstructionLoss(nn.Module):
    def __init__(self, prefix=None, **params):
        super(ReconstructionLoss, self).__init__()
        self.prefix = prefix
        self.norm = Norm(**params)

    def forward(self, **params):
        reconstruction, target, mask = params[f"{f'{self.prefix}_'*(not self.prefix is None)}reconstruction"], params['target'], params['mask']
        diff = reconstruction - target
        loss = self.norm(diff, mask)
        return loss

class NonNegativityLoss(nn.Module):
    def __init__(self, prefix=None, **params):
        super(NonNegativityLoss, self).__init__()
        self.prefix = prefix
        self.norm = Norm(**params)

    def forward(self, **params):
        deconvolved, mask = params[f"{f'{self.prefix}_'*(not self.prefix is None)}deconvolved"], params['mask']
        fodf_neg = torch.min(deconvolved, torch.zeros_like(deconvolved))
        loss = self.norm(fodf_neg, mask.unsqueeze(1))
        return loss

class SparsityLoss(nn.Module):
    def __init__(self, prefix=None, **params):
        super(SparsityLoss, self).__init__()
        self.prefix = prefix
        self.norm = Norm(**params)

    def forward(self, **params):
        deconvolved, mask = params[f"{f'{self.prefix}_'*(not self.prefix is None)}deconvolved"], params['mask']
        loss = self.norm(deconvolved, mask.unsqueeze(1))
        return loss

class GFALoss(nn.Module):
    def __init__(self, prefix=None, **params):
        super(GFALoss, self).__init__()
        self.prefix = prefix
        self.norm = Norm(**params)

    def forward(self, **params):
        deconvolved, mask = params[f"{f'{self.prefix}_'*(not self.prefix is None)}deconvolved"], params['mask']
        rms_square = torch.mean(deconvolved**2, dim=2)
        std_square = torch.var(deconvolved, dim=2)
        loss = self.norm(1 - std_square / (rms_square + 1e-16), mask)
        return loss

class TotalVariationLoss(nn.Module):
    def __init__(self, prefix=None, **params):
        super(TotalVariationLoss, self).__init__()
        self.prefix = prefix
        self.norm = Norm(**params)
        self.use_shc = params['use_shc']
        self.use_normed = params['use_normed']
        self.input_name = f"{f'{self.prefix}_'*(not self.prefix is None)}deconvolved{'_shc'*self.use_shc}{'_normed'*self.use_normed}"
        print(self.input_name )

    def forward(self, **params):
        input, mask = params[self.input_name ], params['mask']
        # h diff
        diff = input[:, :, :, 1:-1] - input[:, :, :, 2:]
        loss = self.norm(diff, mask[:, :, 1:-1].unsqueeze(1))
        diff = input[:, :, :, 1:-1] - input[:, :, :, :-2]
        loss += self.norm(diff, mask[:, :, 1:-1].unsqueeze(1))
        # w diff
        diff = input[:, :, :, :, 1:-1] - input[:, :, :, :, 2:]
        loss += self.norm(diff, mask[:, :, :, 1:-1].unsqueeze(1))
        diff = input[:, :, :, :, 1:-1] - input[:, :, :, :, :-2]
        loss += self.norm(diff, mask[:, :, :, 1:-1].unsqueeze(1))
        # d diff
        diff = input[:, :, :, :, :, 1:-1] - input[:, :, :, :, :, 2:]
        loss += self.norm(diff, mask[:, :, :, :, 1:-1].unsqueeze(1))
        diff = input[:, :, :, :, :, 1:-1] - input[:, :, :, :, :, :-2]
        loss += self.norm(diff, mask[:, :, :, :, 1:-1].unsqueeze(1))
        return 1/6 * loss

class PVENormLoss(nn.Module):
    def __init__(self, prefix=None, **params):
        super(PVENormLoss, self).__init__()
        self.prefix = prefix
        self.norm = Norm(**params)

    def forward(self, **params):
        deconvolved_shc, mask = params[f"{f'{self.prefix}_'*(not self.prefix is None)}deconvolved_shc"], params['mask']
        loss = self.norm(deconvolved_shc[:, :, 0] * np.sqrt(4*np.pi), mask)
        return loss

class PriorRFLoss(nn.Module):
    def __init__(self, prefix=None, **params):
        super(PriorRFLoss, self).__init__()
        self.prefix = prefix
        self.norm = Norm(**params)

    def forward(self, **params):
        polar_filter_shc, target_polar_filter_shc = params[f"{f'{self.prefix}_'*(not self.prefix is None)}polar_filter_shc"], params[f"{f'{self.prefix}_'*(not self.prefix is None)}target_polar_filter_shc"]
        diff = polar_filter_shc - target_polar_filter_shc
        loss = self.norm(diff)
        return loss

class NNRFLoss(nn.Module):
    def __init__(self, prefix=None, **params):
        super(NNRFLoss, self).__init__()
        self.prefix = prefix
        self.norm = Norm(**params)
        
    def forward(self, **params):
        polar_filter = params[f"{f'{self.prefix}_'*(not self.prefix is None)}polar_filter"]
        polar_filter_neg = polar_filter[:, :, 1:] - polar_filter[:, :, :-1]
        polar_filter_neg = torch.min(polar_filter_neg, torch.zeros_like(polar_filter_neg))
        loss = self.norm(polar_filter_neg)
        return loss
    
class FodfReconstructionLoss(nn.Module):
    def __init__(self, prefix=None, **params):
        super(FodfReconstructionLoss, self).__init__()
        self.prefix = prefix
        self.norm = Norm(**params)
        
    def forward(self, **params):
        fodf, target_fodf, mask = params[f"{f'{self.prefix}_'*(not self.prefix is None)}deconvolved_shc"], params[f"{f'{self.prefix}_'*(not self.prefix is None)}deconvolved_shc_target"], params['mask']
        n_shc_min = min(fodf.shape[2], target_fodf.shape[2])
        diff = fodf[:, :, :n_shc_min] - target_fodf[:, :, :n_shc_min]
        loss = self.norm(diff, mask.unsqueeze(1))
        return loss


class fODFInterpolator(nn.Module):
    def __init__(self, n_side=16, sh_degree=18, use_hemisphere=True, use_legacy=False):
        super(fODFInterpolator, self).__init__()
        npix = hp.nside2npix(n_side)
        indexes = np.arange(npix)
        x, y, z = hp.pix2vec(n_side, indexes, nest=True)
        coords = np.stack([x, y, z], axis=1)
        SH2S = _sh_matrix_sh2s(sh_degree, coords)
        self.register_buffer('SH2S', torch.Tensor(SH2S))

    def forward(self, x):
        min_n_shc = min(x.shape[2], self.SH2S.shape[0])
        x = torch.einsum('btcxyz,cv->btvxyz', x[:, :, :min_n_shc], self.SH2S[:min_n_shc])
        return x

class RFInterpolator(nn.Module):
    def __init__(self, sh_degree=18):
        super(RFInterpolator, self).__init__()
        theta = np.arange(0, np.pi/2, (np.pi/2)/100)
        x = np.sin(theta)
        z = np.cos(theta)
        vec = np.stack([x, np.zeros_like(x), z], axis=-1)
        _, sh2s_rf_reg = _sh_matrix(sh_degree, vec, with_order=0, symmetric=True)  
        self.register_buffer('SH2S', torch.Tensor(sh2s_rf_reg))

    def forward(self, x):
        min_n_shc = min(x.shape[2], self.SH2S.shape[0])
        x = torch.einsum('tsc,cv->tsv', x[:, :, :min_n_shc], self.SH2S[:min_n_shc])
        return x

class Norm(torch.nn.Module):
    def __init__(self, norm_name, **params):
        """
        Parameters
        ----------
        norm_name : str
            Name of the loss.
        sigma : float
            Hyper parameter of the loss.
        """
        super(Norm, self).__init__()
        norm_name = norm_name.lower()
        available_norms = {'l2': L2, 'l1': L1, 'cauchy': Cauchy, 'welsch': Welsch, 'geman': Geman}
        if norm_name not in available_norms.keys():
            raise NotImplementedError(f'Expected one of {available_norms.keys()} but got {norm_name}')
        self.norm_name = norm_name
        self.norm = available_norms[norm_name](**params)

    def forward(self, input, mask=None):
        """
        Parameters
        ----------
        img1 : torch.Tensor
            Prediction tensor
        img2 : torch.Tensor
            Ground truth tensor
        wts: torch.nn.Parameter
            If specified, the weight of the grid.
        Returns
        -------
         loss : torch.Tensor
            Loss of the predicted tensor
        """
        # If provided, use mask
        if not mask is None:
            mask = mask.expand(input.size())
            input = input[mask>0]
        loss = self.norm(input)
        loss = loss.mean()
        return loss
    

class L2(torch.nn.Module):
    def __init__(self, **params):
        super(L2, self).__init__()

    def forward(self, input):        
        return input**2

class L1(torch.nn.Module):
    def __init__(self, **params):
        super(L1, self).__init__()

    def forward(self, input):        
        return torch.abs(input)

class Cauchy(torch.nn.Module):
    def __init__(self, **params):
        super(Cauchy, self).__init__()
        self.sigma = float(params['sigma'])

    def forward(self, input):        
        return 2 * torch.log(1 + ((input)**2 / (2*self.sigma)))

class Welsch(torch.nn.Module):
    def __init__(self, **params):
        super(Welsch, self).__init__()
        self.sigma = float(params['sigma'])

    def forward(self, input):        
        return 2 * (1-torch.exp(-0.5 * ((input)**2 / self.sigma)))

class Geman(torch.nn.Module):
    def __init__(self, **params):
        super(Geman, self).__init__()
        self.sigma = float(params['sigma'])

    def forward(self, input):        
        return 2 * (2*((input)**2 / self.sigma) / ((input)**2 / self.sigma + 4))