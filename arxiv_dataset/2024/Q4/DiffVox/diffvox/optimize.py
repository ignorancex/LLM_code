import time

import torch
import wandb

from tqdm import tqdm
from .reconstruct import Reconstruction, TVLoss3D
from torchmetrics.regression import PearsonCorrCoef
from monai.losses import LocalNormalizedCrossCorrelationLoss
from monai.metrics import SSIMMetric, PSNRMetric


def optimize(dataloader, n_itr, lr, lr_tv, shift, loss_fn, drr_params, density_regulator, tv_type, drr_scale, beta, log_wandb):
    """main optimization loop"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        Warning("Using CPU, this will be slow and cause problems")
    print(f"Using device: {device}")

    recon = Reconstruction(dataloader.subject, device, dataloader.init_vol, drr_params, shift, density_regulator, beta)
    ## instead of applying the affine transformation to each batch do it all at once
    ## note: this is not possible with the new implementation of the drr renderer
    # dataloader.apply_function(recon.drr.affine_inverse)
    tv_calc = TVLoss3D(lr_tv, tv_type)
 

    optimizer = torch.optim.Adam(recon.parameters(), lr=lr)
    if loss_fn == "l1":
        criterion = torch.nn.L1Loss()
    elif loss_fn == "l2":
        criterion = torch.nn.MSELoss()
    elif loss_fn == "pcc":
        Warning("Using PCC loss, work in progress")
        criterion = PearsonCorrCoef()
    elif loss_fn == 'ncc':
        KeyError("NCC loss not implemented (yet)")
        criterion = LocalNormalizedCrossCorrelationLoss(spatial_dims=2)
    else:
        raise ValueError(f"Unrecognized loss function : {loss_fn}")
    
    
    subject_volume = dataloader.subject.volume.data.cuda().requires_grad_(False)
    max_val = (subject_volume).max()
    ssim_calc = SSIMMetric(3, max_val)
    psnr_calc = PSNRMetric(max_val)
    pcc_calc = PearsonCorrCoef().to(device)
    mse_calc = torch.nn.MSELoss()
    # ncc_calc = LocalNormalizedCrossCorrelationLoss(spatial_dims=3) # needs patch based rendering

    ## lr schedulers we've tried
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    # lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=n_itr, power=1.0) # starts at lr and decays to 0 over n_itr iterations
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=n_itr,pct_start=5/n_itr, anneal_strategy='linear')


    losses = []
    tvs = []
    ssims = []
    psnrs = []
    pccs = []
    time_deltas = []
    for itr in (pbar := tqdm(range(n_itr), ncols=100)):
        start_time = time.perf_counter()
        for source, target, gt in dataloader:
            optimizer.zero_grad()
            source = source.to(device)
            target = target.to(device)
            est = recon(source, target)
            tv_norm = tv_calc(recon.density[None, None])
            loss = criterion(drr_scale * est, gt.cuda()) + tv_norm
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            tvs.append(tv_norm.item())
        end_time = time.perf_counter()
        time_deltas.append(end_time - start_time)
        pbar.set_description(f"loss : {loss.item():.06f} tv : {tv_norm.item():06f}")
        lr_scheduler.step()
        with torch.no_grad():
            # ssim = ssim_calc(recon.density[None, None], subject_volume[None]) # expensive to calculate
            subject_volume = subject_volume.to('cuda')
            psnr = psnr_calc(recon.density[None, None], subject_volume[None])
            pcc = pcc_calc(recon.density.flatten(), subject_volume.flatten())
            mse = mse_calc(recon.density[None, None], subject_volume[None])
            subject_volume = subject_volume.cpu()
            # ncc = ncc_calc(recon.density[None, None], subject_volume[None]).cpu()
            # ssims.append(ssim.item()) # this is computationally expensive so we only calculate it in the end
            psnrs.append(psnr.item())
            pccs.append(pcc.item())
        if log_wandb:
            wandb.log({"loss": loss.item(), "tv_loss": tv_norm.item(), "psnr": psnr.item(), 'pcc': pcc.item(), 'vol_mse': mse, 'lr_decay': lr_scheduler.get_last_lr()[0]})
    with torch.no_grad():
        ssims.append(ssim_calc(recon.density[None, None], subject_volume[None].cuda()).item())
    return recon.density, losses, tvs, ssims, psnrs, pccs, time_deltas

