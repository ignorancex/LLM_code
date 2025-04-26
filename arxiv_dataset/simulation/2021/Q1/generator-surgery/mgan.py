import argparse
import os
import shutil
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from forward_model import GaussianCompressiveSensing
from model.began import Generator128
from utils import (get_z_vector, load_target_image, load_trained_net,
                   psnr_from_mse)

warnings.filterwarnings("ignore")


def _mgan_recover(x,
                  gen,
                  n_cuts,
                  forward_model,
                  optimizer_type='sgd',
                  mode='zero',
                  limit=1,
                  z_lr=1,
                  n_steps=2000,
                  z_number=20,
                  run_dir=None,
                  run_name=None,
                  disable_tqdm=False,
                  **kwargs):
    """
    Args:
        x - input image, torch tensor (C x H x W)
        gen - generator, already loaded with checkpoint weights
        forward_model - corrupts the image
        n_cuts - the intermediate layer to combine z vectors
        n_steps - number of optimization steps during recovery
        run_name - use None for no logging
    """

    z1_dim, _ = gen.input_shapes[0]
    _, z2_dim = gen.input_shapes[n_cuts]

    if (isinstance(forward_model, GaussianCompressiveSensing)):
        n_pixel_bora = 64 * 64 * 3
        n_pixel = np.prod(x.shape)
        noise = torch.randn(1, forward_model.n_measure, device=x.device)
        noise *= 0.1 * torch.sqrt(torch.tensor(n_pixel / forward_model.n_measure / n_pixel_bora))

    z1 = torch.nn.Parameter(get_z_vector((z_number, *z1_dim), mode=mode, limit=limit, device=x.device))
    alpha = torch.nn.Parameter(
        get_z_vector((z_number, gen.input_shapes[n_cuts][0][0]), mode=mode, limit=limit, device=x.device))
    params = [z1, alpha]
    if len(z2_dim) > 0:
        z2 = torch.nn.Parameter(get_z_vector((1, *z2_dim), mode=mode, limit=limit, device=x.device))
        params.append(z2)
    else:
        z2 = None

    if optimizer_type == 'sgd':
        optimizer_z = torch.optim.SGD(params, lr=z_lr)
        scheduler_z = None
        save_img_every_n = 50
    elif optimizer_type == 'adam':
        optimizer_z = torch.optim.Adam(params, lr=z_lr)
        scheduler_z = None
        # scheduler_z = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer_z, n_steps, 0.05 * z_lr)
        save_img_every_n = 50
    else:
        raise NotImplementedError()

    if run_name is not None:
        logdir = os.path.join('recovery_tensorboard_logs', run_dir, run_name)
        if os.path.exists(logdir):
            print("Overwriting pre-existing logs!")
            shutil.rmtree(logdir)
        writer = SummaryWriter(logdir)

    # Save original and distorted image
    if run_name is not None:
        writer.add_image("Original/Clamp", x.clamp(0, 1))
        if forward_model.viewable:
            writer.add_image("Distorted/Clamp", forward_model(x.unsqueeze(0).clamp(0, 1)).squeeze(0))

    # Recover image under forward model
    x = x.expand(1, *x.shape)
    y_observed = forward_model(x)
    if (isinstance(forward_model, GaussianCompressiveSensing)):
        y_observed += noise

    for j in trange(n_steps, leave=False, desc='Recovery', disable=disable_tqdm):

        optimizer_z.zero_grad()
        F_l = gen.forward(z1, None, n_cuts=0, end=n_cuts, **kwargs)
        F_l_2 = (F_l * alpha[:, :, None, None]).sum(0, keepdim=True)
        x_hats = gen.forward(F_l_2, z2, n_cuts=n_cuts, end=None, **kwargs)
        if gen.rescale:
            x_hats = (x_hats + 1) / 2
        train_mse = F.mse_loss(forward_model(x_hats), y_observed)
        train_mse.backward()
        optimizer_z.step()

        train_mse_clamped = F.mse_loss(forward_model(x_hats.detach().clamp(0, 1)), y_observed)
        orig_mse_clamped = F.mse_loss(x_hats.detach().clamp(0, 1), x)

        if run_name is not None and j == 0:
            writer.add_image('Start', x_hats.clamp(0, 1).squeeze(0))

        if run_name is not None:
            writer.add_scalar('TRAIN_MSE', train_mse_clamped, j + 1)
            writer.add_scalar('ORIG_MSE', orig_mse_clamped, j + 1)
            writer.add_scalar('ORIG_PSNR', psnr_from_mse(orig_mse_clamped), j + 1)

            if j % save_img_every_n == 0:
                writer.add_image('Recovered', x_hats.clamp(0, 1).squeeze(0), j + 1)

        if scheduler_z is not None:
            scheduler_z.step()

    if run_name is not None:
        writer.add_image('Final', x_hats.clamp(0, 1).squeeze(0))

    return x_hats.squeeze(0), forward_model(x)[0], train_mse_clamped


def mgan_recover(x,
                 gen,
                 n_cuts,
                 forward_model,
                 optimizer_type='sgd',
                 mode='zero',
                 limit=1,
                 z_lr=1,
                 n_steps=2000,
                 z_number=20,
                 restarts=1,
                 run_dir=None,
                 run_name=None,
                 disable_tqdm=False,
                 **kwargs):

    best_psnr = -float("inf")
    best_return_val = None

    for i in trange(restarts, desc='Restarts', leave=False, disable=disable_tqdm):
        if run_name is not None:
            current_run_name = f'{run_name}_{i}'
        else:
            current_run_name = None
        return_val = _mgan_recover(x=x,
                                   gen=gen,
                                   n_cuts=n_cuts,
                                   forward_model=forward_model,
                                   optimizer_type=optimizer_type,
                                   mode=mode,
                                   limit=limit,
                                   z_lr=z_lr,
                                   n_steps=n_steps,
                                   z_number=z_number,
                                   run_dir=run_dir,
                                   run_name=current_run_name,
                                   disable_tqdm=disable_tqdm,
                                   **kwargs)
        p = psnr_from_mse(return_val[2])
        if p > best_psnr:
            best_psnr = p
            best_return_val = return_val

    return best_return_val


if __name__ == '__main__':
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    a = argparse.ArgumentParser()
    a.add_argument('--img_dir', required=True)
    a.add_argument('--n_cuts', type=int, required=True)
    a.add_argument('--disable_tqdm', default=False)
    args = a.parse_args()

    gen = Generator128(64)
    gen = load_trained_net(gen, ('./checkpoints/celeba_began.withskips.bs32.cosine.min=0.25'
                                 '.n_cuts=0/gen_ckpt.49.pt'))
    gen = gen.eval().to(DEVICE)

    img_size = 128
    img_shape = (3, img_size, img_size)

    forward_model = GaussianCompressiveSensing(n_measure=20000, img_shape=img_shape)
    # forward_model = NoOp()

    for img_name in tqdm(os.listdir(args.img_dir), desc='Images', leave=True, disable=args.disable_tqdm):
        orig_img = load_target_image(os.path.join(args.img_dir, img_name), img_size).to(DEVICE)
        img_basename, _ = os.path.splitext(img_name)
        x_hat, x_degraded, _ = mgan_recover(orig_img,
                                            gen,
                                            n_cuts=args.n_cuts,
                                            forward_model=forward_model,
                                            run_dir='mgan_prior',
                                            run_name=img_basename)
