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
from model.dcgan import Generator as dcgan_generator
from model.vae import VAE
from utils import (dict_to_str, get_z_vector, load_target_image,
                   load_trained_net, psnr_from_mse)

warnings.filterwarnings("ignore")


def _iagan_recover(
        x,
        gen,
        forward_model,
        optimizer_type='adam',
        mode='clamped_normal',
        limit=1,
        z_lr1=1e-4,
        z_lr2=1e-5,
        model_lr=1e-5,
        z_steps1=1600,
        z_steps2=3000,
        run_dir=None,  # IAGAN
        run_name=None,  # datetime or config
        disable_tqdm=False,
        **kwargs):

    # Keep batch_size = 1
    batch_size = 1

    z1_dim, z2_dim = gen.input_shapes[0]  # n_cuts = 0

    if (isinstance(forward_model, GaussianCompressiveSensing)):
        n_pixel_bora = 64 * 64 * 3
        n_pixel = np.prod(x.shape)
        noise = torch.randn(batch_size,
                            forward_model.n_measure,
                            device=x.device)
        noise *= 0.1 * torch.sqrt(
            torch.tensor(n_pixel / forward_model.n_measure / n_pixel_bora))

    # z1 is the actual latent code.
    # z2 is the additional input for n_cuts logic (not used here)
    z1 = torch.nn.Parameter(
        get_z_vector((batch_size, *z1_dim),
                     mode=mode,
                     limit=limit,
                     device=x.device))
    params = [z1]
    if len(z2_dim) > 0:
        z2 = torch.nn.Parameter(
            get_z_vector((batch_size, *z2_dim),
                         mode=mode,
                         limit=limit,
                         device=x.device))
        params.append(z2)
    else:
        z2 = None

    if optimizer_type == 'adam':
        optimizer_z = torch.optim.Adam([z1], lr=z_lr1)
        optimizer_model = torch.optim.Adam(gen.parameters(), lr=model_lr)
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
            writer.add_image(
                "Distorted/Clamp",
                forward_model(x.unsqueeze(0).clamp(0, 1)).squeeze(0))

    # Make noisy gaussian measurements
    x = x.expand(batch_size, *x.shape)
    y_observed = forward_model(x)
    if (isinstance(forward_model, GaussianCompressiveSensing)):
        y_observed += noise

    # Stage 1: optimize latent code only
    save_img_every_n = 50
    for j in trange(z_steps1, desc='Stage1', leave=False):
        optimizer_z.zero_grad()
        x_hat = gen.forward(z1, z2, n_cuts=0, **kwargs)
        if gen.rescale:
            x_hat = (x_hat + 1) / 2
        train_mse = F.mse_loss(forward_model(x_hat), y_observed)
        train_mse.backward()
        optimizer_z.step()

        train_mse_clamped = F.mse_loss(
            forward_model(x_hat.detach().clamp(0, 1)), y_observed)

        orig_mse_clamped = F.mse_loss(x_hat.detach().clamp(0, 1), x)

        if run_name is not None and j == 0:
            writer.add_image('Stage1/Start', x_hat.squeeze().clamp(0, 1))

        if run_name is not None:
            writer.add_scalar('Stage1/TRAIN_MSE', train_mse_clamped, j + 1)
            writer.add_scalar('Stage1/ORIG_MSE', orig_mse_clamped, j + 1)
            writer.add_scalar('Stage1/ORIG_PSNR',
                              psnr_from_mse(orig_mse_clamped), j + 1)

            if j % save_img_every_n == 0:
                writer.add_image('Stage1/Recovered',
                                 x_hat.squeeze().clamp(0, 1), j + 1)

    if run_name is not None:
        writer.add_image('Stage1_Final', x_hat.squeeze().clamp(0, 1))

    # Stage 2: optimize latent code and model
    save_img_every_n = 20
    optimizer_z = torch.optim.Adam([z1], lr=z_lr2)
    for j in trange(z_steps2, desc='Stage2', leave=False):
        optimizer_z.zero_grad()
        optimizer_model.zero_grad()
        x_hat = gen.forward(z1, z2, n_cuts=0, **kwargs)
        if gen.rescale:
            x_hat = (x_hat + 1) / 2
        train_mse = F.mse_loss(forward_model(x_hat), y_observed)
        train_mse.backward()
        optimizer_z.step()
        optimizer_model.step()

        train_mse_clamped = F.mse_loss(
            forward_model(x_hat.detach().clamp(0, 1)), y_observed)

        orig_mse_clamped = F.mse_loss(x_hat.detach().clamp(0, 1), x)

        if run_name is not None and j == 0:
            writer.add_image('Stage2/Start', x_hat.squeeze().clamp(0, 1))

        if run_name is not None:
            writer.add_scalar('Stage2/TRAIN_MSE', train_mse_clamped, j + 1)
            writer.add_scalar('Stage2/ORIG_MSE', orig_mse_clamped, j + 1)
            writer.add_scalar('Stage2/ORIG_PSNR',
                              psnr_from_mse(orig_mse_clamped), j + 1)

            if j % save_img_every_n == 0:
                writer.add_image('Stage2/Recovered',
                                 x_hat.squeeze().clamp(0, 1), j + 1)

    if run_name is not None:
        writer.add_image('Stage2_Final', x_hat.squeeze().clamp(0, 1))

    return x_hat.squeeze(), forward_model(x).squeeze(), train_mse_clamped


def iagan_recover(x,
                  gen,
                  forward_model,
                  optimizer_type,
                  mode='clamped_normal',
                  limit=1,
                  z_lr1=1e-4,
                  z_lr2=1e-5,
                  model_lr=1e-5,
                  z_steps1=1600,
                  z_steps2=3000,
                  restarts=1,
                  run_dir=None,
                  run_name=None,
                  disable_tqdm=False,
                  **kwargs):

    best_psnr = -float("inf")
    best_return_val = None

    for i in trange(restarts,
                    desc='Restarts',
                    leave=False,
                    disable=disable_tqdm):
        if run_name is not None:
            current_run_name = f'{run_name}_{i}'
        else:
            current_run_name = None
        return_val = _iagan_recover(x=x,
                                    gen=gen,
                                    forward_model=forward_model,
                                    optimizer_type=optimizer_type,
                                    mode=mode,
                                    limit=limit,
                                    z_lr1=z_lr1,
                                    z_lr2=z_lr2,
                                    model_lr=model_lr,
                                    z_steps1=z_steps1,
                                    z_steps2=z_steps2,
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
    a.add_argument('--disable_tqdm', default=False)
    a.add_argument('--run_name_suffix', default='')
    args = a.parse_args()

    params = {
        'z_lr1': 1e-4,
        'z_lr2': 1e-5,
        'model_lr': 1e-5,
        'z_steps1': 1600,
        'z_steps2': 3000,
    }

    def reset_gen(model):
        if model == 'began':
            gen = Generator128(64)
            gen = load_trained_net(
                gen,
                ('./checkpoints/celeba_began.withskips.bs32.cosine.min=0.25'
                 '.n_cuts=0/gen_ckpt.49.pt'))
            gen = gen.eval().to(DEVICE)
            img_size = 128
        elif model == 'vae':
            gen = VAE()
            t = torch.load('./vae_checkpoints/vae_bs=128_beta=1.0/epoch_19.pt')
            gen.load_state_dict(t)
            gen = gen.eval().to(DEVICE)
            gen = gen.decoder
            img_size = 128
        elif model == 'dcgan':
            gen = dcgan_generator()
            t = torch.load(('./dcgan_checkpoints/netG.epoch_24.n_cuts_0.bs_64'
                            '.b1_0.5.lr_0.0002.pt'))
            gen.load_state_dict(t)
            gen = gen.eval().to(DEVICE)
            img_size = 64
        return gen, img_size

    for model in tqdm(['dcgan', 'vae', 'began'], desc='Models', leave=True):
        gen, img_size = reset_gen(model)
        if img_size == 64:
            n_measures = [600, 2000]
        else:
            n_measures = [2400, 8000]

        for n_measure in tqdm(n_measures, desc='N_measures', leave=False):
            img_shape = (3, img_size, img_size)
            forward_model = GaussianCompressiveSensing(n_measure=n_measure,
                                                       img_shape=img_shape)
            # forward_model = NoOp()

            for img_name in tqdm(os.listdir(args.img_dir),
                                 desc='Images',
                                 leave=False,
                                 disable=args.disable_tqdm):
                gen, img_size = reset_gen(model)
                orig_img = load_target_image(
                    os.path.join(args.img_dir, img_name), img_size).to(DEVICE)
                img_basename, _ = os.path.splitext(img_name)
                x_hat, x_degraded, _ = iagan_recover(
                    orig_img,
                    gen,
                    forward_model,
                    run_dir='iagan',
                    run_name=(img_basename + args.run_name_suffix +
                              dict_to_str(params)),
                    **params)
