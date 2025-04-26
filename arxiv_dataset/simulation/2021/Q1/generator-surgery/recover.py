import argparse
import os
import shutil
import warnings

import numpy as np
import scipy.fftpack as fftpack
import torch
import torch.nn.functional as F
from sklearn.linear_model import Lasso
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from forward_model import GaussianCompressiveSensing, NoOp
from model.began import Generator128
from utils import (get_z_vector, load_target_image, load_trained_net, psnr,
                   psnr_from_mse)

warnings.filterwarnings("ignore")

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def _recover(x,
             gen,
             optimizer_type,
             n_cuts,
             forward_model,
             mode='clamped_normal',
             limit=1,
             z_lr=0.5,
             n_steps=2000,
             run_dir=None,
             run_name=None,
             disable_tqdm=False,
             return_z1_z2=False,
             **kwargs):
    """
    Args:
        x - input image, torch tensor (C x H x W)
        gen - generator, already loaded with checkpoint weights
        forward_model - corrupts the image
        n_steps - number of optimization steps during recovery
        run_name - use None for no logging
    """

    # Keep batch_size = 1
    batch_size = 1

    z1_dim, z2_dim = gen.input_shapes[n_cuts]

    if (isinstance(forward_model, GaussianCompressiveSensing)):
        n_pixel_bora = 64 * 64 * 3
        n_pixel = np.prod(x.shape)
        noise = torch.randn(batch_size,
                            forward_model.n_measure,
                            device=x.device)
        noise *= 0.1 * torch.sqrt(
            torch.tensor(n_pixel / forward_model.n_measure / n_pixel_bora))

    if mode == 'lasso_inverse' and isinstance(forward_model,
                                              GaussianCompressiveSensing):
        lasso_x_hat = recover_dct(x.cpu().numpy().transpose([1, 2, 0]),
                                  forward_model.n_measure,
                                  0.01,
                                  128,
                                  A=forward_model.A.cpu().numpy(),
                                  noise=noise.cpu().numpy())

        _, _, _, z1_z2_dict = recover(torch.tensor(
            lasso_x_hat.transpose([2, 0, 1]), dtype=torch.float).to(DEVICE),
                                      gen,
                                      optimizer_type=optimizer_type,
                                      n_cuts=n_cuts,
                                      forward_model=forward_model,
                                      mode='clamped_normal',
                                      limit=limit,
                                      z_lr=z_lr,
                                      n_steps=n_steps,
                                      restarts=1,
                                      return_z1_z2=True)
        z1 = torch.nn.Parameter(z1_z2_dict['z1'])
        params = [z1]
        if len(z2_dim) > 0:
            z2 = torch.nn.Parameter(z1_z2_dict['z2'])
            params.append(z2)
        else:
            z2 = None

    else:
        z1 = torch.nn.Parameter(
            get_z_vector((batch_size, *z1_dim),
                         mode=mode,
                         limit=limit,
                         device=x.device))
        # print('z1: ', z1.min(), z1.max())
        params = [z1]
        if len(z2_dim) > 0:
            z2 = torch.nn.Parameter(
                get_z_vector((batch_size, *z2_dim),
                             mode=mode,
                             limit=limit,
                             device=x.device))
            # print('z2: ', z2.min(), z2.max())
            params.append(z2)
        else:
            z2 = None

    if optimizer_type == 'adamw':
        optimizer_z = torch.optim.AdamW(params,
                                        lr=z_lr,
                                        betas=(0.5, 0.999),
                                        weight_decay=0)
        scheduler_z = None
        save_img_every_n = 50
    elif optimizer_type == 'lbfgs':
        optimizer_z = torch.optim.LBFGS(params, lr=z_lr)
        scheduler_z = None
        save_img_every_n = 2
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

    # Recover image under forward model
    x = x.expand(batch_size, *x.shape)

    y_observed = forward_model(x)
    if (isinstance(forward_model, GaussianCompressiveSensing)):
        y_observed += noise

    for j in trange(n_steps,
                    leave=False,
                    desc='Recovery',
                    disable=disable_tqdm):

        def closure():
            optimizer_z.zero_grad()
            x_hats = gen.forward(z1, z2, n_cuts=n_cuts, **kwargs)
            if gen.rescale:
                x_hats = (x_hats + 1) / 2
            train_mses = F.mse_loss(forward_model(x_hats),
                                    y_observed,
                                    reduction='none')
            train_mses = train_mses.view(batch_size, -1).mean(1)

            train_mse = train_mses.sum()
            train_mse.backward()
            return train_mse

        # Step first, then identify the current "best" and "worst"
        optimizer_z.step(closure)
        with torch.no_grad():
            x_hats = gen.forward(z1, z2, n_cuts=n_cuts, **kwargs)
            if gen.rescale:
                x_hats = (x_hats + 1) / 2
            train_mses = F.mse_loss(forward_model(x_hats),
                                    y_observed,
                                    reduction='none')
            train_mses = train_mses.view(batch_size, -1).mean(1)
            train_mse = train_mses.sum()

        train_mses_clamped = F.mse_loss(forward_model(x_hats.detach().clamp(
            0, 1)),
                                        y_observed,
                                        reduction='none').view(batch_size,
                                                               -1).mean(1)
        orig_mses_clamped = F.mse_loss(x_hats.detach().clamp(0, 1),
                                       x,
                                       reduction='none').view(batch_size,
                                                              -1).mean(1)

        # batch_size = 1, so best and worst are meaningless.
        # Restarts is handled in outer function
        best_train_mse, best_idx = train_mses_clamped.min(0)
        worst_train_mse, worst_idx = train_mses_clamped.max(0)
        best_orig_mse = orig_mses_clamped[best_idx]
        worst_orig_mse = orig_mses_clamped[worst_idx]

        if run_name is not None and j == 0:
            writer.add_image('Start', x_hats[best_idx].clamp(0, 1))

        if run_name is not None:
            writer.add_scalar('TRAIN_MSE/best', best_train_mse, j + 1)
            writer.add_scalar('TRAIN_MSE/worst', worst_train_mse, j + 1)
            writer.add_scalar('TRAIN_MSE/sum', train_mse, j + 1)
            writer.add_scalar('ORIG_MSE/best', best_orig_mse, j + 1)
            writer.add_scalar('ORIG_MSE/worst', worst_orig_mse, j + 1)
            writer.add_scalar('ORIG_PSNR/best', psnr_from_mse(best_orig_mse),
                              j + 1)
            writer.add_scalar('ORIG_PSNR/worst', psnr_from_mse(worst_orig_mse),
                              j + 1)

            if j % save_img_every_n == 0:
                writer.add_image('Recovered/Best',
                                 x_hats[best_idx].clamp(0, 1), j + 1)

        if scheduler_z is not None:
            scheduler_z.step()

    if run_name is not None:
        writer.add_image('Final', x_hats[best_idx].clamp(0, 1))

    if return_z1_z2:
        return x_hats[best_idx], forward_model(x)[0], best_train_mse, {
            'z1': z1,
            'z2': z2
        }
    else:
        return x_hats[best_idx], forward_model(x)[0], best_train_mse


def recover(x,
            gen,
            optimizer_type,
            n_cuts,
            forward_model,
            mode='clamped_normal',
            limit=1,
            z_lr=0.5,
            n_steps=2000,
            restarts=1,
            run_dir=None,
            run_name=None,
            disable_tqdm=False,
            return_z1_z2=False,
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
        return_val = _recover(x=x,
                              gen=gen,
                              optimizer_type=optimizer_type,
                              n_cuts=n_cuts,
                              forward_model=forward_model,
                              mode=mode,
                              limit=limit,
                              z_lr=z_lr,
                              n_steps=n_steps,
                              run_dir=run_dir,
                              run_name=current_run_name,
                              disable_tqdm=disable_tqdm,
                              return_z1_z2=return_z1_z2,
                              **kwargs)
        p = psnr_from_mse(return_val[2])
        if p > best_psnr:
            best_psnr = p
            best_return_val = return_val

    return best_return_val


def recover_dct(x_test, n_measure, gamma, size, A=None, noise=None):
    n_pixel_bora = 64 * 64 * 3
    n_pixel = size * size * 3

    if A is None:
        A = np.random.normal(0,
                             1 / np.sqrt(n_measure),
                             size=(n_pixel, n_measure))

    if noise is None:
        noise = np.random.normal(0, 1, size=(n_measure))
        noise *= 0.1 * np.sqrt(n_pixel / n_measure / n_pixel_bora)

    y_true = np.matmul(x_test.reshape(-1), A) + noise

    for i in range(A.shape[1]):
        A[:, i] = vec([dct2(channel) for channel in devec(A[:, i], s=size)],
                      s=size)

    z_hat = solve_lasso(
        np.sqrt(2 * n_measure) * A,
        np.sqrt(2 * n_measure) * y_true, gamma)

    x_hat = vec([idct2(channel) for channel in devec(z_hat, s=size)], s=size).T

    x_hat = np.maximum(np.minimum(x_hat, 1), -1)

    x_hat = np.array(x_hat)
    x_hat = x_hat.reshape(size, size, 3)

    x_hat = np.clip(x_hat, 0, 1)
    return x_hat


def dct2(image_channel):
    return fftpack.dct(fftpack.dct(image_channel.T, norm='ortho').T,
                       norm='ortho')


def idct2(image_channel):
    return fftpack.idct(fftpack.idct(image_channel.T, norm='ortho').T,
                        norm='ortho')


def vec(channels, s=64):
    image = np.zeros((s, s, 3))
    for i, channel in enumerate(channels):
        image[:, :, i] = channel
    return image.reshape([-1])


def devec(vector, s=64):
    image = np.reshape(vector, [s, s, 3])
    channels = [image[:, :, i] for i in range(3)]
    return channels


def solve_lasso(A_val, y_val, gamma):
    lasso_est = Lasso(alpha=gamma)
    lasso_est.fit(A_val.T, y_val.reshape(-1))
    x_hat = lasso_est.coef_
    x_hat = np.reshape(x_hat, [-1])
    return x_hat


if __name__ == '__main__':
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    a = argparse.ArgumentParser()
    a.add_argument('--img_dir', required=True)
    a.add_argument('--disable_tqdm', default=False)
    args = a.parse_args()

    gen = Generator128(64)
    gen = load_trained_net(
        gen, ('./checkpoints/celeba_began.withskips.bs32.cosine.min=0.25'
              '.n_cuts=0/gen_ckpt.49.pt'))
    gen = gen.eval().to(DEVICE)

    n_cuts = 3

    img_size = 128
    img_shape = (3, img_size, img_size)

    forward_model = GaussianCompressiveSensing(n_measure=2500,
                                               img_shape=img_shape)
    # forward_model = NoOp()

    for img_name in tqdm(os.listdir(args.img_dir),
                         desc='Images',
                         leave=True,
                         disable=args.disable_tqdm):
        orig_img = load_target_image(os.path.join(args.img_dir, img_name),
                                     img_size).to(DEVICE)
        img_basename, _ = os.path.splitext(img_name)
        x_hat, x_degraded, _ = recover(orig_img,
                                       gen,
                                       optimizer_type='lbfgs',
                                       n_cuts=n_cuts,
                                       forward_model=forward_model,
                                       z_lr=1.0,
                                       n_steps=25,
                                       run_dir='ours',
                                       run_name=img_basename)
