import argparse
import os
import shutil
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from forward_model import GaussianCompressiveSensing, NoOp
from model.deep_decoder import DeepDecoder
from utils import dict_to_str, load_target_image, psnr_from_mse

warnings.filterwarnings("ignore")


def _deep_decoder_recover(
    x,
    forward_model,
    optimizer,
    num_filters,
    depth,
    lr,
    img_size,
    steps,
    run_dir,
    run_name,
    disable_tqdm,
    **kwargs,
):
    # Keep batch_size = 1
    batch_size = 1

    if (isinstance(forward_model, GaussianCompressiveSensing)):
        n_pixel_bora = 64 * 64 * 3
        n_pixel = np.prod(x.shape)
        noise = torch.randn(batch_size,
                            forward_model.n_measure,
                            device=x.device)
        noise *= 0.1 * torch.sqrt(
            torch.tensor(n_pixel / forward_model.n_measure / n_pixel_bora))

    # z is a fixed latent vector
    start_imsize = int(np.log2(img_size)) - depth + 1
    z = torch.randn(batch_size,
                    num_filters,
                    start_imsize,
                    start_imsize,
                    device=x.device)

    # make a fresh DD model for every run
    model = DeepDecoder(num_filters=num_filters,
                        img_size=img_size,
                        depth=depth).to(x.device)

    if optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        save_img_every_n = 50
    elif optimizer == 'lbfgs':
        optimizer = torch.optim.LBFGS(model.parameters(), lr=lr)
        save_img_every_n = 2
    else:
        raise NotImplementedError()

    if run_name is not None:
        logdir = os.path.join('recovery_tensorboard_logs', run_dir, run_name)
        if os.path.exists(logdir):
            print("Overwriting pre-existing logs!")
            shutil.rmtree(logdir)
        writer = SummaryWriter(logdir)
    else:
        writer = None

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

    def closure():
        optimizer.zero_grad()
        x_hat = model.forward(z)
        loss = F.mse_loss(forward_model(x_hat), y_observed)
        loss.backward()
        return loss

    for j in trange(steps, desc='Fit', leave=False):
        optimizer.step(closure)
        with torch.no_grad():
            x_hat = model.forward(z)

        train_mse_clamped = F.mse_loss(
            forward_model(x_hat.detach().clamp(0, 1)), y_observed)
        if writer is not None:
            writer.add_scalar('TRAIN_MSE', train_mse_clamped, j + 1)
            writer.add_scalar('TRAIN_PSNR', psnr_from_mse(train_mse_clamped),
                              j + 1)

            orig_mse_clamped = F.mse_loss(x_hat.detach().clamp(0, 1), x)
            writer.add_scalar('ORIG_MSE', orig_mse_clamped, j + 1)
            writer.add_scalar('ORIG_PSNR', psnr_from_mse(orig_mse_clamped),
                              j + 1)
            if j % save_img_every_n == 0:
                writer.add_image('Recovered',
                                 x_hat.squeeze().clamp(0, 1), j + 1)

    if writer is not None:
        writer.add_image('Final', x_hat.squeeze().clamp(0, 1))

    return x_hat.squeeze(), forward_model(x).squeeze(), train_mse_clamped


def deep_decoder_recover(
        x,
        forward_model,
        optimizer='lbfgs',
        num_filters=64,
        depth=6,  # TODO
        lr=1,
        img_size=64,
        steps=50,
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
        return_val = _deep_decoder_recover(x=x,
                                           forward_model=forward_model,
                                           optimizer=optimizer,
                                           num_filters=num_filters,
                                           depth=depth,
                                           lr=lr,
                                           img_size=img_size,
                                           steps=steps,
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

    params_64 = {
        'depth': 5,
        'num_filters': 250,
        'lr': 1e-2,
        'steps': 5000,
        'restarts': 1,
        'optimizer': 'adam'
    }

    params_128 = {
        'depth': 6,
        'num_filters': 700,
        'lr': 1e-2,
        'steps': 5000,
        'restarts': 1,
        'optimizer': 'adam'
    }

    for img_size, n_measures, params in tqdm([(64, [600, 2000], params_64),
                                              (128, [2400, 8000], params_128)],
                                             desc='ImgSizes',
                                             leave=True,
                                             disable=args.disable_tqdm):

        for n_measure in tqdm(n_measures,
                              desc='N_measures',
                              leave=False,
                              disable=args.disable_tqdm):
            img_shape = (3, img_size, img_size)
            forward_model = GaussianCompressiveSensing(n_measure=n_measure,
                                                       img_shape=img_shape)
            # forward_model = NoOp()

            for img_name in tqdm(os.listdir(args.img_dir),
                                 desc='Images',
                                 leave=False,
                                 disable=args.disable_tqdm):
                orig_img = load_target_image(
                    os.path.join(args.img_dir, img_name), img_size).to(DEVICE)
                img_basename, _ = os.path.splitext(img_name)
                x_hat, x_degraded, _ = deep_decoder_recover(
                    orig_img,
                    forward_model,
                    run_dir='deep_decoder',
                    run_name=(img_basename + args.run_name_suffix + '.' +
                              dict_to_str(params) +
                              f'.n_measure={n_measure}.img_size={img_size}'),
                    img_size=img_size,
                    **params)
