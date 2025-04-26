# Based off https://github.com/pytorch/examples/blob/master/vae/main.py
import argparse
import os

import torch
import torch.utils.data
from data.dataloaders import get_dataloader
from model.vae import VAE
from torch import optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm, trange

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--run_name', default='vae')
parser.add_argument('--beta', type=float, default=1.0)
args = parser.parse_args()


torch.backends.cudnn.benchmark = True

device = torch.device('cuda:0')

train_loader, img_size = get_dataloader('./data/celeba_preprocessed',
                                        args.batch_size,
                                        n_train=-1,
                                        train=True)
test_loader, _ = get_dataloader('./data/celeba_preprocessed',
                                args.batch_size,
                                n_train=-1,
                                train=False)

save_img_every_n = 20

latent_dim = 512
lr = 2e-4
model = VAE(latent_dim).to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.ExponentialLR(
    optimizer, gamma=0.9)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    reconstr = F.mse_loss(recon_x, x,
                          reduction='none').sum(3).sum(2).sum(1).mean(0)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = torch.mean(-0.5 *
                     torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1),
                     dim=0)

    return reconstr + args.beta * KLD, reconstr, KLD


def train(epoch, writer):
    model.train()
    for batch_idx, data in enumerate(
            tqdm(train_loader, leave=False, desc='Train')):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss, reconstr, kld = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        optimizer.step()

        x_value = len(train_loader) * epoch + batch_idx
        if batch_idx % save_img_every_n == 0:
            n = min(data.size(0), 8)
            writer.add_image('Train/Reconstructions',
                             make_grid(torch.cat([data[:n], recon_batch[:n]])),
                             x_value)
            with torch.no_grad():
                sample = torch.randn(n, latent_dim).to(device)
                sample = model.decoder(sample)
                writer.add_image('Train/Samples', make_grid(sample), x_value)

        writer.add_scalar('Train/Loss_Combined', loss, x_value)
        writer.add_scalar('Train/Reconstr', reconstr, x_value)
        writer.add_scalar('Train/KLD', kld, x_value)


def test(epoch, writer):
    model.eval()
    test_loss = 0.
    test_reconstr = 0.
    test_kld = 0.
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, leave=False, desc='Test')):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss, reconstr, kld = loss_function(recon_batch, data, mu, logvar)
            test_loss += loss
            test_reconstr += reconstr
            test_kld += kld
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon_batch[:n]])
                writer.add_image('Test/Reconstructions', make_grid(comparison),
                                 epoch)

    test_loss /= len(test_loader)
    test_reconstr /= len(test_loader)
    test_kld /= len(test_loader)

    # After finishing epoch 0, we have seen 1 * len(train_loader) batches
    x_value = len(train_loader) * (epoch + 1)
    writer.add_scalar('Test/Loss_Combined', test_loss, x_value)
    writer.add_scalar('Test/Reconstr', test_reconstr, x_value)
    writer.add_scalar('Test/KLD', test_kld, x_value)


if __name__ == "__main__":
    run_name = f'{args.run_name}_bs={args.batch_size}_beta={args.beta}'
    writer = SummaryWriter(f'./vae_tensorboard_logs/{run_name}')
    checkpoint_dir = f'./vae_checkpoints/{run_name}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    for epoch in trange(args.epochs, leave=True, desc='Epoch'):
        train(epoch, writer)
        test(epoch, writer)
        scheduler.step()
        torch.save(model.state_dict(), f'{checkpoint_dir}/epoch_{epoch}.pt')
