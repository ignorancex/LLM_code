# Based off https://github.com/pytorch/examples/blob/master/dcgan/main.py
import argparse
import os
import random

import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from data.dataloaders import get_dataloader
from model.dcgan import Discriminator, Generator

parser = argparse.ArgumentParser()
parser.add_argument('--n_cuts', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--b1', type=float, default=0.5, help='Adam beta1')
parser.add_argument('--netG', default='', help='checkpoint for resume')
parser.add_argument('--netD', default='', help='checkpoint for resume')
parser.add_argument('--manualSeed', type=int, help='manual seed')
opt = parser.parse_args()

opt.outf = './dcgan_checkpoints'
opt.batch_size = 64
opt.niter = 500
opt.cuda = True
opt.num_gen_updates = 1
print(opt)

os.makedirs(opt.outf, exist_ok=True)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

torch.backends.cudnn.benchmark = True

nc = 3
dataloader, _ = get_dataloader('dataset/celeba64x64_preprocessed',
                               opt.batch_size,
                               n_train=-1,
                               train=True)
device = torch.device("cuda:0" if opt.cuda else "cpu")

nz = 100
ngf = 64
ndf = 64


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


netG = Generator(nz=nz, ngf=ngf, nc=nc).to(device)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = Discriminator(nc=nc, ndf=ndf).to(device)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = torch.nn.BCELoss()

input_shape, _ = netG.input_shapes[opt.n_cuts]
print('n_cuts: ', opt.n_cuts)
print('input_shape: ', input_shape)

fixed_noise = torch.randn(opt.batch_size, *input_shape, device=device)
real_label = 1
fake_label = 0

# setup optimizer
optD = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.b1, 0.999))
optG = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.b1, 0.999))

settings = (f'n_cuts_{opt.n_cuts}.bs_{opt.batch_size}'
            f'.b1_{opt.b1}.lr_{opt.lr}.pt')

writer = SummaryWriter(f'./dcgan_tensorboard_logs/{settings}')

for epoch in trange(opt.niter, desc='Epochs', leave=True):
    for i, data in enumerate(tqdm(dataloader, desc='Batches', leave=False)):
        # real
        opt.batch_size = data.shape[0]
        real_cpu = data.to(device)
        label = torch.full((opt.batch_size, ), real_label, device=device)

        # fake
        noise = torch.randn(opt.batch_size, *input_shape, device=device)
        fake = netG(noise, n_cuts=opt.n_cuts)
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        if i % opt.num_gen_updates == 0:
            netD.zero_grad()

            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optG.step()

        x_value = len(dataloader) * epoch + i
        writer.add_scalar('loss_d', errD.item(), x_value)
        writer.add_scalar('loss_g', errG.item(), x_value)
        if i % 100 == 0:
            fake_img_grid = (netG(fixed_noise, n_cuts=opt.n_cuts).detach() +
                             1) / 2
            fake_img_grid = torchvision.utils.make_grid(
                fake_img_grid.clamp(0, 1))
            writer.add_image('fake_images', fake_img_grid, x_value)

    # do checkpointing
    torch.save(netG.state_dict(), f'{opt.outf}/netG.epoch_{epoch}.{settings}')
    torch.save(netD.state_dict(), f'{opt.outf}/netD.epoch_{epoch}.{settings}')
