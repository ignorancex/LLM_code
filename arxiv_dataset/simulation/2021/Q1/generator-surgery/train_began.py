import argparse
import os

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from data.dataloaders import get_dataloader
from model.began import Discriminator128, Generator128
from utils import get_z_vector, normalize


def save(path, epoch, model, optimizer, scheduler):
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, path)
    return


def load(path, model, optimizer, scheduler):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    return epoch


def ac_loss(input, disc):
    # pixelwise L1 - for each pixel for each image in the batch
    return torch.mean(torch.abs(input - disc.forward(input)))


def main(args):
    checkpoint_path = f"checkpoints/{args.dataset}_{args.run_name}"
    tensorboard_path = f"tensorboard_logs/{args.dataset}_{args.run_name}"
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(tensorboard_path)

    dataloader, _ = get_dataloader(args.dataset_dir, args.batch_size,
                                   args.n_train, True)

    gen = Generator128(args.latent_dim).to(device)
    disc = Discriminator128(args.latent_dim).to(device)

    # Get latent_shape for x1 only
    latent_shape = gen.input_shapes[args.n_cuts][0]

    if torch.cuda.device_count() > 1:
        gen = torch.nn.DataParallel(gen)
        disc = torch.nn.DataParallel(disc)

    gen_optimizer = torch.optim.Adam(gen.parameters(), args.lr)
    gen_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        gen_optimizer,
        len(dataloader) * args.epochs, 0.25 * args.lr)
    disc_optimizer = torch.optim.Adam(disc.parameters(), args.lr)
    disc_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        disc_optimizer,
        len(dataloader) * args.epochs, 0.25 * args.lr)

    current_checkpoint = 0
    if (not os.path.exists(checkpoint_path)):
        os.makedirs(checkpoint_path)
    else:
        print("Restoring from checkpoint...")
        paths = os.listdir(checkpoint_path)
        try:
            available = sorted(set([int(x.split(".")[1]) for x in paths]))

            # Find a checkpoint that both gen AND disc have reached
            # Reaching zero will cause IndexError during pop()
            while True:
                latest_idx = available.pop()
                latest_disc = os.path.join(checkpoint_path,
                                           f"disc_ckpt.{latest_idx}.pt")
                latest_gen = os.path.join(checkpoint_path,
                                          f"gen_ckpt.{latest_idx}.pt")
                if os.path.exists(latest_disc) and os.path.exists(latest_gen):
                    break

            current_checkpoint = latest_idx
            disc_epoch = load(latest_disc, disc, disc_optimizer,
                              disc_scheduler)
            gen_epoch = load(latest_gen, gen, gen_optimizer, gen_scheduler)
            assert disc_epoch == gen_epoch, \
                'Checkpoint contents are mismatched!'
            print(f"Loaded checkpoint {current_checkpoint}")
        except Exception as e:
            print(e)
            print("Unable to load from checkpoint.")

    k = 0

    # Uniform from -1 to 1
    const_sample = get_z_vector((args.batch_size, *latent_shape),
                                mode='uniform',
                                dtype=torch.float,
                                device=device)

    n_gen_param = sum([x.numel() for x in gen.parameters() if x.requires_grad])
    n_disc_param = sum(
        [x.numel() for x in disc.parameters() if x.requires_grad])
    print(f"{n_gen_param + n_disc_param} Trainable Parameters")

    if current_checkpoint < args.epochs - 1:
        for e in trange(current_checkpoint,
                        args.epochs,
                        initial=current_checkpoint,
                        desc='Epoch',
                        leave=True,
                        disable=args.disable_tqdm):
            for i, img_batch in tqdm(enumerate(dataloader),
                                     total=len(dataloader),
                                     leave=False,
                                     disable=args.disable_tqdm):
                disc_optimizer.zero_grad()
                gen_optimizer.zero_grad()

                img_batch = img_batch.to(device)

                # Uniform from -1 to 1
                d_latent_sample = get_z_vector(
                    (args.batch_size, *latent_shape),
                    mode='uniform',
                    dtype=torch.float,
                    device=device)

                g_latent_sample = get_z_vector(
                    (args.batch_size, *latent_shape),
                    mode='uniform',
                    dtype=torch.float,
                    device=device)

                batch_ac_loss = ac_loss(img_batch, disc)
                d_fake_ac_loss = ac_loss(
                    gen.forward(d_latent_sample, x2=None,
                                n_cuts=args.n_cuts).detach(), disc)
                g_fake_ac_loss = ac_loss(
                    gen.forward(g_latent_sample, x2=None, n_cuts=args.n_cuts),
                    disc)

                def d_loss():
                    loss = batch_ac_loss - k * d_fake_ac_loss
                    loss.backward()
                    return loss

                def g_loss():
                    loss = g_fake_ac_loss
                    loss.backward()
                    return loss

                disc_optimizer.step(d_loss)
                gen_optimizer.step(g_loss)
                disc_scheduler.step()
                gen_scheduler.step()

                k = k + args.prop_gain * \
                    (args.gamma * batch_ac_loss.item() - g_fake_ac_loss.item())

                m = ac_loss(img_batch, disc) + \
                    torch.abs(args.gamma * batch_ac_loss - g_fake_ac_loss)
                writer.add_scalar("Convergence", m, len(dataloader) * e + i)

                if (i % args.log_every == 0):
                    ex_img = gen.forward(g_latent_sample,
                                         x2=None,
                                         n_cuts=args.n_cuts)[0]
                    writer.add_image("Random/Raw", ex_img,
                                     len(dataloader) * e + i)
                    writer.add_image("Random/Clamp", ex_img.clamp(0, 1),
                                     len(dataloader) * e + i)
                    writer.add_image("Random/Normalize", normalize(ex_img),
                                     len(dataloader) * e + i)
                    ex_img_const = gen.forward(const_sample,
                                               x2=None,
                                               n_cuts=args.n_cuts)[0]
                    writer.add_image("Constant/Raw", ex_img_const,
                                     len(dataloader) * e + i)
                    writer.add_image("Constant/Clamp",
                                     ex_img_const.clamp(0, 1),
                                     len(dataloader) * e + i)
                    writer.add_image("Constant/Normalize",
                                     normalize(ex_img_const),
                                     len(dataloader) * e + i)

            save(os.path.join(checkpoint_path, f"gen_ckpt.{e}.pt"), e, gen,
                 gen_optimizer, gen_scheduler)
            save(os.path.join(checkpoint_path, f"disc_ckpt.{e}.pt"), e, disc,
                 disc_optimizer, disc_scheduler)


def float01(x):
    x = float(x)
    if x > 1 or x < 0:
        raise argparse.ArgumentError()
    return x


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_dir', required=True)
    p.add_argument('--batch_size', type=int, required=True)
    p.add_argument('--run_name', required=True)

    p.add_argument('--n_cuts', type=int, default=0, choices=[0, 1])
    p.add_argument('--latent_dim', type=int, default=64)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--n_train', type=int, default=-1)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--b1', type=float, default=0.9, help='Adam beta1')
    p.add_argument('--b2', type=float, default=0.999, help='Adam beta2')
    p.add_argument('--gamma',
                   type=float01,
                   default=0.5,
                   help='BEGAN diversity parameter')
    p.add_argument('--prop_gain',
                   type=float,
                   default=0.001,
                   help='Proportional gain for k')
    p.add_argument('--log_every',
                   type=int,
                   default=100,
                   help='tensorboard logging interval')

    p.add_argument('--disable_tqdm', action='store_true')

    args = p.parse_args()
    args.dataset = 'celeba'
    main(args)
