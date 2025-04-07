import sys

sys.path.append(".")
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import argparse
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.metrics import dice
from utils.pytuils import AverageMeter
from unet.unet_model import UNet
from torch.nn.functional import cross_entropy, one_hot
from torch.cuda.amp import autocast, GradScaler
import importlib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--data_module", type=str)
    parser.add_argument("--lab_path", type=str)
    parser.add_argument("--logdir", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--index", type=str)
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--val_iters", type=int)
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpus", type=str)
    args = parser.parse_args()
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    model = torch.nn.DataParallel(
        UNet(n_channels=1, n_classes=args.num_classes, bilinear=True)
    ).cuda()

    data_module = importlib.import_module(f"{args.data_module}.dataset")
    train_dataset, val_dataset, _ = data_module.get_seg_dataset(
        args.data_path, args.lab_path
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=4,
    )

    args.max_iters = args.max_epochs * len(train_loader)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=args.lr,
        total_steps=args.max_iters,
    )
    scaler = GradScaler()

    writer = SummaryWriter(os.path.join(args.logdir, args.index))

    pbar = tqdm(range(1, args.max_iters + 1), ncols=100, desc="iter")

    train_loader_iter = iter(train_loader)
    for n_iter in pbar:
        optimizer.zero_grad()
        model.train()
        try:
            datapack = next(train_loader_iter)

        except:
            train_loader_iter = iter(train_loader)
            datapack = next(train_loader_iter)

        imgs = datapack["img"].cuda()
        segs = datapack["seg"].cuda()
        labs = datapack["lab"].cuda()

        with autocast():
            x = model(imgs)
            loss = cross_entropy(x, labs, weight=torch.tensor([10, 1]).cuda())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

        pred = (
            one_hot(
                x.argmax(dim=1),
                args.num_classes,
            )
            .permute(0, 3, 1, 2)
            .float()
        )
        score, cnt = dice(pred[:, 1:], segs)
        score /= cnt

        if n_iter % args.val_iters == 0:
            model.eval()
            val_score = AverageMeter()
            with torch.no_grad():
                for pack in val_loader:
                    imgs = pack["img"].cuda()
                    segs = pack["seg"].cuda()
                    with autocast():
                        x = model(imgs)
                        pred = (
                            one_hot(
                                x.argmax(dim=1),
                                args.num_classes,
                            )
                            .permute(0, 3, 1, 2)
                            .float()
                        )

                    val_score.add(*dice(pred[:, 1:], segs))
            writer.add_scalar("unet val/val score", val_score.get(), n_iter)

            model.train()

        writer.add_scalar("unet train/train loss", loss.item(), n_iter)
        writer.add_scalar("unet train/train score", score, n_iter)
        writer.add_scalar("unet train/lr", optimizer.param_groups[0]["lr"], n_iter)

        pbar.set_postfix(
            {
                "tl": loss.item(),
                "ts": score,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

    torch.save(
        model.module.state_dict(),
        os.path.join("tblog", args.index, f"{args.index}.pth"),
    )


if __name__ == "__main__":
    main()
