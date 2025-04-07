from torch.utils.data import DataLoader
import torch
import argparse
import os
from samus.build_sam_us import samus_model_registry
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import importlib
from utils.pytuils import AverageMeter
import torch.nn.functional as F
from utils.metrics import dice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--data_module", type=str)
    parser.add_argument("--vit_name", type=str)
    parser.add_argument("--sam_ckpt", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--index", type=str)
    parser.add_argument("--samus_ckpt", type=str)
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--val_iters", type=int)
    parser.add_argument("--parent_classes", type=int)
    parser.add_argument("--child_classes", type=int)
    parser.add_argument("--child_weight", type=float)
    parser.add_argument("--cluster_file", type=str)
    parser.add_argument("--logdir", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpus", type=str)
    args = parser.parse_args()
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    model = samus_model_registry["vit_b"](
        parent_classes=args.parent_classes,
        child_classes=args.child_classes,
        checkpoint=args.sam_ckpt,
    )
    model = torch.nn.DataParallel(model).cuda()
    if args.samus_ckpt:
        checkpoint = torch.load(args.samus_ckpt)
        model.load_state_dict(checkpoint)

    data_module = importlib.import_module(f"{args.data_module}.dataset")
    train_dataset, val_dataset, _ = data_module.get_dataset(
        args.data_path, args.child_classes, args.cluster_file
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer, max_lr=args.lr, total_steps=args.max_iters
    )

    writer = SummaryWriter(os.path.join(args.logdir, args.index))

    pbar = tqdm(range(1, args.max_iters + 1), ncols=100)
    train_loader_iter = iter(train_loader)
    for n_iter in pbar:
        model.train()
        optimizer.zero_grad()
        try:
            datapack = next(train_loader_iter)

        except:
            train_loader_iter = iter(train_loader)
            datapack = next(train_loader_iter)

        imgs = datapack["img"].cuda()
        parent_labs = datapack["plab"].cuda()
        child_labs = datapack["clab"].cuda()

        parent_x, child_x, _ = model(imgs)

        parent_loss = F.binary_cross_entropy_with_logits(
            parent_x,
            parent_labs,
        )

        child_loss = F.binary_cross_entropy_with_logits(child_x, child_labs)
        loss = parent_loss + args.child_weight * child_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        parent_pred = (torch.sigmoid(parent_x) > 0.5).float()
        parent_score = torch.eq(parent_pred, parent_labs).sum() / parent_labs.numel()

        child_pred = (torch.sigmoid(child_x) > 0.5).float()
        child_score = torch.eq(child_pred, child_labs).sum() / child_labs.numel()

        writer.add_scalar("train/train loss", loss.item(), n_iter)
        writer.add_scalar("train/parent loss", parent_loss.item(), n_iter)
        writer.add_scalar("train/child loss", child_loss.item(), n_iter)
        writer.add_scalar("train/parent score", parent_score.item(), n_iter)
        writer.add_scalar("train/child score", child_score.item(), n_iter)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], n_iter)

        if n_iter % args.val_iters == 0:
            model.eval()
            val_loss = AverageMeter()
            val_score = AverageMeter()
            with torch.no_grad():
                for pack in val_loader:
                    imgs = pack["img"].cuda()
                    labs = pack["plab"].float().cuda()
                    x, _, _ = model(imgs)
                    val_loss.add(F.binary_cross_entropy_with_logits(x, labs).item())
                    pred = (torch.sigmoid(x) > 0.5).float()

                    val_score.add(torch.eq(pred, labs).sum().item(), labs.numel())

            model.train()
            writer.add_scalar("val/val loss", val_loss.get(), n_iter)
            writer.add_scalar("val/val score", val_score.get(), n_iter)

        pbar.set_postfix(
            {
                "tl": loss.item(),
                "ts": parent_score.item(),
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

    torch.save(
        model.module.state_dict(),
        os.path.join(args.logdir, args.index, f"{args.index}.pth"),
        _use_new_zipfile_serialization=False,
    )


if __name__ == "__main__":
    main()
