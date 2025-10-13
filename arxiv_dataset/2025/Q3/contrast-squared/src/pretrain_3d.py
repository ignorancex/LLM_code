import glob
import os
import model
import preprocess_3d
import torch
import wandb
import logging
import argparse
import monai as mn
from contextlib import nullcontext
from tqdm import tqdm
from torchvision.utils import make_grid
logging.getLogger("monai").setLevel(logging.ERROR)
import warnings

warnings.filterwarnings(
    "ignore",
    ".*pixdim*.",
)
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")


def run_model(args, device, train_loader, train_transform):
    
    if args.net == "cnn":
        encoder = model.CNNEncoder(
            spatial_dims=3, 
            in_channels=1, 
            features=(64, 128, 256, 512, 768), 
            act="GELU", 
            norm="instance", 
            bias=True, 
            dropout=0.2
        ).to(device)
    elif args.net == "vit":
        encoder = model.ViTEncoder(
            spatial_dims=3,
            in_channels=1,
            img_size=(48 if args.lowres else 96),
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            dropout_rate=0.2,
            qkv_bias=True,
            save_attn=False
        ).to(device)

    projector = model.Projector(
        in_features=768, hidden_size=512, out_features=128
    ).to(device)

    if not args.no_recon:
        reconstructor = model.Reconstructor(
            in_features=768, out_channels=1
        ).to(device)

    if args.resume or args.resume_best:
        ckpts = glob.glob(
            os.path.join(
                args.logdir,
                args.name,
                "checkpoint.pt" if args.resume else "checkpoint_best.pt",
            )
        )
        if len(ckpts) == 0:
            args.resume = False
            print("\nNo checkpoints found. Beginning from epoch #0")
        else:
            checkpoint = torch.load(ckpts[0], map_location=device)
            print(
                "\nResuming from epoch #{} with WandB ID {}".format(
                    checkpoint["epoch"], checkpoint["wandb"]
                )
            )
    print()

    wandb.init(
        project="contrast-squared",
        entity="atlas-ploras",
        save_code=True,
        name=args.name,
        settings=wandb.Settings(start_method="fork"),
        resume="must" if args.resume and not args.reset_wandb else None,
        id=checkpoint["wandb"] if (args.resume or args.resume_best) and not args.reset_wandb else None,
    )
    if not args.resume and not args.resume_best:
        wandb.config.update(args)
    wandb.watch(encoder, projector)

    if args.loss == "simclr":
        crit = mn.losses.ContrastiveLoss()
    elif args.loss == "barlow":
        crit = mn.losses.BarlowTwinsLoss()

    class WandBID:
        def __init__(self, wandb_id):
            self.wandb_id = wandb_id

        def state_dict(self):
            return self.wandb_id

    class Epoch:
        def __init__(self, epoch):
            self.epoch = epoch

        def state_dict(self):
            return self.epoch

    class Metric:
        def __init__(self, metric):
            self.metric = metric

        def state_dict(self):
            return self.metric

    params = list(encoder.parameters()) + list(projector.parameters())
    if not args.no_recon:
        params += list(reconstructor.parameters())
    # try:
    #     opt = torch.optim.AdamW(params, args.lr, fused=torch.cuda.is_available(), foreach=False)
    # except:
    opt = torch.optim.AdamW(params, args.lr, fused=False, foreach=False)
    # Try to load most recent weight
    if args.resume or args.resume_best:
        encoder.load_state_dict(
            checkpoint["encoder"]
        )  # strict False in case of switch between subpixel and transpose
        if "projector" in checkpoint:
            projector.load_state_dict(checkpoint["projector"], strict=False)
        if not args.no_recon and "reconstructor" in checkpoint:
            reconstructor.load_state_dict(checkpoint["reconstructor"], strict=False)
        if "opt" in checkpoint:
            try:
                opt.load_state_dict(checkpoint["opt"])
            except:
                print("Warning: optimizer state dict not found. Continuing with existing optimizer state.")
        start_epoch = checkpoint["epoch"] + 1
        metric_best = checkpoint["metric"]
        # correct scheduler in cases where max epochs has changed
        def lambda1(epoch):
            return (1 - (epoch + start_epoch) / args.epochs) ** 0.9

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=[lambda1])
    else:
        start_epoch = 0
        metric_best = 0

        def lambda1(epoch):
            return (1 - (epoch) / args.epochs) ** 0.9

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=[lambda1])

    os.makedirs(os.path.join(args.logdir, args.name), exist_ok=True)

    train_iter = None
    for epoch in range(start_epoch, args.epochs):
        if args.debug:
            saver1 = mn.transforms.SaveImage(
                output_dir=os.path.join(args.logdir, args.name, "debug-train"),
                output_postfix="img1",
                separate_folder=False,
                print_log=False,
            )
            saver2 = mn.transforms.SaveImage(
                output_dir=os.path.join(args.logdir, args.name, "debug-train"),
                output_postfix="img2",
                separate_folder=False,
                print_log=False,
            )
        encoder.train()
        projector.train()
        if not args.no_recon:
            reconstructor.train()
        epoch_loss = 0
        step_deficit = -1e-7
        if args.amp:
            ctx = torch.autocast("cuda" if torch.cuda.is_available() else "cpu")
            scaler = torch.cuda.amp.GradScaler()
        else:
            ctx = nullcontext()
        progress_bar = tqdm(range(args.epoch_length), total=args.epoch_length, ncols=90)
        progress_bar.set_description(f"Epoch {epoch}")
        if train_iter is None:
            train_iter = iter(train_loader)

        for step in progress_bar:
            try:
                batch = next(train_iter)
            except:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            img1 = batch["image1"].to(device)
            img2 = batch["image2"].to(device)
            if not args.no_recon:
                recon_target1 = batch["image1_recon"].to(device)
                recon_target2 = batch["image2_recon"].to(device)
            opt.zero_grad(set_to_none=True)

            if args.debug and step < 5:
                saver1(torch.Tensor(img1[0].cpu().float()))
                saver2(torch.Tensor(img2[0].cpu().float()))

            with ctx:
                features1 = encoder(img1)
                if len(features1.shape) == 3 and features1.shape[1] != 768 and features1.shape[2] == 768:
                    features1 = torch.movedim(features1, 2, 1)
                if not args.no_recon:
                    recon1 = reconstructor(features1)
                features1 = features1.view(img1.size(0), 768, -1).mean(dim=-1)

                features2 = encoder(img2)
                if len(features2.shape) == 3 and features2.shape[1] != 768 and features2.shape[2] == 768:
                    features2 = torch.movedim(features2, 2, 1)
                if not args.no_recon:
                    recon2 = reconstructor(features2)
                features2 = features2.view(img2.size(0), 768, -1).mean(dim=-1)

                embeddings1 = projector(features1)
                embeddings2 = projector(features2)

                ssl_loss = crit(embeddings1, embeddings2)
                if args.loss == "barlow":
                    ssl_loss = ssl_loss / 128

                recon_loss = 0
                if not args.no_recon:
                    recon_loss = torch.nn.functional.l1_loss(recon1, recon_target1) + torch.nn.functional.l1_loss(recon2, recon_target2)
                loss = ssl_loss + recon_loss

            if type(loss) == float or loss.isnan().sum() != 0:
                print("NaN found in loss!")
                step_deficit += 1
            else:
                if args.amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), 12 if args.net == "cnn" else 1)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), 12 if args.net == "cnn" else 1)
                    opt.step()

                epoch_loss += loss.item()

                loss_update = {"train/ssl_loss": ssl_loss.item()}
                if not args.no_recon:
                    loss_update["train/recon_loss"] = recon_loss.item()
                wandb.log(loss_update)

            progress_bar.set_postfix(
                {"loss": epoch_loss / (step + 1 - step_deficit)}
            )
        
        wandb.log({"train/lr": opt.param_groups[0]["lr"]})
        lr_scheduler.step()

        state_dict = {
            "encoder": encoder.state_dict(),
            "projector": projector.state_dict(),
            "opt": opt.state_dict(),
            "lr": lr_scheduler.state_dict(),
            "wandb": WandBID(wandb.run.id).state_dict(),
            "epoch": Epoch(epoch).state_dict(),
            "metric": Metric(metric_best).state_dict(),
        }
        if not args.no_recon:
            state_dict["reconstructor"] = reconstructor.state_dict()

        if epoch_loss < metric_best:
            metric_best = epoch_loss
            state_dict["metric"] = Metric(metric_best).state_dict()
            torch.save(
                state_dict,
                os.path.join(args.logdir, args.name, "checkpoint_best.pt"),
            )
        torch.save(
            state_dict,
            os.path.join(args.logdir, args.name, "checkpoint.pt"),
        )

        # Upload some sample pairs to wandb
        if img1.size(0) >= 4:
            img1_list = []
            img2_list = []
            recon1_list = []
            recon2_list = []
            for i in range(4):
                img1_list.append(img1[i][..., img1.shape[-1]//2])
                img2_list.append(img2[i][..., img2.shape[-1]//2])
                if not args.no_recon:
                    recon1_list.append(recon1[i][..., recon1.shape[-1]//2])
                    recon2_list.append(recon2[i][..., recon2.shape[-1]//2])
            grid_image1 = make_grid(
                        img1_list,
                        nrow=int(2),
                        padding=5,
                        normalize=True,
                        scale_each=True,
                    )
            grid_image2 = make_grid(
                        img2_list,
                        nrow=int(2),
                        padding=5,
                        normalize=True,
                        scale_each=True,
                    )
            if not args.no_recon:
                grid_recon1 = make_grid(
                            recon1_list,
                            nrow=int(2),
                            padding=5,
                            normalize=True,
                            scale_each=True,
                        )
                grid_recon2 = make_grid(
                            recon2_list,
                            nrow=int(2),
                            padding=5,
                            normalize=True,
                            scale_each=True,
                        )
                
            wandb_examples = [
                wandb.Image(
                    grid_image1[0].cpu().numpy(), caption="Image view #1"
                ),
                wandb.Image(
                    grid_image2[0].cpu().numpy(), caption="Image view #2"
                ),
            ]
            if not args.no_recon:
                wandb_examples += [
                    wandb.Image(
                        grid_recon1[0].cpu().numpy(), caption="Recon view #1"
                    ),
                    wandb.Image(
                        grid_recon2[0].cpu().numpy(), caption="Recon view #2"
                    ),
                ]

            wandb.log({"examples": wandb_examples})


def set_up():
    parser = argparse.ArgumentParser(argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--name", type=str, help="Name of WandB run.")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs for training.")
    parser.add_argument("--epoch_length", type=int, default=200, help="Number of iterations per epoch.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--val_interval", type=int, default=2, help="Validation interval.")
    parser.add_argument("--batch_size", type=int, default=512, help="Number of subjects to use per batch.")
    parser.add_argument(
        "--loss",
        type=str,
        help="Loss function to use. Options: [simclr]",
        choices=["simclr", "barlow"],
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Data setup to use. Options: [mprage, bloch, bloch-paired]",
        choices=["mprage", "bloch", "bloch-paired"],
    )
    parser.add_argument(
        "--net", 
        type=str, 
        help="Encoder network to use. Options: [cnn, vit]. Defaults to cnn.", 
        choices=["cnn", "vit"],
        default="cnn"
    )
    parser.add_argument("--amp", default=False, action="store_true")
    parser.add_argument("--logdir", type=str, default="./", help="Path to saved outputs")
    parser.add_argument("--resume", default=False, action="store_true")
    parser.add_argument("--resume_best", default=False, action="store_true")
    parser.add_argument("--device", type=str, default=None, help="Device to use. If not specified then will check for CUDA.")
    parser.add_argument("--lowres", default=False, action="store_true", help="Train with 2mm resolution images.")
    parser.add_argument("--debug", default=False, action="store_true", help="Save sample images before training.")
    parser.add_argument("--no_recon", default=False, action="store_true", help="Don't use reconstruction loss.")
    parser.add_argument("--reset_wandb", default=False, action="store_true", help="Reset WandB run.")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.logdir, args.name), exist_ok=True)
    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print("Using device:", device)
    print()
    # Additional Info when using cuda
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
        print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")
    if args.data == "mprage":
        debug_loader, _ = preprocess_3d.get_mprage_loader(batch_size=1, device=device, lowres=args.lowres)
        train_loader, train_transform = preprocess_3d.get_mprage_loader(batch_size=args.batch_size, device=device, lowres=args.lowres)
    elif args.data == "bloch":
        debug_loader, _ = preprocess_3d.get_bloch_loader(batch_size=1, device=device, lowres=args.lowres, same_contrast=True)
        train_loader, train_transform = preprocess_3d.get_bloch_loader(batch_size=args.batch_size, device=device, lowres=args.lowres, same_contrast=True)
    elif args.data == "bloch-paired":
        debug_loader, _ = preprocess_3d.get_bloch_loader(batch_size=1, device=device, lowres=args.lowres, same_contrast=False)
        train_loader, train_transform = preprocess_3d.get_bloch_loader(batch_size=args.batch_size, device=device, lowres=args.lowres, same_contrast=False)

    if args.debug:
        saver1 = mn.transforms.SaveImage(
            output_dir=os.path.join(args.logdir, args.name, "debug"),
            output_postfix="img1",
            separate_folder=False,
            print_log=False,
        )
        saver2 = mn.transforms.SaveImage(
            output_dir=os.path.join(args.logdir, args.name, "debug"),
            output_postfix="img2",
            separate_folder=False,
            print_log=False,
        )
        for i, batch in enumerate(debug_loader):
            if i > 5:
                break
            else:
                print(
                    "Image #1: ",
                    batch["image1"].shape,
                    "min={}".format(batch["image1"].min()),
                    "max={}".format(batch["image1"].max()),
                )
                saver1(
                    torch.Tensor(batch["image1"][0].cpu().float()),
                )
                print(
                    "Image #2: ",
                    batch["image2"].shape,
                    "min={}".format(batch["image2"].min()),
                    "max={}".format(batch["image2"].max()),
                )
                saver2(
                    torch.Tensor(batch["image2"][0].cpu().float())
                )

    return args, device, train_loader, train_transform


def main():
    args, device, train_loader, train_transform = set_up()
    run_model(args, device, train_loader, train_transform)


if __name__ == "__main__":
    main()
