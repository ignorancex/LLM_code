import argparse
import random
import comet_ml
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
import warmup_scheduler
from collections import defaultdict
import time
from pytorch_lightning.callbacks import Callback
import math
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from utils import get_model, get_dataset, get_experiment_name, get_criterion
from da import CutMix, MixUp
import torch.backends.cudnn as cudnn
parser = argparse.ArgumentParser()
parser.add_argument("--api-key", help="API Key for Comet.ml")
parser.add_argument("--dataset", default="c10", type=str, help="[c10, c100, svhn]")
parser.add_argument("--num-classes", default=10, type=int)
parser.add_argument("--model-name", default="res20", help="[vit]", type=str)
parser.add_argument("--patch", default=8, type=int)
parser.add_argument("--batch-size", default=256, type=int)
parser.add_argument("--eval-batch-size", default=1024, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--min-lr", default=1e-5, type=float)
parser.add_argument("--beta1", default=0.9, type=float)
parser.add_argument("--beta2", default=0.999, type=float)
parser.add_argument("--off-benchmark", action="store_true")
parser.add_argument("--max-epochs", default=200, type=int)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--weight-decay", default=5e-5, type=float)
parser.add_argument("--warmup-epoch", default=5, type=int)
parser.add_argument("--precision", default=32, type=int)
parser.add_argument("--autoaugment", action="store_true")
parser.add_argument("--criterion", default="ce")
parser.add_argument("--label-smoothing", action="store_true")
parser.add_argument("--smoothing", default=0.1, type=float)
parser.add_argument("--rcpaste", action="store_true")
parser.add_argument("--cutmix", action="store_true")
parser.add_argument("--mixup", action="store_true")
parser.add_argument("--dropout", default=0.0, type=float)
parser.add_argument("--head", default=12, type=int)
parser.add_argument("--num-layers", default=7, type=int)
parser.add_argument("--hidden", default=384, type=int)
parser.add_argument("--mlp-hidden", default=384, type=int)
parser.add_argument("--off-cls-token", action="store_true")
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--project-name", default="VisionTransformer")
parser.add_argument("--sam", action="store_true")
parser.add_argument("--zsam", action="store_true")
parser.add_argument("--alpha", default=0.95, type=float)
parser.add_argument("--fsam", action="store_true")
parser.add_argument("--samsung", action="store_true")
parser.add_argument("--sgd", action="store_true", help="SGD")

args = parser.parse_args()

class EpochTimerCallback(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()
    def on_train_epoch_end(self, trainer, pl_module, outputs):
        epoch_time = time.time() - self.epoch_start_time
        epoch = trainer.current_epoch
        print(f"Epoch {epoch} time: {epoch_time:.2f} seconds")
        pl_module.log("epoch_time_sec", epoch_time, prog_bar=True, on_epoch=True, logger=True)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)

seed_everything(args.seed)
args.benchmark = True if not args.off_benchmark else False
args.gpus = torch.cuda.device_count()
args.num_workers = 4*args.gpus if args.gpus else 8
args.is_cls_token = True if not args.off_cls_token else False
if not args.gpus:
    args.precision=32

if args.mlp_hidden != args.hidden*4:
    print(f"[INFO] In original paper, mlp_hidden(CURRENT:{args.mlp_hidden}) is set to: {args.hidden*4}(={args.hidden}*4)")

train_ds, test_ds = get_dataset(args)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args.eval_batch_size, num_workers=args.num_workers, pin_memory=True)

class Net(pl.LightningModule):
    def __init__(self, hparams):
        super(Net, self).__init__()
        self.hparams.update(vars(hparams))
        self.model = get_model(hparams)
        self.criterion = get_criterion(args)
        if hparams.cutmix:
            self.cutmix = CutMix(hparams.size, beta=1.)
        if hparams.mixup:
            self.mixup = MixUp(alpha=1.)
        self.log_image_flag = hparams.api_key is None
        self.automatic_optimization = False

    def forward(self, x):
        return self.model(x)

    def on_after_backward(self):
        tb_logger = None
        for logger in self.logger:
            if isinstance(logger, TensorBoardLogger):
                tb_logger = logger
                break
        if tb_logger is None:
            return

        for name, param in self.model.named_parameters():
                if param.grad is not None:
                    tb_logger.experiment.add_histogram(f"gradients/{name}", param.grad, self.global_step)

    class FriendlySAM(torch.optim.Optimizer):
        def __init__(self, params, base_optimizer, rho=0.05, sigma=1, lmbda=0.9, adaptive=False, **kwargs):
            assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

            defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
            super(Net.FriendlySAM, self).__init__(params, defaults)

            self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
            self.param_groups = self.base_optimizer.param_groups
            self.defaults.update(self.base_optimizer.defaults)
            self.sigma = sigma
            self.lmbda = lmbda
            print ('FriendlySAM sigma:', self.sigma, 'lambda:', self.lmbda)

        @torch.no_grad()
        def first_step(self, zero_grad=False):

            for group in self.param_groups:
                for p in group["params"]:      
                    if p.grad is None: continue       
                    grad = p.grad.clone()
                    if not "momentum" in self.state[p]:
                        self.state[p]["momentum"] = grad
                    else:
                        p.grad -= self.state[p]["momentum"] * self.sigma
                        self.state[p]["momentum"] = self.state[p]["momentum"] * self.lmbda + grad * (1 - self.lmbda)
                
            grad_norm = self._grad_norm()
            for group in self.param_groups:
                scale = group["rho"] / (grad_norm + 1e-12)

                for p in group["params"]:
                    if p.grad is None: continue
                    self.state[p]["old_p"] = p.data.clone()
                    e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                    p.add_(e_w)

            if zero_grad: self.zero_grad()

        @torch.no_grad()
        def second_step(self, zero_grad=False):
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None: continue
                    p.data = self.state[p]["old_p"]

            self.base_optimizer.step()

            if zero_grad: self.zero_grad()

        @torch.no_grad()
        def step(self, closure=None):
            assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
            closure = torch.enable_grad()(closure)  
            self.first_step(zero_grad=True)
            closure()
            self.second_step()

        def _grad_norm(self):
            shared_device = self.param_groups[0]["params"][0].device
            norm = torch.norm(
                        torch.stack([
                            ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                            for group in self.param_groups for p in group["params"]
                            if p.grad is not None
                        ]),
                        p=2
                )
            return norm

        def load_state_dict(self, state_dict):
            super().load_state_dict(state_dict)
            self.base_optimizer.param_groups = self.param_groups     

    class ASAM:
        def __init__(self, optimizer, model, rho=0.5, eta=0.01):
            self.optimizer = optimizer
            self.model = model
            self.rho = rho
            self.eta = eta
            self.state = defaultdict(dict)

        @torch.no_grad()
        def ascent_step(self):
            wgrads = []
            for n, p in self.model.named_parameters():
                if p.grad is None:
                    continue
                t_w = self.state[p].get("eps")
                if t_w is None:
                    t_w = torch.clone(p).detach()
                    self.state[p]["eps"] = t_w
                if 'weight' in n:
                    t_w[...] = p[...]
                    t_w.abs_().add_(self.eta)
                    p.grad.mul_(t_w)
                wgrads.append(torch.norm(p.grad, p=2))
            wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.e-16
            for n, p in self.model.named_parameters():
                if p.grad is None:
                    continue
                t_w = self.state[p].get("eps")
                if 'weight' in n:
                    p.grad.mul_(t_w)
                eps = t_w
                eps[...] = p.grad[...]
                eps.mul_(self.rho / wgrad_norm)
                p.add_(eps)
            self.optimizer.zero_grad()

        @torch.no_grad()
        def descent_step(self):
            for n, p in self.model.named_parameters():
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["eps"])
            self.optimizer.step()
            self.optimizer.zero_grad()

    class SAM(torch.optim.Optimizer):
        def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, hparams=None, **kwargs):
            assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
            defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
            super().__init__(params, defaults)
            self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
            self.param_groups = self.base_optimizer.param_groups
            self.defaults.update(self.base_optimizer.defaults)
            self.zsam_enabled = getattr(hparams, "zsam", False) if hparams else False
            self.zsam_alpha = getattr(hparams, "alpha", False) if hparams else False

        @torch.no_grad()
        def first_step(self, closure=None, zero_grad=False):
            grad_norm = self._grad_norm()

            for group in self.param_groups:
                rho = group["rho"]
                scale = rho / (grad_norm + 1e-12)
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    self.state[p]["old_p"] = p.data.clone()
                    grad = p.grad
                    if self.zsam_enabled and grad.ndim > 1:
                        grad_mean = grad.mean()
                        grad_std = grad.std() + 1e-12
                        z_norm_grad = (grad - grad_mean) / grad_std
                        threshold = torch.quantile(z_norm_grad.abs(), self.zsam_alpha) 
                        mask = z_norm_grad.abs() > threshold  
                        filtered_grad = grad.clone()
                        filtered_grad[~mask] = 0.0
                        e_w = rho * filtered_grad / (filtered_grad.norm(p=2) + 1e-12) if filtered_grad.norm(p=2) > 0 else grad * scale
                    else:
                        e_w = grad * scale
                    p.add_(e_w)

                    self.state[p]["prev_ew"] = e_w.clone()

            if zero_grad:
                self.zero_grad()

        @torch.no_grad()
        def second_step(self, zero_grad=False):
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    p.data = self.state[p]["old_p"]
            self.base_optimizer.step()
            if zero_grad:
                self.zero_grad()

        def step(self, closure=None):
            assert closure is not None, "SAM requires closure"
            closure = torch.enable_grad()(closure)
            self.first_step(closure=closure)
            closure()
            self.second_step()

        def _grad_norm(self):
            shared_device = self.param_groups[0]["params"][0].device
            norm = torch.norm(torch.stack([
                (torch.abs(p) if group["adaptive"] else 1.0) * p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]), p=2)
            return norm

    class Zharp(SAM):
        def __init__(self, params, hparams, rho=0.05, adaptive=False, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0, eps=1e-8):
            if hparams.sgd:
                base_opt = torch.optim.SGD
            else:
                base_opt = torch.optim.Adam
            super().__init__(
                params=params,
                base_optimizer=base_opt,
                rho=rho,
                adaptive=adaptive,
                hparams=hparams, 
                lr=lr,
                betas=betas,
                weight_decay=weight_decay,
                eps=eps
            )

    def configure_optimizers(self):
        if self.hparams.sam:
            print("[INFO] Using SAM + Modulated Adam")
            optimizer = self.Zharp(
                self.model.parameters(),
                hparams=self.hparams,
                lr=self.hparams.lr,
                betas=(self.hparams.beta1, self.hparams.beta2),
                weight_decay=self.hparams.weight_decay
            )
        
        # SAM_SAMSUNG
        elif self.hparams.fsam:
            print("[INFO] Using SAMSUNG + Modulated Adam")
            base_optimizer_cls = torch.optim.Adam
            optimizer = Net.FriendlySAM(
                self.model.parameters(),
                base_optimizer=base_optimizer_cls,
                lr=self.hparams.lr,
                betas=(self.hparams.beta1, self.hparams.beta2),
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.samsung:
            base_optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.hparams.lr, 
                betas=(self.hparams.beta1, self.hparams.beta2), 
                weight_decay=self.hparams.weight_decay
            )
            self.asam = Net.ASAM(
                optimizer=base_optimizer,
                model=self.model,
                rho=getattr(self.hparams, "rho", 0.5),
                eta=getattr(self.hparams, "eta", 0.01)
            )
            optimizer = self.asam.optimizer
        else:
            print("[INFO] Using Standard Adam (Modulation OFF)")
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, 
                                         betas=(self.hparams.beta1, self.hparams.beta2), 
                                         weight_decay=self.hparams.weight_decay)  

        if  self.hparams.samsung:
            optimizer = self.asam.optimizer
            scheduler = warmup_scheduler.GradualWarmupScheduler(
                optimizer,
                multiplier=1.0,
                total_epoch=self.hparams.warmup_epoch,
                after_scheduler=torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=30, gamma=0.75
                )
            )
        else:
            scheduler = warmup_scheduler.GradualWarmupScheduler(
                optimizer,
                multiplier=1.0,
                total_epoch=self.hparams.warmup_epoch,
                after_scheduler=torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=30, gamma=0.75
                )
            )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        img, label = batch
        optimizer = self.optimizers()

        use_mix = self.hparams.cutmix or self.hparams.mixup
        apply_mix = False
        if self.hparams.cutmix:
            img, label, rand_label, lambda_ = self.cutmix((img, label))
            apply_mix = True
        elif self.hparams.mixup:
            if np.random.rand() <= 0.8:
                img, label, rand_label, lambda_ = self.mixup((img, label))
                apply_mix = True
            else:
                rand_label = torch.zeros_like(label)
                lambda_ = 1.0 

        if self.hparams.sam or self.hparams.fsam:
            out = self(img)
            if apply_mix:
                loss = self.criterion(out, label) * lambda_ + self.criterion(out, rand_label) * (1. - lambda_)
            else:
                loss = self.criterion(out, label)
            self.manual_backward(loss)
            optimizer.first_step(zero_grad=True)

            out = self(img)
            if apply_mix:
                loss2 = self.criterion(out, label) * lambda_ + self.criterion(out, rand_label) * (1. - lambda_)
            else:
                loss2 = self.criterion(out, label)
            self.manual_backward(loss2)
            optimizer.second_step(zero_grad=True)
            final_loss = loss2
        
        elif self.hparams.samsung:
            out = self(img)
            if apply_mix:
                loss = self.criterion(out, label) * lambda_ + self.criterion(out, rand_label) * (1. - lambda_)
            else:
                loss = self.criterion(out, label)
            self.manual_backward(loss)
            self.asam.ascent_step()

            out = self(img)
            if apply_mix:
                loss2 = self.criterion(out, label) * lambda_ + self.criterion(out, rand_label) * (1. - lambda_)
            else:
                loss2 = self.criterion(out, label)
            self.manual_backward(loss2)
            self.asam.descent_step()
            final_loss = loss2
        else:
            out = self(img)
            if apply_mix:
                loss = self.criterion(out, label) * lambda_ + self.criterion(out, rand_label) * (1. - lambda_)
            else:
                loss = self.criterion(out, label)
            self.manual_backward(loss)
            optimizer.step()
            final_loss = loss

        optimizer.zero_grad()

        acc = torch.eq(out.argmax(-1), label).float().mean()
        self.log("loss", final_loss, on_step=False, on_epoch=True)
        self.log("acc", acc, on_step=False, on_epoch=True)
        return final_loss

    def _log_image(self, image):
        grid = torchvision.utils.make_grid(image, nrow=4)
        self.logger.experiment.log_image(grid.permute(1,2,0))
        print("[INFO] LOG IMAGE!!!")

    def validation_step(self, batch, batch_idx):
        img, label = batch
        out = self(img)
        loss = self.criterion(out, label)
        acc = torch.eq(out.argmax(-1), label).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True)
        return loss

    def training_epoch_end(self, outputs):
        optimizer = self.optimizers()
        self.log("lr", optimizer.param_groups[0]["lr"], on_epoch=True)


if __name__ == "__main__":
    experiment_name = get_experiment_name(args)
    if args.sam:
        experiment_name += "-SAM"
    if args.zsam:
        experiment_name += "-ZSAM"
    if args.samsung:
        experiment_name += "-SAMSUNG"
    if args.fsam:
        experiment_name += "-FSAM"
    if args.cutmix:
        experiment_name += "-CUTMIX"
    if args.mixup:
        experiment_name += "-MIXUP"
    if args.label_smoothing:
        experiment_name += "-LABELSMOOTH"
    if args.autoaugment:
        experiment_name += "-AUTOAUG"
    else:
        experiment_name += "-ETC"

    timer_callback = EpochTimerCallback()
    print(f"[INFO] Experiment Name: {experiment_name}")
    print(experiment_name)
    print("[INFO] Log with CSV")
    logger = pl.loggers.CSVLogger(
        save_dir="logs",
        name=experiment_name
    )
    refresh_rate = 1
    net = Net(args)
    trainer = pl.Trainer(precision=args.precision,fast_dev_run=args.dry_run, gpus=args.gpus, benchmark=args.benchmark, logger=logger, max_epochs=args.max_epochs, weights_summary="full", progress_bar_refresh_rate=refresh_rate, callbacks=[timer_callback])
    trainer.fit(model=net, train_dataloader=train_dl, val_dataloaders=test_dl)
    if not args.dry_run:
        model_path = f"weights/{experiment_name}.pth"
        torch.save(net.state_dict(), model_path)
