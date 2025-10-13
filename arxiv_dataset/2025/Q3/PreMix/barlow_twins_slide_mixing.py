import pandas as pd
import numpy as np
import hydra
import math
import torch
import torch.nn as nn
import torch.optim as optim

from omegaconf import DictConfig
from pathlib import Path
from functools import partial

from source.models import ModelFactory
from source.dataset import PretrainFeaturesDataset
from source.dataset_utils import pretrain_collate_features
from source.utils import seed_torch

# CutMix -> applying it to features
def cut_non_padding_regions_with_rand_indices(mask, lam):
    """
    Cut non-padding regions with random indices.

    Parameters:
    - mask (ndarray): Binary mask indicating padding regions ("True": zero padding region, "False": non-padding region)
    - lam (float): Sampled from a Beta distribution (1,1)

    Returns:
    - start_idx (int): Starting index of the cut region.
    - end_idx (int): Ending index of the cut region.
    - cut_len (int): Length of the cut non-padding region.
    - non_padding_regions_len (int): Total length of non-padding regions.
    """

    ratio = np.sqrt(1 - lam) # ratio of cutting
    non_pad_len = (~mask).sum().item() # count "False" in mask list
    cut_len = int(non_pad_len * ratio)
    
    # Initialize the indices
    start_idx, end_idx = None, None
    
    while True:
        cr = torch.randint(cut_len // 2, non_pad_len - cut_len // 2, size=(1,)).item()
        start_idx = cr - cut_len // 2
        end_idx = cr + cut_len // 2

        # Check if the selected indices are not masked
        if not mask[start_idx: end_idx].any():
            break

    return start_idx, end_idx, cut_len, non_pad_len

def slide_mixing(x, mask, lam):
    # x: [BS, R, 192]
    if lam is None:
        lam = np.random.beta(1., 1., size=[x.shape[0],1,1]).astype(np.float32)
        lam = torch.from_numpy(lam).cuda()
    new_lams = lam * 0.8 + 0.1 # the mixing will be within the range of 10% to 90%

    x_flip = x.flip(0).clone() # flip the batch size order [A, B, C, D] -> [D, B, C, A]
    mask_flip = mask.flip(0).clone()
    mask_final = mask.clone()

    for i in range(x.shape[0]):
        # x[i] = [R, 192]
        # mask[i] = [R]
        # new_lams[i] = [1, 1]

        start_idx, end_idx, cut_len, non_pad_len = cut_non_padding_regions_with_rand_indices(mask[i], new_lams[i])

        new_lams[i] = (1. - (cut_len / non_pad_len)) # ratio of cutting

        # mask: ("True": zero padding region, "False": non-padding region)
        if (not mask[i, start_idx:end_idx].any()) and (not mask_flip[i, start_idx:end_idx].any()): # check if its not a mask in both features1 and features1.flip()
            x[i:i+1, start_idx:end_idx, :] = x_flip[i:i+1, start_idx:end_idx, :]

    return x, mask_final, new_lams

def adjust_learning_rate(cfg, optimizer, loader, step):
    max_steps = cfg.nepochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = cfg.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * cfg.BarlowTwins.optimizer.learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * cfg.BarlowTwins.optimizer.learning_rate_biases

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BarlowTwins(nn.Module):
    def __init__(self, backbone, batch_size, lambd):
        super().__init__()
        self.backbone = backbone
        self.batch_size = batch_size
        self.lambd = lambd

        # projector
        sizes = [192, 768, 768, 768]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)
    
    def original_loss(self, z1, z2):

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        c.div_(self.batch_size)

        # calculate loss
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss

    def mixing_loss(self, z1, z2, lam1, lam2):
        # calculate loss for original labels
        original_loss = self.original_loss(z1, z2)

        z2_flipped = z2.flip(0)

        # calculate loss for flipped labels
        flipped_loss = self.original_loss(z1, z2_flipped)

        # mix the losses using the provided weights (lam1 and lam2)
        mixed_loss = lam1 * original_loss + lam2 * flipped_loss
        return mixed_loss.mean()


    def forward(self, features1, features2, mask1, mask2):
        lam = np.random.beta(1., 1., size=[features1.shape[0],1,1]).astype(np.float32)

        swap = torch.rand([1]).cuda()
        if swap[0] > 0.5:
            features1, features2 = features2, features1
            mask1, mask2 = mask2, mask1

        features1_mix, mask1_mix, new_lam = slide_mixing(features1.clone(), mask1, lam) # intra-batch mixing
        features2_mix, mask2_mix, new_lam = slide_mixing(features2.clone(), mask2, lam) # intra-batch mixing

        # z1 = self.projector(self.backbone(features1, mask1, pretrain=True, mixup=False, manifold_mixup=False, mixup_type="random"))
        # z2 = self.projector(self.backbone(features2, mask2, pretrain=True, mixup=False, manifold_mixup=False, mixup_type="random"))
        # z1_mix = self.projector(self.backbone(features1_mix, mask1_mix, pretrain=True, mixup=False, manifold_mixup=False, mixup_type="random"))
        # z2_mix = self.projector(self.backbone(features2_mix, mask2_mix, pretrain=True, mixup=False, manifold_mixup=False, mixup_type="random"))

        z1 = self.projector(self.backbone(features1, mask1))
        z2 = self.projector(self.backbone(features2, mask2))
        z1_mix = self.projector(self.backbone(features1_mix, mask1_mix))
        z2_mix = self.projector(self.backbone(features2_mix, mask2_mix))

        if new_lam is not None:
            # only get new_lam from features2
            lam = new_lam

        source_loss = self.original_loss(z1, z2)

        lam = torch.from_numpy(lam).cuda()
        mix_source_loss = self.mixing_loss(z2_mix, z1, lam, 1-lam)

        common = torch.min(lam, 1-lam.flip(0).clone()) + torch.min(1-lam, lam.flip(0).clone())
        mix_loss = self.mixing_loss(z2_mix, z1_mix, 1/(1+common), common/(1+common))

        return source_loss, mix_source_loss, mix_loss

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = self.backbone.to(device)
        self.projector = self.projector.to(device)
        self.bn = self.bn.to(device)

class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)


    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])

@hydra.main(version_base="1.2.0", config_path="config/training", config_name="pretrain")
def main(cfg: DictConfig):

    seed_torch(seed=cfg.seed)

    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    output_dir = Path(cfg.output_dir, cfg.dataset_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(output_dir, "checkpoints", cfg.level)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    features_dir = Path(output_dir, "features", cfg.experiment_name, cfg.level, "slide")

    # Data
    print('==> Preparing data..')
    train_df_path = Path(cfg.data_dir, cfg.dataset_name) / f"{cfg.csv_name}.csv"

    train_df = pd.read_csv(train_df_path)

    train_set = PretrainFeaturesDataset(
        train_df,
        features_dir,
        augmentation=cfg.augmentation,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=partial(pretrain_collate_features),
        num_workers=8,
        pin_memory=True,
    )

    print('==> Building model..')
    backbone = ModelFactory(cfg.level, cfg.model).get_model()
    print(backbone)

    model = BarlowTwins(backbone, batch_size=cfg.batch_size, lambd=cfg.BarlowTwins.lambd)
    model.relocate()
    print(model)

    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    optimizer = LARS(parameters, lr=0, weight_decay=cfg.BarlowTwins.optimizer.weight_decay, weight_decay_filter=True, lars_adaptation_filter=True)

    scaler = torch.cuda.amp.GradScaler()

    file_path = log_dir / f"{cfg.model_name}.txt"
    with open(file_path, "a") as f:
        header = "Epoch, Source Loss, Mix Source Loss, Mix Loss, Total Loss\n"
        f.write(header)
    
    loss_hyperparameters = cfg.BarlowTwins.loss

    # Training
    for epoch in range(1, cfg.nepochs+1):
        print('\nEpoch: %d' % epoch)
        model.train()
        train_source_loss = 0
        train_mix_source_loss = 0
        train_mix_loss = 0
        train_total_loss = 0

        for batch_idx, (slide_id, features1, features2, mask1, mask2) in enumerate(train_loader):
            features1, features2 = features1.cuda(non_blocking=True), features2.cuda(non_blocking=True)
            mask1, mask2 = mask1.cuda(non_blocking=True), mask2.cuda(non_blocking=True)

            adjust_learning_rate(cfg, optimizer, train_loader, batch_idx)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                source_loss, mix_source_loss, mix_loss = model.forward(features1, features2, mask1, mask2)

                # adjusting slide mixing loss wrt loss hyperparameters (best combination alpha, beta, gamma: 1, 0.5, 0.5 (refer to Table 7 in the paper for the ablation study)
                source_loss, mix_source_loss, mix_loss = source_loss * loss_hyperparameters.alpha, mix_source_loss * loss_hyperparameters.beta, mix_loss * loss_hyperparameters.gamma
                loss = source_loss + mix_source_loss + mix_loss

                train_source_loss += source_loss.item()
                train_mix_source_loss += mix_source_loss.item()
                train_mix_loss += mix_loss.item()
                train_total_loss += loss.item()
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_source_loss /= len(train_loader)
        train_mix_source_loss /= len(train_loader)
        train_mix_loss /= len(train_loader)
        train_total_loss /= len(train_loader)

        print(f"Epoch: {epoch}, Source Loss: {train_source_loss}, Mix Source Loss: {train_mix_source_loss}, Mix Loss: {train_mix_loss}, Total Loss: {train_total_loss}")
        
        with open(file_path, "a") as f:

            formatted_line = f"{epoch}, {train_source_loss:.6f}, {train_mix_source_loss:.6f}, {train_mix_loss:.6f}, {train_total_loss:.6f}\n"
            f.write(formatted_line)

            if epoch % cfg.save_every == 0:
                print("Saving..")
                state = {
                    "epoch": epoch,
                    "net": model.backbone.state_dict(),
                    "source_loss": train_source_loss,
                    "mix_source_loss": train_mix_source_loss,
                    "mix_loss": train_mix_loss,
                    "total_loss": train_total_loss,
                }
                torch.save(state, Path(checkpoint_dir, f"{cfg.model_name}_{epoch}.pth"))

if __name__ == "__main__":
    
    main()