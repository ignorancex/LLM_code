import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import hydra

from omegaconf import DictConfig
from pathlib import Path
from functools import partial

from source.models import ModelFactory
from source.dataset import PretrainFeaturesDataset
from source.dataset_utils import pretrain_collate_features
from source.utils import seed_torch
from loss.nt_xent import NTXentLoss

class SimCLR(nn.Module):
    def __init__(self, backbone, batch_size, device, temperature, use_cosine_similarity):
        super().__init__()
        self.backbone = backbone
        self.batch_size = batch_size
        self.device = device
        self.temperature = temperature
        self.use_cosine_similarity = use_cosine_similarity

        self.ntxentloss = NTXentLoss(self.device, self.batch_size, self.temperature, self.use_cosine_similarity)

        self.projector = nn.Sequential(
            nn.Linear(192, 96)
        )

    def forward(self, features1, features2, mask1, mask2, pretrain=False, mixup=False, manifold_mixup=False):
        
        z1 = self.projector(self.backbone(features1, mask1))
        z2 = self.projector(self.backbone(features2, mask2))
        
        z1 = F.normalize(z1, p=2, dim=1) # L2-normalization: dividing them by their L2-norm (euclidean norm) -> square root of the sum of the squares of its elements
        z2 = F.normalize(z2, p=2, dim=1) # L2-normalization: dividing them by their L2-norm (euclidean norm) -> square root of the sum of the squares of its elements

        z1 = torch.clamp(z1, min=1e-5, max=1. - 1e-5)
        z2 = torch.clamp(z2, min=1e-5, max=1. - 1e-5)

        loss = self.ntxentloss(z1, z2)
        return loss

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = self.backbone.to(device)
        self.projector = self.projector.to(device)

@hydra.main(version_base="1.2.0", config_path="config/training", config_name="pretrain")
def main(cfg: DictConfig):

    seed_torch(seed=cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    model = SimCLR(backbone, cfg.batch_size, device, cfg.SimCLR.temperature, cfg.SimCLR.use_cosine_similarity)
    model.relocate()
    print(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SimCLR.optimizer.lr, betas=(0.5, 0.9), weight_decay=cfg.SimCLR.optimizer.wd)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.nepochs, 0.000005)

    scaler = torch.cuda.amp.GradScaler()

    file_path = log_dir / f"{cfg.model_name}.txt"

    # Training
    for epoch in range(cfg.nepochs):
        print('\nEpoch: %d' % epoch)
        model.train()
        train_loss = 0
        for batch_idx, (slide_id, features1, features2, mask1, mask2) in enumerate(train_loader):
            features1, features2 = features1.cuda(non_blocking=True), features2.cuda(non_blocking=True)
            mask1, mask2 = mask1.cuda(non_blocking=True), mask2.cuda(non_blocking=True)
            
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                loss = model.forward(features1, features2, mask1, mask2)
                train_loss += loss.item()
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        train_loss /= len(train_loader)
        print(f"Epoch: {epoch}, Train Loss: {train_loss}")
        with open(file_path, "a") as f:
            f.write(str(epoch)+':'+str(train_loss)+'\n')
        
        if epoch % cfg.save_every == 0:
            print("Saving..")
            state = {
                "net": model.backbone.state_dict(),
                "train_loss": train_loss,
                "epoch": epoch,
            }
            torch.save(state, Path(checkpoint_dir, f"{cfg.model_name}_{epoch}.pth"))
        
        # scheduler.step()

if __name__ == "__main__":

    main()
