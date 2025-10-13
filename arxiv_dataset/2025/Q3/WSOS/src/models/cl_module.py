import os
from typing import Any, Dict
import torch
from lightning import LightningModule
from torch import nn

from src.models.components.ae import AE
from src.models.components.ae_gmm import AEGMM
from src.losses.infonce import InfoNCELoss
from src.shared.SinkhornKMeans import SinkhornKMeans
from src.utils.utils import normalize_images


class ContrastiveLearningModule(LightningModule):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 compile: bool,
                 net: AE,
                 clusters_list,
                 initial_temperature: float = 0.07):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = net
        self.clusters_list = clusters_list
        self.criterion = nn.BCEWithLogitsLoss()
        self.temperature = nn.Parameter(torch.tensor(initial_temperature))
        self.infonceloss = InfoNCELoss(self.temperature)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/total_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def forward(self, x):
        _, z = self.net.encoder(x)
        return z

    def contrastive_loss(self, patches):
        patches = patches[0]

        patches = normalize_images(patches)
        # patches is assumed to be a tensor of shape (batch_size, n_patches, channels, width, height)
        batch_size, n_patches, channels, width, height = patches.size()

        # Flatten patches to (batch_size * n_patches, channels, width, height)
        patches = patches.view(-1, channels, width, height)

        # Encode patches to obtain embeddings
        z = self.net.encoder(patches.float())  # Assuming self.net.encoder is your encoder model

        # Reshape z to (batch_size, n_patches, latent_dim)
        z = z.view(batch_size, n_patches, -1)

        # Randomly select two different patches from each image
        idx1, idx2 = torch.randperm(n_patches)[:2]
        # just pick one patch
        z_1 = z[:, idx1, :]  # First patch in each set
        z_2 = z[:, idx2, :]  # Second ptch in each set

        return self.infonceloss(z_1, z_2)

    def training_step(self, batch, batch_idx):
        loss = self.contrastive_loss(batch)
        self.log('train/total_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.contrastive_loss(batch)
        self.log('val/total_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.contrastive_loss(batch)
        self.log('test/total_loss', loss)
        return loss

    def get_features(self, data_loader):
        features = []
        for batch_idx, batch in enumerate(data_loader):
            patches = batch[0]
            patches = normalize_images(patches)
            batch_size, n_patches, channels, width, height = patches.size()
            # Flatten patches to (batch_size * n_patches, channels, width, height)
            patches = patches.view(-1, channels, width, height)
            features.extend(self.net.encoder(patches.to(self.device)))
        features = torch.stack(features)
        return features.cpu().numpy()

    def on_test_end(self):
        training_features = self.get_features(self.trainer.datamodule.train_dataloader())
        validation_features = self.get_features(self.trainer.datamodule.val_dataloader())
        test_features = self.get_features(self.trainer.datamodule.test_dataloader())

        all_features = torch.cat(
            [torch.tensor(training_features), torch.tensor(validation_features), torch.tensor(test_features)])

        for cluster_num in self.clusters_list:
            # kmeans = KMeans(n_clusters=cluster_num, n_init=10)
            kmeans = SinkhornKMeans(n_clusters=cluster_num, epsilon=.1, max_iter=10000, tol=1e-4,device=self.device)
            kmeans = kmeans.fit(all_features.detach().cpu().numpy())
            ae_kmeans = AEGMM(self.net, kmeans)
            directory_path = os.path.join(self.logger.save_dir, 'ae_gmm')
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            torch.save(ae_kmeans, os.path.join(directory_path, f'{cluster_num}'))
            print(f"Saved ae_kmeans in the path: {directory_path}")
