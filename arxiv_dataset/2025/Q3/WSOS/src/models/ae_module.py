import logging
import os
from typing import Tuple, Dict, Any

import wandb
import torch
from lightning import LightningModule
from sklearn.metrics import normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import numpy as np
from torchvision.utils import make_grid

from src import ForegroundTextureDataModule
from src.data.birds_datamodule import BirdsDataModule
from src.data.sss_datamodule  import SSSDataModule
from src.losses import DSW, ProjNet
from src.models.components import AE
from src.models.components.ae_gmm import AEGMM
from src.shared.SinkhornKMeans import SinkhornKMeans
from src.utils.images_utils import get_random_closed_shape_mask, add_smooth_intensity_to_masks, \
    unify_images_intensities
from src.utils.utils import mse_dsw, plotly_to_tensor, normalize_images, apply_colormap_to_tensor


class AEModule(LightningModule):
    def __init__(self,
                 net: AE,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 compile: bool,
                 testing_dataset: ForegroundTextureDataModule,
                 clusters_list,
                 type='',
                 psudo_colormap=None
                 ):
        """
        aug tells the class to use augmentation to make AE robust to fg objects.
        if aug is False, then it will assume that the datamodule passed has a type of ForegroundTextureDataModule and
        it will ignore the passed testing_dataset.
        """
        super(AEModule, self).__init__()
        self.net = net
        self.save_hyperparameters(logger=False)
        self.reconstruction_loss = mse_dsw
        self.divergence_loss = self.create_dsw(net.latent_dim)
        self.type = type
        self.testing_dataset = testing_dataset
        self.clusters_list = clusters_list
        self.psudo_colormap = psudo_colormap

    def setup(self, stage: str) -> None:
        super().setup(stage)
        if stage == 'train':
            if (self.type != 'occlusion' and
                    not (isinstance(self.trainer.datamodule, ForegroundTextureDataModule) or
                         isinstance(self.trainer.datamodule, BirdsDataModule))):
                raise RuntimeError(
                    f'if type is not `occlusion` you need to pass datamodule has to be `ForegroundTextureDataModule` or '
                    f'`BirdsDataModule`')
        if stage == 'test':
                self.testing_dataset = self.trainer.datamodule

    def create_dsw(self, input_dim: int) -> DSW:
        embedding_norm = 1.0
        projnet = ProjNet(input_dim)
        op_projnet = torch.optim.Adam(projnet.parameters(), lr=1)
        p = 2
        dsw = DSW(None, embedding_norm, 1000, projnet, op_projnet, p,
                  300, .01)
        return dsw

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latents = self.net.encoder(x.float())
        return self.net.decoder(latents), latents

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        loss = self.pass_to_model(batch, 'train')
        return loss

    def pass_to_model(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        if self.type == 'occlusion':
            bg_images, _ = batch
            shape_bg_images = self.generate_random_shape(bg_images)
            bg_images = bg_images / 255
            shape_bg_images = shape_bg_images / 255
            bg_images_hat, latents = self.forward(bg_images)
            _, latents_shape = self.forward(shape_bg_images)
            latent_difference = 100 * self.reconstruction_loss(latents, latents_shape)
            self.log(f"{stage}/latent_difference", latent_difference,
                     on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            specific_losses = latent_difference
        elif self.type == 'divergence':
            fg_images, bg_images, masks, bg_bg_labels, bg_fg_labels, _ = batch
            fg_images = fg_images / 255
            bg_images = bg_images / 255
            bg_images_hat, latents = self.forward(bg_images)
            fg_images_hat, latents_foreground = self.forward(fg_images)
            if self.current_epoch > 3:
                divergence_loss = self.divergence_loss(latents, latents_foreground)
                # variance_loss = (torch.sqrt(torch.var(bg_images, dim=(0, 1))).sum()
                #                  - torch.sqrt(torch.var(latents, dim=(0,))).sum()) ** 2
            else:
                divergence_loss = 0
                variance_loss = 0
            variance_loss = self.reconstruction_loss(fg_images_hat, fg_images.float())
            specific_losses = divergence_loss + variance_loss
            self.log(f"{stage}/divergence_loss", divergence_loss,
                     on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{stage}/variance_loss", variance_loss,
                     on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        else:
            if isinstance(self.trainer.datamodule, SSSDataModule) and stage=='test':
                images = batch[0]
                masks = batch[1]
                is_background = masks.sum((1,2,3))  == 0
                bg_images = images[is_background]
            else:
                bg_images = batch[1]
            bg_images = normalize_images(bg_images)
            bg_images_hat, latents = self.forward(bg_images)
            specific_losses = 0
        reconstruction_loss = self.reconstruction_loss(bg_images_hat, bg_images.float())
        total_loss = specific_losses + reconstruction_loss
        self.log(f"{stage}/total_loss", total_loss,
                 on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/reconstruction_loss", reconstruction_loss,
                 on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        if stage == 'val' or stage == 'test':
            self.plot_images(bg_images[:32], bg_images_hat[:32], stage=stage)
        return total_loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx) -> torch.Tensor:
        loss = self.pass_to_model(batch, 'val')
        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx) -> torch.Tensor:
        loss = self.pass_to_model(batch, 'test')
        return loss

    def on_test_start(self) -> None:
        logger = logging.getLogger("lightning.pytorch")
        if logger:
            logger.setLevel(logging.ERROR)
        else:
            print("Logger 'lightning.pytorch' does not exist.")

    def on_test_end(self) -> None:
        if isinstance(self.trainer.datamodule, ForegroundTextureDataModule):
            training_features, training_labels = self.get_features_and_labels(self.testing_dataset.train_dataloader(),
                                                                              'bg')
            validation_features, validation_labels = self.get_features_and_labels(self.testing_dataset.val_dataloader(),
                                                                                  'both')
            test_features, test_labels = self.get_features_and_labels(self.testing_dataset.test_dataloader(), 'both')
        elif isinstance(self.trainer.datamodule, BirdsDataModule)  or isinstance(self.trainer.datamodule, SSSDataModule):
            training_features, _ = self.get_features_and_labels(self.testing_dataset.train_dataloader(), 'bg')
            validation_features, _ = self.get_features_and_labels(self.testing_dataset.val_dataloader(), 'both')
            test_features, _ = self.get_features_and_labels(self.testing_dataset.test_dataloader(), 'both')
        else:
            raise NotImplementedError(f'{type(self.trainer.datamodule)} is not supported')

        nmi_values = []
        for cluster_num in self.clusters_list:
            # gmm = KMeans(cluster_num, n_init=10)
            gmm = SinkhornKMeans(n_clusters=cluster_num,max_iter=10000, epsilon=1, tol=1e-8, device=self.device)
            gmm = gmm.fit(training_features)
            ae_gmm = AEGMM(self.net, gmm)
            directory_path = os.path.join(self.trainer.log_dir, 'ae_gmm')
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            torch.save(ae_gmm, self.trainer.log_dir + '/ae_gmm/' + str(cluster_num))
            print(f"saved ae_gmm in the path: {self.trainer.log_dir}")
            if isinstance(self.trainer.datamodule, ForegroundTextureDataModule):
                labels_hat = gmm.predict(np.concatenate((validation_features, test_features)))
                nmi_value = normalized_mutual_info_score(labels_hat, np.concatenate((validation_labels, test_labels)),
                                                         average_method='max')
                nmi_values.append(nmi_value)
        if isinstance(self.trainer.datamodule, ForegroundTextureDataModule):
            trace = go.Scatter(
                x=np.array(self.clusters_list), y=np.array(nmi_values), mode='lines+markers', marker_symbol='star')
            layout = go.Layout(title='NMI Scores', xaxis=dict(title='clusters number'), yaxis=dict(title='NMI'))
            fig = go.Figure(data=[trace], layout=layout)
            logger = self.logger.experiment

            if hasattr(logger, 'add_image'):
                logger.add_image("test/divergence", plotly_to_tensor(fig), global_step=self.current_epoch)
            else:
                # assume its w&b
                logger.log({"test/divergence": fig}, step=self.current_epoch)

    def plot_images(self, x: torch.Tensor, x_hat: torch.Tensor, stage: str):
        combined = torch.concat((x, x_hat))
        combined = make_grid(combined, x.shape[0])
        if self.psudo_colormap is not None:
            combined = apply_colormap_to_tensor(combined.cpu(), self.psudo_colormap)
        logger = self.logger.experiment

        if hasattr(logger, 'add_image'):
            logger.add_image(f'{stage}/True_Estimated', combined.cpu(), global_step=self.current_epoch)
        else:
            # assume its w&b
            logger.log({f'{stage}/True_Estimated': wandb.Image(combined.cpu())})

    def get_features(self, data_loader, task='both'):
        features = []
        for batch_idx, batch in enumerate(data_loader):
            fg_images, bg_images, masks = batch
            if task == 'both':
                images = torch.cat([fg_images, bg_images])
            elif task == 'fg':
                images = fg_images
            elif task == 'bg':
                images = bg_images
            else:
                raise RuntimeError(f'{task} is not a valid value for `task` ')
            images = normalize_images(images)
            features.extend(self.net.encoder(images.float().to(self.device)))
        features = torch.stack(features)
        return features.cpu().numpy()

    def get_features_and_labels(self, data_loader, task='both'):
        labels, features = [], []
        for batch_idx, batch in enumerate(data_loader):
            if len(batch) == 6:
                fg_images, bg_images, masks, bg_bg_labels, bg_fg_labels, _ = batch
                if task == 'both':
                    images = torch.cat([fg_images, bg_images])
                    labels_combined = torch.cat([bg_fg_labels, bg_bg_labels])
                elif task == 'fg':
                    images = fg_images
                    labels_combined = bg_fg_labels
                elif task == 'bg':
                    images = bg_images
                    labels_combined = bg_bg_labels
                else:
                    raise RuntimeError(f'{task} is not a valid value for `task` ')
            elif len(batch) == 3:
                images, masks, labels_combined = batch
            images = normalize_images(images)
            features.extend(self.net.encoder(images.float().to(self.device)))
            labels.extend(labels_combined)
        features = torch.stack(features)
        labels = torch.stack(labels)
        return features.cpu().numpy(), labels.cpu().numpy()

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

    def generate_random_shape(self, images):
        shape_masks = torch.stack([get_random_closed_shape_mask(image.shape[1:]) for image in images])
        shapes_with_intensity = add_smooth_intensity_to_masks(shape_masks, torch.randint(1, 100, size=(
            shape_masks.shape[0], 1, 1, 1)))
        shapes = [
            unify_images_intensities(shape_with_intensity.unsqueeze(0), torch.zeros(1, ),
                                     torch.randint(100, 200, size=(1,)), torch.randint(200, 255, size=(1,)))
            for shape_with_intensity in shapes_with_intensity]
        shapes = torch.concat(shapes)
        shape_masks = shape_masks.to(images)
        shapes = shapes.to(images)
        return shapes * shape_masks + (1 - shape_masks) * images
