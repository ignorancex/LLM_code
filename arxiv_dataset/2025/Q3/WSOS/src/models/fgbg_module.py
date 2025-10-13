import logging
from collections import namedtuple
from typing import Any, Dict, Tuple, List

import wandb
import torch
from lightning import LightningModule
import torchmetrics.classification as evalution_criteria
import plotly.express as px
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torch import nn

from src.data.sss_datamodule  import SSSDataModule
from src.losses.infonce import InfoNCELoss
from src.losses import DSW, ProjNet
from src.losses.ebsw import EBSW
from src.losses.shared import calculate_batch_certainty
from src.utils import RankedLogger

from src.utils.images_utils import repeat_to_match_size
from src.utils.utils import load_net_state_from_checkpoint, mse_dsw, plotly_to_tensor, get_background_clusters_num, \
    apply_colormap_to_tensor, normalize_images

log = RankedLogger(__name__, rank_zero_only=True)


class FgBgLitModule(LightningModule):
    CalculateLossesParams = namedtuple('CalculateLossesParams', ['pairs', 'logits_bg', 'bg_estimated_masks', 'stage'])

    def __init__(
            self,
            net: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            compile: bool,
            num_projections: int,
            max_iter_dswd: int,
            sw_encoder,
            sw_encoder_checkpoint: str,
            regularization_lambda_dswd: float,
            image_size,
            input_channels: int,
            embedding_dim: float,
            masks_dim: int = 1,
            divergence_loss='dswd',
            real_counterfactual_distance='fid',
            cirtic_task: str = None,
            psudo_colormap: str = None,
            images_to_plot_num: int = 32
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.real_counterfactual_distance = real_counterfactual_distance
        self.encoder = None
        self.net = net
        self.image_size = image_size
        self.input_channels = input_channels
        self.divergence_loss = divergence_loss
        self.cirtic_task = cirtic_task
        self.masks_dim = masks_dim
        if sw_encoder:
            assert sw_encoder_checkpoint, 'sw_encoder checkpoint shound not be None'
            self.encoder = sw_encoder
            self.encoder.load_state_dict(
                load_net_state_from_checkpoint(sw_encoder_checkpoint, self.encoder.state_dict(), 'net'))
            self.encoder = self.encoder.encoder
            self.transforms_set = transforms.Compose([
                transforms.RandomResizedCrop(size=(64, 64), scale=(.8, 1), ratio=(1, 1), antialias=True),
                transforms.RandomHorizontalFlip(),
            ])
            self.bce_loss = nn.CrossEntropyLoss()
            temperature = torch.load(sw_encoder_checkpoint)['state_dict']['temperature'] if self.encoder else None
            self.infonceloss = InfoNCELoss(nn.Parameter(temperature))
            self.encoder.train()
            for param in self.encoder.parameters():
                param.requires_grad = True
        self.images_num = images_to_plot_num
        p = 2
        self.bg_criterion = mse_dsw
        if divergence_loss == 'dswd':
            encoder = None
            projnet = ProjNet(embedding_dim)
            embedding_norm = 1.0
            op_projnet = torch.optim.Adam(projnet.parameters(), lr=1)
            self.fg_criterion = DSW(encoder, embedding_norm, num_projections, projnet, op_projnet, p,
                                    max_iter_dswd, regularization_lambda_dswd)
        elif divergence_loss == 'ebsw':
            self.fg_criterion = EBSW(num_projections, p, self.encoder)
        else:
            self.fg_criterion = None
        self.pixel_wise_entropy = calculate_batch_certainty

        self.avg_precision = [evalution_criteria.BinaryAveragePrecision() for _ in range(self.masks_dim)]
        self.confusion_matrix = [evalution_criteria.BinaryConfusionMatrix() for _ in range(self.masks_dim)]
        self.auc_roc = [evalution_criteria.BinaryAUROC() for _ in range(self.masks_dim)]

        self.classification_layer = None
        self.critic_loss = 0
        self.psudo_colormap = psudo_colormap

    def on_train_end(self) -> None:
        return super().on_train_end()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def generate_predictions_targets(self, fg_images: torch.Tensor, bg_images: torch.Tensor,
                                     fg_bg_labels: torch.Tensor, bg_bg_labels: torch.Tensor,
                                     fg_estimated_masks: torch.Tensor) -> List[dict]:
        unique_bg_labels = torch.unique(fg_bg_labels)
        predicted_targets_pairs = []
        self.log(f"{self.trainer.state.stage}/len_bg_labels", len(unique_bg_labels), sync_dist=True, on_epoch=True)
        results = []
        for bg_label in torch.sort(unique_bg_labels)[0]:
            same_bg = bg_images[bg_bg_labels == bg_label][:10]
            if len(same_bg) < 10:
                num_black_images = 10 - len(same_bg)
                black_images = torch.zeros(num_black_images, *same_bg.shape[1:], device=self.device)
                same_bg = torch.cat([same_bg, black_images], dim=0)
            results.append(same_bg)
        results = torch.cat(results)
        logger = self.logger.experiment
        # logger.log({f"{self.trainer.state.stage}/background_clusters": wandb.Image(self.log_images(results, 10))})
        for bg_label in unique_bg_labels:
            fg_with_different_bg = fg_images[fg_bg_labels != bg_label]
            same_bg = bg_images[bg_bg_labels == bg_label]
            # conditional_fg_masks and fg_with_different_bg have the same shape
            conditional_fg_masks = fg_estimated_masks[fg_bg_labels != bg_label]
            if fg_with_different_bg.shape[0] == 0 or same_bg.shape[0] == 0 or conditional_fg_masks.shape[0] == 0:
                continue
            same_bg, fg_with_different_bg, conditional_fg_masks = repeat_to_match_size(same_bg, fg_with_different_bg,
                                                                                       conditional_fg_masks)
            counterfactuals = conditional_fg_masks * fg_with_different_bg + (
                    1 - conditional_fg_masks) * same_bg
            pair = {}
            pair['counterfactuals_for_quantile'] = counterfactuals
            pair['bg_for_quantile'] = same_bg
            pair['mask_for_quantile'] = conditional_fg_masks
            fg_bg_matching_labels = fg_bg_labels == bg_label
            fg_with_same_bg = fg_images[fg_bg_matching_labels]
            pair['input'] = fg_with_different_bg.detach().cpu()
            pair['counterfactuals_input'] = counterfactuals.detach().cpu()
            counterfactuals, fg_with_same_bg, _ = repeat_to_match_size(counterfactuals, fg_with_same_bg)
            counterfactuals, same_bg, conditional_fg_masks = repeat_to_match_size(counterfactuals, same_bg,
                                                                                  conditional_fg_masks)

            counterfactuals, fg_with_different_bg, _ = repeat_to_match_size(counterfactuals, fg_with_different_bg)

            if counterfactuals.shape[0] == 0:
                continue
            # Create a new dictionary for each bg_label
            pair['predicted'] = counterfactuals
            pair['target'] = fg_with_same_bg
            pair['background'] = same_bg
            pair['masks'] = conditional_fg_masks
            pair['bg_labels_counterfactuals'] = bg_label * torch.ones(counterfactuals.shape[0], device=self.device)
            pair['bg_labels_target'] = bg_label * torch.ones(fg_with_same_bg.shape[0], device=self.device)
            pair['real_different_cluster'] = fg_with_different_bg
            pair['bg_label'] = bg_label
            predicted_targets_pairs.append(pair)
            # if fg_with_same_bg.shape[0] >= .49 * len(fg_images):
            #     break

        return predicted_targets_pairs

    def cl_loss(self, sample):
        z1 = self.encoder(self.transforms_set(sample))
        z2 = self.encoder(self.transforms_set(sample))
        return self.infonceloss(z1, z2)

    def calculate_cl_and_bce_losses(self, predicted_targets_pairs: List[dict]):
        cl_loss = 0
        bce_loss = 0
        counterfactual_acc = real_acc = 0
        for pair in predicted_targets_pairs:
            first_sample = pair['predicted'].float()
            second_sample = pair['target'].float()
            z_counterfactual_labels = pair['bg_labels_counterfactuals'].long()
            z_real_labels = pair['bg_labels_target'].long()
            cl_loss += self.cl_loss(first_sample.detach() * 255) + self.cl_loss(second_sample.detach() * 255)
            with torch.no_grad():
                z_counterfactual = self.encoder(first_sample * 255)
                z_real = self.encoder(second_sample * 255)
            if not self.classification_layer:
                # I initialize it here so it can be adaptive to any class number
                self.classification_layer = nn.Linear(z_counterfactual.shape[-1], self.clusters_num).to(self.device)
            logits_counterfactuals = self.classification_layer(z_counterfactual)
            logits_real = self.classification_layer(z_real)
            bce_loss += self.bce_loss(logits_real, z_real_labels) + self.bce_loss(logits_counterfactuals,
                                                                                  z_counterfactual_labels)

            counterfactual_acc += (
                    torch.argmax(logits_counterfactuals, dim=1) == z_counterfactual_labels).float().mean().item()
            real_acc += (torch.argmax(logits_real, dim=1) == z_real_labels).float().mean().item()

        return cl_loss, bce_loss, real_acc / len(predicted_targets_pairs), counterfactual_acc / len(
            predicted_targets_pairs)

    def calculate_dsw_loss(self, predicted_targets_pairs: List[dict]) -> torch.Tensor:
        loss = 0
        for pair in predicted_targets_pairs:
            loss += self.fg_criterion(pair['predicted'].float(), pair['target'].float())
        return loss

    def quantile_loss(self, predicted_targets_pairs, quantile):
        loss = 0
        for pair in predicted_targets_pairs:
            quantile_threshold = torch.quantile(pair['mask_for_quantile'], quantile).item()
            # estimated_logits = probs_to_logits(
            #     pair['counterfactuals_for_quantile'][pair['mask_for_quantile'] < quantile_threshold].flatten())
            # loss += torch.nn.functional.cross_entropy(
            #     estimated_logits
            #     , pair['bg_for_quantile'][pair['mask_for_quantile'] < quantile_threshold].flatten())
            loss += ((pair['counterfactuals_for_quantile'][pair['mask_for_quantile'] < quantile_threshold] -
                      pair['bg_for_quantile'][pair['mask_for_quantile'] < quantile_threshold].flatten()) ** 2).sum()
        return loss

    def calculate_critic_loss(self, predicted_targets_pairs: List[dict]):
        loss1, loss2 = 0, 0
        embellished = []
        for pair in predicted_targets_pairs:
            masking_loss, critic_loss, embellished_counterfactuals = self.fg_criterion(pair['predicted'].float(),
                                                                                       pair['target'].float(),
                                                                                       pair['background'].float(),
                                                                                       pair[
                                                                                           'real_different_cluster'].float(),
                                                                                       pair['masks'].float(),
                                                                                       pair['bg_label'])
            loss1 += masking_loss
            loss2 += critic_loss
            embellished.append(embellished_counterfactuals)
        return loss1, loss2, embellished

    def model_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            stage: str
    ) -> Tuple[
        List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[dict], CalculateLossesParams]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
        """
        fg_images, bg_images, masks, bg_bg_labels, bg_fg_labels, _ = batch
        # Note that bg_images are already patches but it is guaranteed that they don't have fg object parts.
        fg_images = normalize_images(fg_images)
        bg_images = normalize_images(bg_images)
        bg_estimated_masks, fg_estimated_masks, fg_logits, bg_logits = self.pass_to_mask_net(bg_images, fg_images, stage)

        pairs = self.generate_predictions_targets(fg_images, bg_images, bg_fg_labels, bg_bg_labels, fg_estimated_masks)
        if len(pairs) == 0:
            raise RuntimeError('pairs cannot be empty!!')
        losses = self.calculate_losses(pairs, bg_estimated_masks, stage)
        # losses = []
        # losses.append(torch.nn.functional.binary_cross_entropy(fg_estimated_masks, masks.float()))
        # losses.append(torch.nn.functional.binary_cross_entropy(bg_estimated_masks, torch.zeros_like(bg_estimated_masks)))
        # pixel_wise_entropy_loss = self.pixel_wise_entropy(estimated_masks)
        fg_estimated_masks, masks, bg_estimated_masks = fg_estimated_masks.detach().cpu(), masks.cpu(), bg_estimated_masks.detach().cpu()
        for pair in pairs:
            target_tmp = pair['target'].repeat(1, 3, 1, 1) if pair['target'].shape[1] == 1 else pair['target']
            predicted_tmp = pair['predicted'].repeat(1, 3, 1, 1) if pair['predicted'].shape[1] == 1 else pair[
                'predicted']
            self.fids[pair['bg_label'].cpu().item()].update(target_tmp, real=True)
            self.fids[pair['bg_label'].cpu().item()].update(predicted_tmp, real=False)

        return losses, fg_images, bg_images, fg_estimated_masks, masks, bg_estimated_masks, pairs

    def pass_to_mask_net(self, bg_images, fg_images, stage):
        data = torch.concatenate((fg_images, bg_images))
        split_index = len(fg_images)
        logits = self.forward(data.float())
        if stage == 'train':
            random_shift = ((torch.rand(1)) - .5).item()
        else:
            random_shift = 0
        # random_shift = 0
        estimated_masks = torch.sigmoid(logits + random_shift)
        fg_logits = logits[:split_index]
        bg_logits = logits[split_index:]
        fg_estimated_masks = estimated_masks[:split_index]
        bg_estimated_masks = estimated_masks[split_index:]
        return bg_estimated_masks, fg_estimated_masks, fg_logits, bg_logits

    def calculate_losses(self, pairs, bg_estimated_masks, stage):

        losses = []
        fg_loss = self.calculate_dsw_loss(pairs)
        losses.append(fg_loss)
        if self.bg_criterion:
            bg_loss = self.bg_criterion(bg_estimated_masks, torch.zeros_like(bg_estimated_masks))
            losses.append(bg_loss)
        if self.encoder:
            cl_loss, bce_loss, real_acc, counterfactual_acc = self.calculate_cl_and_bce_losses(pairs)
            self.log(f'{stage}/real_acc', real_acc, sync_dist=True, on_epoch=True)
            self.log(f'{stage}/counterfactual_acc', counterfactual_acc, sync_dist=True, on_epoch=True)
            losses.append(cl_loss)
        if isinstance(self.trainer.datamodule, SSSDataModule) or (hasattr(self.trainer.datamodule, "background_type") and self.trainer.datamodule.background_type == 'sas'):
            # losses.append(self.quantile_loss(pairs, .5))
            pass
        return losses

    def calculate_fids(self, stage: str):
        fid_value = 0
        for fid in self.fids.values():
            try:
                fid_value += fid.compute()
            except RuntimeError as e:
                logging.warning(e)
            fid.reset()
        fid_value /= len(self.fids)
        self.log(f'{stage}/fid', fid_value)


    def on_validation_epoch_end(self):
        [ avg_precision.reset() for avg_precision in self.avg_precision]
        [ confusion_matrix.reset() for confusion_matrix in self.confusion_matrix]
        [ auc_roc.reset() for auc_roc in self.auc_roc]
        if not self.trainer.sanity_checking:
            self.calculate_fids('val')

    def training_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        losses, fg_images, bg_images, fg_estimated_masks, masks, bg_estimated_masks, pairs = self.model_step(
            batch,
            'train')
        self.perform_log_evaluation_metrics(losses, masks, fg_estimated_masks, bg_estimated_masks, "train", batch_idx)

        return torch.sum(torch.stack(losses))

    def perform_log_evaluation_metrics(self, losses: List[torch.Tensor], masks: torch.Tensor,
                                       fg_estimated_masks: torch.Tensor,
                                       bg_estimated_masks: torch.Tensor,
                                       stage: str,
                                       idx: int):

        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
        for mask_channel in range(masks.shape[1]):
            fg_avg_precision = self.avg_precision[mask_channel].cpu()(fg_estimated_masks.flatten(), masks[:,mask_channel].flatten())
            fg_confusion_matrix = self.confusion_matrix[mask_channel].cpu()(fg_estimated_masks.flatten(), masks[:,mask_channel].flatten())
            fg_aucroc = self.auc_roc[mask_channel].cpu()(fg_estimated_masks.flatten(), masks[:,mask_channel].flatten())
            mean_ioc = fg_confusion_matrix[1, 1] / (
                    fg_confusion_matrix[1, 1] + fg_confusion_matrix[1, 0] + fg_confusion_matrix[0, 1])
            self.log(f"{stage}/fg/mean_ioc_{mask_channel}", mean_ioc, sync_dist=True, on_epoch=True)
            self.log(f"{stage}/fg/avg_precision_{mask_channel}", fg_avg_precision, sync_dist=True, on_epoch=True)
            self.log(f"{stage}/fg/aucroc_{mask_channel}", fg_aucroc, sync_dist=True, on_epoch=True)
        losses_sum = 0
        for idx, loss in enumerate(losses):
            losses_sum += loss
            self.log(f"{stage}/loss_{idx}", loss, sync_dist=True, on_epoch=True)
        self.log(f"{stage}/total_loss", losses_sum, sync_dist=True, on_epoch=True)
        # self.plot_confusion_matrix(fg_confusion_matrix, f'{stage}/fg/Foreground_Confusion_Matrix', idx)
        # self.plot_confusion_matrix(bg_confusion_matrix, f'{stage}/bg/Background_Confusion_Matrix', idx)

    def plot_confusion_matrix(self, confusion_matrix: torch.Tensor, name: str, idx: int):
        normalized_confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)
        fig = px.imshow(normalized_confusion_matrix, text_auto=True)
        logger = self.logger.experiment  # noqa
        if hasattr(logger, 'add_image'):
            logger.add_image(name, plotly_to_tensor(fig), global_step=idx)
        else:
            # assume its w&b
            logger.log({name: fig}, step=idx)

    def plot_images(self, fg_images, bg_images, masks, fg_estimated_masks, bg_estimated_masks, stage, idx):
        if idx % 500 != 0 and stage != 'test':
            return
        logger = self.logger.experiment
        fg_images = fg_images[:self.images_num].cpu().float()
        bg_images = bg_images[:self.images_num].cpu().float()
        fg_images = fg_images / fg_images.max()
        bg_images = bg_images / bg_images.max()
        masks = masks[:self.images_num].float().cpu().float()
        if self.psudo_colormap and fg_images.shape[1] == 1 and bg_images.shape[1] == 1:
            fg_images = apply_colormap_to_tensor(fg_images, self.psudo_colormap)
            bg_images = apply_colormap_to_tensor(bg_images, self.psudo_colormap)
            masks = masks.repeat(1, 3, 1, 1)
        thresholded_fg = (fg_estimated_masks[:self.images_num] > .5).float().cpu()
        thresholded_bg = (bg_estimated_masks[:self.images_num] > .5).float().cpu()
        fg_estimated_masks = fg_estimated_masks[:self.images_num].cpu().float()
        bg_estimated_masks = bg_estimated_masks[:self.images_num].cpu().float()
        keys = [f'{stage}/Foreground_Info',
                f'{stage}/Background_Info', ]
        fg_info = self.startified_images_plotting(fg_images[:8], fg_estimated_masks[:8], masks[:8], thresholded_fg[:8],
                                                  num_cols=8)
        bg_info = self.startified_images_plotting(bg_images[:8], bg_estimated_masks[:8], thresholded_bg[:8], num_cols=8)
        if hasattr(logger, 'add_image'):
            logger.add_image(keys[0], fg_info, global_step=idx)
            logger.add_image(keys[1], bg_info, global_step=idx)
        else:
            logger.log({keys[0]: wandb.Image(fg_info),
                        keys[1]: wandb.Image(bg_info)})

    def plot_counterfactuals(self, pairs, stage, index):
        if index % 500 != 0:
            return
        # will plot a table of counterfactuals images labels_num x 8
        # plot for each 8 foreground images, the corresponding background image per each label and the counterfactual
        counterfactuals = []
        counter = 0
        logger = self.logger.experiment
        for pair in pairs:
            # label for each background
            tmp = pair['counterfactuals_input']
            images = pair['input']
            rand_indexes = torch.randperm(tmp.size(0))
            tmp = tmp[rand_indexes]
            images = images[rand_indexes]
            # 8 because the default setting for self.log_images is 8 per row
            counterfactuals = torch.cat((images[:8], tmp[:8].cpu()))
            if len(counterfactuals) == 0:
                continue
            if self.psudo_colormap and counterfactuals.shape[1]:
                counterfactuals = apply_colormap_to_tensor(counterfactuals, self.psudo_colormap)
            if hasattr(logger, 'add_image'):
                logger.add_image(f'{stage}/counterfactuals', self.log_images(counterfactuals), global_step=index)
            else:
                logger.log({f'{stage}/counterfactuals': wandb.Image(self.log_images(counterfactuals),
                                                                    caption=f'Counterfactuals: {counter}')})
            counter += 1

    def startified_images_plotting(self, *image_sets, num_cols=8):
        num_images = min([images.shape[0] for images in image_sets])
        # Create a list to hold the interleaved images
        interleaved_images = []
        for i in range(num_images):
            column_images = []
            for images in image_sets:
                image = images[i]
                if image.shape[0] == 1:
                    image = image.repeat(3, 1, 1)
                column_images.append(image)
            interleaved_images.append(torch.cat(column_images, dim=-2))  # Concatenate images vertically

        # Stack and create a grid
        grid = make_grid(torch.stack(interleaved_images), nrow=num_cols, padding=2, normalize=True)
        return grid

    def log_images(self, images: torch.Tensor, n_items_in_row: int = None):
        if n_items_in_row is None:
            images = make_grid(images, 8)
        else:
            images = make_grid(images, n_items_in_row)
        return images.cpu()

    def create_image_grid(self, composite, background, mask):
        composite_batch_size, channels, height, width = composite.shape
        background_batch_size = background.shape[0]
        # Create a black image for grid[0,0]
        black_image = torch.zeros((channels, height, width))

        # Initialize the grid with empty tensors for the required size
        grid = torch.zeros(
            (composite_batch_size + 1, background_batch_size + 1, channels, height, width)
        )

        # Set the top-left corner as a black image
        grid[0, 0] = black_image

        # Fill the first column with composite images
        grid[1:, 0] = composite

        # Fill the first row with background images
        grid[0, 1:] = background

        # Populate the rest of the grid using the given equation
        for i in range(composite_batch_size):
            for j in range(background_batch_size):
                grid[i + 1, j + 1] = (
                        mask[i] * composite[i] + (1 - mask[i]) * background[j]
                )

        # Flatten the grid for visualization using make_grid
        grid = grid.flatten(0, 1)  # Shape: (num_images, channels, height, width)
        grid_image = make_grid(grid, nrow=background_batch_size + 1, padding=2)

        return grid_image

    def plot_counterfactuls_v2(self, fg_images, bg_images, masks, bg_bg_labels, bg_fg_labels, stage):
        candidate_bg_images = []
        candidate_fg_images = []
        candidate_masks = []
        for bg_label in torch.unique(bg_bg_labels):
            bg_images_per_label = bg_images[bg_bg_labels == bg_label]
            candidate_bg_images.append(bg_images_per_label[torch.randint(0, bg_images_per_label.shape[0], (1,)).item()])
        for fg_label in torch.unique(bg_fg_labels):
            fg_images_per_label = fg_images[bg_fg_labels == fg_label]
            random_index = torch.randint(0, fg_images_per_label.shape[0], (1,)).item()
            candidate_fg_images.append(fg_images_per_label[random_index])
            candidate_masks.append(masks[bg_fg_labels == fg_label][random_index])
        candidate_fg_images = torch.stack(candidate_fg_images).detach().cpu()
        candidate_bg_images = torch.stack(candidate_bg_images).detach().cpu()
        candidate_masks = torch.stack(candidate_masks).detach().cpu()
        if self.psudo_colormap and candidate_fg_images.shape[1] == 1 and candidate_bg_images.shape[1] == 1:
            candidate_masks = candidate_masks.repeat(1, 3, 1, 1)
            candidate_bg_images = apply_colormap_to_tensor(candidate_bg_images, self.psudo_colormap)
            candidate_fg_images = apply_colormap_to_tensor(candidate_fg_images, self.psudo_colormap)
        grid = self.create_image_grid(candidate_fg_images, candidate_bg_images, candidate_masks)

        logger = self.logger.experiment
        if hasattr(logger, 'add_image'):
            logger.add_image(f'{stage}/counterfactuals_v2', self.log_images(grid))
        else:
            logger.log({f'{stage}/counterfactuals_v2': wandb.Image(grid)})

    def validation_step(self, batch: Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                        batch_idx: int) -> torch.Tensor:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        losses, fg_images, bg_images, fg_estimated_masks, masks, bg_estimated_masks, pairs = self.model_step(batch,
                                                                                                             'val')

        self.perform_log_evaluation_metrics(losses, masks, fg_estimated_masks, bg_estimated_masks, "val", batch_idx)
        if batch_idx == 0:
            self.plot_images(fg_images, bg_images, masks, fg_estimated_masks, bg_estimated_masks, "val", batch_idx)
            self.plot_counterfactuals(pairs, 'val', batch_idx)
            _, _, _, bg_bg_labels, bg_fg_labels, _ = batch
            self.plot_counterfactuls_v2(fg_images.cpu(), bg_images.cpu(), fg_estimated_masks.cpu(), bg_bg_labels.cpu(),
                                        bg_fg_labels.cpu(), 'val')

        return torch.sum(torch.stack(losses))

    def test_step(self,
                  batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                  batch_idx: int) -> torch.Tensor:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        losses, fg_images, bg_images, fg_estimated_masks, masks, bg_estimated_masks, pairs = self.model_step(batch,
                                                                                                             'test')

        self.perform_log_evaluation_metrics(losses, masks, fg_estimated_masks, bg_estimated_masks, "test", batch_idx)
        self.plot_images(fg_images, bg_images, masks, fg_estimated_masks, bg_estimated_masks, "test", batch_idx)
        self.plot_counterfactuals(pairs, "test", batch_idx)

        _, _, _, bg_bg_labels, bg_fg_labels, _ = batch
        for _ in range(5):
            self.plot_counterfactuls_v2(fg_images.cpu(), bg_images.cpu(), fg_estimated_masks.cpu(), bg_bg_labels.cpu(),
                                        bg_fg_labels.cpu(), 'test')

        return torch.sum(torch.stack(losses))

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
            if self.encoder:
                self.encoder = self.encoder.to(self.device)
        self.clusters_num = get_background_clusters_num(self.trainer.datamodule)
        if not self.clusters_num:
            self.clusters_num = 15
            log.warning("set clusters_num to 15")
        self.fids = {}
        for cluster_num in range(self.clusters_num):
            self.fids[cluster_num] = FrechetInceptionDistance(normalize=True).to(self.device)
        if self.fg_criterion is None:
            raise RuntimeError(f'{self.divergence_loss} is not supported as a divergence loss')

    def on_validation_epoch_start(self):
        [ avg_precision.reset() for avg_precision in self.avg_precision]
        [ confusion_matrix.reset() for confusion_matrix in self.confusion_matrix]
        [ auc_roc.reset() for auc_roc in self.auc_roc]
        if not self.trainer.sanity_checking:
            self.calculate_fids('train')

    def configure_optimizers(self) -> Dict[str, Any]:

        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        optimizers = [optimizer]
        schedulers = []
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            schedulers = [{
                "scheduler": scheduler,
                "monitor": "val/total_loss",
                "interval": "epoch",
                "frequency": 1,
            }]
        return optimizers, schedulers

    @classmethod
    def load_model(cls, checkpoint_path, strict=False):
        print('the model was called here')
        model = cls.load_from_checkpoint(checkpoint_path, strict=False)
        return model


if __name__ == "__main__":
    _ = FgBgLitModule(None, None, None, None, None, None, None)
