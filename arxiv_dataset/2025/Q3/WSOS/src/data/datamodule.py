from abc import abstractmethod, ABC
from typing import Optional, Any
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset, Subset
import wandb
import numpy as np
from matplotlib import pyplot as plt
from torchvision.utils import make_grid

from src.models.components.ae_gmm import AEGMM
from src.utils.utils import load_net_state_from_checkpoint, stratified_cat, apply_colormap_to_tensor

from src.utils import RankedLogger
    
log = RankedLogger(__name__, rank_zero_only=True)


class DataModule(LightningDataModule, ABC):
    def __init__(self, background_classifier: str, order_background_labels: bool):
        super().__init__()
        # bg_fg_labels reads background labels for fg image
        self.fg_images, self.bg_images, self.masks, self.bg_fg_labels, self.bg_bg_labels, self.fg_labels = None, None, None, None, None, None
        self.background_classifier = None
        if background_classifier:
            self.background_classifier: AEGMM = torch.load(background_classifier).cpu()
            self.background_classifier.eval()
            # self.background_classifier = GMVAENet(64*64*3,64,20)
            # self.background_classifier.load_state_dict(load_net_state_from_checkpoint(background_classifier, self.background_classifier.state_dict(),'net'))
        self.background_classifier_name = background_classifier
        self.order_background_labels = order_background_labels
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.train_shuffle = True
        self.logger = None

    def order(self):
        fg_images = torch.stack([self.data_train[i][0] for i in range(len(self.data_train))])
        bg_images = torch.stack([self.data_train[i][1] for i in range(len(self.data_train))])
        masks = torch.stack([self.data_train[i][2] for i in range(len(self.data_train))])
        bg_bg_labels = torch.stack([self.data_train[i][3] for i in range(len(self.data_train))])
        bg_fg_labels = torch.stack([self.data_train[i][4] for i in range(len(self.data_train))])
        fg_labels = torch.stack([self.data_train[i][5] for i in range(len(self.data_train))])

        # in this case bg_fg_labels will belong to fg_images_ground_truth we don't really care about bg labels of
        # fg labels
        bg_images_ordered, bg_images_not_ordered = [], []
        bg_bg_labels_ordered, bg_bg_labels_not_ordered = [], []
        bg_fg_labels_ordered, bg_fg_labels_not_ordered = [], []
        fg_images_ordered, fg_images_not_ordered = [], []
        masks_ordered, masks_not_ordered = [], []
        fg_labels_ordered, fg_labels_not_ordered = [], []

        print(torch.unique(bg_fg_labels))
        for bg_label in torch.unique(bg_fg_labels):
            matched_bg_indexes = bg_fg_labels == bg_label
            matched_bg_bg_indexes = bg_bg_labels == bg_label
            matched_bg_num = matched_bg_indexes.sum().item()
            matched_bg_bg_num = matched_bg_bg_indexes.sum().item()
            if matched_bg_bg_num == 0:
                # if there is no background label just put whatever. There is nothing I can do
                # matched_bg_bg_indexes = self.bg_fg_labels != bg_label
                # matched_bg_bg_num = matched_bg_bg_indexes.sum().item()
                continue
            selected_elements_for_bg = torch.multinomial(torch.ones(matched_bg_bg_num), matched_bg_num,
                                                         replacement=True)
            print(f'number of composite images that have label {bg_label}: {matched_bg_num}')
            bg_images_ordered.append(bg_images[matched_bg_bg_indexes][selected_elements_for_bg])
            fg_images_ordered.append(fg_images[matched_bg_indexes])
            bg_fg_labels_ordered.append(bg_fg_labels[matched_bg_indexes])
            bg_bg_labels_ordered.append(bg_bg_labels[matched_bg_bg_indexes][selected_elements_for_bg])
            masks_ordered.append(masks[matched_bg_indexes])
            fg_labels_ordered.append(fg_labels[matched_bg_indexes])

            notmatched_bg_indexes = bg_fg_labels != bg_label
            selected_elements = torch.multinomial(torch.ones(torch.sum(notmatched_bg_indexes)), matched_bg_num,
                                                  replacement=True)
            selected_elements_for_bg = torch.multinomial(torch.ones(torch.sum(bg_bg_labels != bg_label)),
                                                         matched_bg_num,
                                                         replacement=True)
            fg_images_not_ordered.append(fg_images[notmatched_bg_indexes][selected_elements])
            bg_images_not_ordered.append(fg_images[bg_bg_labels != bg_label][selected_elements_for_bg])
            bg_bg_labels_not_ordered.append(bg_bg_labels[bg_bg_labels != bg_label][selected_elements_for_bg])
            bg_fg_labels_not_ordered.append(bg_fg_labels[notmatched_bg_indexes][selected_elements])
            masks_not_ordered.append(masks[notmatched_bg_indexes][selected_elements])
            fg_labels_not_ordered.append(fg_labels[notmatched_bg_indexes][selected_elements])

        fg_images_ordered = torch.cat(fg_images_ordered)
        bg_images_ordered = torch.cat(bg_images_ordered)
        bg_fg_labels_ordered = torch.cat(bg_fg_labels_ordered)
        bg_fg_labels_not_ordered = torch.cat(bg_fg_labels_not_ordered)
        bg_bg_labels_ordered = torch.cat(bg_bg_labels_ordered)
        bg_bg_labels_not_ordered = torch.cat(bg_bg_labels_not_ordered)
        fg_images_not_ordered = torch.cat(fg_images_not_ordered)
        bg_images_not_ordered = torch.cat(bg_images_not_ordered)
        masks_ordered = torch.cat(masks_ordered)
        masks_not_ordered = torch.cat(masks_not_ordered)
        fg_labels_ordered = torch.cat(fg_labels_ordered)
        fg_labels_not_ordered = torch.cat(fg_labels_not_ordered)

        num_samples, num_channels, w, d = fg_images.shape
        chunk_size0 = int(self.hparams.batch_size * .5)
        chunk_size1 = int(self.hparams.batch_size * .5)
        fg_images = stratified_cat(fg_images_ordered.unsqueeze(1),
                                   fg_images_not_ordered.unsqueeze(1), chunk_size0, chunk_size1).view(
            -1, num_channels, w, d)
        num_samples, num_channels, w, d = bg_images.shape

        bg_images = stratified_cat(bg_images_ordered.unsqueeze(1),
                                   bg_images_ordered.unsqueeze(1), chunk_size0, chunk_size1).view(
            -1, num_channels, w, d)

        num_samples, num_channels, w, d = masks.shape
        masks = stratified_cat(masks_ordered.unsqueeze(1), masks_not_ordered.unsqueeze(1), chunk_size0,
                               chunk_size1).view(
            -1, num_channels, w, d)

        bg_bg_labels = stratified_cat(bg_bg_labels_ordered.unsqueeze(1),
                                      bg_bg_labels_ordered.unsqueeze(1), chunk_size0, chunk_size1).view(-1)
        bg_fg_labels = stratified_cat(bg_fg_labels_ordered.unsqueeze(1), bg_fg_labels_not_ordered.unsqueeze(1),
                                      chunk_size0, chunk_size1).view(-1)
        fg_labels = stratified_cat(fg_labels_ordered.unsqueeze(1),
                                   fg_labels_not_ordered.unsqueeze(1), chunk_size0, chunk_size1).view(
            -1)
        self.data_train = TensorDataset(fg_images, bg_images, masks,
                                        bg_bg_labels, bg_fg_labels,
                                        fg_labels)

    def setup(self, stage: str = None) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            train_percentage, val_percentage, test_percentage = self.hparams.train_val_test_split
            self.assign_bg_labels()
            if self.fg_labels is None:
                self.fg_labels = torch.tensor([0] * len(self.fg_images))
            dataset = TensorDataset(self.fg_images, self.bg_images, self.masks,
                                    self.bg_bg_labels, self.bg_fg_labels,
                                    self.fg_labels)
            if not hasattr(self, 'train_test_split'):
                self.data_train, self.data_val, self.data_test = random_split(dataset,
                                                                              [train_percentage, val_percentage,
                                                                               test_percentage],
                                                                              torch.Generator().manual_seed(42))
            else:
                train_indices = [i for i, is_train in enumerate(self.train_test_split) if is_train == 1]
                test_indices = [i for i, is_train in enumerate(self.train_test_split) if is_train == 0]

                train_dataset = Subset(dataset, train_indices)
                self.data_test = Subset(dataset, test_indices)
                self.data_train, self.data_val = random_split(train_dataset,
                                                              [train_percentage, val_percentage + test_percentage],
                                                              torch.Generator().manual_seed(42))
            if self.order_background_labels:
                print('ordering according to background labels...')
                self.order()
                self.train_shuffle = False
            self.log_histograms()
            self.plot_images_per_background_label()

    @abstractmethod
    def assign_bg_labels(self):
        pass

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=self.train_shuffle,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def set_logger(self, logger):
        self.logger = logger

    def log_histograms(self):

        if not self.logger:
            log.warn(f'logger in datamoduel is None. Skipping logging the histogram of the background labels ...')
            return
        bg_fg_data = self.bg_fg_labels.numpy()
        bg_bg_data = self.bg_bg_labels.numpy()

        # Compute histograms with automatic binning
        fg_labels_count = len(np.unique(bg_fg_data))
        bg_labels_count = len(np.unique(bg_fg_data))
        bg_fg_hist_data, bg_fg_bins = np.histogram(bg_fg_data, bins=fg_labels_count, density=True)
        bg_bg_hist_data, bg_bg_bins = np.histogram(bg_bg_data, bins=bg_labels_count, density=True)

        # Sort histograms by density in descending order
        sorted_bg_fg_indices = np.argsort(bg_fg_hist_data)[::-1]
        sorted_bg_bg_indices = np.argsort(bg_bg_hist_data)[::-1]

        sorted_bg_fg_hist_data = bg_fg_hist_data[sorted_bg_fg_indices]
        sorted_bg_bg_hist_data = bg_bg_hist_data[sorted_bg_bg_indices]

        # Plot histograms using matplotlib
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        axes[0].bar(list(range(fg_labels_count)), sorted_bg_fg_hist_data,
                    align="edge", alpha=0.75, label='Composite')
        axes[0].set_xlabel('Label')
        axes[0].set_ylabel('Density')
        axes[0].legend()

        axes[1].bar(list(range(bg_labels_count)), sorted_bg_bg_hist_data,
                    align="edge", alpha=0.75, label='Background')
        axes[1].set_xlabel('Label')
        axes[1].set_ylabel('Density')
        axes[1].legend()

        plt.tight_layout()

        if hasattr(self.logger[0].experiment, 'add_image'):
            # self.logger[0].experiment.add_image("charts\histograms", fig)
            log.warning("Tensorboard doesn't support logging matplotlib images")
        else:
            self.logger[0].experiment.log({"charts\histograms": wandb.Image(fig)})

    def plot_images_per_background_label(self, colormap=None):
        def concatenate_images_by_label(images, labels):
            unique_labels, counts = labels.unique(return_counts=True)
            unique_labels = unique_labels[counts.sort()[1].flip(0)]
            sorted_images = []
            for label in unique_labels:
                idx = (labels == label).nonzero(as_tuple=True)[0][:10]
                sorted_images.append(images[idx])
            return torch.cat(sorted_images, dim=0)

        fg_images_sorted = concatenate_images_by_label(self.fg_images, self.bg_fg_labels)
        bg_images_sorted = concatenate_images_by_label(self.bg_images, self.bg_bg_labels)

        grid1 = make_grid(fg_images_sorted, nrow=10, padding=2)
        grid2 = make_grid(bg_images_sorted, nrow=10, padding=2)     

        if colormap is not None:
            grid1 = apply_colormap_to_tensor(grid1, colormap)
            grid2 = apply_colormap_to_tensor(grid2, colormap)
        if hasattr(self.logger[0].experiment, 'add_image'):
            self.logger[0].experiment.add_image("charts\composite_background_label_images_examples", grid1)
            self.logger[0].experiment.add_image("charts\bg_background_label_images_examples", grid2)
        else:
            self.logger[0].experiment.log({"charts\composite_background_label_images_examples": wandb.Image(grid1)})
            self.logger[0].experiment.log({"charts\bg_background_label_images_examples": wandb.Image(grid2)})
