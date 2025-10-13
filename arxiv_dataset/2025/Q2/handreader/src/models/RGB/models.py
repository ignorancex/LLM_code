import gc
from pathlib import Path

import rootutils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._C import default_generator
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.Atrainer import AbstractTrainer
from src.models.modules import FeatureExtracter, RNNHead
from src.utils import LOGGER, compute_acc, get_score


class TSM_Resnet(nn.Module):
    """
    A Temporal Spatial Memory (TSM) module based on ResNet architecture for sequence classification.

    Parameters:
    -----------
    encoder : str
        The name or configuration of the ResNet model to be used as the backbone.
    decoder : nn.Module
        The module used for decoding the features extracted by the ResNet backbone.
    num_classes : int
        The number of output classes for classification.
    decoder_net : RNNHead
        A recurrent neural network head that processes the output of the ResNet and predicts the final logits.
    unidirection : bool, optional
        Whether the backbone ResNet should be unidirectional (default: False).
    """

    def __init__(
        self,
        encoder: str,
        decoder,
        num_classes,
        decoder_net: RNNHead,
        unidirection: bool = False,
    ):
        """
        Initializes the TSM Resnet instance.

        Parameters
        ----------
        encoder : str
            The name or configuration of the ResNet model to be used as the backbone.
        decoder
            The module used for decoding the features extracted by the ResNet backbone.
        num_classes : int
            The number of output classes for classification.
        decoder_net : RNNHead
            A recurrent neural network head that processes the output of the ResNet and predicts the final logits.
        unidirection : bool, optional
            Whether the backbone ResNet should be unidirectional (default: False).

        Returns
        -------
        None
        """

        super().__init__()

        self.backbone = FeatureExtracter(resnet=encoder, unidirection=unidirection)

        self.decoder = decoder
        self.decoder_net = decoder_net

    def forward(self, x, input_lenghts=None):
        """Returns logits."""

        x = self.backbone(x, input_lenghts)
        logits = self.decoder_net(x)

        return logits


class Trainer(AbstractTrainer):
    """
    A trainer class for models using Connectionist Temporal Classification (CTC) loss.

    This trainer manages training, validation, and checkpointing for models using
    CTC loss, supporting features such as gradient accumulation, learning rate
    scheduling, and Levenshtein accuracy computation for evaluation.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained and evaluated.

    optimizer : torch.optim.Optimizer
        The optimizer used for updating model weights.

    loss_fn : callable
        The loss function used during training (e.g., CTC loss).

    device : str or torch.device
        The device on which computations are performed (e.g., "cuda" or "cpu").

    decoder : object
        A decoder object that provides methods for decoding predictions
        (`greedy_decode`) and mapping integers to characters (`int_to_char`).

    val_every : int
        The frequency (in epochs) of validation.

    path_to_save_weights : str
        The directory path to save model weights and checkpoints.

    get_metric_train : bool
        Whether to compute evaluation metrics (e.g., Levenshtein accuracy) during training.

    resume_path : str, optional
        Path to a checkpoint file to resume training. Default is None.

    scheduler : torch.optim.lr_scheduler._LRScheduler, optional
        A learning rate scheduler. Default is None.

    grad_accumulate : float, optional
        The number of steps to accumulate gradients before performing an optimizer step.
        Default is 1.0.

    path_to_states : str, optional
        Path to a file containing random state information for reproducibility. Default is None.

    Methods
    -------
    train_step(batch)
        Performs a single training step for a given batch.

    validate_step(batch)
        Performs a single validation step for a given batch.

    fit(train_loader, val_loader=None, epochs=1)
        Trains the model for the specified number of epochs.

    on_epoch_end(epoch, train_loss, val_loss, val_metric, preds_val, gts_val, train_metric=None, preds_train=None, gts_train=None)
        Performs logging and checkpointing at the end of each epoch.

    save_checkpoint(name, path_to_save, model, epoch=None)
        Saves the model and optimizer states to a checkpoint file.

    log_metrics(metric_name, metric_value, epoch)
        Logs scalar metrics using TensorBoard.

    log_phrase(metric_name, text, epoch)
        Logs text-based metrics using TensorBoard.
    """

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        device,
        val_every: int,
        path_to_save_weights: str,
        get_metric_train: bool,
        resume_path: str = None,
        scheduler=None,
        loss_scale: int = 1,
        path_to_states: str = None,
    ) -> None:
        """
        Initializes the Trainer class.

        This constructor sets up the model, optimizer, loss function, and various
        training parameters. It also handles loading from a checkpoint if provided.

        Parameters
        ----------
        model : torch.nn.Module
            The model to be trained and evaluated.
        optimizer : torch.optim.Optimizer
            The optimizer for updating model parameters.
        loss_fn : callable
            The loss function used during training.
        device : str or torch.device
            The device on which computations are performed (e.g., "cuda" or "cpu").
        val_every : int
            The frequency (in epochs) of validation.
        path_to_save_weights : str
            The directory path to save model weights and checkpoints.
        get_metric_train : bool
            Whether to compute evaluation metrics during training.
        resume_path : str, optional
            Path to a checkpoint file to resume training. Default is None.
        scheduler : torch.optim.lr_scheduler._LRScheduler, optional
            A learning rate scheduler. Default is None.
        loss_scale : int, optional
            The scale factor for the loss. Default is 1.
        path_to_states : str, optional
            Path to a file containing random state information for reproducibility. Default is None.
        """
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = torch.device(device)
        self.model.to(self.device)
        self.val_every = val_every
        self.best_lev = 0
        self.logger = LOGGER
        self.path_to_save_weights = path_to_save_weights
        self.writer = SummaryWriter()
        self.scheduler = scheduler
        self.get_metric_train = get_metric_train
        self.loss_scale = loss_scale
        self.iter = 0
        self.start_epoch = -1

        if path_to_states is not None:
            checkpoint = torch.load(path_to_states)
            self.logger.info(f"Uploading states from {path_to_states}")
            default_generator.set_state(checkpoint["state"])
            torch.cuda.set_rng_state(checkpoint["cuda_state"], self.device)

        if resume_path is not None:
            checkpoint = torch.load(resume_path)
            self.logger.info(f"Uploading checkpoint from {resume_path}")
            self.model.load_state_dict(checkpoint["model"])
            self.scheduler.load_state_dict(checkpoint["lr_sched"].state_dict())
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.start_epoch = checkpoint["epoch"]
            self.best_lev = checkpoint["best_lev"]
            default_generator.set_state(checkpoint["state"])
            torch.cuda.set_rng_state(checkpoint["cuda_state"], self.device)

    def train_step(self, batch):
        """
        A single training step.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            A tuple of input tensors (inputs, targets, input_lengths, target_lengths).

        Returns
        -------
        dict
            A dictionary containing the loss value, and optionally the Levenshtein accuracy
            of greedy decoded predictions, as well as the predicted and ground truth phrases.
        """
        inputs, targets, input_lengths, target_lengths = batch
        inputs, targets, input_lengths, target_lengths = (
            inputs.to(self.device),
            targets.to(self.device),
            input_lengths.to(self.device),
            target_lengths.to(self.device),
        )

        self.optimizer.zero_grad()

        outputs = self.model(inputs, input_lengths)

        log_probs = F.log_softmax(outputs, dim=-1).permute(1, 0, 2)
        loss = self.loss_scale * self.loss_fn(log_probs, targets, input_lengths, target_lengths)

        loss.backward()
        self.optimizer.step()

        if self.get_metric_train:
            probs = F.softmax(outputs, dim=-1)
            phrase_preds = []
            phrase_gts = []

            for i in range(len(outputs)):
                curr_pred = self.model.decoder.greedy_decode(probs[i].detach().cpu().numpy())
                curr_pred = "".join(curr_pred)

                phrase_preds.append(curr_pred)

                # decoding gt phrase

                cur_gt = "".join(
                    [
                        self.model.decoder.int_to_char[num]
                        for num in targets[i].detach().cpu().numpy()[: target_lengths[i]]
                    ]
                )
                phrase_gts.append(cur_gt)

            lev_acc_train = get_score(phrase_gts, phrase_preds)

            return {
                "loss": loss.detach().cpu().item(),
                "lev": lev_acc_train,
                "phrase_preds": phrase_preds,
                "phrase_gts": phrase_gts,
            }

        return {"loss": loss.detach().cpu()}

    def validate_step(self, batch):
        """
        Runs a single validation step.

        This method takes a batch of data, forwards it through the model, computes the
        loss, and computes Levenshtein accuracy.

        Parameters
        ----------
        batch : tuple
            A tuple of inputs, targets, input_lengths, and target_lengths.

        Returns
        -------
        dict
            A dictionary containing the loss, and optionally Levenshtein accuracy,
            phrase predictions, and ground truth phrases.
        """
        inputs, targets, input_lengths, target_lengths = batch
        inputs, targets, input_lengths, target_lengths = (
            inputs.to(self.device),
            targets.to(self.device),
            input_lengths.to(self.device),
            target_lengths.to(self.device),
        )

        outputs = self.model(inputs, input_lengths)

        log_probs = F.log_softmax(outputs, dim=-1).permute(1, 0, 2)
        loss = self.loss_scale * self.loss_fn(log_probs, targets, input_lengths, target_lengths)

        probs = F.softmax(outputs, dim=-1)

        phrase_preds = []
        phrase_gts = []

        for i in range(len(outputs)):
            curr_pred = self.model.decoder.greedy_decode(probs[i].detach().cpu().numpy())
            curr_pred = "".join(curr_pred)

            phrase_preds.append(curr_pred)

            # decoding gt phrase
            cur_gt = "".join(
                [
                    self.model.decoder.int_to_char[num]
                    for num in targets[i].detach().cpu().numpy()[: target_lengths[i]]
                ]
            )
            phrase_gts.append(cur_gt)

        lev_acc_val = get_score(phrase_gts, phrase_preds)

        return {
            "loss": loss.detach().cpu().item(),
            "lev": lev_acc_val,
            "phrase_preds": phrase_preds,
            "phrase_gts": phrase_gts,
        }

    def fit(self, train_loader, val_loader=None, epochs=1):
        """
        Trains the model for the specified number of epochs.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            The data loader for the training data.

        val_loader : torch.utils.data.DataLoader, optional
            The data loader for the validation data. Default is None.

        epochs : int, optional
            The number of epochs to train. Default is 1.

        Returns
        -------
        None
        """
        for epoch in range(self.start_epoch + 1, epochs):
            torch.cuda.empty_cache()
            gc.collect()

            self.model.train()
            train_loss = 0.0
            cur_loss_train = 0
            lev_acc_train = 0
            preds_train = []
            gts_train = []
            curr_lev_train = 0.0
            pbar_train = tqdm(train_loader)

            # train step
            for batch in pbar_train:
                metrics_train = self.train_step(batch)
                cur_loss_train = metrics_train["loss"]
                train_loss += cur_loss_train
                self.iter += 1

                if self.get_metric_train:
                    curr_lev_train = metrics_train["lev"]
                    lev_acc_train += metrics_train["lev"]
                    preds_train.extend(metrics_train["phrase_preds"])
                    gts_train.extend(metrics_train["phrase_gts"])
                    pbar_train.set_description(
                        f"train loss {cur_loss_train} epoch {epoch} lev {curr_lev_train}"
                    )
                else:
                    pbar_train.set_description(f"train loss {cur_loss_train} epoch {epoch}")

            train_loss /= len(train_loader)

            if self.get_metric_train:
                lev_acc_train = compute_acc(preds_train, gts_train)

            # val step
            val_loss = None
            lev_acc_val = None
            preds_val = None
            gts_val = None

            if val_loader is not None and epoch % self.val_every == 0:
                self.model.eval()
                val_loss = 0.0
                lev_acc_val = 0
                preds_val = []
                gts_val = []
                curr_lev = 0.0
                curr_loss_val = 0.0
                with torch.no_grad():
                    pbar_val = tqdm(val_loader)
                    for batch in pbar_val:
                        metrics_val = self.validate_step(batch)
                        curr_lev = metrics_val["lev"]
                        curr_loss_val = metrics_val["loss"]
                        pbar_val.set_description(
                            f"val {epoch} loss {curr_loss_val} lev {curr_lev}"
                        )
                        val_loss += metrics_val["loss"]
                        lev_acc_val += metrics_val["lev"]
                        preds_val.extend(metrics_val["phrase_preds"])
                        gts_val.extend(metrics_val["phrase_gts"])

                val_loss /= len(val_loader)

                lev_acc_val = compute_acc(preds_val, gts_val)
                if lev_acc_val > self.best_lev:
                    self.logger.info("Updating metrics and save best state dict")
                    self.best_lev = lev_acc_val
                    self.save_checkpoint(
                        name="best",
                        path_to_save=self.path_to_save_weights,
                        model=self.model,
                    )

            if self.get_metric_train:
                self.on_epoch_end(
                    epoch,
                    train_loss,
                    val_loss,
                    lev_acc_val,
                    preds_val,
                    gts_val,
                    lev_acc_train,
                    preds_train,
                    gts_train,
                )
            else:
                self.on_epoch_end(epoch, train_loss, val_loss, lev_acc_val, preds_val, gts_val)
            del (train_loss, val_loss, curr_loss_val, cur_loss_train)

    def on_epoch_end(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_metric: float,
        preds_val: list[str],
        gts_val: list[str],
        train_metric: float = None,
        preds_train: list[str] = None,
        gts_train: list[str] = None,
    ):
        """Logs and saves metrics at the end of each epoch.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        train_loss : float
            The loss computed on the training data for the current epoch.
        val_loss : float or None
            The loss computed on the validation data for the current epoch.
        val_metric : float
            The Levenshtein accuracy computed on the validation data for the current epoch.
        preds_val : list[str]
            The predicted phrases for the validation data.
        gts_val : list[str]
            The ground truth phrases for the validation data.
        train_metric : float or None
            The Levenshtein accuracy computed on the training data for the current epoch.
        preds_train : list[str] or None
            The predicted phrases for the training data.
        gts_train : list[str] or None
            The ground truth phrases for the training data.
        """
        self.logger.info(f"Train loss: {train_loss} epoch: {epoch}")
        if self.get_metric_train:
            self.logger.info(f"Train loss: {train_loss} epoch {epoch}")
            self.logger.info(f"Train lev: {train_metric}")
            self.log_phrase("phrases pred/train", r" \ ".join(preds_train), epoch)
            self.log_phrase("phrases gt/train", r" \ ".join(gts_train), epoch)
            self.log_metrics("Acc/train", train_metric, epoch)

        if val_loss is not None:
            self.log_metrics("Loss/val", val_loss, epoch)
            self.log_metrics("Acc/val", val_metric, epoch)
            self.logger.info(f"Val loss: {val_loss} Val lev : {val_metric} epoch: {epoch}")
            self.logger.info(f"new lev: {val_metric} best lev: {self.best_lev}")
            self.log_phrase("phrases pred/val", r" \ ".join(preds_val), epoch)
            self.log_phrase("phrases gt/val", r" \ ".join(gts_val), epoch)

        self.log_metrics("Loss/train", train_loss, epoch)

        if self.scheduler is not None:
            self.scheduler.step()
            self.log_metrics("Learning rate", self.scheduler.get_lr()[0], epoch)

        self.logger.info(f"Saving last.pt at {self.path_to_save_weights}")
        self.save_checkpoint(
            name="last", path_to_save=self.path_to_save_weights, model=self.model, epoch=epoch
        )
        self.logger.info(f"Train loss: {train_loss} epoch: {epoch}")
        if self.get_metric_train:
            self.logger.info(f"Train loss: {train_loss} epoch {epoch}")
            self.logger.info(f"Train lev: {train_metric}")
            self.log_phrase("phrases pred/train", r" \ ".join(preds_train), epoch)
            self.log_phrase("phrases gt/train", r" \ ".join(gts_train), epoch)
            self.log_metrics("Acc/train", train_metric, epoch)

        if val_loss is not None:
            self.log_metrics("Loss/val", val_loss, epoch)
            self.log_metrics("Acc/val", val_metric, epoch)
            self.logger.info(f"Val loss: {val_loss} Val lev : {val_metric} epoch: {epoch}")
            self.logger.info(f"new lev: {val_metric} best lev: {self.best_lev}")
            self.log_phrase("phrases pred/val", r" \ ".join(preds_val), epoch)
            self.log_phrase("phrases gt/val", r" \ ".join(gts_val), epoch)

        self.log_metrics("Loss/train", train_loss, epoch)

        if self.scheduler is not None:
            self.scheduler.step()
            self.log_metrics("Learning rate", self.scheduler.get_lr()[0], epoch)

        self.logger.info(f"Saving last.pt at {self.path_to_save_weights}")
        self.save_checkpoint(
            name="last", path_to_save=self.path_to_save_weights, model=self.model, epoch=epoch
        )

    def save_checkpoint(self, name: str, path_to_save: str, model, epoch: int = None):
        """
        Saves a checkpoint of the model, optimizer, and other relevant state.

        Parameters
        ----------
        name : str
            The name of the checkpoint file to save.
        path_to_save : str
            The directory path to save the checkpoint file.
        model : torch.nn.Module
            The model to save.
        epoch : int, optional
            The current epoch number. If provided, the checkpoint will be saved
            with the epoch number. Default is None.
        """
        p = Path(path_to_save) / (name + ".pt")
        if epoch is not None:
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr_sched": self.scheduler,
                "best_lev": self.best_lev,
                "state": default_generator.get_state(),
                "cuda_state": torch.cuda.get_rng_state(self.device),
            }

            torch.save(checkpoint, str(p))
        else:
            torch.save(model.state_dict(), str(p))

    def log_metrics(self, metric_name: str, metric_value: float, epoch: int):
        """Logs a metric to the writer.

        Parameters
        ----------
        metric_name : str
            The name of the metric to log.
        metric_value : float
            The value of the metric to log.
        epoch : int
            The current epoch number.
        """
        self.writer.add_scalar(metric_name, metric_value, epoch)

    def log_phrase(self, metric_name: str, text: str, epoch: int):
        """Logs a phrase or text to the tensorboard writer.

        Parameters
        ----------
        metric_name : str
            The name of the metric to be logged.
        text : str
            The text to be logged.
        epoch : int
            The current epoch number.
        """
        self.writer.add_text(metric_name, text, epoch)
