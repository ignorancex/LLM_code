import sys

import numpy as np
import torch
import torch.nn as nn

from wyckoff_generation.common import utils
from wyckoff_generation.common.registry import registry
from wyckoff_generation.runners.base_runner import BaseRunner


@registry.register_runner("train_d3pm")
class D3PMTrainer(BaseRunner):
    def __init__(self, config):
        super().__init__(config)
        if config["load"]:
            self.load_checkpoint(config)
        else:
            self.epoch = None
        self.init_epoch_step()

    def init_model(self, config):
        self.model = registry.get_model_class("d3pm")(config).to(self.device)

    def init_epoch_step(self):
        self.num_steps_per_epoch = len(self.train_loader)
        if self.epoch is not None:
            # epoch is from saved model
            # lag between saved epoch number and number of epochs that have been performed

            self.step = self.epoch * self.num_steps_per_epoch + self.num_steps_per_epoch
            self.epoch = self.epoch + 1
        else:
            self.step = 0
            self.epoch = 0

    def load_checkpoint(self, config):
        checkpoint_dict = utils.get_pretrained_checkpoint(config["load"], best=False)
        self.model.load_state_dict(checkpoint_dict["model_state_dict"])
        self.epoch = checkpoint_dict["epoch"]
        # TODO: optimizer
        self.optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])

    def init_optimizer(self, config):
        self.optimizer = getattr(torch.optim, self.config["optimizer"])(
            self.model.parameters(),
            lr=config["lr"],
            weight_decay=config["l2_reg"],
        )

    def train(self):
        print(f"Starting training of D3PM", file=sys.stdout)

        # Set model to train mode
        self.model.train(True)

        for epoch_i in range(self.epoch, self.num_epochs):
            epoch_losses = []
            zero_df_losses = []
            inf_df_losses = []
            verbosed_losses_epoch = {}

            for batch_i, batch in enumerate(self.train_loader):
                # Train
                self.step += 1
                self.optimizer.zero_grad()
                batch = batch.to(self.device)

                x_t, t = self.model.get_noisy_data(batch)
                logits_zero_df, logits_inf_df = self.model(x_t, t)
                (
                    loss_zero_dof,
                    loss_inf_dof,
                    verbosed_losses,
                ) = self.model.training_losses(
                    logits_zero_df, logits_inf_df, batch, x_t, t
                )
                loss = loss_zero_dof + loss_inf_dof  # TODO: weight these appropriately

                loss.backward()
                self.optimizer.step()

                # Save and log loss
                epoch_losses.append(loss.item())
                zero_df_losses.append(loss_zero_dof.item())
                inf_df_losses.append(loss_inf_dof.item())

                verbose_log_dict = {}
                if verbosed_losses is not None:
                    for verbosed_loss, verbosed_loss_value in verbosed_losses.items():

                        prev_loss = verbosed_losses_epoch.get(verbosed_loss)
                        if prev_loss is None:
                            verbosed_losses_epoch[verbosed_loss] = [
                                verbosed_loss_value.item()
                            ]
                        else:
                            verbosed_losses_epoch[verbosed_loss] += [
                                verbosed_loss_value.item()
                            ]

                        verbose_log_dict[verbosed_loss] = verbosed_loss_value.item()

                loss_dict = {
                    "train/step_loss": loss.item(),
                    "train/zero_df_loss": loss_zero_dof.item(),
                    "train/inf_df_loss": loss_inf_dof.item(),
                }
                loss_dict.update(verbose_log_dict)

                self.logger.log(
                    loss_dict,
                    step=self.step,
                )

                # store current state
                current_state = {
                    "epoch": epoch_i,
                    "model_state_dict": self.model.state_dict(),
                    "config": self.config,
                    "optimizer_state_dict": self.optimizer.state_dict(),
                }

                # Evaluate
                if (
                    (epoch_i * self.num_steps_per_epoch + batch_i + 1)
                    % self.val_interval
                ) == 0:
                    self.model.eval()
                    # TODO: evaluate? Right now, we don't have any metrics to evaluate
                    # evaluate_wyckoff_samples(self.model, self.config, val=True, logger=self.logger,
                    #                                    step=self.step)
                    self.model.train()

            mean_epoch_loss = np.mean(epoch_losses)
            mean_zero_df_loss = np.mean(zero_df_losses)
            mean_inf_df_loss = np.mean(inf_df_losses)

            mean_verbosed_losses_epoch_log_dict = {}
            if verbosed_losses is not None:
                for verbosed_loss, verbosed_loss_value in verbosed_losses_epoch.items():
                    mean_verbosed_losses_epoch_log_dict[
                        f"train/epoch_{verbosed_loss}"
                    ] = np.mean(verbosed_loss_value)

            epoch_loss_dict = {
                "train/epoch_loss": mean_epoch_loss,
                "train/epoch_zero_df_loss": mean_zero_df_loss,
                "train/epoch_inf_df_loss": mean_inf_df_loss,
            }
            epoch_loss_dict.update(mean_verbosed_losses_epoch_log_dict)

            self.logger.log(
                epoch_loss_dict,
                step=self.step,
            )

            print(
                f"Epoch {epoch_i}\n\t loss: {mean_epoch_loss:.4}\n\t zero df loss: {mean_zero_df_loss:.4} \n\t inf df_loss: {mean_inf_df_loss:.4}",
                file=sys.stdout,
            )
            if verbosed_losses is not None:
                print(
                    "Verbosed losses",
                    file=sys.stdout,
                )
                for key, value in mean_verbosed_losses_epoch_log_dict.items():
                    print(
                        f"\t {key}: {value:.4}",
                        file=sys.stdout,
                    )

            self.logger.save_checkpoint(current_state, best=False)
            if (epoch_i + 1) % 100 == 0:
                self.logger.save_checkpoint(current_state, best=False, epoch=epoch_i)
            self.logger.log({"epoch": epoch_i}, step=self.step)

        print("Training finished!", file=sys.stdout)
        self.model.train(False)

    def run(self):
        self.train()
