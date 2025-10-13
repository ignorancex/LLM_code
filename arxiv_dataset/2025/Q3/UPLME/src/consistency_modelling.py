import os
import logging
from pathlib import Path
import torch
from torch import Tensor
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import WandbLogger

import optuna
from optuna.integration import PyTorchLightningPruningCallback
import plotly # required for optuna.visualisation

from paired_texts_modelling import PairedTextModelController, CrossEncoderProbModel, LitPairedTextModel
from utils import log_info

logger = logging.getLogger(__name__)

class LitTwoModels(LitPairedTextModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.approach == "cross-prob":
            # we already have self.model from the parent class.
            self.model_2 = CrossEncoderProbModel(plm_name=self.hparams.plm_names[1])
        else:
            raise ValueError(f"Invalid or not-implemented approach: {self.approach}")

    def _enable_dropout_at_inference(self):
        for m in self.model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()

        for m in self.model_2.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()

    def test_step(self, batch, batch_idx):
        mean_1s, mean_2s, var_1s, var_2s = [], [], [], []

        if self.num_passes > 1:
            self._enable_dropout_at_inference()

        for _ in range(self.num_passes):
            
            mean_1, var_1, _, _ = self.model(
                input_ids=batch["input_ids_1"],
                attention_mask=batch["attention_mask_1"]
            )
            mean_2, var_2, _, _ = self.model_2(
                input_ids=batch["input_ids_2"],
                attention_mask=batch["attention_mask_2"]
            )

            mean_1s.append(mean_1)
            mean_2s.append(mean_2)
            var_1s.append(var_1)
            var_2s.append(var_2)

        mean_1 = torch.stack(mean_1s, dim=0).mean(dim=0) # (bsz)
        mean_2 = torch.stack(mean_2s, dim=0).mean(dim=0)
        var_1 = torch.stack(var_1s, dim=0).mean(dim=0)
        var_2 = torch.stack(var_2s, dim=0).mean(dim=0)

        mean = 0.5 * (mean_1 + mean_2)
        var = 0.5 * (var_1 + var_2)

        outputs = {
            "mean": mean.unsqueeze(0) if mean.dim() == 0 else mean,
            "var": var.unsqueeze(0) if var.dim() == 0 else var
        }

        if "labels" in batch:
            outputs["labels"] = batch["labels"].unsqueeze(0) if batch["labels"].dim() == 0 else batch["labels"]

        self.test_outputs.append(outputs)

class LitUCVME(LitTwoModels):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _compute_loss(
            self, mean_1: Tensor, mean_2: Tensor, var_1: Tensor, var_2: Tensor,
            labels: Tensor, prefix: str
        ) -> tuple[Tensor, dict]:
        loss_dict = {}

        avg_var = 0.5 * (var_1 + var_2)
        
        nll_1 = F.gaussian_nll_loss(mean_1, labels.squeeze(), avg_var)
        nll_2 = F.gaussian_nll_loss(mean_2, labels.squeeze(), avg_var)
        nll = nll_1 + nll_2
        loss_dict[f"{prefix}_nll"] = nll.item()

        consistency_1 = F.mse_loss(var_1, avg_var)
        consistency_2 = F.mse_loss(var_2, avg_var)
        consistency = consistency_1 + consistency_2
        loss_dict[f"{prefix}_consistency"] = consistency.item()
        
        loss = nll + consistency
        loss_dict[f"{prefix}_loss"] = loss.item()

        return loss, loss_dict

    def training_step(self, batch: dict, batch_idx):
        mean_1s, mean_2s, var_1s, var_2s = [], [], [], []
        with torch.no_grad():
            for _ in range(self.num_passes):
                mean_1, var_1, _, _ = self.model(
                    input_ids=batch["input_ids_1"],
                    attention_mask=batch["attention_mask_1"]
                )
                mean_2, var_2, _, _ = self.model_2(
                    input_ids=batch["input_ids_2"],
                    attention_mask=batch["attention_mask_2"]
                )

                mean_1s.append(mean_1)
                var_1s.append(var_1)
                mean_2s.append(mean_2)
                var_2s.append(var_2)
            
        mean_1 = torch.stack(mean_1s, dim=0).mean(dim=0) # (num_passes, bsz) -> (bsz)
        mean_2 = torch.stack(mean_2s, dim=0).mean(dim=0)
        var_1 = torch.stack(var_1s, dim=0).mean(dim=0)
        var_2 = torch.stack(var_2s, dim=0).mean(dim=0)

        loss_reg_cps, _ = self._compute_loss(
            mean_1=mean_1, mean_2=mean_2, 
            var_1=var_1, var_2=var_2, 
            labels=batch["labels"], prefix="train_lbl"
        )

        outcome = batch["labels"]

        mean_1, var_1, _, _ = self.model(
            input_ids=batch["input_ids_1"],
            attention_mask=batch["attention_mask_1"]
        )
        mean_2, var_2, _, _ = self.model_2(
            input_ids=batch["input_ids_2"],
            attention_mask=batch["attention_mask_2"]
        )

        # we are not scaling-rescaling the labels, so no y_mean and y_std
        loss_mse_1 = F.mse_loss(mean_1, outcome)
        loss1 = torch.mul(torch.exp(-(var_1 + var_2) / 2), loss_mse_1)
        loss2 = (var_1 + var_2) / 2
        loss_1 = .5 * (loss1 + loss2)

        loss_reg_1 = loss_1.mean()

        loss_mse_2 = F.mse_loss(mean_2, outcome)
        loss1 = torch.mul(torch.exp(-(var_1 + var_2) / 2), loss_mse_2)
        loss2 = (var_1 + var_2) / 2
        loss_2 = .5 * (loss1 + loss2)

        loss_reg_2 = loss_2.mean()

        loss_reg = (loss_reg_1 + loss_reg_2)
        loss = loss_reg + self.lambdas[0] * loss_reg_cps +  ((var_2 - var_1) ** 2).mean()
        
        return loss
    
    def validation_step(self, batch: dict, batch_idx):
        mean_1s, mean_2s, var_1s, var_2s = [], [], [], []

        if self.num_passes > 1:
            self._enable_dropout_at_inference()
        
        for _ in range(self.num_passes):
            
            mean_1, var_1, _, _ = self.model(
                input_ids=batch["input_ids_1"],
                attention_mask=batch["attention_mask_1"]
            )
            mean_2, var_2, _, _ = self.model_2(
                input_ids=batch["input_ids_2"],
                attention_mask=batch["attention_mask_2"]
            )

            mean_1s.append(mean_1)
            mean_2s.append(mean_2)
            var_1s.append(var_1)
            var_2s.append(var_2)

        mean_1 = torch.stack(mean_1s, dim=0).mean(dim=0) # (bsz)
        mean_2 = torch.stack(mean_2s, dim=0).mean(dim=0)
        var_1 = torch.stack(var_1s, dim=0).mean(dim=0)
        var_2 = torch.stack(var_2s, dim=0).mean(dim=0)

        mean = 0.5 * (mean_1 + mean_2)
        var = 0.5 * (var_1 + var_2)

        self.validation_outputs.append({
            "mean": mean.unsqueeze(0) if mean.dim() == 0 else mean,
            "var": var.unsqueeze(0) if var.dim() == 0 else var,
            "labels": batch["labels"].unsqueeze(0) if batch["labels"].dim() == 0 else batch["labels"]
        })

    
class TwoModelsController(PairedTextModelController):
    def __init__(
            self,
            is_ucvme: bool,
            *args, **kwargs
        ):
        
        super().__init__(*args, **kwargs)

        self.is_ucvme = is_ucvme
        if self.is_ucvme:
            self.model_class = LitUCVME
        else:
            self.model_class = LitTwoModels

    def _seed_wise_train_validate(self, seed: int, curr_log_dir: str, extra_callbacks: list | None = None) -> tuple[str, dict]:
        L.seed_everything(seed)

        train_dl = self.dm.get_train_dl(
            data_path_list=self.train_file,
            batch_size=self.train_bsz,
            sanitise_newsemp_labels=False,
            add_noise=False,
            seed=seed,
            is_newsemp=self.is_newsemp_main,
            lbl_split=self.lbl_split,
            do_augment=self.do_augment
        )

        trainer = self._prepare_trainer(curr_log_dir=curr_log_dir, extra_callbacks=extra_callbacks)
        
        # https://lightning.ai/docs/pytorch/stable/advanced/model_init.html
        if os.path.exists(curr_log_dir) and not self.do_tune:
            log_info(logger, f"Seed-level logging directory already exists: {curr_log_dir}. So, validating on the saved ckpt...")
            ckpt_list = list(Path(curr_log_dir).rglob("*.ckpt"))
            assert len(ckpt_list) == 1, f"Number of ckpt is not 1."
            best_model_ckpt = ckpt_list[0]
        else:
            log_info(logger, f"Training ...")  
            with trainer.init_module():
                # model created here directly goes to GPU
                model = self.model_class(
                    num_passes=self.num_passes,
                    plm_names=self.plm_names,
                    lr=self.lr,
                    log_dir=curr_log_dir,
                    save_uc_metrics=self.save_uc_metrics,
                    error_decay_factor=self.error_decay_factor,
                    lambdas=self.lambdas,
                    approach=self.approach,
                    sep_token_id=self.dm.tokeniser.sep_token_id
                )
            
            trainer.fit(
                model=model,
                train_dataloaders=train_dl,
                val_dataloaders=self.val_dl
            )

            # getting the best model from the trainer
            best_model_ckpt = trainer.checkpoint_callback.best_model_path
        
        # final validation at the end of training
        log_info(logger, f"Loading the best model from {best_model_ckpt}")
        with trainer.init_module(empty_init=True):
            model = self.model_class.load_from_checkpoint(best_model_ckpt)

        trainer.validate(model=model, dataloaders=self.val_dl)

        metrics = {key: value.item() for key, value in trainer.callback_metrics.items()}

        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.experiment.finish()

        # experimental - stop the trainer
        # if trainer.is_global_zero:
        del trainer
        torch.cuda.empty_cache()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        return best_model_ckpt, metrics

    def evaluate(self, model_path: str):
        tester = L.Trainer(
            logger=False,
            devices=1,
            max_epochs=1
        )

        with tester.init_module(empty_init=True):
            model = self.model_class.load_from_checkpoint(model_path)

        tester.test(model=model, dataloaders=self.test_dl, verbose=True)

        metrics = {key: value.item() for key, value in tester.callback_metrics.items()}
        
        return metrics

    def optuna_objective(self, trial: optuna.trial.Trial, optuna_seed: int, optuna_log_dir: str) -> float:
        if self.is_ucvme:
            # only one hp
            self.lambdas = [trial.suggest_float("lambda_0", 0.0, 50.0)]
        else:            
            lambda_1 = trial.suggest_float("lambda_1", 0.0, 50.0)
            lambda_2 = trial.suggest_float("lambda_2", 0.0, 50.0)
            lambda_3 = trial.suggest_float("lambda_3", 0.0, 50.0)
            lambda_4 = trial.suggest_float("lambda_4", 0.0, 50.0)
            self.lambdas = [1.0, lambda_1, lambda_2, lambda_3, lambda_4]

            # self.error_decay_factor = trial.suggest_float("error_decay_factor", 0.0, 3.0, step=0.5)

        pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_ccc")
        _, metrics = self._seed_wise_train_validate(
            seed=optuna_seed,
            curr_log_dir=optuna_log_dir,
            extra_callbacks=[pruning_callback]
        )

        return metrics["val_ccc"]
    