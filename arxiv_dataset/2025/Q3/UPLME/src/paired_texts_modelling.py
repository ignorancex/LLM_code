import torch
from torch import Tensor
import torch.nn.functional as F
import lightning as L
from transformers import (
    AutoModel, 
    get_linear_schedule_with_warmup
)
from torchmetrics.functional import pearson_corrcoef, concordance_corrcoef, mean_squared_error, spearman_corrcoef
import logging
import numpy as np
import os
from pathlib import Path
import pandas as pd
from zipfile import ZipFile

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

import optuna
from optuna.integration import PyTorchLightningPruningCallback
import plotly # required for optuna.visualisation
from functools import partial

from preprocess import PairedTextDataModule
from utils import log_info, plot_uncertainy, process_seedwise_metrics, DelayedStartEarlyStopping, beta_nll_loss
from util_uncertainty_metrics import calculate_unc_metrics

logger = logging.getLogger(__name__)
    
class CrossEncoderProbModel(torch.nn.Module):
    def __init__(self, plm_name: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(plm_name)

        if plm_name.startswith("roberta"):
            # only applicable for roberta
            self.pooling = "roberta-pooler"
        else:
            self.pooling = "cls"

        self.out_proj_m = torch.nn.Sequential(
            torch.nn.LayerNorm(self.model.config.hidden_size),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(self.model.config.hidden_size, 1)
        )

        self.out_proj_v = torch.nn.Sequential(
            torch.nn.LayerNorm(self.model.config.hidden_size),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(self.model.config.hidden_size, 1),
            torch.nn.Softplus()
        )

    def forward(self, input_ids, attention_mask):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        if self.pooling == "mean":
            sentence_representation = (
                (output.last_hidden_state * attention_mask.unsqueeze(-1)).sum(-2) /
                attention_mask.sum(dim=-1).unsqueeze(-1)
            )
        elif self.pooling == "cls":
            sentence_representation = output.last_hidden_state[:, 0, :]
        elif self.pooling == "roberta-pooler":
            sentence_representation = output.pooler_output # (batch_size, hidden_dim)

        mean = self.out_proj_m(sentence_representation)
        var = self.out_proj_v(sentence_representation)
        var = torch.clamp(var, min=1e-8, max=1000) # following Seitzer-NeurIPS2022

        return mean.squeeze(), var.squeeze(), sentence_representation, output.last_hidden_state

class CrossEncoderBasicModel(torch.nn.Module):
    def __init__(self, plm_name: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(plm_name)
        self.pooling = "roberta-pooler"

        self.out_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(self.model.config.hidden_size),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(self.model.config.hidden_size, 1)
        )

    def forward(self, batch):
        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )

        if self.pooling == "mean":
            attn = batch['attention_mask']
            sentence_representation = (output.last_hidden_state * attn.unsqueeze(-1)).sum(-2) / attn.sum(dim=-1).unsqueeze(-1)
        elif self.pooling == "cls":
            sentence_representation = output.last_hidden_state[:, 0, :]
        elif self.pooling == "roberta-pooler":
            sentence_representation = output.pooler_output

        y_hat = self.out_proj(sentence_representation) 

        return (
            y_hat.squeeze(), 
            output.last_hidden_state
        )
    
class LitPairedTextModel(L.LightningModule):
    def __init__(
        self,
        plm_names: list[str],
        lr: float,
        log_dir: str,
        save_uc_metrics: bool,
        error_decay_factor: float,
        approach: str,
        sep_token_id: int, # required for alignment loss
        lambdas: list[float] = [], # initlisaed to compatible with old saved checkpoints
        num_passes: int = 4
    ):
        super().__init__()
        self.save_hyperparameters()

        self.approach = approach
        if self.approach == "cross-basic":
            self.model = CrossEncoderBasicModel(plm_name=plm_names[0])
        elif self.approach == "cross-prob":
            self.model = CrossEncoderProbModel(plm_name=plm_names[0])
        else:
            raise ValueError(f"Invalid approach: {self.approach}")

        self.lr = lr
        self.log_dir = log_dir
        self.save_uc_metrics = save_uc_metrics

        self.error_decay_factor = error_decay_factor
        
        self.lambdas = lambdas
        self.sep_token_id = sep_token_id
        self.num_passes = num_passes
        
        self.penalty_type = "exp-decay"

        self.validation_outputs = []
        self.test_outputs = []

    def configure_optimizers(self):
        optimiser = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=0.1
        )

        # Caution: self.trainer.num_training_batchs is inf if train_dataloader is not loaded
        # calculating num_training_steps using self.trainer.estimated_stepping_batches loads the train_dataloader
        # so, the below code "in the following order" is important
        num_training_steps = self.trainer.estimated_stepping_batches
        num_warmup_steps = int(0.03 * num_training_steps)
        # moved to 3% from 6% as I doubled the number of steps (epochs) compared to RoBERTa paper
        
        # RoBERTa paper used 10 epoch for fine-tuning, so I can calculate the number of warmup steps using 10 epochs
        # Also note num_training_batch is inf if I only set max_steps without max_epochs (I guess)
        # num_warmup_steps = int(0.06 * self.trainer.num_training_batches * 10)
        
        log_info(logger, f"Num training steps: {num_training_steps}")
        log_info(logger, f"Num warmup steps: {num_warmup_steps}")
        lr_scheduler = get_linear_schedule_with_warmup(
            optimiser,
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps
        )

        return {
            "optimizer": optimiser,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1 
            }
        }
    
    def _compute_penalty_loss(self, mean: Tensor, var: Tensor, labels: Tensor) -> Tensor:
        errors = (mean - labels) ** 2

        if self.penalty_type == "exp-decay":
            weight = torch.exp(-self.error_decay_factor * errors)
            var = var * weight
            penalty_loss = torch.linalg.norm(var, ord=2) / var.numel()
        else:
            raise ValueError(f"Invalid penalty type: {self.penalty_type}")

        return penalty_loss

    def _rescale_labels(self, labels: Tensor) -> Tensor:
        if "newsemp" in self.log_dir: # very loose check. TODO: take a variable input; not done now because then I will not be able to load saved models
            labels = (labels.float() - 4.0) / 3.0 # 1 is -1, 4 is 0, 7 is 1; like cos_sim
        elif "empstories" in self.log_dir:
            # generic, but not tested on newsemp
            min_label = 1.0
            max_label = 4.0
            labels = 2 * (labels.float() - min_label) / (max_label - min_label) - 1
        else:
            raise ValueError(f"Alignment loss requires newsemp or empstories in the log directory name. Got {self.log_dir}")
        return labels
    
    def _compute_alignment_betn_texts(self, input_ids: Tensor, hidden_state: Tensor, labels: Tensor) -> Tensor:
        # input_ids: (bsz, seq_len)
        # hidden_state: (bsz, seq_len, hidden_dim)
        bsz = input_ids.shape[0]

        input1_reprs, input2_reprs = [], []
        for i in range(bsz):
            # doing for each sample in the batch
            input_ids_sample = input_ids[i] # (seq_len)
            h_i = hidden_state[i] # (seq_len, hidden_dim)

            sep_positions = (input_ids_sample == self.sep_token_id).nonzero(as_tuple=True)[0]

            # if deberta
            # token pattern: [CLS] input1 [SEP] input2 [SEP]
            # assert len(sep_positions) == 2, f"Number of sep positions is not 2. {sep_positions}"
            # first_sep = sep_positions[0]
            # second_sep = sep_positions[1]

            # input1_repr = h_i[1:first_sep] # excluding special tokens, (first_sep-1, hidden_dim)
            # input2_repr = h_i[first_sep+1:second_sep] # excluding special tokens, (second_sep-first_sep-1, hidden_dim)
            # ^ important to not include till -1 because there could be padded stuffs

            # if roberta
            # token pattern: [CLS] input1 [SEP][SEP] input2 [SEP]
            assert len(sep_positions) == 3, f"Number of sep positions is not 3. {sep_positions}"
            first_sep = sep_positions[0]
            second_sep = sep_positions[1]
            third_sep = sep_positions[2]
            
            input1_repr = h_i[1:first_sep] # excluding special tokens, (first_sep-1, hidden_dim)
            input2_repr = h_i[second_sep+1:third_sep] # excluding special tokens, (third_sep-second_sep-1, hidden_dim)

            # Pool representation
            input1_repr = input1_repr.mean(dim=0) # (hidden_dim)
            input2_repr = input2_repr.mean(dim=0)

            input1_reprs.append(input1_repr)
            input2_reprs.append(input2_repr)
        
        input1_reprs = torch.stack(input1_reprs) # (bsz, hidden_dim)
        input2_reprs = torch.stack(input2_reprs)

        cos_sim = F.cosine_similarity(input1_reprs, input2_reprs)
                
        labels = self._rescale_labels(labels)

        loss = F.mse_loss(cos_sim, labels)
        return loss
    
    def _compute_loss(self, mean: Tensor, var: Tensor, labels: Tensor, prefix: str,
                      input_ids: Tensor, hidden_state: Tensor) -> tuple[Tensor, dict]:
        loss_dict = {}
        
        if self.approach == "cross-basic":
            # Basic model with MSE loss
            total_loss = F.mse_loss(mean, labels.squeeze())
            loss_dict[f"{prefix}_mse"] = total_loss.item()
        else:
            total_loss = 0 # initiliasing to be safe, required when both lambdas are 0
            if self.lambdas[0] != 0:
                nll_loss = self.lambdas[0] * beta_nll_loss(mean, var, labels.squeeze())
                loss_dict[f"{prefix}_nll"] = nll_loss.item()
            else:
                nll_loss = 0

            if self.lambdas[1] != 0:
                penalty_loss = self.lambdas[1] * self._compute_penalty_loss(mean=mean, var=var, labels=labels)
                loss_dict[f"{prefix}_penalty"] = penalty_loss.item()
            else:
                penalty_loss = 0

            total_loss = nll_loss + penalty_loss

        if self.lambdas[2] != 0:
            assert hidden_state is not None, "Hidden state is required for alignment loss"
            alignment_loss = self.lambdas[2] * self._compute_alignment_betn_texts(
                input_ids=input_ids,
                hidden_state=hidden_state,
                labels=labels
            )
            loss_dict[f"{prefix}_alignment"] = alignment_loss.item()

            total_loss += alignment_loss
            loss_dict[f"{prefix}_loss"] = total_loss.item()

        return total_loss, loss_dict

    def _forward_pass(self, batch: dict) -> tuple[Tensor, Tensor, Tensor]:
        means, varss, hidden_states = [], [], []

        for _ in range(self.num_passes):
            if self.approach == "cross-prob":
                mean, var, _, hidden_state = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
            elif self.approach == "cross-basic":
                mean, hidden_state = self.model(batch)
                var = torch.zeros_like(mean)
            
            means.append(mean)
            varss.append(var)
            hidden_states.append(hidden_state)
        
        mean = torch.stack(means, dim=0).mean(dim=0)
        var = torch.stack(varss, dim=0).mean(dim=0)
        hidden_state = torch.stack(hidden_states, dim=0).mean(dim=0)

        return mean, var, hidden_state
    
    def training_step(self, batch, batch_idx):
        mean, var, hidden_state = self._forward_pass(batch)
        
        loss, loss_dict = self._compute_loss(mean, var, batch["labels"], prefix="train",
                                             input_ids=batch['input_ids'],
                                             hidden_state=hidden_state)
        self.log_dict(
            loss_dict,
            on_step=True,
            on_epoch=False,
            logger=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch["labels"].shape[0]
        )
        return loss
    
    def _enable_dropout_at_inference(self):
        for m in self.model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()

    def validation_step(self, batch, batch_idx):
        if self.num_passes > 1:
            self._enable_dropout_at_inference()

        mean, var, hidden_state = self._forward_pass(batch)

        # Note: it's important to check dim and unsqeeze as we get 0-dim if the last batch has only one sample
        # Otherwise, we get an error when concatenating tensors later
        mean = mean.unsqueeze(0) if mean.dim() == 0 else mean
        var = var.unsqueeze(0) if var.dim() == 0 else var
        labels = batch["labels"].unsqueeze(0) if batch["labels"].dim() == 0 else batch["labels"]
        self.validation_outputs.append({
            "mean": mean,
            "var": var,
            "labels": labels
        })

        _, loss_dict = self._compute_loss(
            mean=mean, var=var, labels=labels, prefix="val",
            input_ids=batch['input_ids'], hidden_state=hidden_state
        )
        self.log_dict(
            loss_dict,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=labels.shape[0]
        )


    def _calculate_metrics(self, mean: Tensor, var: Tensor, label: Tensor, mode: str) -> dict:
        # requires the tensors to be on the correct device as these are used for early stopping, checkpointing, etc.
        pcc = pearson_corrcoef(mean, label).to(self.device)
        ccc = concordance_corrcoef(mean, label).to(self.device)
        scc = spearman_corrcoef(mean, label).to(self.device)
        rmse = mean_squared_error(mean, label, squared=False).to(self.device)

        metrics_dict = {
            f"{mode}_pcc": pcc.item(),
            f"{mode}_ccc": ccc.item(),
            f"{mode}_scc": scc.item(),
            f"{mode}_rmse": rmse.item()
        }

        if self.approach != "cross-basic":
            # meaning that the model is probabilistic
            # In my understanding, it is fine to have unc_metrics in CPU as it's not used for any further computation
            unc_metrics = calculate_unc_metrics(mean=mean, var=var, label=label)
            unc_metrics = {f"{mode}_{k}": v for k, v in unc_metrics.items()}
            metrics_dict.update(unc_metrics)
        
        return metrics_dict
    
    def on_validation_epoch_end(self):
        all_means = torch.cat([out["mean"] for out in self.validation_outputs])
        all_labels = torch.cat([out["labels"] for out in self.validation_outputs])
        all_vars = torch.cat([out["var"] for out in self.validation_outputs])

        all_means = all_means.to(torch.float64).cpu()
        all_labels = all_labels.to(torch.float64).cpu()
        all_vars = all_vars.to(torch.float64).cpu()

        metric_dict = self._calculate_metrics(mean=all_means, var=all_vars, label=all_labels, mode="val")

        self.log_dict(
            metric_dict,
            on_step=False, # must be False
            on_epoch=True, # must be True
            logger=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=self.validation_outputs[0]["mean"].shape[0]
        )

        self.validation_outputs.clear()

    def _save_and_plot_uncertainty(self, output: list[dict]):
        # convert tensors to numpy arrays
        output_dict = {}
        for out in output:
            for key in out:
                array = out[key].cpu().numpy()
                if key in output_dict:
                    output_dict[key] = np.concatenate((output_dict[key], array), axis=0)
                else:
                    output_dict[key] = array
        
        # save for later use
        save_as = f"{self.log_dir}/outputs.npy"
        if os.path.exists(save_as):
            save_as = f"{self.log_dir}/outputs_new.npy"
        np.save(save_as, output_dict)
        log_info(logger, f"Saved output to {save_as}")

        if "labels" in output_dict:
            plot_uncertainy(output_dict, f"{self.log_dir}/uncertainty.pdf")

    def test_step(self, batch, batch_idx):
        if self.num_passes > 1:
            self._enable_dropout_at_inference()

        mean, var, _ = self._forward_pass(batch)

        outputs = {
            "mean": mean.unsqueeze(0) if mean.dim() == 0 else mean,
            "var": var.unsqueeze(0) if var.dim() == 0 else var
        }

        if "labels" in batch:
            outputs["labels"] = batch["labels"]
        
        if "noise" in batch:
            outputs["noise"] = batch["noise"]

        self.test_outputs.append(outputs)

    def on_test_epoch_end(self):
        all_means = torch.cat([out["mean"] for out in self.test_outputs])
        all_means = all_means.to(torch.float64).cpu()
        
        all_vars = torch.cat([out["var"] for out in self.test_outputs])
        all_vars = all_vars.to(torch.float64).cpu()

        if "labels" in self.test_outputs[0]:
            all_labels = torch.cat([out["labels"] for out in self.test_outputs])
            all_labels = all_labels.to(torch.float64).cpu()

            self.log_dict(
                self._calculate_metrics(mean=all_means, var=all_vars, label=all_labels, mode="test"),
                logger=False,
                prog_bar=False,
                sync_dist=True,
                batch_size=self.test_outputs[0]["mean"].shape[0]
            )
        else:
            log_info(logger, "No labels found in test outputs. Assuming metrics to be calculated separately. Saving predictions.")
            pred_df = pd.DataFrame({'emp': all_means, 'dis': all_means}) # we're not predicting distress, just aligning with submission system
            save_as = f"{self.log_dir}/predictions_EMP.tsv"
            pred_df.to_csv(save_as, sep='\t', index=None, header=None)
            # CodaLab requires zip file
            save_as = f"{self.log_dir}/predictions.zip"
            with ZipFile(save_as, 'w') as zipf:
                zipf.write(f"{self.log_dir}/predictions_EMP.tsv", arcname="predictions_EMP.tsv") # TODO: not tested after adding arcname, but it is required to avoid the file being saved in a subdirectory
            log_info(logger, f"Saved predictions to {save_as}")
            # TODO: if the above zip is successful, then remove the tsv file

        if self.approach != "cross-basic":
            # meaning that the model is probabilistic
            self._save_and_plot_uncertainty(self.test_outputs)

        self.test_outputs.clear()

class PairedTextModelController(object):
    def __init__(
        self,
        labelled_train_files: list[str],
        val_files: list[str],
        test_files: list[str],
        lr: float,
        train_bsz: int,
        eval_bsz: int,
        # num_epochs: int,
        max_steps: int,
        val_check_interval: int | float,
        noise_level: float,
        delta: float | None,
        expt_name: str,
        debug: bool,
        do_tune: bool = False,
        do_train: bool = True,
        do_test: bool = False,
        error_decay_factor: float = 0.5,
        lambdas: list[float] = [1.0],
        approach: str = "cross-prob",
        main_data: str = "newsemp",
        lbl_split: float = 1.0,
        plm_names: list[str] = ["roberta-base"],
        num_passes: int = 1,
        sanitise_labels: bool = False,
        add_noise_train: bool = False,
        add_noise_test: bool = False,
        do_augment: bool = False
    ):
        self.train_file = labelled_train_files
        self.val_file = val_files
        self.test_file = test_files
         
        self.lr = lr
        self.train_bsz = train_bsz
        self.eval_bsz = eval_bsz
        self.max_steps = max_steps
        self.val_check_interval = val_check_interval

        self.noise_level = noise_level
        self.delta = delta
        self.expt_name = expt_name
        self.debug = debug
        self.do_tune = do_tune
        self.do_train = do_train
        self.do_test = do_test
        self.error_decay_factor = error_decay_factor
        self.lambdas = lambdas
        self.approach = approach
        self.num_passes = num_passes
        self.sanitise_labels = sanitise_labels
        self.add_noise_train = add_noise_train
        self.do_augment = do_augment

        self.enable_early_stopping = True
        self.early_stopping_start_epoch = 5 # applicable for DelayedStartEarlyStopping
        self.enable_checkpointing = True
        self.save_uc_metrics = False

        self.plm_names = plm_names
        self.lbl_split = lbl_split

        for plm_name in self.plm_names:
            if not plm_name.startswith("roberta"): # TODO: also add condition of lambda (not done now because lambda is incosistent betn two and single models)
                raise ValueError(f"Only roberta is supported for alignment loss. Got {plm_name}")

        self.dm = PairedTextDataModule(
            noise_level=self.noise_level,
            delta=self.delta,
            tokeniser_plms=self.plm_names,
            tokenise_paired_texts_each_tokeniser=True \
                if self.approach in ["cross-basic", "cross-prob"] else False
            # TODO: check if the above should be True for other approaches
            # is_separate_tokeniser=True if self.approach in ["bi-prob", "siamese"] else False # NOTE: moving to len(plm_names) based decision
        )

        # train_dl is seed-dependent, so it's created in the seed-wise training
        self.is_newsemp_main = (main_data == "newsemp")
        if self.do_train or self.do_tune:
            self.val_dl = self.dm.get_val_dl(
                data_path_list=self.val_file,
                batch_size=self.eval_bsz,
                sanitise_newsemp_labels=self.sanitise_labels,
                add_noise=False,
                is_newsemp=self.is_newsemp_main
            )
        if self.do_test:
            self.test_dl = self.dm.get_test_dl(
                data_path_list=self.test_file, batch_size=self.eval_bsz, 
                sanitise_newsemp_labels=self.sanitise_labels, add_noise=add_noise_test,
                is_newsemp=self.is_newsemp_main
            )
           
    def _prepare_trainer(self, curr_log_dir: str, extra_callbacks: list | None = None):
        callbacks = []

        if self.enable_early_stopping:
            early_stopping = DelayedStartEarlyStopping(
                start_epoch=self.early_stopping_start_epoch,
                monitor="val_ccc",
                patience=2,
                mode="max",
                min_delta=0,
                verbose=True
            )

            callbacks.append(early_stopping)
        else:
            log_info(logger, "Early stopping disabled")

        if self.enable_checkpointing:            
            # last only
            checkpoint = ModelCheckpoint(
                save_top_k=1 # saves the last checkpoint; no need to save_last=True as it will save another checkpoint unnecessarily
            )
            
            callbacks.append(checkpoint)

        lr_monitor = LearningRateMonitor()
        callbacks.append(lr_monitor)

        callbacks.extend(extra_callbacks) if extra_callbacks else None

        wandb_logger = WandbLogger(
            name=self.expt_name,
            project="NoisEmpathy",
            save_dir=curr_log_dir,
            offline=self.debug
        )

        trainer = L.Trainer(
            # max_epochs=self.num_epochs,
            val_check_interval=self.val_check_interval,
            check_val_every_n_epoch=None, # eniirely steps-based validation
            max_steps=self.max_steps,
            default_root_dir=curr_log_dir,
            deterministic=True,
            logger=wandb_logger,
            log_every_n_steps=50, # frequent logging will require more memory; watch video at https://lightning.ai/docs/pytorch/stable/common/trainer.html#log-every-n-steps
            callbacks=callbacks,
            devices="auto",
            enable_checkpointing=self.enable_checkpointing,
            limit_train_batches=0.1 if self.debug else 1.0 
        )

        return trainer

    def _seed_wise_train_validate(
            self, seed: int, curr_log_dir: str, extra_callbacks: list | None = None
        ) -> tuple[str, dict]:
        L.seed_everything(seed)

        train_dl = self.dm.get_train_dl(
            data_path_list=self.train_file,
            batch_size=self.train_bsz,
            sanitise_newsemp_labels=self.sanitise_labels,
            add_noise=self.add_noise_train,
            seed=seed,
            is_newsemp=self.is_newsemp_main,
            do_augment=self.do_augment,
            lbl_split=self.lbl_split
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
                model = LitPairedTextModel(
                    plm_names=self.plm_names,
                    lr=self.lr,
                    log_dir=curr_log_dir,
                    save_uc_metrics=self.save_uc_metrics,
                    error_decay_factor=self.error_decay_factor,
                    lambdas=self.lambdas,
                    approach=self.approach,
                    sep_token_id=self.dm.tokeniser.sep_token_id,
                    num_passes=self.num_passes
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
            model = LitPairedTextModel.load_from_checkpoint(best_model_ckpt)

        trainer.validate(model=model, dataloaders=self.val_dl)

        metrics = {key: value.item() for key, value in trainer.callback_metrics.items()}

        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.experiment.finish()

        return best_model_ckpt, metrics
    
    def evaluate(self, model_path: str):
        tester = L.Trainer(
            logger=False,
            devices=1,
            max_epochs=1
        )

        with tester.init_module(empty_init=True):
            model = LitPairedTextModel.load_from_checkpoint(model_path)

        tester.test(model=model, dataloaders=self.test_dl, verbose=True)

        metrics = {key: value.item() for key, value in tester.callback_metrics.items()}
        
        return metrics
    
    def train_test(
        self,
        seeds: list[int],
        parent_log_dir: str
    ) -> None:

        results = []
        for seed in seeds:
            log_info(logger, f"Current seed: {seed}")

            # releasing the memory if allocated
            # torch.cuda.empty_cache()

            curr_log_dir = os.path.join(parent_log_dir, f"seed_{seed}")
            if self.do_train:
                best_model_ckpt, metrics = self._seed_wise_train_validate(
                    seed=seed,
                    curr_log_dir=curr_log_dir
                )
            else:
                best_model_ckpt = list(Path(curr_log_dir).rglob("*.ckpt"))
                assert len(best_model_ckpt) == 1, f"Found {len(best_model_ckpt)} ckpt files."
                best_model_ckpt = best_model_ckpt[0]
                metrics = {} # empty val metrics
            
            if self.do_test:
                # subsequent testing
                log_info(logger, f"Testing using: {best_model_ckpt}")
                test_metrics = self.evaluate(best_model_ckpt)
                metrics = {**metrics, **test_metrics} # merge the two dictionaries 

            metrics["seed"] = seed
            log_info(logger, f"Metrics: {metrics}")
            results.append(metrics)
        save_as = os.path.join(parent_log_dir, f"results_val-{self.do_train}_test-{self.do_test}.csv")
        process_seedwise_metrics(results, save_as)

    def optuna_objective(self, trial: optuna.trial.Trial, optuna_seed: int, optuna_log_dir: str) -> float:
        lambda_1 = trial.suggest_float("lambda_1", 0.0, 50.0)
        lambda_2 = trial.suggest_float("lambda_2", 0.0, 50.0)
        self.lambdas = [1.0, lambda_1, lambda_2]
        
        # self.error_decay_factor = trial.suggest_float("error_decay_factor", 0.0, 3.0, step=0.5)

        pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_ccc")
        _, metrics = self._seed_wise_train_validate(
            seed=optuna_seed,
            curr_log_dir=optuna_log_dir,
            extra_callbacks=[pruning_callback]
        )

        # releasing the memory - trying to solve that OOM issue with slurm job 
        torch.cuda.empty_cache()

        return metrics["val_ccc"]

    def tune_train_test(self, n_trials: int, parent_log_dir: str, seeds: list[int] = [0]) -> None:
        if self.do_tune:
            optuna_log_dir = os.path.join(parent_log_dir, "optuna_logs")
            os.makedirs(optuna_log_dir, exist_ok=True)
            study = optuna.create_study(
                direction="maximize",
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
                study_name=self.expt_name,
                storage=f"sqlite:///{optuna_log_dir}/optuna.db",
                load_if_exists=True
            )

            optuna_seed = seeds[0] # using the first seed for optuna tuning
            
            objective_params = partial(
                self.optuna_objective,
                optuna_seed=optuna_seed,
                optuna_log_dir=optuna_log_dir
            )

            study.optimize(objective_params, n_trials=n_trials, show_progress_bar=False)

            study.trials_dataframe().to_csv(os.path.join(optuna_log_dir, "trial_results.csv"), index=False)
            
            best_trial = study.best_trial
            log_info(logger, f"Best trial:{best_trial.value}")

            with open(os.path.join(optuna_log_dir, "best_trial_params.txt"), 'w') as f:
                for key, value in best_trial.params.items():
                    f.write(f"{key}: {value}\n")
            
            fig_slice = optuna.visualization.plot_slice(study)
            fig_slice.write_image(os.path.join(parent_log_dir, "Optuna_slice.pdf"))

            fig_imp = optuna.visualization.plot_param_importances(study)
            fig_imp.write_image(os.path.join(parent_log_dir, "Optuna_param_importances.pdf"))
            
            if self.do_train:
                raise ValueError("Train right after tune with updated parameters is not using updated parameters. Set the parameters and train separately after tuning.")
                                #  " update is no longer correct as there is no self.lambda_1 for example."
            for key, value in best_trial.params.items():
                setattr(self, key, value)
        
        if self.do_train or self.do_test:
            self.train_test(seeds=seeds, parent_log_dir=parent_log_dir)
