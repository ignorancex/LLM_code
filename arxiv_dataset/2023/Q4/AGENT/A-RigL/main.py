import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import torch
from tqdm import tqdm
import wandb
from omegaconf import DictConfig, OmegaConf
import numpy as np
from scipy.special import softmax
import pickle
import matplotlib.pyplot as plt

import pickle

from data import get_dataloaders
from loss import LabelSmoothingCrossEntropy
from models import registry as model_registry
from sparselearning.core import Masking
from sparselearning.funcs.decay import registry as decay_registry
from sparselearning.utils.accuracy_helper import get_topk_accuracy
from sparselearning.utils.smoothen_value import SmoothenValue
from sparselearning.utils.train_helper import (
    get_optimizer_snap,
    get_optimizer,
    load_weights,
    save_weights,
)
from sparselearning.utils import layer_wise_density

if TYPE_CHECKING:
    from sparselearning.utils.typing_alias import *


def train(
    model: "nn.Module",
    model_snap: "nn.Module",
    mask: "Masking",
    #mask_snap: "Masking",
    train_loader: "DataLoader",
    optimizer: "optim",
    optimizer_snap: "optim",
    lr_scheduler: "lr_scheduler",
    global_step: int,
    epoch: int,
    device: torch.device,
    label_smoothing: float = 0.0,
    log_interval: int = 100,
    use_wandb: bool = False,
    masking_apply_when: str = "epoch_end",
    masking_interval: int = 1,
    masking_end_when: int = -1,
    masking_print_FLOPs: bool = False,
    tau: float = 0.1,
    alpha: float = 0.3,
    gamma: float = 0.1,
) -> "Union[float,int]":
    assert masking_apply_when in ["step_end", "epoch_end"]
    model.train()
    model_snap.train()
    _mask_update_counter = 0
    _loss_collector = SmoothenValue()
    pbar = tqdm(total=len(train_loader), dynamic_ncols=True)
    smooth_CE = LabelSmoothingCrossEntropy(label_smoothing)

    #te_acc_seq = []; te_loss_seq = []
    tr_acc_seq = []; tr_loss_seq = []
    tau_seq = []; tau_curr_seq = []

    loss_curr = []; loss_last = []
    #mask_snap.optimizer.zero_grad()  # zero_grad outside for loop, accumulate gradient inside
    optimizer_snap.zero_grad() 
    optimizer.zero_grad() 
    mul = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        mul += 1
        data, target = data.to(device), target.to(device)
        #optimizer.zero_grad()
        output = model(data)
        output_snap = model_snap(data)
        loss = smooth_CE(output, target)
        loss_snap = smooth_CE(output_snap, target)
        loss.backward()

        loss_curr.append(loss.detach().cpu().numpy())
        loss_last.append(loss_snap.detach().cpu().numpy())
    
    tau_curr = np.cov(loss_curr, loss_last, ddof=1)[0][1] / np.var(loss_last, ddof=1)
    tau = (1 - alpha) * tau + alpha * tau_curr
    tau_seq.append(tau); tau_curr_seq.append(tau_curr)
    print("curr tau: {}; tau used: {}".format(tau_curr, tau))

    #u = mask_snap.optimizer.get_param_groups()
    u = optimizer.get_param_groups()
    mask.optimizer.set_u(u)
    #optimizer.set_u(u)

    #mask_snap.optimizer.set_param_groups(mask.optimizer.get_param_groups())
    optimizer_snap.set_param_groups(optimizer.get_param_groups())

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        mask.optimizer.zero_grad()
        optimizer.zero_grad()

        output = model(data)
        loss = smooth_CE(output, target)
        loss.backward()
        # L2 Regularization

        #mask_snap.optimizer.zero_grad()
        optimizer_snap.zero_grad()
        output_snap = model_snap(data)
        loss_snap = smooth_CE(output_snap, target)
        loss_snap.backward()

        #mask.step(mask_snap.get_param_groups(), tau * gamma, mul)

        # Exp avg collection
        _loss_collector.add_value(loss.item())

        # Mask the gradient step
        stepper = mask if mask else optimizer
        if (
            mask
            and masking_apply_when == "step_end"
            and global_step < masking_end_when
            and ((global_step + 1) % masking_interval) == 0
        ):
            mask.update_connections()
            _mask_update_counter += 1
        else:
            if mask:
                #tau = 0
                stepper.step(optimizer_snap.get_param_groups(), tau * gamma, mul)
            else:
                #tau = 0
                stepper.step(optimizer_snap.get_param_groups(), tau * gamma, mul)

        # Lr scheduler
        lr_scheduler.step()
        pbar.update(1)
        global_step += 1

        def accuracy(yhat, labels):
            _, indices = yhat.max(1)
            return (indices == labels).sum().data.item() / float(len(labels))

        acc_iter = accuracy(output, target)
        tr_acc_seq.append(acc_iter); tr_loss_seq.append(loss.detach().cpu().numpy())

        if batch_idx % log_interval == 0:
            msg = f"Train Epoch {epoch} Iters {global_step} Mask Updates {_mask_update_counter} Train loss {_loss_collector.smooth:.6f}"
            pbar.set_description(msg)

            if use_wandb:
                log_dict = {"train_loss": loss, "lr": lr_scheduler.get_lr()[0]}
                if mask:
                    density = mask.stats.total_density
                    log_dict = {
                        **log_dict,
                        "prune_rate": mask.prune_rate,
                        "density": density,
                    }
                wandb.log(
                    log_dict,
                    step=global_step,
                )

    density = mask.stats.total_density if mask else 1.0
    msg = f"Train Epoch {epoch} Iters {global_step} Mask Updates {_mask_update_counter} Train loss {_loss_collector.smooth:.6f} Prune Rate {mask.prune_rate if mask else 0:.5f} Density {density:.5f}"

    if masking_print_FLOPs:
        log_dict = {
            "Inference FLOPs": mask.inference_FLOPs / mask.dense_FLOPs,
            "Avg Inference FLOPs": mask.avg_inference_FLOPs / mask.dense_FLOPs,
        }

        log_dict_str = " ".join([f"{k}: {v:.4f}" for (k, v) in log_dict.items()])
        msg = f"{msg} {log_dict_str}"
        if use_wandb:
            wandb.log(
                {
                    **log_dict,
                    "layer-wise-density": layer_wise_density.wandb_bar(mask),
                },
                step=global_step,
            )

    logging.info(msg)

    return _loss_collector.smooth, global_step, tr_acc_seq, tr_loss_seq, tau_seq, tau_curr_seq, tau


def evaluate(
    model: "nn.Module",
    loader: "DataLoader",
    cfg: 'DictConfig',
    global_step: int,
    epoch: int,
    device: torch.device,
    is_test_set: bool = False,
    use_wandb: bool = False,
    is_train_set: bool = False,
) -> "Union[float, float]":
    model.eval()

    loss = 0
    correct = 0
    n = 0
    pbar = tqdm(total=len(loader), dynamic_ncols=True)
    smooth_CE = LabelSmoothingCrossEntropy(0.0)  # No smoothing for val

    top_1_accuracy_ll = []
    top_5_accuracy_ll = []

    logits_list = []; labels_list = []

    with torch.no_grad():
        idx = 0
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            probs = np.exp(output.cpu().numpy())
            probs_max = np.reshape(np.max(probs, axis=1), (-1,1))
            preds = np.reshape([np.argmax(probs[m, ]) for m in range(probs.shape[0])], (-1,1))

            if idx == 0:
                probs_list = probs
                probs_max_list = probs_max
                preds_list = preds
                y_true = target.cpu().numpy()
            else:
                probs_list = np.concatenate((probs_list, probs), axis=0)
                probs_max_list = np.concatenate((probs_max_list, probs_max), axis=0)
                preds_list = np.concatenate((preds_list, preds), axis=0)
                y_true = np.concatenate((y_true, target.cpu().numpy()), axis=0)

            #logits_list.append(output)
            #labels_list.append(target)

            loss += smooth_CE(output, target).item()  # sum up batch loss

            top_1_accuracy, top_5_accuracy = get_topk_accuracy(
                output, target, topk=(1, 5)
            )
            top_1_accuracy_ll.append(top_1_accuracy)
            top_5_accuracy_ll.append(top_5_accuracy)

            pbar.update(1)

            idx += 1

        loss /= len(loader)
        top_1_accuracy = torch.tensor(top_1_accuracy_ll).mean()
        top_5_accuracy = torch.tensor(top_5_accuracy_ll).mean()

    val_or_test = "val" if not is_test_set else "test"
    val_or_test = val_or_test if not is_train_set else 'train'
    msg = f"{val_or_test.capitalize()} Epoch {epoch} Iters {global_step} {val_or_test} loss {loss:.6f} top-1 accuracy {top_1_accuracy:.4f} top-5 accuracy {top_5_accuracy:.4f} density level {cfg.masking.density}"
    pbar.set_description(msg)
    logging.info(msg)

    # Log loss, accuracy
    if use_wandb:
        wandb.log({f"{val_or_test}_loss": loss}, step=global_step)
        wandb.log({f"{val_or_test}_accuracy": top_1_accuracy}, step=global_step)
        wandb.log({f"{val_or_test}_top_5_accuracy": top_5_accuracy}, step=global_step)

    return loss, top_1_accuracy#, top_5_accuracy, ece


def single_seed_run(cfg: DictConfig) -> float:
    print(OmegaConf.to_yaml(cfg))

    # Manual seeds
    torch.manual_seed(cfg.seed)

    # Set device
    if cfg.device == "cuda" and torch.cuda.is_available():
        device = torch.device(cfg.device)
    else:
        device = torch.device("cpu")

    # Get data
    train_loader, val_loader, test_loader = get_dataloaders(**cfg.dataset)

    # Select model
    assert (
        cfg.model in model_registry.keys()
    ), f"Select from {','.join(model_registry.keys())}"
    model_class, model_args = model_registry[cfg.model]
    _small_density = cfg.masking.density if cfg.masking.name == "Small_Dense" else 1.0
    model = model_class(*model_args, _small_density).to(device)
    model_snap = model_class(*model_args, _small_density).to(device)

    # wandb
    if cfg.wandb.use:
        with open(cfg.wandb.api_key) as f:
            os.environ["WANDB_API_KEY"] = f.read().strip()
            os.environ["WANDB_START_METHOD"] = "thread"

        wandb.init(
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            reinit=True,
            save_code=True,
        )
        wandb.watch(model)

    # Training multiplier
    cfg.optimizer.decay_frequency *= cfg.optimizer.training_multiplier
    cfg.optimizer.decay_frequency = int(cfg.optimizer.decay_frequency)

    cfg.optimizer.epochs *= cfg.optimizer.training_multiplier
    cfg.optimizer.epochs = int(cfg.optimizer.epochs)

    if cfg.masking.get("end_when", None):
        cfg.masking.end_when *= cfg.optimizer.training_multiplier
        cfg.masking.end_when = int(cfg.masking.end_when)

    # Setup optimizers, lr schedulers
    optimizer, (lr_scheduler, warmup_scheduler) = get_optimizer(model, **cfg.optimizer)
    optimizer_snap = get_optimizer_snap(model_snap, **cfg.optimizer)

    # Setup mask
    mask = None
    if not cfg.masking.dense:
        max_iter = (
            cfg.masking.end_when
            if cfg.masking.apply_when == "step_end"
            else cfg.masking.end_when * len(train_loader)
        )

        kwargs = {"prune_rate": cfg.masking.prune_rate, "T_max": max_iter}

        if cfg.masking.decay_schedule == "magnitude-prune":
            kwargs = {
                "final_sparsity": 1 - cfg.masking.final_density,
                "T_max": max_iter,
                "T_start": cfg.masking.start_when,
                "interval": cfg.masking.interval,
            }

        decay = decay_registry[cfg.masking.decay_schedule](**kwargs)

        mask = Masking(
            optimizer,
            decay,
            density=cfg.masking.density,
            dense_gradients=cfg.masking.dense_gradients,
            sparse_init=cfg.masking.sparse_init,
            prune_mode=cfg.masking.prune_mode,
            growth_mode=cfg.masking.growth_mode,
            redistribution_mode=cfg.masking.redistribution_mode,
        )
        """
        mask_snap = Masking(
            optimizer_snap,
            decay,
            density=cfg.masking.density,
            dense_gradients=cfg.masking.dense_gradients,
            sparse_init=cfg.masking.sparse_init,
            prune_mode=cfg.masking.prune_mode,
            growth_mode=cfg.masking.growth_mode,
            redistribution_mode=cfg.masking.redistribution_mode,
        )
        """
        # Support for lottery mask
        lottery_mask_path = Path(cfg.masking.get("lottery_mask_path", ""))
        mask.add_module(model, lottery_mask_path)
        #mask_snap.add_module(model_snap, lottery_mask_path)

    # Load from checkpoint
    model, optimizer, mask, step, start_epoch, best_val_loss = load_weights(
        model, optimizer, mask, ckpt_dir=cfg.ckpt_dir, resume=cfg.resume
    )
    """
    model_snap, optimizer_snap, mask_snap, step, start_epoch, best_val_loss = load_weights(
        model_snap, optimizer_snap, mask_snap, ckpt_dir=cfg.ckpt_dir, resume=cfg.resume
    )
    """

    # Train model
    epoch = 0
    warmup_steps = cfg.optimizer.get("warmup_steps", 0)
    warmup_epochs = warmup_steps / len(train_loader)

    if (cfg.masking.print_FLOPs and cfg.wandb.use) and (start_epoch, step == (0, 0)):
        if mask:
            # Log initial inference flops etc
            log_dict = {
                "Inference FLOPs": mask.inference_FLOPs / mask.dense_FLOPs,
                "Avg Inference FLOPs": mask.avg_inference_FLOPs / mask.dense_FLOPs,
                "layer-wise-density": layer_wise_density.wandb_bar(mask),
            }
            wandb.log(log_dict, step=0)

    tr_acc_seq_total = []; tr_loss_seq_total = []
    vr_acc_seq_total = []; vr_loss_seq_total = []
    tau_seq_total = []; tau_curr_seq_total = []
    tau = 0.1

    sp = str(1 - cfg.masking.density)
    if len(sp) > 6:
        sp = sp[:6]
    sp = sp.split('.')
    if len(sp) == 2:
        sp = sp[0] + '_' + sp[1]
    else:
        sp = sp[0]
    f_name = str(cfg.dataset['name']) + "_" + str(cfg.model) + '_' + sp + '.pickle'

    for epoch in range(start_epoch, cfg.optimizer.epochs):
        # step here is training iters not global steps
        _masking_args = {}
        if mask:
            _masking_args = {
                "masking_apply_when": cfg.masking.apply_when,
                "masking_interval": cfg.masking.interval,
                "masking_end_when": cfg.masking.end_when,
                "masking_print_FLOPs": cfg.masking.get("print_FLOPs", False),
            }

        scheduler = lr_scheduler if (epoch >= warmup_epochs) else warmup_scheduler
        _, step, tr_acc_seq, tr_loss_seq, tau_seq, tau_curr_seq, tau = train(
            model,
            model_snap,
            mask,
            #mask_snap,
            train_loader,
            optimizer,
            optimizer_snap,
            scheduler,
            step,
            epoch + 1,
            device,
            label_smoothing=cfg.optimizer.label_smoothing,
            log_interval=cfg.log_interval,
            use_wandb=cfg.wandb.use,
            tau = tau,
            **_masking_args,
        )

        tr_acc_seq_total.append(tr_acc_seq); tr_loss_seq_total.append(tr_loss_seq)
        tau_seq_total.append(tau_seq); tau_curr_seq_total.append(tau_curr_seq)

        # Run validation
        if epoch % cfg.val_interval == 0:
            
            val_loss, val_accuracy = evaluate(
                model,
                val_loader,
                cfg,
                step,
                epoch + 1,
                device,
                use_wandb=cfg.wandb.use,
            )
            vr_acc_seq_total.append(val_accuracy); vr_loss_seq_total.append(val_loss)
            """
            te_loss, te_accuracy = evaluate(
                model,
                test_loader,
                cfg,
                step,
                epoch + 1,
                device,
                is_test_set=True,
                use_wandb=cfg.wandb.use,
            )           
            """

            pickle.dump([tr_acc_seq_total, tr_loss_seq_total, vr_acc_seq_total, vr_loss_seq_total, tau_seq_total, tau_curr_seq_total], open('/home/lyy19002/bowen/rigl_make_vr/stats/' + f_name, 'wb'))


            # Save weights
            if (epoch + 1 == cfg.optimizer.epochs) or (
                (epoch + 1) % cfg.ckpt_interval == 0
            ):
                if val_loss < best_val_loss:
                    is_min = True
                    best_val_loss = val_loss
                else:
                    is_min = False

                save_weights(
                    model,
                    optimizer,
                    mask,
                    val_loss,
                    step,
                    epoch + 1,
                    ckpt_dir=cfg.ckpt_dir,
                    is_min=is_min,
                )

        # Apply mask
        if (
            mask
            and cfg.masking.apply_when == "epoch_end"
            and epoch < cfg.masking.end_when
        ):
            if epoch % cfg.masking.interval == 0:
                mask.update_connections()

    if not epoch:
        # Run val anyway
        epoch = cfg.optimizer.epochs - 1
        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            cfg,
            step,
            epoch + 1,
            device,
            use_wandb=cfg.wandb.use,
        )

    evaluate(
        model,
        test_loader,
        cfg,
        step,
        epoch + 1,
        device,
        is_test_set=True,
        use_wandb=cfg.wandb.use,
    )

    evaluate(
        model,
        train_loader,
        cfg,
        step,
        epoch + 1,
        device,
        is_test_set=True,
        use_wandb=cfg.wandb.use,
        is_train_set=True,
    )

    sp = str(1 - cfg.masking.density)
    if len(sp) > 6:
        sp = sp[:6]
    sp = sp.split('.')
    if len(sp) == 2:
        sp = sp[0] + '_' + sp[1]
    else:
        sp = sp[0]
    f_name = str(cfg.dataset['name']) + "_" + str(cfg.model) + '_' + sp + '.pickle'
    pickle.dump([tr_acc_seq_total, tr_loss_seq_total, vr_acc_seq_total, vr_loss_seq_total, tau_seq_total, tau_curr_seq_total], open('/home/lyy19002/bowen/rigl_make_vr/stats/' + f_name, 'wb'))

    if cfg.wandb.use:
        # Close wandb context
        wandb.join()

    return val_accuracy


@hydra.main(config_name="config", config_path="conf")
def main(cfg: DictConfig) -> float:
    if cfg.multi_seed:
        val_accuracy_ll = []
        for seed in cfg.multi_seed:
            run_cfg = deepcopy(cfg)
            run_cfg.seed = seed
            run_cfg.ckpt_dir = f"{cfg.ckpt_dir}_seed={seed}"
            val_accuracy = single_seed_run(run_cfg)
            val_accuracy_ll.append(val_accuracy)

        return sum(val_accuracy_ll) / len(val_accuracy_ll)
    else:
        val_accuracy = single_seed_run(cfg)
        return val_accuracy


if __name__ == "__main__":
    main()
