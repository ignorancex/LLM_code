import copy
import functools
import logging
import os
import sys

import models.registry as registry
import numpy as np
import torch
import wandb
from torch.utils import tensorboard
from torchinfo import summary
from tqdm import tqdm

from sade.datasets.loaders import get_dataloaders
from sade.models.ema import ExponentialMovingAverage
from sade.models.flows import PatchFlow
from sade.run.utils import get_flow_rundir, restore_pretrained_weights


def build_score_getter(dataset_or_loader, scorer, cache_size=-1, device="cuda"):

    if cache_size > -1:
        cache = {}
        def get_next_score_tensor(index):
            if index not in cache:
                x = dataset_or_loader[index]['image']
                x = torch.tensor(x).unsqueeze(0).to(device)
                scores = scorer(x).cpu()
                x = x.cpu()
                cache[index] = (x, scores)

            return cache[index]
    else:
        def get_next_score_tensor():
            x = next(dataset_or_loader)['image']
            x = torch.tensor(x).to(device)
            scores = scorer(x)
            return x, scores
    
    return get_next_score_tensor


def flow_trainer(config, workdir):
    kimg = config.flow.training_kimg
    log_tensorboard = config.flow.log_tensorboard
    lr = config.flow.lr
    log_interval = config.training.log_freq
    device = config.device
    ema_halflife_kimg = config.flow.ema_halflife_kimg
    ema_rampup_ratio = config.flow.ema_rampup_ratio
    fast_training_mode = config.flow.training_fast_mode

    # Initialize score model
    fp16_flag = config.fp16
    config.fp16 = True
    score_model = registry.create_model(config, log_grads=False)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(model=score_model, ema=ema, step=0)

    # Get the score model checkpoint from pretrained run
    state = restore_pretrained_weights(
        config.training.pretrained_checkpoint, state, config.device
    )
    score_model.eval().requires_grad_(False)
    scorer = registry.get_msma_score_fn(config, score_model, return_norm=False)
    config.fp16 = fp16_flag

    # Initialize flow model
    flownet = registry.create_flow(config)

    summary(flownet, depth=0, verbose=2)

    # Build data iterators
    dataloaders, datasets = get_dataloaders(
        config,
        evaluation=False,
        num_workers=4,
        infinite_sampler=True,
    )

    train_ds, eval_ds, _ = datasets

    if fast_training_mode:
        # In this mode, we only compute the scores once for each image
        # We also cache the scores for the entire dataset
        # Note that this mode skips data augmentations and thus the model may overfit
        train_score_getter = build_score_getter(
            train_ds, scorer, cache_size=len(train_ds)
        )
        eval_score_getter = build_score_getter(eval_ds, scorer, cache_size=len(eval_ds))
    else:
        train_dl, eval_dl, _ = dataloaders
        train_iter = iter(train_dl)
        eval_iter = iter(eval_dl)
        
        train_score_getter = build_score_getter(train_iter, scorer)
        eval_score_getter = build_score_getter(eval_iter, scorer)

    run_dir = get_flow_rundir(config, workdir)
    run_dir = run_dir + "_" + wandb.run.id
    os.makedirs(run_dir, exist_ok=True)
    logging.info(f"Saving checkpoints to {run_dir}")

    flow_checkpoint_path = f"{run_dir}/checkpoint.pth"
    flow_checkpoint_meta_path = f"{run_dir}/checkpoint-meta.pth"

    if os.path.exists(flow_checkpoint_meta_path):
        state_dict = torch.load(flow_checkpoint_meta_path, map_location=torch.device("cpu"))
        _ = state_dict["model_state_dict"].pop("position_encoder.cached_penc", None)
        flownet.load_state_dict(state_dict["model_state_dict"], strict=True)
        logging.info(f"Resuming checkpoint from {state_dict['kimg']}")

    # Set logger so that it outputs to both console and file
    gfile_stream = open(os.path.join(run_dir, "stdout.txt"), "w")
    file_handler = logging.StreamHandler(gfile_stream)
    stdout_handler = logging.StreamHandler(sys.stdout)

    # Override root handler
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(filename)s - %(asctime)s - %(message)s",
        handlers=[file_handler, stdout_handler],
    )

    if log_tensorboard:
        writer = tensorboard.SummaryWriter(log_dir=run_dir)

    flownet = flownet.to(device)
    # Main copy to be used for "fast" weight updates
    teacher_flow_model = copy.deepcopy(flownet).to(device)
    # Model will be updated with EMA weights
    flownet = flownet.eval().requires_grad_(False)

    # Defining optimization step
    opt = torch.optim.AdamW(teacher_flow_model.parameters(), lr=lr, weight_decay=1e-5)
    flow_train_step = functools.partial(
        PatchFlow.stochastic_step,
        train=True,
        flow_model=teacher_flow_model,
        opt=opt,
        n_patches=config.flow.patches_per_train_step,
    )
    flow_eval_step = functools.partial(
        PatchFlow.stochastic_step,
        train=False,
        flow_model=flownet,
        n_patches=config.flow.patches_per_train_step,
    )

    losses = []
    batch_sz = 1  if fast_training_mode else config.training.batch_size
    total_iters = kimg * 1000 // batch_sz + 1
    progbar = tqdm(range(total_iters))
    niter = 0
    imgcount = 0
    best_val_loss = np.inf
    checkpoint_interval = 10
    loss_dict = {}
    randint_generator = torch.Generator()

    for niter in progbar:
        if fast_training_mode:
            randidx = torch.randint(high=len(train_ds), size=(1,), generator=randint_generator)
            x_batch, scores = train_score_getter(randidx.item())
        else:
            x_batch, scores = train_score_getter()
        
        x_batch = x_batch.to(device)
        scores = scores.to(device)

        loss_dict["train_loss"] = flow_train_step(scores, x_batch)
        imgcount += x_batch.shape[0]

        # Ramp up EMA beta
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, imgcount * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_sz / max(ema_halflife_nimg, 1e-8))
        writer.add_scalar("ema_beta", ema_beta, niter)

        # Update original model with EMA weights
        for p_ema, p_net in zip(flownet.parameters(), teacher_flow_model.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        if niter % log_interval == 0:
            flownet.eval()

            with torch.no_grad():
                if fast_training_mode:
                    randidx = torch.randint(
                        high=len(eval_ds), size=(1,), generator=randint_generator
                    )
                    x_batch, scores = eval_score_getter(randidx.item())
                else:
                    x_batch, scores = eval_score_getter()
                x_batch = x_batch.to(device)
                scores = scores.to(device)

                val_loss = flow_eval_step(scores, x_batch)
                loss_dict["val_loss"] = val_loss

            progbar.set_description(f"Val Loss: {val_loss:.4f}")
            losses.append(val_loss)

        if log_tensorboard:
            for loss_type in loss_dict:
                loss_name = f"{'fast-' if fast_training_mode else ''}loss/{loss_type}"
                writer.add_scalar(loss_name, loss_dict[loss_type], niter)

        progbar.set_postfix(batch=f"{imgcount}/{kimg}K")

        if niter % checkpoint_interval == 0 and val_loss < best_val_loss:
            # if the current validation loss is the best one
            best_val_loss = val_loss  # Update the best validation loss
            torch.save(
                {
                    "kimg": niter,
                    "model_state_dict": flownet.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "val_loss": val_loss,
                },
                flow_checkpoint_meta_path,
            )

        # progbar.set_description(f"Loss: {loss:.4f}")
    if val_loss < best_val_loss:
        torch.save(
            {
                "kimg": niter,
                "model_state_dict": flownet.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "val_loss": val_loss,
            },
            flow_checkpoint_path,
        )
    else:  # Rename checkpoint_meta to checkpoint
        os.rename(flow_checkpoint_meta_path, flow_checkpoint_path)

    if log_tensorboard:
        writer.close()

    return losses


def flow_evaluator(config, workdir):
    # Initialize score model
    score_model = registry.create_model(config, log_grads=False)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(model=score_model, ema=ema, step=0, model_checkpoint_step=0)

    # Get the score model checkpoint from pretrained run
    state = restore_pretrained_weights(
        config.training.pretrained_checkpoint, state, config.device
    )

    score_model.eval().requires_grad_(False)
    scorer = registry.get_msma_score_fn(config, score_model, return_norm=False)

    # Initialize flow model
    flownet = registry.create_flow(config).eval().requires_grad_(False)

    flow_path = workdir
    ckpt_path = f"{flow_path}/checkpoint.pth"
    if not os.path.exists(ckpt_path):
        ckpt_path = f"{flow_path}/checkpoint-meta.pth"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Could not find checkpoint or checkpoint-meta at {ckpt_path}"
        )
    else:
        logging.info(f"Found checkpoint at {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))
    _ = state_dict["model_state_dict"].pop("position_encoder.cached_penc", None)
    flownet.load_state_dict(state_dict["model_state_dict"], strict=True)
    logging.info(
        f"Loaded flow model at iter={state_dict['kimg']}, val_loss={state_dict['val_loss']}"
    )
    # Use user-specified chunk size for evaluation
    flownet.patch_batch_size = config.flow.patch_batch_size

    # Create save directory
    checkpoint_step = state_dict["kimg"]
    save_dir = os.path.join(workdir, "eval", f"ckpt_{checkpoint_step}")
    os.makedirs(save_dir, exist_ok=True)
    logging.info(f"Saving flow evaluation results to {save_dir}")
    experiment = config.eval.experiment
    experiment_name = f"{experiment.inlier}_{experiment.ood}"
    enhance_lesions = False
    if "-enhanced" in experiment.ood:
        enhance_lesions = True
        experiment.ood = experiment.ood.split("-")[0]

    logging.info(f"Running experiment {experiment_name}")

    # Load datasets
    # Build data iterators
    dataloaders, _ = get_dataloaders(
        config,
        evaluation=True,
        ood_eval=True,
        num_workers=4,
        infinite_sampler=False,
    )

    _, inlier_dl, ood_dl = dataloaders

    # Get negative log-likelihoods
    x_ood_nlls = []
    for x_img_dict in tqdm(ood_dl):
        x = x_img_dict["image"].to(config.device)

        if enhance_lesions:
            labels = x_img_dict["label"].to(config.device)
            x = x * labels * 1.5 + x * (1 - labels)

        h = scorer(x)
        z = -flownet.log_density(h, x, fast=True).cpu()
        # h = h.to("cpu")
        del h
        x_ood_nlls.append(z)

    x_inlier_nlls = []
    for x in tqdm(inlier_dl):
        x = x["image"].to(config.device)
        h = scorer(x)
        z = -flownet.log_density(h, x, fast=True).cpu()
        # h.to("cpu")
        del h
        x_inlier_nlls.append(z)

    x_inlier_nlls = torch.cat(x_inlier_nlls).numpy()
    x_ood_nlls = torch.cat(x_ood_nlls).numpy()

    np.savez_compressed(
        f"{save_dir}/{experiment_name}_results.npz",
        **{"inliers": x_inlier_nlls, "ood": x_ood_nlls},
    )

    return
