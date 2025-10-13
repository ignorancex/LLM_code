import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["WANDB_MODE"] = "disabled"
# os.environ["JAX_TRACEBACK_FILTERING"] = "off"
# os.environ["JAX_DISABLE_JIT"] = "True"
# os.environ["JAX_DEBUG_NANS"] = "True"

import argparse
import functools as fts
import math
from pathlib import Path
from typing import Any

import dotenv
import einops as ein
import equinox as eqx

# import jax
import numpy as np
import optax
import yaml
from jax import nn as jnn
from jax import numpy as jnp
from jax import random as jr
from jax import vmap
from jaxtyping import Array, Float, PRNGKeyArray, PyTree, Scalar
from optax.losses import softmax_cross_entropy_with_integer_labels
from tqdm import tqdm

import wandb
from utils.backbones import Model
from utils.data import load_data
from utils.metrics import (
    expert_overlap_max,
    expert_overlap_min,
    get_parameter_count,
    img_table,
    make_wandb_histlike,
)
from utils.misc import load_model, save_model
from plots.plot_bitmap import plot_bitmaps
from plots.plot_acc_curves import plot_curves
import matplotlib.pyplot as plt

dotenv.load_dotenv()
plt.style.use("plots/myplots.mplstyle")


# change data dir if needed:
wandb_dir = os.getenv("WANDB_DATA_DIR")
if wandb_dir is None:
    print("No WANDB directory found, default to ./wandb/")
    wandb_dir = "./wandb/"

# read api key and strip newlines
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
if WANDB_API_KEY is None:
    raise ValueError(
        "WANDB_API_KEY not found. Please set it in your environment variables."
    )
wandb.login(key=WANDB_API_KEY)


def train(conf: dict = None):
    PROJECT_KWARGS = dict(
        entity="silvretta",
        project="PaperSweeps",
        group=conf["dset"],
        settings=wandb.Settings(code_dir="."),
        dir=wandb_dir,
    )
    with wandb.init(**PROJECT_KWARGS, config=conf) as run:
        cfg = wandb.config
        print("Config:")
        for k, v in cfg.items():
            print(f"{k}: {v}")

        @fts.partial(eqx.filter_value_and_grad, has_aux=True)
        def compute_loss(
            model: Model,
            x: Float[Array, "b ..."],
            y: Float[Array, "b ..."],
            temp: Float[Scalar, ""],
            key: PRNGKeyArray,
            **kwargs,
        ) -> Any:
            keys = jr.split(key, len(x))
            h0 = vmap(model.input_block)(x)
            temps = jnp.full(len(x), temp)
            magic_sequences = vmap(model.rtmoe.act_sequence)(h0, temps, keys)
            acts, frs, actmasks, frs_seq, syn_seq, out_active = magic_sequences
            magic_indices = jnp.argmax(out_active, axis=-1)
            masks = actmasks[jnp.arange(len(actmasks)), magic_indices]
            h_final = vmap(model.rtmoe.expert_net)(h0, masks)
            preds = vmap(model.output_block)(h_final)
            loss = softmax_cross_entropy_with_integer_labels(preds, y).mean()
            return (loss, (preds, magic_sequences, magic_indices))

        @eqx.filter_jit()
        def training_step(
            model: Model,
            x: Float[Array, "b ..."],
            y: Float[Array, "b ..."],
            opt_state: PyTree,
            temp: Float[Scalar, ""],
            *,
            key: PRNGKeyArray,
            **kwargs,
        ):
            vals, grads = compute_loss(model, x, y, temp, key)
            loss, (preds, magic_sequences, magic_idxs) = vals
            params = eqx.filter(model, eqx.is_inexact_array)
            updates, opt_state = optim.update(grads, opt_state, params)
            model = eqx.apply_updates(model, updates)
            return loss, model, opt_state, preds, magic_sequences, magic_idxs, grads

        @eqx.filter_jit
        def eval_step(
            model: Model,
            x: Float[Array, "b ..."],
            y: Float[Array, "b ..."],
            temp: Float[Scalar, ""],
            key: PRNGKeyArray,
        ):
            keys = jr.split(key, len(x))
            h0 = vmap(model.input_block)(x)
            temps = jnp.full(len(x), temp)
            magic_sequences = vmap(model.rtmoe.act_sequence)(h0, temps, keys)
            acts, frs, actmasks, frs_seq, syn_seq, out_active = magic_sequences
            magic_indices = jnp.argmax(out_active, axis=-1)
            masks = actmasks[jnp.arange(len(actmasks)), magic_indices]  # B, L, E
            h_final = vmap(model.rtmoe.expert_net)(h0, masks)
            preds = vmap(model.output_block)(h_final)
            return preds, acts, actmasks, syn_seq, magic_indices, frs

        global_step = 0

        # load data
        reshape = False if cfg.input_block == "conv" else True
        data = load_data(cfg.dset, cfg.bs, seed=cfg.seed, reshape=reshape)
        train_data, eval_data, test_data, ds_info = data
        original_shape = ds_info.features["image"].shape
        len_val_data = sum([len(x) for x, _ in eval_data])
        # instantiate models
        rng = jr.PRNGKey(cfg.seed)
        modkey, rng = jr.split(rng)
        model = Model(cfg, modkey)
        optim = optax.adamw(cfg.lr)
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
        tot_exps = cfg.n_exp_per_l * cfg.n_layers + 1

        # --------------------------------------------------------------------------------
        # TRAINING
        best_val_loss = jnp.inf
        patience_counter = 0
        num_batches = math.ceil(len(train_data))
        temps = jnp.linspace(cfg.temp, cfg.temp, cfg.epochs)  # TODO: future annealing?
        for epoch in tqdm(range(cfg.epochs)):
            # --------------------------------------------------------------------------------
            # TRAIN EPOCH
            train_data = train_data.shuffle(buffer_size=len(train_data), seed=cfg.seed)
            temp = temps[epoch]
            for b_idx, (xs, ys) in enumerate(train_data.as_numpy_iterator()):
                rng, thkey = jr.split(rng)
                # forward pass and unpack logging values
                outs = training_step(model, xs, ys, opt_state, temp, key=thkey)
                loss_val, model, opt_state, preds, magic_sequence, magic_indices, grads = outs
                acts, frs, actmasks, frs_seq, syn_seq, out_active = magic_sequence
                # logging
                accuracy = (jnn.softmax(preds).argmax(1) == ys).mean()
                magic_actmasks = actmasks[jnp.arange(len(xs)), magic_indices]
                active_experts = ein.reduce(magic_actmasks, "b l e -> b", "sum")
                mean_act_per_num_experts = ein.reduce(acts, "b le -> le", "mean")
                act_per_num_experts_line = {
                    f"train/act_per_num_exp/{n + 1}": mean_act_per_num_experts[n].item()
                    for n in range(tot_exps)
                }
                global_step = epoch * num_batches + b_idx
                wandb.log(
                    {
                        "train/epoch": epoch,
                        "train/step": global_step,
                        "train/loss": loss_val.item(),
                        "train/log_loss": jnp.log(loss_val).item(),
                        "train/accuracy": accuracy.item(),
                        "train/temp": temp.item(),
                        "train/active_experts/avg": active_experts.mean().item(),
                        "train/active_experts/min": active_experts.min().item(),
                        "train/active_experts/max": active_experts.max().item(),
                        "train/active_experts/1sp": active_experts.mean().item()
                        + active_experts.std().item(),
                        "train/active_experts/1sm": active_experts.mean().item()
                        - active_experts.std().item(),
                        **act_per_num_experts_line,
                    },
                    commit=False,
                    step=global_step,
                )
                # only log stuff once in a while
                if global_step % cfg.log_frequency == 0:
                    wandb.log({}, commit=True, step=global_step)
            # --------------------------------------------------------------------------------
            # VALIDATION (and early stopping stuff)
            if epoch > 0 and cfg.eval_every > 0 and epoch % cfg.eval_every == 0:
                len_test_data = sum([len(x) for x, _ in test_data])
                eval_accs = np.zeros((len_val_data,), dtype=float)
                eval_losses = np.zeros((len_val_data,), dtype=float)
                for b_idx, (xs, ys) in enumerate(eval_data.as_numpy_iterator()):
                    rng, thkey = jr.split(rng)
                    slc = slice(b_idx * cfg.bs, (b_idx + 1) * cfg.bs)
                    eval_stuff = eval_step(model, xs, ys, temp, thkey)
                    preds, acts, actmasks, syn_seq, magic_indices, frs = eval_stuff
                    eval_accs[slc] = jnn.softmax(preds).argmax(1) == ys
                    eval_losses[slc] = softmax_cross_entropy_with_integer_labels(
                        preds, ys
                    )
                val_accuracy = jnp.mean(eval_accs)
                val_loss = jnp.mean(eval_losses)
                wandb.log(
                    {
                        "val/loss": val_loss.item(),
                        "val/accuracy": val_accuracy.item(),
                        "val/log_loss": jnp.log(val_loss).item(),
                    },
                    step=global_step + 1,
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # save the model
                    save_model(
                        f"checkpoints/best_model_{wandb.run.id}.eqx",
                        wandb.config,
                        model,
                    )
                else:
                    patience_counter += 1
                    if patience_counter > cfg.patience:
                        print("Early stopping")
                        model, _ = load_model(
                            f"checkpoints/best_model_{wandb.run.id}.eqx", Model
                        )
                        break
        # one last log hit at the end of the training
        wandb.log({}, commit=True)

        # --------------------------------------------------------------------------------
        # EVAL CODE
        len_test_data = sum([len(x) for x, _ in test_data])
        test_preds = np.zeros((len_test_data, cfg.odim), dtype=float)
        magic_idxs = np.zeros(len_test_data, dtype=int)
        acts_sequence = np.zeros((len_test_data, tot_exps), dtype=float)
        active_experts = np.zeros((len_test_data, tot_exps - 1), dtype=bool)
        bitmaps = np.zeros(
            (len_test_data, tot_exps, cfg.n_layers, cfg.n_exp_per_l), dtype=bool
        )
        all_preds = np.zeros((len_test_data, tot_exps, cfg.odim), dtype=float)
        experts_frs = np.zeros(
            (len_test_data, tot_exps, cfg.n_layers, cfg.n_exp_per_l), dtype=float
        )
        for b_idx, (xs, ys) in enumerate(test_data.as_numpy_iterator()):
            rng, thkey = jr.split(rng)
            slc = slice(b_idx * cfg.bs, (b_idx + 1) * cfg.bs)
            eval_stuff = eval_step(model, xs, ys, temp, thkey)
            preds, acts, actmasks, syn_seq, magic_indices, frs = eval_stuff
            test_preds[slc] = preds
            magic_idxs[slc] = magic_indices
            active_experts[slc] = actmasks[jnp.arange(len(xs)), magic_indices].reshape(
                len(xs), -1
            )
            acts_sequence[slc] = acts
            bitmaps[slc] = actmasks
            #
            h0 = vmap(model.input_block)(xs)
            exps_out = vmap(vmap(model.rtmoe.expert_net, (None, 0)))(h0, actmasks)
            all_preds[slc] = vmap(vmap(model.output_block))(exps_out)
            experts_frs[slc] = frs
        # Logging everything
        test_labs = jnp.concatenate([ys for _, ys in test_data.as_numpy_iterator()])
        expert_names = [
            f"l{l}, e{e}" for l in range(cfg.n_layers) for e in range(cfg.n_exp_per_l)
        ]
        preds = jnn.softmax(test_preds).argmax(axis=1)
        test_accuracy = jnp.mean(preds == test_labs)
        overlaps_min = expert_overlap_min(active_experts)
        overlaps_max = expert_overlap_max(active_experts)
        actperexp = ein.reduce(active_experts, "s e -> e", "sum")
        act_counts = ein.reduce(active_experts, "s e -> s", "sum")

        # Store heavy data that can not realistically be logged onto wandb
        run_dir = Path(f"results/{run.name}")
        data_dir = run_dir / "data"
        plot_dir = run_dir / "plot"
        data_dir.mkdir(parents=True, exist_ok=True)
        plot_dir.mkdir(parents=True, exist_ok=True)
        jnp.save(data_dir / "all_preds", all_preds)
        jnp.save(data_dir / "acts_seqs", acts_sequence)
        jnp.save(data_dir / "experts_frs", experts_frs)
        jnp.save(data_dir / "labels", test_labs)
        jnp.save(data_dir / "magic_idxs", magic_idxs)
        jnp.save(data_dir / "bitmaps", bitmaps)
        if cfg.plot:
            plot_bitmaps(cfg, data_dir, plot_dir)
            plot_curves(cfg, data_dir, plot_dir)
        # Log metrics
        wandb.summary["total_param_count"] = get_parameter_count(model)
        wandb.summary["expert_param_count"] = get_parameter_count(model.rtmoe.expert_params) / (tot_exps - 1)
        wandb.summary["test accuracy"] = test_accuracy.item()
        run.log(
            {
                "test/expert_activation": wandb.plot.bar(
                    wandb.Table(
                        data=[[n, a] for n, a in zip(expert_names, actperexp)],
                        columns=["expert", "activation"],
                    ),
                    "Expert",
                    "Activation",
                    title="Expert activation",
                ),
                "test/overlaps_min": wandb.plot.bar(
                    make_wandb_histlike(overlaps_min, bins=20, range=(0, 1)),
                    "bin",
                    "count",
                    title="Overlap (min)",
                ),
                "test/overlaps_max": wandb.plot.bar(
                    make_wandb_histlike(overlaps_max, bins=20, range=(0, 1)),
                    "bin",
                    "count",
                    title="Overlap (max)",
                ),
                "test/act_counts": wandb.plot.bar(
                    make_wandb_histlike(act_counts, bins=np.arange(tot_exps) + 0.5),
                    "bin",
                    "count",
                    title="Num. active",
                ),
            }
        )

        test_data_array = jnp.concatenate(
            [x for x, _ in test_data.as_numpy_iterator()], axis=0
        )
        test_labs_array = jnp.concatenate(
            [y for _, y in test_data.as_numpy_iterator()], axis=0
        )
        test_preds_array = preds
        if len(test_data_array.shape) == 2:  # if input = fc, unflatten the images
            idx = [2, 0, 1]
            test_data_array = test_data_array.reshape(
                (-1, *tuple(original_shape[i] for i in idx))
            )
            print(test_data_array.shape)
        test_data_array = jnp.transpose(test_data_array, (0, 2, 3, 1))
        img_log_table = img_table(
            act_counts, test_data_array, test_labs_array, test_preds_array
        )
        run.log({"test/difficulty_analysis": img_log_table})


def main(config=None):
    print(config is None)
    if config is None:
        args = parse_args()
        config = load_config(args)
    config["idim"] = (
        28 * 28 if config["dset"] in ("mnist", "fashion_mnist") else 32 * 32 * 3
    )
    config["odim"] = (
        10 if config["dset"] in ("mnist", "fashion_mnist", "cifar10") else 100
    )

    train(config)


if __name__ == "__main__":

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    def parse_args():
        parser = argparse.ArgumentParser(description="Train a model.")
        parser.add_argument(
            "--config",
            type=str,
            default="configs/default_run.yaml",
            help="Path to the config file.",
        )
        parser.add_argument(
            "--seed",
            type=int,
            help="Random seed for training.",
        )
        parser.add_argument(
            "--dset",
            type=str,
            help="Dataset to use for training.",
        )
        parser.add_argument(
            "--bs",
            type=int,
            help="Batch size for training.",
        )
        parser.add_argument(
            "--epochs",
            type=int,
            help="Number of epochs to train for.",
        )
        parser.add_argument(
            "--lr",
            type=float,
            help="Learning rate for training.",
        )
        parser.add_argument(
            "--n_exp_per_l",
            type=int,
            help="Number of experts per layer.",
        )
        parser.add_argument("--n_layers", type=int, help="Number of layers.")
        parser.add_argument(
            "--input_block",
            type=str,
            help="Type of input block to use.",
        )
        parser.add_argument(
            "--num_gates",
            type=int,
            help="number of initial gates"
        )
        parser.add_argument(
            "--temp",
            type=float,
            help="Gumbel-softmax temperature"
        )
        parser.add_argument(
            "--hdim",
            type=int,
            help="Hidden dimension for the model.",
        )
        parser.add_argument(
            "--log_frequency",
            type=int,
            help="Frequency of logging.",
        )
        parser.add_argument(
            "--patience",
            type=int,
            help="Patience for early stopping.",
        )
        parser.add_argument(
            "--eval_every",
            type=int,
            help="Evaluate every n epochs.",
        )
        parser.add_argument(
            "--res",
            type=str2bool,
            nargs="?",
            help="Enable residual connections.",
        )
        parser.add_argument(
            "--plot",
            type=str2bool,
            nargs="?",
            help="Enable plotting of bitmaps and accuracy curves."
        )
        return parser.parse_args()

    def load_config(args):
        config = {}
        # 1) Load from YAML if provided
        if args.config:
            with open(args.config) as f:
                config = yaml.safe_load(f)

        # 2) Override with CLI args if they are not None
        for key, value in vars(args).items():
            if key == "config":
                continue
            if value is not None:
                config[key] = value
            config["cfg_file"] = Path(args.config).stem
        return config

    main()
