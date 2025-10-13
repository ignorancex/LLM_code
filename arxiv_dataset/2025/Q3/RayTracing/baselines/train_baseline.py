import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["WANDB_MODE"] = "disabled"
import jax
#jax.config.update("jax_disable_jit", True)
#jax.config.update("jax_debug_nans", True)
import math
import argparse
import functools as fts
from itertools import product
from pathlib import Path
import wandb
from tqdm import tqdm
import numpy as np
import yaml
from jax import numpy as jnp, random as jr, nn as jnn, vmap
import equinox as eqx
from jaxtyping import Float, PRNGKeyArray, PyTree, Scalar, Array
from typing import Any, Sequence
import optax
from optax.losses import softmax_cross_entropy_with_integer_labels
import einops as ein
import itertools as its
#from utils import load_data, get_parameter_count, save_model, load_model, create_input_block, create_output_block
from utils.backbones import create_input_block, create_output_block
from utils.data import load_data
from utils.misc import load_model, save_model
from utils.metrics import get_parameter_count
from baselines.baseline_moe import ThresholdMoeLayer,TopKMoeLayer, MLP, MoE, get_experts, get_router, get_deepseek_bias, get_deepseek_bias_threshold
import dotenv

dotenv.load_dotenv()
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
if WANDB_API_KEY is None:
    raise ValueError(
        "WANDB_API_KEY not found. Please set it in your environment variables."
    )

wandb.login(key=WANDB_API_KEY)

PROJECT_KWARGS = dict(
    entity="silvretta",
    project="Baseline_MoE",
    group="mnist",
    settings=wandb.Settings(code_dir="."),
)


def get_model(cfg):
    key = jr.PRNGKey(cfg.seed)
    # common input and output blocks
    ikey, okey, key = jr.split(key, 3)
    input_block = create_input_block(cfg, ikey)
    output_block = eqx.nn.Linear(
        cfg.hdim,
        cfg.odim,
        key=okey,
    )

    if cfg.model == "topk":
        rkey, key = jr.split(key)
        layers=[]
        for layer in range(cfg.num_layers_model):
            experts = []
            keys = jr.split(key, cfg.n_exps + 2)
            ekeys, rkey, key = keys[:-2], keys[-2], keys[-1]
            router = eqx.nn.Linear(
                cfg.hdim,
                cfg.n_exps,
                key=rkey,
            )
            for i in range(cfg.n_exps):
                experts.append(
                    eqx.nn.MLP(
                        cfg.hdim,
                        cfg.hdim,
                        cfg.hdim,
                        depth=cfg.num_layers_expert,
                        key=ekeys[i],
                    )
                )
        
            layers.append(
                TopKMoeLayer(
                    router=router,
                    experts=experts,
                    top_k=cfg.top_k,
                )
            )
        model = MoE(
            input_block=input_block,
            layers=layers,
            output_block=output_block,
        )
    
    elif cfg.model == 'threshold':
        rkey, key = jr.split(key)
        layers=[]
        for layer in range(cfg.num_layers_model):
            experts = []
            keys = jr.split(key, cfg.n_exps + 2)
            ekeys, rkey, key = keys[:-2], keys[-2], keys[-1]
            router = eqx.nn.Linear(
                cfg.hdim,
                cfg.n_exps,
                key=rkey,
            )
            for i in range(cfg.n_exps):
                experts.append(
                    eqx.nn.MLP(
                        cfg.hdim,
                        cfg.hdim,
                        cfg.hdim,
                        depth=cfg.num_layers_expert,
                        key=ekeys[i],
                    )
                )
            layers.append(
                ThresholdMoeLayer(
                    router,
                    experts,
                    threshold=cfg.threshold
                )
            )
        model = MoE(
            input_block=input_block,
            layers=layers,
            output_block=output_block,
        )
        

    elif cfg.model == "mlp":
        mlp = eqx.nn.MLP(
            cfg.hdim,
            cfg.hdim,
            cfg.hdim,
            cfg.num_layers_model, # one for the output block
            key=key,
        )
        # add dummy output beacuse in the moe case we have a mask returned by the model
        model = MLP(
            input_block,
            mlp,
            output_block
        )
    return model


# load data


def train(conf=None):
    run = wandb.run or wandb.init(**PROJECT_KWARGS, config=conf)
    with run:
        @fts.partial(eqx.filter_value_and_grad, has_aux=True)
        def loss_fn(model, x, y, biases):
            outs = vmap(model, in_axes=(0, None))(x, biases)
            logits, mask, _ = outs
            loss = softmax_cross_entropy_with_integer_labels(logits, y)
            return loss.mean(), (logits, mask)

        def accuracy_fn(logits, y):
            pred = logits.argmax(axis=-1)
            acc = (pred == y).mean()
            return acc

        @eqx.filter_jit
        def train_step(
            model: eqx.Module,
            x: Float[Array, "batch_size ..."],
            y: Float[Array, "batch_size"],
            optim: optax.GradientTransformation,
            biases: Sequence[Float[Array, "n_exps_l"]],
            opt_state: PyTree,
        ):
            (loss, (logits, mask)), grads = loss_fn(model, x, y, biases)
            updates, opt_state = optim.update(grads, opt_state, params= eqx.filter(model, eqx.is_inexact_array))
            model = eqx.apply_updates(model, updates)
            acc = accuracy_fn(logits, y)
            utilization = jax.tree.map(lambda m: np.mean(m > 0, axis=0), mask)
            if cfg.model == 'topk':
                biases = get_deepseek_bias(utilization, biases, cfg.top_k, cfg.gamma)
            elif cfg.model == 'threshold':
                biases = get_deepseek_bias_threshold(utilization, biases, cfg.threshold, cfg.gamma)
            return model, opt_state, biases, loss, acc

        @eqx.filter_jit
        def eval_step(
            model: eqx.Module,
            x: Float[Array, "batch_size ..."],
            y: Float[Array, "batch_size"],
            biases
        ):
            outs = vmap(model, in_axes=(0, None))(x, biases)
            logits, mask, _ = outs
            acc = accuracy_fn(logits, y)
            return acc, logits, mask


        global_step = 0
        cfg = wandb.config
        key = jr.PRNGKey(cfg.seed)

        # Load data
        train_data, eval_data, test_data, ds_info = load_data(cfg.dset, 
                                                              cfg.bs, 
                                                              cfg.seed, 
                                                              reshape=True if cfg.input_block == 'fc' else False)


        # Initialize model
        model = get_model(cfg)
        # Get model info
        params_count = get_parameter_count(model)
        wandb.log(
            {
                "model_info/param_count": params_count,
                "model_info/model": str(model),
            }
        )

        # Initialize optimizer
        optim = optax.adamw(
            learning_rate=cfg.lr,
            weight_decay=cfg.weight_decay
        )

        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

        # TRAIN CODE


        best_val_loss = float("inf")
        patience_counter = 0
        biases = [jnp.zeros((cfg.n_exps)) for _ in range(cfg.num_layers_model)]
        zero_biases = biases.copy()
        for epoch in tqdm(range(cfg.epochs)):
            train_data = train_data.shuffle(buffer_size=len(train_data))
            key, pkey = jr.split(key)
            for b_idx, (xs, ys) in enumerate(train_data.as_numpy_iterator()):
                slc = slice(b_idx * cfg.bs, (b_idx + 1) * cfg.bs)
                model, opt_state, biases, loss, acc = train_step(
                    model,
                    xs,
                    ys,
                    optim,
                    biases,
                    opt_state,
                )
                global_step += 1
                if cfg.model == 'topk':
                    biases_dict = {
                        f"biases/bias_{n}, {i}": biases[n][i].item()
                        for (n, i) in product(range(cfg.num_layers_model), range(len(biases[0])))
                    }
                else:
                    biases_dict={}
                wandb.log(
                    {
                        "train/epoch": epoch,
                        "train/loss": loss,
                        "train/acc": acc,
                        "global_step": global_step,
                        **biases_dict
                    }
                )
            # validation
            if epoch  > 0 and cfg.eval_every > 0 and epoch % cfg.eval_every == 0:
                len_val_data = sum([len(x) for x, _ in eval_data])
                eval_accs = np.zeros((len_val_data,), dtype=float)
                eval_losses = np.zeros((len_val_data,), dtype=float)
                for b_idx, (xs, ys) in enumerate(eval_data.as_numpy_iterator()):
                    slc = slice(b_idx * cfg.bs, (b_idx + 1) * cfg.bs)
                    acc, logits, mask = eval_step(model, xs, ys, biases)
                    eval_accs[slc] = jnn.softmax(logits).argmax(1) == ys
                    eval_losses[slc] = softmax_cross_entropy_with_integer_labels(
                        logits, ys
                    )
                val_acc = eval_accs.mean()
                val_loss = eval_losses.mean()
                wandb.log(
                    {
                        "val/epoch": epoch,
                        "val/acc": val_acc,
                        "val/loss": val_loss,
                        "val/log_loss":jnp.log(val_loss)
                    },
                    step=global_step+1,
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
                            f"checkpoints/best_model_{wandb.run.id}.eqx", get_model
                        )
                        break
                    

        # Evaluate
        len_test_data = sum([len(x) for x, _ in test_data])
        masks = np.zeros((len_test_data, cfg.n_exps * cfg.num_layers_model), dtype=jnp.float32)
        eval_accs = np.zeros((len_test_data,), dtype=float)
        eval_losses = np.zeros((len_test_data,), dtype=float)
        for b_idx, (xs, ys) in enumerate(test_data.as_numpy_iterator()):
            slc = slice(b_idx * cfg.bs, (b_idx + 1) * cfg.bs)
            acc, logits, mask = eval_step(model, xs, ys, biases)
            eval_accs[slc] = jnn.softmax(logits).argmax(1) == ys
            eval_losses[slc] = softmax_cross_entropy_with_integer_labels(
                logits, ys
            )
            if cfg.model == "topk" or cfg.model == "threshold":
                masks[slc] = jnp.concatenate(mask, axis=-1)
        test_acc = eval_accs.mean()
        test_loss = eval_losses.mean()
        wandb.log(
            {
                "test/epoch": epoch,
                "test/acc": test_acc,
                "test/loss": test_loss,
            },
        )
                # get n. of active experts
        if cfg.model == "topk" or cfg.model == "threshold":
            active_experts_percent = np.sum(masks > 0, axis=0) / len_test_data
            # bar plot
            active_table = wandb.Table(
                data=[[i, a] for i, a in enumerate(active_experts_percent)],
                columns=["expert", "activation"],
            )
            wandb.log(
                {
                    "test/active_experts": wandb.plot.bar(
                        active_table,
                        "expert",
                        "activation",
                        title="Active Experts",
                    )
                }
            )
        

def main(config=None):
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

    def parse_args():
        parser = argparse.ArgumentParser(description="Train a model.")
        parser.add_argument(
            "--config",
            type=str,
            default="configs/run_topk.yaml",
            help="Path to the config file.",
        )
        parser.add_argument(
            "--model",
            type=str,
            help="Model to use for training.",
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
            "--n_exps",
            type=int,
            help="Number of experts in each layer.",
        )
        parser.add_argument(
            "--top_k",
            type=int,
            help="Number of experts to use in the top-k selection.",
        )
        parser.add_argument(
            "--threshold",
            type=float,
            help="Threshold for threshod MoE."
        )
        parser.add_argument(
            "--num_layers_model",
            type=int,
            help="Number of layers in the model.",
        )
        parser.add_argument(
            "--num_layers_expert",
            type=int,
            help="Number of layers for each MLP expert.",
        )
        parser.add_argument(
            "--input_block",
            type=str,
            help="Type of input block to use.",
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
            "--gamma",
            type=float,
            help="step-size for deepseek bias update"
        )
        return parser.parse_args()

    def load_config(args):
        config = {}
        # 1) Load from YAML if provided
        if args.config:
            with open(args.config) as f:
                config = yaml.safe_load(f)
            config["cfg_file"] = Path(args.config).stem

        # 2) Override with CLI args if they are not None
        for key, value in vars(args).items():
            if key == "config":
                continue
            if value is not None:
                config[key] = value

        return config

    main()
