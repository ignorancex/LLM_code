"""
Example usages:

python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/upbeat-wildflower-739--checkpoint:v9 \
    -jc json/arc-agi_evaluation_challenges.json \
    -js json/arc-agi_evaluation_solutions.json \
    -i mean \
    && \
python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/upbeat-wildflower-739--checkpoint:v9 \
    -jc json/arc-agi_evaluation_challenges.json \
    -js json/arc-agi_evaluation_solutions.json \
    -i gradient_ascent \
    --num-steps 20 \
    --lr 5e-2 \
    && \
python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/playful-monkey-758--checkpoint:v1 \
    -jc json/arc-agi_evaluation_challenges.json \
    -js json/arc-agi_evaluation_solutions.json \
    -i random_search \
    --num-samples 100 \
    --scale 1.0 \
    --scan-batch-size 10 \
    --random-search-seed 0

    
python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/solar-salad-1050--checkpoint:v18 \
    -jc json/arc-agi_training_challenges.json \
    -js json/arc-agi_training_solutions.json \
    -i gradient_ascent \
    --num-steps 200 \
    --lr 1.0 \
    --lr-schedule true \
    --optimizer adam \
    --optimizer-kwargs '{"b2": 0.9}' \
    && \
python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/solar-salad-1050--checkpoint:v18 \
    -jc json/arc-agi_evaluation_challenges.json \
    -js json/arc-agi_evaluation_solutions.json \
    -i gradient_ascent \
    --num-steps 200 \
    --lr 1.0 \
    --lr-schedule true \
    --optimizer adam \
    --optimizer-kwargs '{"b2": 0.9}'





python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/playful-sun-1060--checkpoint:v1 \
    -jc json/arc-agi_evaluation_challenges.json \
    -js json/arc-agi_evaluation_solutions.json \
    -i gradient_ascent \
    --num-steps 200 \
    --lr 0.3 \
    --lr-schedule true \
    --optimizer adam \
    --optimizer-kwargs '{"b2": 0.9}' \
    --accumulate-gradients-decoder-pairs true \
    --random-perturbation '{"num_samples": 5, "scale": 0.1}' \
    --include-all-latents true
    

    

python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/ominous-monster-839--checkpoint:v2 \
    -jc json/arc-agi_training_challenges.json \
    -js json/arc-agi_training_solutions.json \
    --only-n-tasks 10 \
    -i gradient_ascent \
    --num-steps 10 \
    --lr 0.1 \
    --random-perturbation '{"num_samples": 5, "scale": 0.1}' \
    && \
python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/fanciful-pyramid-761--checkpoint:v5 \
    -jc json/arc-agi_evaluation_challenges.json \
    -js json/arc-agi_evaluation_solutions.json \
    -i gradient_ascent \
    --num-steps 125 \
    --lr 0.5 \
    && \
python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/fanciful-pyramid-761--checkpoint:v5 \
    -jc json/arc-agi_training_challenges.json \
    -js json/arc-agi_training_solutions.json \
    -i gradient_ascent \
    --num-steps 125 \
    --lr 1.0 \
    && \
python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/fanciful-pyramid-761--checkpoint:v5 \
    -jc json/arc-agi_evaluation_challenges.json \
    -js json/arc-agi_evaluation_solutions.json \
    -i gradient_ascent \
    --num-steps 125 \
    --lr 1.0 \
    && \
python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/fanciful-pyramid-761--checkpoint:v5 \
    -jc json/arc-agi_training_challenges.json \
    -js json/arc-agi_training_solutions.json \
    -i gradient_ascent \
    --num-steps 125 \
    --lr 5.0 \
    && \
python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/fanciful-pyramid-761--checkpoint:v5 \
    -jc json/arc-agi_evaluation_challenges.json \
    -js json/arc-agi_evaluation_solutions.json \
    -i gradient_ascent \
    --num-steps 125 \
    --lr 5.0 \
    && \
python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/fanciful-pyramid-761--checkpoint:v5 \
    -jc json/arc-agi_training_challenges.json \
    -js json/arc-agi_training_solutions.json \
    -i gradient_ascent \
    --num-steps 125 \
    --lr 10.0 \
    && \
python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/fanciful-pyramid-761--checkpoint:v5 \
    -jc json/arc-agi_evaluation_challenges.json \
    -js json/arc-agi_evaluation_solutions.json \
    -i gradient_ascent \
    --num-steps 125 \
    --lr 10.0 \
    && \
python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/fanciful-pyramid-761--checkpoint:v5 \
    -jc json/arc-agi_training_challenges.json \
    -js json/arc-agi_training_solutions.json \
    -i gradient_ascent \
    --num-steps 125 \
    --lr 50.0 \
    && \
python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/fanciful-pyramid-761--checkpoint:v5 \
    -jc json/arc-agi_evaluation_challenges.json \
    -js json/arc-agi_evaluation_solutions.json \
    -i gradient_ascent \
    --num-steps 125 \
    --lr 50.0    

    
# Evaluate on the ARC json datasets (only -w, -jc, and -js are required):
## Random Search
>> python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/faithful-dawn-316--checkpoint:v76 \
    -jc json/arc-agi_training_challenges.json \
    -js json/arc-agi_training_solutions.json \
    --only-n-tasks 1 \
    -i random_search \
    --num-samples 100 \
    --scale 1.0 \
    --scan-batch-size 10 \
    --random-search-seed 0
## Gradient Ascent
>> python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/faithful-dawn-316--checkpoint:v76 \
    -jc json/arc-agi_training_challenges.json \
    -js json/arc-agi_training_solutions.json \
    --only-n-tasks 1 \
    -i gradient_ascent \
    --num-steps 5 \
    --lr 5e-2

# Evaluate on a custom dataset (only -w and -d are required):
## Random Search
>> python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/faithful-dawn-316--checkpoint:v76 \
    -d storage/v0_main_fix_test \
    --dataset-length 32 \
    --dataset-batch-size 8 \
    --dataset-seed 0 \
    -i random_search \
    --num-samples 100 \
    --scale 1.0 \
    --scan-batch-size 10 \
    --random-search-seed 0
## Gradient Ascent
>> python src/evaluate_checkpoint.py \
    -w TheThinker/ARC/faithful-dawn-316--checkpoint:v76 \
    -d storage/v0_main_fix_test \
    --dataset-length 32 \
    --dataset-batch-size 8 \
    --dataset-seed 0 \
    -i gradient_ascent \
    --num-steps 5 \
    --lr 5e-2
"""

import argparse
import os
from typing import Optional

import chex
import wandb
import hydra
import omegaconf
import jax
from jax.tree_util import tree_map
import jax.numpy as jnp
import json
import optax
from tqdm import trange
from flax.training.train_state import TrainState
from flax.serialization import from_bytes

from src.models.lpn import LPN
from src.evaluator import Evaluator
from src.models.transformer import EncoderTransformer, DecoderTransformer
from src.train import Trainer, load_datasets, instantiate_config_for_mpt
from src.data_utils import make_leave_one_out, DATASETS_BASE_PATH


def instantiate_model(cfg: omegaconf.DictConfig, mixed_precision: bool) -> LPN:
    if mixed_precision:
        encoder = EncoderTransformer(instantiate_config_for_mpt(cfg.encoder_transformer))
        decoder = DecoderTransformer(instantiate_config_for_mpt(cfg.decoder_transformer))
    else:
        encoder = EncoderTransformer(hydra.utils.instantiate(cfg.encoder_transformer))
        decoder = DecoderTransformer(hydra.utils.instantiate(cfg.decoder_transformer))
    lpn = LPN(encoder=encoder, decoder=decoder)
    return lpn


def instantiate_train_state(lpn: LPN) -> TrainState:
    key = jax.random.PRNGKey(0)
    decoder = lpn.decoder
    grids = jax.random.randint(
        key,
        (1, 3, decoder.config.max_rows, decoder.config.max_cols, 2),
        minval=0,
        maxval=decoder.config.vocab_size,
    )
    shapes = jax.random.randint(
        key,
        (1, 3, 2, 2),
        minval=1,
        maxval=min(decoder.config.max_rows, decoder.config.max_cols) + 1,
    )
    variables = lpn.init(
        key, grids, shapes, dropout_eval=False, prior_kl_coeff=0.0, pairwise_kl_coeff=0.0, mode="mean"
    )

    learning_rate, linear_warmup_steps = 0, 0
    linear_warmup_scheduler = optax.warmup_exponential_decay_schedule(
        init_value=learning_rate / (linear_warmup_steps + 1),
        peak_value=learning_rate,
        warmup_steps=linear_warmup_steps,
        transition_steps=1,
        end_value=learning_rate,
        decay_rate=1.0,
    )
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(linear_warmup_scheduler))
    optimizer = optax.MultiSteps(optimizer, every_k_schedule=1)
    train_state = TrainState.create(apply_fn=lpn.apply, tx=optimizer, params=variables["params"])
    return train_state


def load_model_weights(
    train_state: TrainState, artifact_dir: str, ckpt_name: str = "state.msgpack"
) -> TrainState:
    with open(os.path.join(artifact_dir, ckpt_name), "rb") as data_file:
        byte_data = data_file.read()
    loaded_state = from_bytes(train_state, byte_data)
    return loaded_state


def build_generate_output_batch_to_be_pmapped(
    model: LPN, eval_inference_mode: str, eval_inference_mode_kwargs: dict
) -> callable:
    def generate_output_batch_to_be_pmapped(
        params, leave_one_out_grids, leave_one_out_shapes, dataset_grids, dataset_shapes, keys
    ) -> dict[str, chex.Array]:
        grids_inputs, labels_grids_outputs = dataset_grids[..., 0], dataset_grids[..., 1]
        shapes_inputs, labels_shapes_outputs = dataset_shapes[..., 0], dataset_shapes[..., 1]
        generated_grids_outputs, generated_shapes_outputs, _ = model.apply(
            {"params": params},
            leave_one_out_grids,
            leave_one_out_shapes,
            grids_inputs,
            shapes_inputs,
            keys,
            dropout_eval=True,
            mode=eval_inference_mode,
            **eval_inference_mode_kwargs,
            method=model.generate_output,
        )

        correct_shapes = jnp.all(generated_shapes_outputs == labels_shapes_outputs, axis=-1)
        batch_ndims = len(grids_inputs.shape[:-2])

        row_arange_broadcast = jnp.arange(grids_inputs.shape[-2]).reshape(
            (*batch_ndims * (1,), grids_inputs.shape[-2])
        )
        input_row_mask = row_arange_broadcast < labels_shapes_outputs[..., :1]
        col_arange_broadcast = jnp.arange(grids_inputs.shape[-1]).reshape(
            (*batch_ndims * (1,), grids_inputs.shape[-1])
        )
        input_col_mask = col_arange_broadcast < labels_shapes_outputs[..., 1:]
        input_mask = input_row_mask[..., None] & input_col_mask[..., None, :]

        pixels_equal = jnp.where(
            input_mask & correct_shapes[..., None, None],
            (generated_grids_outputs == labels_grids_outputs),
            False,
        )
        pixel_correctness = pixels_equal.sum(axis=(-1, -2)) / labels_shapes_outputs.prod(axis=-1)
        accuracy = pixels_equal.sum(axis=(-1, -2)) == labels_shapes_outputs.prod(axis=-1)

        metrics = {
            "correct_shapes": jnp.mean(correct_shapes),
            "pixel_correctness": jnp.mean(pixel_correctness),
            "accuracy": jnp.mean(accuracy),
        }
        return metrics

    return generate_output_batch_to_be_pmapped


def evaluate_json(
    train_state: TrainState,
    evaluator: Evaluator,
    json_challenges_file: str,
    json_solutions_file: str,
    only_n_tasks: Optional[int],
    random_search_seed: int,
) -> dict:
    print(f"Evaluating the model on {json_challenges_file.rstrip().split('/')[-1]}...")
    metrics, fig = Trainer.test_json_submission(
        train_state,
        evaluator,
        json_challenges_file=os.path.join(DATASETS_BASE_PATH, json_challenges_file),
        json_solutions_file=os.path.join(DATASETS_BASE_PATH, json_solutions_file),
        test_name="",
        key=jax.random.PRNGKey(random_search_seed),
        only_n_tasks=only_n_tasks,  # 'None' to run on all tasks
        progress_bar=True,
        num_tasks_to_show=0,
    )
    metrics = {k.split("/")[-1]: v for k, v in metrics.items()}
    metrics["fig"] = fig
    return metrics


def evaluate_custom_dataset(
    train_state: TrainState,
    evaluator: Evaluator,
    dataset_folder: str,
    dataset_length: Optional[int],
    dataset_batch_size: int,
    dataset_use_hf: bool,
    dataset_seed: int,
    random_search_seed: int,
) -> dict:
    print(f"Evaluating the model on the {dataset_folder.rstrip().split('/')[-1]} dataset...")

    # Load data
    grids, shapes, _ = load_datasets([dataset_folder], use_hf=dataset_use_hf)[0]
    if dataset_length is not None:
        key = jax.random.PRNGKey(dataset_seed)
        indices = jax.random.permutation(key, len(grids))[:dataset_length]
        grids, shapes = grids[indices], shapes[indices]
    # Drop the last batch if it's smaller than the batch size
    num_batches = grids.shape[0] // dataset_batch_size
    grids, shapes = grids[: num_batches * dataset_batch_size], shapes[: num_batches * dataset_batch_size]

    leave_one_out_grids = make_leave_one_out(grids, axis=-4)
    leave_one_out_shapes = make_leave_one_out(shapes, axis=-3)

    num_devices = len(evaluator.devices)
    # Split the dataset onto devices.
    assert grids.shape[0] % num_devices == 0
    leave_one_out_grids, leave_one_out_shapes, grids, shapes = tree_map(
        lambda x: x.reshape((num_devices, x.shape[0] // num_devices, *x.shape[1:])),
        (leave_one_out_grids, leave_one_out_shapes, grids, shapes),
    )
    # Split the dataset into batches for each device.
    batch_size_per_device = dataset_batch_size // num_devices
    assert grids.shape[1] % batch_size_per_device == 0
    leave_one_out_grids, leave_one_out_shapes, grids, shapes = tree_map(
        lambda x: x.reshape(
            (x.shape[0], x.shape[1] // batch_size_per_device, batch_size_per_device, *x.shape[2:])
        ),
        (leave_one_out_grids, leave_one_out_shapes, grids, shapes),
    )
    keys = jax.random.split(
        jax.random.PRNGKey(random_search_seed), (num_devices, grids.shape[1])
    )  # (num_devices, num_batches)

    pmap_dataset_generate_output_batch = jax.pmap(
        build_generate_output_batch_to_be_pmapped(
            model=evaluator.model,
            eval_inference_mode=evaluator.inference_mode,
            eval_inference_mode_kwargs=evaluator.inference_mode_kwargs,
        ),
        axis_name="num_devices",
    )
    metrics_list = [
        pmap_dataset_generate_output_batch(
            train_state.params,
            leave_one_out_grids[:, i],
            leave_one_out_shapes[:, i],
            grids[:, i],
            shapes[:, i],
            keys[:, i],
        )
        for i in trange(grids.shape[1], desc="Generating solutions")
    ]
    # Aggregate the metrics over the devices and the batches.
    metrics = {k: jnp.stack([m[k] for m in metrics_list]).mean() for k in metrics_list[0].keys()}
    return metrics


def pretty_print(metrics: dict) -> None:
    print("Metrics:")
    for k, v in metrics.items():
        if isinstance(v, (jnp.ndarray, float, int)):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: not a scalar")


def main(
    artifact_path: str,
    json_challenges_file: Optional[str],
    json_solutions_file: Optional[str],
    only_n_tasks: Optional[int],
    dataset_folder: Optional[str],
    dataset_length: Optional[int],
    dataset_batch_size: Optional[int],
    dataset_use_hf: bool,
    dataset_seed: int,
    inference_mode: str,
    inference_mode_kwargs: dict,
    random_search_seed: int,
    mixed_precision: bool,
) -> None:
    # Make sure the wandb mode is enabled.
    os.environ["WANDB_MODE"] = "run"

    print("Downloading the model artifact...")
    # Download the artifact and save the config file
    run = wandb.init()
    artifact = run.use_artifact(artifact_path, type="model")
    run.finish()
    cfg = omegaconf.OmegaConf.create(artifact.metadata)
    artifact_dir = artifact.download()
    omegaconf.OmegaConf.save(config=cfg, f=os.path.join(artifact_dir, "config.yaml"))

    print("Instantiating the model and the train state...")
    lpn = instantiate_model(cfg, mixed_precision)
    train_state = instantiate_train_state(lpn)
    evaluator = Evaluator(
        lpn,
        inference_mode=inference_mode,
        inference_mode_kwargs=inference_mode_kwargs,
        devices=None,
    )

    # Load the model weights
    print("Loading the model weights...")
    train_state = load_model_weights(train_state, artifact_dir)

    # Put the train state on the device(s)
    train_state = jax.device_put_replicated(train_state, evaluator.devices)

    # Evaluate the model
    print(f"Inference mode: {evaluator.inference_mode}")
    kwargs = {k: v for k, v in evaluator.inference_mode_kwargs.items() if v is not None}
    if kwargs:
        print(f"Inference mode kwargs: {kwargs}")
    if json_challenges_file and json_solutions_file:
        metrics = evaluate_json(
            train_state,
            evaluator,
            json_challenges_file,
            json_solutions_file,
            only_n_tasks,
            random_search_seed,
        )
        pretty_print(metrics)
    if dataset_folder:
        metrics = evaluate_custom_dataset(
            train_state,
            evaluator,
            dataset_folder,
            dataset_length,
            dataset_batch_size,
            dataset_use_hf,
            dataset_seed,
            random_search_seed,
        )
        pretty_print(metrics)


def true_or_false_from_arg(arg: str) -> bool:
    if arg.lower() == "true":
        return True
    if arg.lower() == "false":
        return False
    raise ValueError(f"Invalid boolean argument '{arg}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a model checkpoint on either the ARC json datasets or custom datasets."
            "Must provide arguments for -w, and, either -jc and -js, or -d."
        )
    )
    parser.add_argument(
        "-w",
        "--wandb-artifact-path",
        type=str,
        required=True,
        help="WandB path to the desired artifact. E.g. 'TheThinker/ARC/faithful-dawn-316--checkpoint:v76'.",
    )
    parser.add_argument(
        "-jc",
        "--json-challenges-file",
        type=str,
        required=False,
        default=None,
        help="Path to the JSON file with the ARC challenges. E.g. 'json/arc-agi_training_challenges.json'.",
    )
    parser.add_argument(
        "-js",
        "--json-solutions-file",
        type=str,
        required=False,
        default=None,
        help="Path to the JSON file with the ARC solutions. E.g. 'json/arc-agi_training_solutions.json'.",
    )
    parser.add_argument(
        "--only-n-tasks",
        type=int,
        required=False,
        default=None,
        help="Number of tasks to evaluate the model on. 'None' to run on all tasks.",
    )
    parser.add_argument(
        "-d",
        "--dataset-folder",
        type=str,
        required=False,
        default=None,
        help="Path to the folder with the custom dataset. E.g. 'storage/v0_main_fix_test'.",
    )
    parser.add_argument(
        "--dataset-length",
        type=int,
        required=False,
        default=None,
        help="Number of examples to evaluate the model on. 'None' to run on all examples.",
    )
    parser.add_argument(
        "--dataset-batch-size",
        type=int,
        required=False,
        default=None,
        help="Batch size for the custom dataset evaluation. 'None' to use the length of the dataset.",
    )
    parser.add_argument(
        "--dataset-use-hf",
        type=true_or_false_from_arg,
        required=False,
        default=True,
        help="Whether to use Hugging Face to load the datasets (otherwise loads locally).",
    )
    parser.add_argument(
        "--dataset-seed",
        type=int,
        required=False,
        default=0,
        help="Seed to sample a subset of the custom dataset for evaluation.",
    )
    parser.add_argument(
        "-i",
        "--inference-mode",
        type=str,
        required=False,
        default="mean",
        help="Inference mode to use, choose from ['mean', 'first', 'random_search', 'gradient_ascent'].",
    )
    parser.add_argument(
        "--random-search-seed",
        type=int,
        required=False,
        default=0,
        help="Seed for the 'random_search' inference mode.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        required=False,
        default=None,
        help="Number of samples for the 'random_search' inference mode.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        required=False,
        default=None,
        help="Scale for the random noise added during the 'random_search' inference mode.",
    )
    parser.add_argument(
        "--scan-batch-size",
        type=int,
        required=False,
        default=None,
        help="Batch size for the 'random_search' inference mode.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        required=False,
        default=None,
        help="Number of steps for the 'gradient_ascent' inference mode.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        required=False,
        default=None,
        help="Learning rate for the 'gradient_ascent' inference mode.",
    )
    parser.add_argument(
        "--lr-schedule",
        type=true_or_false_from_arg,
        required=False,
        default=None,
        help="Whether to use a cosine decay learning rate schedule for the 'gradient_ascent' inference mode.",
    )
    parser.add_argument(
        "--lr-schedule-exponent",
        type=float,
        required=False,
        default=None,
        help="Exponent for the cosine decay learning rate schedule for the 'gradient_ascent' inference mode.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        required=False,
        default=None,
        help="Optimizer to use for the 'gradient_ascent' inference mode.",
    )
    parser.add_argument(
        "--optimizer-kwargs",
        type=json.loads,
        required=False,
        default=None,
        help="Optimizer kwargs for the 'gradient_ascent' inference mode.",
    )
    parser.add_argument(
        "--accumulate-gradients-decoder-pairs",
        type=true_or_false_from_arg,
        required=False,
        default=None,
        help="Whether to accumulate gradients for the decoder pairs in the 'gradient_ascent' inference mode.",
    )
    parser.add_argument(
        "--scan-gradients-latents",
        type=true_or_false_from_arg,
        required=False,
        default=None,
        help="Whether to scan gradients for the latents in the 'gradient_ascent' inference mode.",
    )
    parser.add_argument(
        "--include-mean-latent",
        type=true_or_false_from_arg,
        required=False,
        default=None,
        help="Whether to include the mean latent in the 'random_search' or 'gradient_ascent' inference mode.",
    )
    parser.add_argument(
        "--include-all-latents",
        type=true_or_false_from_arg,
        required=False,
        default=None,
        help="Whether to include all latents in the 'random_search' or 'gradient_ascent' inference mode.",
    )
    parser.add_argument(
        "--random-perturbation",
        type=json.loads,
        required=False,
        default=None,
        help="Random perturbation kwargs. Requires 'num_samples' and 'scale' keys.",
    )
    parser.add_argument(
        "--mixed-precision",
        type=true_or_false_from_arg,
        required=False,
        default=True,
        help="Whether to use mixed precision for inference.",
    )
    args = parser.parse_args()
    if (
        args.json_challenges_file is None
        and args.json_solutions_file is not None
        or args.json_challenges_file is not None
        and args.json_solutions_file is None
    ):
        parser.error("Must provide both the json challenges (-jc) and solutions (-js) files.")
    if args.json_challenges_file is None and args.dataset_folder is None:
        parser.error(
            "Must provide either the json challenges (-jc) and solutions (-js) files or the dataset folder (-d)."
        )
    if args.inference_mode not in ["mean", "first", "random_search", "gradient_ascent"]:
        parser.error(
            "Invalid inference mode. Choose from ['mean', 'first', 'random_search', 'gradient_ascent']."
        )
    if args.inference_mode == "random_search":
        if args.num_samples is None:
            parser.error("The 'random_search' inference mode requires the --num-samples argument.")
        if args.scale is None:
            parser.error("The 'random_search' inference mode requires the --scale argument.")
    if args.inference_mode == "gradient_ascent":
        if args.num_steps is None:
            parser.error("The 'gradient_ascent' inference mode requires the --num-steps argument.")
        if args.lr is None:
            parser.error("The 'gradient_ascent' inference mode requires the --lr argument.")
    inference_mode_kwargs = {
        "num_samples": args.num_samples,
        "scale": args.scale,
        "num_steps": args.num_steps,
        "lr": args.lr,
    }
    for arg in [
        "scan_batch_size",
        "include_mean_latent",
        "include_all_latents",
        "lr_schedule",
        "lr_schedule_exponent",
        "optimizer",
        "optimizer_kwargs",
        "scan_gradients_latents",
        "accumulate_gradients_decoder_pairs",
        "random_perturbation",
    ]:
        if getattr(args, arg) is not None:
            inference_mode_kwargs[arg] = getattr(args, arg)
    main(
        artifact_path=args.wandb_artifact_path,
        json_challenges_file=args.json_challenges_file,
        json_solutions_file=args.json_solutions_file,
        only_n_tasks=args.only_n_tasks,
        dataset_folder=args.dataset_folder,
        dataset_length=args.dataset_length,
        dataset_batch_size=args.dataset_batch_size,
        dataset_use_hf=args.dataset_use_hf,
        dataset_seed=args.dataset_seed,
        inference_mode=args.inference_mode,
        inference_mode_kwargs=inference_mode_kwargs,
        random_search_seed=args.random_search_seed,
        mixed_precision=args.mixed_precision,
    )
