"""
We only change the weights of the model, and use the original tokenizer and config.
The sft is conducted with higher version of transformers (in trl), so we need to convert the weights to the original format.

python train/sft/convert_weights.py \
    --source_model Qwen/Qwen2.5-VL-3B-Instruct \
    --saved_path outputs/saved_source_models/qwen2.5-vl-3b-instruct \
    --target_model outputs/sft-m23k/3b/qwen_med-vlrm/med-vlm-m23k-tokenized_bs16_lr1e-5_epoch5_wd1e-4_20250606_093310/checkpoint-7335 \
    --output_dir outputs/sft-m23k-converted/3b-m23k-checkpoint-7335

arg_list_file:
<source_model> <saved_path> <target_model> <output_dir>
Qwen/Qwen2.5-VL-3B-Instruct \
outputs/saved_source_models/qwen2.5-vl-3b-instruct \
outputs/sft-m23k/3b/qwen_med-vlrm/med-vlm-m23k-tokenized_bs16_lr1e-5_epoch5_wd1e-4_20250606_093310/checkpoint-7335 \
outputs/sft-m23k-converted/3b-m23k-checkpoint-7335
"""

import dotenv

dotenv.load_dotenv(override=True)

import pprint
import shutil
from pathlib import Path
from types import SimpleNamespace

import click
from huggingface_hub import snapshot_download


@click.command()
@click.option(
    "--source_model",
    type=str,
    default="Qwen/Qwen2.5-VL-3B-Instruct",
    help="Source model to download.",
)
@click.option(
    "--saved_path",
    type=str,
    default="outputs/saved_source_models/qwen2.5-vl-3b-instruct",
    help="Path to save the downloaded source model.",
)
@click.option(
    "--target_model",
    type=str,
    default="outputs/sft-m23k/3b/qwen_med-vlrm/med-vlm-m23k-tokenized_bs16_lr1e-5_epoch5_wd1e-4_20250606_093310/checkpoint-7335",
    help="Target model to convert.",
)
@click.option(
    "--output_dir",
    type=str,
    default="outputs/sft-m23k-converted/3b-m23k-checkpoint-7335",
    help="Output directory to save the converted model.",
)
@click.option(
    "--arg_list_file",
    type=str,
    default=None,
    help="Path to a file containing a list of arguments to override the defaults.",
)
def main(**kwargs):
    args = SimpleNamespace(**kwargs)

    arg_list = []
    arg_list_file = args.arg_list_file
    if arg_list_file is None:
        source_model = args.source_model
        saved_path = args.saved_path
        target_model = args.target_model
        output_dir = args.output_dir
        arg_list.append((source_model, saved_path, target_model, output_dir))
    else:
        with open(arg_list_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    source_model, saved_path, target_model, output_dir = line.split()
                    arg_list.append(
                        (source_model, saved_path, target_model, output_dir)
                    )

    for source_model, saved_path, target_model, output_dir in arg_list:
        print(f"Converting model: {source_model} to {target_model}")
        saved_path = Path(saved_path)
        output_dir = Path(output_dir)
        saved_path.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        convert_model(source_model, saved_path, target_model, output_dir)
        print(f"Saving converted model to: {output_dir}")


def convert_model(source_model, saved_path, target_model, output_dir):
    snapshot_download(
        repo_id=source_model,
        repo_type="model",
        local_dir=saved_path,
    )
    print(f"Downloaded source model to {saved_path}")

    target_model = Path(target_model)
    pprint.pprint(list(target_model.glob("model*")))

    saved_path = Path(saved_path)
    pprint.pprint(list(saved_path.glob("model*")))

    # copy saved model to output_dir
    output_dir = Path(output_dir)
    shutil.copytree(saved_path, output_dir, dirs_exist_ok=True)

    # remove model* files in output_dir
    for model_file in output_dir.glob("model*"):
        if model_file.is_file():
            model_file.unlink()
            print(f"Removed file: {model_file}")
        elif model_file.is_dir():
            raise NotImplementedError("Model directories are not supported yet.")

    # copy model* files from target_model to output_dir
    for model_file in target_model.glob("model*"):
        if model_file.is_file():
            shutil.copy(model_file, output_dir)
            print(f"Copied file: {model_file} to {output_dir}")
        elif model_file.is_dir():
            raise NotImplementedError("Model directories are not supported yet.")


if __name__ == "__main__":
    main()
