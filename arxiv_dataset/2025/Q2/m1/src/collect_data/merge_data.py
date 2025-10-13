import dotenv

dotenv.load_dotenv()

import importlib
import logging

import click
import datasets
from decontaminate import decontaminate
from deduplicate import deduplicate

logging.basicConfig(level=logging.INFO)

DATASET_FILE_FUNC = {
    "medqa": "load_medqa",
    "pubmedqa": "load_pubmedqa",
    "headqa": "load_headqa",
    "medmcqa": "load_medmcqa",
}

# Import and create a dictionary of the actual functions
dataset_loaders = {}
for dataset_name, func_name in DATASET_FILE_FUNC.items():
    try:
        # Assuming the functions are in modules with the same name as the dataset
        module = importlib.import_module(f"{dataset_name}")
        dataset_loaders[dataset_name] = getattr(module, func_name)
    except (ImportError, AttributeError) as e:
        print(f"Error loading function {func_name} for dataset {dataset_name}: {e}")


@click.command()
@click.option("--debug", is_flag=True)
@click.option("--hf_repo_id", default="mmqm/m196k", help="Hugging Face Hub repo ID")
@click.option("--force_merge", is_flag=True)
@click.option("--force_deduplicate", is_flag=True)
@click.option("--force_decontaminate", is_flag=True)
def main(
    debug=False,
    hf_repo_id="mmqm/m196k",
    force_merge=False,
    force_deduplicate=False,
    force_decontaminate=False,
):
    dataset_dict = {}
    for dataset_name, func in dataset_loaders.items():
        dataset_dict[dataset_name] = func()

    is_hf_uploaded = False
    try:
        merged_dataset = datasets.load_dataset(hf_repo_id, split="train")
        is_hf_uploaded = True
    except Exception as e:
        logging.info(f"Dataset not found on Hugging Face Hub: {hf_repo_id}")
        pass
    if force_merge or not is_hf_uploaded:
        merged_dataset = datasets.concatenate_datasets(list(dataset_dict.values()))
        upload_dataset(merged_dataset, hf_repo_id)

    is_hf_uploaded = False
    try:
        dedup_merged_dataset = datasets.load_dataset(
            f"{hf_repo_id}-dedup", split="train"
        )
        is_hf_uploaded = True
    except Exception as e:
        logging.info(f"Dataset not found on Hugging Face Hub: {hf_repo_id}-dedup")
        pass
    if force_deduplicate or not is_hf_uploaded:
        dedup_merged_dataset = deduplicate(merged_dataset, column="prompt")
        upload_dataset(dedup_merged_dataset, f"{hf_repo_id}-dedup")

    is_hf_uploaded = False
    try:
        dedup_decon_merged_dataset = datasets.load_dataset(
            f"{hf_repo_id}-dedup-decon", split="train"
        )
        is_hf_uploaded = True
    except Exception as e:
        logging.info(f"Dataset not found on Hugging Face Hub: {hf_repo_id}-dedup-decon")
        pass
    if force_decontaminate or not is_hf_uploaded:
        dedup_decon_merged_dataset = decontaminate(
            dedup_merged_dataset, column="prompt"
        )
        upload_dataset(dedup_decon_merged_dataset, f"{hf_repo_id}-dedup-decon")

    if debug:
        # fmt: off
        import IPython; IPython.embed()
        # fmt: on


def upload_dataset(dataset, repo_id):
    dataset.push_to_hub(repo_id)
    logging.info(f"Dataset uploaded to Hugging Face Hub: {repo_id}")


if __name__ == "__main__":
    main()
