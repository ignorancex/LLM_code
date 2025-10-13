import dotenv

dotenv.load_dotenv()

import json
import multiprocessing as mp
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from typing import List, Tuple

import click
import datasets
from bespokelabs import curator
from classify_sample import classify_sample_with_llm
from omegaconf import OmegaConf
from rapidfuzz import fuzz, process
from tqdm import tqdm


@dataclass
class Config:
    repo_id: str = (
        "mmqm/m196k-dedup-decon-filter_easy-r1-filter_wrong-decon_eval-tokenized-120325-domain-classification"
    )
    split: str = "train"
    upload_id: str = (
        "mmqm/m196k-dedup-decon-filter_easy-r1-filter_wrong-decon_eval-tokenized-120325-extract-domain-code"
    )
    dry_run: bool = False
    failed_extraction_save_path: str = "misc/failed_extraction.json"
    domain_prompt_json_path: str = "src/select_data/mesh_qualifier.json"


@click.command()
@click.option(
    "--config_path",
    "-c",
    type=str,
    default=None,
)
@click.option(
    "--update_config_by_dotlist",
    "-u",
    type=str,
    default=None,
    help="Update config by dotlist",
)
def main(
    config_path=None,
    update_config_by_dotlist=None,
):
    config: Config = OmegaConf.structured(Config)
    if config_path is not None:
        config = OmegaConf.load(config_path)
    if update_config_by_dotlist is not None:
        update_config_by_dotlist = update_config_by_dotlist.split(",")
        update_config_by_dotlist = OmegaConf.from_dotlist(update_config_by_dotlist)
        config = OmegaConf.merge(config, update_config_by_dotlist)
    print(OmegaConf.to_yaml(config))

    repo_id = config.repo_id
    split = config.split

    dataset = datasets.load_dataset(repo_id, split=split)

    extract_domain_dataset_list = []
    extract_domain_dataset, failed_extraction = extract_code_for_dataset(
        config, dataset
    )
    extract_domain_dataset_list.append(extract_domain_dataset)
    failed_extraction.to_json(config.failed_extraction_save_path)
    print(f"Failed extraction saved to {config.failed_extraction_save_path}")

    # Retry failed extraction
    failed_extraction = classify_sample_with_llm(
        config.domain_prompt_json_path, failed_extraction
    )
    extract_domain_dataset_tmp, failed_extraction_tmp = extract_code_for_dataset(
        config, failed_extraction
    )
    if len(failed_extraction_tmp) != 0:
        # NOTE: gpt-4o is quite stable, so this should not happen.
        raise ValueError("Failed extraction is not empty, which should not happened.")
    extract_domain_dataset_list.append(extract_domain_dataset_tmp)

    extract_domain_dataset = datasets.concatenate_datasets(
        extract_domain_dataset_list, axis=0
    )
    extract_domain_dataset = extract_domain_dataset.filter(
        lambda x: x["is_domain_code_extracted"]
    )
    if len(extract_domain_dataset) != len(dataset):
        raise ValueError(
            f"len(extract_domain_dataset) {len(extract_domain_dataset)} != len(dataset) {len(dataset)}"
        )

    upload_repo_id = config.upload_id
    extract_domain_dataset.push_to_hub(upload_repo_id)
    print(f"Uploaded to {upload_repo_id}")


def extract_code_for_dataset(config, dataset):
    with open(config.domain_prompt_json_path, "r") as f:
        domain_prompt = json.load(f)
    domain_code_dict = {subject["code"]: subject["title"] for subject in domain_prompt}

    extract_domain_dataset = dataset.map(
        partial(extract_code, domain_code_dict=domain_code_dict)
    )
    failed_extraction = extract_domain_dataset.filter(
        lambda x: not x["is_domain_code_extracted"]
    )

    return extract_domain_dataset, failed_extraction


def extract_code(
    sample: dict,
    domain_code_dict: dict,
) -> Tuple[str, str]:
    domain_text = sample["domain"]

    # replace "*03*" with "03"
    domain_text = domain_text.replace("*", "")
    domain_text = domain_text.replace("`", "")
    domain_text = domain_text.strip()
    domain_text = domain_text.rstrip(".")

    domain_name = ""
    is_domain_code_extracted = True

    domain_code = domain_text[-2:]
    if len(domain_code) != 2:
        is_domain_code_extracted = False
    elif not domain_code.isdigit():
        is_domain_code_extracted = False
    elif domain_code not in domain_code_dict:
        is_domain_code_extracted = False
    else:
        domain_name = domain_code_dict[domain_code]

    return {
        "domain_code": domain_code,
        "domain_name": domain_name,
        "is_domain_code_extracted": is_domain_code_extracted,
    }


if __name__ == "__main__":
    main()
