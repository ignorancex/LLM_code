#!/usr/bin/env python3
# Simplified script to download only Natural Questions dataset

import os
import json
from pathlib import Path
import argparse
import shutil
import tarfile

import wget

BASE_URL = "https://dl.fbaipublicfiles.com/atlas"

def maybe_download_file(source, target):

    if not os.path.exists(target):

        # os.makedirs(os.path.dirname(target), exist_ok=True)

        print(f"Downloading {source} to {target}")

        wget.download(source, out=str(target))

        print()

def get_s3_path(path):

    return f"{BASE_URL}/{path}"

def get_download_path(output_dir, path):

    return os.path.join(output_dir, path)

# random 64 examples used with Atlas
nq_64shot = [
    27144, 14489, 49702, 38094, 6988, 60660, 65643, 48249, 48085, 52629,
    48431, 7262, 34659, 24332, 44839, 17721, 50819, 62279, 37021, 77405,
    52556, 23802, 40974, 64678, 69673, 77277, 18419, 25635, 1513, 11930,
    5542, 13453, 52754, 65663, 67400, 42409, 74541, 33159, 65445, 28572,
    74069, 7162, 19204, 63509, 12244, 48532, 72778, 37507, 70300, 29927,
    18186, 27579, 58411, 63559, 4347, 59383, 57392, 42014, 77920, 45592,
    32321, 3422, 61041, 34051,
]


def convert_nq(ex):
    return {"question": ex["question"], "answers": ex["answer"]}


def preprocess_nq(orig_dir, output_dir, index_dir):
    data, index = {}, {}
    for split in ["train", "dev", "test"]:
        with open(index_dir / ("NQ." + split + ".idx.json"), "r") as fin:
            index[split] = json.load(fin)

    originaltrain, originaldev = [], []
    with open(orig_dir / "NQ-open.dev.jsonl") as fin:
        for k, example in enumerate(fin):
            example = json.loads(example)
            originaldev.append(example)

    with open(orig_dir / "NQ-open.train.jsonl") as fin:
        for k, example in enumerate(fin):
            example = json.loads(example)
            originaltrain.append(example)

    data["train"] = [convert_nq(originaltrain[k]) for k in index["train"]]
    data["train.64-shot"] = [convert_nq(originaltrain[k]) for k in nq_64shot]
    data["dev"] = [convert_nq(originaltrain[k]) for k in index["dev"]]
    data["test"] = [convert_nq(originaldev[k]) for k in index["test"]]

    for split in data:
        with open(output_dir / (split + ".jsonl"), "w") as fout:
            for ex in data[split]:
                json.dump(ex, fout, ensure_ascii=False)
                fout.write("\n")


def main(args):
    output_dir = Path(args.output_directory)

    index_tar = output_dir / "index.tar"
    index_dir = output_dir / "dataindex"

    nq_dir = output_dir
    original_nq_dir = output_dir / "original_naturalquestions"


    # Download data index
    index_url = "https://dl.fbaipublicfiles.com/FiD/data/dataindex.tar.gz"
    maybe_download_file(index_url, index_tar)
    if not os.path.exists(index_dir):
        os.makedirs(index_dir, exist_ok=True)
        with tarfile.open(index_tar) as tar:
            tar.extractall(index_dir)

    # Create directories and download NQ data
    nq_dir.mkdir(parents=True, exist_ok=True)
    original_nq_dir.mkdir(parents=True, exist_ok=True)
    
    nq_dev_url = "https://raw.githubusercontent.com/google-research-datasets/natural-questions/master/nq_open/NQ-open.dev.jsonl"
    nq_train_url = "https://raw.githubusercontent.com/google-research-datasets/natural-questions/master/nq_open/NQ-open.train.jsonl"
    
    maybe_download_file(nq_dev_url, original_nq_dir / "NQ-open.dev.jsonl")
    maybe_download_file(nq_train_url, original_nq_dir / "NQ-open.train.jsonl")
    
    preprocess_nq(original_nq_dir, nq_dir, index_dir)


    # Clean up temporary files
    index_tar.unlink(missing_ok=True)
    if original_nq_dir.exists():
        shutil.rmtree(original_nq_dir)
    if index_dir.exists():
        shutil.rmtree(index_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_directory",
        type=str,
        default="./",
        help="Path to the directory where the dataset will be saved."
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing data")
    args = parser.parse_args()
    main(args)