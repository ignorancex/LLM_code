import json
import logging
import os
import tarfile
from argparse import ArgumentParser
from datetime import timedelta
from urllib import parse

import boto3
import jsonlines
import torch
import torch.distributed as dist
import torch.nn.functional as F
from qafacteval import QAFactEval
from other_utils import (
    extract_summary_from_txt,
    load_documents,
    str2bool,
)


def split_by_rank(rank, world_size, data):
    chunk_size = int(len(data) / world_size) + 1
    return data[rank * chunk_size : (rank + 1) * chunk_size]


def run_qafacteval(rank, documents, predicted_summaries):
    kwargs = {
        "cuda_device": rank,
        "use_lerc_quip": True,
        "verbose": True,
        "generation_batch_size": 32,
        "answering_batch_size": 32,
        "lerc_batch_size": 8,
    }

    model_folder = "/home/ec2-user/SageMaker/QAFactEval/models"  # path to models downloaded with download_models.sh
    metric = QAFactEval(
        lerc_quip_path=f"{model_folder}/quip-512-mocha",
        generation_model_path=f"{model_folder}/generation/model.tar.gz",
        answering_model_dir=f"{model_folder}/answering",
        lerc_model_path=f"{model_folder}/lerc/model.tar.gz",
        lerc_pretrained_model_path=f"{model_folder}/lerc/pretraining.tar.gz",
        **kwargs,
    )

    results = metric.score_batch_qafacteval(
        source=documents, summaries=predicted_summaries, return_qa_pairs=True
    )
    scores = [score[0]["qa-eval"]["lerc_quip"] for score in results]
    return scores


def download_s3_file(tarfile_uri, local_target="tmp/output.tar.gz"):
    s3 = boto3.client("s3")
    url = parse.urlparse(tarfile_uri)
    bucket, key = url.netloc, url.path.lstrip("/")
    s3.download_file(bucket, key, local_target)
    return


def extract_output_file(target="tmp"):
    for file in os.listdir(target):
        found_file = False
        if file.endswith("txt.json"):
            found_file = True
            summaries = []
            with jsonlines.open(os.path.join(target, file), "r") as f:
                for line in f:
                    # For e2e, we separate plan from summary. Otherwise, it shouldn't touch the summary.
                    # Note for QAFactEval the summary needs to be placed in a list
                    summaries.append([extract_summary_from_txt(line["output"])])
            return summaries
    if not found_file:
        raise ValueError(
            f"Could not find prediction .txt file in {target}. Please make sure the output.tar.gz contains it."
        )
    return


def get_summaries_from_predictions_file(predictions_file):
    summaries = []
    with open(predictions_file, "r") as fhr:
        for line in fhr:
            line = json.loads(line.strip())
            summaries.append([extract_summary_from_txt(line["output"])])
    return summaries


def cleanup_tmp_files():
    for file in os.listdir("tmp"):
        if file.endswith(".tar.gz"):
            os.remove(os.path.join("tmp", file))
        elif file.endswith(".json"):
            os.remove(os.path.join("tmp", file))
    return


def distributed_setup():
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=100000))


def main(
    save_dir,
    predictions_file,
    s3_summaries_tarfile_uri,
    dataset_name,
    local_download_path,
    is_sampled,
    num_reference_summaries,
    min_num_sentences,
):
    distributed_setup()
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    if not predictions_file and rank == 0:
        download_s3_file(s3_summaries_tarfile_uri)
        with tarfile.open(local_download_path, "r:gz") as tar:
            tar.extractall(local_download_path.split("/output.tar.gz")[0])

    dist.barrier()

    if predictions_file:
        predicted_summaries = get_summaries_from_predictions_file(predictions_file)
    else:
        predicted_summaries = extract_output_file(
            local_download_path.replace("output.tar.gz", "")
        )

    documents = load_documents(dataset_name, min_num_sentences)

    if rank == 0:
        print("Found {} summaries".format(len(predicted_summaries), rank))
        print("Found {} documents".format(len(documents), rank))

    if num_reference_summaries > 1:
        if not is_sampled:
            # more than one ref summary and gen summary is not sampled
            dataset_size = len(documents) // num_reference_summaries
            predicted_summaries = predicted_summaries[:dataset_size]
            documents = documents[:dataset_size]

    assert len(predicted_summaries) == len(documents)

    if rank == 0:
        print("Found {} summaries".format(len(predicted_summaries), rank))
        print("Found {} documents".format(len(documents), rank))

    part_pred_summaries = split_by_rank(rank, world_size, predicted_summaries)
    part_documents = split_by_rank(rank, world_size, documents)

    print(
        "Processing {} documents on process rank {}".format(len(part_documents), rank)
    )
    scores = run_qafacteval(rank, part_documents, part_pred_summaries)
    scores = torch.tensor(scores, dtype=torch.float, device=device)

    pad_size = world_size - (len(documents) % world_size)
    if rank == world_size - 1:
        # Gather must take the same size tensor for all processes, so we add padding to final one
        scores = F.pad(scores, (0, pad_size), value=-1)

    tensor_out = torch.zeros(
        len(documents) + pad_size, dtype=torch.float, device=device
    )
    dist.all_gather_into_tensor(tensor_out, scores)

    tensor_out = tensor_out[tensor_out != -1.0]

    if rank == 0:
        if predictions_file:
            mean_score = torch.mean(tensor_out).item()
            with open(os.path.join(save_dir, "qafacteval.txt"), "w") as fhw:
                fhw.write(f"qafacteval: {mean_score:.4f}\n")
        else:
            url = parse.urlparse(s3_summaries_tarfile_uri)
            setting_name = url.path.split("/")[1]
            mean_score = torch.mean(tensor_out).item()
            with open(os.path.join(save_dir, "qafacteval.txt"), "w") as fhw:
                fhw.write(f"{setting_name} qafacteval: {mean_score:.4f}\n")
            cleanup_tmp_files()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--s3_summaries_tarfile_uri",
        type=str,
        help="Should be S3 URI to a output.tar.gz file",
    )
    parser.add_argument(
        "--predictions_file",
        type=str,
        default="",
        help="Path to predictions jsonl file with one json per line. If --predictions_file is passed then --s3_summaries_tarfile_uri will be ignored.",
    )
    parser.add_argument("--dataset_name", type=str, help="multinews or summscreen")
    parser.add_argument(
        "--local_download_path",
        type=str,
        help="Where to download the output.tar.gz file",
        default="tmp/output.tar.gz",
    )
    parser.add_argument(
        "--is_sampled",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Is sampling enabled during generation",
    )
    parser.add_argument(
        "--num_reference_summaries",
        type=int,
        default=1,
        help="number of reference summaries per data point",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="Path where alignscore result should be saved",
    )
    parser.add_argument(
        "--min_num_sentences",
        type=int,
        default=1,
        help="minimum number of sentences in the reference summary to retain that data point",
    )

    args = parser.parse_args()

    os.environ["TRANSFORMERS_CACHE"] = "/home/ec2-user/SageMaker/transformers_cache"
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/home/ec2-user/SageMaker/transformers_cache"
    os.environ["HF_DATASETS_CACHE"] = "/home/ec2-user/SageMaker/hf_data_cache"
    os.environ["TMPDIR"] = "/home/ec2-user/SageMaker/tmp/"
    os.environ["SACREROUGE_DATA_ROOT"] = "/home/ec2-user/SageMaker/sacrerouge_data_root"

    if not args.predictions_file:
        # Validate args
        if args.s3_summaries_tarfile_uri.split("/")[-1] != "output.tar.gz":
            raise ValueError(
                f"Incorrect URI: {args.s3_summaries_tarfile_uri} should be a file with output.tar.gz"
            )

    # This one is specific for this script, in order to set timeout in init_process_group
    os.environ["NCCL_BLOCKING_WAIT"] = "1"

    logging.disable(logging.WARNING)
    torch.set_grad_enabled(False)
    main(
        args.save_dir,
        args.predictions_file,
        args.s3_summaries_tarfile_uri,
        args.dataset_name,
        args.local_download_path,
        args.is_sampled,
        args.num_reference_summaries,
        args.min_num_sentences,
    )
