import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
import time
torch.set_printoptions(sci_mode=False)

from data import (
    prepare_dataloaders,
    prepare_dataloaders_split,
)
from sentence_transformers import SentenceTransformer
from timm.utils import accuracy
from transformers import AutoTokenizer
from utils import (MetricLogger, get_model, load_yaml_with_base, print_gpu_utilization, set_seed, load_saved_weights, load_saved_weights_original)
from pdb import set_trace as st 

# torch.autograd.set_detect_anomaly(True)
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
BASE_KEY = "_BASE_"
logger = logging.getLogger(__name__)
run = None

global sanity_check
sanity_check = False

# Dataset paths
base_data_dir_dict = {
    "nq320k": "/home/thuy0050/ft49/thuy0050/data/incdsi/nq320k",
    "msmarco": "/home/thuy0050/ft49/thuy0050/data/incdsi/msmarco",
}

global counts
counts = torch.zeros(91)

# Initial checkpoints paths trained on $D_0$ corpus
initial_model_ckpt_path_dict = {
    "nq320k_bert": "/home/thuy0050/ft49_scratch/thuy0050/incdsi/task1_checkpoints/incdsi_full_output_98743_cls/base_model_epoch20",
    "nq320k_sbert": "/home/thuy0050/ft49_scratch/thuy0050/incdsi/task1_checkpoints/sbert_incdsi_full_output_98743_cls/base_model_epoch20",
    "msmarco_bert": "/home/thuy0050/ft49_scratch/thuy0050/incdsi/task1_checkpoints/msmarco_full_output_289424_cls/base_model_epoch7_best",
    "msmarco_sbert": "/home/thuy0050/ft49_scratch/thuy0050/incdsi/task1_checkpoints/sbert_msmarco_full_output_289424_cls/base_model_epoch8_best",
    "nq320k_roberta": "/home/thuy0050/ft49_scratch/thuy0050/promptdsi/task1_checkpoints/roberta_incdsi_full_output_98743_cls/base_model_epoch20"
}


def get_arguments(config_file=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        default="" if config_file is None else config_file,
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--exp_name",
        default="exp1",
        type=str,
        help="Experiment name",
    )
    parser.add_argument(
        "--output_dir",
        default="output/",
        type=str,
        help="Output folder that store different experiment folders",
    )
    parser.add_argument(
        "--val_only",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--test_only",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Seed for reproducibility",
    )
    # parser.add_argument(
    #     "--e_prompt_layer_idx",
    #     default=[0],
    #     type=int,
    #     nargs="+",
    #     help="the layer index of the E-Prompt",
    # )
    # parser.add_argument(
    #     "--prompt_feature_layer",
    #     default=0,
    #     type=int,
    # )
    args = parser.parse_args()

    # Load YAML configuration file
    config = None
    if args.config_file:
        # Load YAML configuration file
        if Path(args.config_file).suffix == ".yaml":
            config = load_yaml_with_base(args.config_file)
        elif Path(args.config_file).suffix == ".json":
            with open(args.config_file) as fi:
                config = json.load(fi)
        # Merge command-line arguments with YAML configuration
        config = {**config, **vars(args)}
    # else:
    #     config = vars(args)

    # st()
    if config is not None:
        args = argparse.Namespace(**config)

    return args


@torch.no_grad()
def evaluate(
    args,
    model: torch.nn.Module,
    original_model: torch.nn.Module,
    data_loader,
    device,
    f,
    task_id=-1,
    class_mask=None,
):
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test: [Task {}]".format(task_id + 1)

    # switch to evaluation mode
    if original_model is not None:
        original_model.eval()
        
    model.eval()

    logger.info("Evaluating...")
    with torch.no_grad():
        times = 0.0 # Benchmarking single-pass PCL methods speedup
        
        for inputs in tqdm(data_loader):
            inputs["input_ids"] = inputs["input_ids"].to(device)
            inputs["attention_mask"] = inputs["attention_mask"].to(device)
            inputs["labels"] = inputs["labels"].to(device)
            
            
            start = time.perf_counter()
            if original_model is not None:
                if args.sbert:
                    cls_features = original_model.encode(
                        inputs["texts"],
                        convert_to_tensor=True,
                        batch_size=128,
                        show_progress_bar=False,
                    )
                else:
                    # Get the cls features from the original model
                    cls_features = original_model(
                        inputs["input_ids"],
                        inputs["attention_mask"],
                        return_hidden_emb=True,
                        # layerwise_allocation=(
                        #     args.layerwise_allocation
                        #     and args.model_encoder == "flexprompt_layerwise_cls_feature"
                        # ),
                    )

            else:
                cls_features = None

            logits, output = model(
                inputs["input_ids"],
                inputs["attention_mask"],
                task_id=task_id,
                cls_features=cls_features,
                train=False,
                f=f,
            )
            end = time.perf_counter()
            times += (end-start)
            
            temp = torch.unique(output["idx"], return_counts=True)

            global counts
            counts[temp[0].cpu()] += temp[1].cpu()

            loss = criterion(logits, inputs["labels"])

            hits_at_1, hits_at_5, hits_at_10 = accuracy(
                logits, inputs["labels"], topk=(1, 5, 10)
            )
            # compute recall@10
            max_idxs_10 = torch.argsort(logits, 1, descending=True)[:, :10]

            # compute mrr@10. Sum will later be divided by number of elements
            mrr_at_10 = (
                (
                    1 / (torch.where(max_idxs_10 == inputs["labels"][:, None])[1] + 1)
                ).sum()
                * 100.0
                / inputs["input_ids"].shape[0]
            )

            ndcg_at_10 = (
                (
                    1
                    / (
                        torch.log2(
                            torch.where(max_idxs_10 == inputs["labels"][:, None])[1] + 2
                        )
                    )
                ).sum()
                * 100.0
                / inputs["input_ids"].shape[0]
            )

            metric_logger.meters["Hits@1"].update(
                hits_at_1.item(), n=inputs["input_ids"].shape[0]
            )
            metric_logger.meters["Hits@5"].update(
                hits_at_5.item(), n=inputs["input_ids"].shape[0]
            )
            metric_logger.meters["Hits@10"].update(
                hits_at_10.item(), n=inputs["input_ids"].shape[0]
            )
            metric_logger.meters["MRR@10"].update(
                mrr_at_10.item(), n=inputs["input_ids"].shape[0]
            )
            metric_logger.meters["nDCG@10"].update(
                ndcg_at_10.item(), n=inputs["input_ids"].shape[0]
            )
            metric_logger.meters["loss"].update(
                loss.item(), n=inputs["input_ids"].shape[0]
            )
            if sanity_check:
                return {}

        # print(times / len(data_loader))
        print(f"Task {task_id}")
        print("Selection above 0:", len(counts[counts > 0]))
        another_temp = counts / counts.sum() * 100
        print("Check prob of another_temp", another_temp.sum())
        print("Number of prompts with over uniform selection percentage:", len(another_temp[another_temp >= (100.0/len(counts))]))
        counts = torch.zeros(91)
        st()

    log_string = (
        "\n######  TASK {} ###### \nHits@1 {top1.global_avg:.2f}\n"
        "Hits@5 {top5.global_avg:.2f}\n"
        "Hits@10 {top10.global_avg:.2f}\n"
        "MRR@10 {mrr.global_avg:.2f}\n"
        # "nDCG@10 {ndcg.global_avg:.2f}\n"
        "loss {losses.global_avg:.2f}".format(
            task_id + 1,
            top1=metric_logger.meters["Hits@1"],
            top5=metric_logger.meters["Hits@5"],
            top10=metric_logger.meters["Hits@10"],
            mrr=metric_logger.meters["MRR@10"],
            # ndcg=metric_logger.meters["nDCG@10"],
            losses=metric_logger.meters["loss"],
        )
    )
    logger.info(log_string)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_till_now(
    args,
    model,
    original_model,
    data_loaders,
    device,
    acc_matrix,
    task_id=-1,
    class_mask=None,
    split="val",
    f=0,
    log_prefix="",
):
    stat_matrix = np.zeros((6, 6))
    # 5 for Hits@1, Hits@5, Hits@10, MRR@10, Loss
    # 6 for initial corpus and 5 new corpora

    log_string = ""

    for i in range(task_id + 1):
        data_loader = data_loaders[i][split]
        test_stats = evaluate(
            args,
            model=model,
            original_model=original_model,
            data_loader=data_loader,
            device=device,
            task_id=i,
            class_mask=class_mask,
            f=f,
        )
        if sanity_check:
            return {}

        stat_matrix[0, i if i == 0 else -1] = round(test_stats["Hits@1"], 4)
        stat_matrix[1, i if i == 0 else -1] = round(test_stats["Hits@5"], 4)
        stat_matrix[2, i if i == 0 else -1] = round(test_stats["Hits@10"], 4)
        stat_matrix[3, i if i == 0 else -1] = round(test_stats["MRR@10"], 4)
        stat_matrix[4, i if i == 0 else -1] = round(test_stats["nDCG@10"], 4)
        stat_matrix[5, i if i == 0 else -1] = round(test_stats["loss"], 4)

        acc_matrix[i if i == 0 else -1, task_id if task_id == 0 else -1] = test_stats[
            "Hits@1"
        ]

        log_string += f"Hits@1_{split}_task{i+1}_trained_upto_task{task_id+1}: {test_stats['Hits@1']:.2f}\nHits@5_{split}_task{i+1}_trained_upto_task{task_id+1}: {test_stats['Hits@5']:.2f}\nHits@10_{split}_task{i+1}_trained_upto_task{task_id+1}: {test_stats['Hits@10']:.2f}\nMRR@10_{split}_task{i+1}_trained_upto_task{task_id+1}: {test_stats['MRR@10']:.2f}\nnDCG@10_{split}_task{i+1}_trained_upto_task{task_id+1}: {test_stats['nDCG@10']:.2f}\nloss_{split}_task{i+1}_trained_upto_task{task_id+1}: {test_stats['loss']:.2f}\n"


    avg_stat = np.divide(np.sum(stat_matrix, axis=1), 2)
    beta = args.beta
    current_metric = (
        (1 + beta**2)
        * (stat_matrix[3][1] * stat_matrix[3][0])
        / ((beta**2) * stat_matrix[3][1] + stat_matrix[3][0])
    )

    diagonal = np.diag(acc_matrix)

    result_str = f"[Average accuracy till task{task_id+1}]\nHits@1: {avg_stat[0]:.2f}\tHits@5: {avg_stat[1]:.2f}\tHits@10: {avg_stat[2]:.2f}\tMRR@10: {avg_stat[3]:.2f}\tLoss: {avg_stat[4]:.2f}\n"

    log_string += result_str

    log_string += "##### AVERAGE STATS #####\n"
    log_string += f"Average_Hits@1_{split}_trained_upto_task{task_id+1}: {avg_stat[0]:.2f}\nAverage_Hits@5_{split}_trained_upto_task{task_id+1}: {avg_stat[1]:.2f}\nAverage_Hits@10_{split}_trained_upto_task{task_id+1}: {avg_stat[2]:.2f}\nAverage_MRR@10_{split}_trained_upto_task{task_id+1}:{avg_stat[3]:.2f}\nAverage_nDCG@10_{split}_trained_upto_task{task_id+1}:{avg_stat[4]:.2f}\nAverage_loss_{split}_trained_upto_task{task_id+1}: {avg_stat[5]:.2f}\n"

    print(log_string)

    with open(os.path.join(args.output_dir, f"{log_prefix}{task_id+1}.txt"), "a") as fi:
        fi.write(log_string + "\n")

    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) - acc_matrix[:, -1])[:-1])
        backward = np.mean((acc_matrix[:, -1] - diagonal)[:-1])
        result_str += "\nForgetting: {:.2f}\tBackward: {:.2f}".format(
            forgetting, backward
        )

        log_string = f"Forgetting_trained_upto_task{task_id+1}: {forgetting:.2f}\nBackward_trained_upto_task{task_id+1}: {backward:.2f}\n"
        print(log_string)

        with open(
            os.path.join(args.output_dir, f"{log_prefix}{task_id+1}.txt"), "a"
        ) as fi:
            fi.write(log_string + "\n")

    logger.info(result_str)

    return test_stats, current_metric


def init():
    logger.info("Initializing...")
    args = get_arguments()
    set_seed(args.seed)
    global run

    args.output_dir = os.path.join(args.output_dir, args.exp_name)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logging.basicConfig(
        filename=f"{args.output_dir}/out.log", encoding="utf-8", level=logging.DEBUG
    )

    device = torch.device("cuda")
    logging.info(f"Device: {device}")

    if args.dataset is not None:
        if args.dataset == "nq320k":
            args.base_data_dir = base_data_dir_dict["nq320k"]
        elif args.dataset == "msmarco":
            args.base_data_dir = base_data_dir_dict["msmarco"]

    return args, device


def main():
    args, device = init()
    args.num_tasks = 6
    args.num_workers = 1

    if "sbert" in args.config_file:
        args.sbert = True
    else:
        args.sbert = False

    if args.sbert:
        from model.SbertModel import GeneralQueryClassifier

        tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-mpnet-base-v2", cache_dir="cache"
        )
    elif "roberta" in args.model_encoder:
        from model.RobertaModel import GeneralQueryClassifier
        tokenizer = AutoTokenizer.from_pretrained(
            "roberta-base", cache_dir="cache"
        )
    else:
        from model.BertModel import GeneralQueryClassifier

        tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased", cache_dir="cache"
        )

    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    print_gpu_utilization()

    logger.info("Loading old dataloader")
    (
        old_train_dataloader,
        old_val_dataloader,
        old_test_dataloader,
        old_class_num,
    ) = prepare_dataloaders(args, tokenizer, doc_type="old", train=False)

    ### New ###
    (
        new_train_dataloader1,
        new_val_dataloader1,
        new_test_dataloader1,
        new_class_num1,
    ) = prepare_dataloaders_split(args, tokenizer, doc_type="new", split=1, train=False)
    (
        new_train_dataloader2,
        new_val_dataloader2,
        new_test_dataloader2,
        new_class_num2,
    ) = prepare_dataloaders_split(args, tokenizer, doc_type="new", split=2, train=False)
    (
        new_train_dataloader3,
        new_val_dataloader3,
        new_test_dataloader3,
        new_class_num3,
    ) = prepare_dataloaders_split(args, tokenizer, doc_type="new", split=3, train=False)
    (
        new_train_dataloader4,
        new_val_dataloader4,
        new_test_dataloader4,
        new_class_num4,
    ) = prepare_dataloaders_split(args, tokenizer, doc_type="new", split=4, train=False)
    (
        new_train_dataloader5,
        new_val_dataloader5,
        new_test_dataloader5,
        new_class_num5,
    ) = prepare_dataloaders_split(args, tokenizer, doc_type="new", split=5, train=False)

    dataloaders = [
        {
            "train": old_train_dataloader,
            "val": old_val_dataloader,
            "test": old_test_dataloader,
        },
        {
            "train": new_train_dataloader1,
            "val": new_val_dataloader1,
            "test": new_test_dataloader1,
        },
        {
            "train": new_train_dataloader2,
            "val": new_val_dataloader2,
            "test": new_test_dataloader2,
        },
        {
            "train": new_train_dataloader3,
            "val": new_val_dataloader3,
            "test": new_test_dataloader3,
        },
        {
            "train": new_train_dataloader4,
            "val": new_val_dataloader4,
            "test": new_test_dataloader4,
        },
        {
            "train": new_train_dataloader5,
            "val": new_val_dataloader5,
            "test": new_test_dataloader5,
        },
    ]

    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

    logger.info(f"##### {'VAL' if args.val_only else 'TEST'} ONLY MODE #####")

    # for task_id in range(1, 6):
    for task_id in range(5, 6):
        logger.info(
            f"##### {'VAL' if args.val_only else 'TEST'} MODEL TRAINED UPTO TASK {task_id+1} #####\n"
        )

        if "nq320k" in args.base_data_dir:
            args.class_num = 98743 + task_id * 2000
            if args.class_num > 108617:
                args.class_num = 108617
                
            if args.sbert:
                name = "nq320k_sbert"
            elif "roberta" in args.model_encoder:
                name = "nq320k_roberta"
            else:
                name = "nq320k_bert"

            if task_id == 0:
                ckpt_path = initial_model_ckpt_path_dict[name]
            else:
                ckpt_path = os.path.join(
                    args.output_dir, f"task{task_id+1}_best_checkpoint.pth"
                )
        elif "msmarco" in args.base_data_dir:
            args.class_num = 289424 + task_id * 2000
            if args.class_num > 289424:
                args.class_num = 299424
                
            if args.sbert:
                name = "msmarco_sbert"
            elif "roberta" in args.model_encoder:
                name = "msmarco_roberta"
            else:
                name = "msmarco_bert"

            if task_id == 0:
                ckpt_path = initial_model_ckpt_path_dict[
                    "msmarco_sbert" if args.sbert else "msmarco_bert"
                ]
            else:
                ckpt_path = os.path.join(
                    args.output_dir, f"task{task_id+1}_best_checkpoint.pth"
                )
        if "os" in args.config_file: 
            # one-stage, using bert's intermediate layers as query vectors.
            original_model = None
        elif args.sbert:
            original_model = SentenceTransformer("all-mpnet-base-v2")
        else:
            original_model = get_model(args, device, original=True)
            
        global counts
            
        # if args.sbert and "msmarco" in args.base_data_dir:
        #     counts = torch.zeros(193)
        # elif args.sbert and "nq320k" in args.base_data_dir:
        #     counts = torch.zeros(91)
        # elif "msmarco" in args.base_data_dir:
        #     counts = torch.zeros(182)
        # elif "nq320k" in args.base_data_dir:
        #     counts = torch.zeros(91)
        
        global counts 
        counts= torch.zeros(91)

        # ckpt_path = "/home/thuy0050/ft49_scratch/thuy0050/incdsi/seed0/nq320k/incdsi_batch_bert/final_model_ep10"
        logger.info("loaded model from {}".format(ckpt_path))
        model = GeneralQueryClassifier(args, question_model=args.model_encoder)
        f, _ = load_saved_weights(model, ckpt_path)

        model.to(device)

        with open(
            os.path.join(
                args.output_dir,
                f"final_{'test' if args.test_only else 'val'}_log_task{task_id+1}.txt",
            ),
            "w",
        ) as fi:
            fi.write("")

        _ = evaluate_till_now(
            args,
            model=model,
            original_model=original_model,
            data_loaders=dataloaders,
            device=device,
            task_id=task_id,
            acc_matrix=acc_matrix,
            split="test" if args.test_only else "val",
            f=f,
            log_prefix=f"final_{'test' if args.test_only else 'val'}_log_task",
        )

        log_string = f"\nDone evaluating {'val' if args.val_only else 'test'} of task {task_id+1}\n############################################ \n\n"
        logger.info(log_string)

        with open(
            os.path.join(
                args.output_dir,
                f"final_{'test' if args.test_only else 'val'}_log_task{task_id+1}.txt",
            ),
            "a",
        ) as fi:
            fi.write(log_string + "\n")


if __name__ == "__main__":
    main()
