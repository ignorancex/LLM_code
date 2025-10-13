import argparse
import datetime
import json
import logging
import math
import os
from pdb import set_trace as st

BASE_KEY = "_BASE_"
logger = logging.getLogger(__name__)
run = None

import sys
import time
from pathlib import Path
from pprint import pprint
from typing import Iterable

import joblib
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm


from timm.optim import create_optimizer
from timm.utils import accuracy

from torch.optim import SGD, AdamW
from torch.utils.data import ConcatDataset
from transformers import AutoTokenizer, get_constant_schedule_with_warmup

from data import (
    get_dataloader,
    get_dataset,
    load_dataset_helper,
    prepare_data,
    prepare_dataloaders, 
    prepare_dataloaders_split
)
from utils import (
    CosineSchedule,
    MetricLogger,
    SmoothedValue,
    get_params_info,
    load_saved_weights,
    load_saved_weights_original,
    load_yaml_with_base,
    print_gpu_utilization,
    save_on_master,
    set_seed,
)

from eval_promptdsi import evaluate

global sanity_check, split_length
sanity_check = False

split_length = {
    1: (0, 2000),
    2: (2000, 4000),
    3: (4000, 6000),
    4: (6000, 8000),
    5: (8000, 10000),
}

initial_model_ckpt_path_dict = {
    "nq320k_bert": "/home/thuy0050/ft49_scratch/thuy0050/promptdsi/task1_checkpoints/incdsi_full_output_98743_cls/base_model_epoch20",
    "nq320k_sbert": "/home/thuy0050/ft49_scratch/thuy0050/promptdsi/task1_checkpoints/sbert_incdsi_full_output_98743_cls/base_model_epoch20",
    "msmarco_bert": "/home/thuy0050/ft49_scratch/thuy0050/promptdsi/task1_checkpoints/msmarco_full_output_289424_cls/base_model_epoch7_best",
    "msmarco_sbert": "/home/thuy0050/ft49_scratch/thuy0050/promptdsi/task1_checkpoints/sbert_msmarco_full_output_289424_cls/base_model_epoch8_best",
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
        "--logging_step",
        default=200,
        type=int,
    )
    parser.add_argument(
        "--output_dir",
        default="output/",
        type=str,
        help="Output folder that store different experiment folders",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Seed for reproducibility",
    )

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

    if config is not None:
        args = argparse.Namespace(**config)

    return args


def get_scheduler(args, optimizer, K=None):
    if args.sched != "constant":
        if args.sched == "linear_warmup":
            return get_constant_schedule_with_warmup(optimizer, args.warmup_steps)
        elif args.sched == "cosine":
            if not K and args.epochs > 1:
                K = args.epochs
                return CosineSchedule(optimizer, K=K)
            elif K and K > 1:
                return CosineSchedule(optimizer, K=K)
            return None


def get_optimizer(args, model):
    if args.opt == "adamw":
        optimizer = get_adamw(
            model,
            learning_rate=args.lr,
            adam_eps=args.opt_eps,
            weight_decay=args.weight_decay,
        )
    elif args.opt == "sgd":
        optimizer = SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = create_optimizer(args, model)
    return optimizer


def get_adamw(
    model: nn.Module,
    learning_rate: float = 1e-5,
    adam_eps: float = 1e-8,
    weight_decay: float = 0.0,
    key_lr: float = 1e-3,
) -> torch.optim.Optimizer:
    no_decay = ["bias", "LayerNorm.weight"]
    prompt_key = ["prompt_key", "prompt_keys"]  # larger learning rate
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        eps=adam_eps,
        fused=True,
    )
    return optimizer


def get_model(
    args,
    device,
    original=False,
    train=True,
    task_id=None,
):
    logger.info(f"Loading model")

    # Original model for extracting CLS token
    if original:
        original_model = GeneralQueryClassifier(args, question_model="incdsi")
        logger.info("original model loaded")

        # Load the pre-trained bert model weights only
        if args.load_weight_original:
            logger.info("initializing original model weight")
            load_saved_weights_original(original_model, args.original_model)
        else:
            logger.info(
                "initializing sentence-transformers/all-mpnet-base-v2 model weight"
            )

        for p in original_model.parameters():
            p.requires_grad = False
        original_model.to(device)

        return original_model


def prepare_dataloaders(args, tokenizer, doc_type="old", train=True):
    docs_list = joblib.load(
        os.path.join(args.base_data_dir, f"{doc_type}_docs", "doc_list.pkl")
    )
    if doc_type == "new" and "msmarco" in args.base_data_dir:
        docs_list = docs_list[:10000]
    doc_class = joblib.load(
        os.path.join(args.base_data_dir, f"{doc_type}_docs", "doc_class.pkl")
    )

    train_length = 0
    train_dataloader = None

    # Train
    if train:
        train_data = load_dataset_helper(
            os.path.join(args.base_data_dir, f"{doc_type}_docs", "trainqueries.json")
        )
        generated_queries = load_dataset_helper(
            os.path.join(args.base_data_dir, f"{doc_type}_docs", "passages_seen.json")
        )
        if doc_type == "new" and "msmarco" in args.base_data_dir:
            train_data = train_data.filter(
                lambda example: example["doc_id"] in docs_list
            )
            generated_queries = generated_queries.filter(
                lambda example: example["doc_id"] in docs_list
            )
        train_length = len(train_data) + len(generated_queries)

        natural_queries = get_dataset(
            tokenizer=tokenizer,
            datadict=train_data,
            doc_class=doc_class,
        )
        gen_queries = get_dataset(
            tokenizer=tokenizer,
            datadict=generated_queries,
            doc_class=doc_class,
        )
        train_dataset = ConcatDataset([natural_queries, gen_queries])

        train_dataloader = get_dataloader(
            train_dataset,
            args.batch_size,
            tokenizer,
            shuffle=True,
            num_workers=args.num_workers,
        )
        logger.info(
            f"Loaded {args.base_data_dir.split('/')[-1]}_{doc_type}_(train+passages_seen)"
        )

    # Val
    _, val_length, val_dataloader = prepare_data(
        args.base_data_dir,
        doc_type,
        "val",
        tokenizer,
        args.batch_size,
        docs_list,
        doc_class,
        logger,
        num_workers=args.num_workers,
    )

    # Test
    _, test_length, test_dataloader = prepare_data(
        args.base_data_dir,
        doc_type,
        "test",
        tokenizer,
        args.batch_size,
        docs_list,
        doc_class,
        logger,
        num_workers=args.num_workers,
    )

    logger.info(f"{doc_type} train dataset size: {train_length}")
    logger.info(f"{doc_type} val dataset size: {val_length}")
    logger.info(f"{doc_type} test dataset size: {test_length}")

    class_num = len(docs_list)
    return train_dataloader, val_dataloader, test_dataloader, class_num


def train(
    args,
    original_model,
    dataloaders,
    device: torch.device,
    class_mask=None,
    log_prefix="test_log_task",
):
    # create matrix to save end-of-task accuracies
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    optimizer = None
    lr_scheduler = None

    for task_id in range(2, 6):
        with open(
            os.path.join(args.output_dir, f"train_log_task{task_id+1}.txt"), "w"
        ) as fi:
            fi.write("")

        with open(
            os.path.join(args.output_dir, f"{log_prefix}{task_id+1}.txt"), "w"
        ) as fi:
            fi.write(f"############# TRAINED UPTO TASK {task_id+1} #############")

        log_string = "#" * 10 + f"Task {task_id+1}" + "#" * 10
        logger.info(log_string)
        with open(
            os.path.join(args.output_dir, f"train_log_task{task_id+1}.txt"), "a"
        ) as fi:
            fi.write(log_string)
            
        if args.sbert:
            from model.SbertModel import GeneralQueryClassifier
        else:
            from model.BertModel import GeneralQueryClassifier

        if "nq320k" in args.base_data_dir:
            args.class_num = 98743 + task_id * 2000
            if args.class_num > 108617:
                args.class_num = 108617

            if task_id <= 1:
                ckpt_path = initial_model_ckpt_path_dict[
                    "nq320k_sbert" if args.sbert else "nq320k_bert"
                ]
            else:
                ckpt_path = os.path.join(
                    args.output_dir, f"task{task_id}_best_checkpoint.pth"
                )
        elif "msmarco" in args.base_data_dir:
            args.class_num = 289424 + task_id * 2000
            if args.class_num > 299424:
                args.class_num = 299424

            if task_id <= 1:
                ckpt_path = initial_model_ckpt_path_dict[
                    "msmarco_sbert" if args.sbert else "msmarco_bert"
                ]
            else:
                ckpt_path = os.path.join(
                    args.output_dir, f"task{task_id}_best_checkpoint.pth"
                )
                
        model = GeneralQueryClassifier(args, question_model=args.model_encoder)
        f, _ = load_saved_weights(model, ckpt_path)

        model.to(device)
        
        with open(
            os.path.join(args.output_dir, f"train_log_task{task_id+1}.txt"), "a"
        ) as fi:
            fi.write(log_string + "\n")
        
        get_params_info(model)

        # Create new optimizer for each task to clear optimizer status
        if optimizer is not None and task_id > 0 and args.reinit_optimizer:
            del optimizer

        optimizer = get_optimizer(args, model)
        lr_scheduler = get_scheduler(args, optimizer)

        train_stats = {}

        print("Optimizer:", optimizer)
        print("Scheduler:", lr_scheduler)
        logger.info(f"Using task {task_id + 1} train dataloader")
        logger.info("Printing gpu utilization of after loading model to gpu")
        print_gpu_utilization()

        with open(
            os.path.join(args.output_dir, f"train_log_task{task_id+1}.txt"), "a"
        ) as fi:
            fi.write("Optimizer: {}\n".format(optimizer))
            fi.write("Scheduler: {}\n".format(lr_scheduler))
            fi.write("Using task {} train dataloader\n".format(task_id + 1))

        training_span = range(args.epochs)
        best_loss = float("inf")
        st()

        for epoch in training_span:

            if lr_scheduler is not None and epoch > 0:
                lr_scheduler.step()

            train_stats = train_one_epoch(
                args=args,
                model=model,
                original_model=original_model,
                criterion=criterion,
                data_loader=dataloaders[task_id]["train"],
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                max_norm=args.clip_grad,
                set_training_mode=True,
                task_id=task_id,
                class_mask=class_mask,
            )
            torch.cuda.empty_cache()

            checkpoint_path = os.path.join(
                args.output_dir,
                f"task{task_id+1}_ep{epoch+1}_checkpoint.pth",
            )

            state_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch
                + 1,  # For continue training and loading the correct starting epoch
                "args": args,
            }
            if lr_scheduler is not None:
                state_dict["lr_scheduler"] = lr_scheduler.state_dict()

            logger.info(f"Saved checkpoint at epoch {epoch+1} to {checkpoint_path}")
            # Disable to save disk space
            # save_on_master(
            #     state_dict,
            #     checkpoint_path,
            # )

            with open(
                os.path.join(args.output_dir, f"{log_prefix}{task_id+1}.txt"), "a"
            ) as fi:
                fi.write("\nEPOCH: {}\n".format(epoch + 1))

            test_stats, current_loss = evaluate_till_now(
                args,
                model=model,
                original_model=original_model,
                data_loaders=dataloaders,
                device=device,
                task_id=task_id,
                class_mask=class_mask,
                acc_matrix=acc_matrix,
                log_prefix=log_prefix,
            )

            if current_loss < best_loss:
                best_loss = current_loss

                best_checkpoint_path = os.path.join(
                    args.output_dir,
                    f"task{task_id+1}_best_checkpoint.pth",
                )

                log_string = f"Saved best checkpoint at epoch {epoch+1} to {best_checkpoint_path} with loss {best_loss}"
                logger.info(log_string)
                save_on_master(
                    state_dict,
                    best_checkpoint_path,
                )
                with open(
                    os.path.join(args.output_dir, f"{log_prefix}{task_id+1}.txt"), "a"
                ) as fi:
                    fi.write(f"\n{log_string}\n")

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": args.epochs,
        }

        output_path = os.path.join(
            args.output_dir,
            "{}_stats.txt".format(
                datetime.datetime.now().strftime("log_%Y_%m_%d_%H_%M")
            ),
        )
        with open(os.path.join(output_path), "w") as fi:
            fi.write(json.dumps(log_stats) + "\n")


def train_one_epoch(
    args,
    model: torch.nn.Module,
    original_model: torch.nn.Module,
    criterion,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    set_training_mode=True,
    task_id=-1,
    class_mask=None,
    f=0,
    previous_task_key_centroids=None,
    previous_task_doc_embeddings=None,
):
    model.train(set_training_mode)
    if original_model is not None:
        original_model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("Lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "train_loss", SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    header = f"Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]"

    tr_loss = 0
    sim_loss = 0
    contrastive_loss = 0
    nce_loss = 0

    # here is the trick to mask out classes of non-current tasks
    if args.train_mask and class_mask is not None:
        mask = class_mask[task_id]
        not_mask = np.setdiff1d(np.arange(args.class_num), mask)
        not_mask = torch.tensor(not_mask, dtype=torch.int64, device=device)

    with open(
        os.path.join(args.output_dir, f"train_log_task{task_id+1}.txt"), "a"
    ) as fi:
        for i, inputs in tqdm(
            enumerate(metric_logger.log_every(data_loader, args.logging_step, header))
        ):
            model.zero_grad(set_to_none=True)
            inputs["input_ids"] = inputs["input_ids"].to(device, non_blocking=True)
            inputs["attention_mask"] = inputs["attention_mask"].to(
                device, non_blocking=True
            )
            inputs["labels"] = inputs["labels"].to(device, non_blocking=True)

            cls_features = None

            logits, output = model(
                inputs["input_ids"],
                inputs["attention_mask"],
                task_id=task_id,
                cls_features=cls_features,
                train=set_training_mode,
                return_hidden_emb=False,
                f=f,
                previous_task_key_centroids=previous_task_key_centroids,
            )

            loss = +criterion(logits, inputs["labels"])

            tr_loss += loss.item()
            hits_at_1 = accuracy(logits, inputs["labels"], topk=(1,))[0]

            if not math.isfinite(loss.item()):
                logger.info("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)
            loss.backward()
            if i == 0:
                logger.info("Printing gpu utilization of the first backward pass")
                print_gpu_utilization()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            metric_logger.update(train_loss=loss.item())
            metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters["Hits@1"].update(
                hits_at_1.item(), n=inputs["input_ids"].shape[0]
            )

            if (i + 1) % args.logging_step == 0:
                temp_hits_at_1 = metric_logger.meters["Hits@1"].global_avg
                log_string = f"Train step: {i}, Loss: {(tr_loss/(i+1)):.2f}, Hits@1: {temp_hits_at_1:.2f}"
                logger.info(log_string)
                fi.write(log_string + "\n")
                # print(torch.unique(output["idx"], return_counts=True))

            if sanity_check:
                # return {}
                break

    log_string = f"Averaged stats:, {metric_logger}"
    logger.info(log_string)
    with open(
        os.path.join(args.output_dir, f"train_log_task{task_id+1}.txt"), "a"
    ) as fi:
        fi.write(log_string + "\n")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    args,
    model: torch.nn.Module,
    original_model: torch.nn.Module,
    data_loader,
    device,
    task_id=-1,
    class_mask=None,
    f=0,
):
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test: [Task {}]".format(task_id + 1)

    # switch to evaluation mode
    model.eval()

    logger.info("Evaluating...")
    with torch.no_grad():
        for inputs in tqdm(data_loader):
            inputs["input_ids"] = inputs["input_ids"].to(device, non_blocking=True)
            inputs["attention_mask"] = inputs["attention_mask"].to(
                device, non_blocking=True
            )
            inputs["labels"] = inputs["labels"].to(device, non_blocking=True)

            cls_features = None

            logits, output = model(
                inputs["input_ids"],
                inputs["attention_mask"],
                task_id=task_id,
                cls_features=cls_features,
                train=False,
                f=f,
            )
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
            metric_logger.meters["loss"].update(
                loss.item(), n=inputs["input_ids"].shape[0]
            )

    log_string = (
        "\n######  TASK {} ###### \nHits@1 {top1.global_avg:.2f}\n"
        "Hits@5 {top5.global_avg:.2f}\n"
        "Hits@10 {top10.global_avg:.2f}\n"
        "MRR@10 {mrr.global_avg:.2f}\n"
        "loss {losses.global_avg:.2f}".format(
            task_id + 1,
            top1=metric_logger.meters["Hits@1"],
            top5=metric_logger.meters["Hits@5"],
            top10=metric_logger.meters["Hits@10"],
            mrr=metric_logger.meters["MRR@10"],
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
    f=0,
    task_id=-1,
    class_mask=None,
    split="val",
    log_prefix="",
):
    stat_matrix = np.zeros((5, 2))
    # 5 for Hits@1, Hits@5, Hits@10, MRR@10, Loss
    # 2 for initial corpus and the current corpus

    log_string = ""

    for i in [0] + [task_id]:  # Evaluate on the old task and the current task
        # for i in [task_id]: # Evaluate on the current task only
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
            return test_stats, 0

        stat_matrix[0, i if i == 0 else -1] = round(test_stats["Hits@1"], 4)
        stat_matrix[1, i if i == 0 else -1] = round(test_stats["Hits@5"], 4)
        stat_matrix[2, i if i == 0 else -1] = round(test_stats["Hits@10"], 4)
        stat_matrix[3, i if i == 0 else -1] = round(test_stats["MRR@10"], 4)
        stat_matrix[4, i if i == 0 else -1] = round(test_stats["loss"], 4)

        acc_matrix[i if i == 0 else -1, task_id if task_id == 0 else -1] = test_stats[
            "Hits@1"
        ]

        log_string += f"Hits@1_{split}_task{i+1}_trained_upto_task{task_id+1}: {test_stats['Hits@1']:.2f}\nHits@5_{split}_task{i+1}_trained_upto_task{task_id+1}: {test_stats['Hits@5']:.2f}\nHits@10_{split}_task{i+1}_trained_upto_task{task_id+1}: {test_stats['Hits@10']:.2f}\nMRR@10_{split}_task{i+1}_trained_upto_task{task_id+1}: {test_stats['MRR@10']:.2f}\nloss_{split}_task{i+1}_trained_upto_task{task_id+1}: {test_stats['loss']:.2f}\n"

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
    log_string += f"Average_Hits@1_{split}_trained_upto_task{task_id+1}: {avg_stat[0]:.2f}\nAverage_Hits@5_{split}_trained_upto_task{task_id+1}: {avg_stat[1]:.2f}\nAverage_Hits@10_{split}_trained_upto_task{task_id+1}: {avg_stat[2]:.2f}\nAverage_MRR@10_{split}_trained_upto_task{task_id+1}: {avg_stat[3]:.2f}\nAverage_loss_{split}_trained_upto_task{task_id+1}: {avg_stat[4]:.2f}\n"

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

    with open(os.path.join(args.output_dir, "config.json"), "w") as fi:
        json.dump(vars(args), fi, indent=4)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logging.basicConfig(
        filename=f"{args.output_dir}/out.log", encoding="utf-8", level=logging.DEBUG
    )

    # Save a copy of the config file
    output_path = os.path.join(args.output_dir, "config.json")
    with open(output_path, "w") as fi:
        json.dump(vars(args), fi, indent=4)

    device = torch.device("cuda")
    logging.info(f"Device: {device}")

    return args, device


def main():
    args, device = init()
    global split_length
    args.num_tasks = len(split_length) + 1
        
    if "sbert" in args.config_file:
        args.sbert = True
    else:
        args.sbert = False

    if args.sbert:
        tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-mpnet-base-v2", cache_dir="cache"
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased", cache_dir="cache"
        )

    # Load dataloader
    logger.info("Loading old dataloader")
    (
        old_train_dataloader,
        old_val_dataloader,
        old_test_dataloader,
        old_class_num,
    ) = prepare_dataloaders(args, tokenizer, doc_type="old", train=True)

    logger.info("Loading new dataloader")
    (
        new_train_dataloader1,
        new_val_dataloader1,
        new_test_dataloader1,
        new_class_num1,
        new_train_dataloader2,
        new_val_dataloader2,
        new_test_dataloader2,
        new_class_num2,
        new_train_dataloader3,
        new_val_dataloader3,
        new_test_dataloader3,
        new_class_num3,
        new_train_dataloader4,
        new_val_dataloader4,
        new_test_dataloader4,
        new_class_num4,
        new_train_dataloader5,
        new_val_dataloader5,
        new_test_dataloader5,
        new_class_num5,
    ) = prepare_new_dataloaders(args, tokenizer, train=True)

    dataloaders, class_mask = get_dataloaders_class_mask(
        args,
        old_train_dataloader,
        old_val_dataloader,
        old_test_dataloader,
        old_class_num,
        new_train_dataloader1,
        new_val_dataloader1,
        new_test_dataloader1,
        new_class_num1,
        new_train_dataloader2,
        new_val_dataloader2,
        new_test_dataloader2,
        new_class_num2,
        new_train_dataloader3,
        new_val_dataloader3,
        new_test_dataloader3,
        new_class_num3,
        new_train_dataloader4,
        new_val_dataloader4,
        new_test_dataloader4,
        new_class_num4,
        new_train_dataloader5,
        new_val_dataloader5,
        new_test_dataloader5,
        new_class_num5,
    )

    class_mask = None
    original_model = None
    
    logger.info("Training configuration")
    pprint(vars(args))

    logger.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    train(
        args,
        original_model=original_model,
        dataloaders=dataloaders,
        device=device,
        class_mask=class_mask,
    )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Total training time: {total_time_str}")


def get_dataloaders_class_mask(
    args,
    old_train_dataloader,
    old_val_dataloader,
    old_test_dataloader,
    old_class_num,
    new_train_dataloader1,
    new_val_dataloader1,
    new_test_dataloader1,
    new_class_num1,
    new_train_dataloader2,
    new_val_dataloader2,
    new_test_dataloader2,
    new_class_num2,
    new_train_dataloader3,
    new_val_dataloader3,
    new_test_dataloader3,
    new_class_num3,
    new_train_dataloader4,
    new_val_dataloader4,
    new_test_dataloader4,
    new_class_num4,
    new_train_dataloader5,
    new_val_dataloader5,
    new_test_dataloader5,
    new_class_num5,
):
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

    args.class_num = (
        old_class_num
        + new_class_num1
        + new_class_num2
        + new_class_num3
        + new_class_num4
        + new_class_num5
    )

    t1 = old_class_num + new_class_num1  # 100 new docs
    t2 = t1 + new_class_num2  # 900 new docs
    t3 = t2 + new_class_num3  # 9000 new docs
    t4 = t3 + new_class_num4  # 9000 new docs
    t5 = t4 + new_class_num5  # 9000 new docs

    class_mask = [
        [i for i in range(old_class_num)],
        [i for i in range(old_class_num, t1)],
        [i for i in range(t1, t2)],
        [i for i in range(t2, t3)],
        [i for i in range(t3, t4)],
        [i for i in range(t4, t5)],
    ]

    return dataloaders, class_mask


def prepare_new_dataloaders(args, tokenizer, train=True):
    (
        new_train_dataloader1,
        new_val_dataloader1,
        new_test_dataloader1,
        new_class_num1,
    ) = prepare_dataloaders_split(args, tokenizer, doc_type="new", split=1, train=train)
    (
        new_train_dataloader2,
        new_val_dataloader2,
        new_test_dataloader2,
        new_class_num2,
    ) = prepare_dataloaders_split(args, tokenizer, doc_type="new", split=2, train=train)
    (
        new_train_dataloader3,
        new_val_dataloader3,
        new_test_dataloader3,
        new_class_num3,
    ) = prepare_dataloaders_split(args, tokenizer, doc_type="new", split=3, train=train)
    (
        new_train_dataloader4,
        new_val_dataloader4,
        new_test_dataloader4,
        new_class_num4,
    ) = prepare_dataloaders_split(args, tokenizer, doc_type="new", split=4, train=train)
    (
        new_train_dataloader5,
        new_val_dataloader5,
        new_test_dataloader5,
        new_class_num5,
    ) = prepare_dataloaders_split(args, tokenizer, doc_type="new", split=5, train=train)

    return (
        new_train_dataloader1,
        new_val_dataloader1,
        new_test_dataloader1,
        new_class_num1,
        new_train_dataloader2,
        new_val_dataloader2,
        new_test_dataloader2,
        new_class_num2,
        new_train_dataloader3,
        new_val_dataloader3,
        new_test_dataloader3,
        new_class_num3,
        new_train_dataloader4,
        new_val_dataloader4,
        new_test_dataloader4,
        new_class_num4,
        new_train_dataloader5,
        new_val_dataloader5,
        new_test_dataloader5,
        new_class_num5,
    )



if __name__ == "__main__":
    main()
