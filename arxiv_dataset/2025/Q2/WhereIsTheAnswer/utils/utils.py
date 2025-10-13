# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import torch
import random
import numpy as np
from transformers import set_seed, AutoTokenizer
import json
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import torch.nn as nn
import copy
from datasets import load_dataset
import evaluate
from functools import partial
from evaluate import load
from collections import defaultdict

def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)


def _z3_params_to_fetch(param_list):
    return [
        p
        for p in param_list
        if hasattr(p, "ds_id") and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]

def save_zero_three_model(model_ema, global_rank, save_dir, zero_stage=0):
    zero_stage_3 = zero_stage == 3
    os.makedirs(save_dir, exist_ok=True)
    WEIGHTS_NAME = "pytorch_model.bin"
    output_model_file = os.path.join(save_dir, WEIGHTS_NAME)

    model_to_save = model_ema.module if hasattr(model_ema, "module") else model_ema
    if not zero_stage_3:
        if global_rank == 0:
            torch.save(model_to_save.state_dict(), output_model_file)
    else:
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():
            if hasattr(v, "ds_id"):
                with deepspeed.zero.GatheredParameters(
                    _z3_params_to_fetch([v]), enabled=zero_stage_3
                ):
                    v_p = v.data.cpu()
            else:
                v_p = v.cpu()
            if global_rank == 0 and "lora" not in k:
                output_state_dict[k] = v_p
        if global_rank == 0:
            torch.save(output_state_dict, output_model_file)
        del output_state_dict


def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


def get_all_reduce_tensor(tensor):
    torch.distributed.all_reduce(tensor)
    return tensor


def eval_perplex(model, eval_dataloader, model_args, device, tokenizer=None):
    losses = 0
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    perplex_index = defaultdict(list)
    for step, batch in enumerate(eval_dataloader):        
        batch = to_device(batch, device)
        index_list = None
        if "index_list" in batch:
            index_list = copy.deepcopy(batch["index_list"])
            del batch["index_list"]
        if "end_question" in batch:
            end_question = copy.deepcopy(batch["end_question"])
            del batch["end_question"]
        if len(batch["input_ids"].shape) == 3:
            assert batch["input_ids"].shape[1] == 1
            batch["input_ids"] = batch["input_ids"][:, 0, :]
            batch["attention_mask"] = batch["attention_mask"][:, 0, :]
            batch["labels"] = batch["labels"][:, 0, :]
        with torch.no_grad():
            outputs = model(**batch, use_cache=not model_args.use_flash_attn)
        max_inds = torch.argmax(outputs.logits, dim=2)
        # if tokenizer is not None:
        #     print("question")
        #     print(tokenizer.decode(batch["input_ids"][0][: end_question[0]]))
        #     print("answer")
        #     print(tokenizer.decode(max_inds[0][end_question[0] : 64]))
        loss = outputs.loss        
        losses += loss.float()
        #loss_tmp = criterion(outputs["logits"].squeeze()[:-1], batch["labels"].squeeze()[1:])
        if index_list is not None:
            loss_list = criterion(
                outputs["logits"].squeeze()[:-1], batch["labels"].squeeze()[1:]
            )
            labels_list = batch["labels"].squeeze()[1:]
            res_dict = defaultdict(list)
            for i, (index, label) in enumerate(zip(index_list[0], labels_list)):
                if index == -1:
                    break
                if label != -100:
                    res_dict[index.item()].append(loss_list[i])
            res_dict = {k: sum(v) / len(v) for k, v in res_dict.items()}
            for k, v in res_dict.items():
                perplex_index[k].append(v)
        #import pdb
        #pdb.set_trace()
    losses = losses / (step + 1)
    torch.distributed.barrier()
    try:
        perplexity = torch.exp(losses+1e-8)
    except OverflowError:
        perplexity = float("inf")
    try:
        perplexity = get_all_reduce_mean(perplexity).item()
        loss = get_all_reduce_mean(losses+1e-8).item()
    except:
        pass
    res = {"perplexity": perplexity, "loss": loss}
    if index_list is not None:
        perplex_index = {
            k: torch.exp(sum(v) / len(v)) for k, v in perplex_index.items()
        }
        perplex_index = {
            k: get_all_reduce_mean(v).item() for k, v in perplex_index.items()
        }
        res.update({"perplex_index": perplex_index})
    return res




def eval_f1(model, eval_dataloader, model_args, device, tokenizer=None, num_tokens=256):
    predictions = []
    references = []
    questions = []
    indexes = []
    # squad_v2_metric = load("squad_v2")
    if "Mistral" in str(model):
        model = model.to(device)
    for k, batch in enumerate(eval_dataloader):
        batch = to_device(batch, device)
        if "end_question" in batch:
            end_question = copy.deepcopy(batch["end_question"])
            del batch["end_question"]
        if len(batch["input_ids"].shape) == 3:
            assert batch["input_ids"].shape[1] == 1
            batch["input_ids"] = batch["input_ids"][:, 0, :]
            batch["question_ids"] = batch["question_ids"][:, 0, :]
            batch["attention_mask"] = batch["attention_mask"][:, 0, :]
            batch["labels"] = batch["labels"][:, 0, :]
            # outputs = model(**batch, use_cache=not model_args.use_flash_attn)
        batch["attention_mask"] = batch["attention_mask"].to(batch["input_ids"].device)
        try:
            with torch.no_grad():
                if "Mistral" in str(model):
                    attn_mask = batch["attention_mask"].to(batch["input_ids"].device)[
                        :, : batch["input_ids"].shape[1]
                    ]
                else:
                    attn_mask = None
                outputs = model.generate(
                    input_ids=batch["input_ids"],
                    max_new_tokens=num_tokens,
                    do_sample=True,
                    num_beams=1,
                    top_p=0.9,
                    temperature=0.6,
                    use_cache=True,
                    top_k=50,
                    repetition_penalty=1.2,
                    attention_mask=attn_mask,
                )
            # print(outputs)
            for i, output in enumerate(outputs):
                eval_id = i + k * len(batch["input_ids"])
                answer = output[end_question[i] :]
                answer = tokenizer.decode(answer, skip_special_tokens=True)
                gt = tokenizer.decode(batch["answer"][i])
                gt = gt.replace("<s>", "")
                question = tokenizer.decode(batch["input_ids"][i])
                question = question.replace("<s>", "")
                answer = answer.replace("<s>", "")
                prediction = {
                    "prediction_text": answer,
                    "id": str(eval_id),
                    "no_answer_probability": 0.0,
                }
                reference = {
                    "answers": {"answer_start": [0], "text": [gt]},
                    "id": str(eval_id),
                }
                print("Answer:", answer, "|| GT: ", gt)
                predictions.append(prediction)
                references.append(reference)
                questions.append(question)
                if "index" in batch:
                    indexes.append(batch["index"][i].item())
                else:
                    indexes.append(0)
        except:
            print("failed")
    # results = squad_v2_metric.compute(predictions=predictions, references=references)
    # print(results)
    return (questions, predictions, references, indexes)


def eval_classification(model, eval_dataloader, model_args, device):
    metric = evaluate.combine(["accuracy"])
    for step, batch in enumerate(eval_dataloader):
        batch = to_device(batch, device)
        if len(batch["input_ids"].shape) == 3:
            assert batch["input_ids"].shape[1] == 1
            batch["input_ids"] = batch["input_ids"][:, 0, :]
            batch["attention_mask"] = batch["attention_mask"][:, 0, :]
            batch["labels"] = batch["labels"][:, 0, :]
        with torch.no_grad():
            outputs = model(**batch, use_cache=not model_args.use_flash_attn)
            predictions = outputs.logits.argmax(dim=-1)
            # predictions = get_all_reduce_tensor(predictions)
            # eferences = get_all_reduce_tensor(batch["labels"])
            # print(references)
            # print(predictions.shape)
            # print(batch["labels"].shape)
            # print(type(predictions))
            # print(type(batch['labels']))
            metric.add_batch(
                predictions=predictions,
                references=batch["labels"],
            )
    eval_metric = metric.compute()
    return eval_metric


def evaluation(model, eval_dataloader, model_args, device, task_type, tokenizer=None):
    model.eval()
    eval_func = eval_classification if task_type == "SEQ_CLS" else eval_perplex
    eval_func = text_generation if task_type == "Generation" else eval_func
    eval_func = eval_f1 if task_type in ["QA", "Multi-QA"] else eval_func
    num_tokens = 256 if task_type != "Multi-QA" else 1
    if tokenizer is not None and task_type in ["Generation", "QA", "Multi-QA"]:
        eval_func = partial(eval_func, tokenizer=tokenizer, num_tokens=num_tokens)
    result = eval_func(model, eval_dataloader, model_args, device, tokenizer=tokenizer)
    model.train()
    return result


def prepare_demodata(num_sample=10, num_cond_words=10):
    data = load_dataset(
        "json",
        data_files={
            "train": "/scratch/acf15494gz/moonshot_data/arxiv_data/conference_markdown.json"
        },
        cache_dir="/groups/gcf51104/data_cache/",
    )
    pick_data = random.choices(list(range(len(data["train"]))), k=num_sample)
    demo_data = []
    for choice in pick_data:
        title = data["train"]["title"][choice]
        abst = data["train"]["abst"][choice]
        abst = " ".join(abst.split(" ")[:num_cond_words])
        text = "Title: " + title + "Main Text: " + abst
        demo_data.append(text)

    return demo_data


def prepare_demodata_qa(num_sample=10, num_cond_words=10):
    data = load_dataset(
        "json",
        data_files={
            "train": "/scratch/acf15494gz/moonshot_data/arxiv_data/ai_arxiv_passage_question_answer.json"
        },
        cache_dir="/groups/gcf51104/data_cache/",
    )
    pick_data = random.choices(list(range(len(data["train"]))), k=num_sample)
    demo_data = []
    for choice in pick_data:
        example = data["train"]
        context = example["passage"][choice]
        question = example["question"][choice]
        demo_data.append(context + question)
    return demo_data
