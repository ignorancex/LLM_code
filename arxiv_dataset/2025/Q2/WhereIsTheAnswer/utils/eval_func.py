import torch
import numpy as np
from transformers import set_seed, AutoTokenizer
import torch.nn as nn
import copy
import evaluate
from functools import partial
from collections import defaultdict
from utils.utils import get_all_reduce_mean, to_device


def text_generation(model, eval_dataloader, 
                    device, tokenizer=None, **kwargs):
    res = []
    correct = 0
    correct_original = 0
    count = 0
    for _, batch in enumerate(eval_dataloader):
        # print(batch.keys())
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
        with torch.no_grad():
            outputs = model.generate(
                input_ids=batch["input_ids"],
                max_new_tokens=64,
                do_sample=False,
                top_p=0.9,
                temperature=1.0,
                use_cache=True,
                top_k=3,
                repetition_penalty=1.0,
                length_penalty=1,
            )
        # print(outputs)
        for i, output in enumerate(outputs):
            output_text = tokenizer.decode(output, skip_special_tokens=True)
            q = tokenizer.decode(batch["input_ids"][i])
            gt = tokenizer.decode(batch["answer"][i])
            ans = output_text
            print("question")
            print(q)
            print("answer")
            print(ans)
            print("gt")
            print(gt)
            gt = gt.replace("<s>", "")
            gt = gt.replace(" ", "")

    print("accuracy: ", float(correct / count))
    print("accuracy with original: ", float(correct_original / count))
    return res


def eval_f1(model, eval_dataloader, device, tokenizer=None, num_tokens=256):
    predictions = []
    references = []
    questions = []
    indexes = []
    model = model.to(device)
    bos_token = tokenizer.bos_token
    for k, batch in enumerate(eval_dataloader):
        batch = to_device(batch, device)
        end_question = copy.deepcopy(batch["end_question"])
        del batch["end_question"]

        if len(batch["input_ids"].shape) == 3:
            assert batch["input_ids"].shape[1] == 1
            batch["input_ids"] = batch["input_ids"][:, 0, :]
            batch["question_ids"] = batch["question_ids"][:, 0, :]
            batch["attention_mask"] = batch["attention_mask"][:, 0, :]
            batch["labels"] = batch["labels"][:, 0, :]

        batch["attention_mask"] = batch["attention_mask"].to(batch["input_ids"].device)
        with torch.no_grad():
            if "Mistral" in str(model) or "Qwen" in str(model):
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
        for i, output in enumerate(outputs):
            eval_id = i + k * len(batch["input_ids"])
            answer = output[end_question[i] :]
            answer = tokenizer.decode(answer, skip_special_tokens=True)
            gt = tokenizer.decode(batch["answer"][i])            
            question = tokenizer.decode(batch["input_ids"][i])
            if bos_token is not None:
                gt = gt.replace(bos_token, "")
                question = question.replace(bos_token, "")
                answer = answer.replace(bos_token, "")
            prediction = {
                "prediction_text": answer,
                "id": str(eval_id),
                "no_answer_probability": 0.0,
            }
            reference = {
                "answers": {"answer_start": [0], "text": [gt]},
                "id": str(eval_id),
            }
            print("question:", question ,"Answer:", answer, "|| GT: ", gt)
            predictions.append(prediction)
            references.append(reference)
            questions.append(question)        
            if "index" in batch:
                indexes.append(batch["index"][i].item())
            else:
                indexes.append(0)
    return (questions, predictions, references, indexes)



def eval_perplex(model, eval_dataloader, device, tokenizer=None, **kwargs):
    losses = 0
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    perplex_index = defaultdict(list)
    model = model.to(device)
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
            outputs = model(**batch)
        #max_inds = torch.argmax(outputs.logits, dim=2)
        loss = outputs.loss        
        losses += loss.float()
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



def evaluation(model, eval_dataloader, device, task_type, tokenizer=None):
    model.eval()
    if task_type == "perplex": 
        eval_func = eval_perplex
    elif task_type == "QA":
        eval_func = eval_f1
    else:
        eval_func = text_generation
    num_tokens = 256
    result = eval_func(model, eval_dataloader, device, tokenizer=tokenizer, num_tokens=num_tokens)
    model.train()
    return result


