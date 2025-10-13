import random
import warnings
import torch
import subprocess
import numpy as np
import deepspeed
from transformers import (
    default_data_collator,
)
from common.log import logger_rank0 as logger
from evaluate import load

warnings.filterwarnings("ignore")
from utils.eval_func import evaluation
from utils.model_utils import build_model
from utils.create_dataset import create_dataset
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from train import setup_parser
import os
import json

SYNTHTASK = [
            "birthday",
            "birthplace",
            "school",
            "major",
            "company",
            "occ",
            "sports",
            "food",
            "hobby",
        ]
WIKITASK =  ["val", "test"]


def write_eval_res(result, train_config):
    questions, predictions, references, indexes = result
    rank = torch.distributed.get_rank()
    model_name = train_config['model_name']
    output_dir = train_config['output_dir']
    task_name = train_config['task_name']
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "result"), exist_ok=True)
    print(os.path.join(output_dir, "result"))
    with open(
        os.path.join(
            output_dir,
            "result",
            model_name
            + "_"
            + task_name
            + f"_rank{rank}_"
            + 'result_output.json',
        ),
        "w",
    ) as f:
        for q, p, r, ind in zip(questions, predictions, references, indexes):
            write_content = {}
            write_content["question"] = q
            write_content["prediction"] = p
            write_content["reference"] = r
            write_content["index"] = ind
            json.dump(write_content, f)
            f.write("\n")
    torch.distributed.barrier()
    if rank == 0:
        pred_all = []
        ref_all = []
        offset = 0
        for index in range(torch.distributed.get_world_size()):
            path_output = os.path.join(
                    output_dir,
                    "result",
                    model_name
                    + "_"
                    + task_name
                    + f"_rank{index}_"
                    + 'result_output.json',
                )
            with open(path_output,"r") as f:
                data = [json.loads(line) for line in f]
            pred_add = [line["prediction"] for line in data]
            ref_add = [line["reference"] for line in data]
            for pred, ref in zip(pred_add, ref_add):
                pred["id"] = str(int(pred["id"]) + offset)
                ref["id"] = str(int(ref["id"]) + offset)
                pred_all.append(pred)
                ref_all.append(ref)
            print(index, len(pred_all), len(ref_all))
            offset += len(pred_add)
        squad_v2_metric = load("squad_v2")
        print("-----Overall Performance ------")
        res = squad_v2_metric.compute(predictions=pred_all, references=ref_all)
        print(res)
        with open(
            os.path.join(output_dir, "result", f"eval_{task_name}.json"),
            "w",
        ) as f:
            json.dump(res, f)
        # Define the command you want to run
        command = ["python", "utils/eval_index.py", path_output]
        # Execute the command
        result = subprocess.run(command, check=True,stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)    
        print(result.stdout)
        # Save the output to a file
        with open(os.path.join(output_dir, "result", f"eval_position_{task_name}.txt"), "w") as f:
            f.write(result.stdout)

def main():    
    trainer_args, train_config = setup_parser()    
    if trainer_args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(trainer_args.local_rank)
        device = torch.device("cuda", trainer_args.local_rank)
    deepspeed.init_distributed(dist_backend="nccl")
    torch.cuda.set_device(trainer_args.local_rank)
    random.seed(trainer_args.seed)
    np.random.seed(trainer_args.seed)
    torch.manual_seed(trainer_args.seed)    
    deepspeed.runtime.utils.set_random_seed(trainer_args.seed)
    tokenizer, model = build_model(trainer_args)
    batch_size = 1
    if "synth_language5" in train_config['qa_dataset']:
        task_list = [f"synth_language5_test_qa_{i}.jsonl" for i in range(5)]
    elif "wiki" in train_config['qa_dataset']:
        #task_list = WIKITASK
        task_list = ["film_qa_val.jsonl", "film_qa_test.jsonl"]
    elif "medquad" in train_config['qa_dataset']:
        task_list = ['']
    else:
        raise ValueError('no appropriate dataset available')
      
    for task_name in task_list:
        train_config['task_name'] = task_name
        train_config["path_qa_eval"] = task_name
        eval_dataset = create_dataset(tokenizer, train_config)
        if trainer_args.local_rank == -1:
            eval_sampler = SequentialSampler(eval_dataset)
            device = torch.device("cuda")
        else:
            eval_sampler = DistributedSampler(eval_dataset)
            torch.cuda.set_device(train_config['local_rank'])
        device = torch.device("cuda", train_config['local_rank'])
        eval_dataloader = DataLoader(
            eval_dataset,
            collate_fn=default_data_collator,
            sampler=eval_sampler,
            batch_size=batch_size,
            drop_last=True,
        )
        logger.info(f"Eval dataset with {len(eval_dataloader.dataset)} samples")
        print(f"start eval {task_name}")
        print(os.path.join(train_config['output_dir'], "result"))
        result = evaluation(
            model,
            eval_dataloader,
            device,
            train_config['task_type'],
            tokenizer,
        )
        write_eval_res(result, train_config)

    return


if __name__ == "__main__":
    main()
