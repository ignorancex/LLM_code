# Write function to preprocess data from different sources and save them as json in a specific format

import json
import torch
import os
from datasets import load_dataset
from collections import Counter
import random
from datasets import concatenate_datasets
from tqdm import tqdm

from src import InferenceModule, DPOCurationModule
from data import CoTDataset, CoTRationaleGenerationDataset, DPOCurationDataset
from torch.utils.data import DataLoader

from transformers import AutoModelForCausalLM, AutoTokenizer
import pytorch_lightning as pl


def get_merged_list(args, device, data_dict, cache_path):
    with open(f"{cache_path}_{device}.json", "r") as f:
        data = json.load(f)
    if data_dict is None:
        data_dict = {key: [] for key in list(data.keys())}
    for key in list(data.keys()):
        data_dict[key].extend(data[key])
    return data_dict
    

def merge_data(args, cache_path):
    merged_data = None
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    for device in devices:
        if os.path.exists(f"{cache_path}_{device}.json"):
            merged_data = get_merged_list(args, device, merged_data, cache_path)
        else:
            return    
    with open(f"{cache_path}.json", "w") as f:
        json.dump(merged_data, f)
    for device in devices:
        os.remove(f"{cache_path}_{device}.json")


def preprocess_data(args):

    if args.dataset_name == "CoT":
        def sample_from_task(dataset, task_name, n_samples):
            task_subset = dataset.filter(lambda x: x['task'] == task_name)
            sampled_indices = random.sample(range(len(task_subset)), n_samples)
            return task_subset.select(sampled_indices)

        dataset = load_dataset(args.hf_dataset_path, args.hf_dataset_config, trust_remote_code=True)
        data = dataset['train']

        tasks_list = sorted(list(set(list(data["task"]))))

        counter = Counter(list(data["task"]))
        
        # Extract counts for the strings in the set
        tasks_count = {task: counter[task] for task in tasks_list}

        tasks_count_map = {"task": list(tasks_count.keys()), "data_count": list(tasks_count.values())}

        percentage = 0.163227286379 # percentage to sample exactly 300k samples from CoT collection (out of 1840k samples)
        sample_counts = {task: int(count * percentage) if int(count * percentage) != 0 else count for task, count in tasks_count.items()}
        
        sampled_datasets = [sample_from_task(data, task, n_samples) for idx, (task, n_samples) in tqdm(enumerate(sample_counts.items()), position = 0, leave = True, total = len(list(sample_counts.keys())))]

        sampled_dataset = concatenate_datasets(sampled_datasets)
        sampled_dataset = sampled_dataset.shuffle(seed = args.seed)

        sampled_indices = [i for i in range(len(sampled_dataset["source"]))]
        random.shuffle(sampled_indices)
        sampled_indices = sampled_indices[:300000]

        # Write code to use 100k samples for rationale refinement task
        # Use 100k samples for training
        rationale_refinement_dataset = sampled_dataset.select(sampled_indices[:100000])
        # rationale_refinement_dataset = sampled_dataset.select(sampled_indices[:200])
        cot_rationale_generation_dataset = CoTRationaleGenerationDataset(args, rationale_refinement_dataset)
        dataloader = DataLoader(cot_rationale_generation_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False, collate_fn=cot_rationale_generation_dataset.collate_fn, num_workers=args.num_workers)

        model = AutoModelForCausalLM.from_pretrained(args.hf_model_path, token = args.hf_token, torch_dtype = torch.bfloat16)
        if args.ckpt_path != "":
            model.load_state_dict(torch.load(args.ckpt_path, map_location = "cpu"))
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, token=args.hf_token)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = args.padding_side

        devices = [i for i in range(torch.cuda.device_count())]

        inference_module = InferenceModule(model=model, tokenizer=tokenizer, generation_mode="rationale", cache_path="./dataset/warmup_ift_llama3_8b_cot_dataset", args=args)
        trainer = pl.Trainer(accelerator="gpu", strategy="deepspeed_stage_2_offload", devices=devices, precision="bf16-true")

        trainer.predict(inference_module, dataloaders=dataloader)
        merge_data(args, "./dataset/warmup_ift_llama3_8b_cot_dataset")


        with open("./dataset/warmup_ift_llama3_8b_cot_dataset.json", "r") as f:
            data = json.load(f)

        data["instructions"].extend(sampled_dataset.select(sampled_indices[100000:])["source"])
        data["rationales"].extend([""] * (len(sampled_dataset) - 100000))
        data["gt_rationales"].extend(sampled_dataset.select(sampled_indices[100000:])["rationale"])
        data["responses"].extend(sampled_dataset.select(sampled_indices[100000:])["target"])
        data["response_types"].extend(["rationale_conditioned_response"] * (100000))
        data["response_types"].extend(["rationale"] * (len(sampled_dataset) - 200000))

        natural_instructions_dataset = load_dataset("jayelm/natural-instructions")
        natural_instructions_dataset = natural_instructions_dataset["train"]
        task_names = ["mmlu", "arc_", "math", "commonsense", "nli_answer_generation", "hellaswag", "piqa_answer_generation", "bigbench"]
        filtered_dataset1 = natural_instructions_dataset.filter(lambda example: task_names[0] in example['task_name'])
        filtered_dataset2 = natural_instructions_dataset.filter(lambda example: task_names[1] in example['task_name'])
        filtered_dataset3 = natural_instructions_dataset.filter(lambda example: task_names[2] in example['task_name'])
        filtered_dataset4 = natural_instructions_dataset.filter(lambda example: task_names[3] in example['task_name'])
        filtered_dataset5 = natural_instructions_dataset.filter(lambda example: task_names[4] in example['task_name'])
        filtered_dataset6 = natural_instructions_dataset.filter(lambda example: task_names[5] in example['task_name'])
        filtered_dataset7 = natural_instructions_dataset.filter(lambda example: task_names[6] in example['task_name'])
        filtered_dataset8 = natural_instructions_dataset.filter(lambda example: task_names[7] in example['task_name'])

        natural_instructions_dataset = concatenate_datasets([filtered_dataset1, filtered_dataset2, filtered_dataset3, filtered_dataset4, filtered_dataset5, filtered_dataset6, filtered_dataset7, filtered_dataset8])

        natural_instructions_dataset = natural_instructions_dataset.shuffle(seed = args.seed)
        natural_instructions_dataset = natural_instructions_dataset.select(range(100000))
        # natural_instructions_dataset = natural_instructions_dataset[:100000]
        def concatenate_natural_instructions_columns(example):
            example['instructions'] = example['definition'] + '\nQuestion: ' + example['inputs']
            return example

        # Apply the function to the dataset
        natural_instructions_dataset = natural_instructions_dataset.map(concatenate_natural_instructions_columns)

        # natural_instructions_dataset_instructions = [natural_instructions_dataset["definition"][i] + "\nQuestion: " + natural_instructions_dataset["inputs"][i] for i in tqdm(range(len(natural_instructions_dataset)))]
        natural_instructions_dataset_instructions = natural_instructions_dataset["instructions"]
        natural_instructions_dataset_responses = natural_instructions_dataset["targets"]

        data["instructions"].extend(natural_instructions_dataset_instructions)
        data["rationales"].extend([""] * len(natural_instructions_dataset))
        data["gt_rationales"].extend([""] * len(natural_instructions_dataset))
        data["responses"].extend(natural_instructions_dataset_responses)
        data["response_types"].extend(["response"] * len(natural_instructions_dataset))

        aqua_rat_dataset = load_dataset("deepmind/aqua_rat")
        aqua_rat_dataset = aqua_rat_dataset["train"]
        aqua_rat_dataset = aqua_rat_dataset.shuffle(seed = args.seed)
        aqua_rat_dataset = aqua_rat_dataset.select(range(30000))

        # print(len(aqua_rat_dataset))
        def concatenate_aqua_rat_instructions_columns(example):
            example['instructions'] = "Given a high school level math problem, you have to solve it and choose the correct option.\nQuestion: " + example['question'] + " " + " ".join(example['options'])
            return example

        # Apply the function to the dataset
        aqua_rat_dataset = aqua_rat_dataset.map(concatenate_aqua_rat_instructions_columns)

        # aqua_rat_dataset_instructions = ["Given a high school level math problem, you have to solve it and choose the correct option.\nQuestion: " + aqua_rat_dataset["question"][i] + " " + " ".join(aqua_rat_dataset["options"][i]) for i in tqdm(range(len(aqua_rat_dataset)))]

        aqua_rat_dataset_instructions = aqua_rat_dataset["instructions"]
        aqua_rat_dataset_responses = aqua_rat_dataset["rationale"]
        aqua_rat_dataset_rationales = [""] * len(aqua_rat_dataset)
        aqua_rat_dataset_gt_rationales = [""] * len(aqua_rat_dataset)
        aqua_rat_dataset_response_types = ["response"] * len(aqua_rat_dataset)


        data["instructions"].extend(aqua_rat_dataset_instructions)
        data["rationales"].extend(aqua_rat_dataset_rationales)
        data["gt_rationales"].extend(aqua_rat_dataset_gt_rationales)
        data["responses"].extend(aqua_rat_dataset_responses)
        data["response_types"].extend(aqua_rat_dataset_response_types)

        with open("./dataset/warmup_ift_llama3_8b_dataset.json", "w") as f:
            json.dump(data, f)

    elif args.dataset_name == "stage1-CoT-dpo":

        dataset = load_dataset(args.hf_dataset_path, args.hf_dataset_config, trust_remote_code=True)
        data = dataset['train']

        tasks_list = sorted(list(set(list(data["task"]))))
        counter = Counter(list(data["task"]))

        tasks_count = {task: counter[task] for task in tasks_list}

        tasks_count_map = {"task": list(tasks_count.keys()), "data_count": list(tasks_count.values())}

        percentage = 0.163227286379 # percentage to sample exactly 300k samples from CoT collection (out of 1840k samples)
        
        sample_counts = {task: int(count * percentage) if int(count * percentage) != 0 else count for task, count in tasks_count.items()}

        sampled_datasets = [sample_from_task(data, task, n_samples) for idx, (task, n_samples) in tqdm(enumerate(sample_counts.items()), position = 0, leave = True, total = len(list(sample_counts.keys())))]
        sampled_dataset = concatenate_datasets(sampled_datasets)
        sampled_dataset = sampled_dataset.shuffle(seed = args.seed)

        sampled_indices = [i for i in range(len(sampled_dataset["source"]))]
        random.shuffle(sampled_indices)

        # Use 300k samples from CoT dataset for training

    # NOTE: Filtering the dataset based on the two masters, to curate the dpo dataset
    elif args.dataset_name == "task-specific-dpo":

        with open(args.dpo_turn1_task_dataset1, "r") as f:
            turn1_task_dataset1 = json.load(f)
        
        with open(args.dpo_turn2_task_dataset1, "r") as f:
            turn2_task_dataset1 = json.load(f)

        with open(args.dpo_turn1_task_dataset2, "r") as f:
            turn1_task_dataset2 = json.load(f)

        with open(args.dpo_turn2_task_dataset2, "r") as f:
            turn2_task_dataset2 = json.load(f)


        instruction_mapping_turn1_task_dataset1 = {}
        for idx, instruction in enumerate(turn1_task_dataset1["instructions"]):
            instruction_mapping_turn1_task_dataset1[instruction] = {
                "rationale": turn1_task_dataset1["rationales"][idx], "response": turn1_task_dataset1["responses"][idx],
                "response_type": turn1_task_dataset1["response_types"][idx], "gt_rationale": turn1_task_dataset1["gt_rationales"][idx]
            }

        instruction_mapping_turn1_task_dataset2 = {}
        for idx, instruction in enumerate(turn1_task_dataset2["instructions"]):
            instruction_mapping_turn1_task_dataset2[instruction] = {
                "rationale": turn1_task_dataset2["rationales"][idx], "response": turn1_task_dataset2["responses"][idx],
                "response_type": turn1_task_dataset2["response_types"][idx], "gt_rationale": turn1_task_dataset2["gt_rationales"][idx]
            }

        instruction_mapping_turn2_task_dataset1 = {}
        for idx, instruction in enumerate(turn2_task_dataset1["instructions"]):
            instruction_mapping_turn2_task_dataset1[instruction] = {
                "rationale": turn2_task_dataset1["rationales"][idx], "response": turn2_task_dataset1["responses"][idx],
                "response_type": turn2_task_dataset1["response_types"][idx], "gt_rationale": turn2_task_dataset1["gt_rationales"][idx],
                "refined_rationale": turn2_task_dataset1["refined_rationales"][idx]
            }

        instruction_mapping_turn2_task_dataset2 = {}
        for idx, instruction in enumerate(turn2_task_dataset2["instructions"]):
            instruction_mapping_turn2_task_dataset2[instruction] = {
                "rationale": turn2_task_dataset2["rationales"][idx], "response": turn2_task_dataset2["responses"][idx],
                "response_type": turn2_task_dataset2["response_types"][idx], "gt_rationale": turn2_task_dataset2["gt_rationales"][idx],
                "refined_rationale": turn2_task_dataset2["refined_rationales"][idx]
            }

        curation_data_dict = {"instructions": [], "responses": [], "turn1_rationale1": [], "turn1_rationale2": [], "turn2_rationale1": [], "turn2_rationale2": [], "response_types": []}
        

        # NOTE: There are two guides, so split the dataset into 2 subsets for each guide
        # Once the dataset is split, we need to split each subset again into two parts since there are two types of samples to be curated
        # The first type is the one where given an instruction, the guide has to generate rationale, and there are two candidates for that
        # The second type is the one where given an instruction and rationale generated by the specified guide index, both the guides need to refine that rationale
        # We store this data in curation_data_dict

        instructions = list(instruction_mapping_turn1_task_dataset1.keys())
        if args.guide_index == 1:
            for i in range(len(instructions) // 4):
                curation_data_dict["instructions"].append(instructions[i])
                curation_data_dict["responses"].append(instruction_mapping_turn1_task_dataset1[instructions[i]]["response"])
                curation_data_dict["turn1_rationale1"].append(instruction_mapping_turn1_task_dataset1[instructions[i]]["rationale"])
                curation_data_dict["turn1_rationale2"].append(instruction_mapping_turn1_task_dataset2[instructions[i]]["rationale"])
                curation_data_dict["turn2_rationale1"].append("")
                curation_data_dict["turn2_rationale2"].append("")
                curation_data_dict["response_types"].append(instruction_mapping_turn1_task_dataset1[instructions[i]]["response_type"])
            for i in range(len(instructions) // 4, 2 * len(instructions) // 4):
                curation_data_dict["instructions"].append(instructions[i])
                curation_data_dict["responses"].append(instruction_mapping_turn1_task_dataset1[instructions[i]]["response"])
                curation_data_dict["turn1_rationale1"].append(instruction_mapping_turn1_task_dataset1[instructions[i]]["rationale"])
                curation_data_dict["turn1_rationale2"].append(instruction_mapping_turn1_task_dataset2[instructions[i]]["rationale"])
                curation_data_dict["turn2_rationale1"].append(instruction_mapping_turn2_task_dataset1[instructions[i]]["refined_rationale"])
                curation_data_dict["turn2_rationale2"].append(instruction_mapping_turn2_task_dataset2[instructions[i]]["refined_rationale"])
                curation_data_dict["response_types"].append(instruction_mapping_turn2_task_dataset1[instructions[i]]["response_type"])

        elif args.guide_index == 2:
            for i in range(2 * len(instructions) // 4, 3 * len(instructions) // 4):
                curation_data_dict["instructions"].append(instructions[i])
                curation_data_dict["responses"].append(instruction_mapping_turn1_task_dataset1[instructions[i]]["response"])
                curation_data_dict["turn1_rationale1"].append(instruction_mapping_turn1_task_dataset1[instructions[i]]["rationale"])
                curation_data_dict["turn1_rationale2"].append(instruction_mapping_turn1_task_dataset2[instructions[i]]["rationale"])
                curation_data_dict["turn2_rationale1"].append("")
                curation_data_dict["turn2_rationale2"].append("")
                curation_data_dict["response_types"].append(instruction_mapping_turn1_task_dataset1[instructions[i]]["response_type"])
            for i in range(3 * len(instructions) // 4, len(instructions)):
                curation_data_dict["instructions"].append(instructions[i])
                curation_data_dict["responses"].append(instruction_mapping_turn1_task_dataset1[instructions[i]]["response"])
                curation_data_dict["turn1_rationale1"].append(instruction_mapping_turn1_task_dataset1[instructions[i]]["rationale"])
                curation_data_dict["turn1_rationale2"].append(instruction_mapping_turn1_task_dataset2[instructions[i]]["rationale"])
                curation_data_dict["turn2_rationale1"].append(instruction_mapping_turn2_task_dataset1[instructions[i]]["refined_rationale"])
                curation_data_dict["turn2_rationale2"].append(instruction_mapping_turn2_task_dataset2[instructions[i]]["refined_rationale"])
                curation_data_dict["response_types"].append(instruction_mapping_turn2_task_dataset1[instructions[i]]["response_type"])
        
        dataset = DPOCurationDataset(curation_data_dict, args, split="train")
        dataloader = DataLoader(dataset, batch_size=args.per_device_eval_batch_size, shuffle=False, collate_fn=dataset._collate_fn, num_workers=args.num_workers)

        model1 = AutoModelForCausalLM.from_pretrained(args.hf_model_path, token = args.hf_token)
        model1.load_state_dict(torch.load(args.ckpt_path1, map_location = "cpu"))

        model2 = AutoModelForCausalLM.from_pretrained(args.hf_model_path, token = args.hf_token)
        model2.load_state_dict(torch.load(args.ckpt_path2, map_location = "cpu"))

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, token=args.hf_token)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = args.padding_side

        cache_path = os.path.join("./dataset", f"dpo_dataset_{args.task_name}_{args.model_name}_stage{args.stage_index}_guide{args.guide_index}")
        dpo_curation_module = DPOCurationModule(model1, model2, tokenizer, args, cache_path=cache_path)

        devices = [i for i in range(torch.cuda.device_count())]
        trainer = pl.Trainer(accelerator="gpu", strategy="ddp", devices=devices)

        trainer.validate(dpo_curation_module, dataloaders=dataloader)
        merge_data(args, cache_path)
