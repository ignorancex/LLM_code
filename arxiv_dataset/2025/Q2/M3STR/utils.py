import os
import json
import random


model_path_map = {
    "llava": {
        "llava-1.5-7b": "LlaVA/llava-1.5-7b-hf",
        "llava-llama3-8b": "LlaVA/llava-llama-3-8b-v1_1-transformers",
        "llava-v1.6-vicuna-7b": "LlaVA/llava-v1.6-vicuna-7b-hf"
    },
    "minicpm": {
        "minicpm-llama3-v-2.5": "MiniCPM/MiniCPM-Llama3-V-2_5",
        # bug with minicpm-v and minicpm-v-2
        "minicpm-v-2": "MiniCPM/MiniCPM-V-2",
        "minicpm-v-2.6": "MiniCPM/MiniCPM-V-2_6",
    },
    "qwen-vl": {
        "qwen2-vl-2b": "Qwen/Qwen2-VL-2B-Instruct",
        "qwen2-vl-7b": "Qwen/Qwen2-VL-7B-Instruct",
        "qwen2-vl-72b": "Qwen/Qwen2-VL-72B-Instruct-AWQ",
        "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
        "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
        "qwen2.5-vl-72b": "Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
    },
    # Not supported by vllm
    "instructblip": {
        "instructblip-13b": "InstructBLIP/instructblip-vicuna-13b",
        # "instructblip-7b": "InstructBLIP/instructblip-vicuna-7b"
    },
    "deepseek-vl": {
        "deepseek-vl-1.3b": "Deepseek/deepseek-vl-1.3b-chat",
        "deepseek-vl-7b": "Deepseek/deepseek-vl-7b-chat",
        "deepseek-vl2-small": "Deepseek/deepseek-vl2-small",
        "deepseek-vl2-tiny": "Deepseek/deepseek-vl2-tiny"
    },
    "internvl": {
        "internvl-2.5-1b": "InternVL/InternVL2_5-1B",
        "internvl-2.5-8b": "InternVL/InternVL2_5-8B"
    },
    "phi": {
        # "Phi-3-vision": "Phi/Phi-3-vision-128k-instruct",
        "Phi-3.5-vision": "Phi/Phi-3.5-vision-instruct"
    },
    "chameleon": {
        "chameleon-7b": "Others/chameleon-7b"
    }
}


task1_prompt_template_entity = """
We show a multi-modal knowledge graph in the input image. The entities are presented with their texts and images as nodes while the relations are edges. How many different entities are there in it? Please directly answer a number and do not return anything else.
"""

task1_prompt_template_relation = """
We show a multi-modal knowledge graph in the input image. The entities are presented with their texts and images as nodes while the relations are edges.  How many different relations are there in it? Please directly answer a number and do not return anything else.
"""

task2_prompt_template_entity = """
We show a multi-modal knowledge graph in the input image. The entities are presented with their texts and images as nodes while the relations are edges. Does any anomalies or factually incorrect entity (node) exist in the image? If exists answer Yes otherwise answer No. 
Please directly answer Yes or No and do not return anything else.
"""

task2_prompt_template_relation = """
We show a multi-modal knowledge graph in the input image. The entities are presented with their texts and images as nodes while the relations are edges. Does any anomalies or factually incorrect relation (edge) exist in the image? If exists answer Yes otherwise answer No. 
Please directly answer Yes or No and do not return anything else.
"""

task2_prompt_template_mix = """
We show a multi-modal knowledge graph in the input image. The entities are presented with their texts and images as nodes while the relations are edges. Does any anomalies or factually incorrect entity (node) or relation (edge) exist in the image? If exists answer Yes otherwise answer No. 
Please directly answer Yes or No and do not return anything else.
"""


task3_prompt_template_entity = """
We show a multi-modal knowledge graph in the input image. The entities are presented with their texts and images as nodes while the relations are edges. 
Please select a correct answer from the given choices for the missing entity (marked as ?????) in the image.
Your choices:
{}
Please directly answer the choice (A/B/C/D/E) and do not return anything else.
"""


task3_prompt_template_relation = """
We show a multi-modal knowledge graph in the input image. The entities are presented with their texts and images as nodes while the relations are edges. 
Please select a correct answer from the given choices for the missing relation (marked as ?????) in the image.
Your choices:
{}
Please directly answer the choice (A/B/C/D/E) and do not return anything else.
"""




def load_task1_dataset():
    task_data = json.load(open("dataset/task1_count.json", "r"))
    entity_prompts = []
    relation_prompts = []
    for data in task_data:
        if data["num_entity"] < 10:
            entity_prompts.append({
                "image_file": data["image_file"],
                "input": task1_prompt_template_entity,
                "answer": data["num_entity"]
            })
        relation_prompts.append({
            "image_file": data["image_file"],
            "input": task1_prompt_template_relation,
            "answer": data["num_relation"]
        })
    return entity_prompts, relation_prompts




def load_task2_dataset():
    task_data = json.load(open("dataset/task2_detection.json", "r"))
    entity_prompts = []
    relation_prompts = []
    mixed_prompts = []
    for data in task_data:
        entity_answer = "Yes" if data["modified_type"] == 'entity' else "No"
        entity_prompts.append({
            "image_file": data["image_file"],
            "input": task2_prompt_template_entity,
            "answer": entity_answer
        })
        relation_answer = "Yes" if data["modified_type"] == 'relation' else "No"
        relation_prompts.append({
            "image_file": data["image_file"],
            "input": task2_prompt_template_relation,
            "answer": relation_answer
        })
        mix_answer = "Yes" if data["modified_type"] != '' else "No"
        mixed_prompts.append({
            "image_file": data["image_file"],
            "input": task2_prompt_template_mix,
            "answer": mix_answer
        })
        # print(entity_answer, relation_answer, mix_answer)
    return {
        "entity": entity_prompts,
        "relation": relation_prompts,
        "mix": mixed_prompts
    }



def load_task3_dataset():
    entity_data = json.load(open("dataset/task3_entity_with_answer.json", "r"))
    relation_data = json.load(open("dataset/task3_relation_with_answer.json", "r"))
    return {
        "entity": entity_data,
        "relation": relation_data
    }



def load_task4_dataset():
    pass



def set_answers_entity():
    task_data = json.load(open("dataset/task3_entity_reason.json", "r"))
    entity_prompts = []
    choices = ["A", "B", "C", "D", "E"]
    for data in task_data:
        correct_answer = data["modified_element"]
        candidates = data["choices"]
        correct_choice = random.choice(choices)
        ordered_choices = []
        count = 0
        for choice in choices:
            if choice == correct_choice:
                ordered_choices.append("{}. {}".format(choice, correct_answer))
            else:
                ordered_choices.append("{}. {}".format(choice, candidates[count]))
                count += 1
        choice_prompt = "\n".join(ordered_choices)
        entity_prompts.append({
            "image_file": data["image_file"],
            "input": task3_prompt_template_entity.format(choice_prompt),
            "answer": correct_choice
        })
    json.dump(entity_prompts, open("dataset/task3_entity_with_answer.json", "w"), ensure_ascii=False)


def set_answers_relation():
    task_data = json.load(open("dataset/task4_relation_reason.json", "r"))
    entity_prompts = []
    choices = ["A", "B", "C", "D", "E"]
    for data in task_data:
        correct_answer = data["modified_element"]
        candidates = data["choices"]
        correct_choice = random.choice(choices)
        ordered_choices = []
        count = 0
        for choice in choices:
            if choice == correct_choice:
                ordered_choices.append("{}. {}".format(choice, correct_answer))
            else:
                ordered_choices.append("{}. {}".format(choice, candidates[count]))
                count += 1
        choice_prompt = "\n".join(ordered_choices)
        entity_prompts.append({
            "image_file": data["image_file"],
            "input": task3_prompt_template_relation.format(choice_prompt),
            "answer": correct_choice
        })
    json.dump(entity_prompts, open("dataset/task3_relation_with_answer.json", "w"), ensure_ascii=False)


if __name__ == "__main__":
    # entity_prompts, relation_prompts = load_task1_dataset()
    # print(len(entity_prompts), len(relation_prompts))
    # set_answers_relation()
    dataset = load_task3_dataset()
    print(dataset)

