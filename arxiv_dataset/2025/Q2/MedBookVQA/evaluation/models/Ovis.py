# import sys
# sys.path.append("/home/sunanhe/fyp2024slyipae/FYP2024_VQAgeneration/DOAB/MCeval/models/ovis")
from PIL import Image
from ovis.serve.runner import RunnerArguments, OvisRunner


class Ovis:
    def __init__(self, args):
        self.args = args
        runner_args = RunnerArguments(model_path=args.model_path)
        self.model = OvisRunner(runner_args)

    def inference(self, prompt_batch, image_path_batch):
        output_batch = []
        for image_path, prompt in zip(image_path_batch, prompt_batch):
            image = Image.open(image_path)
            output = self.model.run([image, prompt])
            output = output["output"]
            output_batch.append(output)
        return output_batch
    




######################################################################################################

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


template1 = """
Please answer the following multiple choice question based on the image above.
    <multiple choice question start>
    
"""
template2 = """
    <multiple choice question end>

    Respond only with the letter of the correct answer, do not include specific content of the choice.
    Respond ONLY with 'A', 'B', 'C', or 'D'.
    """

def get_device(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def format_input(item):
    choice_content = [item["correct_choice"] + ". " + item["Answer"]]
    for i in range(3):
        choice_content.append(item["other_choices"][i] + ". " + item["Distractors"][i])

    choice_content = sorted(choice_content)
    choice_content = " | ".join(choice_content)

    input = f"""
    Question: {item["Question"]}
    Choices : {choice_content}
    """
    return input

def format_prompt(input):
    prompt = "<image>\n"
    prompt += f"""
    Please answer the following multiple choice question based on the image above.
    <multiple choice question start>
    {input}
    <multiple choice question end>

    Respond only with the letter of the correct answer, do not include specific content of the choice.
    Respond ONLY with 'A', 'B', 'C', or 'D'.
    """
    return prompt

def format_prompt_reasoning(input):
    prompt = "<image>\n"
    prompt += f"""
    Please answer the following multiple choice question based on the image above.
    <multiple choice question start>
    {input}
    <multiple choice question end>
    """
    prompt += """
    Please solve the problem step by step.
    Provide your reasoning and then give with the letter of the correct answer formatted as: [ANSWER: X], where X is 'A', 'B', 'C', or 'D'.
    You response sould be in the following format: Your reasoning\n\n[ANSWER: X].
    """
    return prompt

class dataset_MC(Dataset):
    def __init__(self, data, reasoning=False):
        self.data = data
        self.reasoning = reasoning

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        _id = item["_id"]
        image_path = item["image_path"]
        input = format_input(item)
        if self.reasoning:
            prompt = format_prompt_reasoning(input)
        else:
            prompt = format_prompt(input)

        return prompt, image_path, _id
    
def evaluate_MC(all_MC, res_path, model, args):
    dataset = dataset_MC(all_MC, reasoning=args.reasoning)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    
    with open(res_path, 'a') as f:
        for batch in tqdm(dataloader):
            prompt_batch, image_path_batch, _id_batch = batch
            response_batch = model.inference(prompt_batch, image_path_batch)

            for i, response in enumerate(response_batch):
                res_item = {
                    "response": response,
                    "prompt": prompt_batch[i].split("<multiple choice question start>")[1].split("<multiple choice question end>")[0].strip(),
                    "image_path": image_path_batch[i],
                    "_id": _id_batch[i]
                }
                f.write(json.dumps(res_item) + "\n")
                f.flush()


def parse_reasoning_response(res_path, parsed_path, error_path):
    with open(res_path, 'r') as f:
        res = [json.loads(line) for line in f.readlines()]
    parsed = []
    error = []

    for item in res:
        response = item["response"]
        try: # [ANSWER: A]
            Answer = response.split("[ANSWER: ")[1].split("]")[0]
            if Answer not in ["A", "B", "C", "D"]:
                if Answer.startswith("A. ") or Answer.startswith("B. ") or Answer.startswith("C. ") or Answer.startswith("D. "):
                    Answer = Answer[0]
                else:
                    error.append(item)
                    print(Answer)
            item["response"] = Answer
            item["reasoning"] = response
            parsed.append(item)
        except:
            try: # ANSWER: A
                Answer = response.split("ANSWER:")[1].strip()
                if Answer not in ["A", "B", "C", "D"]:
                    if Answer.startswith("A. ") or Answer.startswith("B. ") or Answer.startswith("C. ") or Answer.startswith("D. "):
                        Answer = Answer[0]
                    else:
                        error.append(item)
                        print(Answer)
                item["response"] = Answer
                item["reasoning"] = response
                parsed.append(item)
            except:
                try: # The correct answer is A (format error in InternVL2)
                    Answer = response.split("he correct answer is ")[1]
                    Answer = Answer[0]
                    if Answer not in ["A", "B", "C", "D"]:
                        error.append(item)
                        print(Answer)
                    item["response"] = Answer
                    item["reasoning"] = response
                    parsed.append(item)
                except:
                    try: # **Answer: A (format error in InternVL2_5)
                        Answer = response.split("\n\n**Answer: ")[1]
                        Answer = Answer[0]
                        if Answer not in ["A", "B", "C", "D"]:
                            error.append(item)
                            print(Answer)
                        item["response"] = Answer
                        item["reasoning"] = response
                        parsed.append(item)
                    except:
                        error.append(item)
                        # print("ERROR")
                        item["response"] = "ERROR"
                        item["reasoning"] = response
                        parsed.append(item)

    with open(error_path, 'w') as f:
        json.dump(error, f, indent=2)
    with open(parsed_path, 'w') as f:
        json.dump(parsed, f, indent=2)
    print(f"{len(error)} errors.")












#############################################################################################################
import os
import torch
import time
import json

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess PDF files for VQA generation.")
    parser.add_argument('--workspace', type=str, required=True, help='Path to workspace folder')
    parser.add_argument('--cuda_visible_devices', type=str, default='0', help='CUDA visible devices')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--HealthGPT_ViT_path', type=str, default='', help='Path to the VIT model for HealthGPT')
    parser.add_argument('--model_format', type=str, default='awq', help='Model format')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for DataLoader')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for DataLoader')
    parser.add_argument('--reasoning', action='store_true', help='Whether to generate reasoning response')
    parser.add_argument('--parse_reasoning', action='store_true', help='Whether to ONLY parse reasoning response and skip the inference')
    return parser.parse_args()
    
def main():
    args = parse_args()
    reasoning = args.reasoning
    print(args.parse_reasoning)
    print(args.reasoning)

    workspace = args.workspace
    with open(os.path.join(workspace, "all_MC.json"), "r") as f:
        all_MC = json.load(f)

    if not reasoning:
        res_path = os.path.join(workspace, f"MCresults_{args.model_path.split('/')[-1]}.jsonl")
    else:
        res_path = os.path.join(workspace, f"reasoning_MCresults_{args.model_path.split('/')[-1]}.jsonl")
    if not os.path.exists(res_path):
        with open(res_path, 'w') as f:
            pass
    with open(res_path, 'r') as f:
        finished = [json.loads(line) for line in f.readlines()]
        finished = [item["_id"] for item in finished]
    all_MC = [item for item in all_MC if item["_id"] not in finished]
    if reasoning:
        include_types = ["Symptom Recognition", "Surgery & Operation", "Disease Diagnosis"]
        all_MC = [item for item in all_MC if item["QAtype"] in include_types]

    print(f"Number of questions to evaluate: {len(all_MC)}")

    # perform evaluation inference
    print(args.cuda_visible_devices)
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Device: {device}")
    model = Ovis(args)
            

    evaluate_MC(all_MC, res_path, model, args)



if __name__ == "__main__":
    main()