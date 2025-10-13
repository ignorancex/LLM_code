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
    Provide your reasoning and then give with the letter of the correct answer formatted as: $\\boxed{X}$, where X is 'A', 'B', 'C', or 'D'.
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
                if response.startswith("Error: "):
                    print(response)
                    continue
                res_item = {
                    "response": response,
                    "prompt": prompt_batch[i].split("<multiple choice question start>")[1].split("<multiple choice question end>")[0].strip(),
                    "image_path": image_path_batch[i],
                    "_id": _id_batch[i]
                }
                f.write(json.dumps(res_item) + "\n")
                f.flush()


# \\boxed{X} format
def parse_reasoning_response(res_path, parsed_path, error_path):
    with open(res_path, 'r') as f:
        res = [json.loads(line) for line in f.readlines()]
    parsed = []
    error = []

    for item in res:
        response = item["response"]
        try: # \\boxed{X}
            Answer = response.split("\\boxed{")[-1].split("}")[0]
            if Answer not in ["A", "B", "C", "D"]:
                if Answer.startswith("A. ") or Answer.startswith("B. ") or Answer.startswith("C. ") or Answer.startswith("D. "):
                    Answer = Answer[0]
        except:
            pass
        if Answer not in ["A", "B", "C", "D"]:
            error.append(item)
            print(item["response"])
        item["response"] = Answer
        item["reasoning"] = response
        parsed.append(item)

    with open(error_path, 'w') as f:
        json.dump(error, f, indent=2)
    with open(parsed_path, 'w') as f:
        for item in parsed:
            f.write(json.dumps(item) + "\n")
            f.flush()
    print(f"{len(error)} errors.")

