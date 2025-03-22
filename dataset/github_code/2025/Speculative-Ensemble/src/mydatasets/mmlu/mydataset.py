import re
import pdb
from jinja2 import Template
from datasets import load_from_disk

from src.myutils.file import read_txt, load_json


class MyDataset:
    def __init__(self, size=False, use_fewshot=False, dataset_name="mmlu"):
        self.dataset_name = dataset_name
        self.size = size
        self.use_fewshot = use_fewshot
        self.TINY_NUM = 5
        self.SMALL_NUM = 200
        self.init_dataset()

    def init_dataset(self):
        # init dataset
        hf_dataset = load_from_disk(f"src/mydatasets/{self.dataset_name}/dataset")[
            "test"
        ]
        if self.size == "tiny":
            hf_dataset = hf_dataset.shuffle(seed=42).select(range(self.TINY_NUM))
        elif self.size == "small":
            hf_dataset = hf_dataset.shuffle(seed=42).select(range(self.SMALL_NUM))
        elif isinstance(self.size, int):
            hf_dataset = hf_dataset.shuffle(seed=42).select(range(self.size))
        self.datas = hf_dataset
        # generate prefixs
        if self.use_fewshot:
            self.prefixs = load_json(
                f"src/mydatasets/{self.dataset_name}/{self.dataset_name}-cot-claude-single.json"
            )

    def get_prompts(self):
        if self.use_fewshot:
            template = Template(
                read_txt(f"src/mydatasets/{self.dataset_name}/prompt_fewshot.txt")
            )
        else:
            template = Template(
                read_txt(f"src/mydatasets/{self.dataset_name}/prompt_zeroshot.txt")
            )
        prompts = []
        for data in self.datas:
            prompt = template.render(**self.modified_data(data))
            if self.use_fewshot:
                prefix = self.prefixs[data["subject"]]
                prompt = f"{prefix}\n\n{prompt}"
            prompts.append(prompt)
        return prompts

    def evaluate(self, preds):
        preds = [p.split("\n\nQ:")[0] for p in preds]
        choices = ["A", "B", "C", "D"]
        correct = 0
        for i, pred in enumerate(preds):
            ans_pred = self.find_answer(pred)
            gold = choices[self.datas[i]["answer"]]
            if ans_pred == gold:
                correct += 1
        result = float(correct) / len(preds)
        return result

    def find_answer(self, text):
        patterns = [
            r"\b([A-D])\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
        return "Z"

    def modified_data(self, data):
        choices = ["A", "B", "C", "D"]
        data["answer"] = choices[data["answer"]]
        choices_str = ""
        for char, value in zip(["A", "B", "C", "D"], data["choices"]):
            choices_str += f"({char}) {value}\n"
        data["choices_str"] = choices_str.strip()
        return data


if __name__ == "__main__":
    dataset = MyDataset(tiny=True)
