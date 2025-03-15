import re
import pdb
from jinja2 import Template
from datasets import load_from_disk

from src.myutils.file import read_txt


class MyDataset:
    def __init__(self, size="tiny", use_fewshot=False, dataset_name="gsm8k"):
        self.dataset_name = dataset_name
        self.size = size
        self.use_fewshot = use_fewshot
        self.TINY_NUM = 5
        self.SMALL_NUM = 200
        self.init_dataset()

    def init_dataset(self):
        hf_dataset = load_from_disk(f"src/mydatasets/{self.dataset_name}/dataset")[
            "test"
        ]
        if self.size == "tiny":
            hf_dataset = hf_dataset.select(range(self.TINY_NUM))
        elif self.size == "small":
            hf_dataset = hf_dataset.shuffle(seed=42).select(range(self.SMALL_NUM))
        elif isinstance(self.size, int):
            hf_dataset = hf_dataset.shuffle(seed=42).select(range(self.size))
        self.datas = hf_dataset

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
            prompt = template.render(**data)
            prompts.append(prompt)
        return prompts

    def evaluate(self, preds):
        preds = [p.split("\n\nQuestion:")[0] for p in preds]
        correct = 0
        for i, pred in enumerate(preds):
            pred_ans = self.find_answer(pred)
            ground_ans = self.find_answer(self.datas[i]["answer"])
            if pred_ans == ground_ans:
                correct += 1
        result = float(correct) / len(preds)
        return result

    def find_answer(self, text):
        match = re.search(r"###\s*(-?\d+)", text.replace(",", ""))
        if match:
            return round(float(match.group(1)))
        else:
            all_m = re.findall(r"(?<!\d)-?\d+(?:\.\d+)?", text.replace(",", ""))
            if all_m:
                return round(float(all_m[-1]))
        return "No answer found"


if __name__ == "__main__":
    dataset = MyDataset(tiny=True)
