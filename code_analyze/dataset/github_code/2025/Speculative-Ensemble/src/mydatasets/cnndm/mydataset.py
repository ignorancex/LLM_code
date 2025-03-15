import re
import pdb
from jinja2 import Template
from datasets import load_from_disk
import evaluate

from src.myutils.file import read_txt


class MyDataset:
    def __init__(self, size="tiny", use_fewshot=False, dataset_name="cnndm"):
        self.dataset_name = dataset_name
        self.size = size
        self.use_fewshot = use_fewshot
        self.TINY_NUM = 5
        self.SMALL_NUM = 200
        self.init_dataset()
        self.rouge = evaluate.load("./src/mydatasets/cnndm/rouge")

    def init_dataset(self):
        hf_dataset = load_from_disk(f"src/mydatasets/{self.dataset_name}/dataset")
        if self.size == "tiny":
            hf_dataset = hf_dataset.select(range(self.TINY_NUM))
        elif self.size == "small":
            hf_dataset = hf_dataset.shuffle(seed=42).select(range(self.SMALL_NUM))
        elif isinstance(self.size, int):
            hf_dataset = hf_dataset.shuffle(seed=42).select(range(self.size))
        self.datas = hf_dataset
        self.references = [data["highlights"] for data in self.datas]

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

    def evaluate(self, predictions):
        split_str = "\n\nArticle:"
        predictions = [re.split(split_str, p)[0] for p in predictions]
        results = self.rouge.compute(
            predictions=predictions, references=self.references
        )
        return results


if __name__ == "__main__":
    dataset = MyDataset(tiny=True)
