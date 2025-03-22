import re
import pdb
from jinja2 import Template
from datasets import load_from_disk
import subprocess
import warnings

from src.myutils.file import read_txt, load_json


class MyDataset:
    def __init__(self, size="tiny", use_fewshot=False, dataset_name="humaneval"):
        self.dataset_name = dataset_name
        if use_fewshot:
            warnings.warn("Few-shot learning is not supported for humaneval dataset")
        self.size = size
        self.TINY_NUM = 5
        self.init_dataset()

    def init_dataset(self):
        hf_dataset = load_from_disk(f"src/mydatasets/{self.dataset_name}/dataset")[
            "test"
        ]
        if self.size == "tiny":
            hf_dataset = hf_dataset.select(range(self.TINY_NUM))
        elif isinstance(self.size, int):
            hf_dataset = hf_dataset.shuffle(seed=42).select(range(self.size))
        self.datas = hf_dataset

    def get_prompts(self):
        return self.datas["prompt"]

    def evaluate(self, preds):
        for i, p in enumerate(preds):
            preds[i] = self.clean_pred(p)
        n_correct = 0
        for i, pred in enumerate(preds):
            n_correct += self.check_correctness(self.datas[i], pred, timeout=5)
        return n_correct / len(preds)

    def check_correctness(self, problem, completion, timeout):
        func = problem["prompt"] + "    " + completion.strip()
        code = (
            func
            + "\n\n"
            + problem["test"]
            + "\n\n"
            + f'check({problem["entry_point"]})'
        )
        return not self.check_python_program_error(code)

    def check_python_program_error(self, python_program):
        try:
            result = subprocess.run(
                ["python", "-c", python_program],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode != 0
        except subprocess.TimeoutExpired:
            return True
        except Exception as e:
            return True

    def clean_pred(self, pred):
        clean_list = ["</s><s>", "\n\n\n", "\n\ndef"]
        for cl in clean_list:
            if cl in pred:
                pred = pred.split(cl)[0]
        return pred


if __name__ == "__main__":
    dataset = MyDataset(tiny=True)
