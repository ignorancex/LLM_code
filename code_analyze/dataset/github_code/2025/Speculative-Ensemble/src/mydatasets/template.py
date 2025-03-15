from jinja2 import Template

from src.myutils.file import read_txt

class DatasetWrapper:
    def __init__(self, size="tiny", use_fewshot=False, dataset_name="your_dataset"):
        self.dataset_name = dataset_name
        self.size = size
        self.use_fewshot = use_fewshot
        self.TINY_NUM = None
        self.SMALL_NUM = None
        self.init_dataset()
    
    def init_dataset(self):
        dataset = None # Load the dataset
        if self.size == "tiny":
            dataset = dataset.select(range(self.TINY_NUM))
        elif self.size == "small":
            dataset = dataset.shuffle(seed=42).select(range(self.SMALL_NUM))
        elif isinstance(self.size, int):
            dataset = dataset.shuffle(seed=42).select(range(self.size))
        self.datas = dataset
    
    def get_prompts(self):
        """
        Generate prompts for the dataset
        """
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
        """
        Evaluate the model output
        """