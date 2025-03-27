import json
import os
import random
import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """
@article{wang2022benchmarking,
  title={Benchmarking Generalization via In-Context Instructions on 1,600+ Language Tasks},
  author={Wang, Yizhong and Mishra, Swaroop and Alipoormolabashi, Pegah and Kordi, Yeganeh and others},
  journal={arXiv preprint arXiv:2204.07705},
  year={2022}
}
"""

_DESCRIPTION = """
Natural-Instructions v2 is a benchmark of 1,600+ diverse language tasks and their expert-written instructions. 
It covers 70+ distinct task types, such as tagging, in-filling, and rewriting. 
These tasks are collected with contributions of NLP practitioners in the community and 
through an iterative peer review process to ensure their quality. 
"""

_URL = "https://instructions.apps.allenai.org/"

class NIConfig(datasets.BuilderConfig):
    def __init__(self, *args, task_dir=None, max_num_instances_per_task=None, max_num_instances_per_eval_task=None, corruption_base_file=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_dir: str = task_dir
        self.max_num_instances_per_eval_task: int = max_num_instances_per_eval_task
        self.corruption_base_file: str = corruption_base_file


class NaturalInstructions(datasets.GeneratorBasedBuilder):
    """NaturalInstructions Dataset."""

    VERSION = datasets.Version("2.0.0")
    BUILDER_CONFIG_CLASS = NIConfig
    BUILDER_CONFIGS = [
        NIConfig(name="default", description="Default config for NaturalInstructions")
    ]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "Task": datasets.Value("string"),
                    "Contributors": datasets.Value("string"),
                    "Source": [datasets.Value("string")],
                    "URL": [datasets.Value("string")],
                    "Categories": [datasets.Value("string")],
                    "Reasoning": [datasets.Value("string")],
                    "Definition": [datasets.Value("string")],
                    "Positive Examples": [{
                        "input": datasets.Value("string"),
                        "output": datasets.Value("string"),
                        "explanation": datasets.Value("string")
                    }],
                    "Negative Examples": [{
                        "input": datasets.Value("string"),
                        "output": datasets.Value("string"),
                        "explanation": datasets.Value("string")
                    }],
                    "Input_language": [datasets.Value("string")],
                    "Output_language": [datasets.Value("string")],
                    "Instruction_language": [datasets.Value("string")],
                    "Domains": [datasets.Value("string")],
                    "Instance": {
                        "id": datasets.Value("string"),
                        "input": datasets.Value("string"),
                        "output": datasets.Value("string")
                    },
                    "Instance License": datasets.Value("string"),
                    "corruption_id":datasets.Value("string"),
                    "inline-instruction": datasets.Value("string")
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/allenai/natural-instructions",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if self.config.data_dir is None or self.config.task_dir is None:
            dl_path = dl_manager.download_and_extract(_URL)
            self.config.data_dir = self.config.data_dir or os.path.join(dl_path, "splits")
            self.config.task_dir = self.config.task_dir or os.path.join(dl_path, "tasks")
            self.config.corruption_base_file = self.corruption_base_file
            self.config.max_num_instances_per_eval_task = self.max_num_instances_per_eval_task

        split_dir = self.config.data_dir
        task_dir = self.config.task_dir

        print(os.path.join(split_dir, "test_tasks.txt"),  task_dir, self.config.max_num_instances_per_eval_task, self.config.corruption_base_file)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "path": os.path.join(split_dir, "test_tasks.txt"), 
                    "task_dir": task_dir, 
                    "max_num_instances_per_task": self.config.max_num_instances_per_eval_task,
                    "corruption_base_file": self.config.corruption_base_file,
                    "subset":"test"
                }),
        ]

    def _generate_examples(self, path=None, task_dir=None, max_num_instances_per_task=None, corruption_base_file=None, subset=None):
        """Yields examples."""
      
        if subset=="test":
            logger.info(f"Generating tasks from = {path}")
            
            with open(path, encoding="utf-8") as split_f:
                for line in split_f:
                    task_name = line.strip()
                    task_path = os.path.join(task_dir, task_name + "_"+ corruption_base_file + ".json")

                    if os.path.exists(task_path): 
                        with open(task_path, encoding="utf-8") as task_f:
                            s = task_f.read()
                            task_data = json.loads(s)
                            task_data["Task"] = task_name
                            if "Instruction Source" in task_data:
                                task_data.pop("Instruction Source")
                            all_instances = task_data.pop("Instances")
                            instances = all_instances[:max_num_instances_per_task] # keep 100 for test
                            # if max_num_instances_per_task is not None and max_num_instances_per_task >= 0:
                            #     random.shuffle(instances)
                            #     instances = instances[:max_num_instances_per_task]
                            for idx, instance in enumerate(instances):
                                example = task_data.copy()
                                example["id"] = instance["id"]
                                example["Instance"] = instance
                                yield f"{task_name}_{idx}", example
          

