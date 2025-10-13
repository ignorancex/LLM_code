import json
import yaml
import math
import random
import os
import megfile
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset
from open_r1.arguments import GRPOScriptArguments, SFTScriptArguments
from open_r1.constants import system_prompt_registry, question_template_registry, answer_template_registry
from qwen_vl_utils import smart_resize
from tqdm import tqdm

class MMR1Dataset(Dataset):
    def __init__(self, script_args: GRPOScriptArguments):
        super(MMR1Dataset, self).__init__()
        self.data_path = script_args.dataset_name
        self.script_args = script_args
        self.system_prompt_template = system_prompt_registry[script_args.system_prompt_template]
        self.question_template = question_template_registry[script_args.question_template]
        self.answer_template = answer_template_registry[script_args.answer_template]
        self.list_data_dict = []

        if self.data_path.endswith(".yaml"):
            self.list_data_dict = self.get_datas_from_yaml(self.data_path)
        
        elif self.data_path.endswith(".json"):
            with open(self.data_path, "r") as file:
                self.list_data_dict = json.load(file)
            print(f"Loaded {len(self.list_data_dict)} samples from {self.data_path}")
        else:
            try:
                # huggingface dataset
                hf_dataset = load_dataset(self.data_path)
                if 'train' in hf_dataset:
                    split = 'train'
                else:
                    split = list(hf_dataset.keys())[0]
                if self.script_args.train_sample_size is not None:
                    self.list_data_dict = hf_dataset[split].select(range(self.script_args.train_sample_size)).to_list()
                else:
                    self.list_data_dict = hf_dataset[split].to_list()
                print(f"Loaded {len(self.list_data_dict)} samples from {self.data_path}")
            except Exception as e:
                raise ValueError(f"Unsupported file type: {self.data_path}")

        if self.script_args.train_sample_size is not None:
                self.list_data_dict = self.list_data_dict[:self.script_args.train_sample_size]

    def get_datas_from_yaml(self, yaml_path):
        '''
        file should be in the format of:
        datasets:
            - json_path: xxxx1.json
            sampling_strategy: first:1000
            - json_path: xxxx2.json
            sampling_strategy: end:3000
            - json_path: xxxx3.json
            sampling_strategy: random:999
        '''
        datas_from_yaml = []
        with open(yaml_path, "r") as file:
            yaml_data = yaml.safe_load(file)
            datasets = yaml_data.get("datasets")

            for data in datasets:
                json_path = data.get("json_path")
                sampling_strategy = data.get("sampling_strategy", "all")
                sampling_number = None

                if json_path.endswith(".jsonl"):
                    cur_data_dict = []
                    with open(json_path, "r") as json_file:
                        for line in json_file:
                            cur_data_dict.append(json.loads(line.strip()))
                elif json_path.endswith(".json"):
                    with open(json_path, "r") as json_file:
                        cur_data_dict = json.load(json_file)
                else:
                    raise ValueError(f"Unsupported file type: {json_path}")

                if ":" in sampling_strategy:
                    sampling_strategy, sampling_number = sampling_strategy.split(":")
                    if "%" in sampling_number:
                        sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                    else:
                        sampling_number = int(sampling_number)

                # Apply the sampling strategy
                if sampling_strategy == "first" and sampling_number is not None:
                    cur_data_dict = cur_data_dict[:sampling_number]
                elif sampling_strategy == "end" and sampling_number is not None:
                    cur_data_dict = cur_data_dict[-sampling_number:]
                elif sampling_strategy == "random" and sampling_number is not None:
                    random.shuffle(cur_data_dict)
                    cur_data_dict = cur_data_dict[:sampling_number]
                print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                datas_from_yaml.extend(cur_data_dict)
        return datas_from_yaml
    
    def __len__(self):
        return len(self.list_data_dict)

    def load_image(self, image_path):
        # when image is on oss, please run `unset http_proxy https_proxy all_proxy no_proxy`
        from io import BytesIO
        import os
        os.environ['OSS_ENDPOINT'] = 'http://oss.i.basemind.com'
        if isinstance(image_path, bytes):
            image = Image.open(BytesIO(image_path), "r").convert('RGB')
        elif 's3://' in image_path:
            with megfile.smart_open(image_path, "rb") as f:
                bytes_data = f.read()
            image = Image.open(BytesIO(bytes_data), "r").convert('RGB')
        else:
            image = Image.open(image_path).convert('RGB')
        return image

    def __getitem__(self, i):
        problem_key = self.script_args.problem_key
        answer_key = self.script_args.answer_key
        image_key = self.script_args.image_key
        image_dir = self.script_args.image_dir

        def make_conversation(example):
            return json.dumps(
                [
                    {"role": "system", "content": self.system_prompt_template.format(question=example[problem_key])},
                    {"role": "user", "content": self.question_template.format(question=example[problem_key])},
                ],
            )

        def make_conversation_image(example):
            return json.dumps(
                [
                    {"role": "system", "content": self.system_prompt_template.format(question=example[problem_key])},
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": self.question_template.format(question=example[problem_key])},
                    ]},
                ],
            )
        
        example = self.list_data_dict[i]
        if image_key in example:
            # huggingface dataset -> image item is a PIL.Image.Image object
            if isinstance(example[image_key], str):
                image_path = os.path.join(image_dir, example[image_key])
                os.environ['OSS_ENDPOINT'] = 'http://oss.i.basemind.com'
                # In case the image is not found
                while not megfile.smart_exists(image_path):
                    print(f"Warning: Image {image_path} not found, randomly selecting another image")
                    new_index = random.randint(0, len(self.list_data_dict)-1)
                    example = self.list_data_dict[new_index]
                    image_path = os.path.join(image_dir, example[image_key])
            else:
                image_path = example[image_key]['bytes']
            image = self.load_image(image_path)

            width, height = image.size
            min_pixels = self.script_args.min_pixels
            max_pixels = self.script_args.max_pixels
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=28,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            image = image.resize((resized_width, resized_height))
        
        else:
            image = None
        
        prompt = make_conversation(example) if 'image' not in example else make_conversation_image(example)

        return {
            'image': image,
            'problem': example[problem_key],
            'solution': example[answer_key],
            'prompt': prompt,
        }

class MMR1GRPODataset(MMR1Dataset):
    def __init__(self, script_args: GRPOScriptArguments):
        super(MMR1GRPODataset, self).__init__(script_args)
        self.script_args = script_args

class MMR1SFTDataset(MMR1Dataset):
    def __init__(self, script_args: SFTScriptArguments):
        super(MMR1SFTDataset, self).__init__(script_args)
        self.script_args = script_args
    
    def __getitem__(self, i):
        example = super(MMR1SFTDataset, self).__getitem__(i)
        prompt = json.loads(example['prompt'])
        prompt.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": str(self.answer_template.format(answer=example['solution']))}
            ]
        })
        return {
            'image': example['image'],
            'problem': example['problem'],
            'solution': example['solution'],
            'prompt': json.dumps(prompt),
        }

if __name__ == "__main__":
    # we provide some examples here, you can use them as templates to create your own dataset

    # GRPO Dataset for json files
    script_args = GRPOScriptArguments(
        dataset_name="path/to/your/dataset.json",
        system_prompt_template="qwen",
        question_template="<question_template>",
        answer_key="<answer_key>",
        image_dir="path/to/your/image_dir",
    )
    dataset = MMR1GRPODataset(script_args=script_args)
    print(dataset[0])

    # GRPO Dataset for huggingface datasets
    script_args = GRPOScriptArguments(
        dataset_name="path/to/your/dataset",
        system_prompt_template="qwen",
        question_template="<question_template>",
        answer_key="<answer_key>",
    )
    dataset = MMR1GRPODataset(script_args=script_args)
    print(dataset[0])

    # # SFT Dataset for json files
    script_args = SFTScriptArguments(
        dataset_name="path/to/your/dataset.json",
        system_prompt_template="qwen",
        question_template="<question_template>",
        answer_template="<answer_template>",
        answer_key="<answer_key>",
    )
    dataset = MMR1SFTDataset(script_args=script_args)
    print(dataset[0])