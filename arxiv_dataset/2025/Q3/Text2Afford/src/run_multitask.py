import numpy as np
import torch
import torch.nn as nn
import transformers
import datasets
import os
import gc
# import nlp
import logging
import json
from tqdm import tqdm
import pandas as pd
import datasets
from sklearn.model_selection import train_test_split
from transformers import BertPreTrainedModel, BertModel
import torch.nn.functional as F
logging.basicConfig(level=logging.INFO)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

dataset_dict = {
    # "swag": datasets.load_dataset("swag", "regular"),
    "piqa": datasets.load_dataset("piqa"),
    # "commonsense_qa": datasets.load_dataset("commonsense_qa")
}

# model_name = "SpanBERT/spanbert-base-cased"
model_name = "bert-base-uncased"

class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, encoder, taskmodels_dict):
        """
        Setting MultitaskModel up as a PretrainedModel allows us
        to take better advantage of Trainer features
        """
        super().__init__(transformers.PretrainedConfig())

        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def create(cls, model_name, model_type_dict, model_config_dict):
        """
        This creates a MultitaskModel using the model class and config objects
        from single-task models. 

        We do this by creating each single-task model, and having them share
        the same encoder transformer.
        """
        shared_encoder = None
        taskmodels_dict = {}
        for task_name, model_type in model_type_dict.items():
            model = model_type.from_pretrained(
                model_name, 
                config=model_config_dict[task_name],
            )
            if shared_encoder is None:
                shared_encoder = getattr(model, cls.get_encoder_attr_name(model))
            else:
                setattr(model, cls.get_encoder_attr_name(model), shared_encoder)
            taskmodels_dict[task_name] = model
        return cls(encoder=shared_encoder, taskmodels_dict=taskmodels_dict)

    @classmethod
    def get_encoder_attr_name(cls, model):
        """
        The encoder transformer is named differently in each model "architecture".
        This method lets us get the name of the encoder attribute
        """
        model_class_name = model.__class__.__name__
        if model_class_name.startswith("Bert"):
            return "bert"
        elif model_class_name.startswith("Roberta"):
            return "roberta"
        elif model_class_name.startswith("Albert"):
            return "albert"
        elif model_class_name.startswith("DistilBert"):
            return "distilbert"
        elif model_class_name.startswith("Span"):
            return "SpanBERT/spanbert-base-cased"
        else:
            raise KeyError(f"Add support for new model {model_class_name}")

    def forward(self, task_name, **kwargs):
        return self.taskmodels_dict[task_name](**kwargs)
    


multitask_model = MultitaskModel.create(
    model_name=model_name,
    model_type_dict={
        "piqa": transformers.AutoModelForMultipleChoice,
        "swag": transformers.AutoModelForMultipleChoice,
        "commonsense_qa": transformers.AutoModelForMultipleChoice
    },
    model_config_dict={
        "piqa": transformers.AutoConfig.from_pretrained(model_name),
        "swag": transformers.AutoConfig.from_pretrained(model_name),
        "commonsense_qa": transformers.AutoConfig.from_pretrained(model_name)
    },
)


tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

max_length = 128 # The maximum length of a feature (question and context)
doc_stride = 32 # The authorized overlap between two part of the context when splitting it is needed.
pad_on_right = tokenizer.padding_side == "right"

def convert_to_commonsense_qa_features(example_batch):
    num_examples = len(example_batch["question"])
    num_choices = len(example_batch["choices"][0]["text"])
    features = {}

    print(num_examples)
    # return features
    for example_i in range(num_examples):

        input = list(zip(
                [example_batch["question"][example_i]] * num_choices,
                example_batch["choices"][example_i]["text"],
            ))
        
        # print("#"*10)
        # print(input) 
        # print("#"* 10)
        choices_inputs = tokenizer.batch_encode_plus(
            input,
            max_length=max_length, pad_to_max_length=True,
            return_tensors='pt'
        )
        for k, v in choices_inputs.items():
            if k not in features:
                features[k] = []
            features[k].append(v)
    labels2id = {char: i for i, char in enumerate("ABCDE")}
    # Dummy answers for test
    if example_batch["answerKey"][0]:
        features["labels"] = [labels2id[ans] for ans in example_batch["answerKey"]]
    else:
        features["labels"] = [0] * num_examples    
    return features


def convert_to_piqa_features(example_batch):

    num_examples = len(example_batch["goal"])

    features = {}
    
    for example_i in range(num_examples):

        goals = [example_batch["goal"][example_i]] * 2

        solutions = [example_batch['sol1'][example_i], example_batch['sol2'][example_i]]

        input = list(zip(goals, solutions))

        # print("#"*10)
        # print(input) 
        # print("#"* 10)
        # Tokenize
        choices_inputs = tokenizer.batch_encode_plus(input, 
                                                        truncation=True, pad_to_max_length=True, max_length=max_length, return_tensors='pt')
        

        for k, v in choices_inputs.items():
            if k not in features:
                features[k] = []
            features[k].append(v)

    # for k, v in features.items():
    #     features[k] = np.array(v)

    features["labels"] = example_batch["label"]
    
    

    print(type(features['input_ids']))
    return features


ending_names = ["ending0", "ending1", "ending2", "ending3"]
def convert_to_swag_features(example_batch):

    num_examples = len(example_batch["sent1"])

    features = {}
    
    for example_i in range(num_examples):
        # Repeat each first sentence four times to go with the four possibilities of second sentences.
        first_sentences = [example_batch["sent1"][example_i]] * 4
        # Grab all second sentences possible for each context.
        question_headers = example_batch["sent2"][example_i] * 4

        second_sentences = [f"{example_batch['sent2'][example_i]} {example_batch[end][example_i]}" for end in ending_names]

        input = list(zip(first_sentences, second_sentences))
        # Tokenize
        choices_inputs = tokenizer.batch_encode_plus(input, 
                                                        truncation=True, max_length=max_length, pad_to_max_length=True, return_tensors='pt')
        

        
        for k, v in choices_inputs.items():
            if k not in features:
                features[k] = []
            features[k].append(v)

    features["labels"] = example_batch["label"]

    print(example_batch['label'][0:2])
    return features

convert_func_dict = {
    "piqa": convert_to_piqa_features,
    "swag": convert_to_swag_features,
    "commonsense_qa": convert_to_commonsense_qa_features
}


# ds1 = datasets.load_dataset("swag", "regular")

# # 80% train, 20% test + validation
# swag = ds1['train'].train_test_split(0.02)['test'].train_test_split(0.3)

# ds2 = datasets.load_dataset("commonsense_qa")

# common_qa = ds2['train'].train_test_split(0.2)['test'].train_test_split(0.3)

# ds3 = datasets.load_dataset("piqa")

# piqa = ds3['train'].train_test_split(0.2)['test'].train_test_split(0.3)

# dataset_dict = {
#     "piqa": piqa,
#     # "commonsense_qa": common_qa
# }


columns_dict = {
    "piqa": ['input_ids', 'attention_mask', 'labels'],
    "commonsense_qa": ['input_ids', 'attention_mask', 'labels']
}



features_dict = {}
for task_name, dataset in dataset_dict.items():
    features_dict[task_name] = {}
    if task_name != 'qa':
        
        for phase, phase_dataset in dataset.items():
            features_dict[task_name][phase] = phase_dataset.map(
                convert_func_dict[task_name],
                batched=True,
                load_from_cache_file=False,
                
            )
            print(task_name, phase, len(phase_dataset), len(features_dict[task_name][phase]))
            features_dict[task_name][phase].set_format(
                type="torch", 
                columns=columns_dict[task_name],
            )
        
    else:
        for phase, phase_dataset in dataset.items():
            features_dict[task_name][phase] = phase_dataset.map(
                convert_func_dict[task_name],
                batched=True,
                load_from_cache_file=False,
                remove_columns=phase_dataset.column_names
            )

            features_dict[task_name][phase].set_format(
                type="torch", 
                columns=columns_dict[task_name],
            )

    print(task_name, phase, len(phase_dataset), len(features_dict[task_name][phase]))



import dataclasses
from torch.utils.data.dataloader import DataLoader
# from transformers.training_args import is_tpu_available
# from transformers.trainer import get_tpu_sampler
from transformers import DataCollator, DefaultDataCollator
from transformers.data.data_collator import InputDataClass
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from typing import List, Union, Dict
from transformers import default_data_collator



print(type(default_data_collator))
class NLPDataCollator():
    """
    Extending the existing DataCollator to work with NLP dataset batches
    """
    def __call__(self, features: List[Union[InputDataClass, Dict]]) -> Dict[str, torch.Tensor]:
        first = features[0]
        if isinstance(first, dict) and "labels" in first:
            # NLP data sets current works presents features as lists of dictionary
            # (one per example), so we  will adapt the collate_batch logic for that
            if "labels" in first and first["labels"] is not None:
                if first["labels"].dtype == torch.int64:
                    labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
                else:
                    labels = torch.tensor([f["labels"] for f in features], dtype=torch.float)
                batch = {"labels": labels}
            for k, v in first.items():
                if k != "labels" and v is not None and not isinstance(v, str):
                    # for f in features:
                    #     print(type(f[k]))

                    batch[k] = torch.stack([torch.stack(f[k]) for f in features])
            return batch
        else:
          # otherwise, revert to using the default collate_batch
          return DefaultDataCollator().collate_batch(features)
            # return default_data_collator

# data_collator_ = default_data_collator
data_collator_ = NLPDataCollator()

class StrIgnoreDevice(str):
    """
    This is a hack. The Trainer is going call .to(device) on every input
    value, but we need to pass in an additional `task_name` string.
    This prevents it from throwing an error
    """
    def to(self, device):
        return self


class DataLoaderWithTaskname:
    """
    Wrapper around a DataLoader to also yield a task name
    """
    def __init__(self, task_name, data_loader):
        self.task_name = task_name
        self.data_loader = data_loader

        self.batch_size = data_loader.batch_size
        self.dataset = data_loader.dataset

    def __len__(self):
        return len(self.data_loader)
    
    def __iter__(self):
        for batch in self.data_loader:
            batch["task_name"] = StrIgnoreDevice(self.task_name)
            yield batch


class MultitaskDataloader:
    """
    Data loader that combines and samples from multiple single-task
    data loaders.
    """
    def __init__(self, dataloader_dict):
        self.dataloader_dict = dataloader_dict
        self.num_batches_dict = {
            task_name: len(dataloader) 
            for task_name, dataloader in self.dataloader_dict.items()
        }
        self.task_name_list = list(self.dataloader_dict)
        self.dataset = [None] * sum(
            len(dataloader.dataset) 
            for dataloader in self.dataloader_dict.values()
        )

    def __len__(self):
        return sum(self.num_batches_dict.values())

    def __iter__(self):
        """
        For each batch, sample a task, and yield a batch from the respective
        task Dataloader.

        We use size-proportional sampling, but you could easily modify this
        to sample from some-other distribution.
        """
        task_choice_list = []
        for i, task_name in enumerate(self.task_name_list):
            task_choice_list += [i] * self.num_batches_dict[task_name]
        task_choice_list = np.array(task_choice_list)
        np.random.shuffle(task_choice_list)
        dataloader_iter_dict = {
            task_name: iter(dataloader) 
            for task_name, dataloader in self.dataloader_dict.items()
        }
        for task_choice in task_choice_list:
            task_name = self.task_name_list[task_choice]
            yield next(dataloader_iter_dict[task_name])    

class MultitaskTrainer(transformers.Trainer):

    def get_single_train_dataloader(self, task_name, train_dataset):
        """
        Create a single-task data loader that also yields task names
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = (
            RandomSampler(train_dataset)
            if self.args.local_rank == -1
            else DistributedSampler(train_dataset)
        )

        data_loader = DataLoaderWithTaskname(
            task_name=task_name,
            data_loader=DataLoader(
              train_dataset,
              batch_size=self.args.train_batch_size,
              sampler=train_sampler,
              collate_fn=data_collator_,
            ),
        )

        # if is_tpu_available():
        #     data_loader = pl.ParallelLoader(
        #         data_loader, [self.args.device]
        #     ).per_device_loader(self.args.device)
        return data_loader

    def get_train_dataloader(self):
        """
        Returns a MultitaskDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each 
        task Dataloader
        """
        return MultitaskDataloader({
            task_name: self.get_single_train_dataloader(task_name, task_dataset)
            for task_name, task_dataset in self.train_dataset.items()
        })




import os
os.environ["WANDB_DISABLED"] = "true"
train_dataset = {
    task_name: dataset["train"] 
    for task_name, dataset in features_dict.items()
}


eval_dataset = {
    task_name: dataset["validation"] 
    for task_name, dataset in features_dict.items()
}

# trainer = MultitaskTrainer(
#     model=multitask_model,
#     args=transformers.TrainingArguments(
#         output_dir="./models/multitask_model_bert_only",
#         overwrite_output_dir=True,
# #         evaluation_strategy = 'epoch',
#         learning_rate=5e-5,
#         do_train=True,
#         num_train_epochs=5,
#         # Adjust batch size if this doesn't fit on the Colab GPU
#         per_device_train_batch_size=8,  
#         save_steps=2000,
#         # no_cuda = True
#     ),
#     data_collator=data_collator_,
#     train_dataset=train_dataset,
#     # eval_dataset=eval_dataset,
#     # compute_metrics=compute_metrics,
# )
# trainer.train()

gc.collect()
multitask_model.load_state_dict(torch.load('./models/multitask_model_bert_only/checkpoint-4000/pytorch_model.bin'))
multitask_model.to(device)

def multitask_eval_fn(multitask_model, model_name, features_dict, batch_size=8):
    preds_dict = {}

    set_name = "validation"
    # tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    for task_name in list(dataset_dict.keys()):
        val_len = len(features_dict[task_name][set_name])
        acc = []

        # print(features_dict[task_name][set_name][0:2]['input_ids'])
        """
        inputs = list(zip(example_batch['sentence1'], example_batch['sentence2']))
    features = tokenizer.batch_encode_plus(
        inputs, max_length=max_length, pad_to_max_length=True
    )
    features["labels"] = example_batch["label"]
        """
        for index in range(0, val_len, batch_size):

            # batch = list(zip(features_dict[task_name]["validation"][
            #     index : min(index + batch_size, val_len)
            # ]["sentence1"], 
            # features_dict[task_name]["validation"][
            #     index : min(index + batch_size, val_len)
            # ]["sentence2"]
            # ))
            batch = {}
            batch['input_ids'] = torch.stack([torch.stack(ids) for ids in features_dict[task_name][set_name][
                index : min(index + batch_size, val_len)
            ]['input_ids']]).to(device)
            batch['attention_mask'] = torch.stack([torch.stack(masks) for masks in features_dict[task_name][set_name][
                index : min(index + batch_size, val_len)
            ]['attention_mask']]).to(device)
            true = features_dict[task_name][set_name][
                index : min(index + batch_size, val_len)
            ]['labels']
            # labels = features_dict[task_name]["validation"][
            #     index : min(index + batch_size, val_len)
            # ]["labels"]
            # inputs = tokenizer(batch, max_length=512, padding=True)
            # inputs["input_ids"] = torch.LongTensor(inputs["input_ids"])
            # inputs["attention_mask"] = torch.LongTensor(inputs["attention_mask"])



            outputs = multitask_model(task_name, **batch)
            # pull preds out
            pred = torch.argmax(
                torch.FloatTensor(torch.softmax(outputs['logits'], dim=1).detach().cpu().tolist()),
                dim=1,
            )
            # print("#"*50)
            # print(pred)
            # print(true)
            # calculate accuracy for both and append to accuracy list
            acc.append(((pred == true).sum()/len(pred)).item())
            # acc += sum(np.array(predictions) == np.array(labels))
        acc = sum(acc)/len(acc)
        print(f"Task name: {task_name} \t Accuracy: {acc}")

multitask_eval_fn(multitask_model, model_name, features_dict, batch_size=4)
gc.collect()