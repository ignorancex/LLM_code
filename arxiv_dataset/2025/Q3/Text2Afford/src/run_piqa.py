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
from transformers import default_data_collator
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# torch.cuda.set_device(1)


logging.basicConfig(level=logging.INFO)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

dataset = datasets.load_dataset("piqa")

model_name = "roberta-large"

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

max_length = 128 # The maximum length of a feature (question and context)
def convert_to_piqa_features(example_batch):

    num_examples = len(example_batch["goal"])

    features = {}
    
    for example_i in range(num_examples):

        goals = [example_batch["goal"][example_i]] * 2

        solutions = [example_batch['sol1'][example_i], example_batch['sol2'][example_i]]

        input = list(zip(goals, solutions))
        choices_inputs = tokenizer.batch_encode_plus(input, 
                                                        truncation=True, pad_to_max_length=True, max_length=max_length, return_tensors='pt')
        

        for k, v in choices_inputs.items():
            if k not in features:
                features[k] = []
            features[k].append(v)

    features["labels"] = example_batch["label"]
    return features

tokenized_datasets = dataset.map(convert_to_piqa_features, batched=True, remove_columns=dataset["train"].column_names)



## Fine-Tuning
model = transformers.AutoModelForMultipleChoice.from_pretrained(model_name)

model.to(device)



from transformers import default_data_collator

data_collator = default_data_collator

trainer = transformers.Trainer(
    model=model.to(device),
    args=transformers.TrainingArguments(
        output_dir="./models/roberta_large_piqa",
        overwrite_output_dir=True,
#         evaluation_strategy = 'epoch',
        learning_rate=5e-5,
        do_train=True,
        num_train_epochs=5,
        # Adjust batch size if this doesn't fit on the Colab GPU
        per_device_train_batch_size=8, 
        save_total_limit = 1, 
        # no_cuda = True
    ),
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    # eval_dataset=eval_dataset,
    # compute_metrics=compute_metrics,
)
trainer.train()


# model.load_state_dict(torch.load('./models/bert_large_piqa/checkpoint-5000/pytorch_model.bin'))
model.to(device)

def eval_fn(model, batch_size=8):

    set_name = "validation"
    val_len = len(tokenized_datasets[set_name])
    acc = []

    print(tokenized_datasets[set_name][0]['input_ids'])
    print(tokenized_datasets["train"][0]['input_ids'])
    for index in range(0, val_len):
        batch = {}

        batch['input_ids'] = torch.stack([torch.tensor(ids) for ids in tokenized_datasets[set_name][index]['input_ids']]).to(device)
        batch['attention_mask'] = torch.stack([torch.tensor(masks) for masks in tokenized_datasets[set_name][
            index
        ]['attention_mask']]).to(device)
        true = tokenized_datasets[set_name][
            index
        ]['labels']

        



        outputs = model(**batch)
        # pull preds out
        pred = torch.argmax(
            torch.FloatTensor(torch.softmax(outputs['logits'], dim=1).detach().cpu().tolist()),
            dim=1,
        )
        print("#"*50)
        print(pred)
        print(true)
        # calculate accuracy for both and append to accuracy list
        acc.append(((pred == true).sum()/len(pred)).item())
        # acc += sum(np.array(predictions) == np.array(labels))
    acc = sum(acc)/len(acc)
    print(f"Task name: PiQA \t Accuracy: {acc}")

eval_fn(model, batch_size=8)