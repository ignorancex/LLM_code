#https://huggingface.co/docs/transformers/tasks/sequence_classification
from datasets import ClassLabel
import random
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

SEED = 42

def tokenize_function(examples):
    return tokenizer(examples["text"])

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result



#df = pd.read_csv('/wrk/knc6/AtomNLP/Summarize/cond_mat_small.csv')
df = pd.read_csv('/wrk/knc6/AtomNLP/Summarize/cond_mat.csv')

key = 'abstract'

df["category_id"] = df['categories'].factorize()[0]

train_size = 0.8
train_dataset=df.sample(frac=train_size,random_state = SEED)
val_dataset=df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)


train_dataset['text'] = train_dataset[key]
train_dataset['label']=train_dataset["category_id"]
train_dataset = train_dataset[['text','label']]
train_dataset.to_csv('train_dataset.csv',index=False)


val_dataset['text'] = val_dataset[key] #+' can be described as '+val_dataset['abstract']
val_dataset['label']=val_dataset["category_id"]
val_dataset = val_dataset[['text','label']]
val_dataset.to_csv('val_dataset.csv',index=False)


print("FULL Dataset: {}".format(df.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(val_dataset.shape))



datasets = load_dataset('csv', data_files={'train': 'train_dataset.csv','validation':'val_dataset.csv',  'test': 'val_dataset.csv'})

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)
tokenized_imdb = datasets.map(preprocess_function, batched=True)
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
import evaluate

accuracy = evaluate.load("accuracy")
import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
key = "categories"
category_id_df = (df[[key, "category_id"]].sort_values("category_id"))
category_to_id = dict(category_id_df.values)
label2id = dict(category_id_df.values)
id2label = {value:key for key, value in category_to_id.items()}
#id2label = {0: "NEGATIVE", 1: "POSITIVE"}
#label2id = {"NEGATIVE": 0, "POSITIVE": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=9, id2label=id2label, label2id=label2id
)

batch_size = 64
training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=10,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()




"""
model_checkpoint = "chavinlo/alpaca-native" #"distilgpt2"
# block_size = tokenizer.model_max_length
model_checkpoint = "HuggingFaceM4/tiny-random-LlamaForCausalLM"

block_size = 128

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=1, remove_columns=["text"])
lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=10,
    #batch_size=1000,
    num_proc=1,
    #num_proc=4,
)
model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
model_name = model_checkpoint.split("/")[-1]
training_args = TrainingArguments(
    f"{model_name}-finetuned-wikitext2",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=False,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
)
trainer.train()
import math
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

"""
