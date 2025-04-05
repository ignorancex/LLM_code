#https://github.com/huggingface/notebooks/blob/main/examples/language_modeling.ipynb
from datasets import ClassLabel
import random
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import evaluate
import numpy as np
import torch
from tqdm import tqdm
tqdm.pandas()

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



import pandas as pd
df = pd.read_csv('/wrk/knc6/AtomNLP/Summarize/cond_mat.csv')
df=df[df['categories']=='cond-mat.supr-con'][0:500]
#df=df[df['categories']=='cond-mat.supr-con'][0:500]
#df = pd.read_csv('/wrk/knc6/AtomNLP/Summarize/cond_mat_small.csv')

train_size = 0.8
train_dataset=df.sample(frac=train_size,random_state = SEED)
val_dataset=df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)


train_dataset['text'] = train_dataset['title']+' can be described as '+train_dataset['abstract']
train_dataset = train_dataset['text']
train_dataset.to_csv('train_dataset.csv',index=False)


val_dataset['text'] = val_dataset['title']+' can be described as '+val_dataset['abstract']
val_dataset = val_dataset['text']
val_dataset.to_csv('val_dataset.csv',index=False)


print("FULL Dataset: {}".format(df.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(val_dataset.shape))



datasets = load_dataset('csv', data_files={'train': 'train_dataset.csv','validation':'val_dataset.csv',  'test': 'val_dataset.csv'})


model_checkpoint = "chavinlo/alpaca-native" #"distilgpt2"
# block_size = tokenizer.model_max_length
#model_checkpoint = "HuggingFaceM4/tiny-random-LlamaForCausalLM"
model_checkpoint = "gpt2-medium"
#model_checkpoint = "EleutherAI/gpt-neo-1.3B"

block_size = 512
block_size = 32
block_size = 128
block_size = 64

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=1, remove_columns=["text"])
lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=2,
    #batch_size=1000,
    #num_proc=1,
    num_proc=4,
)
model = AutoModelForCausalLM.from_pretrained(model_checkpoint)



#from transformers import AutoModelForCausalLM, AutoTokenizer
def generate_text(prompt = "Define superconductors", checkpoint = "gpt2-medium",max_new_tokens=200, model='',tokenizer=''):
    #https://towardsdatascience.com/text-generation-with-python-and-gpt-2-1fecbff1635b
    #tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    inputs = tokenizer(prompt, return_tensors="pt")
    #model = AutoModelForCausalLM.from_pretrained(checkpoint)
    #outputs = model.generate(**inputs, do_sample=True)
    outputs = model.generate(**inputs, do_sample=True, max_new_tokens=max_new_tokens)
    out = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return out[0]


calculate_baseline=True
#"""
if calculate_baseline:
 tmp = pd.read_csv('val_dataset.csv')
 print (tmp)

 tmp['title']=tmp['text'].apply(lambda x: x.split(' can be described as ')[0])
 tmp['abstract']=tmp['text'].apply(lambda x: x.split(' can be described as ')[1])
 tmp['pred']=tmp['title'].progress_apply(lambda x:generate_text(prompt=x,model=model,tokenizer=tokenizer))

#tmp['pred'] = tmp['text'].progress_apply(lambda x: generate_text(prompt=x,model=model,tokenizer=tokenizer))

 metric = evaluate.load("rouge")
 print('Baseline',metric.compute(references=tmp['abstract'],predictions=tmp['pred']))
#"""

model_name = model_checkpoint.split("/")[-1]
training_args = TrainingArguments(
    output_dir='./results',
    logging_dir='./logs',
    #f"{model_name}-finetuned-wikitext2",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=False,
    #overwrite_output_dir =True,
    do_eval=True, 
    #dataloader_pin_memory=False,
    #gradient_accumulation_steps=2,
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=4, 
    gradient_checkpointing=True
    #output_dir='results',
)

print ('training_args',training_args)
def compute_metrics(eval_preds):
    #metric = evaluate.load("glue","mrpc")
    metric = evaluate.load("rouge")
    #metric = evaluate.load("glue", "mrpc","rouge")
    logits, labels = eval_preds
    #predictions = np.argmax(logits, axis=-1)
    logits = np.argmax(logits, axis=-1)
    #print ('logits', logits,logits.shape)
    #print ()
    #print ('labels', labels,labels.shape)
    x1 = tokenizer.batch_decode(logits, skip_special_tokens=True)
    x2 = tokenizer.batch_decode(labels, skip_special_tokens=True)
    #print ('x1',x1)
    #print ('x2',x2)
    return metric.compute(predictions=x1, references=x2)
    #return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    compute_metrics=compute_metrics,
)
trainer.train()
import math
eval_results = trainer.evaluate()
print ('eval_results',eval_results)
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

