from transformers import  BartForSequenceClassification, AutoTokenizer,  Trainer, TrainingArguments
import random
from datasets import Dataset
import pandas as pd
import numpy as np
import evaluate
import wandb
import os


wandb.login()

id2labels = ['disagree', 'agree', 'neutral', 'unrelated']


t = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
template = "The stance of the statement is {}"

df = pd.read_csv('../data/train/all_labels.csv', index_col=0).reset_index(drop=True).astype(str)
df.columns = ['text', 'label']
df['text'][pd.isna(df['text'])] = 'empty'
train_set = Dataset.from_pandas(df).shuffle(seed=42)

save_sets = train_set.train_test_split(train_size=0.8, seed=42)
save_sets['train'].save_to_disk('../data/stance_detector_eval/train/train_set')
save_sets['test'].save_to_disk('../data/stance_detector_eval/test/test_set')


def create_input_sequence(sample):
    text =sample['text']

    label = sample["label"][0]
    contradiction_label = random.choice([x for x in id2labels if x!=label])

    encoded_sequence = t(text*2, [template.format(label), template.format(contradiction_label)], truncation = True, padding = 'max_length')
    encoded_sequence["labels"] = [2,0]
    encoded_sequence["input_sentence"] = t.batch_decode(encoded_sequence.input_ids)
    return encoded_sequence

train_set = train_set.map(create_input_sequence, batched=True, batch_size=1,  remove_columns=['text', 'label'])
train_set = train_set.train_test_split(train_size=0.8, seed=42)

metric_acc = evaluate.load("accuracy")
metric_f1 = evaluate.load("f1")
def compute_metrics(eval_pred):
    logits = eval_pred.predictions[0]
    #preds = eval_pred[0] if isinstance(eval_pred, tuple) else eval_pred
    preds = np.argmax(logits, axis = 1)
    result = {}
    result["accuracy"] = metric_acc.compute(predictions = preds, references = eval_pred.label_ids)["accuracy"]
    result["f1"] = metric_f1.compute(predictions = preds, references = eval_pred.label_ids, average = 'macro')["f1"]
    return result

model = BartForSequenceClassification.from_pretrained("facebook/bart-large-mnli", ignore_mismatched_sizes = True)

# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"]="stance_detector"

# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"]="true"

# turn off watch to log faster
os.environ["WANDB_WATCH"]="false"


os.system('mkdir test_trainer logs')


lr = 2e-5

training_args = TrainingArguments(
    output_dir = '../models/bart_main',
    max_steps=1750,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=16,
    evaluation_strategy="steps",
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=10,
    eval_steps=250,
    learning_rate=lr,
    weight_decay=0.2,
    save_strategy="steps",
    save_steps=250,
    warmup_steps=500,
    report_to="wandb",
    run_name = 'final_bart_finetuned'
)


trainer = Trainer(
   model = model,  
    args = training_args,
    train_dataset = train_set['train'],
    eval_dataset = train_set['test'],  
    tokenizer = t,
      compute_metrics=compute_metrics,

)


trainer.train()
wandb.finish()

