import torch
# import evaluate
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from tqdm import tqdm
import pandas as pd


batch_size = 32
model_name_or_path = "./Roberta_large"
task = "sst2"

if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


datasets = load_dataset("./GLUE/glue.py", task)
# print(datasets)
# metric = evaluate.load("glue", task)


def tokenize_function(examples):
    # max_length=None => use the model max length (it's actually the default)
    outputs = tokenizer(examples["sentence"], truncation=True, max_length=None)
    return outputs


tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["idx", "sentence"],
)

# We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
# transformers library
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")


def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")


# Instantiate dataloaders.
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size
)

test_datasets = load_dataset("json", data_files="./dev/" + task + ".json")
test_tokenized_datasets = test_datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["idx", "sentence"],
)
test_tokenized_datasets = test_tokenized_datasets.rename_column("label", "labels")
test_dataloader = DataLoader(
    test_tokenized_datasets["train"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size
)



# 初始化模型
model = RobertaForSequenceClassification.from_pretrained(model_name_or_path, num_labels=2)

# 定义训练参数
lr = 2e-5
num_epochs = 20


# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=lr)

# 学习率调度器
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# 训练模型
device = torch.device("cuda:0")
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)
    model.save_pretrained("./sst_large_ft/" + str(epoch))

    model.eval()
    val_list = []
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()

            logits = outputs.logits
            _, predicted = torch.max(logits, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(eval_dataloader)
    val_accuracy = (val_correct / val_total) * 100
    val_list.append(val_accuracy)

    print(f'Epoch {epoch+1}/{num_epochs}')
    print(f'Training Loss: {avg_train_loss:.4f}')
    print(f'Validation Loss: {avg_val_loss:.4f}')
    print(f'Validation Accuracy: {val_accuracy:.2f}%')

    test_list = []
    test_loss = 0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            test_loss += loss.item()

            logits = outputs.logits
            _, predicted = torch.max(logits, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    avg_test_loss = test_loss / len(test_dataloader)
    test_accuracy = (test_correct / test_total) * 100
    test_list.append(test_accuracy)

    print(f'Validation Loss: {avg_test_loss:.4f}')
    print(f'Validation Accuracy: {test_accuracy:.2f}%')
    for i in range(len(test_list)):
        print("test epoch ", i, ": ", val_list[i])
        print("adversarial epoch ", i, ": ", test_list[i])
