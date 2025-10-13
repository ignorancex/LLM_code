import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup

from peft import AdaLoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model


os.environ["TOKENIZERS_PARALLELISM"] = "false"

batch_size = 16
model_name_or_path = "./Roberta_large"
task = "qnli"
device = torch.device("cuda:5")
num_epochs = 60

# creating model
peft_config = AdaLoraConfig(
    init_r=12,
    target_r=8,
    beta1=0.85,
    beta2=0.85,
    tinit=200,
    tfinal=1000,
    deltaT=10,
    lora_alpha=32,
    lora_dropout=0.1,
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
)

lr = 1e-4
if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

datasets = load_dataset("./GLUE/glue.py", task)


def tokenize_function(examples):
    # max_length=None => use the model max length (it's actually the default)
    outputs = tokenizer(examples["question"], examples["sentence"], truncation=True, max_length=510)
    return outputs

tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["idx", "question", "sentence"],
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
    remove_columns=["idx", "question", "sentence"],
)
test_tokenized_datasets = test_tokenized_datasets.rename_column("label", "labels")
test_dataloader = DataLoader(
    test_tokenized_datasets["train"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size
)

model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True)
peft_config.total_step = len(train_dataloader) * num_epochs
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# optimizer and lr scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# Instantiate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)

model.to(device)
acc_list = []
adversarial_list = []
for epoch in range(num_epochs):
    model.train()
    for step, batch in enumerate(train_dataloader):
        batch.to(device)
        # if batch['input_ids'].shape[-1] == 510:
        #     continue
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    model.save_pretrained("./" + task + "_large_adalora_2/" + str(epoch))
    model.eval()
    val_correct = 0
    val_total = 0
    for step, batch in enumerate(eval_dataloader):
        batch.to(device)
        if batch['input_ids'].shape[-1] == 510:
            continue
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]
        val_total += references.size(0)
        val_correct += (predictions == references).sum().item()
    eval_metric = (val_correct / val_total) * 100
    acc_list.append(eval_metric)
    print(f"epoch {epoch} Test Accuracy:", eval_metric)


    # adversarial validation
    test_correct = 0
    test_total = 0
    for step, batch in enumerate(test_dataloader):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]
        test_total += references.size(0)
        test_correct += (predictions == references).sum().item()

    eval_metric = (test_correct / test_total) * 100
    adversarial_list.append(eval_metric)
    print(f"epoch {epoch} Validation Accuracy:", eval_metric)
    print("---------From start to now-------")
    for i in range(len(acc_list)):
        print("epoch", i, " Test Accuracy : ", acc_list[i])
    for i in range(len(adversarial_list)):
        print("epoch", i, " Validation Accuracy : ", adversarial_list[i])
