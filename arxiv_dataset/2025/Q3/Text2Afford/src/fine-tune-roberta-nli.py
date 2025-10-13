import torch
import torch.nn as nn
from transformers import RobertaForSequenceClassification, RobertaTokenizer, AutoModelForSequenceClassification, AutoTokenizer
from transformers import AdamW
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import random
import numpy as np
import datasets
import transformers
import os
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"]= "1"


model_name = 'roberta-large'

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

set_seed(42)

save_path = './models/roberta-large-finetuned-affordance'

# dataset preparation
filename = "../data/ECCV_affordance_data.tsv"
# filename = "../data/final_annotated_data.tsv" ## for Piqa performance
data = pd.read_csv(filename, sep='\t')
label_names = list(data.columns[2:])
oov_class_map = {"lookthrough": "looking", "siton": "sitting", "pourfrom": "pouring", "writewith": "writing", "typeon": "typing"}

labels_lower = [x.lower() for x in label_names]
classes = []
for lab in labels_lower:
    if lab not in oov_class_map.keys():
        classes.append(lab)
    else:
        classes.append(oov_class_map[lab])

prompt_label_pairs = [] ## List of sentence, prompt, label
for idx, row in data.iterrows():

    object_name = row['Object']
    sentence = row['Sentence']

    # Get the indices where the value is 0
    zeros = np.where(row == 0)[0]
    # Get the indices where the value is 1
    ones = np.where(row == 1)[0]
    
    # Randomly select one index from the zeros
    zero_index = np.random.choice(zeros)
    negative_class = classes[zero_index - 2] ## first 2 columns are Sentence and Object
    # prompt_label_pairs.append((f"{sentence}", f"{object_name} can be used for {negative_class} by human", 0))

    prompt_label_pairs.append({
        "sentence": sentence,
        "prompt": f"{object_name} can be used for {negative_class} by human",
        "label": 0
    })

    try:
        # Randomly select one index from the ones
        one_index = np.random.choice(ones)
        positive_class = classes[one_index - 2] ## first 2 columns are Sentence and Object
        # prompt_label_pairs.append((f"{sentence}", f"{object_name} can be used for {positive_class} by human", 1))
        prompt_label_pairs.append({
        "sentence": sentence,
        "prompt": f"{object_name} can be used for {positive_class} by human",
        "label": 1
    })
    except:
        pass


random.shuffle(prompt_label_pairs)

data_df = pd.DataFrame(prompt_label_pairs)

train_data = datasets.Dataset.from_pandas(data_df).class_encode_column("label")

print(train_data.features)


# Define hyperparameters
batch_size = 16
max_length = 128
num_epochs = 5
learning_rate = 2e-5

# Load RoBERTa tokenizer and model
model = AutoModelForSequenceClassification.from_pretrained('roberta-large', num_labels=2)
tokenizer = AutoTokenizer.from_pretrained('roberta-large')  # Set num_labels to the number of labels in your NLI task
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
model.to(device)
# tokenizer.to()

def convert_examples_to_features(examples):

    features = tokenizer(examples['sentence'], examples['prompt'],
                                            max_length=max_length,
                                            padding='max_length',
                                            # return_token_type_ids=True,
                                            return_attention_mask = True,
                                            add_special_tokens=True,
                                            return_tensors='pt')
    
    features['input_ids'] = torch.tensor(features['input_ids'])
    features['attention_mask'] = torch.tensor(features['attention_mask'])
    features['labels'] = examples['label']
    
    return features
    
tokenized_datasets = train_data.map(convert_examples_to_features, batched=False, remove_columns=train_data.column_names)

print(tokenized_datasets)


print(len(tokenized_datasets['input_ids'][0][0]))
print(len(tokenized_datasets['input_ids'][1][0]))
# print(len(tokenized_datasets['input_ids']))   


from transformers import default_data_collator

data_collator = default_data_collator

from torch.nn.utils.rnn import pad_sequence #(1)

def custom_collate(data): #(2)
    input_ids = [torch.tensor(d['input_ids']) for d in data] #(3)
    attention_mask = [torch.tensor(d['attention_mask']) for d in data]
    labels = [d['labels'] for d in data]

    input_ids = pad_sequence(input_ids, batch_first=True) #(4)
    attention_mask = pad_sequence(attention_mask, batch_first=True)
    labels = torch.tensor(labels) #(5)

    return { #(6)
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }



# trainer = transformers.Trainer(
#     model=model.to(device),
#     args=transformers.TrainingArguments(
#         output_dir=save_path,
#         overwrite_output_dir=True,
# #         evaluation_strategy = 'epoch',
#         learning_rate=2e-5,
#         do_train=True,
#         num_train_epochs=5,
#         # Adjust batch size if this doesn't fit on the Colab GPU
#         per_device_train_batch_size=8, 
#         save_total_limit = 1, 
#         # no_cuda = True,
#         # place_model_on_device = True,
#     ),
#     data_collator=data_collator,
#     train_dataset=tokenized_datasets,
    
#     # eval_dataset=eval_dataset,
#     # compute_metrics=compute_metrics,
# )
# print(trainer.args.device)
# trainer.train()



# # Define the custom dataset for NLI
# class NLIDataset(Dataset):
#     def __init__(self, data, tokenizer, max_length):
#         self.data = data
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]
#         premise = item['premise']
#         hypothesis = item['hypothesis']
#         label = item['label']

#         encoding = self.tokenizer.encode_plus(
#             premise,
#             hypothesis,
#             add_special_tokens=True,
#             max_length=self.max_length,
#             padding='max_length',
#             truncation=True,
#             return_tensors='pt'
#         )

#         input_ids = encoding['input_ids'].squeeze()
#         attention_mask = encoding['attention_mask'].squeeze()

#         return {
#             'input_ids': input_ids,
#             'attention_mask': attention_mask,
#             'label': label
#         }


# # Create dataloaders for train and validation data
# train_dataset = NLIDataset(train_data, tokenizer, max_length)
# val_dataset = NLIDataset(val_data, tokenizer, max_length)

train_dataloader = DataLoader(tokenized_datasets, batch_size=16, shuffle=True, collate_fn=custom_collate)
# val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# Move the model to GPU if available
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the optimizer and loss function
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)
loss_fn = nn.CrossEntropyLoss()

# Fine-tuning loop
for epoch in tqdm(range(num_epochs)):
    # Training
    model.train()
    train_loss = 0.0
    train_acc = 0.0

    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # print(input_ids.shape)
        input_ids = input_ids.squeeze(1)
        attention_mask = attention_mask.squeeze(1)

        # print(input_ids.shape)
       
        labels = batch['labels'].to(device)
        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        logits = outputs.logits
        _, preds = torch.max(logits, dim=1)

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * input_ids.size(0)
        train_acc += torch.sum(preds == labels).item()
        print(labels, preds)
    train_loss = train_loss / len(tokenized_datasets)
    train_acc = train_acc / len(tokenized_datasets)

    print(train_acc, train_loss)

    # # Validation
    # model.eval()
    # val_loss = 0.0

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)