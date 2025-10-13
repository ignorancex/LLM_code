
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

set_seed(42)

save_path = './models/flan-t5-large-finetuned-affordance'

model_name = 'google/flan-t5-large'
tokenizer = AutoTokenizer.from_pretrained(model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(model_name)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
t5_model.to(device)

# optimizer
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in t5_model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
    {
        "params": [p for n, p in t5_model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)


# dataset preparation
# filename = "../data/ECCV_affordance_data.tsv"
filename = "../data/final_annotated_data.tsv" ## for Piqa performance
data = pd.read_csv(filename, sep='\t')
label_names = list(data.columns[2:])
oov_class_map = {"play": "playing", "lookthrough": "looking through", "siton": "sitting on", "pourfrom": "pouring from", "writewith": "writing with", "typeon": "typing"}
labels_lower = [x.lower() for x in label_names]
classes = []
for lab in labels_lower:
    if lab not in oov_class_map.keys():
        classes.append(lab)
    else:
        classes.append(oov_class_map[lab])

prompt_label_pairs = []
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
    prompt_label_pairs.append((f"{sentence}  \nOPTIONS:\n- Human cannot use the {object_name} for {negative_class} \n- Human uses the {object_name} for {negative_class}", f"Human cannot use the {object_name} for {negative_class}"))

    try:
        # Randomly select one index from the ones
        one_index = np.random.choice(ones)
        positive_class = classes[one_index - 2] ## first 2 columns are Sentence and Object
        prompt_label_pairs.append((f"{sentence}  \nOPTIONS:\n- Human cannot use the {object_name} for {positive_class}  \n- Human uses the {object_name} for {positive_class}", f"Human uses the {object_name} for {positive_class}"))
    except:
        pass

    # print(row['Object'], classes[zero_index - 2]) ## first 2 columns are Sentence and Object
    # print(row['Object'], classes[one_index - 2]) ## first 2 columns are Sentence and Object

random.shuffle(prompt_label_pairs)

print(len(prompt_label_pairs))



t5_model.train()

epochs = 5

for epoch in range(epochs):
  print ("epoch ",epoch)
  for input,output in prompt_label_pairs:
    input_sent = "Select the most appropriate option based on the situation:\n\n: "+input + "</s>"
    ouput_sent = output

    tokenized_inp = tokenizer.encode_plus(input_sent,  max_length=128, pad_to_max_length=True,return_tensors="pt")
    tokenized_output = tokenizer.encode_plus(ouput_sent, max_length=128, pad_to_max_length=True,return_tensors="pt")


    input_ids  = tokenized_inp["input_ids"].to(device)
    attention_mask = tokenized_inp["attention_mask"].to(device)

    lm_labels= tokenized_output["input_ids"].to(device)
    decoder_attention_mask=  tokenized_output["attention_mask"].to(device)


    # the forward function automatically creates the correct decoder_input_ids
    output = t5_model(input_ids=input_ids, labels=lm_labels, decoder_attention_mask=decoder_attention_mask,attention_mask=attention_mask)
    loss = output[0]

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

t5_model.eval()
t5_model.save_pretrained(save_path, from_pt=True)

test_sent = 'Answer Yes or No: The metallic jingles of the sistrum added a mesmerizing rhythm to the ancient Egyptian ritual, invoking the divine presence.	Can human play Sistrum? </s>'
test_sent = "Answer Yes or No: The flowing chador enveloped her in modesty and grace, representing the cultural values and religious customs of Iran.	Can human feed Chador?"

test_tokenized = tokenizer.encode_plus(test_sent, return_tensors="pt")

test_input_ids  = test_tokenized["input_ids"].to(device)
test_attention_mask = test_tokenized["attention_mask"].to(device)



# beam_outputs = t5_model.generate(
#     input_ids=test_input_ids,attention_mask=test_attention_mask,
#     max_length=20,
#     early_stopping=True,
#     num_beams=4,
#     num_return_sequences=3,
#     no_repeat_ngram_size=2
# )

output_ids = t5_model.generate(test_input_ids)
predicted_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(test_sent)
print(predicted_text)

# for beam_output in beam_outputs:
#     sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
#     print (sent)


