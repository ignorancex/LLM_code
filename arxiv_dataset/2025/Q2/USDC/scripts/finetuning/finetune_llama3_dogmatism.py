#!/usr/bin/env python
# coding: utf-8

# ## Fine-tune Llama 2 for Dogmatism
# 
# The code imports the os module and sets two environment variables:
# * CUDA_VISIBLE_DEVICES: This environment variable tells PyTorch which GPUs to use. In this case, the code is setting the environment variable to 0, which means that PyTorch will use the first GPU.
# * TOKENIZERS_PARALLELISM: This environment variable tells the Hugging Face Transformers library whether to parallelize the tokenization process. In this case, the code is setting the environment variable to false, which means that the tokenization process will not be parallelized.
# * TRANSFORMERS_CACHE: This will provide the model loaded from the directory where there is no space issues


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TRANSFORMERS_CACHE'] = "./"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from datasets import Dataset
from peft import LoraConfig, PeftConfig
from trl import SFTTrainer
from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer, 
                          BitsAndBytesConfig, 
                          TrainingArguments, 
                          pipeline, 
                          logging)
from sklearn.metrics import (accuracy_score, 
                             classification_report, 
                             confusion_matrix)
from sklearn.model_selection import train_test_split


# In[5]:4

print(f"pytorch version {torch.__version__}")
torch.cuda.empty_cache()

# Label column which is used as ground truth for model training
main_column = 'gpt41106preview_one_shot_dogmatism_label'
#main_column = 'majority_vote_dogmatism_label'

data = pd.read_csv('all_users_dogmatism.csv')
print(data[main_column].value_counts())

X_train = list()
X_test = list()
for sentiment in list(np.unique(data[main_column])):
    train, test  = train_test_split(data[data[main_column]==sentiment], test_size=0.2,
                                    random_state=42)
    X_train.append(train)
    X_test.append(test)

X_train = pd.concat(X_train).sample(frac=1, random_state=10)
X_test = pd.concat(X_test)

eval_idx = [idx for idx in data.index if idx not in list(train.index) + list(test.index)]
X_eval = data[data.index.isin(eval_idx)]
X_eval = (X_eval
          .groupby(main_column, group_keys=False)
          .apply(lambda x: x.sample(n=30, random_state=10, replace=True)))
X_train = X_train.reset_index(drop=True)

# Creating prompt for the training dataset
def generate_prompt(data_point):
    return f"""
            Analyze the comments of a user in conversation enclosed in square brackets. Categorize the opinion fluctuation of the user into one of the following categories based on its change:
- Open to Dialogue
- Firm but Open
- Deeply Rooted
- Flexible
Return the answer as one of the corresponding dogmatism labels.

            [{data_point["comments_string_for_dogmatism"]}] = {data_point[main_column]}
            """.strip()

# Creating prompt for the testing dataset
def generate_test_prompt(data_point):
    return f"""
            Analyze the comments of a user in conversation enclosed in square brackets. Categorize the opinion fluctuation of the user into one of the following categories based on its change:
- Open to Dialogue
- Firm but Open
- Deeply Rooted
- Flexible
Return the answer as one of the corresponding dogmatism labels.

            [{data_point["comments_string_for_dogmatism"]}] = """.strip()

X_train = pd.DataFrame(X_train.apply(generate_prompt, axis=1), 
                       columns=["text"])
X_eval = pd.DataFrame(X_eval.apply(generate_prompt, axis=1), 
                      columns=["text"])

y_true = X_test[main_column]
X_test = pd.DataFrame(X_test.apply(generate_test_prompt, axis=1), columns=["text"])

train_data = Dataset.from_pandas(X_train)
eval_data = Dataset.from_pandas(X_eval)
print(X_train.shape, y_true.shape, X_test.shape)


def evaluate(y_true, y_pred):
    labels = list(np.unique(data[main_column]))
    mapping = {'Open to Dialogure': 0, 'Firm but Open': 1,
    'Deeply Rooted': 2, 'Flexible': 3}
    def map_func(x):
        return mapping.get(x, 1)
    
    y_true = np.vectorize(mapping.get)(y_true)
    y_pred = np.vectorize(mapping.get)(y_pred)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(f'Accuracy: {accuracy:.3f}')
    
    # Generate accuracy report
    unique_labels = set(y_true)  # Get unique labels
    
    for label in unique_labels:
        label_indices = [i for i in range(len(y_true)) 
                         if y_true[i] == label]
        label_y_true = [y_true[i] for i in label_indices]
        label_y_pred = [y_pred[i] for i in label_indices]
        accuracy = accuracy_score(label_y_true, label_y_pred)
        print(f'Accuracy for label {label}: {accuracy:.3f}')
        
    # Generate classification report
    class_report = classification_report(y_true=y_true, y_pred=y_pred)
    print('\nClassification Report:')
    print(class_report)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1, 2, 3, 4])
    print('\nConfusion Matrix:')
    print(conf_matrix)


# Next we need to take care of the model, which is a 7b-hf (7 billion parameters, no RLHF, in the HuggingFace compatible format).
# 
# Model loading and quantization:
# 
# * First the code loads the Llama-2 language model from the Hugging Face Hub.
# * Then the code gets the float16 data type from the torch library. This is the data type that will be used for the computations.
# * Next, it creates a BitsAndBytesConfig object with the following settings:
#     1. load_in_4bit: Load the model weights in 4-bit format.
#     2. bnb_4bit_quant_type: Use the "nf4" quantization type. 4-bit NormalFloat (NF4), is a new data type that is information theoretically optimal for normally distributed weights.
#     3. bnb_4bit_compute_dtype: Use the float16 data type for computations.
#     4. bnb_4bit_use_double_quant: Do not use double quantization (reduces the average memory footprint by quantizing also the quantization constants and saves an additional 0.4 bits per parameter.).
# * Then the code creates a AutoModelForCausalLM object from the pre-trained Llama-2 language model, using the BitsAndBytesConfig object for quantization.
# * After that, the code disables caching for the model.
# * Finally the code sets the pre-training token probability to 1.
# 
# Tokenizer loading:
# 
# * First, the code loads the tokenizer for the Llama-2 language model.
# * Then it sets the padding token to be the end-of-sequence (EOS) token.
# * Finally, the code sets the padding side to be "right", which means that the input sequences will be padded on the right side. This is crucial for correct padding direction (this is the way with Llama 2).

from huggingface_hub import HfApi, HfFolder
hf_token = ""

# Save the token (this will save the token in the Hf folder, akin to using the CLI)
HfFolder.save_token(hf_token)

# Optional: Authenticate using the token
api = HfApi()
user = api.whoami(token=hf_token)
print(f"Logged in as: {user['name']}")

# model names: 
# Llama model: "meta-llama/Llama-2-7b"
# Llama-chat model: "meta-llama/Llama-2-7b-chat-hf"
# Llama-3-instruct: "meta-llama/Meta-Llama-3-8B-Instruct"
# Falcon-7B-instruct: "tiiuae/falcon-7b-instruct"
# Vicuna: "lmsys/vicuna-7b-v1.5"

model_name = "meta-llama/Meta-Llama-3-8B"

compute_dtype = getattr(torch, "bfloat16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",use_auth_token = True,
    quantization_config=bnb_config, cache_dir="./"
)

model.config.use_cache = False
model.config.pretraining_tp = 1
#model.bfloat16()

tokenizer = AutoTokenizer.from_pretrained(model_name,use_auth_token = True, 
                                          trust_remote_code=True,cache_dir="./"
                                         )
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 
# The pipeline() function from the Hugging Face Transformers library is used to generate text from the language model. The task argument specifies that the task is text generation. The model and tokenizer arguments specify the pre-trained Llama-2 language model and the tokenizer for the language model. The max_new_tokens argument specifies the maximum number of new tokens to generate. The temperature argument controls the randomness of the generated text. A lower temperature will produce more predictable text, while a higher temperature will produce more creative and unexpected text.
# 

def predict(test, model, tokenizer):
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=5, temperature=0.1)
    
    y_pred = []
    possible_labels = ['Open to Dialogue', 'Firm but Open', 'Deeply Rooted', 'Flexible']
    
    for i in tqdm(range(len(test))):
        prompt = test.iloc[i]["text"]
        result = pipe(prompt)
        generated_text = result[0]['generated_text']
        
        # Check if the generated text contains an '=' sign
        if '=' in generated_text:
            # Extract the part following the last '=' sign
            answer = generated_text.split('=')[-1].strip()
        else:
            # If no '=' sign, set a default value
            answer = "Open to Dialogue"
        y_pred.append(generated_text)
    
    return y_pred

# In the next cell we set everything ready for the fine-tuning. We configures and initializes a Simple Fine-tuning Trainer (SFTTrainer) for training a large language model using the Parameter-Efficient Fine-Tuning (PEFT) method, which should save time as it operates on a reduced number of parameters compared to the model's overall size. The PEFT method focuses on refining a limited set of (additional) model parameters, while keeping the majority of the pre-trained LLM parameters fixed. This significantly reduces both computational and storage expenses. Additionally, this strategy addresses the challenge of catastrophic forgetting, which often occurs during the complete fine-tuning of LLMs.
# 
# PEFTConfig:
# 
# The peft_config object specifies the parameters for PEFT. The following are some of the most important parameters:
# 
# * lora_alpha: The learning rate for the LoRA update matrices.
# * lora_dropout: The dropout probability for the LoRA update matrices.
# * r: The rank of the LoRA update matrices.
# * bias: The type of bias to use. The possible values are none, additive, and learned.
# * task_type: The type of task that the model is being trained for. The possible values are CAUSAL_LM and MASKED_LM.
# 
# TrainingArguments:
# 
# The training_arguments object specifies the parameters for training the model. The following are some of the most important parameters:
# 
# * output_dir: The directory where the training logs and checkpoints will be saved.
# * num_train_epochs: The number of epochs to train the model for.
# * per_device_train_batch_size: The number of samples in each batch on each device.
# * gradient_accumulation_steps: The number of batches to accumulate gradients before updating the model parameters.
# * optim: The optimizer to use for training the model.
# * save_steps: The number of steps after which to save a checkpoint.
# * logging_steps: The number of steps after which to log the training metrics.
# * learning_rate: The learning rate for the optimizer.
# * weight_decay: The weight decay parameter for the optimizer.
# * fp16: Whether to use 16-bit floating-point precision.
# * bf16: Whether to use BFloat16 precision.
# * max_grad_norm: The maximum gradient norm.
# * max_steps: The maximum number of steps to train the model for.
# * warmup_ratio: The proportion of the training steps to use for warming up the learning rate.
# * group_by_length: Whether to group the training samples by length.
# * lr_scheduler_type: The type of learning rate scheduler to use.
# * report_to: The tools to report the training metrics to.
# * evaluation_strategy: The strategy for evaluating the model during training.
# 
# SFTTrainer:
# 
# The SFTTrainer is a custom trainer class from the PEFT library. It is used to train large language models using the PEFT method.
# 
# The SFTTrainer object is initialized with the following arguments:
# 
# * model: The model to be trained.
# * train_dataset: The training dataset.
# * eval_dataset: The evaluation dataset.
# * peft_config: The PEFT configuration.
# * dataset_text_field: The name of the text field in the dataset.
# * tokenizer: The tokenizer to use.
# * args: The training arguments.
# * packing: Whether to pack the training samples.
# * max_seq_length: The maximum sequence length.
# 
# Once the SFTTrainer object is initialized, it can be used to train the model by calling the train() method

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

training_arguments = TrainingArguments(
    output_dir="logs",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4, # 4
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    #report_to="tensorboard",
    evaluation_strategy="epoch"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    peft_config=peft_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
    max_seq_length=2048,
)


# The following code will train the model using the trainer.train() method and then save the trained model to the trained-model directory. Using The standard GPU P100 offered by Kaggle, the training should be quite fast.


# Train model
trainer.train()

# Save trained model
trainer.model.save_pretrained("trained-model-llama-dogmatism", cache_dir="./")


torch.cuda.empty_cache()
y_pred = predict(X_test, model, tokenizer)

evaluation = pd.DataFrame({'text': X_test["text"], 
                           'y_true':y_true, 
                           'y_pred': y_pred},
                         )
evaluation.to_csv("llama3_gptoneshot_dogmatism_test_predictions.csv", index=False)

