#!/usr/bin/env python
# coding: utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TRANSFORMERS_CACHE'] = "./"
import pandas as pd
from sklearn.model_selection import KFold
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.metrics import accuracy_score
import pandas as pd
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import os
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          HfArgumentParser,
                          Trainer,
                          TrainingArguments,
                          DataCollatorForLanguageModeling,
                          EarlyStoppingCallback,
                          pipeline,
                          logging,
                          set_seed)

#import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, AutoPeftModelForCausalLM
from trl import SFTTrainer
import numpy as np
import pandas as pd
import pickle
from datasets import Dataset


new_model = "llama-3-usdc"
data = pd.read_csv('instructions_user.csv')

def create_prompt_formats(data_point):
    
    """
    Creates a formatted prompt template for a prompt in the instruction dataset

    :param sample: Prompt or sample from the instruction dataset
    """
    
    # Initialize static strings for the prompt template
    intro_blurb_template = """ 
    ### Introduction
    **Objective**: Analyze Reddit conversations enclosed in square bracketsto identify the stance of specific authors on sociopolitical topics and determine their level of dogmatism.
    **Stance Defintion**: Stance is defined as the expression of the author's standpoint and judgement towards a given topic.
    **Dogmatism Defintion**:  Dogmatism is an opinion strongly believed as a fact to support a stance without a question or allowance for conversation.
    **Task**: Given a JSON formatted Reddit post and its comment thread, classify the stance of text segments related to "author1" and "author2" by assigning one of the following five predefined stance labels: `strongly_against`, `somewhat_against`, `somewhat_in_favor`, `strongly_in_favor`, `stance_not_inferrable`. Also, assign a dogmatism label for each author by assigning one of the following four predefined labels: `Deeply Rooted`, `Firm but Open`, `Open to Dialogue`, `Flexible`
    
    ### Stance Labels Description:
    1. **strongly_aganist / strongly_in_favor**: Mark text showing strong opinions, emotional expressions, or argumentative tones.
    2. **somewhat_against / somewhat_in_favor**: Identify texts with openness to discussion, less certainty, or showing interest in different viewpoints.
    3. **stance_not_inferrable**: Use for texts that are neutral, support both stances, or where the stance is unclear despite being on-topic.
    
    ### Dogmatism Labels Description:
    1. **Deeply Rooted**: Reflects a strong, unchangeable belief. This label conveys the idea of someone who is firm in their opinion and unlikely to be swayed.
    2. **Firm but Open**: Indicates a person who is not likely to change their mind but does not impose their views authoritatively. It captures the essence of being steadfast in one’s beliefs without being dismissive of others.
    3. **Open to Dialogue**: Describes someone who holds a certain opinion but is genuinely interested in considering other viewpoints. This label suggests a willingness to engage in meaningful conversation about differing perspectives.
    4. **Flexible**: Denotes a person who is not firmly committed to their stance and is open to changing their opinion. This label is indicative of flexibility and openness to new information or arguments.
    
    ### Input Data Format
    The input data will be in JSON format and should include several key elements to represent a Reddit submission and its associated comments. Each element provides specific information as described below:
    
    - `id`: This is the unique identifier for the Reddit submission.
    - `title`: The title of the post. This is what users see first and often summarizes or hints at the content of the submission.
    - `content`: The main post's detailed description. This text segment provides the core message or information the author wishes to communicate with the Reddit community. It may include narratives, questions, or any information relevant to the title.
    - `author1` or `author2`: The username of our focus author. This field is applicable if the post or comment is made by one of the specific authors we are tracking in the dataset.
    - `comments`: An array (list) of comments related to the Reddit submission. Each comment in this array includes the following fields:
        - `id`: The unique identifier for the comment, allowing for identification and reference within the dataset.
        - `author1` or `author2`: The username of the comment's author, if it is made by one of our focus authors. This helps in tracking contributions by specific individuals.
        - `body`: The text of the comment. This is the main content of the comment where the author responds to the post or another comment, providing insights, opinions, or further information.
        - `replies`: An array of comments that are direct responses to this comment. The structure of each reply follows the same format as the initial comment, including `id`, `author1` or `author2` (if applicable), `body`, and potentially more `replies`.
    
    ### Output Data Format
    Submit your annotations in JSON format, grouping all stance annotations under the key "stance_annotations". Each entry should be a dictionary containing the segment's "id", your "label", and the "reason" for your choice. Include the dogmatism label and its justification under "dogmatism_label" and "dogmatism_reason", respectively.
    
    The output should follow this structure:
    json
    {{
      "author1": {{
        "name": "[author_name]",
        "stance_annotations": [
          {{
            "id": "[segment_id]",
            "label": "[chosen_label]",
            "reason": "[Justification in <50 words]"
          }},
          ...
        ],
        "dogmatism_label": "[chosen_dogmatism_label]",
        "dogmatism_reason": "[Justification in <50 words]"
      }},
      "author2": {{
        "name": "[author_name]",
        "stance_annotations": [
          {{
            "id": "[segment_id]",
            "label": "[chosen_label]",
            "reason": "[Justification in <50 words]"
          }},
          ...
        ],
        "dogmatism_label": "[chosen_dogmatism_label]",
        "dogmatism_reason": "[Justification in <50 words]"
      }}
    }}
    
    ### Instructions for Effective Annotation
    
    1. **Labeling Stance**: For each segment (including the original Reddit submission, comments, or replies) where "author1" or "author2" is mentioned, assign a stance label that best represents the stance expressed towards the discussed topic in the conversation. This comprehensive approach ensures no relevant contribution by "author1" or "author2" is overlooked. Evaluate the stance based on the content's tone, argumentation, and engagement level with the topic.
    2. **Providing Justification**: For each label assigned, include a concise reason, aiming for less than 50 words. Focus on the stance and argumentative indicators present in the text. 
    3. **Dogmatism Assessment**: After reviewing all segments from "author1" and "author2", assign a single dogmatism label reflecting the overall tone and approach in their contributions."""
    
    # Final f-string
    INTRO_BLURB = f"""{intro_blurb_template}
        [{data_point["gpt41106preview_zero_shot_promptmessages"]}] = {data_point["gpt41106preview_zero_shot_output_reordered"]}
    """.strip()
    return INTRO_BLURB

def create_prompt_formats_test(data_point):
    
    """
    Creates a formatted prompt template for a prompt in the instruction dataset

    :param sample: Prompt or sample from the instruction dataset
    """
    
    # Initialize static strings for the prompt template
    intro_blurb_template = """ 
    ### Introduction
    **Objective**: Analyze Reddit conversations enclosed in square bracketsto identify the stance of specific authors on sociopolitical topics and determine their level of dogmatism.
    **Stance Defintion**: Stance is defined as the expression of the author's standpoint and judgement towards a given topic.
    **Dogmatism Defintion**:  Dogmatism is an opinion strongly believed as a fact to support a stance without a question or allowance for conversation.
    **Task**: Given a JSON formatted Reddit post and its comment thread, classify the stance of text segments related to "author1" and "author2" by assigning one of the following five predefined stance labels: `strongly_against`, `somewhat_against`, `somewhat_in_favor`, `strongly_in_favor`, `stance_not_inferrable`. Also, assign a dogmatism label for each author by assigning one of the following four predefined labels: `Deeply Rooted`, `Firm but Open`, `Open to Dialogue`, `Flexible`
    
    ### Stance Labels Description:
    1. **strongly_aganist / strongly_in_favor**: Mark text showing strong opinions, emotional expressions, or argumentative tones.
    2. **somewhat_against / somewhat_in_favor**: Identify texts with openness to discussion, less certainty, or showing interest in different viewpoints.
    3. **stance_not_inferrable**: Use for texts that are neutral, support both stances, or where the stance is unclear despite being on-topic.
    
    ### Dogmatism Labels Description:
    1. **Deeply Rooted**: Reflects a strong, unchangeable belief. This label conveys the idea of someone who is firm in their opinion and unlikely to be swayed.
    2. **Firm but Open**: Indicates a person who is not likely to change their mind but does not impose their views authoritatively. It captures the essence of being steadfast in one’s beliefs without being dismissive of others.
    3. **Open to Dialogue**: Describes someone who holds a certain opinion but is genuinely interested in considering other viewpoints. This label suggests a willingness to engage in meaningful conversation about differing perspectives.
    4. **Flexible**: Denotes a person who is not firmly committed to their stance and is open to changing their opinion. This label is indicative of flexibility and openness to new information or arguments.
    
    ### Input Data Format
    The input data will be in JSON format and should include several key elements to represent a Reddit submission and its associated comments. Each element provides specific information as described below:
    
    - `id`: This is the unique identifier for the Reddit submission.
    - `title`: The title of the post. This is what users see first and often summarizes or hints at the content of the submission.
    - `content`: The main post's detailed description. This text segment provides the core message or information the author wishes to communicate with the Reddit community. It may include narratives, questions, or any information relevant to the title.
    - `author1` or `author2`: The username of our focus author. This field is applicable if the post or comment is made by one of the specific authors we are tracking in the dataset.
    - `comments`: An array (list) of comments related to the Reddit submission. Each comment in this array includes the following fields:
        - `id`: The unique identifier for the comment, allowing for identification and reference within the dataset.
        - `author1` or `author2`: The username of the comment's author, if it is made by one of our focus authors. This helps in tracking contributions by specific individuals.
        - `body`: The text of the comment. This is the main content of the comment where the author responds to the post or another comment, providing insights, opinions, or further information.
        - `replies`: An array of comments that are direct responses to this comment. The structure of each reply follows the same format as the initial comment, including `id`, `author1` or `author2` (if applicable), `body`, and potentially more `replies`.
    
    ### Output Data Format
    Submit your annotations in JSON format, grouping all stance annotations under the key "stance_annotations". Each entry should be a dictionary containing the segment's "id", your "label", and the "reason" for your choice. Include the dogmatism label and its justification under "dogmatism_label" and "dogmatism_reason", respectively.
    
    The output should follow this structure:
    json
    {{
      "author1": {{
        "name": "[author_name]",
        "stance_annotations": [
          {{
            "id": "[segment_id]",
            "label": "[chosen_label]",
            "reason": "[Justification in <50 words]"
          }},
          ...
        ],
        "dogmatism_label": "[chosen_dogmatism_label]",
        "dogmatism_reason": "[Justification in <50 words]"
      }},
      "author2": {{
        "name": "[author_name]",
        "stance_annotations": [
          {{
            "id": "[segment_id]",
            "label": "[chosen_label]",
            "reason": "[Justification in <50 words]"
          }},
          ...
        ],
        "dogmatism_label": "[chosen_dogmatism_label]",
        "dogmatism_reason": "[Justification in <50 words]"
      }}
    }}
    
    ### Instructions for Effective Annotation
    
    1. **Labeling Stance**: For each segment (including the original Reddit submission, comments, or replies) where "author1" or "author2" is mentioned, assign a stance label that best represents the stance expressed towards the discussed topic in the conversation. This comprehensive approach ensures no relevant contribution by "author1" or "author2" is overlooked. Evaluate the stance based on the content's tone, argumentation, and engagement level with the topic.
    2. **Providing Justification**: For each label assigned, include a concise reason, aiming for less than 50 words. Focus on the stance and argumentative indicators present in the text. 
    3. **Dogmatism Assessment**: After reviewing all segments from "author1" and "author2", assign a single dogmatism label reflecting the overall tone and approach in their contributions."""
    
    # Final f-string
    INTRO_BLURB = f"""{intro_blurb_template}
        [{data_point["gpt41106preview_zero_shot_promptmessages"]}] = """.strip()
    return INTRO_BLURB


X_train, X_test  = train_test_split(data, test_size=0.2,random_state=42)
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)

X_train = pd.DataFrame(X_train.apply(create_prompt_formats, axis=1), columns=["text"])
X_eval = pd.DataFrame(X_test.apply(create_prompt_formats, axis=1), columns=["text"])
X_test = pd.DataFrame(X_test.apply(create_prompt_formats_test, axis=1), columns=["text"])
train_data = Dataset.from_pandas(X_train)
test_data = Dataset.from_pandas(X_test)
eval_data = Dataset.from_pandas(X_eval)

torch.cuda.empty_cache()

from huggingface_hub import HfApi, HfFolder
hf_token = ""

# Save the token (this will save the token in the Hf folder, akin to using the CLI)
HfFolder.save_token(hf_token)

# Optional: Authenticate using the token
api = HfApi()
user = api.whoami(token=hf_token)
print(f"Logged in as: {user['name']}")

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

tokenizer = AutoTokenizer.from_pretrained(model_name,use_auth_token = True, 
                                          trust_remote_code=True,cache_dir="./"
                                         )
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def predict(test, model, tokenizer):
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=2048, temperature=0.1)
    
    y_pred = []
    for i in tqdm(range(len(test))):
      prompt = test.iloc[i]["text"]
      result = pipe(prompt)
      generated_text = result[0]['generated_text']
      y_pred.append(generated_text)
    
    return y_pred

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

training_arguments = TrainingArguments(
    output_dir="./results_llama3",
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8, # 4
    optim="paged_adamw_32bit",
    save_steps=0,
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
    max_seq_length=4096,
)
trainer.train()

trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)

y_pred = predict(X_test, model, tokenizer)

evaluation = pd.DataFrame({'text': X_test["text"], 
                           'y_true':X_test["gpt41106preview_zero_shot_output_reordered"], 
                           'y_pred': y_pred},
                         )
evaluation.to_csv("llama3_gptzeroshot_instructiontuned_test_predictions.csv", index=False)
