from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_id = "saved_model/checkpoint-500/"
# model_id = "saved_model/llama-3-8B-instruct-5/"
# model_id = "saved_model/llama-3-8B-instruct_few_shot_prost/"
model_id = "/home/student/heisenberg/Affordance/data_utility/saved_model/llama-3-8B-instruct_few_shot_piqa/"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    load_in_4bit=True,
)


################### For Prost ###############
# dataset = load_dataset("corypaik/prost", split="test")

# prompt_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

# ### Instruction:
# You are an AI assistant that has strong reasoning capability. You are given a question with 4 options and you have to choose the right option. 

# ### Question:
# {question}

# ### Options:
# [0] {option_A}
# [1] {option_B}
# [2] {option_C}
# [3] {option_D}

# Only response the 'answer id'. For example if the answer is [0] then response 0. DO NOT respond anything other than <0, 1, 2, 3>.

# ### Answer:
# """

##################### For Piqa ###########################
dataset = load_dataset("ybisk/piqa", split="validation")

prompt_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
You are an AI assistant that has strong reasoning capability. You are given a situation and asked to choose the most appropriate option from given two options. 

### Situation:
{situation}

### Options:
[0] {option_A}
[1] {option_B}

Only response the 'answer id'. For example if the answer is [0] then response 0. DO NOT respond anything other than <0, 1>.

### Answer:
"""

if tokenizer.pad_token is None:
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

def preprocess_llama_prost(examples, max_length=256):

    prompts = []
    messages = []
    # input_id_list = torch.empty([])
    # input_id_list = []

    for question, option_A, option_B, option_C, option_D, label in zip(examples["ex_question"], examples["A"], examples["B"], examples["C"], examples["D"], examples["label"]):
        prompt = prompt_template.format(question = question, 
                                        option_A=option_A,
                                        option_B=option_B, 
                                        option_C=option_C, 
                                        option_D=option_D)
        message = [
        {"role": "system", "content": "You are a chatbot who always responds to the query!"},
        {"role": "user", "content": prompt},
        ]

        

        # prompts.append(prompt)
        messages.append(message)

    # input_ids = tokenizer.apply_chat_template(
    #         messages,
    #         add_generation_prompt=True,
    #         padding="max_length",
    #         max_length=max_length,
    #         return_tensors="pt",
    #     ).to(model.device)
    

    ## Converting to torch
    # tensor_ids = torch.empty((len(input_id_list[0]), 512)).to(model.device)
    # for id_list in input_id_list:
    #     tensor_ids = torch.stack((tensor_ids, id_list)).to(model.device)
    # input_id_list = torch.stack(input_id_list)
        
    
    return {"messages": messages}


def preprocess_llama_piqa(examples, max_length=256):

    prompts = []
    messages = []
    # input_id_list = torch.empty([])
    # input_id_list = []

    for situation, option_A, option_B, label in zip(examples["goal"], examples["sol1"], examples["sol2"], examples["label"]):
        prompt = prompt_template.format(situation = situation, 
                                        option_A=option_A,
                                        option_B=option_B)
        message = [
        {"role": "system", "content": "You are a chatbot who always responds to the query!"},
        {"role": "user", "content": prompt},
        ]

        
        messages.append(message)
    
    return {"messages": messages}
    


# dataset = dataset.select(range(1000))
dataset = dataset.map(preprocess_llama_piqa, batched=True)
dataset = dataset.with_format("torch")

print(dataset[0])

# test_set = dataset[:100]

# terminators = [
#       tokenizer.eos_token_id,
#       tokenizer.convert_tokens_to_ids("<|eot_id|>")
#   ]

# tokenizer.padding_side = "right" 
# with torch.no_grad():
#     outputs = model.generate(
#         test_set["input_ids"].to(model.device),
#         max_new_tokens=20,
#         eos_token_id=terminators,
#         do_sample=True,
#         temperature=0.6,
#         top_p=0.9,
#         # batch_size=20
#     )
# generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=False)

# print(generated_texts)


from tqdm import tqdm

predictions = []
test_set = dataset

retry_limit = 3
# for (message, group) in tqdm(zip(test_set["messages"], test_set["group"])):

for message in tqdm(test_set["messages"]):

    # print(items)
    # message = items["messages"]
    # print(message)
    input_ids = tokenizer.apply_chat_template(
        message,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    retry = 0
    while retry < retry_limit:
        outputs = model.generate(
            input_ids,
            max_new_tokens=5,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        answer = tokenizer.decode(response, skip_special_tokens=True)
        

        if answer.strip() in ['0', '1', '2', '3', 0, 1, 2, 3]: ## Checking whether answer within the options
            # predictions.append([int(answer), group]) ## To analyze group wise predictions
            predictions.append(int(answer))
            break
        retry += 1
    
    print(retry)
    print(answer)
    if retry == retry_limit:
        answer = 0
        # predictions.append([answer, group]) ## append random answer
        predictions.append(answer)

    # print(predictions)

print(predictions)

import json

# with open('predictions_finetuned_affordance_llama3.json', 'w') as f:
#     json.dump(predictions, f)

# predictions = [int(pred) for pred, group in predictions]

from sklearn.metrics import accuracy_score
print(test_set['label'])
accuracy = accuracy_score(test_set['label'], predictions)
print(f"Accuracy: {accuracy:.2f}")
