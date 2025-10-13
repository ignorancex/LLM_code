import math
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
import datasets
import re, os
from transformers import  BitsAndBytesConfig


device = "cuda" if torch.cuda.is_available() else "cpu"

# model_id = "/home/student/heisenberg/Affordance/data_utility/saved_model/llama-3-8B-instruct_few_shot_text2afford/"
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    load_in_4bit=True,
)

# You are an AI assistant that has strong reasoning capability. You are given a context containing an object, and you are asked to answer a question about the object based on the context. Just response 'Yes' or 'No'

### Context:
# {}



### Example questions and answers:
# {}

alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
You are an AI assistant that has strong reasoning capability. You are given a context containing an object, and you are asked to answer a question about the object based on the context. Just response 'Yes' or 'No'

### Context:
{}

### Object:
{}

### Example questions and answers:
{}

### Question:
{}

### Answer: """


filename = "../../data/final_normal_dataset.tsv"
# filename = "/home/student/heisenberg/Affordance/data/ECCV_affordance_data.tsv"
filename = "/home/student/heisenberg/Affordance/data/data_high_agreement.tsv"

data = pd.read_csv(filename, sep='\t')

print(data.isna().sum())

label_names = list(data.columns[2:])
oov_class_map = {"play": "play", "lookthrough": "look through", "siton": "sit on", "pourfrom": "pour from", "writewith": "write with", "typeon": "type on"}
labels_lower = [x.lower() for x in label_names]
classes = []
for lab in labels_lower:
    if lab not in oov_class_map.keys():
        classes.append(lab)
    else:
        classes.append(oov_class_map[lab])

incontext_examples = """Example 1: can human ride bicycle? Answer: Yes

Example 2: can human lift donkey? Answer: No

Example 3: can human watch the tower? Answer: No

Example 4: can human ride violin? Answer: No

Example 5: can human write with chalk? Answer: Yes """

def predict_affordance_proba(model, sentence, object_name):
    # Empty dict for the probability scores of affordances
    affordance_proba = {}

    for affordance in classes:

        question = f"Can human '{affordance}' the {object_name}?"
        prompt = alpaca_prompt.format(sentence, object_name, incontext_examples, question)

        message = [
        {"role": "system", "content": "You are a chatbot who always responds to the query!"},
        {"role": "user", "content": prompt},
        ]

        input_ids = tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = model.generate(
            input_ids,
            max_new_tokens=5,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        predicted_answer = tokenizer.decode(response, skip_special_tokens=True)
        
        print(predicted_answer)
        if 'yes' in predicted_answer.lower():
            affordance_proba[affordance] = 1
        else:
            affordance_proba[affordance] = 0

    
    return affordance_proba


## Initialize to calculate Accuracy
gt_affordance = {}
correct = 0
wrong = 0
avg_acc = []

## Initialize to calculate Mean Average Precision
ground_truth_classes = []
predicted_classes = []

# f = open("result.txt", "w")

all_predictions = []

for ids, rows in tqdm(data.iterrows()):
# predicted_affordances = predict_affordance_proba(model, sentence, object)
    if ids < 1868:
        continue
    positive_classes = []
    negative_classes = []
    sentence = rows[0]
    object_name = rows[1]

    predicted_affordances = predict_affordance_proba(model, sentence, object_name)
    sorted_predicted_affordances = dict(sorted(predicted_affordances.items(), key=lambda item: item[1], reverse=True))

    all_predictions.append(predicted_affordances)

    for itr in range(2, data.shape[1]):
        class_name = list(data.columns)[itr].lower()
        ## replacing oov classnames
        class_name = class_name if class_name not in oov_class_map.keys() else oov_class_map[class_name]

        gt_affordance[class_name] = rows[itr]

        if not math.isnan(rows[itr]):
            if rows[itr] > 0:
                positive_classes.append(class_name)
            else:
                negative_classes.append(class_name)
        else:
            print("Nan found")

    # print(object_name)
    # print(predicted_affordances)
    # print(gt_affordance)

    prev_cor = correct
    prev_wrong = wrong
    
    print(positive_classes)
    print(negative_classes)
    if (len(positive_classes) > 0) and (len(negative_classes) > 0):
        for pos in positive_classes:
            for neg in negative_classes:
                if predicted_affordances[pos] > predicted_affordances[neg]:
                    correct += 1
                else:
                    wrong += 1

        acc = (correct-prev_cor)/(correct+wrong - prev_cor-prev_wrong)
        avg_acc.append(acc)

        print(acc)

        if acc < 0.3:
            # f.write(f"Sentence: {sentence}, Object: {object_name} \n Positives: {positive_classes} \n Predictions: {predicted_affordances}")
            print(f"Sentence: {sentence}, Object: {object_name} \n Positives: {positive_classes} \n Predictions: {predicted_affordances}")

    ground_truth_classes.append(positive_classes)
    predicted_classes.append(list(sorted_predicted_affordances.keys()))


accuracy = correct / (correct+wrong)

# f.close()
print("Accuracy: %s"%accuracy )
# print("MAP: %s" %eval.mapk(ground_truth_classes, predicted_classes, 15))
print("Accuracy: %s"%(sum(avg_acc)/len(avg_acc)))

df = pd.DataFrame(all_predictions)

print(df.head())

df.to_csv("predictions_llama_3_finetuned.tsv", sep="\t")