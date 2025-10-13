import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import datasets
import re, os
from transformers import  BitsAndBytesConfig
import openai
openai.api_type = "azure"
openai.api_base = "https://gpt35newdec23.openai.azure.com/"
openai.api_version = "2023-09-15-preview"
openai.api_key = ""

device = "cuda" if torch.cuda.is_available() else "cpu"



filename = "../data/ECCV_affordance_data.tsv"
filename = "../data/toloka_annotated_data.tsv"
filename = "../data/final_annotated_data.tsv"
# filename = "../data/Daivik_annotated.tsv"
filename = "../data/rare_object_annotated_data.tsv"
filename = "../data/Rare_Objects_Sayantan_Merged.tsv"
filename = "../data/rare_xnli.tsv"
filename = "../../data/final_normal_dataset.tsv"

data = pd.read_csv(filename, sep='\t')


label_names = list(data.columns[2:])
oov_class_map = {"play": "play using", "lookthrough": "look through", "siton": "sit on", "pourfrom": "pour from", "writewith": "write with", "typeon": "type using"}
labels_lower = [x.lower() for x in label_names]
classes = []
for lab in labels_lower:
    if lab not in oov_class_map.keys():
        classes.append(lab)
    else:
        classes.append(oov_class_map[lab])

incontext_examples = """Example 1: can human ride bicycle? 
Answer: Yes

Example 2: can human write with chalk? 
Answer: Yes

Example 3: can human grasp dog? 
Answer: No

Example 4: can human ride violin? 
Answer: No

Example 5: can human lift donkey? 
Answer: No"""

def predict_affordance_proba(sentence, object_name):
    # Empty dict for the probability scores of affordances
    affordance_proba = {}

    for affordance in classes:
        template = f"{incontext_examples}\n\nConsider the sentence - '{sentence}'. Now, can human {affordance} the {object_name}?\nAnswer Yes or No: "

        response = openai.Completion.create(
                    engine="gpt35tdec23",
                    prompt=template,
                    temperature=1,
                    max_tokens=1,
                    top_p=0.5,
                    frequency_penalty=0.2,
                    presence_penalty=0,
                    stop=None)
        
        predicted_answer = response["choices"][0]["text"].strip()
        
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
    # if ids < 1868:
    #     continue
    # if ids > 1968:
    #     break
    positive_classes = []
    negative_classes = []
    sentence = rows[0]
    object_name = rows[1]

    predicted_affordances = predict_affordance_proba(sentence, object_name)
    sorted_predicted_affordances = dict(sorted(predicted_affordances.items(), key=lambda item: item[1], reverse=True))

    all_predictions.append(predicted_affordances)

    for itr in range(2, data.shape[1]):
        class_name = list(data.columns)[itr].lower()
        ## replacing oov classnames
        class_name = class_name if class_name not in oov_class_map.keys() else oov_class_map[class_name]

        gt_affordance[class_name] = rows[itr]

        if rows[itr] > 0:
            positive_classes.append(class_name)
        else:
            negative_classes.append(class_name)

    # print(object_name)
    # print(predicted_affordances)
    # print(gt_affordance)

    prev_cor = correct
    prev_wrong = wrong
    
    if len(positive_classes) > 0:
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



##### Calculating error rate per class
data_gt = data.tail(df.shape[0]).iloc[:, 2:]

df.columns = data_gt.columns
print(df.head())

df.to_csv("predictions_chatGPT.tsv", sep="\t")

data_gt.reset_index(drop=True, inplace=True)
data_gt.to_csv("ground_truth.tsv", sep="\t")

# Calculating error rate per label
error_rates = (data_gt != df).mean()

print("Error rates per label:")
print(error_rates)
