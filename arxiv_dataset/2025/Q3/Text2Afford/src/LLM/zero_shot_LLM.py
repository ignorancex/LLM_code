import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import datasets
import re, os
from transformers import  BitsAndBytesConfig


device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = "google/flan-t5-xxl"
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    quantization_config=quantization_config,
    device_map="auto"
)

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

incontext_examples = """Example 1: can human ride bicycle? Answer: Yes

Example 2: can human write with chalk? Answer: Yes

Example 3: can human grasp dog? Answer: No

Example 4: can human ride violin? Answer: No

Example 5: can human lift donkey? Answer: No"""

def predict_affordance_proba(model, sentence, object_name):
    # Empty dict for the probability scores of affordances
    affordance_proba = {}

    for affordance in classes:
        template = f"{incontext_examples}\n\nconsider '{sentence}'. Now, can human '{affordance}' the '{object_name}'? Answer Yes or No: "

        inputs = tokenizer(template, return_tensors="pt")

        generated_ids = model.generate(**inputs, max_new_tokens=20)
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        predicted_answer = generated_text[0]
        
        # print(predicted_answer)
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

print(df.head())

df.to_csv("predictions_icontext.tsv", sep="\t")