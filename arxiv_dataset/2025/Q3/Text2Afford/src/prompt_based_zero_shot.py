import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoModelForMaskedLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
import datasets
from utils import eval
import re

# Load the T5 model and tokenizer
model_name = './models/flan-t5-large-finetuned-affordance'
# model_name = 'google/flan-t5-large'
tokenizer_name = 'google/flan-t5-large'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
model.to(device)
print(model_name)
dataset = datasets.load_dataset("piqa", split='validation')
correct = 0
wrong = 0

for idx in tqdm(range(len(dataset))):
    input_format = f"Select the most appropriate option based on the situation:\n\n: {dataset['goal'][idx]} \nOPTIONS:\n-{dataset['sol1'][idx]} \n-{dataset['sol2'][idx]}"
    input_ids = tokenizer.encode(input_format, return_tensors='pt').to(device)
    output_ids = model.generate(input_ids)
    predicted_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    gt_sol = 'sol' + str(dataset['label'][idx] + 1)

    predicted_text = re.sub(' +', ' ', predicted_text.lower().strip())
    ground_truth_text = re.sub(' +', ' ', dataset[gt_sol][idx].lower().strip())
    if predicted_text in ground_truth_text:
        correct += 1
    else:
        wrong += 1
        print(predicted_text)
        print(dataset[gt_sol][idx])
        # print("#"* 50)
        print(correct/(correct + wrong))

print("Accuracy: %s"%(correct/(correct + wrong)))

# filename = "../data/ECCV_affordance_data.tsv"
# filename = "../data/toloka_annotated_data.tsv"
# filename = "../data/final_annotated_data.tsv"
# # filename = "../data/Daivik_annotated.tsv"
# filename = "../data/rare_object_annotated_data.tsv"
# filename = "../data/Rare_Objects_Sayantan_Merged.tsv"
# filename = "../data/rare_xnli.tsv"

# data = pd.read_csv(filename, sep='\t')


# label_names = list(data.columns[2:])
# oov_class_map = {"play": "play using", "lookthrough": "look through", "siton": "sit on", "pourfrom": "pour from", "writewith": "write with", "typeon": "type using"}
# labels_lower = [x.lower() for x in label_names]
# classes = []
# for lab in labels_lower:
#     if lab not in oov_class_map.keys():
#         classes.append(lab)
#     else:
#         classes.append(oov_class_map[lab])

# def predict_affordance_proba(model, sentence, object_name, max_seq_len=128):
#     """Re-ranks the candidates answers for each question.

#     Returns:
#         ranked_ans: list of re-ranked candidate docids
#         sorted_scores: list of relevancy scores of the answers
#     -------------------
#     Arguments:
#         model - PyTorch model
#         q_text - str - query
#         cands -List of retrieved candidate docids
#         max_seq_len - int
#     """
#     # Empty dict for the probability scores of affordances
#     affordance_proba = {}

#     for affordance in classes:
#         prompt = f"Answer Yes or No: {sentence}. Question: Can human {affordance} the {object_name}?"
#         # prompt = f"Context: {sentence}. Question: Can the action '{affordance}' be performed on {object_name} by human?\nOPTIONS:\n-YES \n-NO"

#         input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
#         output_ids = model.generate(input_ids)
#         predicted_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

#         if predicted_text.lower() == 'yes':
#             affordance_proba[affordance] = 1
#         else:
#             affordance_proba[affordance] = 0

    
#     return affordance_proba


# ## Initialize to calculate Accuracy
# gt_affordance = {}
# correct = 0
# wrong = 0
# avg_acc = []

# ## Initialize to calculate Mean Average Precision
# ground_truth_classes = []
# predicted_classes = []

# for ids, rows in tqdm(data.iterrows()):
# # predicted_affordances = predict_affordance_proba(model, sentence, object)
#     # if ids >=200:
#     #     break
#     positive_classes = []
#     negative_classes = []
#     sentence = rows[0]
#     object_name = rows[1]

#     predicted_affordances = predict_affordance_proba(model, sentence, object_name)
#     sorted_predicted_affordances = dict(sorted(predicted_affordances.items(), key=lambda item: item[1], reverse=True))

#     for itr in range(2, data.shape[1]):
#         class_name = list(data.columns)[itr].lower()
#         ## replacing oov classnames
#         class_name = class_name if class_name not in oov_class_map.keys() else oov_class_map[class_name]

#         gt_affordance[class_name] = rows[itr]

#         if rows[itr] > 0:
#             positive_classes.append(class_name)
#         else:
#             negative_classes.append(class_name)

#     # print(object_name)
#     # print(predicted_affordances)
#     # print(gt_affordance)

#     prev_cor = correct
#     prev_wrong = wrong
    
#     if len(positive_classes) > 0:
#         for pos in positive_classes:
#             for neg in negative_classes:
#                 if predicted_affordances[pos] > predicted_affordances[neg]:
#                     correct += 1
#                 else:
#                     wrong += 1

#         avg_acc.append((correct-prev_cor)/(correct+wrong - prev_cor-prev_wrong))

#     ground_truth_classes.append(positive_classes)
#     predicted_classes.append(list(sorted_predicted_affordances.keys()))


# accuracy = correct / (correct+wrong)

# print("Accuracy: %s"%accuracy )
# print("MAP: %s" %eval.mapk(ground_truth_classes, predicted_classes, 15))
# print("Accuracy: %s"%(sum(avg_acc)/len(avg_acc)))