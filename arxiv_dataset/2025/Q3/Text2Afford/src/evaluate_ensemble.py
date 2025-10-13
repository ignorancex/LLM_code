import numpy as np
import pandas as pd
import pickle

from utils import eval


# filename = "../data/ECCV_affordance_data.tsv"
# filename = "../data/toloka_annotated_data.tsv"
filename = "../data/final_annotated_data.tsv"
# filename = "../data/Daivik_annotated.tsv"
# filename = "../data/rare_object_annotated_data.tsv"
# filename = "../data/Rare_Objects_Sayantan_Merged.tsv"
# filename = "../data/rare_xnli.tsv"

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



result_files = ['./results/predictions_nli_roberta.pkl', './results/predictions_nli_bart.pkl', './results/predictions_MLM_roberta-large.pkl', 
                './results/predictions_vilt.pkl', './results/predictions_clip.pkl']

ensemble_weight = 0.59
res_file1 = result_files[1]
res_file2 = result_files[-2]

print(f"Result1: {res_file1}, Result2: {res_file2}")

with open(res_file1, 'rb') as content:
    result1 = pickle.load(content)

with open(res_file2, 'rb') as content:
    result2 = pickle.load(content)

print(f"Result1 objects: {len(result1)}, Result2 obects: {len(result2)}")

## Initialize to calculate Accuracy
gt_affordance = {}
correct = 0
wrong = 0
avg_acc = []

## Initialize to calculate Mean Average Precision
ground_truth_classes = []
predicted_classes = []

## For calculating AUC-ROC
true_labels_list = []
predicted_scores_list = []



for ids, rows in data.iterrows():

    proba1 = result1[ids]
    proba2 = result2[ids]

    positive_classes = []
    negative_classes = []

    ensemble_prediction = {}
    for keys, values in proba1.items():
        ensemble_prediction[keys] = (1-ensemble_weight)*proba1[keys] + ensemble_weight*proba2[keys]

    sorted_predicted_affordances = dict(sorted(ensemble_prediction.items(), key=lambda item: item[1], reverse=True))

    for itr in range(2, data.shape[1]):
        class_name = list(data.columns)[itr].lower()
        ## replacing oov classnames
        class_name = class_name if class_name not in oov_class_map.keys() else oov_class_map[class_name]

        gt_affordance[class_name] = rows[itr]

        if rows[itr] > 0:
            positive_classes.append(class_name)
        else:
            negative_classes.append(class_name)

    per_itr_correct = 0
    per_itr_wrong = 0
    
    if len(positive_classes) > 0:
        for pos in positive_classes:
            for neg in negative_classes:
                if ensemble_prediction[pos] >= ensemble_prediction[neg]:
                    correct += 1
                    per_itr_correct += 1
                else:
                    wrong += 1
                    per_itr_wrong +=1 

        per_itr_accuracy = per_itr_correct/(per_itr_correct+per_itr_wrong)

        # print(per_itr_accuracy)
        avg_acc.append(per_itr_accuracy)

        ground_truth_classes.append(positive_classes)
        predicted_classes.append(list(sorted_predicted_affordances.keys()))

        true_labels_list.append(list(gt_affordance.values()))
        predicted_scores_list.append(list(ensemble_prediction.values()))
    
    # if ids > 15:
    #     break

accuracy = correct / (correct+wrong)

print("Accuracy: %s"%accuracy )
print("MAP: %s" %eval.mapk(ground_truth_classes, predicted_classes, 15))
# print("Accuracy: %s"%(sum(avg_acc)/len(avg_acc)))
eval.calculate_auc_roc(true_labels_list, predicted_scores_list)