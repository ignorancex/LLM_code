import os 
import sys
import torch
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import  confusion_matrix, roc_curve, accuracy_score, precision_recall_fscore_support, auc,precision_recall_curve, average_precision_score


folder = "/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/code/eur-journal/results_JAMA_aug2023/labelled/DenseNet264_cc_65_JAMA_aug23/"

#get all folders in the directory
subdirs = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
#sort the folders
subdirs.sort()

#iterate over all folders 

csvs = []
roc_curves = []
prc_curves = []

val_csvs = [] 



def compute_roc(predictions, labels):
    _fpr, _tpr, _ = roc_curve(labels, predictions,pos_label=1)
    roc_auc = auc(_fpr, _tpr)
    return roc_auc, _fpr, _tpr, _


def compute_prc(predictions, labels):

    precisions, recalls, thresholds = precision_recall_curve(labels, predictions,pos_label=1)
    auprc = average_precision_score(labels, predictions)
    return auprc, precisions, recalls, thresholds   


for subdir in subdirs:

    path = os.path.join(folder,subdir) 
    csv = pd.read_csv(path+"/test_without_thresh_calc_prediction.csv") 
    csvs.append(csv)

    val_csv = pd.read_csv(path+"/val_without_thresh_calc_prediction.csv")
    val_csvs.append(val_csv)
    
    #read roc curve dataframes
    roc_curve_ = pd.read_csv(path+"/test_roccurve.csv") 

    #add column CV fold
    roc_curve_["CV_fold"] = "CV " + str(int(subdir.split("_")[-1]))
    roc_curves.append(roc_curve_)

    #read prc curve dataframes
    prc_curve_ = pd.read_csv(path+"/test_without_thresh_calc_prcurve.csv")

    #add column CV fold
    prc_curve_["CV_fold"] = "CV " + str(int(subdir.split("_")[-1]))
    prc_curves.append(prc_curve_)


#concatenate all roc curve into one dataframe by only taking columns FPR, TPR
roc_curves = pd.concat(roc_curves)
roc_curves = roc_curves[["FPR","TPR","CV_fold"]]

#concatenate all prc curve into one dataframe by only taking columns Precision, Recall
prc_curves = pd.concat(prc_curves)
prc_curves = prc_curves[["Prec_PRCurve","Rec_PRCurve","CV_fold"]]


labels = []
confidence = []
confidence_all = []

std_confidence = []
per_disease_labels = []
for id,smax,label,disease_label in zip(csvs[0]["Disease_to_patient_id"],csvs[0]["smax"],csvs[0]["label"],csvs[0]["disease_label"]):

    labels.append(label)
    per_disease_labels.append(disease_label)

    # select row with same id and smax value in all csvs
    conf = []
    for csv in csvs:

        row = csv.loc[(csv['Disease_to_patient_id'] == id) & (csv['smax'] == smax)]
        conf.append(row["confidence"].values[0])
    
    conf_mean = np.mean(conf)
    conf_std = np.std(conf) 
    confidence_all.append(conf)
    confidence.append(conf_mean)
    
    #print(np.sort(conf)[-3:])
    std_confidence.append(conf_std)

    #print(smax,conf_mean,conf_std)

confidence_all = np.array(confidence_all)
#calculate  AUROC, AUPRC, F1, Precision, Recall, Sensitivity, Specificity, Accuracy  for each CV fold
confidence = np.array(confidence)


auprc_csvs = []
for i in range(len(csvs)):

    #AUROC
    #auroc = roc_auc_score(labels,(np.array(confidence_all)[:,i] > 0.5).astype(int)) 
    auroc = compute_roc(confidence_all[:,i],labels)[0] 
    aucpr = compute_prc(confidence_all[:,i],labels)[0] 
    auprc_csvs.append(aucpr)
    f1 = f1_score(labels,(np.array(confidence_all)[:,i] > 0.5).astype(int))
    precision,recall,f1,_  = precision_recall_fscore_support(labels, (confidence_all[:,i] > 0.5).astype(int) ,pos_label=1, average='binary')
    accuracy = accuracy_score(labels,(confidence_all[:,i] > 0.5).astype(int))

    tn, fp, fn, tp = confusion_matrix(labels, (confidence_all[:,i] > 0.5).astype(int),labels=[0,1]).ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp+fn)
    # specificity 
    #confusion_matrix = confusion_matrix(labels,(confidence_all[:,i] > 0.5).astype(int))
    #specificity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[0,1]) 
    #sensitivity = confusion_matrix[1,1]/(confusion_matrix[1,0]+confusion_matrix[1,1])


    print("CV fold: ",i+1)
    print("AUROC: ",auroc)
    print("AUCPR: ",aucpr)
    print("F1: ",f1)
    print("Precision: ",precision)
    print("Recall: ",recall)
    print("Accuracy: ",accuracy)
    print("Sensitivity: ",sensitivity)
    print("Specificity: ",specificity)
    print("\n")

   

# find best and worst 3 CV folds
auprc_csvs = np.array(auprc_csvs)
best_3 = np.argsort(auprc_csvs)[-3:]
worst_3 = np.argsort(auprc_csvs)[:3]

#choose confidence values for best and worst 3 CV folds
confidence_best_3 = confidence_all[:,best_3] 
confidence_worst_3 = confidence_all[:,worst_3] 

#calculate mean confidence for best and worst 3 CV folds
confidence_best_3 = np.mean(confidence_best_3,axis=1)
confidence_worst_3 = np.mean(confidence_worst_3,axis=1) 


# Calculate AUROC, AUPRC, F1, Precision, Recall, Sensitivity, Specificity, Accuracy 

confidence = np.array(confidence)
# AUROC
auroc = compute_roc(confidence,labels)[0] 
aucpr, precision,recall,_ = compute_prc(confidence,labels)
f1 = f1_score(labels,(confidence > 0.5).astype(int))
accuracy = accuracy_score(labels,(confidence > 0.5).astype(int))
#confusion_matrix = confusion_matrix(labels,(confidence > 0.5).astype(int))
#specificity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[0,1])
#sensitivity = confusion_matrix[1,1]/(confusion_matrix[1,0]+confusion_matrix[1,1])

precision,recall,f1,_  = precision_recall_fscore_support(labels, (confidence > 0.5).astype(int) ,pos_label=1, average='binary')
tn, fp, fn, tp = confusion_matrix(labels, (confidence > 0.5).astype(int),labels=[0,1]).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)


print("AUROC: ",auroc)
print("AUCPR: ",aucpr)
print("F1: ",f1)
print("Precision: ",precision)
print("Recall: ",recall)
print("Accuracy: ",accuracy)
print("Sensitivity: ",sensitivity)
print("Specificity: ",specificity)
print("\n")


# Calculate AUROC, AUPRC, F1, Precision, Recall, Sensitivity, Specificity, Accuracy for best 3
confidence_best_3 = np.array(confidence_best_3)
# AUROC
auroc = compute_roc(confidence_best_3,labels)[0]
aucpr, precision,recall,_ = compute_prc(confidence_best_3,labels)
f1 = f1_score(labels,(confidence_best_3 > 0.5).astype(int))
accuracy = accuracy_score(labels,(confidence_best_3 > 0.5).astype(int))
precision,recall,f1,_  = precision_recall_fscore_support(labels, (confidence_best_3 > 0.5).astype(int) ,pos_label=1, average='binary')
tn, fp, fn, tp = confusion_matrix(labels, (confidence_best_3 > 0.5).astype(int),labels=[0,1]).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)

print("AUROC best 3: ",auroc)
print("AUCPR best 3: ",aucpr)
print("F1 best 3: ",f1)
print("Precision best 3: ",precision)
print("Recall best 3: ",recall)
print("Accuracy best 3: ",accuracy)
print("Sensitivity best 3: ",sensitivity)
print("Specificity best 3: ",specificity)
print("\n")

# disease label 1,2,3,4 are label 1 and disease label 0 is label 0 
# calculate the accuracy for each disease label

preds = (confidence_best_3 > 0.5).astype(int) 
disease_labels = np.array(per_disease_labels)
labels = np.array(labels)
#calculate accuracy for each disease label
for i in range(5):
    
        idx = np.where(disease_labels == i)[0]
        print("Disease label: ",i)
        print("Accuracy: ",accuracy_score(labels[idx],preds[idx]))
        print("Count: ",len(idx))
        print("Count of correct classification:",np.sum(labels[idx] == preds[idx]))
        print("\n")

# Perform McNemar's test for prediction labels of confidence_best_3 against confidence_all 

from mlxtend.evaluate import mcnemar_table,mcnemar

# contingency table
table = [[0, 0], [0, 0]]
for i in range(len(csvs)):

    model_1 = (confidence_best_3 > 0.5).astype(int) 
    model_2 = (confidence_all[:,i] > 0.5).astype(int) 

    tb = mcnemar_table(y_target=labels, y_model1=model_1, y_model2=model_2)
    chi2, p = mcnemar(ary=tb, corrected=True) 
    print('chi-squared:', chi2)
    print('p-value:', p)
    print("Ensemble VS CV fold: ",i+1)
  
    print("\n")


assert False

print(table)
# calculate mcnemar test
result = mcnemar(table, exact=True)
# summarize the finding

# Calculate AUROC, AUPRC, F1, Precision, Recall, Sensitivity, Specificity, Accuracy for worst 3
confidence_worst_3 = np.array(confidence_worst_3)
# AUROC
auroc = compute_roc(confidence_worst_3,labels)[0]
aucpr, precision,recall,_ = compute_prc(confidence_worst_3,labels)
f1 = f1_score(labels,(confidence_worst_3 > 0.5).astype(int))
accuracy = accuracy_score(labels,(confidence_worst_3 > 0.5).astype(int))
precision,recall,f1,_  = precision_recall_fscore_support(labels, (confidence_worst_3 > 0.5).astype(int) ,pos_label=1, average='binary')
tn, fp, fn, tp = confusion_matrix(labels, (confidence_worst_3 > 0.5).astype(int),labels=[0,1]).ravel()
specificity = tn / (tn+fp)
sensitivity = tp / (tp+fn)

print("AUROC worst 3: ",auroc)
print("AUCPR worst 3: ",aucpr)
print("F1 worst 3: ",f1)
print("Precision worst 3: ",precision)
print("Recall worst 3: ",recall)
print("Accuracy worst 3: ",accuracy)
print("Sensitivity worst 3: ",sensitivity)
print("Specificity worst 3: ",specificity)






# ROC curve 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve



fpr, tpr, thresholds = roc_curve(labels, confidence) 
#create new dataframe with fpr, tpr, CV_fold
df_roc = pd.DataFrame({"FPR":fpr,"TPR":tpr,"CV_fold":["Ensemble"]*len(fpr)})

#concatenate with all roc curves
df_roc = pd.concat([df_roc,roc_curves])
# calculate PRC curve 

from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(labels, confidence) 
#create new dataframe with precision, recall, CV_fold
df_prc = pd.DataFrame({"Prec_PRCurve":precision,"Rec_PRCurve":recall,"CV_fold":["Ensemble"]*len(precision)})


#concatenate with all prc curves
df_prc = pd.concat([df_prc,prc_curves])






#Plot roc curve using seaborn, hue = CV_fold
import seaborn as sns
sns.set_theme(style="darkgrid")
sns.lineplot(x="FPR", y="TPR", hue="CV_fold", data=df_roc) 
plt.savefig(folder+"/roc_curve.png") 

plt.close()
plt.clf()
#Plot prc curve using seaborn, hue = CV_fold
sns.lineplot(x="Prec_PRCurve", y="Rec_PRCurve", hue="CV_fold", data=df_prc)
plt.savefig(folder+"/prc_curve.png")



