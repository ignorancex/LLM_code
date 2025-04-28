## Performance metrics
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, balanced_accuracy_score
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

## Define performance measures
def pmeasure(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1score = 2 * tp / (2 * tp + fp + fn)
    yi = (tp/(tp+fn)) + (tn/(tn+fp)) - 1
    return ({'Sensitivity': sensitivity, 'Specificity': specificity, 'F1-Score': f1score, 'Yoden-Index': yi})

def Calculate_Stats(y_actual,y_pred):
    # from scikit-learn
    acc = accuracy_score(y_actual.argmax(axis=1), y_pred.argmax(axis=1))
    mcc = matthews_corrcoef(y_actual.argmax(axis=1), y_pred.argmax(axis=1))
    bacc = balanced_accuracy_score(y_actual.argmax(axis=1), y_pred.argmax(axis=1))    
    pre, rec, _ = precision_recall_curve(y_actual.argmax(axis=1), y_score, pos_label=1)
    fpr, tpr, _ = roc_curve(y_actual.argmax(axis=1), y_score, pos_label=1)
    auroc = auc(fpr, tpr)
    aupr = auc(rec, pre)

    # from custom function - pmeasure
    sen = pmeasure(y_actual.argmax(axis=1), y_pred.argmax(axis=1))['Sensitivity']
    spe = pmeasure(y_actual.argmax(axis=1), y_pred.argmax(axis=1))['Specificity']
    f1 = pmeasure(y_actual.argmax(axis=1), y_pred.argmax(axis=1))['F1-Score']
    yi = pmeasure(y_actual.argmax(axis=1), y_pred.argmax(axis=1))['Yoden-Index']    

    return acc, sen, spe, f1, mcc, bacc, yi, auroc, aupr

def Show_Statistics(msg,Stats):
    print(msg.upper())
    print(70*'-')
    print('Accuracy:',Stats[0])
    print('Sensitivity:',Stats[1])
    print('Specificity:',Stats[2])
    print('F1-Score:',Stats[3]) 
    print('MCC:',Stats[4])
    print('Balance Accuracy:',Stats[5])
    print('Youden-Index:',Stats[6])
    print('AUC:',Stats[7])  
    print('AUPR:',Stats[8])  
    print('Reconstruction MSE:',Stats[9])
    print(70*'-')
