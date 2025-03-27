import pandas as pd
from sklearn.metrics import f1_score
from sklearn import metrics

label = pd.read_csv("E:\sentimix\label/test_labels_hinglish.txt", header=0, delimiter=",", quoting=3)
predict = pd.read_csv("E:\sentimix\Code\gating_2lang_hindi/0.690/answer.txt", header=0, delimiter=",", quoting=3)
label = label["Sentiment"]
predict = predict["Sentiment"]
print("F1 score is: " + str(f1_score(label, predict, average='macro')))

target_names = ['class 0', 'class 1','class 2']
print(metrics.classification_report(label, predict, target_names=target_names))