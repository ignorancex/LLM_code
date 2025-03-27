import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()



nll_cifar = np.load('./array/ER/testindist.npy')

nll_svhn = np.load('./array/ER/testood.npy')

combined = np.concatenate((nll_cifar, nll_svhn))
label_1 = np.ones(len(nll_cifar))
label_2 = np.zeros(len(nll_svhn))
label = np.concatenate((label_1, label_2))

fpr, tpr, thresholds = metrics.roc_curve(label, combined, pos_label=1)

#plot_roc_curve(fpr, tpr)

rocauc = metrics.auc(fpr, tpr)
print('AUC for ER is: ', rocauc)