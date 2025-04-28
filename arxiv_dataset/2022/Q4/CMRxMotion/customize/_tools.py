import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import itertools

class WBVote:
    def __init__(self, weight=[1,2,10]):
        self.weight = weight

    def __call__(self, x):
        x = np.array(x)
        counts = [np.sum(x==0)*self.weight[0], np.sum(x==1)*self.weight[1], np.sum(x==2)*self.weight[2]]
        result = np.argmax(counts)
        return result



class BVote:
    def __init__(self, thresholds):
        self.thresholds = thresholds

    def __call__(self, x):
        try:
            x = np.array(x)
            counts = [np.sum(x==0), np.sum(x==1), np.sum(x==2)]
            total = np.sum(counts)
            if (counts[2] + counts[1]) / total >= self.thresholds[0]:
                if counts[2] / (counts[2] + counts[1]) >= self.thresholds[1]:
                    return 2
                return 1
            return 0
        except RuntimeWarning:
            pass

class Vote:
    def __init__(self, thresholds):
        self.thresholds = thresholds

    def __call__(self, x):
        try:
            x = np.array(x)
            counts = [np.sum(x==0), np.sum(x==1), np.sum(x==2)]
            total = np.sum(counts)
            if counts[2] >= self.thresholds[0]:
                return 2
            if counts[1] >= self.thresholds[1]:
                return 1
            return 0
        except RuntimeWarning:
            pass


class MVote:
    def __init__(self, order='last'):
        self.order = order

    def __call__(self, x):
        x = np.array(x)
        counts = [np.sum(x==0), np.sum(x==1), np.sum(x==2)]
        if self.order == 'last':
            maxi = 2
            if counts[1] > counts[maxi]:
                maxi = 1
            if counts[0] > counts[maxi]:
                maxi = 0
        elif self.order == 'first':
            maxi = 0
            if counts[1] > counts[maxi]:
                maxi = 1
            if counts[2] > counts[maxi]:
                maxi = 2
        return maxi


def fill_imgs(data_list, ratio=None, suffix='*.tif'):
    new_list = []
    for d in data_list:
        sub_list = glob.glob(d + '/' + suffix)
        sub_list = sorted(sub_list)
        if ratio is not None:
            if type(ratio) is float:
                rs = int(len(sub_list) * ratio * 0.5)
                mid = len(sub_list) // 2
                sub_list = sub_list[mid-rs : mid+rs]
            else:
                tmp_list = []
                for sub_file in sub_list:
                    if int(os.path.basename(sub_file).split('_')[0]) in ratio:
                        tmp_list.append(sub_file)
                sub_list = tmp_list

        new_list += sub_list
    return new_list




def plot_confusion_matrix(gt, pred, 
                          classes=['0', '1', '2'],
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    cm = metrics.confusion_matrix(gt, pred)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Ground Truth')
    plt.xlabel('Prediction')

