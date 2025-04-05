import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt


class MCROC(object):
    def __init__(self, ys, yhats, n_classes, labels=None):
        """
        Parameters
        ----------
        ys : list of numpy arrays
            each numpy array is either 1d (class id) or 2d (class prob)
        yhats : list of numpy arrays
            each numpy array is either 1d (class id) or 2d (class prob)
        labels : list of strings
        """
        self.labels = labels
        self.n_classes = n_classes
        self.ys = list()
        self.y_class = list()
        for y in ys:
            if len(y.shape) == 1:
                y_class = y
                y = self.one_hot(y)
            elif len(y.shape) == 2:
                y = y
                y_class = np.argmax(y, axis=1)

            self.ys.append(y)
            self.y_class.append(y_class)

        self.yhats = list()
        for yhat in yhats:
            if len(yhat.shape) == 1:
                yhat = self.one_hot(yhat)
            self.yhats.append(yhat)

    def one_hot(self, y):
        n = len(y)
        yoh = np.zeros((n, self.n_classes))
        for i, index in enumerate(y):
            if i is None: continue
            yoh[i,index] = 1
        return yoh

    def get_fpr_tpr_auc_single(self, y, yhat, average='macro'):
        assert average in ['micro', 'macro', None]
        if average == None:
            fprs, tprs, aucs = list(), list(), list()
            for i in range(self.n_classes):
                fpr, tpr, t = sklearn.metrics.roc_curve(
                    y_true=y[:,i],
                    y_score=yhat[:,i],
                )
                auc = sklearn.metrics.auc(fpr, tpr)
                auc = np.round(auc, 3)
                fprs.append(fpr)
                tprs.append(tpr)
                aucs.append(auc)
            return fprs, tprs, aucs

        elif average == 'macro':
            fprs, tprs, _ = self.get_fpr_tpr_auc_single(y, yhat, average=None)
            all_fpr = np.unique(np.concatenate(fprs))
            mean_tpr = np.zeros(all_fpr.shape)
            for i in range(self.n_classes):
                mean_tpr += np.interp(all_fpr, fprs[i], tprs[i])
            mean_tpr /= float(self.n_classes)
            auc = sklearn.metrics.auc(all_fpr, mean_tpr)
            auc = np.round(auc, 3)
            return all_fpr, mean_tpr, auc

        elif average == 'micro':
            fpr, tpr, t = sklearn.metrics.roc_curve(
                y_true=np.ravel(y),
                y_score=np.ravel(yhat)
            )
            auc = sklearn.metrics.auc(fpr, tpr)
            return fpr, tpr, auc

    def get_fpr_tpr_auc(self, average='macro'):
        self.fprs, self.tprs, self.aucs = list(), list(), list()
        for y, yhat in zip(self.ys, self.yhats):
            fpr, tpr, auc = self.get_fpr_tpr_auc_single(y, yhat, average=average)
            self.fprs.append(fpr)
            self.tprs.append(tpr)
            self.aucs.append(auc)

        return self.fprs, self.tprs, self.aucs

    def get_auc(self, ys, yhats, _round=True):
        self.aucs = list()
        for y, yhat in zip(ys, yhats):
            auc = sklearn.metrics.roc_auc_score(
                y_true=y,
                y_score=yhat,
            )
            if _round: auc = np.round(auc, 3)
            self.aucs.append(auc)

        return self.aucs

    def get_overall_accuracies(self, threshold=0.5, _round=True):
        accs = list()
        for y_class, yhat in zip(self.y_class, self.yhats):
            yhat = np.argmax(yhat, axis=1)
            acc = np.mean(yhat == y_class)
            if _round: acc = np.round(acc, 3)
            accs.append(acc)
        return accs

    def get_perclass_accuracies(self, threshold=0.5, _round=True):
        accs = list()
        for y_class, yhat in zip(self.y_class, self.yhats):
            yhat = np.argmax(yhat, axis=1)
            accs_i = list()
            for i in range(self.n_classes):
                acc = np.mean(yhat[y_class == i] == i)
                if _round: acc = np.round(acc, 3)
                accs_i.append(acc)
            accs.append(accs_i)
        return accs

    def plot_roc(self, fprs, tprs, aucs, w=10, h=10, title='', ax=None):
        sns.set_style('darkgrid')
        sns.set_palette('husl')
        if ax is None: fig, ax = plt.subplots(1,1, figsize=(w,h))
        for fpr, tpr, auc, label in zip(fprs, tprs, aucs, self.labels):
            label = '{} AUC: {}'.format(label, auc)
            ax.plot(fpr, tpr, linewidth=2, alpha=0.5, label=label)
            # ax.scatter(fpr, tpr)

        ax.legend(loc='lower right')
        ax.set_xlim([-0.01,1])
        ax.set_ylim([0,1.01])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves' + title)

    def plot_roc_one_vs_all(self, h=10, title=''):
        sns.set_style('darkgrid')
        sns.set_palette('husl')
        fig, axarr = plt.subplots(1, self.n_classes, figsize=(self.n_classes*h, h))
        for i, ax in zip(range(self.n_classes), axarr):
            fprs = [fpr[i] for fpr in self.fprs]
            tprs = [tpr[i] for tpr in self.tprs]
            aucs = [auc[i] for auc in self.aucs]
            yhats = [yhat[:,i] for yhat in self.yhats]
            ys = [y[:,i] for y in self.ys]
            self.plot_roc(fprs, tprs, aucs, title='\n{}'.format(str(i)), ax=ax)
        plt.show()
