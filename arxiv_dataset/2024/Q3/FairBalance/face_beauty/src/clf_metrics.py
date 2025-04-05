import numpy as np
from collections import Counter

class Clf_Metrics:
    def __init__(self, X, y, y_pred, y_pred_prob, A):
        # y and y_pred are 1-d arrays of true values and predicted values
        self.X = X
        self.y = y
        self.y_pred = y_pred
        self.y_pred_prob = y_pred_prob
        self.groups = {}
        for i in range(len(self.y)):
            group = tuple([self.X[a][i] for a in A])
            if group not in self.groups:
                self.groups[group] = []
            self.groups[group].append(i)

    def accuracy(self):
        return Counter(self.y==self.y_pred)[True] / len(self.y)

    def auc(self):
        order = np.argsort(self.y_pred_prob)[::-1]
        tp = 0
        fp = 0
        fn = Counter(self.y)[1]
        tn = len(self.y) - fn
        tpr = 0
        fpr = 0
        auc = 0
        for i in order:
            if self.y[i] == 1:
                tp += 1
                fn -= 1
                tpr = float(tp) / (tp + fn)
            else:
                fp += 1
                tn -= 1
                pre_fpr = fpr
                fpr = float(fp) / (tn + fp)
                auc += tpr * (fpr - pre_fpr)
        return auc

    def prs(self):
        prs = {}
        for group in self.groups:
            sub = self.groups[group]
            conf = self.confusion(self.y[sub], self.y_pred[sub])
            pr = (conf['tp']+conf['fp']) / len(sub)
            prs[group] = pr
        return prs

    def tprs(self):
        tprs = {}
        for group in self.groups:
            sub = self.groups[group]
            conf = self.confusion(self.y[sub], self.y_pred[sub])
            tpr = conf['tp'] / (conf['tp'] + conf['fn'])
            tprs[group] = tpr
        return tprs

    def tprcs(self):
        tprs = {}
        for group in self.groups:
            sub = self.groups[group]
            conf = self.confusion(self.y[sub], self.y_pred_prob[sub])
            tpr = conf['tp'] / (conf['tp'] + conf['fn'])
            tprs[group] = tpr
        return tprs

    def fprs(self):
        fprs = {}
        for group in self.groups:
            sub = self.groups[group]
            conf = self.confusion(self.y[sub], self.y_pred[sub])
            fpr = conf['fp'] / (conf['fp'] + conf['tn'])
            fprs[group] = fpr
        return fprs

    def fprcs(self):
        fprs = {}
        for group in self.groups:
            sub = self.groups[group]
            conf = self.confusion(self.y[sub], self.y_pred_prob[sub])
            fpr = conf['fp'] / (conf['fp'] + conf['tn'])
            fprs[group] = fpr
        return fprs

    def aos(self):
        tprs = self.tprs()
        fprs = self.fprs()
        aos = {key: (tprs[key]+fprs[key])/2 for key in tprs}
        return aos

    def aocs(self):
        tprcs = self.tprcs()
        fprcs = self.fprcs()
        aocs = {key: (tprcs[key]+fprcs[key])/2 for key in tprcs}
        return aocs


    def eod(self, a=None):
        # return: EOD = max{TPR(A=a)} - min{TPR(A=a)} if a == None
        # return: EOD = TPR(a=1) - TPR(a=0) if a != None
        # TPR = #(y=1, C=1) / #(y=1)
        if a is not None:
            ind0 = np.where(self.X[a] == 0)[0]
            ind1 = np.where(self.X[a] == 1)[0]
            conf0 = self.confusion(self.y[ind0], self.y_pred[ind0])
            conf1 = self.confusion(self.y[ind1], self.y_pred[ind1])
            tpr0 = conf0['tp'] / (conf0['tp'] + conf0['fn'])
            tpr1 = conf1['tp'] / (conf1['tp'] + conf1['fn'])
            return tpr1 - tpr0
        else:
            tprs = self.tprs()
            return max(tprs.values())-min(tprs.values())

    def seod(self, a=None):
        # return: EOD = max{TPR(A=a)} - min{TPR(A=a)} if a == None
        # return: EOD = TPR(a=1) - TPR(a=0) if a != None
        # TPR = #(y=1, C=1) / #(y=1)
        if a is not None:
            ind0 = np.where(self.X[a] == 0)[0]
            ind1 = np.where(self.X[a] == 1)[0]
            conf0 = self.confusion(self.y[ind0], self.y_pred_prob[ind0])
            conf1 = self.confusion(self.y[ind1], self.y_pred_prob[ind1])
            tpr0 = conf0['tp'] / (conf0['tp'] + conf0['fn'])
            tpr1 = conf1['tp'] / (conf1['tp'] + conf1['fn'])
            return tpr1 - tpr0
        else:
            tprcs = self.tprcs()
            return max(tprcs.values())-min(tprcs.values())

    def aod(self, a=None):
        # return: AOD = 0.5 * (max{TPR(A=a)+FPR(A=a)} - min{TPR(A=a)+FPR(A=a)}) if a == None
        # return: AOD = 0.5 * (TPR(a=1) + FPR(a=1) - TPR(a=0) - FPR(a=0)) if a != None
        # TPR = #(y=1, C=1) / #(y=1), FPR = #(y=0, C=1) / #(y=0)
        if a is not None:
            ind0 = np.where(self.X[a] == 0)[0]
            ind1 = np.where(self.X[a] == 1)[0]
            conf0 = self.confusion(self.y[ind0], self.y_pred[ind0])
            conf1 = self.confusion(self.y[ind1], self.y_pred[ind1])
            tpr0 = conf0['tp'] / (conf0['tp'] + conf0['fn'])
            tpr1 = conf1['tp'] / (conf1['tp'] + conf1['fn'])
            fpr0 = conf0['fp'] / (conf0['fp'] + conf0['tn'])
            fpr1 = conf1['fp'] / (conf1['fp'] + conf1['tn'])
            return 0.5 * (tpr1 + fpr1 - tpr0 - fpr0)
        else:
            aos = self.aos()
            return max(aos.values()) - min(aos.values())

    def saod(self, a=None):
        # return: AOD = 0.5 * (max{TPR(A=a)+FPR(A=a)} - min{TPR(A=a)+FPR(A=a)}) if a == None
        # return: AOD = 0.5 * (TPR(a=1) + FPR(a=1) - TPR(a=0) - FPR(a=0)) if a != None
        # TPR = #(y=1, C=1) / #(y=1), FPR = #(y=0, C=1) / #(y=0)
        if a is not None:
            ind0 = np.where(self.X[a] == 0)[0]
            ind1 = np.where(self.X[a] == 1)[0]
            conf0 = self.confusion(self.y[ind0], self.y_pred_prob[ind0])
            conf1 = self.confusion(self.y[ind1], self.y_pred_prob[ind1])
            tpr0 = conf0['tp'] / (conf0['tp'] + conf0['fn'])
            tpr1 = conf1['tp'] / (conf1['tp'] + conf1['fn'])
            fpr0 = conf0['fp'] / (conf0['fp'] + conf0['tn'])
            fpr1 = conf1['fp'] / (conf1['fp'] + conf1['tn'])
            return 0.5 * (tpr1 + fpr1 - tpr0 - fpr0)
        else:
            aocs = self.aocs()
            return max(aocs.values()) - min(aocs.values())

    def spd(self, a=None):
        # return: SPD = max{TPR(A=a)} - min{TPR(A=a)} if a == None
        # return: SPD = PR(a=1) - PR(a=0) if a != None
        # PR = #(C=1) / N
        if a is not None:
            ind0 = np.where(self.X[a] == 0)[0]
            ind1 = np.where(self.X[a] == 1)[0]
            conf0 = self.confusion(self.y[ind0], self.y_pred[ind0])
            conf1 = self.confusion(self.y[ind1], self.y_pred[ind1])
            pr0 = (conf0['tp']+conf0['fp']) / len(ind0)
            pr1 = (conf1['tp']+conf1['fp']) / len(ind1)
            return pr1 - pr0
        else:
            prs = self.prs()
            return max(prs.values())-min(prs.values())

    def confusion(self, y, y_pred):
        conf = {'tp':0, 'tn':0, 'fp':0, 'fn':0}
        for i in range(len(y)):
            if y[i]==0:
                conf['tn'] += 1 - y_pred[i]
                conf['fp'] += y_pred[i]
            elif y[i]==1:
                conf['tp'] += y_pred[i]
                conf['fn'] += 1 - y_pred[i]
        return conf