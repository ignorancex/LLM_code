import copy
import math

from sklearn import metrics
from sklearn.metrics import roc_auc_score, brier_score_loss
import numpy as np
import pandas as pd
from sklearn.metrics import  auc

class measures(object):

    def __init__(self,actual,predicted,loc,labels = [0,1]):
        self.actual = actual
        self.predicted = predicted
        self.loc = loc
        #self.dframe = pd.concat(
        #    [pd.Series(self.actual,name='Actual'), pd.Series(self.predicted,name='Predicted'), self.loc], axis=1)
        self.dframe = pd.DataFrame(list(zip(self.actual,self.predicted,self.loc)),columns = ['Actual','Predicted','LOC'])
        self.dframe = self.dframe.dropna()
        self.dframe = self.dframe.astype({'Actual': int, 'Predicted': int})
        self.dframe_unchanged = copy.deepcopy(self.dframe)
        self.dframe.sort_values(by = ['Predicted','LOC'],inplace=True,ascending=[False,True])
        #print(self.dframe)
        self.dframe['InspectedLOC'] = self.dframe.LOC.cumsum()
        self.dframe_unchanged['InspectedLOC'] = self.dframe_unchanged.LOC.cumsum()
        self.tn, self.fp, self.fn, self.tp = metrics.confusion_matrix(
            actual, predicted, labels=labels).ravel()
        self.pre, self.rec, self.spec, self.fpr, self.npv, self.acc, self.f1,self.pd,self.pf = self.get_performance()
        #print(metrics.classification_report(self.actual,self.predicted))
        self._set_aux_vars()


    def _set_aux_vars(self):
        """
        Set all the auxillary variables used for defect prediction
        """

        self.M = len(self.dframe[self.dframe['Predicted'] == 1])
        self.N = self.dframe.Actual.sum() # have to check the implementation
        #inspected_max = self.dframe.InspectedLOC.max()
        inspected_max = self.dframe.InspectedLOC.max() * 0.2

        inspectedSoFar = 0
        for i in range(self.M):
            inspectedSoFar += self.dframe.InspectedLOC.iloc[i]

            if inspectedSoFar >= 1 * inspected_max:
                # If we have inspected more than 20% of the total LOC
                # break
                break


            # print("inspector = ",inspectedSoFar, inspected_max, self.dframe.InspectedLOC.max())

        if self.M == 0:
            i = 0
            self.M = 1
        self.inspected_50 = self.dframe.iloc[:i]
        # Number of changes when we inspect 20% of LOC
        self.m = len(self.inspected_50)
        self.n = self.inspected_50.Predicted.sum()

    def subtotal(self, x):
        xx = [0]
        for i, t in enumerate(x):
            xx += [xx[-1] + t]
        return xx[1:]

    def get_arecall(self,true):
        total_true = float(len([i for i in true if i == 1]))
        hit = 0.0
        recall = []

        for i in range(len(true)):
            if true[i] == 1:
                hit += 1
            recall += [hit / total_true if total_true else 0.0]
        return recall

    def get_aauc(self, data):
        """The smaller the better"""
        if len(data) == 1:
            return 0
        x_sum = float(sum(data['loc']))
        x = data['loc'].apply(lambda t: t / x_sum)
        xx = self.subtotal(x)
        yy = self.get_arecall(data['bug'].values)
        try:
            ret = round(auc(xx, yy), 3)
        except:
            # print"?"
            ret = 0
        return ret

    def get_popt_20(self):

        tempDF = pd.DataFrame()

        tempDF['bug'] = self.dframe['Actual']
        tempDF['loc'] = self.dframe['LOC']
        tempDF['prediction'] = self.dframe['Predicted']
        data = tempDF.copy(deep=True)
        # data.to_csv('data.csv',index=False)

        data.sort_values(by=["bug", "loc"], ascending=[0, 1], inplace=True)
        x_sum = float(sum(data['loc']))
        x = data['loc'].apply(lambda t: t / x_sum)
        xx = self.subtotal(x)

        # get  AUC_optimal
        yy = self.get_arecall(data['bug'].values)
        xxx = [i for i in xx if i <= 0.2]
        yyy = yy[:len(xxx)]
        s_opt = round(auc(xxx, yyy), 3)

        # get AUC_worst
        xx = self.subtotal(x[::-1])
        yy = self.get_arecall(data['bug'][::-1].values)
        xxx = [i for i in xx if i <= 0.2]
        yyy = yy[:len(xxx)]
        try:
            s_wst = round(auc(xxx, yyy), 3)
        except:
            # print "s_wst forced = 0"
            s_wst = 0

        # get AUC_prediction
        data.sort_values(by=["prediction", "loc"], ascending=[0, 1], inplace=True)
        x = data['loc'].apply(lambda t: t / x_sum)
        xx = self.subtotal(x)
        yy = self.get_arecall(data['bug'].values)
        xxx = [k for k in xx if k <= 0.2]
        yyy = yy[:len(xxx)]

        try:
            s_m = round(auc(xxx, yyy), 3)
        except:
            return 0

        # Popt = (s_m - s_wst) / (s_opt - s_wst)

        Popt = (s_opt - s_m)/(s_opt - s_wst)
        Popt = 1 - Popt
        Popt = round(Popt, 3)

        # if Popt > 1:
        #     data.to_csv('data1.csv',index=False)

        return Popt

    def get_pci_20(self):
        pci_20 = self.m / self.M
        return round(pci_20,2)

    def get_ifa(self):

        # print('Actual')
        # print(self.dframe['Actual'])
        # print('Predicted')
        # print(self.dframe['Predicted'] )

        for i in range(len(self.dframe)):
            if self.dframe['Actual'].iloc[i] == self.dframe['Predicted'].iloc[i] == 1:
                break

        # print('$$$', i)
        return i

    def get_ifa_roc(self):
        ifa_x = []
        ifa_y = []
        for perc in range(1,101,1):
            count = 0
            inspected_max = self.dframe_unchanged.InspectedLOC.max() * (perc/100)
            for i in range(len(self.dframe_unchanged)):
                if self.dframe_unchanged.InspectedLOC.iloc[i] >= 1 * inspected_max:
                    break
                if self.dframe_unchanged['Predicted'].iloc[i] == True:
                    continue
                count += 1 
                if self.dframe_unchanged['Actual'].iloc[i] == self.dframe_unchanged['Predicted'].iloc[i] == 1:
                    break
            ifa_x.append(perc)
            ifa_y.append(count/self.dframe_unchanged[self.dframe_unchanged['Predicted'] == 1].shape[0])
        return np.trapz(ifa_y,x=ifa_x)
    
    def calculate_accuracy(self):

        return  metrics.accuracy_score(self.actual, self.predicted)

    def calculate_recall(self):

        if len(metrics.recall_score(self.actual, self.predicted, average=None)) == 1:
            if self.actual.unique()[0] == True:
                result = round(metrics.recall_score(self.actual, self.predicted, average=None)[0],2)
            else:
                result = 0
        else:
            result = round(metrics.recall_score(self.actual, self.predicted, average=None)[1],2)
        return result


    def mcc(self):

        return metrics.matthews_corrcoef(self.actual, self.predicted)


    def calculate_precision(self):
        if len(metrics.precision_score(self.actual, self.predicted, average=None)) == 1:
            if self.actual.unique()[0] == True:
                result = round(metrics.precision_score(self.actual, self.predicted, average=None)[0],2)
            else:
                result = 0
        else:
            result = round(metrics.precision_score(self.actual, self.predicted, average=None)[1],2)
        return result

    def calculate_f1_score(self):

        # print("all weighted ", metrics.f1_score(self.actual, self.predicted, average='weighted'))
        # print("all ", metrics.f1_score(self.actual, self.predicted, average=None))

        if len(metrics.f1_score(self.actual, self.predicted, average=None)) == 1:
            if self.actual.unique()[0] == True:
                result = round(metrics.f1_score(self.actual, self.predicted, average=None)[0],2)
                # print("normal ",result)

            else:
                result = 0
        else:
            result = round(metrics.f1_score(self.actual, self.predicted, average=None)[1],2)
        return result

    def get_performance(self):
        pre = round(1.0 * self.tp / (self.tp + self.fp),2) if (self.tp + self.fp) != 0 else 0
        rec = round(1.0 * self.tp / (self.tp + self.fn),2) if (self.tp + self.fn) != 0 else 0
        spec = round(1.0 * self.tn / (self.tn + self.fp),2) if (self.tn + self.fp) != 0 else 0
        fpr = round(1 - spec,2)
        npv = round(1.0 * self.tn / (self.tn + self.fn),2) if (self.tn + self.fn) != 0 else 0
        acc = round(1.0 * (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn),2) if (self.tp + self.tn + self.fp + self.fn) != 0 else 0
        f1 = round(2.0 * self.tp / (2.0 * self.tp + self.fp + self.fn),2) if (2.0 * self.tp + self.fp + self.fn) != 0 else 0
        pd = round(1.0 * self.tp / (self.tp + self.fn),2)
        pf =  round(1.0 * self.fp / (self.fp + self.tn),2)
        return pre, rec, spec, fpr, npv, acc, f1,pd,pf

    def get_pd(self):
        return self.pd

    def get_pf(self):
        return self.pf

    def get_tp(self):
        return self.tp

    def get_fp(self):
        return self.fp

    def get_tn(self):
        return self.tn

    def get_fn(self):
        return self.fn

    def calculate_d2h(self):

        far = 0
        recall = 0

        if (self.fp + self.tn) != 0:
            far = self.fp/(self.fp+self.tn)
        if (self.tp + self.fn) != 0:
            recall = self.tp/(self.tp + self.fn)

        dist2heaven = math.sqrt((1 - recall) ** 2 + far ** 2)

        dist2heaven = dist2heaven/math.sqrt(2)

        return dist2heaven

    def get_g_score(self, beta = 0.5):

        if self.pd == 0 and self.pf == 1:
            self.pf = 1.000001 # to avoid divide by 0 error

        g = (1 + beta**2) * (self.pd * (1.0 - self.pf))/ (beta ** 2 * self.pd + (1.0 - self.pf))
        return round(g,2)

    def get_roc_auc_score(self):
        return roc_auc_score(self.actual, self.predicted)

    def negOverPos(self):

        return (self.tn + self.fp)/(self.fn + self.tp)


    def balance(self):
        return 1 - math.sqrt(math.pow(0 - self.get_pf(), 2) + math.pow(1 - self.get_pd(), 2))/math.sqrt(2)

    def brier(self):
        """
        Brier score measures the mean squared difference between (1) the predicted probability assigned to the possible outcomes for item i, and (2) the actual outcome.
        Thus smaller the better.
        :return:
        """

        bl = brier_score_loss(self.actual, self.predicted)

        return bl


def test_d2h(tp, tn, fp, fn):

    far = 0
    recall = 0

    if (fp + tn) != 0:
        far = fp/(fp + tn)
    if (tp + fn) != 0:
        recall = tp/(tp + fn)

    dist2heaven = math.sqrt((1 - recall) ** 2 + far ** 2)
    dist2heaven = dist2heaven / math.sqrt(2)

    return dist2heaven

if __name__ == '__main__':
    tp,     tn,     fp,     fn, =    5,    5,    0,    0
    print(test_d2h(tp, tn, fp, fn))






    