import os
import traceback

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import  DecisionTreeClassifier
# from fasttrees.fasttrees import FastFrugalTreeClassifier
# from hpsklearn import HyperoptEstimator, any_classifier, any_preprocessing
import os

import hyperopt_wrapper

os.environ['OMP_NUM_THREADS'] = "1"
import sys

import time as goodtime

from sklearn.linear_model import *

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC,NuSVC
from sklearn.neural_network import MLPClassifier

from sklearn.base import clone

from identify_bell_project import *
import tca_plus
from sklearn.neural_network import *

from hyperopt_wrapper import *

from dodge import *


# from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval, rand, atpe



# from Inference import *
from Util import *
from Constants import *
import platform
from multiprocessing import Process
from random import *
from project_samples import *


import SMOTE
import feature_selector
import CFS



import release_manager
import numpy as np


from multiprocessing import Pool, cpu_count


import metrices


from  sklearn.calibration import *
from  sklearn.discriminant_analysis import *

from  sklearn.dummy import *
from  sklearn.ensemble import *
from  sklearn.ensemble._hist_gradient_boosting.gradient_boosting import *
from  sklearn.gaussian_process._gpc import *
from  sklearn.neighbors._nearest_centroid import *

from  sklearn.gaussian_process._gpc import *
from  sklearn.linear_model._logistic import *

from  sklearn.linear_model._passive_aggressive import *
from  sklearn.linear_model._perceptron import *
from  sklearn.linear_model._ridge import *

from  sklearn.linear_model._stochastic_gradient import *
from  sklearn.multiclass import *
from  sklearn.multioutput import *

from  sklearn.naive_bayes import *
from  sklearn.neighbors._classification import *
from  sklearn.neural_network._multilayer_perceptron import *
from  sklearn.semi_supervised._label_propagation import *
from  sklearn.svm._classes import *
from  sklearn.tree._classes import *


import warnings
warnings.filterwarnings("ignore")



"""
@author : Anonymous  
Evaluate various train approaches across various learners and records all measures
"""

RESULTS_FOLDER = 'RQ_'+str(RQ)+'_RESULTS'


# TCA_DATA_FOLDER = 'TTD_TCA_DATA_FOLDER'

if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

"""
@author : Anonymous
"""
class MLUtil(object):

    def __init__(self):
        self.cores = 1

    def get_features(self,df):
        fs = feature_selector.featureSelector()
        df,_feature_nums,features = fs.cfs_bfs(df)
        return df,features

    def apply_pca(self, df):
        return df

    def apply_cfs(self,df):

        copyDF = df.copy(deep=True)

        y = copyDF.Buggy.values
        X = copyDF.drop(labels=['Buggy'], axis=1)

        X = X.values

        selected_cols = CFS.cfs(X, y)

        finalColumns = []

        for s in selected_cols:

            if s > -1:
                finalColumns.append(copyDF.columns.tolist()[s])

        finalColumns.append('Buggy')

        if 'loc' in finalColumns:
            finalColumns.remove('loc')


        return None, finalColumns


    def apply_normalize(self, df):
        """
        Not used
        :param df:
        :return:
        """

        return df

    def apply_smote(self,df):

        originalDF = df.copy(deep=True)

        try:
            cols = df.columns
            smt = SMOTE.smote(df)
            df = smt.run()
            df.columns = cols

        except:
            return originalDF

        return df



def getChanges(project, releaseDate):
    releases = project.getReleases()
    for r in releases:
        if r.getReleaseDate() == releaseDate:
            return r.getChanges()

    return None

def getFirstChangesBetween(project, startDate, endDate):

    releases = project.getReleases()


    changeDF = None
    for r in releases:

        if r.getStartDate() >= startDate and r.getReleaseDate() <= endDate - 1:

            if r.getChanges() is not None and len(r.getChanges()) > 1:

                if changeDF is None:
                    changeDF = r.getChanges().copy(deep=True)
                else:
                    changeDF = changeDF.append(r.getChanges().copy(deep=True))



    return changeDF.copy(deep=True)



def getLastXChangesEndDate(startDate, endDate, project):


    releases = project.getReleases()


    changeDF = None
    for r in releases:

        if r.getStartDate() >= startDate and r.getReleaseDate() < endDate:
            if r.getChanges() is not None and len(r.getChanges()) > 1:

                if changeDF is None:
                    changeDF = r.getChanges().copy(deep=True)
                else:
                    changeDF = changeDF.append(r.getChanges().copy(deep=True))


    return changeDF

def getLastXChanges(currentReleaseObj, project, months):


    if currentReleaseObj is None:
        return None

    releases = project.getReleases()

    if months == math.inf:
        startDate = 0
    else:
        startDate = currentReleaseObj.getStartDate() - (months * one_month)

    changeDF = None
    for r in releases:

        if r.getStartDate() >= startDate and r.getReleaseDate() < currentReleaseObj.getStartDate():
            if r.getChanges() is not None and len(r.getChanges()) > 1:

                if changeDF is None:
                    changeDF = r.getChanges().copy(deep=True)
                else:
                    changeDF = changeDF.append(r.getChanges().copy(deep=True))


    return changeDF




def toNominal(changes):

    releaseDF = changes
    releaseDF.loc[releaseDF['Buggy'] >= 1, 'Buggy'] = 1
    releaseDF.loc[releaseDF['Buggy'] <= 0, 'Buggy'] = 0
    d = {1: True, 0: False}
    releaseDF['Buggy'] = releaseDF['Buggy'].map(d)

    return changes


def printRelease(releaseObj):

    changes = releaseObj.getChanges()

    print("\t Info : ", releaseObj.getReleaseDate(), '\t changes : ', len(changes),
          '\t Bug % : ', len(changes[changes['Buggy'] > 0]), '\tLA:', sum(changes['la']),  '\tLD:', sum(changes['ld'])
    , '\tEnt:', sum(changes['entropy']))

    print( )


def validTrainChanges(changes):
    print("Bug count = ",len(changes[changes['Buggy'] > 0]) , len(changes))
    return changes is not None and len(changes[changes['Buggy'] > 0]) > 5 \
           and len(changes[changes['Buggy'] == 0]) > 5


def validTestChanges(changes):
    return changes is not None and len(changes) > 4 and len(changes[changes['Buggy'] > 0]) > 0 \
           and len(changes[changes['Buggy'] == 0]) > 0


def validate(trainChanges, testChanges):
    return validTrainChanges(trainChanges) and validTestChanges(testChanges)


def getReleaseObject(pname, releaseDate):
    project = release_manager.getProject(pname)
    allReleases = project.getReleases()

    for r in allReleases:
        if r.getReleaseDate() == releaseDate:
            return r

    return None

def computeMeasures(test_df, clf, timeRow, codeChurned):

    F = {}

    errorMessage = 'CLF_ERROR'

    try:
        test_y = test_df.Buggy
        test_X = test_df.drop(labels=['Buggy'], axis=1)
        predicted = clf.predict(test_X)
        abcd = metrices.measures(test_y, predicted, codeChurned)
    except Exception as e:
        errorMessage = str(e).replace(',', ' ')
        abcd = None

    try:
        F['f1'] = [abcd.calculate_f1_score()]
    except:
        F['f1'] = ['Error_['+errorMessage+']']

    errorMessage = 'CLF_ERROR'

    try:
        F['precision'] = [abcd.calculate_precision()]
    except:
        F['precision'] = [errorMessage]

    try:
        F['recall'] = [abcd.calculate_recall()]
    except:
        F['recall'] = [errorMessage]



    try:
        F['pf'] = [abcd.get_pf()]
    except:
        F['pf'] = [errorMessage]


    try:
        F['g-score'] = [abcd.get_g_score()]
    except:
        F['g-score'] = [errorMessage]

    try:
        F['d2h'] = [abcd.calculate_d2h()]
    except:
        F['d2h'] = [errorMessage]

    # try:
    #     F['accuracy'] = [abcd.calculate_accuracy() * -1 ]
    # except:
    #     F['accuracy'] = [errorMessage]
    #
    #
    # print('warn')



    try:
        F['pci_20'] =  [1]
    except Exception as e:
        F['pci_20'] = [errorMessage]

    try:

        # tempPredicted = predicted
        # temptest_y = test_y.values.tolist()
        # initialFalseAlarm = 0
        #
        # for i in range(0, len(tempPredicted)):
        #
        #     if tempPredicted[i] == True and temptest_y[i] == False:
        #         initialFalseAlarm += 1
        #     elif not tempPredicted[i]:
        #         continue
        #     elif tempPredicted[i] == True and temptest_y[i] == True:
        #         break
        #
        # F['ifa'] = [initialFalseAlarm]

        F['ifa'] = [abcd.get_ifa()]
    except Exception as e:
    #     print('\t ',errorMessage)
        F['ifa'] = [errorMessage]

    try:
        F['ifa_roc'] = [abcd.get_ifa_roc()]
    except:
        F['ifa_roc'] = [errorMessage]

    try:
        F['roc_auc'] = [abcd.get_roc_auc_score()]
    except:
        F['roc_auc'] = [errorMessage]

    try:
        F['pd'] = [abcd.get_pd()]
    except:
        F['pd'] =[errorMessage]

    try:
        F['tp'] = [abcd.get_tp()]
    except:
        F['tp'] =[errorMessage]

    try:
        F['tn'] = [abcd.get_tn()]
    except:
        F['tn'] = [errorMessage]

    try:
        F['fp'] = [abcd.get_fp()]
    except:
        F['fp'] = [errorMessage]

    try:
        F['fn'] = [abcd.get_fn()]
    except:
        F['fn'] = [errorMessage]

    try:
        F['negopos'] = [abcd.negOverPos()]
    except:
        F['negopos'] = [errorMessage]

    try:
        F['balance'] = [abcd.balance()]
    except:
        F['balance'] = [errorMessage]

    try:
        F['brier'] = [abcd.brier()]
    except:
        F['brier'] = [errorMessage]

    try:
        pt = abcd.get_popt_20()
        F['popt20'] = [ pt ]
    except:
        F['popt20'] = [errorMessage]


    try:
        F['mcc'] = [abcd.mcc()]
    except:
        F['mcc'] = [errorMessage]

    return F


def getAllClassifiers():

    return [
        # KNeighborsClassifier(),
        # DecisionTreeClassifier(),
        boosting(),
        LogisticRegression()
        # RandomForestClassifier(n_estimators=50, random_state=1),
        # GaussianNB(),
        # SVC()
        # MLPClassifier(),
        # NuSVC()
    ]

"""
Classifiers from Ghotra et. al 2013
"""

class Correction(object):

    def __init__(self, projectName = None, base_estimator = LogisticRegression(), threshold=0.35):
        self.base_estimator = base_estimator
        self.projectName = projectName
        self.threshold = threshold

    def setThreshold(self, threshold):
        self.threshold = threshold


    def get_params(self):
        return {}


    def bagLR(self, trainX, trainY):


        for C in np.linspace(math.pow(1e-5,2), math.sqrt(1e-5), 100):
            for solver in ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']:
                for penalty in ['l1', 'l2', 'elasticnet', 'none']:
                    try:
                        classifier = LogisticRegression(C=C, solver=solver, penalty=penalty)
                        classifier.fit(trainX, trainY)
                        self.classifiers.append(classifier)
                    except Exception as e:
                        continue




    def fit(self, trainX, trainY):

        self.base_estimator.fit(trainX, trainY)

        if self.projectName is not None and self.projectName in correctionMap:
            self.classifiers = correctionMap[self.projectName]
        else:

            self.classifiers = []
            self.bagLR(trainX, trainY)


        if self.projectName is not None and self.projectName not in correctionMap:
            correctionMap[self.projectName] = self.classifiers



    def predict(self, testX):

        base_predictions = self.base_estimator.predict(testX)
        # print('before  = ',sum(base_predictions))

        aggregation = [0 for x in range(len(testX))]
        for c in self.classifiers:
            predictions = c.predict(testX)
            aIndex = -1
            for p in predictions:
                aIndex += 1
                aggregation[aIndex] += p

        defective_threshold = int(self.threshold * len(self.classifiers))

        clean_threshold = len(self.classifiers) - defective_threshold

        # print(defective_threshold, clean_threshold, 'defective_threshold:clean_threshold')



        base_prediction_index = -1

        window = int(0.5 * len(base_predictions))

        stopDefects = False
        for opinions in aggregation:

            base_prediction_index += 1

            if base_prediction_index > window:
                stopDefects = True


            if opinions > defective_threshold and stopDefects == False:
                base_predictions[base_prediction_index] = True

            if stopDefects and opinions < clean_threshold:
                base_predictions[base_prediction_index] = False


        # print('after = ', sum(base_predictions))
        return base_predictions


def stacking():

    estimators = []
    for c in  [ LogisticRegression(), SVC(), LinearSVC(), MLPClassifier()]:
        # for c in [  NuSVC(), SVC() ]:
        estimators.append((str(c), c))
    return StackingClassifier(final_estimator=LogisticRegression(), estimators=estimators) # , cv=LeaveOneOut()


def boosting():
    return GradientBoostingClassifier()


def bagging():
    estimators = []

    index = 0
    for c in [     SVC(),  RandomForestClassifier(),
                   GaussianNB()      ]:

        estimators.append((str(c)+str(index), c))
        index += 1

    stack =  StackingClassifier(final_estimator=LogisticRegression(), estimators=estimators) # , cv=LeaveOneOut()
    return BaggingClassifier(base_estimator=stack, n_estimators=100)

def voting():

    estimators = []

    unique_key = 0


    for c in [  LogisticRegression(), SVC()  ]:
        unique_key += 1
        estimators.append((getSimpleName(c) + '-' + str(unique_key), c))

    return VotingClassifier(estimators=estimators)

def voting1(c1, c2):

    estimators = []

    unique_key = 1

    for c in [ c1, c2 ]:
        unique_key += 1
        estimators.append((getSimpleName(c)+'-'+str(unique_key) , c))

    return VotingClassifier(estimators=estimators)


# TODO

# CLASSIFIERS = [  LogisticRegression , LinearSVC, SVC, DecisionTreeClassifier, RandomForestClassifier,
#                  GaussianNB, KNeighborsClassifier, MLPClassifier, RidgeClassifier ]


class HyperOptClassifier(object):

    def __init__(self, policy):
        self.name = 'HyperOptClassifier'
        self.best_classifier = None
        self.policy = policy


    def fit(self, training_commits):


        if self.best_classifier is None:

            split_size = int(len(training_commits)  )
            head = split_size # int ( 0.5 * len(training_commits) )
            tail = split_size # len(training_commits) - head

            print('head tail = ', head, tail)

            if self.policy == 'ALL' or self.policy == 'B' :
                self.best_classifier = hyper_opt_run(training_commits.head(head), training_commits.tail(tail))
            elif self.policy.startswith('E_2') or self.policy.startswith('E_1') or self.policy.startswith('E_B_2'):
                self.best_classifier = hyper_opt_run(early_sample(training_commits.head(head)), early_sample(training_commits.tail(tail)))


        print("hyperopt best classifier = ",self.best_classifier)


    def predict(self, testX):
        return self.best_classifier.predict(testX)




class DODGEClassifier(object):

    def __init__(self, policy):
        self.name = 'DODGEClassifier'
        self.best_classifier = None
        self.best_preprocessor = None
        self.best_score = None
        self.info = None
        self.policy = policy

    def fit(self, training_commits):

        if self.best_classifier is None:

            # split_size = int(len(training_commits)/ 1 )

            if self.policy == 'ALL' or    self.policy == 'B':

                _dodge = DODGE(training_commits, training_commits)
                self.best_classifier, self.best_preprocessor, self.best_score, self.info = _dodge.run()

            elif self.policy.startswith('E_2') or self.policy.startswith('E_B_2') or self.policy.startswith('E_1'):

                _dodge = DODGE(early_sample(training_commits),
                               early_sample(training_commits))

                self.best_classifier, self.best_preprocessor, self.best_score, self.info = _dodge.run()
            else:
                float("unknown policy ", self.policy)
        else:
            print("Already fitted!")

        print("Best score, classifier and pre-processor", self.best_score, self.best_classifier, self.best_preprocessor)



    def predict(self, testX):
        return self.best_classifier.predict(testX)


class CustomLogisticRegression(LogisticRegression):

    def predict(self, testX):

        # strategy # 1
        logistic_regression_predicted = super().predict(testX)

        la_values = testX['la'].values.tolist()
        la_values.sort()

        q1 = max(la_values[0:  int(0.25 * len(la_values))])
        q3 = max(la_values[0:  int(0.75 * len(la_values))])



        for i in range(0, len(logistic_regression_predicted)):

            if testX['la'].values.tolist()[i] >= q3:
                logistic_regression_predicted[i] = True
            elif testX['la'].values.tolist()[i] <= q1:
                logistic_regression_predicted[i] = False

        return logistic_regression_predicted



class ManualUpClassifier(object):

    def __init__(self, strategy):
        self.name = 'ManualUpClassifier'+str(strategy)
        self.strategy = strategy

    def fit(self, x, y):
        return

    def predict(self, testX):

        la_values = testX['la'].values.tolist()
        la_values.sort()
        num = len(la_values)

        q1 = max(la_values[ 0:  int(0.25 * num)])
        q2 = max(la_values[ 0:  int(0.5 * num)])
        q3 =max(la_values[ 0:  int(0.75 * num)])


        print(self.strategy, min(la_values), q1, q2, np.median(la_values), q3, ">>", la_values, '<<', max(la_values))

        try:

            if self.strategy == 1:
                return [False for x in testX['la'].values.tolist()]
            elif self.strategy == 2:
                return [x <= q1 for x in testX['la'].values.tolist()]
            elif self.strategy == 3:
                return [q1 < x <= q2 for x in testX['la'].values.tolist()]
            elif self.strategy == 4:
                return [q2 < x <= q3 for x in testX['la'].values.tolist()]
            elif self.strategy == 5:
                return [ x > q3  for x in testX['la'].values.tolist()]
            elif self.strategy == 6:
                return [x < q2 for x in testX['la'].values.tolist()]

            elif self.strategy == 7:
                return [q1 < x <= q3 for x in testX['la'].values.tolist()]
            elif self.strategy == 8:
                return [ x < q2 for x in testX['la'].values.tolist()]

            elif self.strategy == 9:
                return [ x <= q3 for x in testX['la'].values.tolist()]
            elif self.strategy == 10:
                return [ x > q1 for x in testX['la'].values.tolist()]
            elif self.strategy == 11:
                return [ True for x in testX['la'].values.tolist()]
            elif self.strategy == 12:
                return [  q1 <= x or ( q2 < x <= q3)  in testX['la'].values.tolist()]
            elif self.strategy == 13:
                return [  x > q3 or ( q1 < x <= q2)  in testX['la'].values.tolist()]
            elif self.strategy == 14:
                return [  x <= q1 or x > q3  in testX['la'].values.tolist()]
            elif self.strategy == 15:
                return [  x > q1 and x <= q3  in testX['la'].values.tolist()]
            elif self.strategy == 16:
                return [  x <= q2 or x > q3  in testX['la'].values.tolist()]
            elif self.strategy == 17:
                return [  x <= q1 or x > q2  in testX['la'].values.tolist()]

        except Exception as e:
            print('[ManualUpClassifier ERROR]', e)
            return [False for x in range(0, len(testX))]

    def toString(self):
        return self.name

class ManualDownClassifier(object):

    def __init__(self, strategy):
        self.name = 'ManualDownClassifier'+str(strategy)
        self.strategy = strategy

    def fit(self, x, y):
        return

    def predict(self, testX):

        la_values = testX['la'].values.tolist()
        la_values.sort()
        num = len(la_values)

        q1 = max(la_values[ 0:  int(0.25 * num)])
        q2 = max(la_values[ 0:  int(0.5 * num)])
        q3 =max(la_values[ 0:  int(0.75 * num)])


        print(self.strategy, min(la_values), q1, q2, np.median(la_values), q3, ">>", la_values, '<<', max(la_values))

        try:

            if self.strategy == 1:
                return [False for x in testX['la'].values.tolist()]
            elif self.strategy == 2:
                return [x <= q1 for x in testX['la'].values.tolist()]
            elif self.strategy == 3:
                return [q1 < x <= q2 for x in testX['la'].values.tolist()]
            elif self.strategy == 4:
                return [q2 < x <= q3 for x in testX['la'].values.tolist()]
            elif self.strategy == 5:
                return [ x > q3  for x in testX['la'].values.tolist()]
            elif self.strategy == 6:
                return [x < q2 for x in testX['la'].values.tolist()]

            elif self.strategy == 7:
                return [q1 < x <= q3 for x in testX['la'].values.tolist()]
            elif self.strategy == 8:
                return [ x >= q2 for x in testX['la'].values.tolist()]

            elif self.strategy == 9:
                return [ x <= q3 for x in testX['la'].values.tolist()]
            elif self.strategy == 10:
                return [ x > q1 for x in testX['la'].values.tolist()]
            elif self.strategy == 11:
                return [ True for x in testX['la'].values.tolist()]
            elif self.strategy == 12:
                return [  q1 <= x or ( q2 < x <= q3)  in testX['la'].values.tolist()]
            elif self.strategy == 13:
                return [  x > q3 or ( q1 < x <= q2)  in testX['la'].values.tolist()]
            elif self.strategy == 14:
                return [  x <= q1 or x > q3  in testX['la'].values.tolist()]
            elif self.strategy == 15:
                return [  x > q1 and x <= q3  in testX['la'].values.tolist()]
            elif self.strategy == 16:
                return [  x <= q2 or x > q3  in testX['la'].values.tolist()]
            elif self.strategy == 17:
                return [  x <= q1 or x > q2  in testX['la'].values.tolist()]

        except Exception as e:
            print('[ManualDownClassifier ERROR]', e)
            return [False for x in range(0, len(testX))]

    def toString(self):
        return self.name

class  CustomTLELClassifier(object):

    def __init__(self, policy):
        self.name = 'CustomTLELClassifier'
        self.estimators = []
        self.policy = policy

    def fit(self, training_commits):

        # print("\t training_commits # = ", len(training_commits))

        buggy_commits = training_commits[training_commits['Buggy'] == True]
        clean_commits = training_commits[training_commits['Buggy'] == False]

        # sample_size = min(25, min(len(buggy_commits), len(clean_commits)))

        if self.policy.startswith('E_2') or self.policy.startswith('E_B_2') or self.policy.startswith('E_1'):
            sample_size = min(25, min(len(buggy_commits), len(clean_commits)))
        else:
            sample_size = min(len(buggy_commits), len(clean_commits))

        self.estimators = []

        for i in range(0, 10):

            random_under_sampling = buggy_commits.sample(sample_size).copy(deep=True).append(
                clean_commits.sample(sample_size).copy(deep=True))

            rf = RandomForestClassifier()
            # lr = LogisticRegression()

            trainY = random_under_sampling.Buggy
            trainX = random_under_sampling.drop(labels=['Buggy'], axis=1)

            rf.fit(trainX, trainY)

            self.estimators.append( rf )




    def predict(self, testX):

        predicted_result = None
        for e in self.estimators:

            if predicted_result is None:
                predicted_result =  map(int, e.predict(testX))
            else:
                predicted_result = [a + b for a, b in zip(predicted_result, map(int, e.predict(testX)))]

        tlel_predicted = [x >= len(self.estimators)/2 for x in predicted_result]

        la_values = testX['la'].values.tolist()
        la_values.sort()

        q1 = max(la_values[0:  int(0.25 * len(la_values))])
        q3 = max(la_values[0:  int(0.75 * len(la_values))])

        for i in range(0, len(tlel_predicted)):

            if testX['la'].values.tolist()[i] >= q3:
                tlel_predicted[i] = True
            elif testX['la'].values.tolist()[i] <= q1:
                tlel_predicted[i] = False

        return tlel_predicted


class  TLELClassifier(object):

    def __init__(self, policy):
        self.name = 'TLELClassifier'
        self.estimators = []
        self.policy = policy

    def fit(self, training_commits):

        # print("\t training_commits # = ", len(training_commits))

        buggy_commits = training_commits[training_commits['Buggy'] == True]
        clean_commits = training_commits[training_commits['Buggy'] == False]

        # sample_size = min(25, min(len(buggy_commits), len(clean_commits)))

        if self.policy.startswith('E_2') or self.policy.startswith('E_B_2') or self.policy.startswith('E_1'):
            sample_size = min(25, min(len(buggy_commits), len(clean_commits)))
        else:
            sample_size = min(len(buggy_commits), len(clean_commits))

        self.estimators = []

        for i in range(0, 10):

            random_under_sampling = buggy_commits.sample(sample_size).copy(deep=True).append(
                clean_commits.sample(sample_size).copy(deep=True))

            rf = RandomForestClassifier()
            # lr = LogisticRegression()

            trainY = random_under_sampling.Buggy
            trainX = random_under_sampling.drop(labels=['Buggy'], axis=1)

            rf.fit(trainX, trainY)

            self.estimators.append( rf )

        # for i in range(0, 10):
        #
        #     random_under_sampling = buggy_commits.sample(sample_size).copy(deep=True).append(
        #         clean_commits.sample(sample_size).copy(deep=True))
        #
        #     # rf = RandomForestClassifier()
        #     svc = SVC()
        #
        #     trainY = random_under_sampling.Buggy
        #     trainX = random_under_sampling.drop(labels=['Buggy'], axis=1)
        #
        #     svc.fit(trainX, trainY)
        #
        #     self.estimators.append( svc )



        # self.classifier = StackingClassifier(estimators=estimators, final_estimator=DecisionTreeClassifier())
        #
        # ttrainY = training_commits.copy(deep=True).Buggy
        # ttrainX = training_commits.copy(deep=True).drop(labels=['Buggy'], axis=1)
        #
        #
        # self.classifier.fit(ttrainX, ttrainY)


    def predict(self, testX):

        # testX = test_commits.copy(deep=True).drop(labels=['Buggy'], axis=1)
        # print(testX.columns.values.tolist())

        predicted_result = None
        for e in self.estimators:

            if predicted_result is None:
                predicted_result =  map(int, e.predict(testX))
            else:
                predicted_result = [a + b for a, b in zip(predicted_result, map(int, e.predict(testX)))]

        return [x >= len(self.estimators)/2 for x in predicted_result]



CLASSIFIERS_TRADITIONAL = [LogisticRegression, SVC, DecisionTreeClassifier, RandomForestClassifier,
                   GaussianNB, KNeighborsClassifier]

# shrikanth

if RQ == 1 or RQ == 4:
    CLASSIFIERS = [LogisticRegression, SVC, DecisionTreeClassifier, RandomForestClassifier,
                   GaussianNB, KNeighborsClassifier]
elif RQ == 2:
    CLASSIFIERS = [LogisticRegression, SVC, DecisionTreeClassifier, RandomForestClassifier,
                   GaussianNB, KNeighborsClassifier] + [ ManualDownClassifier(8), ManualUpClassifier(8)]
elif RQ == 3:
     CLASSIFIERS = [ LogisticRegression, HyperOptClassifier, DODGEClassifier ,  TLELClassifier ]
elif RQ == 5:
     CLASSIFIERS = [LogisticRegression, SVC, DecisionTreeClassifier, RandomForestClassifier,
                   GaussianNB, KNeighborsClassifier]




def new_tlel_classifier():

    estimators = []
    for i in range(0, 10):
        rf = RandomForestClassifier()
        estimators.append(('rf' + str(i), rf))

    return StackingClassifier(estimators=estimators, final_estimator=DecisionTreeClassifier())










def getClassifiers():

    return CLASSIFIERS



def getFeatureSelectors():
    return [ 'CFS' ]


def getSimpleNames():

    return [ getSimpleName(clf) for clf in getClassifiers() ]

def getHeader():

    header = ['projectName', 'trainApproach', 'trainReleaseDate', 'testReleaseDate', 'train_changes', 'test_changes',
              'train_Bug_Per', 'test_Bug_Per', 'features_selected', 'classifier', 'featureSelector', 'SMOTE', 'SMOTE_time', 'test_time', 'train_time', 'goal_score']

    header += METRICS_LIST


    return header

def getSimpleName(classifier, tune=False):

        for a in ['SVC','DecisionTreeClassifier','RandomForestClassifier', 'GaussianNB', 'KNeighborsClassifier', 'CustomLogisticRegression', 'HyperOptClassifier', 'CustomTLELClassifier', 'TLELClassifier',  'DODGEClassifier',
            'LogisticRegression', 'SVC', 'DecisionTreeClassifier', 'RandomForestClassifier',
            'GaussianNB', 'KNeighborsClassifier', 'MultinomialNB'  , 'ManualDownClassifier' , 'ManualUpClassifier'  ]:

            if a in str(classifier):
                return  a

        float("error = "+str(classifier))


def getCacheKey(fs, projectName, trainReleaseDate, testReleaseDate):

    return fs+"_"+projectName+"_"+str(trainReleaseDate)+"_"+str(testReleaseDate)


def getFileName(projectName):
        return './'+RESULTS_FOLDER+'/project_' + projectName + "_results.csv"


unitTestResults = []


def toMajorMinor(emails):

    threshold = int(0.05 * len(emails))

    emailMap = {}

    for e in emails:
        if e not in emailMap:
            emailMap[e] = 0
        else:
            emailMap[e] = emailMap[e] + 1

    mminor = []

    for e in emails:

        if emailMap[e] >= threshold:
            mminor.append(1)
        else:
            mminor.append(0)

    # print('threshold = ',threshold, mminor, emails)
    return mminor


def dontDrop(changesDF, consider):

    if 'lt' in changesDF.columns.tolist() or 'fix' in changesDF.columns.tolist():
        for col in ['ns', 'nd', 'nf', 'entropy', 'ld', 'la', 'lt', 'fix', 'ndev', 'age', 'nuc', 'exp', 'rexp', 'sexp']:
            if col not in consider:

                if col in changesDF.columns:
                    changesDF = changesDF.drop(col, 1)

        return changesDF


# def normalize_log(changesDF):
#     """
#         log Normalization
#     """
#     numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
#
#     for c in [c for c in changesDF.columns if changesDF[c].dtype in numerics]:
#
#         if c != 'Buggy' and c != 'fix' and c != 'loc' and c != 'author_email':
#             changesDF[c] = changesDF[c] + abs(changesDF[c].min()) + 0.00001
#
#     for c in [c for c in changesDF.columns if changesDF[c].dtype in numerics]:
#
#         if c != 'Buggy' and c != 'fix' and c != 'loc' and c != 'author_email':
#             changesDF[c] = np.log10(changesDF[c])
#
#     return changesDF








def splitFeatures(type):

    featArr = type.split('$')

    features = []
    for f in featArr:

        if len(f) > 1:
            features.append(f)

    # print("Returning ",features)
    return features


def customPreProcess(changesDF, type, tune):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    for c in [c for c in changesDF.columns if changesDF[c].dtype in numerics]:
        if c != 'Buggy':
            changesDF[c] = changesDF[c] + abs(changesDF[c].min()) + 0.00001

    # changesDF['author_email'] = toMajorMinor(changesDF['author_email'].values.tolist())

    if type == 'diffusion':
        changesDF = dontDrop(changesDF, ['ns', 'nd', 'nf', 'entropy'])
    elif type == 'la':
        changesDF['la'] = changesDF['la'] / changesDF['lt']
        changesDF = dontDrop(changesDF, ['la'])
    elif type == 'size':
        # changesDF = dontDrop(changesDF, ['ld', 'la', 'lt'])
        if 'lt' in changesDF.columns.tolist():
            changesDF = dontDrop(changesDF, ['la', 'lt'])
        else:
            changesDF = changesDF[['la', 'ld', 'Buggy']]
            print("dropping >> ", changesDF.columns.tolist())
    elif type == 'purpose':
        changesDF = dontDrop(changesDF, ['fix'])
    elif type == 'history':
        changesDF = dontDrop(changesDF, ['ndev', 'age', 'nuc'])
    elif type == 'experience':
        changesDF = dontDrop(changesDF, ['exp', 'rexp', 'sexp', 'author_email'])
    elif type == 'top':
        changesDF = dontDrop(changesDF, ['entropy', 'la', 'ld', 'lt', 'exp' ])
    elif type is None:
        # changesDF = changesDF.drop('author_email', 1)
        if 'lt' in changesDF.columns.tolist():
            changesDF['la'] = changesDF['la'] / changesDF['lt']
            changesDF['ld'] = changesDF['ld'] / changesDF['lt']

            changesDF['lt'] = changesDF['lt'] / changesDF['nf']
            changesDF['nuc'] = changesDF['nuc'] / changesDF['nf']

            changesDF = changesDF.drop('nd', 1)
            changesDF = changesDF.drop('rexp', 1)

    elif type == 'i1':
        changesDF = dontDrop(changesDF, ['la', 'lt', 'entropy'])
    elif type == 'i2':
        changesDF = dontDrop(changesDF, ['la', 'lt', 'exp'])
    elif type == 'i3':
        changesDF = dontDrop(changesDF, ['la', 'lt', 'entropy', 'exp'])
    elif type == 'i4':
        changesDF = dontDrop(changesDF, ['la', 'lt', 'ndev'])
    elif type == 'i5':
        changesDF = dontDrop(changesDF, ['la', 'lt', 'entropy', 'ndev', 'exp'])
    elif '$' in type:
        changesDF = dontDrop(changesDF, splitFeatures(type))
    else:
        float("error")


    for c in [c for c in changesDF.columns if changesDF[c].dtype in numerics]:

        if c != 'Buggy' and c != 'fix' and c != 'loc' and c != 'author_email':
            changesDF[c] =  changesDF[c] + 0.0000001

    if not tune:

        """
        log Normalization
        """
        for c in [c for c in changesDF.columns if changesDF[c].dtype in numerics]:

            if c != 'Buggy' and c != 'fix' and c != 'loc' and c!='author_email':

                changesDF[c] = np.log10(changesDF[c])

    return changesDF

def splitEChanges(changesDF):

    defectsPerHalf = math.floor(getBugCount(changesDF)/2)
    nonDefectsPerHalf = math.floor(getNonBugCount(changesDF)/2)

    if len(changesDF) % 2 != 0:
        if getBugCount(changesDF) % 2 != 0:
            defectsPerHalf -= 1
        elif getNonBugCount(changesDF) % 2 != 0:
            nonDefectsPerHalf -= 1

    # print('==> ', defectsPerHalf, nonDefectsPerHalf, len(changesDF))

    onehalfIndex = []
    otherhalfIndex = []

    changesDF = changesDF.reset_index()

    for index, row in changesDF.iterrows():

        added = False
        if row['Buggy'] == True and defectsPerHalf > 0:
            onehalfIndex.append(index)
            defectsPerHalf -= 1
            added = True

        if row['Buggy'] == False and nonDefectsPerHalf > 0:
            onehalfIndex.append(index)
            nonDefectsPerHalf -= 1
            added = True

        if added == False:
            otherhalfIndex.append(index)

    onehalfChanges = changesDF.copy(deep=True)
    otherhalfChanges = changesDF.copy(deep=True)

    onehalfChanges = onehalfChanges.drop(onehalfChanges.index[onehalfIndex]).copy(deep=True)
    otherhalfChanges = otherhalfChanges.drop(otherhalfChanges.index[otherhalfIndex]).copy(deep=True)

    onehalfChanges = onehalfChanges.drop(labels=['index'], axis=1)
    otherhalfChanges = otherhalfChanges.drop(labels=['index'], axis=1)

    # print('==> ', len(otherhalfChanges), len(onehalfChanges))

    return otherhalfChanges, onehalfChanges


def getTrue(labels):

    c = 0

    for x in labels:

        if x or x == 'True':
            c += 1

    return c


def getFalse(labels):
    c = 0

    for x in labels:

        if x == False or x == 'False':
            c += 1

    return c

Dummy_Flag = False

smoteCopyMap = {}

current_time_ms = lambda: int(round(goodtime.time() * 1000))

from sklearn.model_selection import LeavePOut,LeaveOneOut

def new_classifier(learner):
    return clone(learner)


    # for c in CLASSIFIERS:
    #
    #    if c == type(learner):
    #         return c()
    #    elif type(learner) == VotingClassifier:
    #
    #         learner.get_estimators()
    #
    #
    #         return voting()
    #    elif type(learner) == BaggingClassifier:
    #         return bagging()


       # if c == type(learner):
       #     if type(learner) == StackingClassifier:
       #      return stacking()
       #     elif type(learner) == BaggingClassifier:
       #      return bagging()
       #     else:
       #      return c()





hyperParamMap = {}


def crossCheck(tempClf, trainChanges, testChanges, params, r):

    trainY = trainChanges.Buggy
    trainX = trainChanges.drop(labels=['Buggy'], axis=1)

    tempClf.fit(trainX, trainY)
    F = computeMeasures(testChanges, tempClf, [], [0 for x in range(0, len(testChanges))])

    # print(r, '\t Cross-check ',F[DE_GOAL], params)


def getAggregatedLabels(_trainX, _trainY, tuneX):

    labelsList = []

    for clf in getAllClassifiers():
        labelsList.append(new_classifier(clf).fit(_trainX, _trainY).predict(tuneX))

    labelSum = [0 for x in range(0, len(tuneX))]

    for labels in labelsList:

        index = -1
        for label in labels:
            index += 1

            if label:
                labelSum[index] += 1

    finalLabels = []

    ss = len(getAllClassifiers())
    prob = int(0.4 * ss)

    for l in labelSum:
        finalLabels.append( l >= prob )

    pb = percentage(sum(finalLabels), len(finalLabels))
    # print('% Buggy =  ', pb )

    if pb == 0 or pb == 100:
        finalLabels[0] = not finalLabels[0]
        pb = percentage(sum(finalLabels), len(finalLabels))
        # print('% changed Buggy =  ', pb, finalLabels)

    return finalLabels


correctionMap = {}

stackedInstanceMap = {}

projectStackMap = {}











def applyCFS(tune, trainApproach):

    return not (trainApproach.startswith('E_2') or trainApproach.startswith('E_B_2') or trainApproach.startswith('E_1')  or trainApproach.startswith('T_'))


def applySMOTE(tune, trainApproach):

    # if trainApproach.startswith('RESAMPLE_') or 'TCAPLUS' in trainApproach or trainApproach.startswith('150_'):
    #     return False
    # else:
    return trainApproach == 'ALL' or   trainApproach == 'B'


def apply_custom_processing(tune, trainApproach):

    if trainApproach.startswith('T_'):
        return False
    else:
        return True

cfsMap = {}

train_cache = {}

def preprocess(projectName,trainReleaseDate, tune, trainApproach, train_changes, test_changes, tune_changes, learner, retainFew):

    if trainApproach.startswith('T_') or trainApproach.startswith('E_T2_') or trainApproach.startswith('E_F2_T2_'): # TCA plus

        tca_train_changes = MLUtil().apply_smote(train_changes)
        tca_train_changes = customPreProcess(tca_train_changes, retainFew, tune)

        test_changes =  customPreProcess(test_changes, retainFew, tune)

        if trainApproach.startswith('T_'):
            tca_trainChanges, tca_testChanges = tca_plus.apply_tcaplus(tca_train_changes.copy(deep=True), test_changes.copy(deep=True), 5)
            return tca_trainChanges, tca_testChanges, None, 't5', True
        elif trainApproach.startswith('E_T2_'):
            tca_trainChanges, tca_testChanges = tca_plus.apply_tcaplus(tca_train_changes.copy(deep=True),
                                                                       test_changes.copy(deep=True), 2)
            return tca_trainChanges, tca_testChanges, None, 't2', True
        elif trainApproach.startswith('E_F2_T2_'):
            tca_trainChanges, tca_testChanges = tca_plus.apply_tcaplus(tca_train_changes.copy(deep=True),
                                                                       test_changes.copy(deep=True), 2)
            return tca_trainChanges, tca_testChanges, None, 'tsize', True




    process_train_changes = True

    if trainApproach.endswith('_ALL_CHANGES') and trainApproach in train_cache:
        process_train_changes = False

    smote = False

    if getBugCount(train_changes) != getNonBugCount(train_changes) and  applySMOTE(tune, trainApproach):
        smote = True
        train_changes = MLUtil().apply_smote(train_changes)

    fselector = ''


    if apply_custom_processing(tune, trainApproach):

        if process_train_changes:
            train_changes = customPreProcess(train_changes, retainFew, tune)

        test_changes = customPreProcess(test_changes, retainFew, tune)
        if tune_changes is not None:
            tune_changes = customPreProcess(tune_changes, retainFew, tune)

        fselector = 'CUS'




    if applyCFS(tune, trainApproach):

        fselector += '_CFS'

        # cfs_key = projectName + '_' + trainApproach + "_" + trainReleaseDate + "_" + str(len(train_changes))+"_"+getSimpleName(learner, tune)

        cfs_key = projectName + '_' + trainApproach + "_" + trainReleaseDate + "_" + str(len(train_changes))+"_"+str(tune)

        if cfs_key in cfsMap:
            someDF, selected_cols = None, cfsMap[cfs_key]
        else:
            # if tune:
            #     selected_cols = ['la', 'lt', 'Buggy']
            #     cfsMap[cfs_key] = selected_cols
            #     # print("<< warning only cfs >>")
            # else:
            someDF, selected_cols = MLUtil().apply_cfs(train_changes)
            cfsMap[cfs_key] = selected_cols




    else:
        selected_cols = train_changes.columns.tolist()

        # if 'loc' in selected_cols:
        #     selected_cols.remove('loc')

    if process_train_changes:
        train_changes = train_changes[selected_cols]

    test_changes = test_changes[selected_cols]

    if tune_changes is not None:
        tune_changes = tune_changes[selected_cols]

    if not process_train_changes:
        train_changes = train_cache[trainApproach]
        # # print("taking from cache ",trainApproach)
    else:
        if trainApproach.endswith('_ALL_CHANGES'):
            train_cache[trainApproach] = train_changes
            # # print("putting in cache")



    return train_changes, test_changes, tune_changes, fselector, smote


def permissible(tune, methodX, trainApproach):

    if tune == False and methodX == True:
        return False

    if tune and trainApproach.startswith('BELLWETHER'):
        return False
    else:
        return True

goal_scores = []

UNIT_TEST = False
def writeRowEntry(test_changes_processed, clf, trainX, trainY, train_start, projectName, trainApproach, trainReleaseDate, testReleaseDate,
                  train_changes_processed, returnResults, classifierName, fselector, smote, goal_score):

    train_time = goodtime.time() - train_start

    test_start = goodtime.time()


    F = computeMeasures(test_changes_processed, clf, [], [1 for x in range(0, len(test_changes_processed))])


    test_time = goodtime.time() - test_start

    metricsReport = F

    featuresSelectedStr = ''

    if trainX is not None:
        for sc in trainX.columns.tolist():
            featuresSelectedStr += '$' + str(sc)

    if train_changes_processed is not None:
        result = [projectName, trainApproach, trainReleaseDate, testReleaseDate,
                  len(train_changes_processed),
                  len(test_changes_processed),
                  percentage(len(train_changes_processed[train_changes_processed['Buggy'] > 0]),
                             len(train_changes_processed)),
                  percentage(len(test_changes_processed[test_changes_processed['Buggy'] > 0]),
                             len(test_changes_processed)),
                  featuresSelectedStr, classifierName, fselector, str(smote), 0, test_time, train_time, goal_score]
    else:
        result = [projectName,
                  trainApproach,
                  trainReleaseDate,
                  testReleaseDate,
                  None,
                  len(test_changes_processed),
                  None,
                  percentage(len(test_changes_processed[test_changes_processed['Buggy'] > 0]), len(test_changes_processed)),
                  featuresSelectedStr,
                  classifierName,
                  fselector,
                  str(smote),
                  0,
                  test_time,
                  train_time,
                  goal_score
                  ]

    if metricsReport is not None:
        for key in metricsReport.keys():
            result += metricsReport[key]
    else:
        for m in METRICS_LIST:
            result.append(str('UNABLE'))

    if UNIT_TEST == False:

        if returnResults:
            return metricsReport['balance']
        else:
            writeRow(getFileName(projectName), result)

    else:
        print("***************** ", classifierName, fselector, train_changes_processed.columns.tolist(),
              test_changes_processed.columns.tolist(), " Results **************************")
        if metricsReport is not None:

            print('\tprecision', metricsReport['precision'])
            print('\trecall', metricsReport['recall'])
            print('\tpf', metricsReport['pf'])
            print('\troc_auc', metricsReport['roc_auc'])
            print('\td2h', metricsReport['d2h'])
            print('*** \n ')
            return metricsReport['precision'], metricsReport['recall'], metricsReport['pf'], []
        else:

            print(metricsReport, getTrue(trainY), getFalse(trainY))
            return None, None, None, None

THRESHOLDS =  [40,50,60]

fitCache = {}

def makeE(some_df):


    samples = min(25, min(getBugCount(some_df), getNonBugCount(some_df)))

    buggyChangesDF = some_df[some_df['Buggy'] == True]
    nonBuggyChangesDF = some_df[some_df['Buggy'] == False]

    s = buggyChangesDF.sample(samples).copy(deep=True).append(
        nonBuggyChangesDF.sample(samples).copy(deep=True)).copy(deep=True)

    return s


def tca_sample(train_changes):

    return train_changes.head(150).copy(deep=True).append(
        train_changes.tail(150).copy(deep=True)).copy(deep=True)


def early_sample(train_changes, default=25):
    print("er - ", train_changes['Buggy'].values.tolist())
    buggyChangesDF = train_changes[train_changes['Buggy'] == True]
    nonBuggyChangesDF = train_changes[train_changes['Buggy'] == False]

    sample_size = min(default, min(len(buggyChangesDF), len(nonBuggyChangesDF)))

    return buggyChangesDF.sample(sample_size).copy(deep=True).append(
        nonBuggyChangesDF.sample(sample_size).copy(deep=True)).copy(deep=True)

fitCache = {}


def is_valid_experiment(learner,  trainApproach):

    policy_learner_rule_map = {}

    policy_learner_rule_map['LogisticRegression'] = ['E_B_2', 'E_2', 'E', 'ALL']
    policy_learner_rule_map['CustomLogisticRegression'] = ['E_B_2', 'E_2', 'E']
    policy_learner_rule_map['ManualDownClassifier'] = ['NO_TRAINING_DATA']
    policy_learner_rule_map['ManualUpClassifier'] = ['NO_TRAINING_DATA']

    policy_learner_rule_map['DODGEClassifier'] = ['ALL', 'E_2', 'E_B_2', 'B']
    policy_learner_rule_map['HyperOptClassifier'] = ['ALL', 'E_2', 'E_B_2', 'B']
    policy_learner_rule_map['TLELClassifier'] = ['ALL', 'E_2', 'E_B_2', 'B']

    policy_learner_rule_map['CustomTLELClassifier'] = ['ALL', 'E_2', 'E_B_2', 'B']

    policy_learner_rule_map['KNeighborsClassifier'] = ['ALL', 'E' , 'E_2']
    policy_learner_rule_map['DecisionTreeClassifier'] = ['ALL', 'E', 'E_2']
    policy_learner_rule_map['RandomForestClassifier'] = ['ALL', 'E', 'E_2']
    policy_learner_rule_map['GaussianNB'] = ['ALL', 'E', 'E_2']
    policy_learner_rule_map['SVC'] = ['ALL', 'E', 'E_2']





    return getSimpleName(learner) in policy_learner_rule_map and trainApproach in policy_learner_rule_map[getSimpleName(learner)]

def performPredictionRunner(projectName, originalTrainChanges, originalTestChanges, trainReleaseDate, testReleaseDate, trainApproach,
                            testReleaseStartDate=None, retainFew=None, tuneChanges=None, returnResults=False):


    for repeat in range(1, 2):

            if (trainApproach != 'NO_TRAINING_DATA' and validTrainChanges(originalTrainChanges) == False) or \
                    validTestChanges(originalTestChanges) == False:
                return

            for learner in getClassifiers():


                    # try:

                        tune = 'DODGEClassifier' in str(learner) or 'HyperOptClassifier' in str(learner)

                        if not is_valid_experiment(learner, trainApproach):
                            print("skipping ", learner, trainApproach)
                            continue
                        else:
                            print("Not skipping ", learner, trainApproach)

                            process_start =  goodtime.time()

                            clfkey = None
                            if trainApproach != 'NO_TRAINING_DATA':

                                if (trainApproach == 'ALL' or trainApproach == 'B') and (
                                        'DODGEClassifier' in str(learner) or 'HyperOptClassifier' in str(learner)):

                                    if len(originalTrainChanges) < 150:
                                        float('error')
                                    elif len(originalTrainChanges) > 957:
                                        train_tune_sample_size = 957
                                    else:
                                        train_tune_sample_size = len(originalTrainChanges)

                                    train_changes_copy = toNominal(originalTrainChanges.head(train_tune_sample_size).copy(deep=True))
                                else:
                                    train_changes_copy = toNominal(originalTrainChanges.copy(deep=True))

                                test_changes_copy = toNominal(originalTestChanges.copy(deep=True))
                                tune_changes_copy = None

                                if not tune and (trainApproach.startswith('E_2') or trainApproach.startswith('E_1')
                                                  or trainApproach.startswith('E_B_2')):
                                    train_changes_copy = early_sample(train_changes_copy)

                                train_changes_processed, test_changes_processed, tune_changes_processed, fselector, smote = preprocess(
                                    projectName,
                                    trainReleaseDate, tune,
                                    trainApproach,
                                    train_changes_copy,
                                    test_changes_copy, tune_changes_copy, learner, retainFew)

                                if (trainApproach == 'ALL' or trainApproach == 'B') and ('DODGEClassifier' in str(learner) or 'HyperOptClassifier' in str(learner)):
                                    clfkey = trainApproach + '_' + getSimpleName(learner, tune) + '_' + str(len(train_changes_processed))
                                else:
                                    clfkey = trainApproach + '_' + getSimpleName(learner, tune) + '_' + str(trainReleaseDate)
                            else:
                                train_changes_processed = None
                                test_changes_processed = originalTestChanges.copy(deep=True)
                                fselector, smote = None, None




                            if  clfkey is None or clfkey not in fitCache  :

                                print("Not in cache ", clfkey)


                                preprocessor = None
                                info = None

                                if getSimpleName(learner) in ['TLELClassifier']:
                                    classifier = TLELClassifier(trainApproach)
                                    classifier.fit(train_changes_processed)
                                    final_classifier = classifier
                                elif getSimpleName(learner) in ['CustomTLELClassifier']:
                                    classifier = CustomTLELClassifier(trainApproach)
                                    classifier.fit(train_changes_processed)
                                    final_classifier = classifier
                                elif getSimpleName(learner) in ['DODGEClassifier']:
                                    classifier = DODGEClassifier(trainApproach)
                                    classifier.fit(train_changes_processed)
                                    final_classifier = classifier.best_classifier
                                    preprocessor = classifier.best_preprocessor
                                    info = classifier.info
                                elif getSimpleName(learner) in ['HyperOptClassifier']:
                                    classifier = HyperOptClassifier(trainApproach)
                                    classifier.fit(train_changes_processed)
                                    final_classifier = classifier.best_classifier

                                    temp = train_changes_processed.copy(deep=True)
                                    trainY = temp.Buggy
                                    trainX = temp.drop(labels=['Buggy'], axis=1)
                                    print("error @ ", trainApproach)
                                    final_classifier.fit(trainX, trainY)

                                elif getSimpleName(learner) == 'LogisticRegression':
                                    final_classifier =  LogisticRegression()
                                    temp = train_changes_processed.copy(deep=True)
                                    trainY = temp.Buggy
                                    trainX = temp.drop(labels=['Buggy'], axis=1)
                                    final_classifier.fit(trainX, trainY)
                                elif getSimpleName(learner)  == 'SVC':
                                    final_classifier =  SVC()
                                    temp = train_changes_processed.copy(deep=True)
                                    trainY = temp.Buggy
                                    trainX = temp.drop(labels=['Buggy'], axis=1)
                                    final_classifier.fit(trainX, trainY)
                                elif getSimpleName(learner)  == 'DecisionTreeClassifier':
                                    final_classifier =  DecisionTreeClassifier()
                                    temp = train_changes_processed.copy(deep=True)
                                    trainY = temp.Buggy
                                    trainX = temp.drop(labels=['Buggy'], axis=1)
                                    final_classifier.fit(trainX, trainY)
                                elif getSimpleName(learner)  == 'RandomForestClassifier':
                                    final_classifier =  RandomForestClassifier()
                                    temp = train_changes_processed.copy(deep=True)
                                    trainY = temp.Buggy
                                    trainX = temp.drop(labels=['Buggy'], axis=1)
                                    final_classifier.fit(trainX, trainY)
                                elif getSimpleName(learner) == 'GaussianNB':
                                    final_classifier =  GaussianNB()
                                    temp = train_changes_processed.copy(deep=True)
                                    trainY = temp.Buggy
                                    trainX = temp.drop(labels=['Buggy'], axis=1)
                                    final_classifier.fit(trainX, trainY)
                                elif getSimpleName(learner) == 'KNeighborsClassifier':
                                    final_classifier =  KNeighborsClassifier()
                                    temp = train_changes_processed.copy(deep=True)
                                    trainY = temp.Buggy
                                    trainX = temp.drop(labels=['Buggy'], axis=1)
                                    final_classifier.fit(trainX, trainY)

                                elif getSimpleName(learner) == 'ManualDownClassifier':
                                    final_classifier =  learner
                                elif getSimpleName(learner) == 'ManualUpClassifier':
                                    final_classifier =  learner
                                else:
                                    final_classifier = None
                                    float("unknown classifier")


                                fitCache[clfkey] = [final_classifier, preprocessor, info]

                            else:
                                print("Taking from cache ", clfkey, fitCache[clfkey][0])


                            temp_preprocessor = fitCache[clfkey][1]

                            # if temp_preprocessor is not None:
                            #     train_changes_processed = transform(train_changes_processed, temp_preprocessor)
                            #     test_changes_processed = transform(test_changes_processed, temp_preprocessor)
                            #     print("\t Applying pre-processor = ", temp_preprocessor, fitCache[clfkey][0])

                            if trainApproach != 'NO_TRAINING_DATA':
                                trainY = train_changes_processed.Buggy
                                trainX = train_changes_processed.drop(labels=['Buggy'], axis=1)
                                train_instances = str(len(train_changes_processed))
                            else:
                                trainX = None
                                trainY = None
                                train_instances = None
                            #
                            # for repeat in range(1,2):
                            #
                            clf = fitCache[clfkey][0]
                            #
                            #     if False:
                            #         print('\t', repeat, tune, clf)
                            #
                            #     classifierName = getSimpleName(learner, tune)
                            #
                            #     try:
                            #
                            #         # if type(clf) == TLELClassifier or type(clf) == HyperOptClassifier:
                            #         #     clf.fit(train_changes_processed)
                            #         # else:
                            #         #     clf.fit(trainX, trainY)
                            #
                            #     except Exception as e1:
                            #         print('[RUNNER-1-ERROR] ', e1)
                            #         continue

                                # print(">> sending ", clf)

                            if getSimpleName(learner) in ['ManualDownClassifier', 'ManualUpClassifier']:
                                simple_name = learner.toString()
                            else:
                                simple_name = getSimpleName(learner, tune)


                            writeRowEntry(test_changes_processed, clf, trainX, trainY, process_start, projectName, trainApproach,
                                              train_instances, testReleaseDate,
                                              train_changes_processed, returnResults, simple_name, fselector, smote, str(fitCache[clfkey][2]))

                    # except Exception as e2:
                    #     print('[RUNNER-2-ERROR] ', e2)




def getBugCount(  xDF ):

    return len(xDF[xDF['Buggy'] == True])

def getNonBugCount(  xDF):

    return len(xDF[xDF['Buggy'] == False])

def getBuggyPastRelease(pastReleases):

    buggyPastRelease = None

    for pr in pastReleases:

        if buggyPastRelease is None:
            buggyPastRelease = pr
        elif getBugCount(pr.getChanges()) > getBugCount(buggyPastRelease.getChanges()):
            buggyPastRelease = pr


    return buggyPastRelease


# def getPreviousRelease(projectObj, testReleaseDate):
#
#     sourceReleases = projectObj.getReleases()
#
#     previousRelease = None
#     for sr in sourceReleases:
#
#         if sr.getReleaseDate() >= testReleaseDate:
#             break
#
#         previousRelease = sr
#
#     if previousRelease is not None:
#         print('prev = ', previousRelease.getReleaseDate(), testReleaseDate)
#
#     return previousRelease


def getRandomProject(ignoreProject):

    filteredCrossProject = []

    for cl in release_manager.getProjectNames():
        if cl != ignoreProject:
            filteredCrossProject.append(cl)

    randomIndex = randint(0, len(filteredCrossProject) - 1)

    return filteredCrossProject[randomIndex]


def getBugPercentage(xDF):
    # print(xDF)
    return percentage(len(xDF[xDF['Buggy'] == True]), len(xDF))





def getCrossTrainReleaseRandom(testReleaseObj, ignoreProject):

    testReleaseBugs = getBugCount(testReleaseObj.getChanges())

    crossProjectObj = getRandomProject(ignoreProject)

    crossBuggiestRelease = getBuggyPastRelease(crossProjectObj.getReleases())

    return crossBuggiestRelease




def getMostRecentPastRelease(projectObj, testReleaseObj):
    releaseList = projectObj.getReleases()

    mostRecentRelease = None

    deltaDiff = float("-inf")

    for r in releaseList:

        currentDeltaDiff = r.getReleaseDate() - testReleaseObj.getReleaseDate()

        if currentDeltaDiff >= 0:
            break
        elif currentDeltaDiff > deltaDiff:
            mostRecentRelease = r
            deltaDiff = currentDeltaDiff

    return mostRecentRelease

# def getSimilarPastRelease( projectName, testRelease, pastReleases):
#
#         similarRelease = None
#         overallDelta = math.inf
#
#         for pastRelease in pastReleases:
#
#             if similarRelease is None:
#                 similarRelease = pastRelease
#                 overallDelta = computeReleaseSimilarity(projectName, testRelease, pastRelease)
#                 continue
#
#             currentDelta = computeReleaseSimilarity(projectName, testRelease, pastRelease)
#
#             if currentDelta < overallDelta:
#                 similarRelease = pastRelease
#                 overallDelta = currentDelta
#             elif currentDelta == overallDelta:
#                 if getBugCount(pastRelease.getChanges()) > getBugCount(similarRelease.getChanges()):
#                     similarRelease = pastRelease
#             else:
#                 continue
#
#         return similarRelease


def getReleasesInFirstMonths(releaseList, month):

    # print("received ",len(releaseList), month)
    firstReleaseDate = min([r.getStartDate() for r in releaseList])

    start = firstReleaseDate
    end = start + (month * one_month)

    # print(start, end)

    releaseObjects  = []
    for r in releaseList:
        if end > r.getReleaseDate() > start:
            # print(start, r.getReleaseDate(), end)
            releaseObjects.append(r)

    return releaseObjects

def getReleasesFromYear(releaseList, year):


    firstReleaseDate = min([r.getReleaseDate() for r in releaseList])

    start = firstReleaseDate + ( (year - 1) * one_year)
    end = start + one_year

    releaseObjects  = []
    for r in releaseList:
        if end > r.getReleaseDate() >= start:
            releaseObjects.append(r)

    return releaseObjects

def aggregate(releaseObjList):

    changes = None

    for r in releaseObjList:

        if changes is None:
            changes = r.getChanges()
        else:
            changes = changes.append(r.getChanges())

    return changes


def createApproachLabel(trainYear, testYear, t):

    return "Train_"+str(trainYear)+"_Test_"+str(testYear)+"_"+t


def get6Months(testReleaseList):

    first6, next6 = [], []

    if len(testReleaseList) > 0:

        midWay = testReleaseList[0].getReleaseDate() + (6 * one_month)

        for t in testReleaseList:

            if t.getReleaseDate() < midWay:
                first6.append(t)
            else:
                next6.append(t)

    return first6, next6


def isOverlapping(trainReleaseList, testReleaseList):


    for t in trainReleaseList:
        for test in testReleaseList:

            if t.getReleaseDate() == test.getReleaseDate():
                return False

    return True


def getBuggiestYearBefore(projectObj, testYear):

    buggiestYear = None

    releaseList = projectObj.getReleases()

    bugCountSoFar = -1
    for year in range(1, testYear):

        bugCount = getBugCount(aggregate(getReleasesFromYear(releaseList, year)))

        if bugCountSoFar < bugCount:
            buggiestYear = year
            bugCountSoFar = bugCount


    return buggiestYear


def numBuggyInstances(testInstances):
    return round(testInstances/3)

def numCleanInstances(testInstances):
    return round(testInstances/2)




def reportOverlap(trainReleaseList, testRelList):

    for r in trainReleaseList:

        for t in testRelList:

            if r.getReleaseDate() >= t.getReleaseDate():
                float("overlap exists !!!!")


# def isWithin(release, startDate, endDate):
#
#     return release.getStartDate() > startDate and release.getReleaseDate() < endDate
#
#
# def hasOverlap(release, startDate, endDate):
#
#     return release.getStartDate() > startDate or release.getReleaseDate() < endDate


def rq1a(projectName):

    projectObj = release_manager.getProject(projectName)
    releaseList = projectObj.getReleases()
    projectStart = min([r.getStartDate() for r in releaseList])

    for testReleaseObj in releaseList:

        if True: #testReleaseObj.getReleaseDate() - testReleaseObj.getStartDate() > one_month:  # Sizeable release

            trainingRegion = getFirstChangesBetween(projectObj, projectStart, testReleaseObj.getStartDate())

            if trainingRegion is None or len(trainingRegion) < 3:
                continue

            buggyChangesDF = trainingRegion[ trainingRegion['Buggy'] == True]
            nonBuggyChangesDF = trainingRegion[ trainingRegion['Buggy'] == False]

            # For All sel rule
            allBugs = buggyChangesDF.copy(deep=True)
            allNonBugs = nonBuggyChangesDF.copy(deep=True)
            allTrainChangesDF = allBugs.append(allNonBugs)


            if len(buggyChangesDF) > 90 and len(nonBuggyChangesDF) > 90 : # To ensure equal number of experiments

                for samples in [12, 25, 50, 100]:

                    for buggyPercentage in [10, 20, 30, 40, 50, 60, 70, 80, 90]:

                        buggySamples = round(buggyPercentage * samples/100)
                        nonBuggySamples = abs(samples - buggySamples)

                        bugs =  buggyChangesDF.sample(buggySamples).copy(deep=True)
                        nonBugs = nonBuggyChangesDF.sample(nonBuggySamples).copy(deep=True)

                        trainChangesDF = bugs.append(nonBugs)

                        performPredictionRunner(projectName, trainChangesDF, testReleaseObj.getChanges(), str(len(trainChangesDF)),
                                                testReleaseObj.getReleaseDate(), 'RESAMPLE_' + str(buggySamples) +"_" + str(nonBuggySamples))

                        # print("pass ",testReleaseObj.getReleaseDate(), )

            # else:
            #     print('\t condition fail', len(buggyChangesDF), len(nonBuggyChangesDF))


            # ALL
            # print(testReleaseObj.getReleaseDate(), ' ALL ')
            performPredictionRunner(projectName, allTrainChangesDF, testReleaseObj.getChanges(),
                                    str(len(allTrainChangesDF)),
                                    testReleaseObj.getReleaseDate(), 'ALL')
            # print('\t Pass ',testReleaseObj.getReleaseDate(), ' ALL ')


def sampleAndPredict(projectName, year, trainChanges, testReleaseObj, bugCount, nonBugCount):

    if trainChanges is None or len(trainChanges) < 100:
        return

    buggyChangesDF = trainChanges[trainChanges['Buggy'] == True]
    nonBuggyChangesDF = trainChanges[trainChanges['Buggy'] == False]

    if len(buggyChangesDF) < 40 or len(nonBuggyChangesDF) < 60:
        return

    bugs = buggyChangesDF.sample(bugCount).copy(deep=True)
    nonBugs = nonBuggyChangesDF.sample(nonBugCount).copy(deep=True)

    trainChangesDF = bugs.append(nonBugs)

    performPredictionRunner(projectName, trainChangesDF, testReleaseObj.getChanges(), str(len(trainChangesDF)),
                            testReleaseObj.getReleaseDate(), 'RESAMPLE_YEAR_' + str(year) +"_" + str(bugCount) +"_" + str(nonBugCount))



def rq3(projectName):

    projectObj = release_manager.getProject(projectName)
    releaseList = projectObj.getReleases()

    projectStart = min([r.getStartDate() for r in releaseList])
    projectAfterXMonths = projectStart + (one_month * 12)

    trainingRegion = getFirstChangesBetween(projectObj, projectStart, projectAfterXMonths)

    previousRelease = None
    for testReleaseObj in releaseList:

        if testReleaseObj.getReleaseDate() < projectAfterXMonths:
            # print("Ignoring ", projectObj.getName(), projectStart, '>>',testReleaseObj.getStartDate(), testReleaseObj.getReleaseDate(), '<<', projectAfterXMonths)
            continue

        if True:#testReleaseObj.getReleaseDate() - testReleaseObj.getStartDate() > one_month: # sizeable release

            if trainingRegion is None or len(trainingRegion) <= 0:
                continue

            print("Testing ", projectObj.getName(), projectStart, '>>', testReleaseObj.getStartDate(),
                  testReleaseObj.getReleaseDate(), '<<',
                  projectAfterXMonths, len(trainingRegion))


            buggyChangesDF = trainingRegion[trainingRegion['Buggy'] == True]
            nonBuggyChangesDF = trainingRegion[trainingRegion['Buggy'] == False]

            allChangesDF = getLastXChangesEndDate(0, testReleaseObj.getStartDate(), projectObj)
            recent3MonthChanges = getLastXChanges(testReleaseObj, projectObj, 3)
            recent6MonthChanges = getLastXChanges(testReleaseObj, projectObj, 6)

            #
            if len(buggyChangesDF) < 40 or len(nonBuggyChangesDF) < 60 or allChangesDF is None or recent3MonthChanges is None or recent6MonthChanges is None: # fair chance
                continue


            performPredictionRunner(projectName, allChangesDF, testReleaseObj.getChanges(), str(len(allChangesDF)),
                                    testReleaseObj.getReleaseDate(), 'ALL')
            performPredictionRunner(projectName, recent3MonthChanges, testReleaseObj.getChanges(), str(len(recent3MonthChanges)),
                                    testReleaseObj.getReleaseDate(), '3MONTHS')
            performPredictionRunner(projectName, recent6MonthChanges, testReleaseObj.getChanges(), str(len(recent6MonthChanges)),
                                    testReleaseObj.getReleaseDate(), '6MONTHS')

            RESAMPLE_YEAR_1_20_30 = buggyChangesDF.sample(20).copy(deep=True).append(
                nonBuggyChangesDF.sample(30).copy(deep=True)).copy(deep=True)

            RESAMPLE_YEAR_1_40_60 = buggyChangesDF.sample(40).copy(deep=True).append(
                nonBuggyChangesDF.sample(60).copy(deep=True)).copy(deep=True)

            performPredictionRunner(projectName, RESAMPLE_YEAR_1_20_30, testReleaseObj.getChanges(), str(len(RESAMPLE_YEAR_1_20_30)), testReleaseObj.getReleaseDate(), 'RESAMPLE_YEAR_1_20_30')
            performPredictionRunner(projectName, RESAMPLE_YEAR_1_40_60, testReleaseObj.getChanges(), str(len(RESAMPLE_YEAR_1_40_60)), testReleaseObj.getReleaseDate(), 'RESAMPLE_YEAR_1_40_60')

            if previousRelease is None:
                previousRelease = testReleaseObj
                continue
            else:
                performPredictionRunner(projectName, previousRelease.getChanges(), testReleaseObj.getChanges(), previousRelease.getReleaseDate(), testReleaseObj.getReleaseDate(), 'RECENT_RELEASE')

            previousRelease = testReleaseObj



def getYear1NumberOfCommits(projectName):

    projectObj = release_manager.getProject(projectName)
    releaseList = projectObj.getReleases()

    projectStart = min([r.getStartDate() for r in releaseList])
    projectAfterXMonths = projectStart + (one_month * 12)

    trainingRegion = getFirstChangesBetween(projectObj, projectStart, projectAfterXMonths)

    if trainingRegion is None:
        # print(projectName, ' not found')
        return len(releaseList[0].getChanges())

    return len(trainingRegion)





def alreadyPerdicted(crossProject, targetProject, testReleaseDate):

    df = pd.read_csv('./results/project_'+targetProject+'_results.csv')
    df = df[ ( df['trainApproach'].str.strip() == crossProject) & (df['testReleaseDate'] == testReleaseDate) ]

    # print("checking ", crossProject, testReleaseDate, len(df))

    return len(df) > 0


# def performCross(crossProject, targetProject):
#
#     crossChanges = release_manager.getProject(crossProject).getAllChanges()
#     testReleases = release_manager.getProject(targetProject).getReleases()
#
#     for testReleaseObj in testReleases:
#
#         try:
#
#             if valid(testReleaseObj.getChanges()) and not alreadyPerdicted(crossProject, targetProject, testReleaseObj.getReleaseDate()):
#
#                 performPredictionRunner(targetProject, crossChanges, testReleaseObj.getChanges(),
#                                     str(len(crossChanges)), testReleaseObj.getReleaseDate(),
#                                     crossProject)
#         except Exception as e:
#             # print('[ERROR]', targetProject, crossProject, str(testReleaseObj.getReleaseDate()), str(e))
#             continue


# def crossWithAllProjects(targetProject):
#
#     if targetProject in ['active_merchant', 'android', 'AnySoftKeyboard', 'beets', 'bitcoin', 'brackets', 'camel',
#                              'cassandra', 'coreclr', 'curl', 'django-rest-framework', 'django', 'druid',
#                              'elasticsearch', 'fluentd', 'gevent', 'go-ethereum', 'gradle', 'grav', 'guava',
#                              'home-assistant', 'homebrew-cask', 'ionic', 'ipython', 'istio', 'jekyll', 'kafka',
#                              'logstash', 'metasploit-framework', 'netty', 'numpy', 'pandas', 'passenger', 'peewee',
#                              'phpunit', 'piwik', 'postgres', 'redis', 'rubocop', 'ruby', 'swoole', 'symfony']:
#
#         for crossProject in release_manager.getProjectNames():
#
#             if crossProject != targetProject:
#
#                 df = pd.read_csv('./results/project_' + targetProject + '_results.csv')
#                 completedProjects = df['trainApproach'].values.tolist()
#                 completedProjects.sort()
#
#                 if crossProject not in completedProjects or completedProjects[len(completedProjects) - 1] == crossProject:
#                     performCross(crossProject, targetProject)
#                 else:
#                     # print("skipp cross ",crossProject)
#     else:
#         # print("Skipping ",targetProject, " as already 100%")

# def runAllExperiments(projectName):
#
#     projectObj = release_manager.getProject(projectName)
#     releaseList = projectObj.getReleases()
#
#     commitsList = [150]
#     testReleaseList = getReleasesAfter(max(commitsList), releaseList)
#     projectStart = min([r.getStartDate() for r in releaseList])
#
#     for testReleaseObj in testReleaseList:
#
#         if True: #testReleaseObj.getReleaseDate() - testReleaseObj.getStartDate() > one_month:  # sizeable release
#
#             projectChanges = projectObj.getAllChanges()
#             for commits in commitsList:
#
#                 if len(projectChanges) >= max(commitsList): # ensure equal number of experiments
#
#                     trainingRegion = projectChanges.head(commits).copy(deep=True)
#
#                     if trainingRegion is not None and len(trainingRegion) > 0:
#                         buggyChangesDF = trainingRegion[trainingRegion['Buggy'] == True]
#                         nonBuggyChangesDF = trainingRegion[trainingRegion['Buggy'] == False]
#
#                         # for samples in []: #[12, 25, 50]:
#                         # for sampleInfo in [[25,25]]:
#                         buggySamples =  25
#                         nonBuggySamples = 25 #abs(samples - buggySamples)
#
#                         if buggySamples > 5 and nonBuggySamples > 5:
#
#                             if len(buggyChangesDF) >= buggySamples and len(nonBuggyChangesDF) >= nonBuggySamples:
#
#                                 RESAMPLE_COMMITS = buggyChangesDF.sample(buggySamples).copy(deep=True).append(
#                                     nonBuggyChangesDF.sample(nonBuggySamples).copy(deep=True)).copy(deep=True)
#
#                                 performPredictionRunner(projectName, RESAMPLE_COMMITS, testReleaseObj.getChanges(),
#                                                         str(len(RESAMPLE_COMMITS)), testReleaseObj.getReleaseDate(),
#                                                 'RESAMPLE_' + str(commits) + '_' + str(buggySamples) +'_' + str(nonBuggySamples))
#
#
#             trainingRegion = getFirstChangesBetween(projectObj, projectStart, testReleaseObj.getStartDate())
#             buggyChangesDF = trainingRegion[trainingRegion['Buggy'] == True]
#             nonBuggyChangesDF = trainingRegion[trainingRegion['Buggy'] == False]
#
#             # for samples in [12, 25, 50]:
#             #
#             #     for buggyPercentage in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
#             #         buggySamples = round(buggyPercentage * samples / 100)
#             #         nonBuggySamples = abs(samples - buggySamples)
#
#             buggySamples = 25
#             nonBuggySamples = 25
#
#             if buggySamples > 5 and nonBuggySamples > 5:
#
#                 if len(buggyChangesDF) >= buggySamples and len(nonBuggyChangesDF) >= nonBuggySamples:
#
#                     RESAMPLE_COMMITS = buggyChangesDF.sample(buggySamples).copy(deep=True).append(
#                         nonBuggyChangesDF.sample(nonBuggySamples).copy(deep=True)).copy(deep=True)
#
#                     performPredictionRunner(projectName, RESAMPLE_COMMITS, testReleaseObj.getChanges(),
#                                             str(len(RESAMPLE_COMMITS)), testReleaseObj.getReleaseDate(),
#                                         'RESAMPLE_' + str(buggySamples) + '_' + str(nonBuggySamples))
#
#     previousRelease = None
#
#     for testReleaseObj in testReleaseList:
#
#         if True: #testReleaseObj.getReleaseDate() - testReleaseObj.getStartDate() > one_month:
#
#             allChangesDF = getLastXChangesEndDate(0, testReleaseObj.getStartDate(), projectObj)
#             recent3MonthChanges = getLastXChanges(testReleaseObj, projectObj, 3)
#             recent6MonthChanges = getLastXChanges(testReleaseObj, projectObj, 6)
#
#             if allChangesDF is not None and len(allChangesDF) >= max(commitsList):
#                 performPredictionRunner(projectName, allChangesDF, testReleaseObj.getChanges(), str(len(allChangesDF)),
#                                         testReleaseObj.getReleaseDate(), 'ALL')
#
#             if recent3MonthChanges is not None:
#                 performPredictionRunner(projectName, recent3MonthChanges, testReleaseObj.getChanges(),
#                                         str(len(recent3MonthChanges)),
#                                         testReleaseObj.getReleaseDate(), '3MONTHS')
#
#             if recent6MonthChanges is not None:
#                 performPredictionRunner(projectName, recent6MonthChanges, testReleaseObj.getChanges(),
#                                         str(len(recent6MonthChanges)),
#                                         testReleaseObj.getReleaseDate(), '6MONTHS')
#             #
#             if previousRelease is not None:
#                 performPredictionRunner(projectName, previousRelease.getChanges(), testReleaseObj.getChanges(),
#                                         previousRelease.getReleaseDate(), testReleaseObj.getReleaseDate(),
#                                     'RECENT_RELEASE')
#
#
#
#             previousRelease = testReleaseObj











def getCrossCommits(cbug, cclean):
    cross_PROJECT = 'libsass'

    projectChanges = release_manager.getProject(cross_PROJECT).getAllChanges()

    trainingRegion = projectChanges.head(150).copy(deep=True)
    buggyChangesDF = trainingRegion[trainingRegion['Buggy'] == True]
    nonBuggyChangesDF = trainingRegion[trainingRegion['Buggy'] == False]

    if cbug > 0:
        RESAMPLE_COMMITS = buggyChangesDF.sample(cbug).copy(deep=True)

    if cclean > 0:
        if cbug >0:
            RESAMPLE_COMMITS = RESAMPLE_COMMITS.append(nonBuggyChangesDF.sample(cclean).copy(deep=True)).copy(deep=True)
        else:
            RESAMPLE_COMMITS = nonBuggyChangesDF.sample(cclean).copy(deep=True).copy(deep=True)


    # RESAMPLE_COMMITS = nonBuggyChangesDF.sample(25).copy(deep=True).copy(deep=True)


    return RESAMPLE_COMMITS


def intelligent_select(df):

    changesDF = df.copy(deep=True)
    metrics = ['la', 'lt'] #, 'entropy', 'exp']


    commits = None
    for feature in metrics :

        changesDF = changesDF.sort_values(by=feature, ascending=False)

        # medval = changesDF[feature].median()
        # # print('\t >> ',feature, changesDF[feature].min(), changesDF[feature].median(), changesDF[feature].max())

        midway = int(len(changesDF)/2)

        if commits is None:
            commits = changesDF.head(4) # prev 4
            commits = commits.append(changesDF.tail(4)) # prev 4
            commits = commits.append(changesDF[ midway - 2 : midway + 3  ]) # prev , 2 and 3
            # print('len commits = ', len(commits))
        else:
            commits = commits.append(changesDF.head(4))
            commits = commits.append(changesDF.tail(4))
            commits = commits.append(changesDF[midway - 2: midway + 2]) # prev , 2 and 2
            # print('len commits = ', len(commits))

    return commits


def variance_select(changesDF, feat_type):

    copyChanges = changesDF.copy(deep=True)

    if feat_type == 'size':
        for column in copyChanges.columns:
            if column not in ['la', 'lt', 'Buggy']:
                copyChanges = copyChanges.drop(labels=[column], axis = 1)

    # # print(feat_type, copyChanges.columns)

    copyChanges = normalize_log(copyChanges)

    copyChanges["variance"] = copyChanges.sum(axis=1)

    copyChanges = copyChanges.sort_values(by='variance', ascending=False)

    midway = int(len(copyChanges) / 2)

    commits = copyChanges.tail(7)
    commits = commits.append(copyChanges[midway - 5: midway + 5])
    commits = commits.append(copyChanges.tail(8))

    # # print(commits['variance'].values.tolist())

    commits = commits.drop(labels=['variance'], axis=1)

    #  print('commits = ', len(
    #     commits
    # ))

    return commits


def orchestrateCommits(buggyChangesDF, nonBuggyChangesDF, type, feat_type):

    if type == 'intelligent_buggy':
        return  intelligent_select(buggyChangesDF).append(
            nonBuggyChangesDF.sample(25).copy(deep=True)).copy(deep=True)
    elif type == 'intelligent_clean':
        return buggyChangesDF.sample(25).copy(deep=True).append(intelligent_select(nonBuggyChangesDF)).copy(deep=True)
    elif type == 'variance':
        return variance_select(buggyChangesDF, feat_type).append(variance_select(nonBuggyChangesDF, feat_type)).copy(deep=True)
    elif type == 'random':
        sample_size = min(len(buggyChangesDF), len(nonBuggyChangesDF))
        return buggyChangesDF.sample(sample_size).copy(deep=True).append(
            nonBuggyChangesDF.sample(sample_size).copy(deep=True)).copy(deep=True)
    elif type == 'head':
        return buggyChangesDF.head(25).copy(deep=True).append(
            nonBuggyChangesDF.head(25).copy(deep=True)).copy(deep=True)
    elif type =='tail':
        return buggyChangesDF.tail(25).copy(deep=True).append(
            nonBuggyChangesDF.tail(25).copy(deep=True)).copy(deep=True)
    elif type == 'middle':

        midway = int(len(buggyChangesDF) / 2)
        buggyCommits = buggyChangesDF[midway - 12: midway + 13].copy(deep=True)

        midway = int(len(nonBuggyChangesDF) / 2)
        cleanCommits = buggyCommits.append(nonBuggyChangesDF[midway - 12: midway + 13].copy(deep=True))

        return cleanCommits
    else:
        float('unknown')


def balancedSample(trainingRegion, bugs):

    buggyChangesDF = trainingRegion[trainingRegion['Buggy'] == True]
    nonBuggyChangesDF = trainingRegion[trainingRegion['Buggy'] == False]

    return buggyChangesDF.sample(bugs).copy(deep=True).append(
        nonBuggyChangesDF.sample(bugs).copy(deep=True)).copy(deep=True)



def performEarly(projectChanges, crossAllChanges, max_commits, projectName, testReleaseObj, cross):

    if cross:

        trainingRegion = crossAllChanges.head(max_commits).copy(deep=True)

        buggyChangesDF = trainingRegion[trainingRegion['Buggy'] == True]
        nonBuggyChangesDF = trainingRegion[trainingRegion['Buggy'] == False]

        if True: # len(buggyChangesDF) >= 25 and len(nonBuggyChangesDF) >= 25:

            for t in [ [None, 'random'], [ '$la$lt$',  'random'] ]: # [ '$la$lt$',  'head']

                type = t[0]
                orch_type = t[1]

                CROSS_COMMITS = orchestrateCommits(buggyChangesDF, nonBuggyChangesDF, orch_type, type)

                performPredictionRunner(projectName, CROSS_COMMITS, testReleaseObj.getChanges(),
                                        str(len(CROSS_COMMITS)), testReleaseObj.getReleaseDate(),
                                        '150_25_25_BELLWETHER_' + str(orch_type) + '_' + str(type).replace('$', '_'), type)
    else:

        trainingRegion = projectChanges.head(max_commits).copy(deep=True)

        if trainingRegion is not None and len(trainingRegion) > 0:

            buggyChangesDF = trainingRegion[ trainingRegion['Buggy'] == True ]
            nonBuggyChangesDF = trainingRegion[ trainingRegion['Buggy'] == False]

            if True: # len(buggyChangesDF) >= 25 and len(nonBuggyChangesDF) >= 25:

                for t in [ [None, 'random'] ] : #,  [ '$la$lt$', 'random'] ]: # [ '$la$lt$', 'head']

                    type = t[0]
                    orch_type = t[1]

                    EARLY_COMMITS = orchestrateCommits(buggyChangesDF, nonBuggyChangesDF, orch_type, type)


                    performPredictionRunner(projectName, EARLY_COMMITS, testReleaseObj.getChanges(),
                                            str(len(EARLY_COMMITS)), testReleaseObj.getReleaseDate(),
                                            '150_25_25_'+str(orch_type)+'_'+str(type).replace('$', '_'), testReleaseObj.getStartDate(), type)
            # else:
            #
            #     print('>> ', projectName, len(buggyChangesDF), len(nonBuggyChangesDF))





def performBellwether(projectName, crossAllChanges, testReleaseObj):
    performPredictionRunner(projectName, crossAllChanges, testReleaseObj.getChanges(),
                            str(len(crossAllChanges)), testReleaseObj.getReleaseDate(),
                            'BELLWETHER')


BELLWETHER_PROJECT =  'predis' # predis

def run_performance(projectName):

    print("Cores  = ", os.cpu_count())

    if projectName == 'homebrew-cask':

        project_obj = getProject(projectName)

        releaseList = project_obj.getReleases()

        testReleaseList = getReleasesAfter(E_TRAIN_LIMIT, releaseList)

        early_changes = early_sample(project_obj.getAllChanges().head(150))

        for testReleaseObj in testReleaseList:

            all_changes = getFirstChangesBetween(project_obj, 0, testReleaseObj.getStartDate())

            performPredictionRunner(projectName, all_changes.copy(deep=True), testReleaseObj.getChanges(),
                                    str(len(all_changes)), testReleaseObj.getReleaseDate(), 'ALL')

            performPredictionRunner(projectName, early_changes.copy(deep=True), testReleaseObj.getChanges(),
                                    str(len(early_changes)), testReleaseObj.getReleaseDate(), 'E')
    else:
        print("skipping ", projectName)


def print_time():

    all_time = 0.0
    early_time = 0.0

    for p in getProjectNames():


            all_df = pd.read_csv('./results_TTD_ALL_PAIRS/project_'+p+'_results.csv')
            all_df = all_df [ all_df['classifier'] == 'LogisticRegression']

            early_df = pd.read_csv('./results_TTD_ALL_PAIRS_Early/project_' + p + '_results.csv')
            early_df = early_df[early_df['classifier'] == 'LogisticRegression']

            all_time += sum(all_df['total_time'].values.tolist())

            early_time += sum(early_df['test_time'].values.tolist())
            early_time += sum(early_df['train_time'].values.tolist())

            # print(all_time, early_time)



    # print("all time = ", all_time)
    # print("early time = ", early_time)


def bell_project_index(bell_project):

    return bell_project

    # bell_index = 0
    #
    # for b in BELLWETHER_PROJECTS:
    #     bell_index += 1
    #
    #     if bell_project == b:
    #         return bell_index
    #
    # return None


def run_train_time():

    bellproject = getProject('homebrew-cask').getAllChanges()
    targetProject = getProject('camel')

    for testReleaseObj in targetProject.getReleases():

        if len(testReleaseObj.getChanges()) > 25:
            testChanges = testReleaseObj.getChanges().copy(deep=True)
            break

    t1 = []
    t2 = []
    t3 = []
    amtt = []

    for amt in [50, 100, 200, 400, 800, 1600, 3200, 6400]:

        amtt.append(amt)

        test_changes = testChanges.copy(deep=True)
        train_changes = bellproject.head(amt).copy(deep=True)

        current_time_ms = goodtime.time()
        performPredictionRunner('camel', train_changes.copy(deep=True), test_changes.copy(deep=True),
                                str(len(train_changes)),
                                testReleaseObj.getReleaseDate(), 'T_' + str(bell_project_index('homebrew-cask')))
        t1.append(goodtime.time() - current_time_ms)

        current_time_ms = goodtime.time()

        performPredictionRunner('camel', train_changes.copy(deep=True), testReleaseObj.getChanges(),
                                str(len(train_changes)), testReleaseObj.getReleaseDate(), 'B')
        t2.append(goodtime.time() - current_time_ms)
        current_time_ms = goodtime.time()

        train_changes = early_sample(train_changes.head(150))
        performPredictionRunner('camel', train_changes.copy(deep=True), test_changes.copy(deep=True),
                                str(len(train_changes)),
                                testReleaseObj.getReleaseDate(), 'E_B' + str(bell_project_index('homebrew-cask')))

        t3.append(goodtime.time() - current_time_ms)
        # print(amtt, t1, t2, t3)

    # print(amtt, t1, t2, t3)

    df = pd.DataFrame()

    df['amt'] = amtt
    df['t1']= t1
    df['t2'] = t2
    df['t3'] = t3

    df.to_csv('runtime.csv', index=False)



def run_tca_multiple_bell(projectName):

    for bell_project in BELLWETHER_PROJECTS:
        run_tca(projectName, bell_project)

def run_tca(projectName, bell_project):

    project_obj = getProject(projectName)
    releaseList = project_obj.getReleases()
    testReleaseList = getReleasesAfter(E_TRAIN_LIMIT, releaseList)

    bell_project_changes = getProject(bell_project).getAllChanges()

    if len(bell_project_changes) <= 300:
        print(bell_project, ' has less than 300 changes')
        return

    more_train_changes = tca_sample(bell_project_changes)
    early_trainChanges = early_sample(bell_project_changes.copy(deep=True).head(150))


    for testReleaseObj in testReleaseList:

        try:

            testChanges = testReleaseObj.getChanges().copy(deep=True)

            performPredictionRunner(projectName, more_train_changes.copy(deep=True), testChanges.copy(deep=True),
                                    str(len(more_train_changes)),
                                    testReleaseObj.getReleaseDate(), 'T_' + str(bell_project_index(bell_project)))


            performPredictionRunner(projectName, early_trainChanges.copy(deep=True), testChanges.copy(deep=True), str(len(early_trainChanges)),
                                    testReleaseObj.getReleaseDate(), 'E_F2_T2_' + str(bell_project_index(bell_project)),
                                    None, 'size')

            performPredictionRunner(projectName, early_trainChanges.copy(deep=True), testChanges.copy(deep=True), str(len(early_trainChanges)),
                                    testReleaseObj.getReleaseDate(), 'E_T2_' + str(bell_project_index(bell_project)))

        except Exception as e:
            print('TCAPLUS EXCEPTION : ', e)

        # if not path.exists('./'+TCA_DATA_FOLDER+'/train_'+BELLWETHER_PROJECT+'_'+projectName+'_'+str(testReleaseObj.getReleaseDate())+'.csv')  or not path.exists('./'+TCA_DATA_FOLDER+'/test_'+BELLWETHER_PROJECT+'_'+projectName+'_'+str(testReleaseObj.getReleaseDate())+'.csv'):
        #     print("\t Path does not exist")
        #     return False
        #
        # trainChanges = pd.read_csv('./'+TCA_DATA_FOLDER+'/train_'+BELLWETHER_PROJECT+'_'+projectName+'_'+str(testReleaseObj.getReleaseDate())+'.csv')
        # testChanges = pd.read_csv('./'+TCA_DATA_FOLDER+'/test_'+BELLWETHER_PROJECT+'_'+projectName+'_'+str(testReleaseObj.getReleaseDate())+'.csv')






# class DE_Learners(object):
#
#     def __init__(self, clf, train_X, train_Y, test_X, test_Y, goal):
#         """
#         :param clf: classifier, SVM, etc...
#         :param train_X: training data, independent variables.
#         :param train_Y: training labels, dependent variables.
#         :param predict_X: testing data, independent variables.
#         :param predict_Y: testingd labels, dependent variables.
#         :param goal: the objective of your tuning, F, recision,....
#         """
#         self.train_X = train_X
#         self.train_Y = train_Y
#
#         self.test_X = test_X
#         self.test_Y = test_Y
#
#         self.goal = goal
#         self.param_distribution = self.get_param()
#         self.learner = clf
#         self.confusion = None
#         self.params = None
#
#
#     def learn(self, F, **kwargs):
#         """
#         :param F: a dict, holds all scores, can be used during debugging
#         :param kwargs: a dict, all the parameters need to set after tuning.
#         :return: F, scores.
#         """
#         self.scores = { self.goal: [0.0]}
#         try:
#
#             # print('\t >> ', len(self.train_X), len(self.train_Y), len(self.test_X), len(self.test_Y))
#             self.learner.set_params(**kwargs)
#             # predict_result = []
#             # _df = pd.concat([self.train_X, self.train_Y], axis=1)
#             # _df = self.apply_smote(_df,neighbours,r)
#             # y_train = _df.Buggy
#             # X_train = _df.drop(labels=['Buggy'], axis=1)
#             #
#             # print(getBugCount(y_train), getBugCount(test_Y))
#
#
#             # print("df = ", getBugCount(_df), getNonBugCount(_df), len(_df))
#
#             clf = self.learner.fit(self.train_X, self.train_Y)
#
#             # clf = self.learner.fit(self.train_X, self.train_Y)
#             predict_result = clf.predict(self.test_X)
#
#             # print('\t Trying with ', clf.get_params(), len(self.test_X))
#
#             self.abcd = metrices.measures(self.test_Y, predict_result, pd.Series([0] * self.test_Y.shape[0]))
#
#             self.scores = self._Abcd(self.abcd, F)
#             self.confusion = metrics.classification_report(self.test_Y.values.tolist(), predict_result, digits=2)
#             self.params = kwargs
#
#         except Exception as e:
#             print('\n\n \t exception >>>> ', str(e), getSimpleName(self.learner), self.params )
#             # print('locl',e)
#             # print("local exception!")
#             F = {}
#             F[DE_GOAL] = [0]
#             return F
#
#
#         # print('returing ',self.scores)
#         return self.scores
#
#     def _Abcd(self, abcd, F):
#         """
#         :param predicted: predicted results(labels)
#         :param actual: actual results(labels)
#         :param F: previously got scores
#         :return: updated scores.
#         """
#         if 'recall' in self.goal:
#             F['recall'] = [abcd.calculate_recall()]
#             return F
#         elif 'g-score' in self.goal:
#             F['g-score'] = [abcd.get_g_score()]
#             return F
#         elif 'brier' in self.goal:
#             F['brier'] = [abcd.brier()]
#             return F
#         elif 'precision' in self.goal:
#             F['precision'] = [abcd.calculate_precision()]
#             return F
#         elif 'f1' in self.goal:
#             F['f1'] = [ abcd.calculate_f1_score() ]
#             return F
#         elif 'd2h' in self.goal:
#             F['d2h'] = [abcd.calculate_d2h()]
#             return F
#         elif 'roc_auc' in self.goal:
#             F['roc_auc'] = [abcd.get_roc_auc_score()]
#             return F
#         elif 'd2h' in self.goal:
#             F['d2h'] = [abcd.calculate_d2h()]
#             return F
#         elif 'pci_20' in self.goal:
#             F['pci_20'] = [abcd.get_pci_20()]
#         elif 'popt_20' in self.goal:
#             F['popt_20'] = [abcd.get_popt_20()]
#
#     # def predict(self, test_X):
#     #     return self.learner.predict(test_X)









# def fft_scorer(y_true, y_pred, *, normalize=True, sample_weight=None):
#
#     tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
#
#     far = 0
#     recall = 0
#
#     if (fp + tn) != 0:
#         far = fp/(fp + tn)
#     if (tp + fn) != 0:
#         recall = tp/(tp + fn)
#
#     dist2heaven = math.sqrt((1 - recall) ** 2 + far ** 2)
#
#     dist2heaven = dist2heaven/math.sqrt(2)
#
#     return dist2heaven * -1

# def compareEarlyWithCrossTreatments(projectName):
#
#     print("Received ",projectName)
#
#     max_commits = 150
#     projectObj = release_manager.getProject(projectName)
#     projectChanges = projectObj.getAllChanges()
#
#     releaseList = projectObj.getReleases()
#     testReleaseList = getReleasesAfter(max_commits, releaseList)
#
#     crossProjObj = release_manager.getProject(BELLWETHER_PROJECT)
#     crossAllChanges = crossProjObj.getAllChanges()
#     crossBuggiestRelease =  getBuggyPastRelease(crossProjObj.getReleases())
#
#
#     recentRelease = None
#
#     for testReleaseObj in testReleaseList:
#
#         # if recentRelease is None and valid(testReleaseObj.getChanges()):
#         #     recentRelease = testReleaseObj
#         #     continue
#         #
#         #
#         # if (valid(testReleaseObj.getChanges())):
#         #     movingTuning(recentRelease, testReleaseObj, projectChanges, projectName)
#         #     recentRelease = testReleaseObj
#
#
#
#         # performTCAplus(projectName, testReleaseObj)
#         performBellwether(projectName, crossAllChanges, testReleaseObj)
#
#         # allChangesDF = getLastXChangesEndDate(0, testReleaseObj.getStartDate(), projectObj)
#         # performPredictionRunner(projectName, allChangesDF, testReleaseObj.getChanges(), str(len(allChangesDF)),
#         #                         testReleaseObj.getReleaseDate(), 'ALL')
#
#         performEarly(projectChanges, None, max_commits, projectName, testReleaseObj, False)
#         performEarly(None, crossAllChanges, max_commits, projectName, testReleaseObj, True)
#
#
#         # performCrossBuggiest(projectName, crossBuggiestRelease, testReleaseObj)


def validTrainRegion(trainingRegion):

    return trainingRegion is not None and len(trainingRegion) > 2



def runTreatmentsForTSE(projectName):

    # print('runTreatmentsForTSE')

    projectObj = release_manager.getProject(projectName)
    releaseList = projectObj.getReleases()
    projectStart = min([r.getStartDate() for r in releaseList])

    testReleaseList = getReleasesAfter(E_TRAIN_LIMIT, releaseList)
    # print("Fetching releases after > ", E_TRAIN_LIMIT)

    for testReleaseObj in testReleaseList:

        if True:

            train_changes_df = getFirstChangesBetween(projectObj, projectStart, testReleaseObj.getStartDate())

            if validTrainRegion(train_changes_df):

                """
                Train on ALL
                """
                # performPredictionRunner(projectName, train_changes_df.copy(deep=True), testReleaseObj.getChanges(),
                #                         str(len(train_changes_df)),
                #                         testReleaseObj.getReleaseDate(), 'ALL')

                performEarly(projectObj.getAllChanges(), None, MAX_EARLY_COMMITS, projectName, testReleaseObj, False)


# BELLWETHER_PROJECTS = [ 'onadata' , 'restheart', 'scikit-learn', 'pry' ]

# BELLWETHER_PROJECTS  = ['apptentive-android', 'restheart', 'active_merchant', 'pry', 'scikit-learn']

BELLWETHER_PROJECTS  = ['scikit-learn']

# onadata - unpopular  restheart unpopular



def run_multiple_bellwether(projectName, type=None):

    for bellwether_project in BELLWETHER_PROJECTS:

        if projectName == bellwether_project:
            continue

        project_obj = release_manager.getProject(projectName)
        releaseList = project_obj.getReleases()
        testReleaseList = getReleasesAfter(E_TRAIN_LIMIT, releaseList)
        bell_changes = getProject(bellwether_project).getAllChanges()
        early_changes = early_sample(bell_changes.head(150))

        for testReleaseObj in testReleaseList:

            try:

                performPredictionRunner(projectName, bell_changes.copy(deep=True), testReleaseObj.getChanges(),
                                        str(len(bell_changes)), testReleaseObj.getReleaseDate(),
                                        'B_' + str(bell_project_index(bellwether_project)))

                performPredictionRunner(projectName, early_changes.copy(deep=True), testReleaseObj.getChanges(),
                                        str(len(early_changes)), testReleaseObj.getReleaseDate(),
                                        'E_B_' + str(bell_project_index(bellwether_project)) + '_2', None, 'size')

                # if type is None:
                #     performPredictionRunner(projectName, bell_changes.copy(deep=True), testReleaseObj.getChanges(),
                #                         str(len(bell_changes)), testReleaseObj.getReleaseDate(), 'B_'+str(bell_project_index(bellwether_project)), None, type)
                #     # performPredictionRunner(projectName, early_changes.copy(deep=True), testReleaseObj.getChanges(),
                #     #                         str(len(early_changes)), testReleaseObj.getReleaseDate(),
                #     #                         'E_B_' +str(bell_project_index(bellwether_project)), None, type)
                # else:
                #     # performPredictionRunner(projectName, bell_changes.copy(deep=True), testReleaseObj.getChanges(),
                #     #                         str(len(bell_changes)), testReleaseObj.getReleaseDate(),
                #     #                         'B_' + str(bell_project_index(bellwether_project))+'_2', None, type)
                #     performPredictionRunner(projectName, early_changes.copy(deep=True), testReleaseObj.getChanges(),
                #                             str(len(early_changes)), testReleaseObj.getReleaseDate(),
                #                             'E_B_' + str(bell_project_index(bellwether_project))+'_2', None, type)
            except Exception as e:
                print('Exception @ run_multiple_bellwether ', projectName, bellwether_project)


def run_bellwether(projectName):

    if projectName == BELLWETHER_PROJECT:
        return

    project_obj = release_manager.getProject(projectName)
    releaseList = project_obj.getReleases()
    testReleaseList = getReleasesAfter(E_TRAIN_LIMIT, releaseList)
    bell_changes = getProject(BELLWETHER_PROJECT).getAllChanges()

    for testReleaseObj in testReleaseList:

        performPredictionRunner(projectName, bell_changes.copy(deep=True), testReleaseObj.getChanges(),
                                str(len(bell_changes)), testReleaseObj.getReleaseDate(), 'B')





def run_random(projectName):

    project_obj = release_manager.getProject(projectName)
    releaseList = project_obj.getReleases()
    testReleaseList = getReleasesAfter(E_TRAIN_LIMIT, releaseList)

    for testReleaseObj in testReleaseList:

        random_project = getRandomProject(projectName)

        changes_random = getProject(random_project).getAllChanges()

        performPredictionRunner(projectName, changes_random.copy(deep=True), testReleaseObj.getChanges(),
                                random_project+'_'+str(len(changes_random)), testReleaseObj.getReleaseDate(), 'R')

        early_random_changes = early_sample(changes_random.head(150))

        performPredictionRunner(projectName, early_random_changes.copy(deep=True), testReleaseObj.getChanges(),
                                random_project+'_'+str(len(early_random_changes)), testReleaseObj.getReleaseDate(), 'E_R')



def run_early_bellwether(projectName):

    if projectName == BELLWETHER_PROJECT:
        return

    project_obj = release_manager.getProject(projectName)
    releaseList = project_obj.getReleases()
    testReleaseList = getReleasesAfter(E_TRAIN_LIMIT, releaseList)

    early_bell_changes = early_sample(getProject(BELLWETHER_PROJECT).getAllChanges().head(150))

    for testReleaseObj in testReleaseList:
        performPredictionRunner(projectName, early_bell_changes.copy(deep=True), testReleaseObj.getChanges(),
                                str(len(early_bell_changes)), testReleaseObj.getReleaseDate(), 'E_B')



def run_rq1(projectName):

    # run_early(projectName)
    run_multiple_bellwether(projectName)
    # run_all(projectName)
    run_tca_multiple_bell(projectName)


def run_rq2(projectName):

    # run_early(projectName)
    # run_early_2(projectName)
    # run_multiple_bellwether(projectName)
    run_multiple_bellwether(projectName, 'size')


def run_table_1(projectName):

    run_multiple_bellwether(projectName)


def run_table_2(projectName):

    run_tca_multiple_bell(projectName)


def run_table_3_4(projectName):

    run_multiple_bellwether(projectName, 'size')


def test_all_releases(projectName):



    if not resultExists(projectName):

        writeRow(getFileName(projectName), getHeader())

        try:

            if RQ == 1:
                run_early(projectName)
                run_all(projectName)
            elif RQ == 2:
                run_early(projectName)
                run_early_2(projectName)
                run_no_training_data(projectName)
            elif RQ == 3:
                run_early_2(projectName)
                run_all(projectName)
                run_bellwether(projectName)
                run_early_bellwether_2(projectName)
            elif RQ == 4:
                run_early(projectName)
                run_early_2(projectName)
                run_bellwether(projectName)
                run_early_bellwether_2(projectName)
                run_tca(projectName, BELLWETHER_PROJECT)
            elif RQ == 5:
                run_all(projectName)
                # run_early(projectName)
                run_early_2(projectName)

            else:
                float("INVALID RQ")


            # rq1

            # 'E_B_2', 'E_2', 'ALL', 'B', 'NO_TRAINING_DATA'

            # run_early(projectName)
            # run_early_2(projectName)




            # table 2

            # run_all(projectName)
            # run_early_2(projectName)
            # run_bellwether(projectName)
            # run_early_bellwether_2(projectName)
            # run_no_training_data(projectName)



            # run_early_revise(projectName, 3)
            # run_early_revise(projectName, 9)
            # run_early_revise(projectName, 27)
            # run_early_revise(projectName, 111)

            #

            # run_early_1(projectName)

            #
            #
            #

            # run_early_bellwether(projectName)
            #



            # run_all_pairs(projectName)

            # run_tse(projectName)


            # run_tca(projectName, BELLWETHER_PROJECTS[2])
            # run_performance(projectName)
            # run_random(projectName)
            # run_rq1(projectName)
            # run_rq2(projectName)f
            # run_table_1(projectName)
            # run_table_2(projectName)
            # run_table_3_4(projectName)
            # run_tca_multiple_bell(projectName)
            # run_multiple_bellwether(projectName)


        except Exception as e:
            print("Error processing ", projectName, ' for RQ = ', RQ, e)
            traceback.print_exc()





def resultExists(p):

    # if True:
    #     return False

    filePath = './' + RESULTS_FOLDER + '/project_' + p + "_results.csv"

    if 'Windows' in platform.system() and os.path.exists(filePath):
        os.remove(filePath)
        print("removed ", filePath)


    return path.exists(filePath)
    # return False



LOCAL_PROJECT = 'numpy'


def run_all(projectName):

    if projectName == 'qt_tse':
        local_train_limit = 1211
    elif projectName == 'openstack_tse':
        local_train_limit = 8353
    else:
        local_train_limit = E_TRAIN_LIMIT

    project_obj = getProject(projectName)



    releaseList = project_obj.getReleases()


    testReleaseList = getReleasesAfter(local_train_limit, releaseList)



    print("releases == ", len(testReleaseList), len(project_obj.getAllChanges()))

    for testReleaseObj in testReleaseList:

        # print(testReleaseObj.getReleaseDate())

        if projectName in TSE_SZZ_PROJECTS:

            train_changes = getFirstChangesBetween(project_obj, testReleaseObj.getStartDate() - 6 * one_month, testReleaseObj.getStartDate())

            performPredictionRunner(projectName, train_changes, testReleaseObj.getChanges(),
                                    str(len(train_changes)),  testReleaseObj.getReleaseDate(), 'ALL')
        else:
            train_changes = getFirstChangesBetween(project_obj, 0, testReleaseObj.getStartDate())

            performPredictionRunner(projectName, train_changes, testReleaseObj.getChanges(),
                                    str(len(train_changes)), testReleaseObj.getReleaseDate(), 'ALL')


def run_early(projectName):

    project_obj = release_manager.getProject(projectName)
    releaseList = project_obj.getReleases()

    if projectName == 'qt_tse':
        local_train_limit = 1211
    elif projectName == 'openstack_tse':
        local_train_limit = 8353
    else:
        local_train_limit = E_TRAIN_LIMIT

    testReleaseList = getReleasesAfter(local_train_limit, releaseList)
    train_changes = project_obj.getAllChanges().head(local_train_limit)
    early_changes = early_sample(train_changes)

    print("early = ", len(early_changes), len(train_changes) , len(testReleaseList))

    for testReleaseObj in testReleaseList:

        performPredictionRunner(projectName, early_changes.copy(deep=True), testReleaseObj.getChanges(),
                                str(len(train_changes)), testReleaseObj.getReleaseDate(), 'E')




def run_early_revise(projectName, months):

    project_obj = release_manager.getProject(projectName)
    releaseList = project_obj.getReleases()
    testReleaseList = getReleasesAfter(E_TRAIN_LIMIT, releaseList)



    rev_count = 0
    time_passed  = None

    train_changes = project_obj.getAllChanges().head(150)
    early_changes = train_changes

    for testReleaseObj in testReleaseList:
        # projectName, originalTrainChanges, originalTestChanges, trainReleaseDate, testReleaseDate, trainApproach,
        # testReleaseStartDate = None, retainFew = None, tuneChanges = None, returnResults = False

        # print("\t sending ", len(early_changes))
        # performPredictionRunner(projectName, early_changes.copy(deep=True), testReleaseObj.getChanges(),
        #                         str(rev_count), testReleaseObj.getReleaseDate(), 'E_R_1_Y', None, 'size')

        performPredictionRunner(projectName, early_changes.copy(deep=True), testReleaseObj.getChanges(),
                                str(len(train_changes)), testReleaseObj.getReleaseDate(), 'E_R_'+str(months))

        if time_passed is None:
            time_passed = testReleaseObj.getReleaseDate()
        elif testReleaseObj.getReleaseDate() - time_passed > one_month * months:
            print("revising ", rev_count)
            rev_count += 1
            time_passed = testReleaseObj.getReleaseDate()
            early_changes = getFirstChangesBetween(project_obj, 0, testReleaseObj.getStartDate()).copy(deep=True)


def run_no_training_data(projectName):
    project_obj = release_manager.getProject(projectName)
    releaseList = project_obj.getReleases()
    testReleaseList = getReleasesAfter(E_TRAIN_LIMIT, releaseList)
    train_changes = project_obj.getAllChanges().head(150)
    early_changes = train_changes
    # early_changes = early_sample(train_changes)

    for testReleaseObj in testReleaseList:
        # projectName, originalTrainChanges, originalTestChanges, trainReleaseDate, testReleaseDate, trainApproach,
        # testReleaseStartDate = None, retainFew = None, tuneChanges = None, returnResults = False

        performPredictionRunner(projectName, None, testReleaseObj.getChanges(),
                                '0', testReleaseObj.getReleaseDate(), 'NO_TRAINING_DATA', None)



def run_early_1(projectName):

    project_obj = release_manager.getProject(projectName)
    releaseList = project_obj.getReleases()
    testReleaseList = getReleasesAfter(E_TRAIN_LIMIT, releaseList)
    train_changes = project_obj.getAllChanges().head(150)
    early_changes = train_changes
    # early_changes = early_sample(train_changes)

    for testReleaseObj in testReleaseList:
        # projectName, originalTrainChanges, originalTestChanges, trainReleaseDate, testReleaseDate, trainApproach,
        # testReleaseStartDate = None, retainFew = None, tuneChanges = None, returnResults = False

        performPredictionRunner(projectName, early_changes.copy(deep=True), testReleaseObj.getChanges(),
                                str(len(train_changes)), testReleaseObj.getReleaseDate(), 'E_1', None, 'la')

def run_early_2(projectName):

    if projectName == 'qt_tse':
        local_train_limit = 1211
    elif projectName == 'openstack_tse':
        local_train_limit = 8353
    else:
        local_train_limit = E_TRAIN_LIMIT


    project_obj = release_manager.getProject(projectName)
    releaseList = project_obj.getReleases()
    testReleaseList = getReleasesAfter(local_train_limit, releaseList)
    train_changes = project_obj.getAllChanges().head(local_train_limit)
    early_changes = train_changes
    # early_changes = early_sample(train_changes)

    for testReleaseObj in testReleaseList:
        # projectName, originalTrainChanges, originalTestChanges, trainReleaseDate, testReleaseDate, trainApproach,
        # testReleaseStartDate = None, retainFew = None, tuneChanges = None, returnResults = False

        performPredictionRunner(projectName, early_changes.copy(deep=True), testReleaseObj.getChanges(),
                                str(len(train_changes)), testReleaseObj.getReleaseDate(), 'E_2', None, 'size')

def run_early_bellwether_2(projectName):

    if projectName == BELLWETHER_PROJECT:
        return

    project_obj = release_manager.getProject(projectName)
    releaseList = project_obj.getReleases()
    testReleaseList = getReleasesAfter(E_TRAIN_LIMIT, releaseList)

    early_bell_changes = early_sample(getProject(BELLWETHER_PROJECT).getAllChanges().head(150))

    for testReleaseObj in testReleaseList:
        performPredictionRunner(projectName, early_bell_changes.copy(deep=True), testReleaseObj.getChanges(),
                                str(len(early_bell_changes)), testReleaseObj.getReleaseDate(), 'E_B_2' , None, 'size')





def run():

    print("Writing in ", getFileName('<p>'), ' change in two places')

    procs = []

    for name in getProjectNames():
        proc = Process(target=test_all_releases, args=(name,))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()


def printLocal():


    scores = []

    for p in [LOCAL_PROJECT] : #release_manager.getProjectNames():

        df = pd.read_csv('./'+'results_ttd'+'/project_'+p+'_results.csv')
    #
        for c in [getSimpleName(c) for c in getClassifiers()]:

            clf_df = df[df['classifier'] == c ]

            for tune in [False]:

                for m in ['brier']:


                    try:
                        scores.append(str(clf_df[m].median())+'<->'+str(c)+'_'+str(m))
                    except Exception as e:
                        print(e)
                        continue
    #
    # scores.sort()
    #
    for s in scores:
        print(s)

# def compare():
#
#     tunedGreater = []
#     tunedSmaller = []
#     for p in  release_manager.getProjectNames():
#
#         df = pd.read_csv('./results_ttd/project_'+p+'_results.csv')
#
#         for c in getClassifiers():
#             tdf = df[df['classifier'] == getSimpleName(c, True)]
#             untuned_df = df[df['classifier'] == getSimpleName(c, False)]
#
#             try:
#                 if tdf['d2h'].median() < untuned_df['d2h'].median():
#                     tunedGreater.append(p)
#                 elif tdf['d2h'].median() > untuned_df['d2h'].median():
#                     tunedSmaller.append(p)
#             except Exception as e:
#                 print(e)
#                 continue
#
#
#     print('greater = ',  tunedGreater)
#     print('smaller = ',   tunedSmaller)

def augument_goal_score():

    df = pd.read_csv('zzgoal_compare.csv')

    commits = []
    releases = []
    bug_per = []
    for p in df['project'].values.tolist():

        p_obj = release_manager.getProject(p)

        numcommits = len(p_obj.getAllChanges())
        num_releases = len(p_obj.getReleases())
        per = int(getBugCount(p_obj.getAllChanges())/numcommits)

        commits.append(numcommits)
        releases.append(num_releases)
        bug_per.append(per)

    df['commits'] = commits
    df['releases'] = releases
    df['bug_per'] = bug_per
    df['diff'] = df['tuned_loss'] - df['untuned_loss']

    df.to_csv("zzzznew.csv", index=False)






# def report_goal_scores():
#
#     writeRow('aaa.csv', ['projectName', 'tuned', 'untuned', 'diff'])
#     procs = []
#
#     projectNames = release_manager.getProjectNames()
#
#
#
#     for name in projectNames:
#
#         proc = Process(target=_sub_report_goal_score, args=(name,))
#         procs.append(proc)
#         proc.start()
#
#     # complete the processes
#     for proc in procs:
#         proc.join()



DEBUG = False

def ttd():

    tuneable_projects = []
    diff = []

    for p in POPULAR_PROJECTS:
        df = pd.read_csv('./results/project_' + p + '_results.csv')
        df = df[(df['test_changes'] > 5)]

        try:
            tuned_df = df [ df['classifier'] == 'TUNED_LR'  ]
            ut_df = df[df['classifier'] == 'SVM']


            if ut_df['d2h'].median() - tuned_df['d2h'].median() > 0.05 :
                tuneable_projects.append(p)
                diff.append(abs(ut_df['d2h'].median() - tuned_df['d2h'].median()))


        except Exception as e:
            print(p, e)

    print(" Tuneable Projects ", tuneable_projects, len(tuneable_projects))
    print(" Diff Projects ", diff, len(diff), np.median(diff))


def is_ls():

    scores = []
    pure_scores = []

    ls_map = {}

    for p in release_manager.getProjectNames():

        try:
            pobj = release_manager.getProject(p)
            early_changes = early_sample(pobj.getAllChanges().copy(deep=True))
            early_processed = customPreProcess(early_changes, None, False)

            trainY = early_processed.Buggy
            trainX = early_processed.drop(labels=['Buggy'], axis=1)

            linear_svc = SVC(kernel='linear')
            linear_svc.fit(trainX, trainY)


            F = computeMeasures(early_processed, linear_svc, [], [1 for x in range(0, len(early_processed))])

            print(p, F['accuracy'])

            ls_map[p] = F['accuracy'][0]

            pure_scores.append(F['accuracy'])

            scores.append(str(F['accuracy'])+'-'+p)

        except Exception as e:
            print('\t',e)
            continue

    scores.sort()

    print(scores, 'median = ', np.median(pure_scores))

    print(ls_map)




'woboq_codebrowser', 'tatami', 'taiga-back', 'symfony', 'state_machine', 'snp-pipeline', 'snorocket', \
'ruby', 'intellij-pants-plugin', 'django-tastypie', 'django-payments', 'bitcoin', 'SFML'

x = {'Newtonsoft.Json': -1.0, 'metrics-plugins': -1.0, 'soot-infoflow-android': -1.0,
     'azure-mobile-services': -1.0, 'neovim': -0.9866666666666667, 'tigon': -0.9866666666666667,
     'parameter-framework': -0.9866666666666667, 'GooFit': -0.98, 'Hystrix': -0.9733333333333334,
     'jadx': -0.9733333333333334, 'pivotal_workstation': -0.9733333333333334,
     'lutece-core': -0.9733333333333334, 'closure-stylesheets': -0.9689922480620154,
     'apollo': -0.9666666666666667, 'google-api-php-client': -0.9666666666666667,
     'leakcanary': -0.9666666666666667, 'libsodium': -0.9666666666666667,
     'middleman': -0.9666666666666667, 'make-snake': -0.9666666666666667,
     'tori': -0.9666666666666667, 'azure-sdk-for-media-services': -0.9666666666666667,
     'origin-server': -0.9666666666666667, 'gradle': -0.96,
     'metrics': -0.96, 'SalesforceMobileSDK-Android': -0.96,
     'druid': -0.9533333333333334, 'qa-tools': -0.9533333333333334,
     'puppetlabs-powershell': -0.9533333333333334,
     'napa': -0.9533333333333334, 'imeji': -0.9533333333333334,
     'twitter': -0.9466666666666667, 'brackets-app': -0.9466666666666667, 'analytics-java': -0.9466666666666667, 'java-util': -0.9466666666666667, 'oasp4j': -0.9466666666666667, 'codyco-modules': -0.9466666666666667, 'server': -0.9466666666666667, 'coreclr': -0.94, 'seaborn': -0.94, 'patternlab-php': -0.9333333333333333, 'ros_ethercat': -0.9333333333333333, 'brotli': -0.9266666666666666, 'GSYVideoPlayer': -0.9266666666666666, 'ripe.atlas.sagan': -0.9266666666666666, 'deepin-movie': -0.9266666666666666, 'gazebo-yarp-plugins': -0.9266666666666666, 'recog': -0.9266666666666666, 'onadata': -0.9266666666666666, 'AnySoftKeyboard': -0.92, 'ionic': -0.92, 'kafka': -0.92, 'lob-ruby': -0.92, 'portico': -0.92, 'GloboDNS': -0.92, 'okhttp': -0.9133333333333333, 'powerline': -0.9133333333333333, 'rails_best_practices': -0.9133333333333333, 'ThinkUp': -0.9133333333333333, 'geoserver-manager': -0.9133333333333333, 'redpotion': -0.9133333333333333, 'Trooper': -0.9133333333333333, 'aminator': -0.9133333333333333, 'AsTeRICS': -0.9133333333333333, 'MatterControl': -0.9133333333333333, 'formtastic': -0.9066666666666666, 'PHP-CS-Fixer': -0.9066666666666666, 'apptentive-android': -0.9066666666666666, 'bitcoin': -0.9, 'SignalR': -0.9, 'B2BProfessional': -0.9, 'travis.rb': -0.9, 'compute-image-packages': -0.9, 'projecta': -0.9, 'yandex-tank': -0.9, 'stripe-java': -0.9, 'libwebp': -0.9, 'brackets': -0.8933333333333333, 'dompdf': -0.8933333333333333, 'picasso': -0.8933333333333333, 'redisson': -0.8933333333333333, 'ruby': -0.8933333333333333, 'springside4': -0.8933333333333333, 'mcsema': -0.8933333333333333, 'narayana': -0.8933333333333333, 'tatami': -0.8933333333333333, 'CodeIgniter': -0.8866666666666667, 'mosh': -0.8866666666666667, 'sunspot': -0.8866666666666667, 'mailjet-gem': -0.8866666666666667, 'Galicaster': -0.8866666666666667, 'canal': -0.88, 'go-ethereum': -0.88, 'requests': -0.88, 'pienoon': -0.88, 'GloboNetworkAPI-client-python': -0.88, 'KunstmaanGeneratorBundle': -0.88, 'grape': -0.8733333333333333, 'Imagine': -0.8733333333333333, 'jekyll': -0.8733333333333333, 'proxygen': -0.8733333333333333, 'raiden': -0.8733333333333333, 'sysdig': -0.8733333333333333, 'zipline': -0.8733333333333333, 'BigQuery-Python': -0.8733333333333333, 'sauce-java': -0.8733333333333333, 'PerfKitBenchmarker': -0.8733333333333333, 'compass': -0.8666666666666667, 'rhino': -0.8666666666666667, 'scikit-learn': -0.8666666666666667, 'state_machine': -0.8666666666666667, 'Twig': -0.8666666666666667, 'yii': -0.8666666666666667, 'libgdiplus': -0.8666666666666667, 'restheart': -0.8666666666666667, 'ActionBarSherlock': -0.86, 'brakeman': -0.86, 'dagger': -0.86, 'greenDAO': -0.86, 'logstash': -0.86, 'masscan': -0.86, 'memcached': -0.86, 'portia': -0.86, 'zulip': -0.86, 'radosgw-agent': -0.86, 'XBeeJavaLibrary': -0.86, 'mod_dup': -0.86, 'cocaine-plugins': -0.86, 'odemis': -0.86, 'Catch': -0.8533333333333334, 'jieba': -0.8533333333333334, 'Lean': -0.8533333333333334, 'newspaper': -0.8533333333333334, 'predis': -0.8533333333333334, 'swoole': -0.8533333333333334, 'opendht': -0.8533333333333334, 'picketlink': -0.8533333333333334, 'waterbutler': -0.8533333333333334, 'codis': -0.8466666666666667, 'jinja2': -0.8466666666666667, 'rubocop': -0.8466666666666667, 'shadowsocks-csharp': -0.8466666666666667, 'wagtail': -0.8466666666666667, 'iron_mq_java': -0.8466666666666667, 'django-payments': -0.8466666666666667, 'dogapi': -0.8466666666666667, 'sputnik': -0.8466666666666667, 'MDK': -0.8466666666666667, 'active_merchant': -0.84, 'androidannotations': -0.84, 'jsoup': -0.84, 'libgdx': -0.84, 'nanomsg': -0.84, 'Signal-Android': -0.84, 'WP-API': -0.84, 'wp-cli': -0.84, 'whiteboard': -0.84, 'nRF51-ble-bcast-mesh': -0.84, 'opencms-core': -0.84, 'capybara': -0.8333333333333334, 'elasticsearch': -0.8333333333333334, 'eventmachine': -0.8333333333333334, 'fluentd': -0.8333333333333334, 'homebrew-cask': -0.8333333333333334, 'macvim': -0.8333333333333334, 'mybatis-3': -0.8333333333333334, 'puphpet': -0.8333333333333334, 'pyston': -0.8333333333333334, 'thanos': -0.8333333333333334, 'tweepy': -0.8333333333333334, 'snorocket': -0.8333333333333334, 'Cachet': -0.8266666666666667, 'curl': -0.8266666666666667, 'disruptor': -0.8266666666666667, 'guava': -0.8266666666666667, 'junit': -0.8266666666666667, 'metasploit-framework': -0.8266666666666667, 'ServiceStack': -0.8266666666666667, 'facebook-android-sdk': -0.82, 'grav': -0.82, 'home-assistant': -0.82, 'macdown': -0.82, 'pry': -0.82, 'rasa': -0.82, 'vcr': -0.82, 'watchman': -0.82, 'agg-sharp': -0.82, 'android': -0.8133333333333334, 'diaspora': -0.8133333333333334, 'peewee': -0.8133333333333334, 'prawn': -0.8133333333333334, 'Slice': -0.8133333333333334, 'serengeti-ws': -0.8133333333333334, 'bitmask_client': -0.8133333333333334, 'libsass': -0.8066666666666666, 'realm-java': -0.8066666666666666, 'Validation': -0.8066666666666666, 'silverstripe-elemental': -0.8066666666666666, 'supplejack_api': -0.8066666666666666, 'hrpsys-base': -0.8066666666666666, 'ajenti': -0.8, 'backup': -0.8, 'django-rest-framework': -0.8, 'fresco': -0.8, 'gunicorn': -0.8, 'istio': -0.8, 'roboguice': -0.8, 'vert.x': -0.8, 'snp-pipeline': -0.8, 'cassandra': -0.7933333333333333, 'jq': -0.7933333333333333, 'pandas': -0.7933333333333333, 'pyspider': -0.7933333333333333, 'titan': -0.7933333333333333, 'intellij-pants-plugin': -0.7933333333333333, 'midpoint': -0.7933333333333333, 'devise': -0.7866666666666666, 'camel': -0.78, 'mezzanine': -0.78, 'passenger': -0.78, 'zf2': -0.78, 'olingo-odata2': -0.78, 'fat_free_crm': -0.7733333333333333, 'netty': -0.7733333333333333, 'redis': -0.7733333333333333, 'tiled': -0.7733333333333333, 'woboq_codebrowser': -0.7733333333333333, 'KDSoap': -0.7733333333333333, 'postgres': -0.7666666666666667, 'symfony': -0.7666666666666667, 'Ushahidi_Android': -0.7666666666666667, 'django-tastypie': -0.76, 'taiga-back': -0.76, 'glide': -0.7533333333333333, 'mechanize': -0.7533333333333333, 'phpunit': -0.7533333333333333, 'tesseract': -0.7533333333333333, 'azure-linux-extensions': -0.7466666666666667, 'piwik': -0.74, 'sympy': -0.74, 'indico': -0.74, 'assetic': -0.7333333333333333, 'beets': -0.7333333333333333, 'numpy': -0.7266666666666667, 'gevent': -0.72, 'pymel': -0.72, 'ipython': -0.7066666666666667, 'mopidy': -0.7066666666666667, 'SFML': -0.7066666666666667, 'deepin-boot-maker': -0.7, 'django': -0.66}

def compare():

    for c in [getSimpleName(c) for c in getClassifiers() ]:

        scores = []

        for p in  release_manager.getProjectNames():

            # pobject = release_manager.getProject(p)
            # print(p, len(pobject.getAllChanges()), len(pobject.getReleases()))

            try:
                df = pd.read_csv('./' + 'results' + '/project_' + p + '_results.csv')

                a = df[ df['classifier'] == 'LogisticRegression'][DE_GOAL].median()
                b = df[ df['classifier'] == c][DE_GOAL].median()



                if a - b > 0:
                    scores.append(a-b)


            except Exception as e:
                continue

        scores.sort(reverse=True)
        try:
            print(c, len(scores),  np.median(scores) )
        except Exception as e:
            continue



def test_TLELClassifier():

    tlel  = TLELClassifier()
    window = getProject('ActionBarSherlock').getAllChanges().head(300)
    tlel.fit(window.head(150))
    print( tlel.predict(window.tail(150)))

if __name__ == "__main__":

    print("Result csv's will be generated at : ",RESULTS_FOLDER, ' folder!')
    run()

