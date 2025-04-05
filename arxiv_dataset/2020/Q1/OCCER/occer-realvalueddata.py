import numpy as np
import arff

import argparse

import pandas as pd

import random

from math import sqrt
from sklearn.metrics import mean_squared_error

from statistics import mean
from statistics import stdev

from sklearn import linear_model
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score

from pyod.utils.utility import standardizer


from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.cof import COF



def rmse(predictions, targets):
    rms = sqrt(mean_squared_error(targets, predictions))
    return rms


def transformmodulus(rf_predictions):
    modfunc = lambda x: -x if x < 0 else x
    vmodfunc = np.vectorize(modfunc)
    modrf_predictions = vmodfunc(rf_predictions)
    return modrf_predictions



# https://stackoverflow.com/questions/39945410/numpy-equivalent-of-list-pop
def popcol(my_array,pc):
    """ column popping in numpy arrays
    Input: my_array: NumPy array, pc: column index to pop out
    Output: [new_array,popped_col] """
    i = pc
    pop = my_array[:,i]
    new_array = np.hstack((my_array[:,:i],my_array[:,i+1:]))
    return [new_array,pop]


def predict(model, testdf, columnId):
    X_test_norm_copy_updated, testlabelForRegression = popcol(testdf, columnId)
    predictions = model.predict(X_test_norm_copy_updated)
    return predictions, testlabelForRegression


def runColumnWiseRegressorAlgorithm(X_train_norm, X_test_norm, columnId, alg):
    if alg == 0:
        #print("Ridge regression start")
        X_train_norm_copy_updated, trainlabelForRegression = popcol(X_train_norm, columnId)
        ridgeregseed = random.randint(0, 100000)
        print("Using seed for ridge regression : " + str(ridgeregseed))
        model = linear_model.Ridge(random_state = ridgeregseed)
        model.fit(X_train_norm_copy_updated, trainlabelForRegression)

        X_test_norm_copy_updated, testlabelForRegression = popcol(X_test_norm, columnId)
        predictions = model.predict(X_test_norm_copy_updated)
        return model, predictions, testlabelForRegression

    elif alg == 1:
        #print("Lasso regression start")
        X_train_norm_copy_updated, trainlabelForRegression = popcol(X_train_norm, columnId)
        lassoregseed = random.randint(0, 100000)
        print("Using seed for lasso regression : " + str(lassoregseed))
        model = linear_model.Lasso(random_state = lassoregseed)
        model.fit(X_train_norm_copy_updated, trainlabelForRegression)

        X_test_norm_copy_updated, testlabelForRegression = popcol(X_test_norm, columnId)
        predictions = model.predict(X_test_norm_copy_updated)
        return model, predictions, testlabelForRegression

    elif alg == 2:
        #print("Elastic net regression start")
        X_train_norm_copy_updated, trainlabelForRegression = popcol(X_train_norm, columnId)
        elasticnetregseed = random.randint(0, 100000)
        print("Using seed for elasticnet regression : " + str(elasticnetregseed))
        model = ElasticNet(random_state = elasticnetregseed)
        model.fit(X_train_norm_copy_updated, trainlabelForRegression)

        X_test_norm_copy_updated, testlabelForRegression = popcol(X_test_norm, columnId)
        predictions = model.predict(X_test_norm_copy_updated)
        return model, predictions, testlabelForRegression

    elif alg == 3:
        #print("Random forest regression start")
        X_train_norm_copy_updated, trainlabelForRegression = popcol(X_train_norm, columnId)
        rfregseed = random.randint(0, 100000)
        print("Using seed for randomforest regression : " + str(rfregseed))
        model = RandomForestRegressor(random_state = rfregseed, n_estimators=100)
        model.fit(X_train_norm_copy_updated, trainlabelForRegression)

        X_test_norm_copy_updated, testlabelForRegression = popcol(X_test_norm, columnId)
        predictions = model.predict(X_test_norm_copy_updated)
        return model, predictions, testlabelForRegression



def runNovelRegressorMethod(X_train_norm, X_test_norm, y_test, algoIndex):
    num_rows, num_cols = X_train_norm.shape #X_train_norm[1].shape

    dict1 = {}
    dict2 = {}

    for i in range(num_cols):
        #X_train_norm_copy = np.copy(X_train_norm)
        #X_test_norm_copy = np.copy(X_test_norm)
        #X_train_norm_copy2 = np.copy(X_train_norm)


        model, predictions, testlabelForRegression = runColumnWiseRegressorAlgorithm(X_train_norm, X_test_norm, i, algoIndex)
        diff = predictions-testlabelForRegression
        modulo_Diff = transformmodulus(diff)

        #
        trainrf_predictions, trainlabelsForRegression = predict(model, X_train_norm, i)
        rms = rmse(trainrf_predictions, trainlabelsForRegression)
        dict1[i] = rms
        dict2[i] = modulo_Diff
        #

        if i == 0:
            rawDiffSumArrayAll = modulo_Diff
        else:
            rawDiffSumArrayAll = np.add(rawDiffSumArrayAll, modulo_Diff)


    # Create a list of tuples sorted by index 1 i.e. value field
    listofTuples = sorted(dict1.items(), key=lambda x: x[1])




    bestsize1 = 1/4
    limit1 = int(round((bestsize1) * num_cols))

    bestsize2 = 1/2
    limit2 = int(round((bestsize2) * num_cols))

    bestsize3 = 3/4
    limit3 = int(round((bestsize3) * num_cols))

    count = 0

    for elem in listofTuples:
        # print(elem[0], " ::", elem[1])
        if count == limit3:
            break

        elif count == limit1:
            rawDiffSumArray2 = rawDiffSumArray1

        elif count == limit2:
            rawDiffSumArray3 = rawDiffSumArray2


        if count >= 0 and count < limit1:
            if count == 0:
                rawDiffSumArray1 = dict2[elem[0]]
                rawDiffSumArrayZero = dict2[elem[0]]
            else:
                rawDiffSumArray1 = rawDiffSumArray1 + dict2[elem[0]]

        elif count >= limit1 and count < limit2:
            rawDiffSumArray2 = rawDiffSumArray2 + dict2[elem[0]]

        elif count >= limit2 and count < limit3:
            rawDiffSumArray3 = rawDiffSumArray3 + dict2[elem[0]]

        count = count + 1
    #print(count)


    averagerawDiff0 = np.true_divide(rawDiffSumArrayZero, 1)
    auc0 = calculateAUC(y_test, averagerawDiff0)

    averagerawDiff1 = np.true_divide(rawDiffSumArray1, limit1)
    auc1 = calculateAUC(y_test, averagerawDiff1)

    averagerawDiff2 = np.true_divide(rawDiffSumArray2, limit2)
    auc2 = calculateAUC(y_test, averagerawDiff2)

    averagerawDiff3 = np.true_divide(rawDiffSumArray3, limit3)
    auc3 = calculateAUC(y_test, averagerawDiff3)

    averagerawDiffAll = np.true_divide(rawDiffSumArrayAll, num_cols)
    auc4 = calculateAUC(y_test, averagerawDiffAll)

    return auc0, auc1, auc2, auc3, auc4











def runIF(X_train_norm, X_test_norm, y_test):
    ifseed = random.randint(0,100000)
    print("Using seed for Isolation forest : " + str(ifseed))
    ifo = IForest(random_state = ifseed)
    ifo.fit(X_train_norm)
    #y_train_scores = clf.decision_scores_  # raw outlier scores
    y_test_scores = ifo.decision_function(X_test_norm)
    auc = calculateAUC(y_test, y_test_scores)
    return auc


def runLOF(X_train_norm, X_test_norm, y_test):
    lof = LOF(n_neighbors=20)
    lof.fit(X_train_norm)
    #y_train_scores = clf.decision_scores_  # raw outlier scores
    y_test_scores = lof.decision_function(X_test_norm)
    auc = calculateAUC(y_test, y_test_scores)
    return auc


def runOCSVM(X_train_norm, X_test_norm, y_test):
    ocsvm = OCSVM()
    ocsvm.fit(X_train_norm)
    #y_train_scores = clf.decision_scores_  # raw outlier scores
    y_test_scores = ocsvm.decision_function(X_test_norm)
    auc = calculateAUC(y_test, y_test_scores)
    return auc


def runCOF(X_train_norm, X_test_norm, y_test):
    cof = COF()
    cof.fit(X_train_norm)
    # y_train_scores = clf.decision_scores_  # raw outlier scores
    y_test_scores = cof.decision_function(X_test_norm)
    auc = calculateAUC(y_test, y_test_scores)
    return auc


def runAutoEnc(X_train_norm, X_test_norm, y_test):
    num_rows, num_cols = X_train_norm.shape #X_train_norm[1].shape
    #hidden_neurons = [num_cols, int(num_cols/2), int(num_cols/2), num_cols]
    #hidden_neurons = [num_cols, int(num_cols/2),  num_cols]
    hidden_neurons = [int(num_cols/2)]

    autoencseed = random.randint(0, 100000)
    print("Using seed for autoencoder : " + str(autoencseed))
    #clf = AutoEncoder(epochs=100, random_state= autoencseed)
    clf = AutoEncoder(epochs=100, random_state= autoencseed, hidden_neurons=hidden_neurons)
    clf.fit(X_train_norm)
    #y_train_scores = clf.decision_scores_  # raw outlier scores
    y_test_scores = clf.decision_function(X_test_norm)
    auc = calculateAUC(y_test, y_test_scores)
    return auc







def calculateAUC(origlabels, scores):
    return round(roc_auc_score(origlabels, scores), ndigits=4)



def readARFFinput(inputfile):
    dataset = arff.load(open(inputfile))
    data = np.array(dataset['data'])
    labels = data[:, data.shape[1] - 1]
    data = data[:, 0:data.shape[1] - 1]

    return data, labels


def readCSVinput(inputfile):
    dataset = pd.read_csv(inputfile, header=None)
    df = pd.DataFrame(dataset)
    column_index = len(df.columns) - 1  # Last column for true label

    labels = np.array(df.pop(df.columns[column_index]))
    data = df.to_numpy()

    return data, labels



#Extracts positive data for training (Extracts only majority)
def extract_pos_training(data,index,labels):
    X= data[index]
    #Extract labels
    t_labels = labels[index]
    t_labels = np.asarray(list(map(int, t_labels))) #convert to int
    #Flag only the positive labels
    # Changed to 0
    flag=(t_labels==0)
    #Extract only the positive samples
    posdata=X[flag]

    return posdata



#Extract test data and test labels
def extract_test_data(data,index,labels):
    X = data[index]
    X=X.astype('float64')
    #Extract test labels
    test_labels=labels[index]
    test_labels = np.asarray(list(map(int, test_labels))) #convert to int

    return X, test_labels





# 0 majority 1 outlier
if __name__ == "__main__":
    """
    filename = 'testdata.csv'  # Input File name
    inputfile = filename
    """

    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('filename', type=str, help='File name')
    args = parser.parse_args()
    inputfile = args.filename


    if(inputfile.lower().endswith('.csv')) :
        data, labels = readCSVinput(inputfile)  # Read the input file
    else: # Consider as Arff file
        data, labels = readARFFinput(inputfile)  # Read the input file

    rsplitseed = random.randint(0,100000)
    print("Using rseed for repeated stratified kfold split : "+ str(rsplitseed))
    rkf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=rsplitseed)

    auciflist = []
    aucloflist = []
    aucocsvmlist = []
    auccoflist = []
    aucautoenclist = []


    aucNRridgeonelist = []
    aucNRridgeonefourthlist = []
    aucNRridgehalflist = []
    aucNRridgethreefourthlist = []
    aucNRridgelist = []

    aucNRlassoonelist = []
    aucNRlassoonefourthlist = []
    aucNRlassohalflist = []
    aucNRlassothreefourthlist = []
    aucNRlassolist = []

    aucNRelasticonelist = []
    aucNRelasticonefourthlist = []
    aucNRelastichalflist = []
    aucNRelasticthreefourthlist = []
    aucNRelasticlist = []

    aucNRrfonelist = []
    aucNRrfonefourthlist = []
    aucNRrfhalflist = []
    aucNRrfthreefourthlist = []
    aucNRrflist = []

    # Create 5 times 2 fold
    for train_index, test_index in rkf.split(data, labels):
        # Extract only majority instances for training
        posdata = extract_pos_training(data, train_index, labels)
        tdata, tlabel = extract_test_data(data, test_index, labels)

        # standardizing data for processing
        X_train_norm, X_test_norm = standardizer(posdata, tdata)

        ifauc = runIF(X_train_norm, X_test_norm, tlabel)
        auciflist.append(ifauc)

        lofauc = runLOF(X_train_norm, X_test_norm, tlabel)
        aucloflist.append(lofauc)

        ocsvmauc = runOCSVM(X_train_norm, X_test_norm, tlabel)
        aucocsvmlist.append(ocsvmauc)

        cofauc = runCOF(X_train_norm, X_test_norm, tlabel)
        auccoflist.append(cofauc)

        autoencauc = runAutoEnc(X_train_norm, X_test_norm, tlabel)
        aucautoenclist.append(autoencauc)



        aucNRridgeone, aucNRridgeonefourth, aucNRridgehalf, aucNRridgethreefourth, aucNRridgeall = runNovelRegressorMethod(X_train_norm, X_test_norm, tlabel, 0)
        aucNRridgeonelist.append(aucNRridgeone)
        aucNRridgeonefourthlist.append(aucNRridgeonefourth)
        aucNRridgehalflist.append(aucNRridgehalf)
        aucNRridgethreefourthlist.append(aucNRridgethreefourth)
        aucNRridgelist.append(aucNRridgeall)

        aucNRlassoone, aucNRlassoonefourth, aucNRlassohalf, aucNRlassothreefourth, aucNRlassoall = runNovelRegressorMethod(X_train_norm, X_test_norm, tlabel, 1)
        aucNRlassoonelist.append(aucNRlassoone)
        aucNRlassoonefourthlist.append(aucNRlassoonefourth)
        aucNRlassohalflist.append(aucNRlassohalf)
        aucNRlassothreefourthlist.append(aucNRlassothreefourth)
        aucNRlassolist.append(aucNRlassoall)

        aucNRelasticone, aucNRelasticonefourth, aucNRelastichalf, aucNRelasticthreefourth, aucNRelasticall = runNovelRegressorMethod(X_train_norm, X_test_norm, tlabel, 2)
        aucNRelasticonelist.append(aucNRlassoone)
        aucNRelasticonefourthlist.append(aucNRlassoonefourth)
        aucNRelastichalflist.append(aucNRelastichalf)
        aucNRelasticthreefourthlist.append(aucNRelasticthreefourth)
        aucNRelasticlist.append(aucNRelasticall)

        aucNRrfone, aucNRrfonefourth, aucNRrfhalf, aucNRrfthreefourth, aucNRrfall = runNovelRegressorMethod(X_train_norm, X_test_norm, tlabel, 3)
        aucNRrfonelist.append(aucNRrfone)
        aucNRrfonefourthlist.append(aucNRrfonefourth)
        aucNRrfhalflist.append(aucNRrfhalf)
        aucNRrfthreefourthlist.append(aucNRrfthreefourth)
        aucNRrflist.append(aucNRrfall)

        #print("aucif : " + str(aucif))
        #print("auclof : " + str(auclof))
        #print("aucocsvm : " + str(aucocsvm))
        #print("aucautoenc : " + str(aucautoenc))
        #print("aucNRzero : " +str())


    aucifmean = mean(auciflist)
    auclofmean = mean(aucloflist)
    aucocsvmmean = mean(aucocsvmlist)
    auccofmean = mean(auccoflist)
    aucautoencmean = mean(aucautoenclist)

    ###
    aucNRridgeonemean = mean(aucNRridgeonelist)
    aucNRridgeonefourthmean = mean(aucNRridgeonefourthlist)
    aucNRridgehalfmean = mean(aucNRridgehalflist)
    aucNRridgethreefourthmean = mean(aucNRridgethreefourthlist)
    aucNRridgeallmean= mean(aucNRridgelist)

    aucNRlassoonemean = mean(aucNRlassoonelist)
    aucNRlassoonefourthmean = mean(aucNRlassoonefourthlist)
    aucNRlassohalfmean = mean(aucNRlassohalflist)
    aucNRlassothreefourthmean = mean(aucNRlassothreefourthlist)
    aucNRlassoallmean = mean(aucNRlassolist)

    aucNRelasticonemean = mean(aucNRelasticonelist)
    aucNRelasticonefourthmean = mean(aucNRelasticonefourthlist)
    aucNRelastichalfmean = mean(aucNRelastichalflist)
    aucNRelasticthreefourthmean = mean(aucNRelasticthreefourthlist)
    aucNRelasticallmean = mean(aucNRelasticlist)

    aucNRrfonemean = mean(aucNRrfonelist)
    aucNRrfonefourthmean = mean(aucNRrfonefourthlist)
    aucNRrfhalfmean = mean(aucNRrfhalflist)
    aucNRrfthreefourthmean = mean(aucNRrfthreefourthlist)
    aucNRrfallmean = mean(aucNRrflist)
    ###

    aucifstd = stdev(auciflist)
    auclofstd = stdev(aucloflist)
    aucocsvmstd = stdev(aucocsvmlist)
    auccofstd = stdev(auccoflist)
    aucautoencstd = stdev(aucautoenclist)


    aucNRridgeonestd = stdev(aucNRridgeonelist)
    aucNRridgeonefourthstd = stdev(aucNRridgeonefourthlist)
    aucNRridgehalfstd = stdev(aucNRridgehalflist)
    aucNRridgethreefourthstd = stdev(aucNRridgethreefourthlist)
    aucNRridgeallstd = stdev(aucNRridgelist)

    aucNRlassoonestd = stdev(aucNRlassoonelist)
    aucNRlassoonefourthstd = stdev(aucNRlassoonefourthlist)
    aucNRlassohalfstd = stdev(aucNRlassohalflist)
    aucNRlassothreefourthstd = stdev(aucNRlassothreefourthlist)
    aucNRlassoallstd = stdev(aucNRlassolist)

    aucNRelasticonestd = stdev(aucNRelasticonelist)
    aucNRelasticonefourthstd = stdev(aucNRelasticonefourthlist)
    aucNRelastichalfstd = stdev(aucNRelastichalflist)
    aucNRelasticthreefourthstd = stdev(aucNRelasticthreefourthlist)
    aucNRelasticallstd = stdev(aucNRelasticlist)

    aucNRrfonestd = stdev(aucNRrfonelist)
    aucNRrfonefourthstd = stdev(aucNRrfonefourthlist)
    aucNRrfhalfstd = stdev(aucNRrfhalflist)
    aucNRrfthreefourthstd = stdev(aucNRrfthreefourthlist)
    aucNRrfallstd = stdev(aucNRrflist)

    """
    print("Results : ~~~~~~~~~")
    print(" - - - - ")
    print("auc-if-mean : " + str(aucifmean))
    print("auc-lof-mean : " + str(auclofmean))
    print("auc-ocsvm-mean : " + str(aucocsvmmean))
    print("auc-cof-mean : " + str(auccofmean))
    print("auc-autoenc-mean : " + str(aucautoencmean))
    print(" - - - - ")

    print("auc-NR-ridge-mean-one : " + str(aucNRridgeonemean))
    print("auc-NR-ridge-mean-one fourth : " + str(aucNRridgeonefourthmean))
    print("auc-NR-ridge-mean-half : " + str(aucNRridgehalfmean))
    print("auc-NR-ridge-mean-three fourth : " + str(aucNRridgethreefourthmean))
    print("auc-NR-ridge-mean-all : " + str(aucNRridgeallmean))

    print(" - - - - ")

    print("auc-NR-lasso-mean-one : " + str(aucNRlassoonemean))
    print("auc-NR-lasso-mean-one fourth : " + str(aucNRlassoonefourthmean))
    print("auc-NR-lasso-mean-half : " + str(aucNRlassohalfmean))
    print("auc-NR-lasso-mean-three fourth : " + str(aucNRlassothreefourthmean))
    print("auc-NR-lasso-mean-all : " + str(aucNRlassoallmean))

    print(" - - - - ")

    print("auc-NR-elastic-mean-one : " + str(aucNRelasticonemean))
    print("auc-NR-elastic-mean-one fourth : " + str(aucNRelasticonefourthmean))
    print("auc-NR-elastic-mean-half : " + str(aucNRelastichalfmean))
    print("auc-NR-elastic-mean-three fourth : " + str(aucNRelasticthreefourthmean))
    print("auc-NR-elastic-mean-all : " + str(aucNRelasticallmean))

    print(" - - - - ")

    print("auc-NR-RF-mean-one : " + str(aucNRrfonemean))
    print("auc-NR-RF-mean-one fourth : " + str(aucNRrfonefourthmean))
    print("auc-NR-RF-mean-half : " + str(aucNRrfhalfmean))
    print("auc-NR-RF-mean-three fourth : " + str(aucNRrfthreefourthmean))
    print("auc-NR-RF-mean-all : " + str(aucNRrfallmean))

    print(" - - - - ")
    print(" - - - - ")
    print(" - - - - ")

    print("auc-NR-ridge-std-one : " + str(aucNRridgeonestd))
    print("auc-NR-ridge-std-one fourth : " + str(aucNRridgeonefourthstd))
    print("auc-NR-ridge-std-half : " + str(aucNRridgehalfstd))
    print("auc-NR-ridge-std-three fourth : " + str(aucNRridgethreefourthstd))
    print("auc-NR-ridge-std-all : " + str(aucNRridgeallstd))

    print(" - - - - ")

    print("auc-NR-lasso-std-one : " + str(aucNRlassoonestd))
    print("auc-NR-lasso-std-one fourth : " + str(aucNRlassoonefourthstd))
    print("auc-NR-lasso-std-half : " + str(aucNRlassohalfstd))
    print("auc-NR-lasso-std-three fourth : " + str(aucNRlassothreefourthstd))
    print("auc-NR-lasso-std-all : " + str(aucNRlassoallstd))

    print(" - - - - ")

    print("auc-NR-elastic-std-one : " + str(aucNRelasticonestd))
    print("auc-NR-elastic-std-one fourth : " + str(aucNRelasticonefourthstd))
    print("auc-NR-elastic-std-half : " + str(aucNRelastichalfstd))
    print("auc-NR-elastic-std-three fourth : " + str(aucNRelasticthreefourthstd))
    print("auc-NR-elastic-std-all : " + str(aucNRelasticallstd))

    print(" - - - - ")

    print("auc-NR-RF-std-one : " + str(aucNRrfonestd))
    print("auc-NR-RF-std-one fourth : " + str(aucNRrfonefourthstd))
    print("auc-NR-RF-std-half : " + str(aucNRrfhalfstd))
    print("auc-NR-RF-std-three fourth : " + str(aucNRrfthreefourthstd))
    print("auc-NR-RF-std-all : " + str(aucNRrfallstd))

    print("Results : ~~~~~~~~~")
    """

    csvd = ","
    outputfilename1 = inputfile+"result.csv"
    outputfilename2 = inputfile + "resultstd.csv"
    f1 = open(outputfilename1, "w")
    f2 = open(outputfilename2, "w")
    ###
    f1.write(str(aucifmean))
    f1.write(csvd)

    f1.write(str(auclofmean))
    f1.write(csvd)

    f1.write(str(aucocsvmmean))
    f1.write(csvd)

    f1.write(str(auccofmean))
    f1.write(csvd)

    f1.write(str(aucautoencmean))
    f1.write(csvd)
    ###
    f1.write(str(aucNRridgeonemean))
    f1.write(csvd)

    f1.write(str(aucNRridgeonefourthmean))
    f1.write(csvd)

    f1.write(str(aucNRridgehalfmean))
    f1.write(csvd)

    f1.write(str(aucNRridgethreefourthmean))
    f1.write(csvd)

    f1.write(str(aucNRridgeallmean))
    f1.write(csvd)
    #

    f1.write(str(aucNRlassoonemean))
    f1.write(csvd)

    f1.write(str(aucNRlassoonefourthmean))
    f1.write(csvd)

    f1.write(str(aucNRlassohalfmean))
    f1.write(csvd)

    f1.write(str(aucNRlassothreefourthmean))
    f1.write(csvd)

    f1.write(str(aucNRlassoallmean))
    f1.write(csvd)
    #


    f1.write(str(aucNRelasticonemean))
    f1.write(csvd)

    f1.write(str(aucNRelasticonefourthmean))
    f1.write(csvd)

    f1.write(str(aucNRelastichalfmean))
    f1.write(csvd)

    f1.write(str(aucNRelasticthreefourthmean))
    f1.write(csvd)

    f1.write(str(aucNRelasticallmean))
    f1.write(csvd)
    #
    f1.write(str(aucNRrfonemean))
    f1.write(csvd)

    f1.write(str(aucNRrfonefourthmean))
    f1.write(csvd)

    f1.write(str(aucNRrfhalfmean))
    f1.write(csvd)

    f1.write(str(aucNRrfthreefourthmean))
    f1.write(csvd)

    f1.write(str(aucNRrfallmean))
    f1.write(csvd)
    ############
    f2.write(str(aucifstd))
    f2.write(csvd)

    f2.write(str(auclofstd))
    f2.write(csvd)

    f2.write(str(aucocsvmstd))
    f2.write(csvd)

    f2.write(str(auccofstd))
    f2.write(csvd)

    f2.write(str(aucautoencstd))
    f2.write(csvd)


    f2.write(str(aucNRridgeonestd))
    f2.write(csvd)

    f2.write(str(aucNRridgeonefourthstd))
    f2.write(csvd)

    f2.write(str(aucNRridgehalfstd))
    f2.write(csvd)

    f2.write(str(aucNRridgethreefourthstd))
    f2.write(csvd)

    f2.write(str(aucNRridgeallstd))
    f2.write(csvd)
    #

    f2.write(str(aucNRlassoonestd))
    f2.write(csvd)

    f2.write(str(aucNRlassoonefourthstd))
    f2.write(csvd)

    f2.write(str(aucNRlassohalfstd))
    f2.write(csvd)

    f2.write(str(aucNRlassothreefourthstd))
    f2.write(csvd)

    f2.write(str(aucNRlassoallstd))
    f2.write(csvd)
    #


    f2.write(str(aucNRelasticonestd))
    f2.write(csvd)

    f2.write(str(aucNRelasticonefourthstd))
    f2.write(csvd)

    f2.write(str(aucNRelastichalfstd))
    f2.write(csvd)

    f2.write(str(aucNRelasticthreefourthstd))
    f2.write(csvd)

    f2.write(str(aucNRelasticallstd))
    f2.write(csvd)
    #
    f2.write(str(aucNRrfonestd))
    f2.write(csvd)

    f2.write(str(aucNRrfonefourthstd))
    f2.write(csvd)

    f2.write(str(aucNRrfhalfstd))
    f2.write(csvd)

    f2.write(str(aucNRrfthreefourthstd))
    f2.write(csvd)

    f2.write(str(aucNRrfallstd))
    f2.write(csvd)




    f1.close()
    f2.close()