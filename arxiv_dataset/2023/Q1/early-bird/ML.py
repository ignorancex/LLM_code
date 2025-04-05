from __future__ import print_function, division





from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import *

from sklearn.linear_model import *
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from utilities import _randuniform,_randchoice,_randint
from utilities import *

def DT(default=False):


    if default:
        model = DecisionTreeClassifier()
        tmp = "default_" + DecisionTreeClassifier.__name__
        return model, tmp
    else:
        a=_randuniform(0.0,1.0)
        b=_randchoice(['gini','entropy'])
        c=_randchoice(['best','random'])

        model = DecisionTreeClassifier(criterion=b, splitter=c, min_samples_split=a,  min_impurity_decrease=0.0)
        tmp=str(a)+"_"+b+"_"+c+"_"+DecisionTreeClassifier.__name__
        return model,tmp

def RF(default=False):

    if default:
        # a = _randint(50, 150)
        # b = _randchoice(['gini', 'entropy'])
        # c = _randuniform(0.0, 1.0)
        model = RandomForestClassifier()
        tmp = "default_" + RandomForestClassifier.__name__
        return model, tmp
    else:
        a = _randint(50, 150)
        b = _randchoice(['gini', 'entropy'])
        c = _randuniform(0.0, 1.0)
        model = RandomForestClassifier(n_estimators=a,criterion=b,min_samples_split=c)
        tmp=str(a)+"_"+b+"_"+str(round(c,5))+"_"+RandomForestClassifier.__name__
        return model,tmp

def LR(default=False):


    # a = _randchoice(['l1', 'l2'])
    # b = _randuniform(0.0, 0.1)
    # c = _randint(1,500)
    #
    # model = LogisticRegression(penalty=a, tol=b, C=float(c))
    # tmp = str(a) + "_" + str(b) + "_" + str(c) + "_" + LogisticRegression.__name__
    # return model, tmp

    if default:
        model = LogisticRegression()
        tmp = "default_" + RandomForestClassifier.__name__
        return model, tmp
    else:
        C= _randuniform( 1e-12, 100.0)
        tol= _randuniform(1e-8, 10.0)
        l1_ratio= _randuniform( 0, 1)
        fit_intercept= _randchoice([True, False])
        warm_start= _randchoice( [True, False])
        class_weight= _randchoice( ['balanced', None])
        max_iter= _randint(1, 151)

        solver = _randchoice(['newton-cg', 'lbfgs' , 'liblinear' , 'sag', 'saga' ])
        if solver == 'newton-cg':
            penalty = _randchoice(['l2', 'none'])
        elif solver == 'lbfgs':
            penalty = _randchoice(['l2', 'none'])
        elif solver == 'liblinear':
            penalty = _randchoice(['l1'])
        elif solver == 'sag':
            penalty = _randchoice(['l2', 'none'])
        elif solver == 'saga':
            penalty = _randchoice(['l1', 'elasticnet', 'none'])



        model = LogisticRegression(C=C, tol=tol, l1_ratio=l1_ratio, fit_intercept=fit_intercept, warm_start=warm_start,
                                       class_weight=class_weight, max_iter=max_iter, penalty=penalty, solver=solver)



        tmp = str(model.get_params())


        return model, tmp


def NB(default=False):

    if default:
        model = MultinomialNB()
        tmp = "default_" + MultinomialNB.__name__
    else:
        a = _randuniform(0.0, 0.1)
        model = MultinomialNB(alpha=a)
        tmp = str(a) + "_" + MultinomialNB.__name__



    return model, tmp

def KNN(default=False):

    if default:
        model = KNeighborsClassifier()
        tmp = "default_" + KNeighborsClassifier.__name__
        return model, tmp
    else:
        a = _randint(2, 25)
        b = _randchoice(['uniform', 'distance'])
        c = _randchoice(['minkowski','chebyshev'])

        if c=='minkowski':
            d=_randint(1,15)
        else:
            d=2

        model = KNeighborsClassifier(n_neighbors=a, weights=b, algorithm='auto', p=d, metric=c, n_jobs=-1)
        tmp = str(a) + "_" + b + "_" +c+"_"+str(d) + "_" + KNeighborsClassifier.__name__

        return model,tmp


def SVM():
    # from sklearn.preprocessing import MinMaxScaler
    # scaling = MinMaxScaler(feature_range=(-1, 1)).fit(train_data)
    # train_data = scaling.transform(train_data)
    # test_data = scaling.transform(test_data)
    a = _randint(1, 500)
    b = _randchoice(['linear', 'poly', 'rbf', 'sigmoid'])
    c = _randint(2,10)
    d = _randuniform(0.0,1.0)
    e = _randuniform(0.0,0.1)
    f = _randuniform(0.0, 0.1)
    model = SVC(C=float(a), kernel=b, degree=c, gamma=d, coef0=e, tol=f, cache_size=20000)
    tmp = str(a) + "_" + b+"_"+str(c) + "_" + str(round(d,5)) + "_" + str(round(e,5)) + "_"+str(round(f,5)) + "_"+SVC.__name__
    return model, tmp





def new_classifier(classifier):

    if isinstance(classifier, LogisticRegression):
        return LogisticRegression()
    elif isinstance(classifier, KNeighborsClassifier):
        return KNeighborsClassifier()
    elif isinstance(classifier, GaussianNB):
        return GaussianNB()
    elif isinstance(classifier, RandomForestClassifier):
        return RandomForestClassifier()
    elif isinstance(classifier, DecisionTreeClassifier):
        return DecisionTreeClassifier()
    elif isinstance(classifier, Perceptron):
        return Perceptron()
    elif isinstance(classifier, LinearSVC):
        return LinearSVC()



    float('unknown classifier')


def PN():

    penalty = _randchoice(['l2', 'l1'])
    loss = _randchoice([ 'squared_hinge'])
    # dual = _randchoice([True, False])
    tol = _randuniform(1e-6, 1e-2)
    C = _randuniform(1e-12, 100.0)
    multi_class = _randchoice(['ovr', 'crammer_singer'])
    fit_intercept = _randchoice([True, False])
    intercept_scaling = _randuniform(0, 10)
    class_weight = _randchoice(['balanced', None])
    max_iter = _randint(500, 2000)







    # penalty = _randchoice(['l2', 'l1', 'elasticnet', None])
    # alpha = _randuniform(1e-6, 1)
    # l1_ratio = _randuniform(0, 1)
    # fit_intercept = _randchoice([True, False])
    # max_iter = _randint(500, 2000)
    # tol = _randuniform(1e-6, 1)
    # # shuffle = _randchoice([True, False])
    # eta0 =  _randint(1, 50)
    # early_stopping = _randchoice([True, False])
    # # validation_fraction = _randuniform(1e-6, 1)
    # n_iter_no_change = _randint(1, 50)
    # class_weight= _randchoice( ['balanced', None])
    # warm_start= _randchoice( [True, False])
    #
    # model = Perceptron(penalty=penalty, alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept,
    #                    max_iter=max_iter, tol=tol,   eta0=eta0, early_stopping=early_stopping,
    #                      n_iter_no_change=n_iter_no_change, class_weight=class_weight, warm_start=warm_start)


    model = LinearSVC(penalty=penalty, loss=loss, dual=False, tol=tol, C=C, multi_class=multi_class, fit_intercept=fit_intercept,
                      intercept_scaling=intercept_scaling, class_weight=class_weight, max_iter=max_iter)
    tmp = model.get_params()


    return model, tmp




def run_model(train_data,test_data,model,metric,training=-1):
    model.fit(train_data[train_data.columns[:training]], train_data["bug"])
    prediction = model.predict(test_data[test_data.columns[:training]])
    test_data.loc[:,"prediction"]=prediction
    return round(get_score(metric,prediction, test_data["bug"].tolist(),test_data ),5)

