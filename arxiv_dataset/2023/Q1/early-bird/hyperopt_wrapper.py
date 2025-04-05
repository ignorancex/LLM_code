from rq_run import *
from hyperopt import hp, fmin, tpe, Trials



class hyperopt_wrapper(object):
    """
    @author: N.C. Shrikanth (nc.shrikanth@gmail.com)
    A simple wrapper to HyperOpt framework (see http://hyperopt.github.io/hyperopt/)
    """

    def __init__(self, classifier, trainChanges, tuneChanges,  fe=100):
        self.clf = classifier
        self.fe = fe
        self.train_changes = trainChanges
        self.tune_changes = tuneChanges
        self.best_param = None
        self.best_param_list = []
        self.loss_so_far = ERROR_SCORE
        self.visit = 0
        self.visitedParamMap = {}
        # self.projectName = projectName
        self.found = False
        self.def_score = None


    def local_eval(self, train_changes, tune_changes, params):

        trainX = train_changes.drop(labels=['Buggy'], axis=1)
        trainY = train_changes.Buggy

        try:
            self.clf = new_classifier(self.clf)
            self.clf.set_params(**params)
            self.clf.fit(trainX, trainY)

            F = computeMeasures(tune_changes, self.clf, [], [0 for x in range(0, len(tune_changes))])

            if F[DE_GOAL][0] < self.loss_so_far:
                self.loss_so_far = F[DE_GOAL][0]
                self.best_param = params

        except Exception as e:
            print('[HYPER_OPT_LOCAL_EVAL_ERROR]', 'exception = ',e)
            return ERROR_SCORE

        return F[DE_GOAL][0]

    def objFunc(self, rawParams):

        params = self.unbox(rawParams)

        temp_train_changes = self.train_changes.copy(deep=True)
        temp_tune_changes = self.tune_changes.copy(deep=True)

        return  self.local_eval(temp_train_changes, temp_tune_changes, params)

        # return (self.local_eval(temp_train_changes, temp_tune_changes.head(50), params) +
        #         self.local_eval(temp_train_changes, temp_tune_changes.tail(50), params) +
        #         self.local_eval(temp_train_changes, temp_tune_changes.head(100).tail(50), params)) / 3


    # def default_score(self, params):
    #
    #     temp_train_changes = self.train_changes.copy(deep=True)
    #     temp_tune_changes = self.tune_changes.copy(deep=True)
    #
    #     trainX = temp_train_changes.drop(labels=['Buggy'], axis=1)
    #     trainY = temp_train_changes.Buggy
    #
    #     try:
    #         self.clf.set_params(**params)
    #         self.clf.fit(trainX, trainY)
    #         F = computeMeasures(temp_tune_changes, self.clf, [], [0 for x in range(0, len(temp_tune_changes))])
    #         return float(F[DE_GOAL][0])
    #     except Exception as e:
    #         print('{ exception internal 1 >> ' + str(self.projectName) + " >> " + str(e) + ">>" + str(params)+" }")
    #         return ERROR_SCORE

    def getParamSpace(self):


        if getSimpleName(self.clf) == str(RandomForestClassifier.__name__):
            p = {
                'n_estimators': hp.choice('n_estimators', range(50, 150)),
                'criterion': hp.choice('criterion', ['gini', 'entropy']),
                'max_depth': hp.choice('max_depth', [None] + [x for x in range(1, 50)]),
                'min_samples_split': hp.uniform('min_samples_split', 0.1, 1),
                'min_samples_leaf': hp.uniform('min_samples_leaf', 0, 0.5),
                'min_weight_fraction_leaf': hp.uniform('min_weight_fraction_leaf', 0.0, 0.5),
                'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2']),
                'max_leaf_nodes': hp.choice('max_leaf_nodes', [None] + [y for y in range(2, 50)]),
                'min_impurity_decrease': hp.uniform('min_impurity_decrease', 0.0, 1.0),

                'bootstrap': hp.choice('bootstrap', [
                    {
                        'bootstrap': True,
                        'oob_score': hp.choice('oob_score_1', [True, False])
                    },
                    {
                        'bootstrap': False,
                        'oob_score': hp.choice('oob_score_2', [False])
                    }
                ]),
                'class_weight': hp.choice('class_weight', ['balanced', 'balanced_subsample', None]),
                'ccp_alpha': hp.uniform('ccp_alpha',  0.0, 1.0),
                'max_samples': hp.uniform('max_samples', 0.0, 1.0)
            }
            return p
        elif getSimpleName(self.clf) == str(RidgeClassifier.__name__):

            p = {
            'alpha': hp.uniform('alpha', 0.0, 2.0),
            'fit_intercept': hp.choice('fit_intercept',[True, False]),
            'normalize': hp.choice('normalize', [True, False]),
            'copy_X': hp.choice('copy_X', [True, False]),
            'tol': hp.uniform('tol', 1e-6, 1e6),
            'class_weight': hp.choice('class_weight', ['balanced', None]),
            'solver': hp.choice('solver', ['auto', 'svd',  'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'])
            }


            return p
        elif getSimpleName(self.clf) == str(LogisticRegression.__name__):
            p = {
            'C': hp.uniform('C', 0.00001, 100.0),
            'tol': hp.uniform('tol', 1e-8, 10.0),
            'l1_ratio': hp.choice('l1_ratio', [ hp.uniform('l1_ratio_1', 0, 1) ] ),
            'fit_intercept': hp.choice('fit_intercept',[True, False]),
            'warm_start': hp.choice('warm_start', [True, False]),
            'class_weight': hp.choice('class_weight', ['balanced', None]),
            'max_iter': hp.choice('max_iter', range(1, 151)),
            'solver': hp.choice('solver', [
                {
                  'solver': 'newton-cg',
                 'penalty':hp.choice('penalty_1', ['l2', 'none'])
                },
                {'solver': 'lbfgs',
                 'penalty': hp.choice('penalty_2', ['l2', 'none'])
                 },
                {'solver': 'liblinear',
                 'penalty': hp.choice('penalty_3', ['l1']),
                 'intercept_scaling': hp.uniform('intercept_scaling_1', 0.0,2),
                 },
                {'solver': 'sag',
                 'penalty': hp.choice('penalty_4', ['l2', 'none'])
                 },
                {'solver': 'saga',
                 'penalty': hp.choice('penalty_5', ['l1', 'elasticnet', 'none'])
                 }
                ]
                )
            }

            return p
        elif getSimpleName(self.clf) == str(GaussianNB.__name__):
            p = {
                'priors': hp.choice('priors', [ [0.1, 0.9], [0.9,0.1], [0.8,0.2], [0.2,0.8],  [0.5,0.5], [0.4,0.6], [0.6,0.4] ]),
                'var_smoothing': hp.uniform('var_smoothing', 1e-18, 1e-4)
            }
            return p
        elif getSimpleName(self.clf) == str(DecisionTreeClassifier.__name__):
            p = {
                'criterion': hp.choice('criterion', ['gini', 'entropy']),
                'splitter': hp.choice('splitter', ['random', 'best']),
                'max_depth': hp.choice('max_depth', [None] + [x for x in range(1, 50)]),
                'min_samples_split': hp.uniform('min_samples_split', 0.1, 1),
                'min_samples_leaf': hp.uniform('min_samples_leaf', 0, 0.5),
                'min_weight_fraction_leaf': hp.uniform('min_weight_fraction_leaf', 0.0, 0.5),
                'max_features': hp.choice('max_features', [mf for mf in range(1, len(self.train_changes.columns.tolist()) - 1)]),
                'max_leaf_nodes': hp.choice('max_leaf_nodes', [None] + [y for y in range(2, 50)]),
                'min_impurity_decrease': hp.uniform('min_impurity_decrease', 0.0, 1.0),
                'min_impurity_split': hp.uniform('min_impurity_split', 0.0, 1.0),
                'class_weight': hp.choice('class_weight', ['balanced', None]),
                'ccp_alpha': hp.uniform('ccp_alpha',  0.0, 1.0)
            }


            return p

        elif getSimpleName(self.clf) == str(KNeighborsClassifier.__name__):
            p = {
                'n_neighbors': hp.choice('n_neighbors', range(1, len(self.train_changes)-1)),
                'weights': hp.choice('weights', ['uniform', 'distance']),
                'algorithm': hp.choice('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
                'leaf_size': hp.choice('leaf_size', range(1, 51)),
                'p': hp.choice('p', range(1, 51)),
                'metric': hp.choice('metric', ['manhattan', 'chebyshev', 'minkowski'])

            }
            return p
        elif getSimpleName(self.clf) == str(SVC.__name__):
            p = {
                'C': hp.uniform('C', 1, 100),
                'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
                'degree': hp.choice('degree', [2,3,4]),
                'gamma': hp.choice('gamma', ['scale', 'auto']),
                'shrinking': hp.choice('shrinking', [True, False]),
                'probability': hp.choice('probability', [True, False]),
                'tol': hp.uniform('tol', 1e-6, 1e-1 ),
                'class_weight': hp.choice('class_weight', ['balanced', None]),
                'max_iter': hp.choice('max_iter', range(1, 151)),
                # 'decision_function_shape': hp.choice('decision_function_shape', ['ovo', 'ovr']),
                'break_ties': hp.choice('break_ties', [True, False]),
            }
            return p

        elif getSimpleName(self.clf) == str(MLPClassifier.__name__):

            p ={
                'hidden_layer_sizes': hp.choice( 'hidden_layer_sizes',
                                                 [(50, 50, 50), (50, 100, 50), (100,)]),
                'activation': hp.choice('activation', ['identity', 'logistic',
                                                                   'tanh', 'relu']),
                'solver': hp.choice('solver', ['sgd', 'adam', 'lbfgs']),
                'alpha': hp.uniform( 'alpha', 0.0001, 0.05),
                'learning_rate': hp.choice('learning_rate', ['invscaling', 'constant', 'adaptive']),
                'batch_size': hp.choice('batch_size', ['auto', 10,25,50]),
                'learning_rate_init':  hp.uniform( 'learning_rate_init', 0.001, 0.05),
                'power_t': hp.uniform('power_t', 0.5, 1.0),
                'max_iter': hp.choice('max_iter', [50,100,150,200,250]),
                'shuffle': hp.choice('shuffle', [True, False]),
                'tol': hp.uniform('tol', 1e-6, 1e-1),
                'momentum': hp.uniform('momentum', 0.1, 1.0),
                'early_stopping': hp.choice('early_stopping', [True, False]),
                'epsilon': hp.uniform('epsilon', 1e-12, 1e-1)

            }

            return p

        # elif getSimpleName(self.clf) == str(FastFrugalTreeClassifier.__name__):
        #
        #     p = {
        #
        #         'max_categories': hp.choice('max_categories', [x for x in range(4, 5)]),
        #         'max_cuts': hp.choice('max_cuts', [x for x in range (100, 101)]),
        #         'max_levels': hp.choice('max_levels', [y for y in range(2,5)]),
        #         'stopping_param': hp.uniform('stopping_param', 0.1, 0.25),
        #
        #     }
        #
        #     return p

        return {}

    def reconstructParamSpace(self, best_param):

        newParamSpace = {}

        full_space = self.getParamSpace()

        for k, v in best_param.items():
            if k in full_space:
                if isinstance(v, int):
                    print(k, [z for z in range(int(v/2), v*2)] + [v])
                    newParamSpace[k] =  hp.choice(k, [z for z in range(int(v/2), v*2)] + [v] )
                elif isinstance(v, float):
                    print(k, v/2, v*2)
                    newParamSpace[k] =  hp.uniform(k, v/2, v*2)
                else:
                    newParamSpace[k] = hp.choice(k, [v])


        for l, m in full_space.items():
            if l not in newParamSpace:
                newParamSpace[l] = m

        if DEBUG:
            print(newParamSpace)

        return newParamSpace

    def getBestParams(self):

        trials = Trials()
        fmin(self.objFunc, space=self.getParamSpace(), algo=tpe.suggest, max_evals=self.fe, verbose=False,
             trials=trials, show_progressbar=True)

        return self.best_param, self.loss_so_far, -1

    def unbox(self, rawParams):
        p = {}

        for k,v in rawParams.items():

            if isinstance(v, dict):
                for kk, vv in v.items():
                    p[kk] = vv
            else:
                p[k] = v

        return p

def hyper_opt_run(trainChanges, tuneChanges):

    min_score = 1
    best_learner = None

    for c in CLASSIFIERS_TRADITIONAL:

        classifier = c()

        hyper_opt =  hyperopt_wrapper(classifier, trainChanges,
                                      tuneChanges,  100)
        params, score, y = hyper_opt.getBestParams()

        if best_learner is None or min_score > score:
            min_score = score

            print("\t\t HYPER-OPT SETTING BEST: \t ",min_score,
                  getSimpleName(classifier))
            for k,v in params.items():
                print('==>\t',k,v)

            best_learner = new_classifier(classifier)
            best_learner.set_params(**params)

    print("\n\n $ Returning ", getSimpleName(best_learner), best_learner.get_params())
    return best_learner

if __name__ == '__main__':

    project_changes = getProject('numpy').getAllChanges()

    sample_size = int(len(project_changes)/2)

    train, test = project_changes.head(sample_size), project_changes.tail(sample_size)

    hyper_opt_run(train, test, 'numpy')
