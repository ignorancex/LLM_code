import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from IPython import embed
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import xgboost as xgb
import seaborn as sns
random_seed = 412069413
np.random.seed(random_seed)

MAX_EVALS = 200

plt.style.use("seaborn")
sns.set(font_scale=1.8)
plt.rcParams['axes.titlesize'] = 24; plt.rcParams['axes.labelsize'] = 21;
plt.rcParams["xtick.labelsize"] = 21; plt.rcParams["ytick.labelsize"] = 21; plt.rcParams["legend.fontsize"] = 21


class Model():
    """
    Parent class for all model classes
    Contains common functions like plotting, evaluation,
    and feature importance

    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.params  = dataset.params 
        self.X_train, self.X_test, self.y_train, self.y_test = dataset.get_data()

        self.PATH_MODEL = f'{dataset.path_results}/Models/'
        self.PATH_PLOTS = f'{dataset.path_results}/Plots/'


    def plot_sorted_prediction(self, X=None, y=None, fn=''):
        print(f"Plotting sorted predictions for {self.name}")
        PATH_SORTEDPRED = os.path.join(self.PATH_PLOTS, f'SortedPrediction/')

        if not os.path.exists(PATH_SORTEDPRED):
            os.makedirs(PATH_SORTEDPRED)

        if X is None:
            X = self.X_test
        if y is None:
            y = self.y_test

        pred = self.model.predict(X)
        residual = pred - y        
        RMS = np.sqrt( np.mean( residual**2 ) )


        plt.clf()
        plt.figure(figsize=(12,8))


        print('y_test shape: ', y.shape)
        try:
            idxSorted = y.ravel().argsort()
        except:
            idxSorted = y.argsort()
        plt.plot(range(len(pred)), pred[idxSorted], label="Predicted")
        try:
            idxSorted = y.ravel().argsort()
            plt.plot(range(len(pred)), y.ravel()[idxSorted], label="True")
        except:
            embed()

        print(f'RMS for {self.name}: {RMS} arcsecs')
        rms_offset = self.dataset.rms_offset

        plt.xlabel("Sample #")
        plt.ylabel("Offset [\'\']")
        plt.title(f'True and predicted offset')
        plt.legend()

        save_path_sp   = os.path.join(PATH_SORTEDPRED, f"sortpred_{self.name}{fn}.pdf")

        plt.savefig(save_path_sp, dpi = 400)
        plt.clf()
        return RMS, rms_offset

    def plot_histogram(self, X=None, y=None, fn=''):
        print(f"Plotting histogram for {self.name}")
        PATH_HISTOGRAM  = os.path.join(self.PATH_PLOTS, f'Histogram/')
        if not os.path.exists(PATH_HISTOGRAM):
            os.makedirs(PATH_HISTOGRAM)

        if X is None:
            X = self.X_test
        if y is None:
            y = self.y_test

        pred = self.model.predict(X)

        plt.clf()

        n_bins = 25
        plt.figure(figsize=(12,8))
    
        _, bins, _ = plt.hist(y, bins = n_bins, alpha = 0.8, label = 'Current offset')
        plt.hist(y - pred, bins = bins, alpha = 0.8, label='XGB Model')
        plt.xlabel('Offset [\'\']')
        plt.ylabel('Number of samples')

        plt.title('Distribution of pointing offsets with and without XGB model')

        plt.legend()
        plt.tight_layout()

        save_path_hist = os.path.join(PATH_HISTOGRAM, f"hist_{self.name}{fn}.pdf")

        plt.savefig(save_path_hist, dpi = 400)


class XGBoostRegressor(Model):
    def __init__(self, dataset, name = 'XGB', load_model = False, train_model = False, model_path = None):
        super().__init__(dataset)

    
        self.name = name


        self.dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        self.dtest  = xgb.DMatrix(self.X_test, label=self.y_test)

        if not isinstance(self.X_train, type(self.y_train)):
            self.y_train = self.y_train.values.ravel()
            self.y_test  = self.y_test.values.ravel()

        else:
            self.y_train = self.y_train.values.ravel()
            self.y_test  = self.y_test.values.ravel()
            self.X_train = self.X_train.values
            self.X_test  = self.X_test.values

        print(self.X_train.shape, self.y_train.shape)
        print(self.X_test.shape, self.y_test.shape)

        if load_model:
            self.load_model(model_path)

        if train_model:
            self.train()


    def train(self, save = True):
        
        space={'max_depth': hp.quniform("max_depth", 1, 5, 1),
                'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
                'subsample': hp.uniform('subsample', 0.5, 1),
                'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
                'learning_rate': hp.loguniform('learning_rate', -5, 0),
                'n_estimators': hp.quniform('n_estimators', 20, 500, 1),
            }

        trials = Trials()
        self.model_params = fmin(fn = self.objective,
                            space = space,
                            algo = tpe.suggest,
                            max_evals = MAX_EVALS,
                            trials = trials,
                            rstate=np.random.default_rng(random_seed)
                            )


        self.model_params['max_depth'] = int(self.model_params['max_depth'])
        self.model_params['n_estimators'] = int(self.model_params['n_estimators'])

        self.model = xgb.XGBRegressor(**self.model_params, n_jobs = 24)
        self.model.fit(self.X_train,self.y_train)

        if save:
            self.save_model()
        
        return self.model_params




    def objective(self, space):
        evaluation = [(self.X_train, self.y_train), (self.X_test, self.y_test)]
        
        space['n_estimators'] = int(space['n_estimators'])
        space['max_depth'] = int(space['max_depth'])

        regr = xgb.XGBRegressor(**space, n_jobs = 24, seed=random_seed)
        regr.fit(self.X_train, self.y_train, eval_set=evaluation,verbose=False)        

        pred = regr.predict(self.X_test)
        mse = mean_squared_error(self.y_test, pred)
        return {'loss': mse, 'status': STATUS_OK }


    def save_model(self):
        if not os.path.exists(self.PATH_MODEL):
            os.makedirs(self.PATH_MODEL)
        print('Saving model')
        path = os.path.join(self.PATH_MODEL, f'{self.name}.pkl')
        pickle.dump(self.model, open(path, 'wb'))


    def load_model(self, model_path):
        if model_path is not None:
            path = model_path
        else:
            path = os.path.join(self.PATH_MODEL, f'{self.name}.pkl')

        if os.path.exists(path):
            self.model = pickle.load(open(path, 'rb'))
        else:
            print('No model found')

