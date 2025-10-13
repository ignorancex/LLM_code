from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import MiniBatchKMeans
from sklearn.covariance import MinCovDet
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering

import time 
import numpy as np
import matplotlib.pyplot as plt
import parameters
import utils
import random
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
import numpy as np
import utils  # Assuming this is your own utility script for data handling
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# Extended common characteristics

# Train models and extract characteristics

class Algorithm:

    def __init__(self, algorithms, datasetloader):
        self.algorithms_=algorithms
        self.built_algorithms = {}
        self.topk_samples = {}
        self.characteristics={}
        self.X_train, self.X_test, self.y_train, self.y_test = datasetloader.get_data()
        
    def getAlgorithms(self):
        return self.built_algorithms
    def getCharactiristics(self):
        characteristics={}
        for name,model in self.built_algorithms.items():
            common_characteristics={}

            common_characteristics['Has Intercept']=(hasattr(model, 'intercept_'))
            common_characteristics['Has Regularization']=(hasattr(model, 'alpha'))
            common_characteristics['Has Iterations']=(hasattr(model, 'n_iter_') or hasattr(model, 'n_estimators'))
            common_characteristics['Has Distance Metric']=(name == 'K-NN')
            common_characteristics['Is Tree-based']=(name in ['Decision Tree', 'Random Forest', 'Gradient Boosting'])
            common_characteristics['Is Ensemble']=(name in ['Random Forest', 'Gradient Boosting'])
            common_characteristics['Uses Kernel']=(name == 'SVR')
            common_characteristics['Provides Feature Importance']=name in ['Decision Tree', 'Random Forest', 'Gradient Boosting']
            common_characteristics['Is Scalable']=name not in ['K-NN', 'SVR']
            common_characteristics['Has Many Hyperparameters']=name in ['Random Forest', 'Gradient Boosting', 'SVR']
            common_characteristics['Captures Non-linearity']=name not in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Elastic Net']
            common_characteristics['Sensitive to Outliers']= name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']
            common_characteristics['Supports Parallelization']=name in ['Random Forest', 'Gradient Boosting']
            common_characteristics['Supports Online Learning']=False  # Placeholder, none of the models here support online learning
            characteristics[name]=common_characteristics
        return characteristics
        
    def get_topk_samples(self, model, X, y, k, model_type='classification'):
        if model_type == 'classification':
            if hasattr(model, 'predict_proba'):
                scores = model.predict_proba(X)
                top_k_indices = np.argsort(np.max(scores, axis=1))[-k:]
            else:
                return None  # For models that don't support predict_proba
        elif model_type == 'regression':
            y_pred = model.predict(X)
            errors = np.abs(y - y_pred)
            top_k_indices = np.argsort(errors)[:k]
        else:
            return None  # Unsupported model type
        
        top_k_samples = X[top_k_indices]
        top_k_labels = y[top_k_indices]
        return top_k_samples,top_k_labels
    def predict_class(self, model, X, model_type='classification'):
        # Reshape if it's a 1D array
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        if model_type == 'classification':
            # Predict the class labels
            predicted_labels = model.predict(X)
            
           
        elif model_type == 'regression':
            # Predict the output values
            predicted_labels = model.predict(X)
        else:
            return None  # Unsupported model type
        
        return predicted_labels


    
    def getScores(self):
        return self.topk_samples

    def getFeatures(self):
        return self.X_train
    

    def Train(self, k,model_type):
        for name, params in self.algorithms_.items():
            # Classification models
            if name in ['RandomForestClassifier', 'GradientBoostingClassifier', 'LogisticRegression', 'SVC', 'KNeighborsClassifier', 'DecisionTreeClassifier', 'GaussianNB', 'MLPClassifier']:
                model = eval(name)(**params)
                model.fit(self.X_train, self.y_train)
                self.built_algorithms[name]=model

                #self.topk_samples[name] = self.get_topk_samples(model, self.X_train,self.y_train, k,model_type)

            # Regression models
            elif name in ['RandomForestRegressor', 'GradientBoostingRegressor', 'LinearRegression', 'Ridge', 'Lasso', 'SVR', 'KNeighborsRegressor', 'DecisionTreeRegressor', 'MLPRegressor']:
                model = eval(name)(**params)
                model.fit(self.X_train, self.y_train)
                self.built_algorithms[name]=model

                self.topk_samples[name] = self.get_topk_samples(model, self.X_train,self.y_train, k,model_type)  # Modify this line if you have a different way to get top-k samples for regression
            else:
                model = eval(name)(**params)
                model.fit(self.X_train, self.y_train)
                self.built_algorithms[name]=model

                self.topk_samples[name] = self.get_topk_samples(model, self.X_train,self.y_train, k,model_type)  # Modify this line if you have a different way to get top-k samples for regression
        self.characteristics=self.getCharactiristics()
    def Evaluate(self):
        evaluation_metrics = {}
        for name, params in self.algorithms_.items():
            if name in ['RandomForestClassifier', 'GradientBoostingClassifier', 'LogisticRegression', 'SVC', 'KNeighborsClassifier', 'DecisionTreeClassifier', 'GaussianNB', 'MLPClassifier']:
                model = eval(name)(**params)
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred, average='weighted')  # Change average parameter as needed
                recall = recall_score(self.y_test, y_pred, average='weighted')  # Change average parameter as needed
                f1 = f1_score(self.y_test, y_pred, average='weighted')  # Change average parameter as needed
                
                evaluation_metrics[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
        return evaluation_metrics
    def Evaluate1(self,X_train,y_train,X_test,y_test):
        evaluation_metrics = {}
        for name, params in self.algorithms_.items():
            if name in ['RandomForestClassifier', 'GradientBoostingClassifier', 'LogisticRegression', 'SVC', 'KNeighborsClassifier', 'DecisionTreeClassifier', 'GaussianNB', 'MLPClassifier']:
                model = eval(name)(**params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')  # Change average parameter as needed
                recall = recall_score(y_test, y_pred, average='weighted')  # Change average parameter as needed
                f1 = f1_score(y_test, y_pred, average='weighted')  # Change average parameter as needed
                
                evaluation_metrics[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
        return evaluation_metrics
    def getTopK(self):
        return self.topk_samples
