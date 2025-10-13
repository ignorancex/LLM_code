"""
Created on Mon Feb 03 2025, 11:33:03

@author: Amoy Ashesh
"""
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

class instruct:
    load_data = True
    load_models = True

class file_paths:
    raw_data_file = 'data\\1725606664466O-result.fits'
    catalogue_file = 'data\\binary_catalog.fits'

class training:
    test_split = 0.2
    training_random_state = 42
    drop_columns = ['solution_id', 'designation', 'source_id', 'target']
    target_column = 'target'

class algorithms:
    def __init__(self):
        self.ml_algs = {
            'RFC':[], 
            'LR':[], 
            'SVM (RBF)':[], 
            'DTC':[], 
            'AdaBoost':[], 
            'KNN': [], 
            'NB':[], 
            'Bagging': []
        }
            
        self.models = [
            RandomForestClassifier(n_estimators=25, random_state=42),
            LogisticRegression(random_state=42),
            DecisionTreeClassifier(random_state=42),
            AdaBoostClassifier(random_state=42),
            KNeighborsClassifier(n_neighbors=5),
            GaussianNB(),
            BaggingClassifier(random_state=42)
        ]
       
    def train(self, X_train, y_train):
        for model in self.models:
            model.fit(X_train, y_train)
            
    def fill_predictions(self, X_test):
        for algo, model in zip(self.alg_dict.keys(), self.models):
            self.alg_dict[algo] = model.predict(X_test)