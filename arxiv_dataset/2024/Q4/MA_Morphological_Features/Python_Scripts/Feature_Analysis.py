# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 17:10:13 2024

Classify data using LDA and RF and display model evaluation using classification report

@author: Henry
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, classification_report, 
    roc_curve, roc_auc_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from warnings import simplefilter

class CineFeatureAnalysis:
    def __init__(self, root_dir, features_csv):
        self.root_dir = root_dir
        self.features_csv = features_csv
        self.rf_tuning = True
        self.analysis = 'ALL'  # ALL, LDA, RF
        self.n_estimators = None
        self.max_features = None
        self.max_depth = None
        self.max_leaf_nodes = None
        self.data_scaled, self.y, self.identifiers = self.load_and_preprocess_data()

    def load_and_preprocess_data(self):
        """Load and preprocess data, including scaling."""
        # Load feature data
        df = pd.read_csv(self.features_csv)

        identifiers = df['Assession_Number'].tolist()
        y = np.asarray(df['class'])

        NoVD_count = np.sum(y == 0)
        Pathology_count = np.sum(y == 1)
        print(f'Number of cases with No Valvular Disease: {NoVD_count}, with Pathology: {Pathology_count}.')

        df_1 = df.drop(columns=['Unnamed: 0', 'Assession_Number', 'class'])

        # Z-normalization
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df_1.values)

        return data_scaled, y, identifiers

    def split_test_set(self):
        """Select 10 cases per class as an individual test set."""
        NoVD_test, MR_test, selected_idx = [], [], []
        NoVD_identifiers_individual, MR_identifiers_individual = [], []

        for idx, status in enumerate(self.y):
            if status == 0 and len(NoVD_test) < 10:
                NoVD_test.append(self.data_scaled[idx])
                selected_idx.append(idx)
                NoVD_identifiers_individual.append(self.identifiers[idx])
            elif status == 1 and len(MR_test) < 10:
                MR_test.append(self.data_scaled[idx])
                selected_idx.append(idx)
                MR_identifiers_individual.append(self.identifiers[idx])

        data_test_individual = np.asarray(NoVD_test + MR_test)
        y_test_individual = np.asarray([0] * len(NoVD_test) + [1] * len(MR_test))
        identifiers_individual = NoVD_identifiers_individual + MR_identifiers_individual

        # Delete the selected cases from the data and label
        self.data_scaled = np.delete(self.data_scaled, selected_idx, axis=0)
        self.y = np.delete(self.y, selected_idx, axis=0)

        return data_test_individual, y_test_individual, identifiers_individual

    def tune_random_forest(self):
        """Hyperparameter tuning for Random Forest using GridSearchCV."""
        if self.rf_tuning:
            param_grid = { 
                'n_estimators': [100, 150, 200], 
                'max_features': ['sqrt', 'log2'], 
                'max_depth': [6, 9, 12, 15], 
                'max_leaf_nodes': [6, 9, 12, 15]
            }

            grid_search = GridSearchCV(RandomForestClassifier(), param_grid=param_grid)
            grid_search.fit(self.data_scaled, self.y)

            best_params = grid_search.best_params_
            print(f'Best parameters found: {best_params}')

            self.n_estimators = best_params['n_estimators']
            self.max_features = best_params['max_features']
            self.max_depth = best_params['max_depth']
            self.max_leaf_nodes = best_params['max_leaf_nodes']

    def evaluate_model(self, model, data_test_individual, y_test_individual):
        """Evaluate the model and generate plots for confusion matrix and ROC AUC."""
        y_pred = model.predict(data_test_individual)
        
        print('----------Classification Report on the Individual Test Set----------')
        print(classification_report(y_test_individual, y_pred, digits=2))

        self.evaluation_plots(y_test_individual, y_pred)

    @staticmethod
    def evaluation_plots(y_true, y_pred):
        """Plot confusion matrix and ROC AUC."""
        # Plot confusion matrix
        cf_matrix = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(cf_matrix, index=['No VD', 'MR'], columns=['No VD', 'MR'])
        sns.set(font_scale=1.5)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Values')
        plt.ylabel('Actual Values')
        plt.show()

        # Plot ROC AUC
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)
        plt.figure(figsize=(10, 7))
        plt.plot(fpr, tpr, color='green', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.show()

    def run_lda(self, data_test_individual, y_test_individual):
        """Perform Linear Discriminant Analysis (LDA) with 5-fold cross-validation."""
        lda = LinearDiscriminantAnalysis(store_covariance=True)
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)

        metrics = {
            'acc_train': [], 'acc_test': [],
            'specificity_train': [], 'specificity_test': [],
            'sensitivity_train': [], 'sensitivity_test': [],
            'f1_train': [], 'f1_test': [],
            'auc_train': [], 'auc_test': []
        }

        for train_index, test_index in kfold.split(self.data_scaled, self.y):
            X_train, X_test = self.data_scaled[train_index], self.data_scaled[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            lda.fit(X_train, y_train)
            yhat_train, yhat_test = lda.predict(X_train), lda.predict(X_test)

            metrics['acc_train'].append(accuracy_score(y_train, yhat_train))
            metrics['acc_test'].append(accuracy_score(y_test, yhat_test))
            metrics['specificity_train'].append(recall_score(y_train, yhat_train, pos_label=0))
            metrics['specificity_test'].append(recall_score(y_test, yhat_test, pos_label=0))
            metrics['sensitivity_train'].append(recall_score(y_train, yhat_train, pos_label=1))
            metrics['sensitivity_test'].append(recall_score(y_test, yhat_test, pos_label=1))
            metrics['f1_train'].append(f1_score(y_train, yhat_train))
            metrics['f1_test'].append(f1_score(y_test, yhat_test))
            metrics['auc_train'].append(roc_auc_score(y_train, yhat_train))
            metrics['auc_test'].append(roc_auc_score(y_test, yhat_test))

        lda.fit(self.data_scaled, self.y)
        self.evaluate_model(lda, data_test_individual, y_test_individual)

        # Print cross-validation results
        for metric_name, metric_values in metrics.items():
            metric_mean, metric_std = np.mean(metric_values), np.std(metric_values)
            print(f'{metric_name.replace("_", " ").title()} for 5-fold CV: {metric_mean:.2f} +/- {metric_std:.2f}')

    def run_random_forest(self, data_test_individual, y_test_individual):
        """Perform Random Forest Classification with 5-fold cross-validation."""
        rf = RandomForestClassifier(
            max_depth=self.max_depth, max_leaf_nodes=self.max_leaf_nodes,
            n_estimators=self.n_estimators, max_features=self.max_features,
            bootstrap=True, random_state=42, n_jobs=-1
        )
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)

        metrics = {
            'acc_train': [], 'acc_test': [],
            'specificity_train': [], 'specificity_test': [],
            'sensitivity_train': [], 'sensitivity_test': [],
            'f1_train': [], 'f1_test': [],
            'auc_train': [], 'auc_test': []
        }

        for train_index, test_index in kfold.split(self.data_scaled, self.y):
            X_train, X_test = self.data_scaled[train_index], self.data_scaled[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            rf.fit(X_train, y_train)
            yhat_train, yhat_test = rf.predict(X_train), rf.predict(X_test)

            metrics['acc_train'].append(accuracy_score(y_train, yhat_train))
            metrics['acc_test'].append(accuracy_score(y_test, yhat_test))
            metrics['specificity_train'].append(recall_score(y_train, yhat_train, pos_label=0))
            metrics['specificity_test'].append(recall_score(y_test, yhat_test, pos_label=0))
            metrics['sensitivity_train'].append(recall_score(y_train, yhat_train, pos_label=1))
            metrics['sensitivity_test'].append(recall_score(y_test, yhat_test, pos_label=1))
            metrics['f1_train'].append(f1_score(y_train, yhat_train))
            metrics['f1_test'].append(f1_score(y_test, yhat_test))
            metrics['auc_train'].append(roc_auc_score(y_train, yhat_train))
            metrics['auc_test'].append(roc_auc_score(y_test, yhat_test))

        rf.fit(self.data_scaled, self.y)
        self.evaluate_model(rf, data_test_individual, y_test_individual)

        # Print cross-validation results
        for metric_name, metric_values in metrics.items():
            metric_mean, metric_std = np.mean(metric_values), np.std(metric_values)
            print(f'{metric_name.replace("_", " ").title()} for 5-fold CV: {metric_mean:.2f} +/- {metric_std:.2f}')

    def run_analysis(self):
        """Main method to run the analysis."""
        data_test_individual, y_test_individual, _ = self.split_test_set()

        if self.rf_tuning:
            self.tune_random_forest()

        if self.analysis == 'LDA' or self.analysis == 'ALL':
            self.run_lda(data_test_individual, y_test_individual)

        if self.analysis == 'RF' or self.analysis == 'ALL':
            self.run_random_forest(data_test_individual, y_test_individual)