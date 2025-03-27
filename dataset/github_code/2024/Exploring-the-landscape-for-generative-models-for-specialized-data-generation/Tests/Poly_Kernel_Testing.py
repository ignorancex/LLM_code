from sklearn.svm import OneClassSVM
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import IsolationForest



class Poly_Kernel_Testing:
    
    def __init__(self, output_filepath, original_data_filepath,nu,degree):
        
        self.output_filepath = output_filepath
        self.original_data_fielpath = original_data_filepath
        self.nu = nu
        self.degree = degree
        self.data = pd.read_csv(original_data_filepath)
        self.output = pd.read_csv(output_filepath)


    def inliers(self):
        svm_model = OneClassSVM(kernel='poly', degree=self.degree, nu=self.nu)

        random_20_percent = self.data.sample(frac=0.2, random_state=42)  # Set random_state for reproducibility
        data_edited = self.data.drop(random_20_percent.index)
        svm_model.fit(data_edited) 

        predictions_set = []

        predictions1 = svm_model.predict(random_20_percent)
        random_20_percent['prediction'] = predictions1
        num_inliers_out = np.sum(predictions1 == 1)
        percentage_inliers_out = (num_inliers_out / len(predictions1)) * 100
        predictions_set.append(percentage_inliers_out)

        random_20_percent_1 = data_edited.sample(frac=0.2, random_state= 42)
        predictions2 = svm_model.predict(random_20_percent_1)
        random_20_percent_1['prediction'] = predictions2
        num_inliers_out = np.sum(predictions2 == 1)
        percentage_inliers_out = (num_inliers_out / len(predictions2)) * 100
        predictions_set.append(percentage_inliers_out)

        predictions3 = svm_model.predict(self.output)
        self.output['prediction'] = predictions3
        num_inliers_out = np.sum(predictions3 == 1)
        percentage_inliers_out = (num_inliers_out / len(predictions3)) * 100
        predictions_set.append(percentage_inliers_out)

        return predictions_set
    


    def draw(self,predictions):

        predictions_data = ['testing data', 'training data', 'output data']

        plt.figure(figsize=(10, 6))
        plt.bar(predictions_data, predictions, color='skyblue', edgecolor='black')
        plt.title('Bar Chart of Predictions Set', fontsize=15)
        plt.xlabel('Category', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()



