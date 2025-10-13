from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer, fetch_openml
from sklearn.datasets import  load_diabetes, load_linnerud, fetch_california_housing, make_friedman1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import time
class DatasetLoader:
    
    def __init__(self, dataset_name=None, file_path=None):
        self.dataset_name = dataset_name
        self.file_path = file_path
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def load_data(self):
        if self.file_path:
            # Load custom dataset from a file
            df = pd.read_csv(self.file_path, encoding='ISO-8859-1')
            df=df.dropna()
            X = df.drop(df.columns[-1], axis=1)  # Replace 'target_column' with the actual target column name
            y = df[df.columns[-1]]  # Replace 'target_column' with the actual target column name
        else:
            # Load dataset from sklearn
            if self.dataset_name == 'iris':
                data = load_iris()
            elif self.dataset_name == 'digits':
                data = load_digits()
            elif self.dataset_name == 'wine':
                data = load_wine()
            elif self.dataset_name == 'breast_cancer':
                data = load_breast_cancer()
            elif self.dataset_name == 'mnist':
                data = fetch_openml('mnist_784')
            #elif self.dataset_name == 'boston':
                #data = load_boston()
            elif self.dataset_name == 'diabetes':
                data = load_diabetes()
            elif self.dataset_name == 'linnerud':
                data = load_linnerud()
            elif self.dataset_name == 'california_housing':
                data = fetch_california_housing()
            elif self.dataset_name == 'friedman1':
                X, y = make_friedman1(n_samples=1000, n_features=10, noise=0.1, random_state=42)
            else:
                print("Invalid dataset name.")
                return
            
            X = data.data
            y = data.target
        label_encoder = LabelEncoder()
        for col in X.columns:
            if (X[col].dtype == 'object')==True:  # Check if the column is of type 'object' (usually means it's a string)
                X[col] = label_encoder.fit_transform(X[col])
        y= label_encoder.fit_transform(y)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=round(time.time()))
    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test
