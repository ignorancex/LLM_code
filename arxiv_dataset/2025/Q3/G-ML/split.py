"""
Created on Mon Feb 03 2025, 15:33:47

@author: Amoy Ashesh
"""
from sklearn.model_selection import train_test_split
from parameters import training, algs

def split(data):
    """
    This function splits the dataset into training and testing datasets.
    
    Parameters:
    -----------
    data : pandas.core.frame.DataFrame
        Marked dataset.

    Returns:
    --------
    X_train: pandas.core.frame.DataFrame
        Training features.
    X_test: 
        Testing features. 
    y_train:
        Training targets. 
    y_test:
        Testing targets.
    """
    X = data.drop(columns=training.drop_columns)
    y = data[training.target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=training.test_split, random_state=training.training_random_state)
    return X_train, X_test, y_train, y_test