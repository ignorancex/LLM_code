import numpy as np

def average_l2_distance(y_true, y_pred):
    """
    Calculate the average L2 (Euclidean) distance between corresponding vectors in y_true and y_pred.
    
    Parameters:
    - y_true: List or array of ground truth vectors where each vector is represented as a list or array.
    - y_pred: List or array of predicted vectors where each vector is represented as a list or array.
    
    Returns:
    - avg_distance: Average L2 distance.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Check if the shapes of y_true and y_pred are the same
    if y_true.shape != y_pred.shape:
        raise ValueError("The shapes of y_true and y_pred must be the same.")
    
    # Calculate L2 distance between corresponding vectors
    l2_distances = np.linalg.norm(y_true - y_pred, axis=1)
    
    # Calculate average L2 distance
    avg_distance = np.mean(l2_distances)
    
    return avg_distance

def elementwise_average_distance(y_true, y_pred):
    """
    Calculate the element-wise average distance between each pair of vectors in y_true and y_pred.
    
    Parameters:
    - y_true: List or array of ground truth vectors where each vector is represented as a list or array.
    - y_pred: List or array of predicted vectors where each vector is represented as a list or array.
    
    Returns:
    - avg_distances: List of element-wise average distances.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Check if the shapes of y_true and y_pred are the same
    if y_true.shape != y_pred.shape:
        raise ValueError("The shapes of y_true and y_pred must be the same.")
    
    # Calculate element-wise distances between each pair of vectors
    distances = np.abs(y_true - y_pred)
    # print(distances)
    # Calculate average distance for each element position across all vectors
    avg_distances = np.mean(distances, axis=0)
    
    return avg_distances

def mean_squared_error(y_true, y_pred):
    """
    Calculate the Mean Squared Error (MSE) between observed and predicted values.
    
    Parameters:
    - y_true: List, array, or numpy vector of observed values.
    - y_pred: List, array, or numpy vector of predicted values.
    
    Returns:
    - MSE: Mean Squared Error.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Check if the shapes of y_true and y_pred are the same
    if y_true.shape != y_pred.shape:
        raise ValueError("The shapes of y_true and y_pred must be the same.")
    
    # Calculate squared errors element-wise
    squared_errors = np.square(y_true - y_pred)
    
    # Calculate mean squared error element-wise
    mse = np.mean(squared_errors, axis=tuple(range(1, y_true.ndim)))
    
    return mse

def average_values(arr):
    arr = np.array(arr)
    return np.mean(arr, axis=0)
