import os

from typing import Optional

import pandas as pd
import numpy as np
from sklearn import metrics

import matplotlib.pyplot as plt


def optimal_threshold(y_true, y_pred) -> float:
    """
    Calculate the optimal threshold for classification.

    This function calculates the optimal threshold for classification based on the Receiver Operating Characteristic (ROC) curve.
    It takes the true labels and predicted probabilities as input and returns the optimal threshold.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted probabilities.

    Returns:
        float: The optimal threshold for classification.
    """
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    return thresholds[np.argmax(tpr - fpr)]
    
    
def confusion_matrix(y_true, y_pred, threshold: float = 0.5,
                     output_folder: Optional[str] = None) -> None:
    """
    Compute and plot the confusion matrix for binary classification.

    Args:
        y_true (array-like): True labels of the binary classification.
        y_pred (array-like): Predicted labels of the binary classification.
        threshold (float, optional): Threshold value for converting predicted probabilities to binary labels.
            Defaults to 0.5.
        output_folder (str, optional): Path to the output folder where the confusion matrix plot
            will be saved as an image. If not provided, the plot will be displayed on the screen instead.

    Returns:
        None
    """
    y_pred = y_pred >= threshold
    cm = metrics.confusion_matrix(y_true, y_pred)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                          display_labels=np.asarray(['Non-cancerous', 'Cancerous']))
    disp.plot()
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, 'confusion_matrix.png')
        plt.savefig(output_file)
    else:
        plt.show()
    plt.close()
    

def classification_report(y_true, y_pred, threshold: float = 0.5,
                          output_folder: Optional[str] = None) -> None:
    """
    Generate a classification report based on the true labels and predicted labels.

    Args:
        y_true (array-like): The true labels.
        y_pred (array-like): The predicted labels.
        threshold (float, optional): The threshold value for classification. Defaults to 0.5.
        output_folder (str, optional): The folder path to save the classification report. If not provided,
            the classification report will be printed to the screen. Defaults to None.

    Returns:
        None
    """
    y_pred = y_pred >= threshold
    results = metrics.classification_report(y_true, y_pred, target_names=['Non-cancerous', 'Cancerous'],
                                            output_dict=True)
    
    df = pd.DataFrame(results).transpose()
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, 'classification_report.csv')
        df.to_csv(output_file)
    else:
        print(df)


def predictions_to_csv(target_values: np.ndarray, predicted_values: np.ndarray,
                       output_folder: str):
    """
    Save the predicted values and target values to a CSV file.

    Args:
        target_values (np.ndarray): The array of target values.
        predicted_values (np.ndarray): The array of predicted values.
        output_folder (str): The folder path where the CSV file will be saved.

    Returns:
        None
    """
    df = pd.DataFrame({'Prediction': predicted_values,
                       'Target': target_values})
    os.makedirs(output_folder, exist_ok=True)
    df.to_csv(os.path.join(output_folder, 'predictions.csv'))
    

def roc_curve(y_true, y_pred, output_folder: Optional[str] = None) -> None:
    """
    Compute and plot the Receiver Operating Characteristic (ROC) curve.

    Args:
        y_true (array-like): True binary labels.
        y_pred (array-like): Target scores, can either be probability estimates of the positive class,
            confidence values, or non-thresholded measure of decisions.
        output_folder (str, optional): Path to the output folder where the ROC curve plot will be saved.
            If not provided, the plot will be displayed on the screen instead.

    Returns:
        None
    """
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color='darkorange',
        lw=lw,
        label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc='lower right')
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, 'roc_curve.png')
        plt.savefig(output_file)
    else:
        plt.show()
    plt.close()
    
    
def pr_curve(y_true, y_pred, output_folder: Optional[str] = None) -> None:
    """
    Compute and plot the precision-recall curve.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted probabilities or scores.
        output_folder (str, optional): Path to the output folder where the plot will be saved.
            If not provided, the plot will be displayed on the screen instead. Defaults to None.

    Returns:
        None
    """
    prec, recall, _ = metrics.precision_recall_curve(y_true, y_pred, pos_label=1)
    avg_pr = metrics.average_precision_score(y_true, y_pred, pos_label=1)
    
    plt.figure()
    lw = 2
    plt.plot(
        recall,
        prec,
        color='darkorange',
        lw=lw,
        label='PR curve (average = %0.2f)' % avg_pr)
    plt.plot([1, 0], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall Curve')
    plt.legend(loc='lower right')
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, 'pr_curve.png')
        plt.savefig(output_file)
    else:
        plt.show()
    plt.close()
    

def det_curve(y_true, y_pred, output_folder: Optional[str] = None) -> None:
    """
    Plot the Detection Error Tradeoff (DET) curve.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        output_folder (Optional[str], optional): Path to the output folder where the DET curve plot will be saved.
            If not provided, the plot will be displayed on the screen instead. Defaults to None.
    
    Returns:
        None
    """
    fpr, fnr, _ = metrics.det_curve(y_true, y_pred)
    
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        fnr,
        color='darkorange',
        lw=lw,
        label='DET curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')
    plt.title('Detection Error Tradeoff (DET) curves')
    plt.legend(loc='upper right')
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, 'det_curve.png')
        plt.savefig(output_file)
    else:
        plt.show()
    plt.close()