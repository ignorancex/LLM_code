"""
Evaluation metrics that are used to measure the performance of the mechanisms at various tasks.

@author: Noah Burrell <burrelln@umich.edu>
"""

from itertools import combinations
from numpy import isnan
from pandas import DataFrame, qcut
from sklearn.metrics import roc_auc_score, mean_squared_error
from scipy.stats import kendalltau, pearsonr
from statistics import mean
from sys import maxsize

def roc_auc(student_list):
    """
    Computes the ROC AUC score for classifying agents as "active" or "passive" based on their payments.
    Payments were assigned according to some mechanism for completing peer grading tasks over the course of a simulated semester.

    Parameters
    ----------
    student_list : list of Student objects.

    Returns
    -------
    score : float. 
            ROC AUC score.

    """
    true = []
    scores = []
    
    minsize = -maxsize - 1
    
    for student in student_list:
        classification = 0
        if student.type == "active":
            classification = 1
        true.append(classification)
        
        p = student.payment
        
        """
        Formatting to ensure all payments are valid inputs to the roc_auc_score function from sklearn.metrics.
        """
        if p > maxsize:
            p = maxsize
        if p < minsize:
            p = minsize
        if isnan(p):
            #print("Payment is nan")
            p = 0
        
        scores.append(p)
        
    score = roc_auc_score(true, scores)
    return score

def roc_auc_strategic(student_list):
    """
    Computes the ROC AUC score for classifying agents as truthful (strategy="TRUTH") or strategic (strategy="NOISE", "FIX-BIAS", "MERGE", "PRIOR", "ALL10", or "HEDGE") based on their payments.
    Payments were assigned according to some mechanism for completing peer grading tasks over the course of a simulated semester.

    Parameters
    ----------
    student_list : list of StrategicStudent objects.

    Returns
    -------
    score : float.
            ROC AUC score.

    """
    true = []
    scores = []
    
    minsize = -maxsize - 1
    
    for student in student_list:
        classification = 0
        if student.strategy == "TRUTH":
            classification = 1
        true.append(classification)
        
        p = student.payment
        
        """
        Formatting to ensure all payments are valid inputs to the roc_auc_score function from sklearn.metrics.
        """
        if p > maxsize:
            p = maxsize
        if p < minsize:
            p = minsize
        if isnan(p):
            #print("Payment is nan")
            p = 0
        
        scores.append(p)
        
    score = roc_auc_score(true, scores)
    return score

def aucs_mse(student_list, include_q = True):
    """
    Computes AUC scores (binary and quinary AUC) for classifying a student as above or below the median in terms of grading ability (i.e. MSE in grading tasks) according to their payment.
    
    Parameters
    ----------
    student_list : list of Student objects.
        
    include_q : bool, optional
        Indicates wheter to calculate quinary AUC (see paper) in addition to binary.
        Default is True

    Returns
    -------
    binary_score : int
        Binary AUC (see paper)
    quinary_score : int
        Quinary AUC (see paper)
        Zero if include_q is False
    """
    payments = []
    mses = []
    
    minsize = -maxsize - 1
    
    for student in student_list:
        
        p = student.payment
        
        """
        Formatting to ensure all payments are valid inputs to the roc_auc_score function from sklearn.metrics.
        """
        if p > maxsize:
            p = maxsize
        if p < minsize:
            p = minsize
        if isnan(p):
            #print("Payment is nan")
            p = 0
        payments.append(p)
        
        mses.append(-1*student.mse)
    
    d = {"Payment": payments, "MSE": mses}

    df = DataFrame(data=d)
    
    df['Binary'] = qcut(df['MSE'], 2, labels = False)
    if include_q:
        df['Quinary'] = qcut(df['MSE'], 5, labels = False)
    
    binary_score = roc_auc_score(df['Binary'], df['Payment'])
    
    if include_q:
        quinary_aucs = []
        for (i,j) in combinations(range(5), 2):
            q_df = df.loc[((df['Quinary'] == i) | (df['Quinary'] == j)), ['Payment', 'Quinary']]
            pairwise = roc_auc_score(q_df['Quinary'], q_df['Payment'])
            quinary_aucs.append(pairwise)
        quinary_score = mean(quinary_aucs)
    else:
        quinary_score = 0
        
    return binary_score, quinary_score

def correlation_mse(student_list):
    """
    Computes the Pearson correlation coefficient (rho) between the mse of agent reports and their payments.
    Payments were assigned according to some mechanism for completing peer grading tasks over the course of a simulated semester.
                                                       
    Parameters
    ----------
    student_list : A list of Student objects.
        
    Returns
    -------
    rho : float.
          Pearson correlation coefficient.

    """
    rho = 0 
    
    true = []
    scores = []
    
    minsize = -maxsize - 1
    
    for student in student_list:
        score = -1*student.mse
        true.append(score)
        
        p = student.payment
        
        """
        Formatting to ensure all payments are valid inputs to the kendalltau function from scipy.stats.
        """
        if p > maxsize:
            p = maxsize
        if p < minsize:
            p = minsize
        if isnan(p):
            #print("Payment is nan")
            p = 0
        
        scores.append(p)
    
    rho, p_value = pearsonr(true, scores)
    
    return rho
    

def kendall_tau(student_list):
    """
    Computes the Kendall rank correlation coefficient (Kendall's tau_B) between the ranking of agents according to the continuous effort parameter (lam) and the ranking of agents according to their payments.
    Payments were assigned according to some mechanism for completing peer grading tasks over the course of a simulated semester.
                                                       
    Parameters
    ----------
    student_list : A list of Student objects.
        
    Returns
    -------
    tau : float.
          Kendall rank correlation coefficient.

    """
    tau = 0 
    
    true = []
    scores = []
    
    minsize = -maxsize - 1
    
    for student in student_list:
        score = student.lam
        true.append(score)
        
        p = student.payment
        
        """
        Formatting to ensure all payments are valid inputs to the kendalltau function from scipy.stats.
        """
        if p > maxsize:
            p = maxsize
        if p < minsize:
            p = minsize
        if isnan(p):
            #print("Payment is nan")
            p = 0
        
        scores.append(p)
    
    tau, p_value = kendalltau(true, scores)
    
    return tau

def kendall_tau_mse(student_list):
    """
    Computes the Kendall rank correlation coefficient (Kendall's tau_B) between the ranking of agents according to the mse of their reports and the ranking of agents according to their payments.
    Payments were assigned according to some mechanism for completing peer grading tasks over the course of a simulated semester.
                                                       
    Parameters
    ----------
    student_list : A list of Student objects.
        
    Returns
    -------
    tau : float.
          Kendall rank correlation coefficient.

    """
    tau = 0 
    
    true = []
    scores = []
    
    minsize = -maxsize - 1
    
    for student in student_list:
        score = -1*student.mse
        true.append(score)
        
        p = student.payment
        
        """
        Formatting to ensure all payments are valid inputs to the kendalltau function from scipy.stats.
        """
        if p > maxsize:
            p = maxsize
        if p < minsize:
            p = minsize
        if isnan(p):
            #print("Payment is nan")
            p = 0
        
        scores.append(p)
    
    tau, p_value = kendalltau(true, scores)
    
    return tau

def true_grade_mse(true_scores, computed_scores):
    """
    Computes the mean squared error (MSE) of estimates of the ground truth scores for some submissions.

    Parameters
    ----------
    true_scores : list of int 0-10. 
                  The ground truth scores for the submissions.
    computed_scores : list of float. 
                      The estimated scores for the submissions.

    Returns
    -------
    mse : float.
          The mean squared error of the computed scores.

    """
    return mean_squared_error(true_scores, computed_scores)