"""
Implementation of the parametric MSE mechanism and the EM procedure used to estimate the parameters of the parametric model (model PG_1 from Piech et al. 2013).

@author: Noah Burrell <burrelln@umich.edu>
"""

from math import sqrt

import numpy as np
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import gamma as gamma_distribution    

def mse_p_mechanism(grader_dict, student_list, assignment_num, mu, gamma, bias=True, bias_correct=False):
    """
    Computes payments for students according to the MSE_P mechanism.   
    
    Prints a warning if the EM estimation procedure does not converge.
    
    Returns the estimated parameters.

    Parameters
    ----------
    grader_dict : dict.
                  Maps a Submission object to a list of graders (Student objects).
    student_list : list of Student objects.
                   The population of students/graders.
    assignment_num : int.
                     Unique identifier of the assignment for which payments are being computed.
    mu : float.
        The mean of the normal approximation of the distribution of true grades.
    gamma : float.
        The precision (i.e. the inverse of the variance) of the normal approximation of the distribution of true grades.
    bias : bool, optional.
        Indicates whether agents have bias, and therefore whether bias parameters should be estimated. The default is True.

    Returns
    -------
    scores : dict
             {submission.student_id: estimated grade}
    reliability : dict.
             {student_id: estimated reliability}
    biases : dict.
             {student_id: estimated bias}

    """
    biases, reliability, scores, iteration = em_estimate_parameters(grader_dict, student_list, assignment_num, mu, gamma, bias)
    
    if not iteration < 1000:
        print("EM estimation procedure did not converge.")
        for student in student_list:
            student.payment += 0
    
    else:
        for student in student_list:
            tasks = []
            reports = []
            ground_truth = []
            if bias_correct:
                b = biases[student.id]
            else:
                b = 0
            
            for task, report in student.grades[assignment_num].items():
                tasks.append(task)
                reports.append(report - b)
                ground_truth.append(scores[task])
                
            student.payment -= mse(ground_truth, reports)
            
    return scores, reliability, biases

def em_estimate_parameters(grader_dict, student_list, assignment_num, mu, gamma, include_bias=False):
    """
    Estimates parametric model parameters using EM-style algorithm with Bayesian updating. 

    Parameters
    ----------
    grader_dict : dict.
                  Maps a Submission object to a list of graders (Student objects).
    student_list : list of Student objects.
                   The population of students/graders.
    assignment_num : int.
                     Unique identifier of the assignment for which payments are being computed.
    mu : float.
        The mean of the normal approximation of the distribution of true grades.
    gamma : float.
        The precision (i.e. the inverse of the variance) of the normal approximation of the distribution of true grades.
    include_bias : bool, optional.
        Indicates whether agents have bias, and therefore whether bias parameters should be estimated. The default is False.

    Returns
    -------
    scores : dict
             {submission.student_id: estimated grade}
    reliability : dict.
             {student_id: estimated reliability}
    biases : dict.
             {student_id: estimated bias}
             All zeros when bias==False.
    iteration : int.
                The total number of iterations of the EM process.
                Value indicates either that the score estimates conveged or that the score estimates did not converge and the estimation was stopped after 1000 iterations.
    """
    
    biases = {student.id: 0 for student in student_list}
    reliability = {student.id: (2*gamma)for student in student_list} 
    scores = {submission.student_id: int(round(mu)) for submission in grader_dict.keys()}
    
    new_scores = np.zeros(len(scores))
    old_scores = np.ones(len(scores))
    
    iteration = 0
    termination = 0.0001
    
    score = np.linalg.norm((old_scores - new_scores))
    
    while score > termination and iteration < 1000:
        
        old_scores_dict = scores.copy()
    
        #One Iteration of EM
        for submission in grader_dict.keys():
        #First compute the scores
            graders = list(submission.grades.keys())
            
            numerator_list = [sqrt(reliability[g])*(submission.grades[g] - biases[g]) for g in graders]
            denominator_list = [sqrt(reliability[g]) for g in graders]
            
            numerator_sum = sum(numerator_list)
            denominator_sum = sum(denominator_list)
            
            numerator = sqrt(gamma)*mu + numerator_sum
            denominator = sqrt(gamma) + denominator_sum
            
            scores[submission.student_id] = numerator/denominator
        
        if include_bias:
            for student in student_list:
            #Then compute the bias
                """
                BAYESIAN UPDATING: Conjugate prior is a Normal distirbution.
                """
                #prior_mu = 0
                prior_tau = 1
                
                samples = [(s - scores[num]) for num, s in student.grades[assignment_num].items()]
                sample_sum = sum(samples)
                
                n = len(samples)
                
                tau = reliability[student.id]
                
                posterior_tau = prior_tau + n*tau
                posterior_mu = (tau*sample_sum)/posterior_tau

                biases[student.id] = posterior_mu
                
            
        for student in student_list:
        #Then compute the reliability
            """
            BAYESIAN UPDATING: Conjugate Prior is a Gamma distribution
            """
            prior_a = 10.0/1.05
            prior_B = 10.0
            
            residuals = [ (s - (scores[num] + biases[student.id]))**2 for num, s in student.grades[assignment_num].items()]
            n = len(residuals)
            residual_sum = sum(residuals)
            
            posterior_a = prior_a + n/2.0
            posterior_B = prior_B + residual_sum/2.0
            
            posterior_theta = 1.0 / posterior_B
        
            score = gamma_distribution.mean(a=posterior_a, scale=posterior_theta)

            reliability[student.id] = score
        
        idx = 0
        for sid, score in scores.items():
            old_score = old_scores_dict[sid]
            old_scores[idx] = old_score
            new_scores[idx] = score
            idx += 1
            
        score = np.linalg.norm((old_scores - new_scores))
        
        iteration += 1
        
    return biases, reliability, scores, iteration