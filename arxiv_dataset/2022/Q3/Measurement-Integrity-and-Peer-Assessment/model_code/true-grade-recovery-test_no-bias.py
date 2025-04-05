"""
Script for running simulations to test estimation procedures for recovering the true grades of submissions in the continuous effort, unbiased agents setting.

@author: Noah Burrell <burrelln@umich.edu>
"""

import numpy as np
import json

from setup import initialize_student_list, shuffle_students, initialize_submission_list
from grading import assign_grades, assign_graders, get_grading_dict

from mechanisms.baselines import mean_squared_error
from mechanisms.parametric_mse import mse_p_mechanism

from evaluation import true_grade_mse
from graphing import plot_estimation_mses

import warnings

def run_simulation(num_iterations, num_students, continuous_effort=True, bias=False): 
    """
    Iteratively simulates one assignment, estimating true grades according to each of 3 estimation procedures and recording evaluation metrics.

    Parameters
    ----------
    num_iterations : int.
                     The number of times to simulate one assignment.
    num_students : int.
                   The number of students to include in the agent population.
    continuous_effort : bool, optional.
                        Determines whether agents should grade based on continuous or binary effort. The default is True and will not work as expected if set to False.
    bias : bool, optional.
           Determines whether agents should have bias in grading. The default is False.

    Returns
    -------
    score_dict : dict.
                 score_dict maps the names of estimation procedures to dicts that map the string "MSE Scores" to a list of scores.
                 { 
                     Estimation-Procedure-Name (str): { "MSE Scores": [ score (float) ] }
                }
                 Estimation Procedures are: 
                            - "Consensus-Grade"
                            - "Procedure-NB"
    """
    score_dict = {
            "Consensus-Grade": {},
            "Procedure-NB": {}
        }
    
    consensus_mse_scores = []
    procedure_nb_mse_scores = []
    
    for i in range(num_iterations):
        """
        Simulating a "semester"
        """
        students = initialize_student_list(num_students, num_students)
        shuffle_students(students)
            
        """
        Simulating a single assignment
        """
        submissions = initialize_submission_list(students, 0)
        grader_dict = assign_graders(students, submissions, 4)
        grading_dict = get_grading_dict(grader_dict)
        
        #Here is where you can change the number of draws an active grader gets
        assign_grades(grading_dict, 3, 0, continuous_effort, bias)
        
        true_scores = np.zeros(100)
        for submission in submissions:
            true_scores[submission.student_id] = submission.true_grade
            
        mse_scores = np.zeros(100)    
        mse_dict = mean_squared_error(grader_dict)
        for sid, score in mse_dict.items():
            mse_scores[sid] = score
            
        mu = 7
        gamma = 1/2.1
        
        unbiased_scores = np.zeros(100)  
        unbiased_score_dict, unbiased_reliability, zero_list = mse_p_mechanism(grader_dict, students, 0, mu, gamma, False)
        for sid, score in unbiased_score_dict.items():
            unbiased_scores[sid] = score
            
        mse_score = true_grade_mse(true_scores, mse_scores)
        consensus_mse_scores.append(mse_score)
        
        utru_score = true_grade_mse(true_scores, unbiased_scores)
        procedure_nb_mse_scores.append(utru_score)
        
    score_dict["Consensus-Grade"]["MSE Scores"] = consensus_mse_scores
    score_dict["Procedure-NB"]["MSE Scores"] = procedure_nb_mse_scores
    
    return score_dict

def simulate(filename):
    """
    Calls run_simlation for 1000 assignments with 100 students in the agent population.
    
    Saves a file containing the results of the experiment and generates and saves a plot of those results.
    Results are saved as filename.json in the ./results directory.
    Plots are saved as filename.pdf in the ./figures directory.

    Parameters
    ----------
    filename : str.
               The filename used to save the .json file and .pdf plot associated with the experiment.

    Returns
    -------
    None.

    """
    
    results = run_simulation(1000, 100)
    
    """
    Export JSON file of simulation data to results directory
    """
    json_file = "results/" + filename + ".json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    """
    Graphing the results in the figures directory
    """
    plot_estimation_mses(results, filename)

if __name__ == "__main__":
    
    """
    Simulations are controlled and run from here.
    """
    #Supress Warnings in console
    warnings.filterwarnings("ignore")
    
    """
    Change the filename before running a simulation to prevent overwriting previous results
    """
    filename = "true-grade-recovery_no-bias-filename"
    
    """
    The function below runs the experiment.
    """
    simulate(filename)