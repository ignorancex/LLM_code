"""
Script for running experiments to examine the potential for individual students to gain by deviating from truthful reporting using real grading data.

@author: Noah Burrell <burrelln@umich.edu>
"""

from numpy import ones
from statistics import mean, median, variance
import json
from random import randint, seed
from sys import maxsize

import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from grading import get_grading_dict

from mechanisms.baselines import mean_squared_error
from mechanisms.phi_divergence_pairing import phi_divergence_pairing_mechanism, parametric_phi_divergence_pairing_mechanism
from mechanisms.output_agreement import oa_mechanism
from mechanisms.parametric_mse import mse_p_mechanism
from mechanisms.peer_truth_serum import pts_mechanism

from load import load17, load19

import warnings

def run_simulation(strategy, mechanism, mechanism_param, semester, coarsen): 
    """
    Iteratively simulates semesters, scoring students according to a single mechanism, and recording the values of the relevant evaluation metrics.

    Parameters
    ----------
    strategy: str.
           The name of the strategy that the deviator will adopt.
           The list of all possible strategies can be found below in the code for running simulations.
    mechanism : str.
                The name of the mechanism to be used to score the students performance in the grading task.
                One of the following:
                    - "BASELINE"
                    - "OA"
                    - "Phi-DIV"
                    - "PTS"
                    - "MSE_P"
                    - "Phi-DIV_P"
    mechanism_param : str.
                      Denotes different versions of the same mechanism, e.g. the choice phi divergence used in the phi divergence pairing mechanism.
                      "0" for mechanisms that do not require such a parameter.
    semester : str
              Chooses a semester for which to load data.
              One of: "Spring 17", "Fall 17", "Spring 19", "Fall 19"
    coarsen : bool
             Set to true if data should be coarsened so that grades fall in the standard integer [0, 10] range.

    Returns
    -------
    score_dict : dict.
                score_dict maps the names of evaluation metrics to scores for those metrics.
                 { 
                    "Mean Gain": mean_gain (float),
                    "Median Gain": median_gain (float),
                    "Variance Gain": variance_gain (float)
                }
    """
    score_dict = {}
    
    deviator_gains = []
    
    print("        ", mechanism, mechanism_param)
    
    """
    Set semester-specific variables.
    """
    
    if semester == "Spring 17":
        assignment_list = list(range(1, 17))
        all_students, all_submissions = load17("Spring", coarsen, False)
        if coarsen:
            possible_grades = 11
            
            # MLE
            mu = 8.71
            sigma = 1.95
            gamma = 1/(sigma ** 2)
            
        else: 
            return
            
    elif semester == "Fall 17":
        assignment_list = list(range(1, 17))
        all_students, all_submissions = load17("Fall", coarsen, False)
        if coarsen:
            possible_grades = 11
            
            # MLE
            mu = 7.57
            sigma = 2.23
            gamma = 1/(sigma ** 2)
            
        else: 
            return
            
    elif semester == "Spring 19":
        assignment_list = list(range(1, 12)) + [13, 14]
        all_students, all_submissions = load19("Spring", coarsen, False)
        #include_q = True
        if coarsen:
            possible_grades = 11
            
            # MLE
            mu = 7.68
            sigma = 1.92
            gamma = 1/(sigma ** 2)
            
        else: 
            return
            
    elif semester == "Fall 19":
        assignment_list = list(range(1, 15))
        all_students, all_submissions = load19("Fall", coarsen, False)
        if coarsen:
            possible_grades = 11
            
            # MLE
            mu = 8.25
            sigma = 1.69
            gamma = 1/(sigma ** 2)
            
        else: 
            return
        
    else:
        print("Error -- Semester is specified incorrectly.")
        
    prior = mu
    
    #Records the number of payments each student receives.
    for student in all_students:
        student.num_graded = 0
        student.num_graded_initial = 0
        
    grader_dicts = {}
    grading_dicts = {}
    
    nonempty_assignments = []
    for assignment in assignment_list:
        """
        Considering a single assignment at a time.
        """
        submission_list = [sub for sub in all_submissions if sub.assignment_number == assignment]
        
        if len(submission_list) > 0:
            nonempty_assignments.append(assignment)
        
            grader_dict = {}
            for submission in submission_list:
                graders = []
                grader_ids = list(submission.grades.keys())
                for stu in all_students:
                    if stu.id in grader_ids:
                        graders.append(stu)
                        stu.num_graded += 1
                        stu.num_graded_initial += 1
                        
                grader_dict[submission] = graders
                grading_dict = get_grading_dict(grader_dict)
                
            grader_dicts[assignment] = grader_dict
            grading_dicts[assignment] = grading_dict
    
    for deviator in all_students:
        
        deviator_ranks = []
        random_seed = randint(~maxsize, maxsize)
            
        for iteration in range(2):
            seed(random_seed)
            if iteration == 1:
                """
                Change deviator reports to strategic reports for every submission on every assignment
                """
                deviator.truthful_grades = deviator.grades.copy()
                deviator.strategy = strategy
                
                for assignment_num in nonempty_assignments:
                    grading_dict = grading_dicts[assignment_num]
                    
                    if deviator in grading_dict.keys():
                        deviator_submissions = grading_dict[deviator]
                        
                        for submission in deviator_submissions:
                            signal = deviator.grades[assignment_num][submission.student_id]
                            grade = deviator.report(signal, prior)
                        
                            deviator.grades[assignment_num][submission.student_id] = grade
                            submission.grades[deviator.id] = grade
                            
            #necessary for PTS
            H = ones(possible_grades)
            
            for assignment in nonempty_assignments:
                """
                Run the Mechanism
                """
                students = [student for student in all_students if assignment in student.grades.keys()]
                
                grader_dict = grader_dicts[assignment]
                grading_dict = grading_dicts[assignment]
                
                """
                Non-Parametric Mechanisms
                """
                    
                if mechanism == "BASELINE":
                    mean_squared_error(grader_dict)
                        
                elif mechanism == "OA":
                    oa_mechanism(grader_dict)
                        
                elif mechanism == "Phi-DIV":
                    phi_divergence_pairing_mechanism(grader_dict, mechanism_param)
                    
                elif mechanism == "PTS":
                    H = pts_mechanism(grader_dict, H)
                    
                    """
                Parametric Mechanisms
                """
                
                elif mechanism == "MSE_P":
                    mse_p_mechanism(grader_dict, students, assignment, mu, gamma, True)
                    
                elif mechanism == "Phi-DIV_P":
                    parametric_phi_divergence_pairing_mechanism(grader_dict, students, assignment, mu, gamma, False, mechanism_param)
                    
                else:
                    print("Error: The given mechanism name does not match any of the options.")
            
            for stu in all_students:
                num = stu.num_graded
                stu.payment *= (1/num)
            
            '''
            Calculate the rank of the deviator (according to the number of payments that are >= than theirs)
            '''
            val = deviator.payment
            rank = 0
            for student in all_students:
                if student.payment >= val:
                    rank += 1
            deviator_ranks.append(rank)
            
            '''
            Reset the payments for all students.
            '''
            for student in all_students:
                student.num_graded = student.num_graded_initial
                student.payment = 0
                    
        '''
        Reseed the randomness for initializing next semester
        '''        
        seed()
              
        deviator_gain = deviator_ranks[0] - deviator_ranks[1]
        deviator_gains.append(deviator_gain)
        
        deviator.strategy = "TRUTH"
        deviator.grades = deviator.truthful_grades.copy()
        
    mean_gain = mean(deviator_gains)
    score_dict["Mean Gain"] = mean_gain
    
    median_gain = median(deviator_gains)
    score_dict["Median Gain"] = median_gain
    
    variance_gain = variance(deviator_gains, mean_gain)
    score_dict["Variance Gain"] = variance_gain
    
    return score_dict


def compare_mechanisms(strategy, mechanisms, semester, coarsen):
    """
    Iterates over a list of mechanisms, calling run_simulation for each one.

    Parameters
    ----------
    strategy: str.
           The name of the strategy that the deviator will adopt.
           The list of all possible strategies can be found below in the code for running simulations.
    mechanisms : list of 2-tuples of strings. 
                 Describes the mechanisms to be included in the form ("mechanism_name", "mechanism_param").
                 The complete list of possible mechanisms and associated params can be found below in the code for running simulations.
    semester : str
              Chooses a semester for which to load data.
              One of: "Spring 17", "Fall 17", "Spring 19", "Fall 19"
    coarsen : bool
             Set to true if data should be coarsened so that grades fall in the standard integer [0, 10] range.

    Returns
    -------
    eval_dict : dict.
                Maps the string "mechanism_name: mechanism_param" to a score_dict (returned from the call to run_simulation).

    """
    eval_dict = {}
    
    for mechanism, param in mechanisms:
        
        score_dict = run_simulation(strategy, mechanism, param, semester, coarsen)
        
        key = mechanism + ": " + param 
        eval_dict[key] = score_dict
    
    return eval_dict

def simulate(strategies, mechanisms, filename, semester, coarsen):
    """
    Calls compare_mechanisms iteravely for each strategy, varying the number of strategic graders.
    
    Saves a file containing the results of the experiment and generates and saves a plot of those results.
    Results are saved as filename.json in the ./results directory.
    Plots are saved as filename-mean_gain-*MECHANISM*.pdf and filename-variance_gain-*MECHANISM*.pdf in the ./figures directory.

    Parameters
    ----------
    strategies : list of strings.
                 Describes the strategies ot be included.
                 The list of all possible strategies can be found below in the code for running simulations.
    mechanisms : list of 2-tuples of strings. 
                 Describes the mechanisms to be included in the form ("mechanism_name", "mechanism_param").
                 The complete list of possible mechanisms and associated params can be found below in the code for running simulations.
    filename : str.
               The filename used to save the .json file and .pdf plot associated with the experiment.
    semester : str
              Chooses a semester for which to load data.
              One of: "Spring 17", "Fall 17", "Spring 19", "Fall 19"
    coarsen : bool
             Set to true if data should be coarsened so that grades fall in the standard integer [0, 10] range.
    
    Returns
    -------
    None.

    """
    results = {}
    
    for strategy in strategies:
        result = {}
        print("Working on experiments for the following strategy:", strategy)
        
        evals = compare_mechanisms(strategy, mechanisms, semester, coarsen)
        result[1] = evals
        
        results[strategy] = result
        
        json_file = "../results/" + filename + strategy + ".json"
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        
    json_file_all = "../results/" + filename + ".json"
    
    """
    Export JSON file of simulation data to results directory
    """
    
    with open(json_file_all, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    """
    Graphing the results in the figures directory
    
    mean_gain_filename = filename + "-mean_gain"
    plot_mean_rank_changes(results, mean_gain_filename)
    
    variance_gain_filename = filename + "-variance_gain"
    plot_variance_rank_changes(results, variance_gain_filename)
    """

if __name__ == "__main__":
    
    """
    Simulations are controlled and run from here.
    """
    
    #Supress Warnings in console
    warnings.filterwarnings("ignore")
    
    """
    Uncomment the mechanisms to be included in an experiment.
    """
    mechanisms = [
        
            #NON-PARAMETRIC MECHANISMS
            
            ("BASELINE", "MSE"),
            ("OA", "0"),
            ("Phi-DIV", "CHI_SQUARED"),
            ("Phi-DIV", "KL"),
            ("Phi-DIV", "SQUARED_HELLINGER"),
            ("Phi-DIV", "TVD"),
            ("PTS", "0"),
            
            #PARAMETRIC MECHANISMS
            
            ("MSE_P", "0"),
            ("Phi-DIV_P", "CHI_SQUARED"),
            ("Phi-DIV_P", "KL"),
            ("Phi-DIV_P", "SQUARED_HELLINGER"),
            ("Phi-DIV_P", "TVD"),
            
        ]
    
    """ 
    Uncomment the strategies to be included in an experiment.
    """
    
    strategies = [
                        
            "NOISE",
            "MERGE",
            "PRIOR", 
            "ALL10", 
            "HEDGE"
        ]
    
    semesters = ["Spring 17", "Fall 17", "Spring 19", "Fall 19"]
    
    """
    Select which semester should be used in the experiment by changing the index below.
    """
    semester = semesters[0]
    semester_name = semester.replace(" ", "")
    coarsen = True
    print(semester)
    
    """
    Change the filename before running a simulation to prevent overwriting previous results.
    """
    filename = f"individual-robustness_{semester_name}_filename"
    
    """
    The function below runs the experiment.
    """
    simulate(strategies, mechanisms, filename, semester, coarsen)