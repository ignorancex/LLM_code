"""
Script for running simulations to examine the potential for one agent to gain by deviating from truthful reporting in the continuous effort, biased agents setting.

@author: Noah Burrell <burrelln@umich.edu>
"""

from numpy import ones
from statistics import mean, median, variance
import json
from random import randint, seed
from sys import maxsize

from setup import initialize_strategic_student_list, shuffle_students, initialize_submission_list
from grading import assign_grades, assign_graders, get_grading_dict
from grading_dmi import assign_graders_dmi_clusters

from mechanisms.baselines import mean_squared_error
from mechanisms.dmi import dmi_mechanism
from mechanisms.phi_divergence_pairing import phi_divergence_pairing_mechanism, parametric_phi_divergence_pairing_mechanism
from mechanisms.output_agreement import oa_mechanism
from mechanisms.parametric_mse import mse_p_mechanism
from mechanisms.peer_truth_serum import pts_mechanism

from graphing import plot_mean_rank_changes, plot_variance_rank_changes

import warnings

def run_simulation(num_semesters, num_assignments, strategy_map, strat, mechanism, mechanism_param): 
    """
    Iteratively simulates semesters, scoring students according to a single mechanism, and recording the values of the relevant evaluation metrics.

    Parameters
    ----------
    num_semesters : int.
                     The number of semesters to simulate.
    num_assignments : int.
                      The number of assignments to include in each simulated semester.
    strategy_map: dict.
                  Maps the name of a strategy to a number of students who should adopt that strategy in each simulated semester.
    strat: str.
           The name of the strategy that the deviator will adopt.
    mechanism : str.
                The name of the mechanism to be used to score the students performance in the grading task.
                One of the following:
                    - "BASELINE"
                    - "DMI"
                    - "OA"
                    - "Phi-DIV"
                    - "PTS"
                    - "MSE_P"
                    - "Phi-DIV_P"
    mechanism_param : str.
                      Denotes different versions of the same mechanism, e.g. the choice phi divergence used in the phi divergence pairing mechanism.
                      "0" for mechanisms that do not require such a parameter.

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
    avg_truthful_payments = []
    avg_strategic_payments = []
    
    print("        ", mechanism, mechanism_param)
    
    for _ in range(num_semesters):
        """
        Simulating a "semester"
        """
        students = initialize_strategic_student_list(strategy_map)
        shuffle_students(students)
        submission_lists = [initialize_submission_list(students, i) for i in range(num_assignments)]
        
        """
        Select a deviator.
        """
        
        found_deviator = False
        
        i = 0
        while((i < len(students)) and (not found_deviator)):
            s = students[i]
            if s.strategy == "TRUTH":
                deviator = s
                found_deviator = True
            i += 1
            
        grader_dicts = []
        grading_dicts = []
            
        for assignment in range(len(submission_lists)):
            """
            Simulating a grading a single assignment
            """
            submissions = submission_lists[assignment]
            
            if mechanism == "DMI":
                cluster_size = int(mechanism_param)
                grader_dict = assign_graders_dmi_clusters(students, submissions, cluster_size)
            
            else:
                grader_dict = assign_graders(students, submissions, 4)
            
            grading_dict = get_grading_dict(grader_dict)
            
            #Here is where you can change the number of draws an active grader gets
            assign_grades(grading_dict, 3, assignment, True, True)
            
            grader_dicts.append(grader_dict)
            grading_dicts.append(grading_dict)
        
        deviator_ranks = []
        random_seed = randint(~maxsize, maxsize)
            
        for iteration in range(2):
            seed(random_seed)
            if iteration == 1:
                """
                Change deviator reports to strategic reports for every submission on every assignment
                """
                deviator.strategy = strat
                
                for assignment_num in range(len(grading_dicts)):
                    grading_dict = grading_dicts[assignment_num]
                    
                    deviator_submissions = grading_dict[deviator]
                    
                    for submission in deviator_submissions:
                        signal = deviator.grades[assignment_num][submission.student_id]
                        grade = deviator.report(signal)
                    
                        deviator.grades[assignment_num][submission.student_id] = grade
                        submission.grades[deviator.id] = grade
                
            #necessary for PTS
            H = ones(11)
            
            for assignment in range(len(submission_lists)):
                """
                Run the Mechanism
                """
                grader_dict = grader_dicts[assignment]
                grading_dict = grading_dicts[assignment]
                
                """
                Non-Parametric Mechanisms
                """
                    
                if mechanism == "BASELINE":
                    mean_squared_error(grader_dict)
                        
                elif mechanism == "DMI":
                    cluster_size = int(mechanism_param)
                    dmi_mechanism(grader_dict, assignment, cluster_size)
                        
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
                    mu = 7
                    gamma = 1/2.1
                    
                    mse_p_mechanism(grader_dict, students, assignment, mu, gamma, True, False)
                    
                elif mechanism == "Phi-DIV_P":
                    mu = 7
                    gamma = 1/2.1
                    
                    parametric_phi_divergence_pairing_mechanism(grader_dict, students, assignment, mu, gamma, False, mechanism_param)
                    
                else:
                    print("Error: The given mechanism name does not match any of the options.")
            
            '''
            Calculate the rank of the deviator (according to the number of payments that are >= than hers)
            '''
            val = deviator.payment
            rank = 0
            for student in students:
                if student.payment >= val:
                    rank += 1
            deviator_ranks.append(rank)
            
            if iteration == 0:
                '''
                Calculate the average payments for the truthful and strategic agents
                '''
                truthful_payments = []
                strategic_payments = []
                for student in students:
                    pay = student.payment
                    if student.strategy == "TRUTH":
                        truthful_payments.append(pay)
                    else:
                        strategic_payments.append(pay)
                avg_truthful_payment = mean(truthful_payments)
                avg_strategic_payment = mean(strategic_payments)
                avg_truthful_payments.append(avg_truthful_payment)
                avg_strategic_payments.append(avg_strategic_payment)
                
                '''
                Reset the payments for all students.
                '''
                for student in students:
                    student.payment = 0
                    
        '''
        Reseed the randomness for initializing next semester
        '''        
        seed()
              
        deviator_gain = deviator_ranks[0] - deviator_ranks[1]
        deviator_gains.append(deviator_gain)
        
    mean_gain = mean(deviator_gains)
    score_dict["Mean Gain"] = mean_gain
    
    median_gain = median(deviator_gains)
    score_dict["Median Gain"] = median_gain
    
    variance_gain = variance(deviator_gains, mean_gain)
    score_dict["Variance Gain"] = variance_gain
    
    return score_dict


def compare_mechanisms(num_semesters, num_assignments, strategy_map, strategy, mechanisms):
    """
    Iterates over a list of mechanisms, calling run_simulation for each one.

    Parameters
    ----------
    num_semesters : int.
                    The number of semesters to simulate.
    num_assignments : int.
                      The number of assignments to include in each simulated semester.
    strategy_map: dict.
                  Maps the name of a strategy to a number of students who should adopt that strategy in each simulated semester.
    strat: str.
           The name of the strategy that the deviator will adopt.
    mechanisms : list of 2-tuples of strings. 
                 Describes the mechanisms to be included in the form ("mechanism_name", "mechanism_param").
                 The complete list of possible mechanisms and associated params can be found below in the code for running simulations.

    Returns
    -------
    eval_dict : dict.
                Maps the string "mechanism_name: mechanism_param" to a score_dict (returned from the call to run_simulation).

    """
    eval_dict = {}
    
    for mechanism, param in mechanisms:
        
        score_dict = run_simulation(num_semesters, num_assignments, strategy_map, strategy, mechanism, param)
        
        key = mechanism + ": " + param 
        eval_dict[key] = score_dict
    
    return eval_dict

def simulate(strategies, mechanisms, filename):
    """
    Calls compare_mechanisms iteravely for each strategy, varying the number of strategic graders.
    
    Saves a file containing the results of the experiment and generates and saves a plot of those results.
    Results are saved as filename.json in the ./results directory.
    Plots are saved as filename-mean_gain-*MECHANISM*.pdf and filename-variance_gain-*MECHANISM*.pdf in the ./figures directory.

    Parameters
    ----------
    strategies : list of strings.
                 Describes the strategies ot be included.
                 The list of all relevant strategies can be found below in the code for running simulations.
    mechanisms : list of 2-tuples of strings. 
                 Describes the mechanisms to be included in the form ("mechanism_name", "mechanism_param").
                 The complete list of possible mechanisms and associated params can be found below in the code for running simulations.
    filename : str.
               The filename used to save the .json file and .pdf plot associated with the experiment.

    Returns
    -------
    None.

    """
    results = {}
    
    for strategy in strategies:
        result = {}
        print("Working on simulations for the following strategy:", strategy)
        
        for strat in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
            strategy_map = {}
            print("    Working on simulations for", strat, "strategic students.")
        
            strategy_map[strategy] = strat
            strategy_map["TRUTH"] = 100 - strat
        
            evals = compare_mechanisms(100, 10, strategy_map, strategy, mechanisms)
            result[strat] = evals
        
        results[strategy] = result
        
    json_file = "results/" + filename + ".json"
    
    """
    Export JSON file of simulation data to results directory
    """
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    """
    Graphing the results in the figures directory
    """
    mean_gain_filename = filename + "-mean_gain"
    plot_mean_rank_changes(results, mean_gain_filename)
    
    variance_gain_filename = filename + "-variance_gain"
    plot_variance_rank_changes(results, variance_gain_filename)

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
            
            #("BASELINE", "MSE"),
            #("DMI", "4"),
            #("OA", "0"),
            #("Phi-DIV", "CHI_SQUARED"),
            #("Phi-DIV", "KL"),
            #("Phi-DIV", "SQUARED_HELLINGER"),
            #("Phi-DIV", "TVD"),
            #("PTS", "0"),
            
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
            "FIX-BIAS",
            "MERGE",
            "PRIOR", 
            "ALL10", 
            "HEDGE"
        ]
    
    """
    Change the filename before running a simulation to prevent overwriting previous results.
    """
    filename = "incentives_for_deviating-ce-bias-parametric-bias_correct_false"
    
    """
    The function below runs the experiment.
    """
    simulate(strategies, mechanisms, filename)