"""
Script for running simulations to compare the payments of strategic and truthful agents in the continuous effort, biased agents setting.

@author: Noah Burrell <burrelln@umich.edu>
"""

from numpy import ones
from statistics import mean, median, variance
import json

from setup import initialize_strategic_student_list, shuffle_students, initialize_submission_list
from grading import assign_grades, assign_graders, get_grading_dict
from grading_dmi import assign_graders_dmi_clusters

from mechanisms.baselines import mean_squared_error
from mechanisms.dmi import dmi_mechanism
from mechanisms.phi_divergence_pairing import phi_divergence_pairing_mechanism, parametric_phi_divergence_pairing_mechanism
from mechanisms.output_agreement import oa_mechanism
from mechanisms.parametric_mse import mse_p_mechanism
from mechanisms.peer_truth_serum import pts_mechanism

from evaluation import roc_auc_strategic
from graphing import plot_auc_strategic

import warnings

def run_simulation(num_iterations, num_assignments, strategy_map, mechanism, mechanism_param): 
    """
    Iteratively simulates semesters, scoring students according to a single mechanism, and recording the values of the relevant evaluation metrics.

    Parameters
    ----------
    num_iterations : int.
                     The number of semesters to simulate.
    num_assignments : int.
                      The number of assignments to include in each simulated semester.
    strategy_map: dict.
                  Maps the name of a strategy to a number of students who should adopt that strategy in each simulated semester.
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
                     "ROC-AUC Scores": [ score (float)],
                     "Mean ROC-AUC": mean_auc (float),
                     "Median ROC-AUC": median_auc (float),
                     "Variance ROC-AUC":  variance_auc (float)
                }
    """
    score_dict = {}
    auc_scores = []
    
    print("        ", mechanism, mechanism_param)
    
    for i in range(num_iterations):
        """
        Simulating a "semester"
        """
        students = initialize_strategic_student_list(strategy_map)
        shuffle_students(students)
        
        #necessary for PTS
        H = ones(11)
            
        for assignment in range(num_assignments):
            """
            Simulating a single assignment
            """
            submissions = initialize_submission_list(students, assignment)
            if mechanism == "DMI":
                cluster_size = int(mechanism_param)
                grader_dict = assign_graders_dmi_clusters(students, submissions, cluster_size)
            else:
                grader_dict = assign_graders(students, submissions, 4)
            grading_dict = get_grading_dict(grader_dict)
            
            #Here is where you can change the number of draws an active grader gets
            assign_grades(grading_dict, 3, assignment, True, True)
            
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
    
        auc_score = roc_auc_strategic(students)
        auc_scores.append(auc_score)
        
    score_dict["ROC-AUC Scores"] = auc_scores
        
    mean_auc = mean(auc_scores)
    score_dict["Mean ROC-AUC"] = mean_auc
    
    median_auc = median(auc_scores)
    score_dict["Median ROC-AUC"] = median_auc
    
    variance_auc = variance(auc_scores, mean_auc)
    score_dict["Variance ROC-AUC"] = variance_auc
    
    return score_dict

def compare_mechanisms(num_iterations, num_assignments, strategy_map, mechanisms):
    """
    Iterates over a list of mechanisms, calling run_simulation for each one.

    Parameters
    ----------
    num_iterations : int.
                     The number of semesters to simulate.
    num_assignments : int.
                      The number of assignments to include in each simulated semester.
    strategy_map: dict.
                  Maps the name of a strategy to a number of students who should adopt that strategy in each simulated semester.
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
        
        score_dict = run_simulation(num_iterations, num_assignments, strategy_map, mechanism, param)
        
        key = mechanism + ": " + param 
        eval_dict[key] = score_dict
    
    return eval_dict

def simulate(strategies, mechanisms, filename):
    """
    Calls compare_mechanisms iteravely for each strategy, varying the number of strategic graders.
    
    Saves a file containing the results of the experiment and generates and saves a plot of those results.
    Results are saved as filename.json in the ./results directory.
    Plots are saved as filename-*MECHANISM*.pdf in the ./figures directory.

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
        
            evals = compare_mechanisms(100, 10, strategy_map, mechanisms)
            result[strat] = evals
            
        results[strategy] = result
    """
    Export JSON file of simulation data to results directory
    """
    json_file = "results/" + filename + ".json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    """
    Graphing the results in the figures directory
    """
    plot_auc_strategic(results, filename)

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
            
            #("MSE_P", "0"),
            #("Phi-DIV_P", "CHI_SQUARED"),
            #("Phi-DIV_P", "KL"),
            #("Phi-DIV_P", "SQUARED_HELLINGER"),
            #("Phi-DIV_P", "TVD"),
            
        ]
    
    """ 
    Uncomment the strategies to be included in an experiment.
    
    Note that uninformative strategies are not considered in this experiment.
    """
    
    strategies = [
                        
            #"NOISE",
            #"FIX-BIAS",
            #"MERGE",
            #"PRIOR", 
            #"ALL10", 
            #"HEDGE"
        ]
    
    """
    Change the filename before running a simulation to prevent overwriting previous results.
    """
    filename = "truthful_vs_strategic_payments-ce-bias-filename"
    
    """
    The function below runs the experiment.
    """
    simulate(strategies, mechanisms, filename)
    
    
