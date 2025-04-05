"""
Script for running simulations in the continuous effort, unbiased agents setting.

@author: Noah Burrell <burrelln@umich.edu>
"""

from numpy import ones
import json
from statistics import mean

from setup import initialize_student_list, shuffle_students, initialize_submission_list
from grading import assign_grades, assign_graders, get_grading_dict
from grading_dmi import assign_graders_dmi_clusters

from mechanisms.baselines import mean_squared_error
from mechanisms.dmi import dmi_mechanism
from mechanisms.phi_divergence_pairing import phi_divergence_pairing_mechanism, parametric_phi_divergence_pairing_mechanism
from mechanisms.output_agreement import oa_mechanism
from mechanisms.parametric_mse import mse_p_mechanism
from mechanisms.peer_truth_serum import pts_mechanism

from evaluation import aucs_mse, correlation_mse, kendall_tau_mse 

import warnings

def run_simulation(num_iterations, num_assignments, num_students, mechanism, mechanism_param):
    """
    Iteratively simulates semesters, scoring students according to a single mechanism, and recording the values of the relevant evaluation metrics.

    Parameters
    ----------
    num_iterations : int.
                     The number of semesters to simulate.
    num_assignments : int.
                      The number of assignments to include in each simulated semester.
    num_students : int.
                   The size of the student population that should be created for each semester.
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
                      Denotes different versions of the same mechanism, e.g. the choice of Phi-divergence used in the Phi-divergence pairing mechanism.
                      "0" for mechanisms that do not require such a parameter.

    Returns
    -------
    score_dict : dict.
            {
                "Binary AUCs": [ list of AUCs from using payments to classify students as above or below median MSE (floats) ]
                "Quinary AUCs": [ list of average pairwise (over pairs of Quintiles) AUCs from using payments to classify students according to quintile (floats) ]
                "Taus": [ list of Kendall rank correlations between ranking from MSE of reports and ranking from payments (floats) ]
                "Rhos": [ list of Pearson correlations between MSE of reports and payments (floats) ]
            }
    """
    score_dict = {}
    b_aucs = []
    q_aucs = []
    kt_scores = []
    rho_scores = []
    
    print("    ", mechanism, mechanism_param)
    
    for i in range(num_iterations):
        
        """
        Simulating a "semester"
        """
        students = initialize_student_list(num_students, num_students)
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
                
                mse_p_mechanism(grader_dict, students, assignment, mu, gamma, True)
                
            elif mechanism == "Phi-DIV_P":
                mu = 7
                gamma = 1/2.1
                
                parametric_phi_divergence_pairing_mechanism(grader_dict, students, assignment, mu, gamma, False, mechanism_param)
                
            else:
                print("Error: The given mechanism name does not match any of the options.")
                
        b, q = aucs_mse(students)
        kt = kendall_tau_mse(students)
        rho = correlation_mse(students)
        
        b_aucs.append(b)
        q_aucs.append(q)
        kt_scores.append(kt)
        rho_scores.append(rho)

    score_dict["Binary AUCs"] = b_aucs
    score_dict["Quinary AUCs"] = q_aucs
    score_dict["Taus"] = kt_scores
    score_dict["Rhos"] = rho_scores
    
    return score_dict

def compare_mechanisms(num_iterations, num_assignments, num_students, mechanisms):
    """
    Iterates over a list of mechanisms, calling run_simulation for each one.

    Parameters
    ----------
    num_iterations : int.
                     The number of semesters to simulate.
    num_assignments : int.
                      The number of assignments to include in each simulated semester.
    num_students : int.
                   The size of the student population that should be created for each semester.
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
        
        score_dict = run_simulation(num_iterations, num_assignments, num_students, mechanism, param)
        
        key = mechanism + ": " + param 
        eval_dict[key] = score_dict
    
    return eval_dict

def compare_mechanisms_varying_num_assignments(num_iterations, max_num_assignments, num_students, mechanisms):
    """
    Iterates over a list of mechanisms and a range of num_assignments, calling run_simulation for each one.

    Parameters
    ----------
    num_iterations : int.
                     The number of semesters to simulate.
    max_num_assignments : int.
                      The max number of assignments to include in each simulated semester.
    num_students : int.
                   The size of the student population that should be created for each semester.
    mechanisms : list of 2-tuples of strings. 
                 Describes the mechanisms to be included in the form ("mechanism_name", "mechanism_param").
                 The complete list of possible mechanisms and associated params can be found below in the code for running simulations.

    Returns
    -------
    eval_dict : dict.
                Maps the string "mechanism_name: mechanism_param" to dicts that map values of num_assignments to a score_dict (returned from the call to run_simulation).

    """
    eval_dict = {}
    
    
    for mechanism, param in mechanisms:
        mechanism_dict = {}
        for i in range(max_num_assignments):
            num_assignments = i + 1
            score_dict = run_simulation(num_iterations, num_assignments, num_students, mechanism, param)
            mechanism_dict[num_assignments] = score_dict
        
        key = mechanism + ": " + param 
        eval_dict[key] = mechanism_dict
    
    return eval_dict

def simulate(mechanisms, filename):
    """
    Calls compare_mechanisms.
    
    Saves a file containing the results of the experiment and generates and saves a plot of those results.
    Results are saved as filename.json in the ./results directory.
    Plots are saved as filename.pdf in the ./figures directory.

    Parameters
    ----------
    mechanisms : list of 2-tuples of strings. 
                 Describes the mechanisms to be included in the form ("mechanism_name", "mechanism_param").
                 The complete list of possible mechanisms and associated params can be found below in the code for running simulations.
    filename : str.
               The filename used to save the .json file and .pdf plot associated with the experiment.

    Returns
    -------
    None.
    
    """
    print("Working on simulations for 500 students.")
    
    #results = compare_mechanisms(100, 10, 1000, mechanisms)
    results = compare_mechanisms_varying_num_assignments(50, 15, 500, mechanisms) 
    
    json_file = "results/" + filename + ".json"
    
    """
    Export JSON file of simulation data to results directory
    """
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    """
    Print the results when taking the average of each list.
    """
    '''
    avg_results = dict(results)
    for mechanism, metrics in avg_results.items():
        for metric, lst in metrics.items():
            avg = mean(lst)
            metrics[metric] = avg
    '''
    avg_results = dict(results)
    for mechanism, mechanism_dict in avg_results.items():
        for num_assignments, metrics in mechanism_dict.items():
            for metric, lst in metrics.items():
                avg = mean(lst)
                metrics[metric] = avg
    
    print(json.dumps(avg_results, indent=4))
    
    """
    Graphing the results in the figures directory
    """
    #plot_payments_vs_mse(results, filename)

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
            #("Phi-DIV_P", "CHI_SQUARED"),
            #("Phi-DIV_P", "KL"),
            #("Phi-DIV_P", "SQUARED_HELLINGER"),
            #("Phi-DIV_P", "TVD"),
            
        ]
    
    """
    Change the filename before running a simulation to prevent overwriting previous results.
    """
    filename = "payments-vs-mse_mse-p"
    
    """
    The function below runs the experiment.
    """
    simulate(mechanisms, filename)