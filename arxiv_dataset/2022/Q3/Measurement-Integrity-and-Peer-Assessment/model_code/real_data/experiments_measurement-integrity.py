"""
Script for running experiments quantifying measurement integrity using real grading data.

@author: Noah Burrell <burrelln@umich.edu>
"""

from numpy import ones
import json
from statistics import mean

import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from mechanisms.baselines import mean_squared_error
from mechanisms.phi_divergence_pairing import phi_divergence_pairing_mechanism, parametric_phi_divergence_pairing_mechanism
from mechanisms.output_agreement import oa_mechanism
from mechanisms.parametric_mse import mse_p_mechanism
from mechanisms.peer_truth_serum import pts_mechanism

from evaluation import aucs_mse, correlation_mse, kendall_tau_mse

from load import load17, load19

import warnings

def run_simulation(assignment_partition, mechanism, mechanism_param, semester, coarsen=True):
    """
    Iteratively simulates semesters, scoring students according to a single mechanism, and recording the values of the relevant evaluation metrics.

    Parameters
    ----------
    assignment_partition : list of lists of ints
                      Defines the assignments in the semester (by number) and how they should be partitioned in simulations.
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
                      Denotes different versions of the same mechanism, e.g. the choice of Phi-divergence used in the Phi-divergence pairing mechanism.
                      "0" for mechanisms that do not require such a parameter.
    semester : str
              Chooses a semester for which to load data.
              One of: "Spring 17", "Fall 17", "Spring 19", "Fall 19"
    coarsen : bool, optional
             Set to true if data should be coarsened so that grades fall in the standard integer [0, 10] range.
             Default is True.

    Returns
    -------
    score_dict : dict.
        { 
            assignment_partition_number:
                {
                    "Binary AUCs": [ list of AUCs from using payments to classify students as above or below median MSE (floats) ]
                    "Quinary AUCs": [ list of average pairwise (over pairs of Quintiles) AUCs from using payments to classify students according to quintile (floats) ]
                    "Taus": [ list of Kendall rank correlations between ranking from MSE of reports and ranking from payments (floats) ]
                    "Rhos": [ list of Pearson correlations between MSE of reports and payments (floats) ]
                }
        }
    """
    score_dict = {}
    for i in range(1, len(assignment_partition) + 1):
        score_dict[i] = {}
        
        score_dict[i]["Binary AUCs"] = []
        score_dict[i]["Quinary AUCs"] = []
        score_dict[i]["Taus"] = []
        score_dict[i]["Rhos"] = []
    
    print("    ", mechanism, mechanism_param)
    
    """
    Set semester-specific variables.
    """
    
    include_q = True
    
    if semester == "Spring 17":
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
        all_students, all_submissions = load19("Spring", coarsen, False)
        
        if coarsen:
            possible_grades = 11
            
            # MLE
            mu = 7.68
            sigma = 1.92
            gamma = 1/(sigma ** 2)
            
        else: 
            return
            
    elif semester == "Fall 19":
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
        
    #Records the number of payments each student receives.
    for student in all_students:
        student.num_graded = 0
        
    for _ in range(50):
        
        #necessary for PTS
        H = ones(possible_grades)
    
        for idx, part in enumerate(assignment_partition):
            for assignment in part:
                """
                Considering a single assignment at a time.
                """
                submission_list = [sub for sub in all_submissions if sub.assignment_number == assignment]
                students = [student for student in all_students if assignment in student.grades.keys()]
                
                if len(submission_list) < 1:
                    # Skip over empty assignments
                    continue
                
                grader_dict = {}
                for submission in submission_list:
                    graders = []
                    grader_ids = list(submission.grades.keys())
                    for stu in all_students:
                        if stu.id in grader_ids:
                            report = stu.grades[assignment][submission.student_id]
                            stu.update_mse(submission.true_grade, report)
                            graders.append(stu)
                            stu.num_graded += 1
                    grader_dict[submission] = graders
                
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
            
            included_students = [student for student in all_students if student.included]
        
            for stu in included_students:
                num = stu.num_graded
                stu.raw_mse = stu.mse
                stu.raw_payment = stu.payment
                stu.mse *= (1/num)
                stu.payment *= (1/num) 
        
            b, q = aucs_mse(included_students, include_q)
            kt = kendall_tau_mse(included_students)
            rho = correlation_mse(included_students)
            
            for stu in included_students:
                stu.mse = stu.raw_mse
                stu.payment = stu.raw_payment
            
            i = idx + 1
            score_dict[i]["Binary AUCs"].append(b)
            score_dict[i]["Quinary AUCs"].append(q)
            score_dict[i]["Taus"].append(kt)
            score_dict[i]["Rhos"].append(rho)
            
        for student in all_students:
            student.mse = 0
            student.num_graded = 0
            student.payment = 0
            
    return score_dict

def compare_mechanisms_varying_num_assignments(assignment_partition, mechanisms, semester, coarsen):
    """
    Iterates over a list of mechanisms and a range of num_assignments, calling run_simulation for each one.

    Parameters
    ----------
    assignment_partition : list of lists of ints
                      Defines the assignments in the semester (by number) and how they should be partitioned in simulations.
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
                Maps the string "mechanism_name: mechanism_param" to dicts that map values of num_assignments to a score_dict (returned from the call to run_simulation).

    """
    eval_dict = {}
    
    for mechanism, param in mechanisms:
        mechanism_dict = run_simulation(assignment_partition, mechanism, param, semester, coarsen)
        
        key = mechanism + ": " + param 
        eval_dict[key] = mechanism_dict
    
    return eval_dict

def simulate(mechanisms, filename, semester, coarsen):
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
    semester : str
              Chooses a semester for which to load data.
              One of: "Spring 17", "Fall 17", "Spring 19", "Fall 19"
    coarsen : bool
             Set to true if data should be coarsened so that grades fall in the standard integer [0, 10] range.
    
    Returns
    -------
    None.
    
    """
    print("Working on an experiment for " + semester + ".")
    
    assignments_dict = {
            "Spring 17": [
                [1, 2, 3, 4], 
                [5, 6, 7, 8], 
                [9, 10, 11, 12], 
                [13, 14, 15, 16]
            ],
            "Fall 17": [
                [1, 2, 3, 4], 
                [5, 6, 7, 8], 
                [9, 10, 11, 12],
                [13, 14, 15, 16]
            ],
            "Spring 19": [
                [1, 2, 3], 
                [4, 5, 6], 
                [7, 8, 9], 
                [10, 11, 13, 14]
            ],
            "Fall 19": [
                [1, 2, 3], 
                [4, 5, 6], 
                [7, 8, 9, 10], 
                [11, 12, 13, 14]
            ],
        }
    
    assignments = assignments_dict[semester]
    
    results = compare_mechanisms_varying_num_assignments(assignments, mechanisms, semester, coarsen) 
    
    json_file = "../results/" + filename + ".json"
    
    """
    Export JSON file of simulation data to results directory
    """
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    """
    Print the results after taking the average of each list.
    """
    
    avg_results = dict(results)
    for mechanism, mechanism_dict in avg_results.items():
        for num_assignments, metrics in mechanism_dict.items():
            for metric, lst in metrics.items():
                avg = mean(lst)
                metrics[metric] = avg
    
    #print(json.dumps(avg_results, indent=4))
    
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
    
    semesters = ["Spring 17", "Fall 17", "Spring 19", "Fall 19"]
    
    """
    Select which semester should be used in the experiment by changing the index below.
    """
    semester = semesters[0]
    semester_name = semester.replace(" ", "")
    coarsen = True
    
    """
    Change the filename before running a simulation to prevent overwriting previous results.
    """
    filename = f"payments-vs-mse_{semester_name}_filename"
    
    """
    The function below runs the experiment.
    """
    simulate(mechanisms, filename, semester, coarsen)
