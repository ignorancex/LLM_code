"""
Helper functions that perform tasks related to "grading an assignment" in a simulated semester.

@author: Noah Burrell <burrelln@umich.edu>
"""

from networkx import random_regular_graph
from scipy.stats import binom, poisson
from statistics import mean

def assign_graders(student_list, submission_list, num_graders):
    """
    Assigns graders (Student objects) to submissions (Submission objects) that they will "grade" (i.e. for which they will receive a signal and compute a report).      
    
    Parameters
    ----------
    student_list : list of Student objects.
    submission_list : list of Submission objects for a single assignment (i.e. that all have the same assignment_number attribute).
    num_graders : int.
                  Number of graders that are assigned to grade each submission.

    Returns
    -------
    grader_dict : dict. 
                 Maps a Submission object to a list of Student objects will grade that submission.
                 grader_dict = { submission (Submission object): [ graders (Student objects) ] } 
    """
    
    #student_id_map maps student id numbers (int) to the corresponding Student objects
    student_id_map = {s.id: s for s in student_list}
    
    d = num_graders
    n = len(student_list)
    G = random_regular_graph(d, n)
    
    grader_dict = {}
    for submission in submission_list:
        node = submission.student_id
        neighbors = G.neighbors(node)
        graders = [student_id_map[neighbor] for  neighbor in neighbors]
        grader_dict[submission] = graders
    return grader_dict

def get_grading_dict(grader_dict):
    """
    Inverts the information in grader_dict to create a grading_dict that maps a Student object to a list of Submission objects that they will grade.
    
    Parameters
    ----------
    grader_dict :  dict.
                   Maps a Submission object to a list of graders (Student objects).

    Returns
    -------
    grading_dict : dict. 
                   Maps a grader (Student object) to a list of Submission objects.
                   grading_dict = { Student object: [ Submission objects ] }

    """
    
    grading_dict = {}
    for key, val in grader_dict.items():
        for grader in val:
            if grader not in grading_dict.keys():
                grading_dict[grader] = []
            grading_dict[grader].append(key)
    return grading_dict

def assign_grades(grading_dict, num_draws, assignment_num, continuous_effort=False, bias=False):
    """
    Simulates the grading process. Records the appropriate grading reports.
    Students grade the Submissions that they are assigned to grade (according to grading_dict) as follows:
        First, a signal is generated (according to the ground truth score and the bias and effort of the grader).
        Then, a report, which is a function of the signal, is generated and stored in the "grades" attribute (a dict) of the relevant Student and Submission object.
    
    Parameters
    ----------
    grading_dict : dict.
                   Maps a grader (Student object) to a list of Submission objects.
                   grading_dict = { Student object: [ Submission objects ] }
    num_draws: int.
               Number of draws from Binom distribution that an active grader gets to see. 
               Only relevant when continuous_effort = False.

    Returns
    -------
    None.
    
    """
    for grader, submissions in grading_dict.items():
        
        grader.grades[assignment_num] = {}
        
        for submission in submissions:
            ground_truth = submission.true_grade
            bias_val = 0
            
            if bias:
                bias_val += grader.bias
            
            probability = (ground_truth + bias_val)/10.0
            if probability > 1:
                probability = 1.0
            elif probability < 0:
                probability = 0.0
                
            if continuous_effort:
                l = grader.lam
                num = 1 + poisson.rvs(mu=l, loc=0, random_state=None)
        
            else:
                if grader.type == "active":
                    num = num_draws
                else:
                    num = 1
                    
            draws = []
            for i in range(num):
                draw = binom.rvs(n=10, p=probability, random_state=None)
                draws.append(draw)
                
            avg = mean(draws)
            signal = int(round(avg))
            
            grade = grader.report(signal)
                
            grader.grades[assignment_num][submission.student_id] = grade
            submission.grades[grader.id] = grade
            
            grader.update_mse(ground_truth, grade)
    