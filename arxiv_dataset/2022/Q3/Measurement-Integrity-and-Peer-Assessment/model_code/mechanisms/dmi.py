"""
Implementation of the Determinant-based Mutual Information (DMI) Mechanism (Kong 2019).


@author: Noah Burrell <burrelln@umich.edu>
"""

from itertools import combinations
from random import shuffle
import numpy as np
        
def dmi_mechanism(grader_dict, assignment_num, cluster_size):
    """
    Computes payments for students according to the DMI mechanism.

    Parameters
    ----------
    grader_dict : dict.
                  Maps a Submission object to a list of graders (Student objects).
    assignment_num : int.
                     Unique identifier of the assignment for which payments are being computed.
    cluster_size : int.
                   The size of the clusters in which students grade submissions (should evenly divide the number of students.)
                   All students in a cluster grade the same submissions (the submissions from another cluster of students.)

    Returns
    -------
    None.

    """
    grade_map = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 1,
        8: 1,
        9: 1,
        10: 1
        }
    
    submission_dict = {submission.student_id: submission for submission in grader_dict.keys()}
    task_list = [submission.student_id for submission in grader_dict.keys()]
    
    task_list.sort()
    
    for i in range(0, len(task_list), cluster_size):
        submission0 = submission_dict[i]
        graders = grader_dict[submission0]
        tasks = [task_list[i + j] for j in range(cluster_size)]
        shuffle(tasks)
        
        for (j, k) in combinations(graders, 2):
            M_1 = np.zeros((2, 2), dtype=np.uint8)
            M_2 = np.zeros((2, 2), dtype=np.uint8)
            
            
            for t in range(cluster_size):
                j_grade = grade_map[j.grades[assignment_num][tasks[t]]]
                k_grade = grade_map[k.grades[assignment_num][tasks[t]]]
    
                if t < cluster_size/2:
                    M_1[j_grade, k_grade] += 1
                    
                else:
                    M_2[j_grade, k_grade] += 1
        
            d1 = np.linalg.det(M_1)
            d2 = np.linalg.det(M_2)
        
            score = d1*d2
        
            j.payment += score
            k.payment += score