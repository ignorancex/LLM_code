"""
Helper functions that initialize the Student and Submission objects for a simulated semester.

@author: Noah Burrell <burrelln@umich.edu>
"""

from classes import Student, Submission, StrategicStudent

from random import shuffle

def initialize_student_list(num_students, num_active):
    """
    Create a list of Student objects, with a specified number of active graders.
    
    (student_list returned by this function has some structure: all the active graders are first, then all the passive graders)

    Parameters
    ----------
    num_students : int 
                   Number of Student objects to create.
    num_active : int 
                 Number of Students who should have type="active".

    Returns
    -------
    student_list : list of Student objects.

    """
    num_passive = num_students - num_active
    active_list = [Student(i, "active") for i in range(num_active)]
    passive_list = [Student(i + num_active, "passive") for i in range(num_passive)]
    student_list = active_list + passive_list
    return student_list

def initialize_strategic_student_list(strategy_map):
    """
    Creates a list of StrategicStudent objects, according to a given description of which strategies should be included in the population and how many agents should adopt each such strategy. 

    (student_list returned by this function has some structure based on the order given by the strategy_map)

    Parameters
    ----------
    strategy_map: dict.
                  A dictionary mapping the names of strategies to the number of students set to adopt that strategy.
                    e.g. {
                            "TRUTH": 80,
                            "NOISE": 0,
                            "FIX_BIAS": 0,
                            "MERGE": 20,
                            "PRIOR": 0,
                            "ALL10": 0,
                            "HEDGE": 0
                        }

    Returns
    -------
    student_list : list of StrategicStudent objects.

    """
    student_list = []
    i = 0
    for strat, num in strategy_map.items():
        for _ in range(num):
            s = StrategicStudent(i, strat)
            student_list.append(s)
            i += 1
    return student_list

def shuffle_students(student_list):
    """
    Removes the structure from an existing list of Student objects by shuffling and then re-numbering accordingly.

    Parameters
    ----------
    student_list : list of Student objects.

    Returns
    -------
    None.

    """
    shuffle(student_list)
    for i in range(len(student_list)):
        student = student_list[i]
        student.id = i
        
def initialize_submission_list(student_list, assignment_number):
    """
    Creates a Submission object for each Student in student_list for the given assignment.

    Parameters
    ----------
    student_list : list of Student objects.
    assignment_number : int.
                        Unique identifier for a specific assignment.

    Returns
    -------
    submission_list : list of Submission objects.

    """
    submission_list = [Submission(student.id, assignment_number) for student in student_list]
    return submission_list
    
        
    
    
        
    