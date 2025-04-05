"""
Implementation of the baseline MSE mechanism.

@author: Noah Burrell <burrelln@umich.edu>
"""

from statistics import mean

def mean_squared_error(grader_dict):
    """
    Computes payments for students according to the baseline MSE mechanism.
    Also computes a ``consensus grade''---an estimate of the true score computed as the average of the reports given by the graders---for each submission.
    
    Parameters
    ----------
    grader_dict :  dict.
                   Maps a Submission object to a list of graders (Student objects).

    Returns
    -------
    scores : a dict.
             {submission.student_id: ``consensus grade''}.

    """
    scores = {}
    
    for submission, grader_list in grader_dict.items():
        
        consensus_grade = mean(list(submission.grades.values()))
        graders = {grader.id : grader for grader in grader_list}
        
        scores[submission.student_id] = consensus_grade
        
        for grader_id, grade in submission.grades.items():
            
            grader = graders[grader_id]
            
            squared_error = (grade - consensus_grade)**2
            avg_squared_error = 0.25 * squared_error
            
            grader.payment -= avg_squared_error

    return scores            