"""
Implementation of the Output Agreement mechanism.

@author: Noah Burrell <burrelln@umich.edu>
"""

import numpy as np
from itertools import combinations

def oa_mechanism(grader_dict):
    """
    Computes payments for students according to the OA mechanism.

    Parameters
    ----------
    grader_dict :  dict.
                   Maps a Submission object to a list of graders (Student objects).

    Returns
    -------
    None.

    """
    
    H = np.ones(11)
    R = np.multiply(H, (1.0/np.sum(H)))
    
    for submission, graders in grader_dict.items():
        
        """
        COMPUTING THE SCORES
        """
        assignment = submission.assignment_number
        
        task = submission.student_id
        
        constant = 1/(len(graders) - 1)
        
        pairs = combinations(graders, 2)
        
        for pair in pairs:
            one = pair[0]
            two = pair[1]
            
            one_report = one.grades[assignment][task]
            two_report = two.grades[assignment][task]
            
            score = 0 
            
            if one_report == two_report:
                score = 1.0 / R[one_report]
                
            one.payment += constant*score
            two.payment += constant*score