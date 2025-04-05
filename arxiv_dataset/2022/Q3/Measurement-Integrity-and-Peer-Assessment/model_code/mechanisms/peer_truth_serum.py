"""
Implementation of the Peer Truth Serum Mechanism (Faltings, Jurca, and Radanovic 2017).

@author: Noah Burrell <burrelln@umich.edu>
"""

import numpy as np
from itertools import combinations

def pts_mechanism(grader_dict, H_init):
    """
    Computes payments for students according to the PTS mechanism.

    Parameters
    ----------
    grader_dict :  dict.
                   Maps a Submission object to a list of graders (Student objects).
    
    H_init : np.array (or list) of ints. 
             Initial histogram of report values.

    Returns
    -------
    H : np.array of ints.
        Updated histogram of report values.

    """
    
    H = np.array(H_init)
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
            
            #Simplest PTS Mechanism: f(rr) = 0, C = 1.
            score = 0 
            
            if one_report == two_report:
                score = 1.0 / R[one_report]
                
            one.payment += constant*score
            two.payment += constant*score
            
            H[one_report] += 1
            H[two_report] += 1
            
    return H