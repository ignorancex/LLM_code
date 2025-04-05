"""
Helper function that assigns graders when the DMI mechanism is being used, effectively superseding the analogous function in grading.py.
(The DMI mechanism is not functional when the functions from grading.py are used to assign graders.)

@author: Noah Burrell <burrelln@umich.edu>
"""

def assign_graders_dmi_clusters(student_list, submission_list, cluster_size):
    """
    Assigns graders (Student objects) to submissions (Submission objects) that they will "grade" (i.e. for which they will receive a signal and compute a report).
        
    Graders are clustered in groups such that a group will all grade the same Submissions 
    (which will be the Submissions submitted by the Students from another cluster). 
       
    Parameters
    ----------
    student_list : list of Student objects.
    submission_list : list of Submission objects for a single assignment number.
    cluste_size : int.
                  Number of graders in each cluster.

    Returns
    -------
    grader_dict : dict.
                  Maps a Submission object to a list of Student objects will grade that submission. 
                  grader_dict = { submission (Submission object): [ graders (Student objects) ] } 

    """
    
    grader_dict = {}
    
    submissions = submission_list[cluster_size:] + submission_list[:cluster_size]
    
    for i in range(0, len(submissions), cluster_size):
        students = [student_list[i + j] for j in range(cluster_size)]
        for j in range(cluster_size):
            grader_dict[submissions[i + j]] = students
    
    return grader_dict