"""
Implementation of the non-parametric and parametric Phi-Divergence pairing mechanisms (Schoenebeck and Yu, 2021).

@author: Noah Burrell <burrelln@umich.edu>
"""
import numpy as np
from random import shuffle, choice
from math import log, exp, sqrt
from sys import maxsize
from itertools import combinations

from .parametric_mse import em_estimate_parameters

def phi_divergence_pairing_mechanism(grader_dict, phi_divergence="TVD"):
    """
    Computes payments for students according to the non-parametric Phi-Div pairing mechanism. 
    
    Estimating the Scoring Matrices is outsourced to a different function.
    
    Each Submission is graded by 4 students.

    Parameters
    ----------
    grader_dict : dict.
                  Maps a Submission object to a list of graders (Student objects).
    phi_divergence : str, optional. Should be one of the options below, default is TVD.
                     Defines an Phi-Mutual Information measure (phi is denoted by f below).
                        - TVD:
                            - f(a) = 1/2|a - 1|
                            - f*(b) = b if |b| <= 1/2; infty otherwise.
                            - df(a) = 1/2 if a > 1, -1/2 if a < 1, [-1/2, 1/2] if a = 1.
                        - KL: KL-Divergence
                            - f(a) = a log a
                            - f*(b) = exp(b - 1)
                            - df(a) = 1 + log(a)
                        - CHI_SQUARED: 
                            - f(a) = a^2 - 1
                            - f*(b) = b^2/4 + 1
                            - df(a) = 2a
                        - SQUARED_HELLINGER:
                            - f(a) = (1 - sqrt(a))^2
                            - f*(b) = -b/b-1, b < 1; infty otherwise.
                            - df(a) = 1 - 1/sqrt(a)

    Returns
    -------
    None.

    """
    
    minsize = -maxsize - 1
    
    A, B, S_A, S_B = estimate_pairwise_scoring_matrices(grader_dict, phi_divergence)
    
    for submission, graders in grader_dict.items():
        
        """
        COMPUTING THE SCORES
        
        1) Randomly separate the four agents into pairs
    
        2) For each pair: Choose a penalty task for each agent, score the pair.
    
        (Take an average over this process)
        
        """
        assignment = submission.assignment_number
        
        bonus = submission.student_id
        
        constant_dict = {}
        temp_scores = {}
        
        for grader in graders:
            constant_dict[grader.id] = (len(graders) - 1)
            temp_scores[grader.id] = 0
        
        pairs = combinations(graders, 2)
        
        for pair in pairs:
            one = pair[0]
            two = pair[1]
            
            bonus_one_grade = one.grades[assignment][bonus]
            bonus_two_grade = two.grades[assignment][bonus]
            
            penalties_one = list(one.grades[assignment].keys()) 
            if assignment in one.penalty_tasks.keys():
                penalties_one += list(one.penalty_tasks[assignment].keys())
            penalties_one.remove(bonus)
            penalties_two = list(two.grades[assignment].keys())
            if assignment in two.penalty_tasks.keys():
               penalties_two += list(two.penalty_tasks[assignment].keys())
            penalties_two.remove(bonus)
            
            i = 0
            found = False
            shuffle(penalties_one)
            while (i < len(penalties_one)) and (not found):
                possible = penalties_one[i]
                if possible not in penalties_two:
                    penalty_one = possible
                    found = True
                elif len(penalties_two) > 1:
                    penalty_one = possible
                    penalties_two.remove(penalty_one)
                    found = True
                i += 1
                
            if not found:
                #print("Pair of students without possible penalty task:", one.id, two.id)
                constant_dict[one.id] -= 1
                constant_dict[two.id] -= 1
                continue
                
            else:
                if penalty_one in one.grades[assignment].keys():
                    penalty_one_grade = one.grades[assignment][penalty_one]
                else:
                    penalty_one_grade = one.penalty_tasks[assignment][penalty_one]
                    
                penalty_two = choice(penalties_two)
                if penalty_two in two.grades[assignment].keys():
                    penalty_two_grade = two.grades[assignment][penalty_two]
                else:
                    penalty_two_grade = two.penalty_tasks[assignment][penalty_two]
            
            if phi_divergence == "TVD":
                conjugate = lambda b: b
                    
            elif phi_divergence == "KL":
                conjugate = lambda b: exp(b - 1)
            
            elif phi_divergence == "CHI_SQUARED":
                conjugate = lambda b: (b*b)/4 + 1
                
            elif phi_divergence == "SQUARED_HELLINGER":
                conjugate = lambda b: (-b)/(b - 1)
            
            if bonus in A:
                penalty_score = conjugate(S_A[penalty_one_grade, penalty_two_grade])
                
                if phi_divergence == "SQUARED_HELLINGER" and np.isnan(penalty_score):
                    #Conjugate may evaluate to infty/-infty, limit as b -> -infty is -1.
                    penalty_score = -1
                    
                score = S_A[bonus_one_grade, bonus_two_grade] - penalty_score
            
            elif bonus in B:
                penalty_score = conjugate(S_B[penalty_one_grade, penalty_two_grade])
                
                if phi_divergence == "SQUARED_HELLINGER" and np.isnan(penalty_score):
                    #Conjugate may evaluate to infty/-infty, limit as b -> -infty is -1.
                    penalty_score = -1
                    
                score = S_B[bonus_one_grade, bonus_two_grade] - penalty_score
            
            #Get rid of numpy -infty vales (raises error in scoring)
            if score < minsize:
                score = minsize
                
            temp_scores[one.id] += score
            temp_scores[two.id] += score
            
        for grader in graders:
            constant_inv = constant_dict[grader.id]
            if constant_inv > 0:
                constant = 1/constant_inv
                temp_score = temp_scores[grader.id]
                grader.payment += constant*temp_score
            else:
                grader.num_graded -= 1
            
def estimate_pairwise_scoring_matrices(grader_dict, phi_divergence="TVD"):
    """
    Estimates the scoring matrices used by the non-parametric Phi-Div pairing mechanism. 

    Parameters
    ----------
    grader_dict : dict.
                  Maps a Submission object to a list of graders (Student objects).
    phi_divergence : str, optional. Should be one of the options below, default is TVD.
                     Defines an Phi-Mutual Information measure (phi is denoted by f below).
                        - TVD:
                            - f(a) = 1/2|a - 1|
                            - f*(b) = b if |b| <= 1/2; infty otherwise.
                            - df(a) = 1/2 if a > 1, -1/2 if a < 1, [-1/2, 1/2] if a = 1.
                        - KL: KL-Divergence
                            - f(a) = a log a
                            - f*(b) = exp(b - 1)
                            - df(a) = 1 + log(a)
                        - CHI_SQUARED: 
                            - f(a) = a^2 - 1
                            - f*(b) = b^2/4 + 1
                            - df(a) = 2a
                        - SQUARED_HELLINGER:
                            - f(a) = (1 - sqrt(a))^2
                            - f*(b) = -b/b-1, b < 1; infty otherwise.
                            - df(a) = 1 - 1/sqrt(a)
        
    Returns
    -------
    A, B :  lists of ints (submission/task identifiers).
            Partitions of the set of tasks into two equal-sized sets.
    
    S_A,  S_B : 11x11 numpy 2d-arrays.
                Used for scoring the tasks in lists A and B, respectively, based on a pair of agent reports.

    """
    A = []
    B = []
    
    """
    Partition the set of tasks into two equal-sized sets A and B.
    """
    
    tasks = [submission.student_id for submission in grader_dict.keys()]
    shuffle(tasks)
        
    halfway = int(round(len(tasks)/2))
        
    A = tasks[:halfway]
    B = tasks[halfway:]
    
    """
    
    COMPUTE EXPECTED RANDOM ESTIMATE OF DISTRIBUTIONS
    
    """
        
    normalize_A = 1/len(A)
    normalize_B = 1/len(B)
    
    # JOINT DISTRIBUTION ESTIMATES
    JA = np.zeros(shape=(11, 11))
    JB = np.zeros(shape=(11, 11))
    
    # MARGINAL DISTRIBUTION ESTIMATES
    MA = np.zeros(11)
    MB = np.zeros(11)
    
    """
    Compute the expectation of the process of estimating the joint distribution of signals
    along w/the expectation of the process of estimating the marginal distribution.
    """
    
    for submission, graders in grader_dict.items():
        
        task = submission.student_id
        counts = np.zeros(11)
        
        for grade in submission.grades.values():
            counts[grade] += 1
        
        matrix = np.outer(counts, counts)
        
        for i in range(len(counts)):
            count = counts[i]
            if count > 0:
                matrix[i, i] = count * (count - 1)
        
        normalization_coefficient = 1/(len(graders)*(len(graders) - 1))
        marginal_normalization_coeff = 1/len(graders)
        
        if task in A:
            normalization_coefficient *= normalize_A
            matrix = np.multiply(matrix, normalization_coefficient)
            JA += matrix
            
            counts *= marginal_normalization_coeff
            MA += counts
            
        else:
            normalization_coefficient *= normalize_B
            matrix = np.multiply(matrix, normalization_coefficient)
            JB += matrix
            
            counts *= marginal_normalization_coeff
            MB += counts
            
    MA *= normalize_A
    MB *= normalize_B
    
    """
    Compute the products of the marginal distrubitions estimates
    """
    
    PMA = np.outer(MA, MA)
    PMB = np.outer(MB, MB)
    
    """
    Tasks in A are scored using the estimates computed from the tasks in B, and vice versa
    """
    
    S_A = compute_K(JB, PMB, phi_divergence) 
    S_B = compute_K(JA, PMA, phi_divergence)
    
    return A, B, S_A, S_B

def parametric_phi_divergence_pairing_mechanism(grader_dict, student_list, assignment_num, mu, gamma, bias_correct=True, phi_divergence="TVD"):
    """
    Computes payments for students according to the parametric Phi-Divergence pairing mechanism, using parametric model estimates for the joint-to-marginal product ratio.
    
    Each Submission is graded by 4 students.

    Parameters
    ----------
    grader_dict : dict.
                  Maps a Submission object to a list of graders (Student objects).
    student_list : list of Student objects.
                   The population of students/graders.
    assignment_num : int.
                     Unique identifier of the assignment for which payments are being computed.
    mu : float.
        The mean of the normal approximation of the distribution of true grades.
    gamma : float.
        The precision (i.e. the inverse of the variance) of the normal approximation of the distribution of true grades.
    bias : bool, optional.
        Indicates whether agents have bias, and therefore whether bias parameters should be estimated. The default is True.
    phi_divergence : str, optional. Should be one of the options below, default is TVD.
                     Defines an Phi-Mutual Information measure (phi is denoted by f below).
                        - TVD:
                            - f(a) = 1/2|a - 1|
                            - f*(b) = b if |b| <= 1/2; infty otherwise.
                            - df(a) = 1/2 if a > 1, -1/2 if a < 1, [-1/2, 1/2] if a = 1.
                        - KL: KL-Divergence
                            - f(a) = a log a
                            - f*(b) = exp(b - 1)
                            - df(a) = 1 + log(a)
                        - CHI_SQUARED: 
                            - f(a) = a^2 - 1
                            - f*(b) = b^2/4 + 1
                            - df(a) = 2a
                        - SQUARED_HELLINGER:
                            - f(a) = (1 - sqrt(a))^2
                            - f*(b) = -b/b-1, b < 1; infty otherwise.
                            - df(a) = 1 - 1/sqrt(a)

    Returns
    -------
    None.

    """
    
    minsize = -maxsize - 1
    
    if bias_correct:
        biases, reliability, scores, iteration = em_estimate_parameters(grader_dict, student_list, assignment_num, mu, gamma, include_bias=True)
        if not iteration < 1000:
            print("EM estimation procedure did not converge.")
            biases = {student.id: 0 for student in student_list}
            reliability = {student.id: 0 for student in student_list}
    
    for submission, graders in grader_dict.items():
        
        """
        COMPUTING THE SCORES
        
        1) Randomly separate the four agents into pairs
    
        2) For each pair: Choose a penalty task for each agent, score the pair.
    
        (Take an average over this process)
        
        """
        assignment = submission.assignment_number
        
        bonus = submission.student_id
        
        constant_dict = {}
        temp_scores = {}
        
        for grader in graders:
            constant_dict[grader.id] = (len(graders) - 1)
            temp_scores[grader.id] = 0
        
        pairs = combinations(graders, 2)
        
        for pair in pairs:
            one = pair[0]
            two = pair[1]
            
            if bias_correct:
                unregularized_tau_1 = reliability[one.id]
                unregularized_tau_2 = reliability[two.id]
            else:
                unregularized_tau_1 = 1/0.7
                unregularized_tau_2 = 1/0.7
            
            tau_1, tau_2 = regularize_reliability(unregularized_tau_1, unregularized_tau_2)
            
            if bias_correct:
                b_1 = biases[one.id]
                b_2 = biases[two.id]
            else:
                b_1 = 0
                b_2 = 0
            
            bonus_one_grade = one.grades[assignment][bonus]
            bonus_two_grade = two.grades[assignment][bonus]
            
            penalties_one = list(one.grades[assignment].keys()) 
            if assignment in one.penalty_tasks.keys():
                penalties_one += list(one.penalty_tasks[assignment].keys())
            penalties_one.remove(bonus)
            penalties_two = list(two.grades[assignment].keys())
            if assignment in two.penalty_tasks.keys():
               penalties_two += list(two.penalty_tasks[assignment].keys())
            penalties_two.remove(bonus)
            
            i = 0
            found = False
            shuffle(penalties_one)
            while (i < len(penalties_one)) and (not found):
                possible = penalties_one[i]
                if possible not in penalties_two:
                    penalty_one = possible
                    found = True
                elif len(penalties_two) > 1:
                    penalty_one = possible
                    penalties_two.remove(penalty_one)
                    found = True
                i += 1
                
            if not found:
                #print("Pair of students without possible penalty task:", one.id, two.id)
                constant_dict[one.id] -= 1
                constant_dict[two.id] -= 1
                continue
            
            else:
                if penalty_one in one.grades[assignment].keys():
                    penalty_one_grade = one.grades[assignment][penalty_one]
                else:
                    penalty_one_grade = one.penalty_tasks[assignment][penalty_one]
                    
                penalty_two = choice(penalties_two)
                if penalty_two in two.grades[assignment].keys():
                    penalty_two_grade = two.grades[assignment][penalty_two]
                else:
                    penalty_two_grade = two.penalty_tasks[assignment][penalty_two]
            
            if phi_divergence == "TVD":
                conjugate = lambda b: b
                    
            elif phi_divergence == "KL":
                conjugate = lambda b: exp(b - 1)
            
            elif phi_divergence == "CHI_SQUARED":
                conjugate = lambda b: (b*b)/4 + 1
                
            elif phi_divergence == "SQUARED_HELLINGER":
                conjugate = lambda b: (-b)/(b - 1)
            
            penalty_val = evaluate_K(penalty_one_grade, penalty_two_grade, mu, gamma, tau_1, tau_2, b_1, b_2, phi_divergence)
            if phi_divergence == "SQUARED_HELLINGER" and penalty_val == 1:
                penalty_score = minsize
            else:
                penalty_score = conjugate(penalty_val)
                
            if phi_divergence == "SQUARED_HELLINGER" and np.isnan(penalty_score):
                #Conjugate may evaluate to infty/-infty, limit as b -> -infty is -1.
                penalty_score = -1
                    
            bonus_score = evaluate_K(bonus_one_grade, bonus_two_grade, mu, gamma, tau_1, tau_2, b_1, b_2, phi_divergence)
            
            score = bonus_score - penalty_score
                
            temp_scores[one.id] += score
            temp_scores[two.id] += score
            
        for grader in graders:
            constant_inv = constant_dict[grader.id]
            if constant_inv > 0:
                constant = 1/constant_inv
                temp_score = temp_scores[grader.id]
                grader.payment += constant*temp_score
            else:
                grader.num_graded -= 1
            
def compute_K(J, PM, phi_divergence):
    """
    Computes the scoring matrix, which is entrywise equal to df(joint-to-marginal-product ratio) when the (estimated) joint-to-marginal-product ratio is defined, 0 otherwise.
    df is determined by the choice of phi_divergence.
    
    Parameters
    ----------
    J : numpy 2D array (11x11).
        Estimate of the Joint Distribution of reports.
    PM : numpy 2D array (11x11).
        Estimate of the Product of the Marginal Distributions of reports. 
    phi_divergence : str; should be one of the options below.
                     Defines an Phi-Mutual Information measure (phi is denoted by f below).
                        - TVD:
                            - f(a) = 1/2|a - 1|
                            - f*(b) = b if |b| <= 1/2; infty otherwise.
                            - df(a) = 1/2 if a > 1, -1/2 if a < 1, [-1/2, 1/2] if a = 1.
                        - KL: KL-Divergence
                            - f(a) = a log a
                            - f*(b) = exp(b - 1)
                            - df(a) = 1 + log(a)
                        - CHI_SQUARED: 
                            - f(a) = a^2 - 1
                            - f*(b) = b^2/4 + 1
                            - df(a) = 2a
                        - SQUARED_HELLINGER:
                            - f(a) = (1 - sqrt(a))^2
                            - f*(b) = -b/b-1, b < 1; infty otherwise.
                            - df(a) = 1 - 1/sqrt(a)

    Returns
    -------
    K : numpy 2D array (11x11).
        Scoring matrix.

    """
    
    K = np.zeros(shape=(11, 11))
    
    i = 0
    
    for row in PM:
        j = 0
        for entry in row:
            if PM[i, j] != 0:
                value = J[i, j] / PM[i, j]
                
                if phi_divergence == "TVD":
                    K[i, j] = tvd_subdifferential(value)
                    
                elif phi_divergence == "KL":
                    K[i, j] = kl_subdifferential(value)
                    
                elif phi_divergence == "CHI_SQUARED":
                    subdifferential = lambda a: 2*a
                    K[i, j] = subdifferential(value)
                        
                elif phi_divergence == "SQUARED_HELLINGER":
                    K[i, j] = squared_hellinger_subdifferential(value)
                    
            j += 1
        i += 1
    return K

def tvd_subdifferential(a):
    """
    Evaluates the subdifferential of the function phi that corresponds to phi divergence being TVD at the value a.

    Parameters
    ----------
    a : float.

    Returns
    -------
    ans : float.
          1/2 if a > 1, 
          -1/2 if a < 1, 
          0 if a = 1.

    """
    ans = 0
    
    if a > 1:
        ans = 0.5
    elif a < 1:
        ans = -0.5
    
    return ans

def kl_subdifferential(a):
    """
    Evaluates the subdifferential of the function phi that corresponds to phi divergence being KL at the value a.

    Parameters
    ----------
    a : float.

    Returns
    -------
    ans : float.
          1 + log(a) if a > 0,
          -inf otherwise.

    """
    ans = -np.inf
    
    if a > 0:
        ans = 1 + log(a)

    return ans

def squared_hellinger_subdifferential(a):
    """
    Evaluates the subdifferential of the function phi that corresponds to phi divergence being Squared Hellinger distance at the value a.

    Parameters
    ----------
    a : float.

    Returns
    -------
    ans : float.
          1 - 1/sqrt(a) if a > 0,
          -inf otherwise.

    """
    ans = -np.inf
    
    if a > 0:
        ans = 1 - 1/sqrt(a)

    return ans

def evaluate_K(x, y, mu, gamma, tau_1, tau_2, b_1, b_2, phi_divergence):
    """ 
    Evaluates the scoring function K(x, y), which is equal to df(joint-to-marginal-product ratio) when the joint-to-marginal-product ratio is computed using parametric model estimates.
    df is determined by the choice of phi_divergence.
    
    Parameters
    ----------
    x: int 0-10.
       Grader 1's report.
    y: int 0-10.
       Grader 2's report.
    mu : float.
        The mean of the normal approximation of the distribution of true grades.
    gamma : float.
        The precision (i.e. the inverse of the variance) of the normal approximation of the distribution of true grades.
    tau_1: float. 
           Estimated reliability of grader 1
    tau_2: float.
           Estimated reliability of grader 2
    b_1: float.
         Estimated bias of grader 1
    b_2: float.
         Estimated bias of grader 2
    phi_divergence : str; should be one of the options below.
                     Defines an Phi-Mutual Information measure (phi is denoted by f below).
                        - TVD:
                            - f(a) = 1/2|a - 1|
                            - f*(b) = b if |b| <= 1/2; infty otherwise.
                            - df(a) = 1/2 if a > 1, -1/2 if a < 1, [-1/2, 1/2] if a = 1.
                        - KL: KL-Divergence
                            - f(a) = a log a
                            - f*(b) = exp(b - 1)
                            - df(a) = 1 + log(a)
                        - CHI_SQUARED: 
                            - f(a) = a^2 - 1
                            - f*(b) = b^2/4 + 1
                            - df(a) = 2a
                        - SQUARED_HELLINGER:
                            - f(a) = (1 - sqrt(a))^2
                            - f*(b) = -b/b-1, b < 1; infty otherwise.
                            - df(a) = 1 - 1/sqrt(a)


    Returns
    -------
    score: float.
            K(x, y)

    """
    
    sig2 = 1/gamma
    
    val = (sig2 + (1/tau_1))*(sig2 + (1/tau_2))
    
    eps = np.array([x - (mu + b_1), y - (mu  + b_2)])
    L = np.array([[sig2*(sig2 + (1/tau_2)), -val],[-val, sig2*(sig2 + (1/tau_1))]])
    
    interim = np.dot(eps, L)
    G = np.dot(interim, eps)
    
    num = val
    denom = val - (sig2**2)
    
    coeff = sqrt(num/denom)
    
    exp_num = -0.5*sig2*tau_1*tau_2*G
    exp_denom = (sig2*tau_1 + sig2*tau_2 + 1)*val
    
    exp_val = exp(exp_num/exp_denom)
    
    jp = coeff*exp_val
    
    if np.isnan(jp):
        print("JP is nan.")
        jp = 0
                
    if phi_divergence == "TVD":
        score = tvd_subdifferential(jp)
                    
    elif phi_divergence == "KL":
        score = kl_subdifferential(jp)
                    
    elif phi_divergence == "CHI_SQUARED":
        subdifferential = lambda a: 2*a
        score = subdifferential(jp)
                        
    elif phi_divergence == "SQUARED_HELLINGER":
        score = squared_hellinger_subdifferential(jp)
                    
    return score

def regularize_reliability(rel_1, rel_2):
    """
    Regularizes the reliability estimates for a pair of graders.

    Parameters
    ----------
    rel_1 : float. 
            Estimated reliability of grader 1.
    rel_2 : float. 
            Estimated reliability of grader 2.

    Returns
    -------
    t_1 : float. 
          Regularized reliability of grader 1.
    t_2 : float. 
          Regularized reliability of grader 2.
    """
    
    val = 1/0.7
    # val = 1/1.05
    
    p = 0.00
    
    t_1 = p*rel_1 + (1-p)*val
    t_2 = p*rel_2 + (1-p)*val
    
    return t_1, t_2