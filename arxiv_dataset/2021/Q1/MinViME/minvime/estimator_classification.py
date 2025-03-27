""" 
    Estimator
    Functions for estimating model performance requirements from business criteria
"""
import numpy as np
import math

######################################################################

def estimate_binary_model_requirements(tp, fp, tn, fn, cases, baserate, minroi=0):
    """
    Determine the minimal performance characteristics of a binary classification model

    :param tp: The benefit of a True Positive Prediction
    :type tp: float, required 
    :param fp: The cost of a False Positive Prediction
    :type fp: float, required 
    :param tn: The benefit of a True Negative Prediction
    :type tn: float, required 
    :param fn: The cost of a False Negative Prediction
    :type fn: float, required 

    :param cases: The number of events/cases that occur within the period of analysis.
    :type cases: integer, required 
    :param baserate: The rate at which the event being predicted occurs
    :type baserate: float, required 
    :param minroi: The minimum required ROI for the model, defaults to 0.0
    :type minroi: float, optional

    *Returns a tuple containing
    min_auc, 
    min_precision, 
    min_recall, 
    fprates 
    tprs

    """

    beta_range = [2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100 ]
    alpha_range = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 
                   0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99]
    fprates = [0.0, 0.00001, 0.0001, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.015, 
               0.02,0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 
               0.075, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
               0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93, 0.94, 
               0.95, 0.96, 0.97, 0.98, 0.99, 0.999, 0.9999, 1.0]
    # PLACEHOLDERS FOR THE RETURN VALUES
    min_auc = 1.0
    min_precision = 1.0
    min_recall = 1.0
    tprs = np.array([1.0 for x in fprates])
    current_min_roi = 9999999999999
    num_pos = cases * baserate
    num_neg = cases - num_pos 
    # #############################################
    # Iterate over all combinations of exponent and
    # alpha mix variables so that we can simulate a
    # range of ROC curves.
    ###############################################
    combinations = 0
    for b_i in range( len(beta_range) ):
        beta = beta_range[b_i]
        for alpha_i in range( len(alpha_range) ):
            alpha = alpha_range[alpha_i]
            auc, x, y = generate_roc_auc(fprates, alpha, beta)
            roi, precision, recall = calculate_peak_roi(
                fprates, y, tp, fp, tn, fn, num_pos, num_neg
            )
            if (auc <= min_auc) & (roi >= minroi):
                min_auc = auc
                min_precision = precision
                min_recall = recall
                current_min_roi = roi
                tprs = y
            combinations =  combinations + 1
    #print("Tested ", combinations, " different AUC plots")
    #print("Number of Exponents", len(beta_range))
    #print("Number of Alpha Weights", len(alpha_range))
    return min_auc, min_precision, min_recall, np.array(fprates), tprs

######################################################################
def generate_roc_auc(fprates, alpha, beta):
    formula = 'alpha*(-(x-1)**(2*beta)+1)+(1-alpha)*x'
    x = np.array(fprates)
    y = eval(formula)
    auc = calculate_auc(x, y)
    return auc, x, y


######################################################################

def calculate_peak_roi(fprates, tprates, tp, fp, tn, fn, num_pos, num_neg):
   """ Calculate the maximal ROI for a given ROC curve (defined by vectors of FPR and TPR) """
   roi = -99999999999999
   result_precision = 0.0
   result_recall = 0.0
   for index in range(len(fprates)):
      false_positive_rate = fprates[index]
      true_positive_rate = tprates[index]
      temp = (num_neg * (1-false_positive_rate) * tn) +\
      (num_neg * false_positive_rate * fp ) +\
      (num_pos * true_positive_rate * tp ) +\
      (num_pos * (1-true_positive_rate) * fn)
      if temp>roi:
         roi = temp
         tps = num_pos * true_positive_rate
         fps = num_neg * false_positive_rate
         if (tps+fps) > 0:
             result_precision = tps/(tps+fps)
         else:
             result_precision = 0
         result_recall = true_positive_rate
   return roi,result_precision,result_recall

######################################################################

def calculate_auc(fprates, tprates):
    """ Calculate the AUC of given ROC curve (defined by vectors of FPR and TPR) """
    curr_fprate = fprates[0]
    curr_tprate = tprates[0]
    area = 0
    for index in range( 1, len(fprates) ):
       next_fprate = fprates[index]
       next_tprate = tprates[index]
       x_delta = next_fprate - curr_fprate
       y_mean = (curr_tprate + next_tprate)/2
       area = area + (x_delta*y_mean)
       curr_fprate = fprates[index]
       curr_tprate = tprates[index]
    return area

########################################################################
 
def estimate_intervention_requirements(cases, baserate, cost, payoff, payback, succrate, backfire):
   if cost < 0:
      cost = 0 - cost
   tp = payoff * succrate - cost
   fp = payback * backfire - cost
   tn = 0
   fn = 0
   return tp, fp, tn, fn


########################################################################

def simplicity_estimate(tp, fp, cases, baserate, minroi=0):
    minp = minroi / tp
    fp = 0 - fp # THE COST IS COLLECTED AS A NEGATIVE NUMBER FROM THE APPLICATION
    p = round(cases * baserate)
    n = cases - p
    if p <= minp:
        return 0.0
    total_area = (p*n)
    tri_y = (p-minp) 
    tri_x = (tp/fp)*(p-minp) 
    temp = nth_triangle(tri_y)
    if tri_x > n:
        temp = temp - nth_triangle(tri_x-n)
    return temp / total_area 

def nth_triangle(n):
    return (math.pow(n,2) + n)/2

########################################################################


