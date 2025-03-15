""" 
    Distribution Generators for use in Performance Estimation for Regression Models
"""
import random
import numpy as np
import pandas as pd

######################################################################
def extract_distribution_from_sample(filepath):
    """ Extract a sample of target values from a file on the given path """
    try:
        df = pd.read_csv(filepath)
        indecies = [ np.issubdtype(x, np.number) for x in df.dtypes]
        only_numeric = df.loc[:,indecies]
        if len(only_numeric.columns)==0:
            return [], " your sample file: Please provide a CSV with a column of numeric values."
        else:
            return list(only_numeric.iloc[:,0]), ""
    #try:
    #    df = pd.read_csv('evaluate.py')
    except pd.errors.ParserError:
        return [], "Problem Parsing your sample file: Please provide a CSV with a column of numeric values."
    except:
        return [], "There was an unanticipated problem with your file. Please provide a CSV with a column of numeric values."

######################################################################
def produce_distribution_sample(mean, max, min):
    """ Given some simple parameters we generate a sample of target values. TODO: This needs work """
    # START WITH SAMPLES BETWEEN MIN AND MAX
    baseline = generate_min_max_baseline(min, max)
    threshold = (max-min)/200
    enhanced = resample_toward_mean(baseline, mean, threshold)
    return enhanced, ""

######################################################################
def resample_toward_mean(baseline, mean, threshold):
    current_mean = np.mean(baseline)
    rez = baseline.copy()
    print("Target Mean:", mean, " baseline sample: ", len(baseline) )
    while abs(mean - current_mean) > threshold:
        temp = rez.copy()
        new_sample = random.sample(rez, 1)[0]
        temp.append(new_sample)
        temp_mean = np.mean(temp)
        if abs(mean-temp_mean)<abs(mean - current_mean):
            current_mean = temp_mean
            rez = temp
            print("Sample accepted. New Mean:", current_mean)
        else:
            print("Sample rejected.")
    return rez

######################################################################
def generate_min_max_baseline(min, max, sample_size=1000):
    difference = max-min
    return [min + (difference * x/(sample_size-1)) for x in range(sample_size)]


########################################################################
def generate_candidate_predictions(dist):
    sigma = np.std(dist)
    factors = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    candidates = []
    for index, factor in enumerate(factors):
        for variant in range(10):
            candidates.append( copy_with_noise(dist, factor*sigma) )
    return candidates

########################################################################
def copy_with_noise(dist, scale):
    rez = list(map(lambda x: x + scale*np.random.normal(), dist))
    return rez

