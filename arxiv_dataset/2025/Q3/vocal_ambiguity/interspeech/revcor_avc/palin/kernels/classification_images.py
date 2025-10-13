#!/usr/bin/env python
'''
PALIN toolbox v0.2
February 2024, Aynaz Adl Zarrabi, JJ Aucouturier, Paige Tuttosi (CNRS/UBFC)

Functions for kernel calculating method in Classification images
'''

import pandas as pd
import numpy as np
from scipy import stats

def compute_kernel_diff(data_df,trial_ids=['experimentor','type','subject','session'], dimension_ids=['segment'],response_id='response', value_id='pitch', normalize=True):
  ''' computes first-order temporal kernels for each participant using the classification image, ie. 
      mean(stimulus features classified as positive) - mean(stimulus features classified as negative)'''

  # for each participant, average stimulus features (e.g. mean pitch for each segment) separately for positive and negative responses, and subtract positives - negatives 
  dimension_mean_value = data_df.groupby(trial_ids+[dimension_ids]+[response_id])[value_id].mean().reset_index()
  positives = dimension_mean_value.loc[dimension_mean_value[response_id] == True].reset_index()
  negatives = dimension_mean_value.loc[dimension_mean_value[response_id] == False].reset_index()
  kernels = pd.merge(positives, negatives, on=trial_ids+[dimension_ids])
  kernels['kernel_value'] = kernels['%s_x'%value_id] - kernels['%s_y'%value_id]

  if(normalize):
    # Kernel are then normalized for each participant/session by dividing them 
    # by the square root of the sum of their squared values.
    kernels['square_value'] = kernels['kernel_value']**2
    for_norm = kernels.groupby(trial_ids)['square_value'].mean().reset_index()
    kernels = pd.merge(kernels, for_norm, on=trial_ids)
    kernels['norm_value'] = kernels['kernel_value']/np.sqrt(kernels['square_value_y'])
    kernels.drop(columns=['index_x','%s_x'%response_id,'%s_x'%value_id,'index_y','%s_y'%response_id,'%s_y'%value_id,'square_value_x', 'square_value_y'], inplace=True)
  
  return kernels

def compute_kernel(data_df,trial_ids=['experimentor','type','subject','session'], dimension_ids=['segment'],response_id='response', value_id='pitch', normalize=True):
  ''' computes first-order temporal kernels for each participant using the classification image
  creating a kernel for each response where the positives were response = 1 and negatives were response = 0'''

  # for each participant, average stimulus features (e.g. mean pitch for each segment) separately for positive and negative responses, and subtract positives - negatives 
  dimension_mean_value = data_df.groupby(trial_ids+[dimension_ids]+[response_id])[value_id].mean().reset_index()
  positives = dimension_mean_value.loc[dimension_mean_value[response_id] == True].reset_index()
  negatives = dimension_mean_value.loc[dimension_mean_value[response_id] == False].reset_index()
  positives['kernel_value'] = positives['%s'%value_id]
  negatives['kernel_value'] = negatives['%s'%value_id]

  if(normalize):
    # Kernel are then normalized for each participant/session by dividing them 
    # by the square root of the sum of their squared values.
    positives['square_value'] = positives['kernel_value']**2
    for_norm = positives.groupby(trial_ids)['square_value'].mean().reset_index()
    positives = pd.merge(positives, for_norm, on=trial_ids)
    positives['norm_value'] = positives['kernel_value']/np.sqrt(positives['square_value_y'])
    positives.drop(columns=['index','%s'%response_id,'%s'%value_id,'square_value_x', 'square_value_y'], inplace=True)

    negatives['square_value'] = negatives['kernel_value']**2
    for_norm = negatives.groupby(trial_ids)['square_value'].mean().reset_index()
    negatives = pd.merge(negatives, for_norm, on=trial_ids)
    negatives['norm_value'] = negatives['kernel_value']/np.sqrt(negatives['square_value_y'])
    negatives.drop(columns=['index','%s'%response_id,'%s'%value_id,'square_value_x', 'square_value_y'], inplace=True)
  
  return positives, negatives

def one_sample(data_df_positives, data_df_negatives, dimension_ids=['segment'], value_id='pitch', response_dict = {'positives' : 'pull', 'negatives' : 'poule'}, mu = 0):
  # run one sample t-test
  segment_values = data_df_positives[dimension_ids].unique()
  t_test_results = {response_dict['negatives'] : {}, response_dict['positives' ]: {}}
  for segment in segment_values:
    negatives_segment = data_df_negatives.loc[data_df_negatives[dimension_ids] == segment].reset_index()
    positives_segment = data_df_positives.loc[data_df_positives[dimension_ids] == segment].reset_index()
    t_test_results[response_dict['negatives']][segment] = stats.ttest_1samp(negatives_segment[value_id], popmean=mu)
    t_test_results[response_dict['positives' ]][segment] = stats.ttest_1samp(positives_segment[value_id], popmean=mu)

  return(t_test_results)

def two_sample(data_df_positives, data_df_negatives, dimension_ids=['segment'], value_id='pitch'):
  # run one sample t-test with mu = 0
  segment_values = data_df_positives[dimension_ids].unique()
  t_test_results = {}
  for segment in segment_values:
    negatives_segment = data_df_negatives.loc[data_df_negatives[dimension_ids] == segment].reset_index()
    positives_segment = data_df_positives.loc[data_df_positives[dimension_ids] == segment].reset_index()
    t_test_results[segment] = stats.ttest_ind(positives_segment[value_id], negatives_segment[value_id], equal_var=False)

  return(t_test_results)

def paired_sample(data_df_positives, data_df_negatives, dimension_ids=['segment'], value_id='pitch'):
  # run one sample t-test with mu = 0
  segment_values = data_df_positives[dimension_ids].unique()
  t_test_results = {}
  for segment in segment_values:
    negatives_segment = data_df_negatives.loc[data_df_negatives[dimension_ids] == segment].reset_index()
    positives_segment = data_df_positives.loc[data_df_positives[dimension_ids] == segment].reset_index()
    t_test_results[segment] = stats.ttest_rel(positives_segment[value_id], negatives_segment[value_id])

  return(t_test_results)