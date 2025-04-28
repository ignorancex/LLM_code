"""

Plot bar-graphs that show (relative) improvement in precision, f-1 and (relative) reduction in recall

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import os
import pandas as pd
import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt

plt.rc('axes', titlesize=16) #fontsize of the title
plt.rc('axes', labelsize=16) #fontsize of the x and y labels
plt.rc('xtick', labelsize=16) #fontsize of the x tick labels
plt.rc('ytick', labelsize=16) #fontsize of the y tick labels

def create_relative_change_df(umc_df, lits_df, metric='Precision'):

    improvement_umc = (np.abs(umc_df['{} (After)'.format(metric)] - umc_df['{} (Before)'.format(metric)])/umc_df['{} (Before)'.format(metric)])*100
    improvement_lits = (np.abs(lits_df['{} (After)'.format(metric)] - lits_df['{} (Before)'.format(metric)])/lits_df['{} (Before)'.format(metric)])*100

    improvement_df = pd.concat([umc_df['Config'], improvement_umc, umc_df['Error {}'.format(metric)]*100, improvement_lits, lits_df['Error {}'.format(metric)]*100], axis=1)
    improvement_df = improvement_df.reset_index(drop=True)
    improvement_df.columns = ['Uncertainty Estimation', 'UMC', 'Error UMC', 'LITS', 'Error LITS']

    return improvement_df


def create_relative_change_df_logsum(umc_df, lits_df, metric='Precision'):

    improvement_df = pd.concat([umc_df['Config'], umc_df[metric], lits_df[metric]],
                               axis=1)

    improvement_df.columns = ['Uncertainty Estimation', 'UMC', 'LITS']

    return improvement_df


def plot_bar_chart(umc_df, lits_df, fname='improvement'):

    precision_improvement = create_relative_change_df(umc_df, lits_df,
                                                      metric='Precision')

    recall_change = create_relative_change_df(umc_df, lits_df,
                                              metric='Recall')

    f1_improvement = create_relative_change_df(umc_df, lits_df,
                                               metric='F1')

    fig, ax = plt.subplots(1, 3, figsize=(18, 4))


    precision_improvement.plot(x="Uncertainty Estimation",
                               y=["UMC", "LITS"],
                               kind="bar",
                               yerr=[precision_improvement['Error UMC'], precision_improvement['Error LITS']],
                               capsize=4,
                               legend=True,
                               ax=ax[0],
                               title='% improvement in precision (higher=better)',
                               ylim=(0, 200))



    recall_change.plot(x="Uncertainty Estimation",
                       y=["UMC", "LITS"],
                       kind="bar",
                       yerr=[recall_change['Error UMC'], recall_change['Error LITS']],
                       capsize=4,
                       legend=True,
                       ax=ax[1],
                       title='% reduction in recall (lower=better)',
                       ylim=(0, 200))


    f1_improvement.plot(x="Uncertainty Estimation",
                        y=["UMC", "LITS"],
                        kind="bar",
                        yerr=[f1_improvement['Error UMC'], f1_improvement['Error LITS']],
                        capsize=4,
                        legend=True,
                        ax=ax[2],
                        title='% improvement in F1-score (higher=better)',
                        ylim=(0, 200))

    fig.savefig('/data/ishaan/{}.pdf'.format(fname), bbox_inches='tight')
    fig.savefig('/data/ishaan/{}.png'.format(fname), bbox_inches='tight')


def plot_bar_chart_logsum(umc_df, lits_df, fname='improvement'):

    precision_improvement = create_relative_change_df_logsum(umc_df, lits_df,
                                                      metric='Precision')

    recall_change = create_relative_change_df_logsum(umc_df, lits_df,
                                                     metric='Recall')

    f1_improvement = create_relative_change_df_logsum(umc_df, lits_df,
                                                      metric='F1')

    fig, ax = plt.subplots(1, 3, figsize=(18, 4))


    precision_improvement.plot(x="Uncertainty Estimation",
                               y=["UMC", "LITS"],
                               kind="bar",
                               capsize=4,
                               legend=True,
                               ax=ax[0],
                               title='% improvement in precision (higher=better)',
                               ylim=(0, 200))


    recall_change.plot(x="Uncertainty Estimation",
                       y=["UMC", "LITS"],
                       kind="bar",
                       capsize=4,
                       legend=True,
                       ax=ax[1],
                       title='% reduction in recall (lower=better)',
                       ylim=(0, 200))


    f1_improvement.plot(x="Uncertainty Estimation",
                        y=["UMC", "LITS"],
                        kind="bar",
                        capsize=4,
                        legend=True,
                        ax=ax[2],
                        title='% improvement in F1-score (higher=better)',
                        ylim=(0, 200))

    fig.savefig('/data/ishaan/{}.pdf'.format(fname), bbox_inches='tight')
    fig.savefig('/data/ishaan/{}.png'.format(fname), bbox_inches='tight')

if __name__ == '__main__':


    umc_df = pd.read_pickle('/data/ishaan/umc/dataset_metrics.pkl')
    lits_df = pd.read_pickle('/data/ishaan/lits/dataset_metrics.pkl')

    plot_bar_chart(umc_df=umc_df,
                   lits_df=lits_df,
                   fname='improvement')

    umc_df_shape = pd.read_pickle('/data/ishaan/umc/dataset_metrics_shape.pkl')
    lits_df_shape = pd.read_pickle('/data/ishaan/lits/dataset_metrics_shape.pkl')

    plot_bar_chart(umc_df=umc_df_shape,
                   lits_df=lits_df_shape,
                   fname='improvement_shape')

    umc_df_shape = pd.read_pickle('/data/ishaan/umc/dataset_metrics_intensity.pkl')
    lits_df_shape = pd.read_pickle('/data/ishaan/lits/dataset_metrics_intensity.pkl')

    plot_bar_chart(umc_df=umc_df_shape,
                   lits_df=lits_df_shape,
                   fname='improvement_intensity')

    umc_df_logsum = pd.read_pickle('/data/ishaan/umc/logsum_results.pkl')
    lits_df_logsum = pd.read_pickle('/data/ishaan/lits/logsum_results.pkl')

    plot_bar_chart_logsum(umc_df=umc_df_logsum,
                          lits_df=lits_df_logsum,
                          fname='improvement_logsum')
