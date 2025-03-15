import sys
from os import listdir
from os.path import isdir, isfile, join
import math
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
import argparse
import homoglyphs as hg
import statsmodels.api as sm
import numpy as np

sys.path.append('./src/')
from utils import constants
from utils import utils
from h04_analysis.print_bert_correlations import get_data

aspect = {
    'height': 7,
    'font_scale': 1.8,
    'labels': True,
    'name_suffix': '',
    'ratio': 1.0625,
    # 'ratio': 2.125,
}
sns.set_palette("muted")
sns.set_context("notebook", font_scale=aspect['font_scale'])
mpl.rc('font', family='serif', serif='Times New Roman')
sns.set_style({'font.family': 'serif', 'font.serif': 'Times New Roman'})

def print_bert_vs_wordnet_correlations(df):
    df['corr'] = -10
    langs = df.Language.unique()
    langs = sorted(langs)
    print('\n\nCorrelation BERT vs Wordnet polysemy correlation  (Pearson, Spearman)')
    for lang in langs:
        df_lang_orig = df[df.Language == lang]
        df_lang = df_lang_orig

        corr_spear, p_value = stats.spearmanr(df_lang.n_senses_log, df_lang.bert_polysemy)

        p_value_spear = '~~~'
        if p_value < 0.01:
            p_value_spear = '$^{**}$'
        elif p_value < 0.1:
            p_value_spear = '$^*$~~'

        corr_pearson, p_value = stats.pearsonr(df_lang.n_senses_log, df_lang.bert_polysemy)
        df.loc[df.Language == lang, 'corr'] = corr_spear

        p_value_pearson = '~~~'
        if p_value < 0.01:
            p_value_pearson = '$^{**}$'
        elif p_value < 0.1:
            p_value_pearson = '$^*$~~'
        # print('Pearson:', stats.pearsonr(df.surprisal, df.bert_polysemy))

        print('~~~~%s & %d & %.2f%s  & %.2f%s \\\\' % (lang, df_lang_orig.shape[0], corr_pearson, p_value_pearson, corr_spear, p_value_spear))


def plot_polysemy(df):
    df.sort_values('Language', inplace=True)

    n_bins = 100
    e_sorted = sorted(df['wordnet_polysemy'].values)
    bins = np.unique(np.array([x for x in e_sorted[int(len(e_sorted)/(2*n_bins))::int(len(e_sorted)/n_bins)]]))
    fig = sns.lmplot('wordnet_polysemy', 'bert_polysemy', df, hue='Language', x_bins=bins, logx=True,
                     height=aspect['height'], aspect=aspect['ratio'])
    fig.set(xscale="log")

    plt.ylabel('Lexical Ambiguity (bits)')
    plt.xlabel('# Senses in WordNet')
    fig._legend.set_title(None)

    plt.xlim([.9, 101])
    plt.ylim([0, df.bert_polysemy.max() * 1.05])

    fig.savefig('plots/full-plot-polysemy.pdf', bbox_inches="tight")
    plt.close()


def print_multivariate_wordnet_vs_bert_analysis(df):
    print('\n\nGetting multivariate polysemy parameters (bert ~ wordnet + frequency)')
    langs = df.Language.unique()
    langs = sorted(langs)
    for lang in langs:
        df_lang = df[df.Language == lang].copy()
        df_lang['n_senses_log'] = (df_lang['n_senses_log'] - df_lang['n_senses_log'].mean()) / df_lang['n_senses_log'].std()
        df_lang['frequency'] = (df_lang['frequency'] - df_lang['frequency'].mean()) / df_lang['frequency'].std()
        df_lang['bert_polysemy'] = (df_lang['bert_polysemy'] - df_lang['bert_polysemy'].mean()) / df_lang['bert_polysemy'].std()

        params = ['n_senses_log', 'frequency']
        X = df_lang[params]
        X = sm.add_constant(X)
        y = df_lang['bert_polysemy']
        model = sm.OLS(y, X).fit()

        p_strings = {}
        for param in params:
            p_strings[param] = '~~~'
            if model.pvalues[param] < 0.01:
                p_strings[param] = '$^{**}$'
            elif model.pvalues[param] < 0.1:
                p_strings[param] = '$^*$~~'

        print('~~~~%s & %d & %.2f%s & %.2f%s \\\\' % \
              (lang, df_lang.shape[0],
               model.params['n_senses_log'], p_strings['n_senses_log'],
               model.params['frequency'], p_strings['frequency'],))

def print_wordnet_vs_surprisal_correlations(df):
    langs = df.Language.unique()
    langs = sorted(langs)
    print('\n\nGetting Wordnet Ambiguity vs Surprisal correlation (Pearson, Spearman)')

    for lang in langs:
        df_lang_orig = df[df.Language == lang]
        df_lang = df_lang_orig

        corr_spear, p_value = stats.spearmanr(df_lang.n_senses_log, df_lang['surprisal'])

        p_value_spear = '~~~'
        if p_value < 0.01:
            p_value_spear = '$^{**}$'
        elif p_value < 0.1:
            p_value_spear = '$^*$~~'

        corr_pearson, p_value = stats.pearsonr(df_lang.n_senses_log, df_lang['surprisal'])

        p_value_pearson = '~~~'
        if p_value < 0.01:
            p_value_pearson = '$^{**}$'
        elif p_value < 0.1:
            p_value_pearson = '$^*$~~'

        print('~~~~%s & %d & %.2f%s  & %.2f%s \\\\' % (lang, df_lang_orig.shape[0], corr_pearson, p_value_pearson, corr_spear, p_value_spear))


def plot_kde(x, y, data, y_axis, title):
    fig = sns.lmplot(y, x, data, fit_reg=False,
                     height=aspect['height'], aspect=aspect['ratio'],
                     ci=None)
    sns.kdeplot(data[y], data[x])
    sns.regplot(data[y], data[x], robust=True, scatter=False,
                ci=None)

    plt.xlabel(y_axis)
    plt.ylabel('# Senses in WordNet (bits)')

    if y == 'surprisal':
        plt.xlim([-0.5, data[y].max() + .5])

    fig.savefig(title, bbox_inches="tight")
    plt.close()

def plots_kdes(df, langs):
    for lang in langs:
        df_lang = df[df.Language == lang]

        plot_kde(y='surprisal', x='n_senses_log', data=df_lang,
            y_axis='Contextual Uncertainty (bits)', title='plots/lang-kde-wordnet_surprisal-%s.png' % lang)


def main():
    df = get_data(check_wordnet=True)
    langs = df.Language.unique()
    langs = sorted(langs)


    print_bert_vs_wordnet_correlations(df)
    plot_polysemy(df)
    print_multivariate_wordnet_vs_bert_analysis(df)

    print_wordnet_vs_surprisal_correlations(df)
    plots_kdes(df, langs)


if __name__ == "__main__":
    main()
