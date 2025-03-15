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
# import homoglyphs as hg
import statsmodels.api as sm

sys.path.append('./src/')
from utils import utils


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


def get_result(lang, fname, is_var=True, check_wordnet=False):
    alphabet = utils.get_alphabet(lang)

    df = pd.read_csv(fname, sep='\t')
    if alphabet is not None:
        df = df[df.word.apply(lambda x: all([char in alphabet for char in str(x)]))].copy()
    df['Language'] = lang

    df = df[df.in_vocab == True]
    if check_wordnet:
        df = df[df.wordnet_polysemy > 0]
        df['n_senses_log'] = df.wordnet_polysemy.apply(math.log2)

    if is_var:
        df = df[df.bert_polysemy > -350]
    else:
        df = df[df.bert_polysemy > -35000]
    df['frequency'] = df['n_samples']

    print('Negative MIs: ', (df.surprisal > df.unigram).sum(), ' of ', df.shape[0])
    df['mi_context'] = df['unigram'] - df['surprisal']
    df['length'] = df.word.apply(lambda x: len(str(x)))

    return df.copy()


def get_data(check_wordnet=False):
    results_path = 'results/'
    corr_fname = 'surprisal_var_polysemy.tsv'
    langs = [f for f in listdir(results_path) if isdir(join(results_path, f)) and ' ' not in f]
    print(langs)
    MIN_WORDS = 100

    # Get data
    dfs = []
    for lang in langs:
        corr_file = '%s/%s/%s' % (results_path, lang, corr_fname)
        print(corr_file)
        if isfile(corr_file):
            df = get_result(lang, corr_file, check_wordnet=check_wordnet)

            if df.shape[0] >= MIN_WORDS:
                dfs += [df]

    return pd.concat(dfs)


def print_correlations(df):
    # Get correlations
    df['corr'] = -10
    langs = df.Language.unique()
    langs = sorted(langs)

    corr_var = 'surprisal'
    print('\nGetting BERT polysemy vs Surprisal correlation  (Pearson, Spearman)')

    for lang in langs:
        df_lang = df[df.Language == lang]

        corr_spear, p_value = stats.spearmanr(df_lang[corr_var], df_lang.bert_polysemy)

        p_value_spear = '~~~'
        if p_value < 0.01:
            p_value_spear = '$^{**}$'
        elif p_value < 0.1:
            p_value_spear = '$^*$~~'

        corr_pearson, p_value = stats.pearsonr(df_lang[corr_var], df_lang.bert_polysemy)
        df.loc[df.Language == lang, 'corr'] = corr_spear

        p_value_pearson = '~~~'
        if p_value < 0.01:
            p_value_pearson = '$^{**}$'
        elif p_value < 0.1:
            p_value_pearson = '$^*$~~'

        df.loc[df.Language == lang, 'corr-%s-%s' % (corr_var, 'pear')] = '%.2f%s' % (corr_pearson, p_value_pearson)
        df.loc[df.Language == lang, 'corr-%s-%s' % (corr_var, 'spear')] = '%.2f%s' % (corr_spear, p_value_spear)

        print('~~~~%s & %d & %.2f%s  & %.2f%s \\\\' % (lang, df_lang.shape[0], corr_pearson, p_value_pearson, corr_spear, p_value_spear))


def plot_kde(x, y, data, x_axis, y_axis, title, fit_reg=True):
    fig = sns.lmplot(x, y, data, fit_reg=False,
                     height=aspect['height'], aspect=aspect['ratio'])
    sns.kdeplot(data[x], data[y])

    if fit_reg:
        sns.regplot(data[x], data[y], robust=True, scatter=False,
                    ci=None)

    plt.xlabel(x_axis)
    plt.ylabel(y_axis)

    fig.savefig(title, bbox_inches="tight")
    plt.close()


# Plot Kernel density estimates for all languages
def plot_kdes(df, langs):
    print('Plotting KDEs')
    for lang in langs:
        df_lang = df[df.Language == lang]

        plot_kde(x='surprisal', y='bert_polysemy', data=df_lang,
            x_axis='Contextual Uncertainty (bits)', y_axis='Lexical Ambiguity (bits)',
            title='plots/lang-kde-bert_polysemy_surprisal-%s.png' % lang)


def plot_surprisals(df):
    print('Plotting surprisals')

    df.sort_values('corr', inplace=True, ascending=False)
    df.sort_values('Language', inplace=True, ascending=True)

    fig = sns.lmplot('surprisal', 'bert_polysemy', df, hue='Language', scatter=False, truncate=False,
                    height=aspect['height'], aspect=aspect['ratio'])

    plt.ylabel('Lexical Ambiguity (bits)')
    plt.xlabel('Contextual Uncertainty (bits)')
    plt.ylim([-500, 500])

    fig._legend.set_title(None)

    fig.savefig('plots/full-plot-surprisal.pdf', bbox_inches="tight")
    plt.close()


def main():
    df = get_data()
    langs = df.Language.unique()
    langs = sorted(langs)

    print_correlations(df)
    print()
    plot_kdes(df, langs)
    plot_surprisals(df)


if __name__ == "__main__":
    main()
