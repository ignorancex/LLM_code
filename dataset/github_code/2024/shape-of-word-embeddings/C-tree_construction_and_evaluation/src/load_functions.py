import json
import pickle
import ete3

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


def load_bars(filename):
    bars = []
    bars_dim = []
    with open(filename, 'r') as file:
        for row in file:
            row = row.strip()
            if len(row) == 0:
                bars.append(np.array(bars_dim))
            elif len(row) == 1:
                bars_dim = []
            else:
                bars_dim.append(tuple(map(float, row.split())))
        bars.append(np.array(bars_dim))

    return bars


def load_strictly_lower_triangular_matrix(filename, restrict_labels=None):
    """Load a strictly triangular matrix from a file. The file format is as follows:
        - the first line is a list of space-separated labels (#labels = #rows = #columns)
        - after that there is #labels-1 rows with 1, 2, ..., #labels-1 space-separated numbers, each
          representing a row in the strictly lower triangular matrix
        If restrict_labels is a list of labels, return the matrix restricted to those labels.
    """
    with open(filename, 'r') as file:
        labels = file.readline().strip().split()
        matrix = np.zeros((len(labels), len(labels)), dtype='float64')
        for i, row in enumerate(file):
            for j, num in enumerate(row.strip().split()):
                matrix[i + 1, j] = float(num)
                matrix[j, i + 1] = float(num)
    if restrict_labels is None:
        return matrix, labels
    else:
        return labeled_matrix_minor(matrix, labels, restrict_labels)


def load_combination_of_strictly_lower_triangular_matrices(*filenames, restrict_labels=None, normalise=True):
    """Load several strictly triangular matrices from a file and combine them"""
    if len(filenames) == 0:
        raise TypeError("Missing a positional argument: at least 1 filename must be given")
    matrices = []
    languages = []
    for filename in filenames:
        mat, lang = load_strictly_lower_triangular_matrix(filename, restrict_labels=restrict_labels)
        matrices.append(mat)
        languages.append(lang)
    if any(languages[0] != lang for lang in languages):
        raise ValueError("The matrices do not contain the same languages.")
    if normalise:
        matrices = [mat / np.mean(mat) for mat in matrices]
    return sum(matrices), languages[0]


def labeled_matrix_minor(matrix, labels, desired_labels):
    """Return a pair: matrix composed of the rows and columns labeled by desired_labels,
    and the desired_labels in the order matching the returned matrix.

    The inputted matrix should be a numpy array. The labels should be unique."""
    if not set(desired_labels) <= set(labels):
        raise KeyError("The desired labels is not a subset of the labels.")
    if len(set(desired_labels)) != len(desired_labels):
        raise ValueError("The desired labels have to be a list of _unique_ elements -- duplicates detected.")
    indices = sorted([labels.index(lab) for lab in desired_labels])
    return matrix[indices][:, indices], [labels[i] for i in indices]


def get_wiki_articles_table(source='20231108'):
    if source == '20231108':
        return pd.read_csv('resources/wiki_articles_df_20231108.csv')
    elif source == 'current':
        url = 'https://meta.wikimedia.org/wiki/List_of_Wikipedias'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'wikitable'})
        return pd.read_html(str(table))[0]
    else:
        raise ValueError('Invalid source parameter.')


def load_languages_with_embedding(source= 'resources/languages_dict.pickle'):
    with open(source, 'rb') as file:
        return set(pickle.load(file))


class EthnologueLanguageTree:
    def __init__(self, resources= 'resources/'):
        self.ethnologue_tree = self.load_ethnologue_tree(
            resources + '/ethnologue_language_tree/ethnologue_language_tree.json')
        self.languages_in_ethnologue = set(self._collect_codes(self.ethnologue_tree))
        self.codes_to_names = {}
        self._collect_codes_to_names(self.ethnologue_tree, self.codes_to_names)

    def load_ethnologue_tree(self, source):
        with open(source, 'r') as file:
            return json.load(file)

    def _collect_codes(self, node):
        if 'two_letter_code' in node:
            code = node['two_letter_code']
            return [code] if code != '--' else [node['chip']]
        else:
            return sum([self._collect_codes(child) for child in node['children']], start=[])

    def _get_newick_tree(self, node, languages_to_collect):
        if 'chip' in node:
            code = node['two_letter_code'] if node['two_letter_code'] != '--' else node['chip']
            if code in languages_to_collect:
                return code
            else:
                return None
        children_trees = [self._get_newick_tree(child, languages_to_collect) for child in node['children']]
        children_trees = [t for t in children_trees if t is not None]
        if len(children_trees) == 1:
            return children_trees[0]
        if len(children_trees) > 1:
            return '(' + ','.join(children_trees) + ')'
        return None

    def _collect_codes_to_names(self, node, dictionary):
        if 'chip' in node:
            dictionary[node['two_letter_code']] = node['name']
        else:
            for child in node['children']:
                self._collect_codes_to_names(child, dictionary)

    def get_newick_tree(self, languages_to_collect):
        return self._get_newick_tree(self.ethnologue_tree, languages_to_collect) + ';'


class ResourcesManager:
    def __init__(self, wiki_articles_source='20231108'):
        self.ethnologue_tree = EthnologueLanguageTree()
        self.indoeuropean_languages_with_data = (self.ethnologue_tree.languages_in_ethnologue
                                                 & load_languages_with_embedding())
        df = get_wiki_articles_table(source=wiki_articles_source)
        self.wiki_articles_df = df[df.Wiki.isin(self.indoeuropean_languages_with_data)]

    def get_languages_with_most_articles(self, number_of_languages):
        return list(self.wiki_articles_df.sort_values(
            by='Articles', key=lambda x: x.astype(int), ascending=False).Wiki.iloc[:number_of_languages])

    def get_ethnologue_with_most_articles(self, number_of_languages):
        return ete3.Tree(self.ethnologue_tree.get_newick_tree(
            self.get_languages_with_most_articles(number_of_languages)
        ))
