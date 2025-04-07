import REUParsing as rp
import numpy as np
import pickle
import conllu
import networkx as nx
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.spatial import distance
import os

def language_counts(hashes):
    counts = {}
    for lang in hashes:
        counts[lang] = 0
        for hash_ in hashes[lang]:
            counts[lang] += hashes[lang][hash_]
    return counts

def relative_hashes(hashes, counts):
    relative_hashes = {}
    for language in hashes:
        relative_hashes[language] = {}
        for hash_ in hashes[language]:
            relative_hashes[language][hash_] = hashes[language][hash_] / counts[language]
    return relative_hashes

def generate_hash_vocab(relative):
    hash_vocab = set()
    for lang in relative:
        for hash_ in relative[lang]:
            hash_vocab.add(hash_)
    hash_vocab = list(hash_vocab)
    return hash_vocab

def language_vectors(relative, hash_vocab):
    vectors = {}
    for lang in tqdm(relative):
        lang_vec = []
        for hash_ in hash_vocab:
            if hash_ in relative[lang]:
                lang_vec.append(relative[lang][hash_])
            else:
                lang_vec.append(0.0)
        vectors[lang] = lang_vec
    return vectors

def language_diversities(hashes, counts):
    diversities = {}
    for lang in language_counts:
        if language_counts[lang] > 0:
            diversities[lang] = len(hashes[lang])/language_counts[lang]
    return diversities

def language_similarity(lang1vec, lang2vec):
    one_hot = np.array(list(map(lambda x: 1 if x != 0 else 0, lang2vec)))
    return np.dot(lang1vec, one_hot)

def similar_languages(langvec, group): #group is a dictionary mapping languages to vectors
    similarities = {}
    for language in tqdm(group):
        similarities[language] = language_similarity(langvec, group[language]) 
    return {lang:similarities[lang] for lang in sorted([lang for lang in group], key=lambda x: similarities[x], reverse=True)}

def leakage(langvec, group):
    vectors = [group[language] for language in group]
    combination = np.sum(vectors, axis=0)/len(vectors)
    one_hot = np.array(list(map(lambda x: 1 if x != 0 else 0, combination)))
    return np.dot(langvec, one_hot)