# import statements
import REUParsing as rp
import POSmethods as pm
import numpy as np
import pickle
import conllu
import networkx as nx
from tqdm import tqdm
from matplotlib import pyplot as plt

# for going through the files
import os
from tqdm import tqdm
import shutil

# root_addr = "D:\\REU Datasets\\Universal Dependencies 2.8.1\\ud-treebanks-v2.8\\" #put the address of Universal dependencies here
root_addr = "UD2/ud-treebanks-v2.8/" #put the address of Universal dependencies here
language_hashes = {}
for dirname in tqdm(os.listdir(root_addr)):
    lang = "_".join(dirname.split("-")[0].split("_")[1:])
    if lang not in language_hashes:
        language_hashes[lang] = {}
    directory = root_addr + dirname
    for file in os.listdir(directory):
        if file.endswith("-test.conllu"): #Change this to "-train.conllu" if you want
            file_addr = directory + "/" + file
            trees = rp.load_sentences(file_addr)
            # hashes = rp.generate_hashes(trees, pos=True)
            hashes = pm.generate_hashes_POS_2stepBFS(trees, quiet=False)
            for hash_ in hashes:
                if hash_ not in language_hashes[lang]:
                    language_hashes[lang][hash_] = 0
                # language_hashes[lang][hash_] += len(hashes[hash_])
                language_hashes[lang][hash_] += hashes[hash_]
with open("2stepBFS_hashes.dict", "wb") as f: #Change this filename to whatever you want
    pickle.dump(language_hashes, f)


with open("2stepBFS_hashes.dict", "rb") as lf:
    bfs_hashes = pickle.load(lf)

print(bfs_hashes)


# LANGUAGE_COUNTS
# creates a dictionary of the number of sentences in each language
language_counts = {}
for lang in bfs_hashes:
    language_counts[lang] = 0
    for hash_ in bfs_hashes[lang]:
        language_counts[lang] += bfs_hashes[lang][hash_]
# for lang in language_counts:             # this is if we don't want any languages with no sentences
#     if language_counts[lang] == 0:
#         del language_hashes[lang]

# prints out the language counts
print({lang:language_counts[lang] for lang in sorted(language_counts, key=lambda x: language_counts[x], reverse=True)})

# prints the number of unique hashes there are
print({lang:len(bfs_hashes[lang]) for lang in sorted(bfs_hashes, key=lambda x: len(bfs_hashes[x]), reverse=True)})


hash_vocab = set()
for lang in bfs_hashes:
    for hash_ in bfs_hashes[lang]:
        hash_vocab.add(hash_)
hash_vocab = list(hash_vocab)
print(len(hash_vocab))


# LANGUAGE_DIVERSITIES
# creates a dictionary {language: diversity score, ...}
# the diversity score is calculated by taking the total number of unique hashes (isomorphisms) and dividing it by the total number of sentences
language_diversities = {}
for lang in language_counts:
    if language_counts[lang] > 0:
        language_diversities[lang] = len(bfs_hashes[lang])/language_counts[lang]

# this prints out the language diversities in decreasing order (most diverse first)
print({lang:language_diversities[lang] for lang in sorted(language_diversities, key=lambda x: language_diversities[x], reverse=True)})


relative_hashes = {}
for language in bfs_hashes:
    relative_hashes[language] = {}
    for hash_ in bfs_hashes[language]:
        relative_hashes[language][hash_] = bfs_hashes[language][hash_] / language_counts[language]


language_vectors = {}
for lang in tqdm(relative_hashes):
    lang_vec = []
    for hash_ in hash_vocab:
        if hash_ in relative_hashes[lang]:
            lang_vec.append(relative_hashes[lang][hash_])
        else:
            lang_vec.append(0.0)
    language_vectors[lang] = lang_vec


def language_similarity(lang1, lang2):
    one_hot = np.array(list(map(lambda x: 1 if x != 0 else 0, language_vectors[lang2])))
    return np.dot(language_vectors[lang1], one_hot)


def leakage(lang, into): #into holds the set we are checking that if it leaks into
    vectors = [language_vectors[language] for language in into]
    combination = np.sum(vectors, axis=0)/len(vectors)
    one_hot = np.array(list(map(lambda x: 1 if x != 0 else 0, combination)))
    return np.dot(language_vectors[lang], one_hot)


language_leakage = {}
for language in tqdm(bfs_hashes):
    into = [lang for lang in bfs_hashes if lang != language]
    language_leakage[language] = leakage(language, into)


print({lang:language_leakage[lang] for lang in sorted(language_leakage, key=lambda x: language_leakage[x], reverse=True)})

# SIZE V LEAKAGE GRAPH
# this code creates a graph that shows the size of data vs the diversity score it has

# this sorts the language_diversities in reversing language count order to create the graph
# {lang: language_leakage[lang] for lang in sorted(language_leakage, key=lambda x: language_counts[x], reverse=True)}

xs = []
ys = []
labels = []
for lang in {lang: language_leakage[lang] for lang in sorted(language_leakage, key=lambda x: language_counts[x], reverse=True)}:
    xs.append(language_counts[lang])
    ys.append(language_leakage[lang])
    labels.append(lang)
# xs = np.log(xs)
fig, ax = plt.subplots(figsize=(50, 50))
ax.scatter(xs, ys)
for idx, label in enumerate(labels):
    ax.annotate(label, (xs[idx], ys[idx]))

# plt.savefig("Size vs Leakage", dpi=300)
plt.show()


def similar_languages(language):
    return {lang:language_similarity(language, lang) for lang in sorted([lang for lang in bfs_hashes], key=lambda x: language_similarity(language, x), reverse=True)}



