
import pickle
import numpy as np
from Bio import SeqIO
from scipy.stats import rankdata
import os
import sys


def features(sequence):

    base = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    feat = np.empty([1,0])
    for b in base:
        feat = np.append(feat, sequence.upper().count(b))

    return feat


def get_feats(filename):
    fs=[]
    ids=[]

    for rec in SeqIO.parse(filename, "fasta"):
        f=list(features(str(rec.seq)))
        fs.append(f)
        ids.append(rec.id)

    feats=np.array(fs)

    return feats, ids


def analyzePhage(filename, output):

    feats, ids = get_feats(filename)

    # get scores for all models
    pic_dir = r"./pickles/"
    pscores = []
    for pic in os.listdir(pic_dir):

        model = pickle.load(open(pic_dir + pic, 'rb'))
        pscores.append(model.predict(feats))


    # get average of the scores from different models
    scores = []
    for j in range(len(pscores[0])):

        a = [x[j] for x in pscores]

        scores.append( sum(a) / len(a) )


    ranks = len(scores) - rankdata(scores, method='ordinal') + 1

    ranks, scores, ids= zip(*sorted(zip(ranks, scores, ids)))

    output_file = output+'.csv'

    with open(output_file,'w') as f:
        f.write('protein ID,Rank,Score\n')

        for i in range(len(ids)):

            f.write(str(ids[i])+','+str(ranks[i])+','+str(scores[i])+'\n')
    f.close()





# =============================================================================
# test_file = r"./test.faa"
# output_file = r"./output_test"
# a, b = get_feats(test_file)
#
# a = analyzePhage(test_file, output_file)
# =============================================================================



