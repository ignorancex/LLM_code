
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

    return ranks, scores, ids





if __name__=='__main__':
    
    # define phage directory
    prot_dir = r"./all_klebs_phages_faa_2May2022/"
    
    # initialise lists
    phage_prot_list = []
    
    pred_ranks = []
    pred_phage = []
    pred_score = []
    pred_ids = []
    
    # import all phage proteomes with known depo
    for file in os.listdir(prot_dir):
        # ignore whole sequence files
        if file[-5:] == "fasta":
            continue                  
        phage_prot_list.append(list(SeqIO.parse(prot_dir + file, "fasta")))
        
        ranks, scores, ids = analyzePhage(prot_dir + file, file)
        
        for i in range(len(scores)):
            if ranks[i] == 1:
                pred_score.append(scores[i])
                pred_phage.append(file)
                pred_ranks.append(ranks[i])
                pred_ids.append(ids[i])
            
# =============================================================================
#             if scores[i] > 0:
#                 pred_score.append(scores[i])
#                 pred_phage.append(file)
#                 pred_ranks.append(ranks[i])
#                 pred_ids.append(ids[i])
# =============================================================================
                

    with open("Positive Preds.csv",'w') as f:
        f.write('Phage ID, Protein ID,Rank,Score\n')

        for j in range(len(pred_ids)):

            f.write(str(pred_phage[j][:-4]) + "," + str(pred_ids[j])+','+str(pred_ranks[j])+','+str(pred_score[j])+'\n')
    f.close()

    records = []
    
    for k in range(len(pred_phage)):
        for rec in SeqIO.parse(r"./all_klebs_phages_faa_2May2022/" + pred_phage[k], "fasta"):
            if rec.id != pred_ids[k]:
                continue
            
            records.append(rec)


    SeqIO.write(records, "Positive Preds.faa", "fasta")
    




