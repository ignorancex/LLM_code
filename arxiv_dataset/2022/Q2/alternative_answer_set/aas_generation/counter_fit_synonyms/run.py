import numpy as np
import os
import json
from tqdm.auto import tqdm
import argparse

# this txt file is donwloaded from https://github.com/nmrksic/counter-fitting/tree/master/word_vectors 
def get_cfv_dic():
    lines = open('counter_fit_synonyms/counter-fitted-vectors.txt', 'r').readlines()

    cfv_dic = {}
    for line in tqdm(lines):
        temp = line.split(' ')
        temp[-1] = temp[-1][:-1]    # remove the \n

        key = temp.pop(0)

        cfv_dic[key] = np.array(temp).astype(np.float32)
    return cfv_dic

def create_W(labels, cfv_dic):
    word_list = []
    W = []
    for word in tqdm(labels):
        try:
            vector = cfv_dic[word]
            word_list.append(word)
            W.append(vector)            
        except:
            
            pass
    return np.array(W), word_list

def cos_similarity(W, x):
    cos = np.dot(W, x).reshape(-1,1) / (
        np.sqrt((W*W).sum(1) + 1e-9).reshape(-1,1) * np.sqrt((x * x).sum(0)).reshape(-1,1))
    return cos

def get_synset(word_vec, W, W_word_list, thresh):
    similarities = cos_similarity(W, word_vec)
    new_synset = []
    for i in range(similarities.shape[0]):
        if similarities[i][0] > thresh:
            try:
                new_synset.append(W_word_list[i])
            except IndexError:
                print(i)

    return new_synset
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get the cfv embeddings of each label')
    parser.add_argument('--label_file',  type=str, default='data/gqa/trainval_label2ans.json',
                        help='the path of label file of dataset')
    parser.add_argument('--save_file',  type=str, default='counter_fit_synonyms/gqa/cfv_aas.json',
                        help='the path to save the aas of each label')
    
    args = parser.parse_args()
    with open(args.label_file) as f:
        labels = json.load(f)
        f.close()
    ans2syn = {}
    cfv_dic = get_cfv_dic()
    thresh = 0.6
    W, W_word_list = create_W(labels, cfv_dic)
    for word in tqdm(labels):
        try:
            word_vec = np.array(cfv_dic[word])
            new_synset = get_synset(word_vec, W, W_word_list, thresh=thresh)
        except KeyError:
            new_synset = [word]
        
        ans2syn[word] = new_synset
    json.dump(ans2syn, open(args.save_file, 'w'), indent=2)
        
    
    