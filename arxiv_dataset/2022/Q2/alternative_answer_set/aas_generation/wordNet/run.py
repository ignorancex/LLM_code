from nltk.corpus import wordnet as wn
import argparse
import json
from tqdm.auto import tqdm
def get_hypernyms(key_word):
    synsets = wn.synsets(key_word)
    
    if not synsets:
        return []

    temp = [x.hypernyms() for x in synsets]
    lemmas = []
    for x in temp:
        for y in x:
            for z in y.lemma_names():
                if z not in lemmas:
                    lemmas.append(z)
    return lemmas

def get_synonyms(key_word):
    synsets = wn.synsets(key_word)
    
    if not synsets:
        return []
    
    lemmas = []
    new_lemmas = synsets[0].lemma_names()
    for lemma in new_lemmas:
        if lemma not in lemmas:
            lemmas.append(lemma)

    return lemmas

def prune_non_similar(key_word, word_list, thresh=0.6):
    return_list = []
    for word in word_list:
        word_synset = wn.synsets(word)[0]
        key_word_ = wn.synsets(key_word)[0]
        
        if word_synset is None:
            print('none word')
            
        if key_word_ is None:
            print('none key')
        
        similarity = key_word_.wup_similarity(word_synset)
        
        if similarity is not None and similarity > thresh:
            return_list.append(word)
    
    return return_list

def get_aas(key_word):
    synset = get_synonyms(key_word)
    hyperset = get_hypernyms(key_word)
    
    aas = prune_non_similar(key_word, list(set().union(synset, hyperset)), thresh=0.6)
    
    return aas

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get the wordNet synset of each label')
    parser.add_argument('--label_file',  type=str, default='data/gqa/trainval_label2ans.json',
                        help='the path of label file of dataset')
    parser.add_argument('--save_file',  type=str, default='wordNet/gqa/wordNet_aas.json',
                        help='the path to save the aas of each label')
    
    args = parser.parse_args()
    with open(args.label_file) as f:
        labels = json.load(f)
        f.close()

    ans2syn={} 
    for label in tqdm(labels, total=len(labels)):
        
        label_syn = get_aas(label)
        ans2syn[label] = label_syn
        
    json.dump(ans2syn, open(args.save_file, 'w'), indent=2)