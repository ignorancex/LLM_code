import re
from fuzzywuzzy import fuzz
def match_sent(sent, txt):
    sent = re.sub(r'[^\w\s]', '', sent)
    sent =  ' '.join(sent.split()).strip() #removing extra spaces in between and leading and trailing spaces
    txt =  ' '.join(txt.split()).strip()
    
    txt = txt.replace(' [ARG]', '')
    if fuzz.partial_ratio(sent, txt) == 100: 
        return True
    else: return False


def check(doc, label):
    #check to see both args are documented in the train/test instance
    count = 0
    spl_args = ['[ARG]']
    for word in doc.split():
        if word in spl_args: count += 1
       
    if count != 4 and label == 0: 
        return False #keeping instances with one arg which will merge and are not duplicates.
    else: return True


def same_sent_diff_arg(txt1, txt2):
    
    #check if they are same sent but have diff args of same type -> then merge them
    spl_args = ['[ARG]']

    for arg in spl_args:
        if arg in txt1 :
            t1 = txt1.replace(arg, '')
            t2 = txt2.replace(arg, '')
    
    t1 =  ' '.join(t1.split()).strip()
    t2 =  ' '.join(t2.split()).strip()
    
    if t1 == t2:
        for arg in spl_args:
            if arg in txt1:
                idx1 = txt1.index(arg)
                idx2 = txt2.index(arg)
                len1 = len(txt1.split(arg)[1])
                len2 = len(txt2.split(arg)[1])
                if idx1 == idx2 or idx1 in range(idx2 - 2, idx2 + len2) or idx2 in range(idx1 - 2, idx1 + len1):
                    return txt1 #return only one text
                else:
                    if idx1 < idx2:
                        txt1 = ' '.join(txt1.split())
                        return txt1[:idx2 + 2*len(arg) + 1] + ' ' + txt2[idx2:] # -2 as txt1 has two spl args in earlier indices
                    else:
                        txt2 = ' '.join(txt2.split())
                        return txt2[:idx1 + 2*len(arg) + 1] + ' ' + txt1[idx1:]



import copy

def get_doc(doc, txt1, txt2, label=1):
       
    sents = nltk.sent_tokenize(doc)
    
    tmp = []
    
    #check for same arg, same sent
    
    if txt2 and same_sent_diff_arg(txt1, txt2):
        txt1 = same_sent_diff_arg(txt1, txt2)
        txt2 = txt1
        
    arg_list = ['[TIME-ARG]', '[PLACE-ARG]', '[CASUALTIES-ARG]', '[AFTER_EFFECTS-ARG]', '[PARTICIPANT-ARG]', '[REASON-ARG]']   
    txt1 = ' '.join(['[ARG]' if word in arg_list else word for word in txt1.split()]) 
    txt2 = ' '.join(['[ARG]' if word in arg_list else word for word in txt2.split()]) #these 2 lines, + change in check()

    for i in range(len(sents)):
        if txt1 and ' [SENT] ' + txt1 + ' [SEP].' not in tmp and match_sent(sents[i], txt1):
                tmp.append(' [SENT] ' + txt1 + ' [SEP].')
        elif txt2 and txt1 != txt2 and ' [SENT] ' + txt2 + ' [SEP].' not in tmp and match_sent(sents[i], txt2):
            tmp.append(' [SENT] ' + txt2 + ' [SEP].')
        elif sents[i] not in tmp:
            tmp.append(sents[i])
    
    doc_new = ' '.join(tmp)
    #print('........New Doc...',doc_new)    
    if len(doc_new.split()) > 200:
        sents = nltk.sent_tokenize(doc_new)
        
        #record the order in which indices can be deleted
        idx = []
        tmp_sents = copy.copy(sents) #shallow copy
        
        for i in range(len(sents)):
            if i != 0 and '[SENT]' not in sents[i]:
                idx.append(i)
                tmp_sents[i] = 'NULL' #keep only the essential 3 sents & rest get added in next steps keeping #tokens < 200
        
        if len(' '.join([s for s in tmp_sents if s != 'NULL']).split()) >= 200: #if the basic sents are v.long
            return ' '.join([s for s in tmp_sents if s != 'NULL'])
        
        else:
            for i in idx:
                tmp_sents[i] = sents[i]
                
                if len(' '.join([s for s in tmp_sents if s != 'NULL']).split()) >= 200: 
                    #else we will add one sent at a time unless it is more than 200
                    return ' '.join([s for s in tmp_sents if s != 'NULL'])
    
    return doc_new

import pickle as pkl
import pandas as pd
import nltk
from tqdm import tqdm

if __name__ == '__main__':
    
    #loading labelled samples (seed samples)
    df_rel = pd.read_csv('../data/relevance_check.csv', usecols = ['Text', 'Doc', 'Event', 'Label'])
    df_rel = df_rel.sample(frac=1, random_state=30).reset_index(drop=True)

    print('--------Formatting data for relevance-----------')
    print('Length of Training Data........', len(df_rel))
    
    train_data = []
    for i, (txt1, doc, event, label) in tqdm(enumerate(zip(df_rel['Text'].tolist(), df_rel['Doc'].tolist(), 
                                    df_rel['Event'].tolist(), df_rel['Label'].tolist()))):
        if get_doc(doc.strip(), txt1, '', label=label, mode=1):
            #print('Event....', event)
            train_data.append([event.strip() + ' [EVENT]' + get_doc(doc.strip(), txt1, '', label=label, mode=1) + ' [DOC]', label] )
    
    assert len(train_data) == len(df_rel)
    
    with open('../data/train_data_rel_orig.pkl', 'wb') as pkl_out:
         pkl.dump(train_data, pkl_out)

 
    df_test = pd.read_csv('../data/test_rel.csv', usecols = ['Text', 'Doc', 'Event', 'Label'])
    df_test.dropna(axis=0, inplace=True)
    print('Length of Testing Data........', len(df_test))

    test_data = []
    for i, (txt1, doc, event, label) in tqdm(enumerate(zip(df_test['Text'].tolist(), df_test['Doc'].tolist(),
                                    df_test['Event'].tolist(), df_test['Label'].tolist()))):
        if get_doc(doc.strip(), txt1, '', label=label, mode=1):
            test_data.append([event.strip() + ' [EVENT]' + get_doc(doc.strip(), txt1, '', label=label, mode=1) + ' [DOC]', label] )

    assert len(test_data) == len(df_test)

    with open('../data/test_data_rel.pkl', 'wb') as pkl_out:
         pkl.dump(test_data, pkl_out)
    
