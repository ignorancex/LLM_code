#load files
def load_file(path, filename):
    with open(path+filename, 'r') as f:
        lines = f.readlines()
    lines = [line.strip('\n') for line in lines]
    print(len(lines))
    return lines

import string
def remove_punct(s):
    return s.translate(str.maketrans('', '', string.punctuation + '£'))


#count tp, fp, fn
import re
from fuzzywuzzy import fuzz

def count(hyp_args, ref_args):
    
    tp = 0
    fp = 0
    fn = 0
    
    regex = '[a-zA-z0-9]$'
    
    for line1, line2 in zip(hyp_args, ref_args):
        hyp = [remove_punct(word) for word in line1.split(',')]
        ref = [remove_punct(word) for word in line2.split(',')]
        
        #print(hyp, ref)
        for word in hyp:
            if not re.search(regex, word): continue #if word is not alphanumeric
            if word in ref or word in line2: tp += 1
            else: fp += 1
                        
        for word in ref:
            if not re.search(regex, word): continue
            flag = 0
            for w in hyp: 
                if fuzz.ratio(word, w) >= 95: 
                    flag = 1
                    break
            if flag == 0: fn += 1
    print('tp, fp, fn = ', tp, fp, fn) 
    print('Total = ', tp + fp + fn)
    return tp, fp, fn

def calculate(tp, fp, fn):
    prec = tp/(tp + fp)
    rec = tp/ (tp + fn)
    f_score = (2*prec*rec)/(prec + rec)
    print(prec, rec, f_score)

if __name__ == '__main__':
    
    path = './results/'
    
    ref_args = load_file(path, 'ref.txt')
    hyp_args = load_file(path, 'hyp.txt')
    assert len(ref_args) == len(hyp_args)
    
    print('Results for ArgFuse')
    
    tp, fp, fn = count(hyp_args, ref_args)
    calculate(tp, fp, fn)
    
    

    