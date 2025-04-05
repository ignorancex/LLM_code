import biased_textrank as bt
from tqdm import tqdm
import pandas as pd
import pickle as pkl
import format_data as fd

from simpletransformers.classification import ClassificationModel, ClassificationArgs
model_rel = ClassificationModel("roberta","outputs_rel/")
def check_relevance(arg, txt, doc, event): #no need of remove_irrelevant.py 
    idx = txt.index(arg)
    txt = txt[:idx] + '[ARG] ' + txt[idx: idx + len(arg)] + ' [ARG]' + txt[idx + len(arg):]
    txt = event.strip() + ' [EVENT]' + fd.get_doc(doc.strip(), txt, '') + ' [DOC]'
    x, _ = model_rel.predict([txt])
    if x == [1]: return False
    
    return True

def add_text(t1, t2, arg):
   alist = []
   if t1:
       alist.append(t1.split(arg)[1])
   if t2:
       alist.append(t2.split(arg)[1])
   return alist

def create_list(df_pred, doc_idx, rev_doc_idx):
    pred_rank = {}
    from simpletransformers.classification import ClassificationModel, ClassificationArgs
    model = ClassificationModel("roberta","outputs_rel/")
    
    for i, (txt1, txt2, doc, event) in tqdm(enumerate(zip(df_pred['Text1'].tolist(), df_pred['Text2'].tolist(),
                                              df_pred['Doc'].tolist(), df_pred['Event'].tolist()))):

        argtype = [arg for arg in list(atext.keys()) if arg in txt1][0]

        doc = doc.strip()
        for key in doc_idx[doc]:
            if key not in pred_rank:
                #print(argtype, txt1, txt2)
                pred_rank[key] = {'event': event, argtype.replace('[','').replace(']',''): add_text(txt1, txt2, argtype)}
            elif argtype.replace('[','').replace(']','') not in pred_rank[key]:
                 pred_rank[key][argtype.replace('[','').replace(']','')] = add_text(txt1, txt2, argtype)
            else:
                 pred_rank[key][argtype.replace('[','').replace(']','')].extend(add_text(txt1, txt2, argtype))

    return pred_rank

def rank_txt(pred_rank):
    from simpletransformers.language_representation import RepresentationModel

    model = RepresentationModel(
        model_type="bert",
        model_name="outputs_red_entropy_args/") #bert-base-cased
    
    atext = {'[TIME-ARG]': '[TIME-ARG]: the time when the EVENT took place.',
             '[PLACE-ARG]' : '[PLACE-ARG]: the location of occurrence of the EVENT or where the EVENT took place.',
             '[REASON-ARG]' : '[REASON-ARG]: the reason of occurrence of the EVENT, the why and how of the EVENT.',
             '[CASUALTIES-ARG]' : '[CASUALTIES-ARG]: the injury sustained, killings and loss of lives due to an EVENT.',
             '[PARTICIPANT-ARG]' : '[PARTICIPANT-ARG]: the people or objects involved in an EVENT',
             '[AFTER_EFFECTS-ARG]' : '[AFTER_EFFECTS-ARG]: the effects that an EVENT had afterwards like damage to property, affecting livelihoods.'}

    ranked = {}

    for k, v in tqdm(pred_rank.items()): #docid
        ranked[k] = {}
        for a_type, arguments in v.items(): #arg type
            #flag = 0 #flag for arg lists that are deemed not relevant -> we save one arg from that list at least
            if a_type != 'event' and a_type != 'doc': arguments = arguments[2]
            else: continue

            if not arguments: continue
            #print(len(arguments), arguments)
            if len(arguments) == 1:
                ranked[k][a_type] = arguments
                continue

            else:
                argmnts = [a for j, a in enumerate(arguments) if check_relevance(a, v[a_type][0][j], v['doc'], v['event'])] #txt, doc, event, argtype
                if not argmnts: #flag=1
                    ranked[k][a_type] = [arguments[0]] 
                    #if none of the args are relevant, choose the argument that appears first in the document 
                    continue                    
                else: arguments = argmnts 
            
            ranked[k][a_type] = []

            b_text = atext['['+a_type+']'].replace('EVENT', v['event']) + ' [EVENT] ' + v['doc']
            bias_vector = model.encode_sentences([b_text], combine_strategy="mean")
            arguments = list(set(arguments))
            text_vectors = model.encode_sentences(arguments, combine_strategy="mean")
            ranks = bt.biased_textrank(text_vectors, bias_vector)
            ranked[k][a_type].extend(bt.select_top_k_texts_preserving_order(arguments, ranks, len(arguments))) #replace len(..) with the number of args you want to keep.
            #if flag == 1: ranked[k][a_type] = ranked[k][a_type][-1] #keeping the most informative arg for all irrelevant arg list

    return ranked

if __name__ == '__main__':
    #biased text rank
    atext = {'[TIME-ARG]': '[TIME-ARG]: the time when the EVENT took place.',
             '[PLACE-ARG]' : '[PLACE-ARG]: the location of occurrence of the EVENT or where the EVENT took place.',
             '[REASON-ARG]' : '[REASON-ARG]: the reason of occurrence of the EVENT, the why and how of the EVENT.',
             '[CASUALTIES-ARG]' : '[CASUALTIES-ARG]: the injury sustained, killings and loss of lives due to an EVENT.',
             '[PARTICIPANT-ARG]' : '[PARTICIPANT-ARG]: the people or objects involved in an EVENT',
             '[AFTER_EFFECTS-ARG]' : '[AFTER_EFFECTS-ARG]: the effects that an EVENT had afterwards like damage to property, affecting livelihoods.'}

    with open('../resources/aggr_data-pred.pickle', 'rb') as pkl_in:  #-> on predicted text
         pred_data_orig = pkl.load(pkl_in)

    '''with open('../resources/aggr_data.pickle', 'rb') as pkl_in: #-> on test/gold data
        _ = pkl.load(pkl_in)
        _ = pkl.load(pkl_in)
        test = pkl.load(pkl_in)

    pred_data_orig = test'''

    #creating doc idx
    doc_idx = {}
    for k, v in pred_data_orig.items():
        doc = ' '.join(v['doc'].split()).strip()
        if doc not in doc_idx: doc_idx[doc] = [k]
        else: doc_idx[doc].append(k)
        
    #creating a reverse idx
    rev_doc_idx = {}
    for k, v in doc_idx.items():
        if len(v) == 1: rev_doc_idx[v[0]] = k
        else:
            for item in v:
                rev_doc_idx[item] = k

    #ranking the list
    ranked = rank_txt(pred_data_orig)

    print('#ranked docs: ', len(ranked))
    print('#original docs: ', len(doc_idx))
    print('Sample of ranked: .....................')
    count = 0
    for k, v in ranked.items():
        if count < 5:
           print(k, ': ', v)
           print('=====================================')
           count += 1
    
    with open('../resources/ranked_pred_args.pkl', 'wb') as pkl_out:
         pkl.dump(ranked, pkl_out)
         pkl.dump(rev_doc_idx, pkl_out)

    print('----------Completed Arg Ranking-------------')
