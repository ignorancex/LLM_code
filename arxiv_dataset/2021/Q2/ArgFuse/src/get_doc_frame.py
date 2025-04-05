import pickle as pkl
import format_data_rel as fd
import format_data_red as fd1
from data_create import attach_special_tokens as dcrt

from simpletransformers.classification import ClassificationModel, ClassificationArgs

def singles(pred_data):
    data = {}
    print('No. of single docs = ', len(pred_data))
    model = ClassificationModel("roberta","outputs_rel/") 
    for k, v in pred_data.items():
        if k not in data: data[k] = {}
        for atype, val in v.items():
            if '-ARG' in atype and val[0]:
                for i in range(len(val[0])):
                    txt1 = dcrt(val[0][i], val[2][i], atype)
                    txt1 = ' '.join(['[ARG]' if word == atype else word for word in txt1.split()])
                    txt = v['event'].strip() + ' [EVENT]' + fd.get_doc(v['doc'].strip(), txt1, '') + ' [DOC]' 
                    x, _ = model.predict([txt])
                    
                    if atype not in data[k]: 
                        if x == [0]: data[k][atype] = [val[2][i]]
                        else: data[k][atype] = ['N.A.']
                    elif val[2][i] not in data[k][atype]:
                        if x == [0]: data[k][atype].append(val[2][i])
                        else: data[k][atype].append('N.A.')

    return data

import re
import explicit_rel as er

def select_from_redundant(ranked, doc_idx, pred_data):
    data = {}
    model = ClassificationModel("bert","outputs_red_entropy_args/")
    for doc_id , v in ranked.items():
        print('-----------', doc_id, '--------------')
        if doc_id not in data:
            data[doc_id] = {}
        for atype, val in v.items():
            reject = []
            if '-ARG' in atype:
                if atype not in data[doc_id]: data[doc_id][atype] = []
                if len(val) == 1:
                    data[doc_id][atype].append(val[0].strip())
                    continue
                else:
                    sidx = {val.index(arg):i for i, arg in enumerate(pred_data[doc_id][atype][2]) if arg in val}
                    sents = [pred_data[doc_id][atype][0][sidx[i]] for i, arg in enumerate(val)]
                    for i, arg1 in enumerate(val):
                        for j, arg2 in enumerate(val):
                            arg1 = arg1.strip()
                            arg2 = arg2.strip()
                            if arg1 != arg2 and arg1 not in reject and arg2 not in reject:

                                #explicit-redundancy
                                if er.explicit(arg1, arg2):
                                    reject.append(arg1)
                                    continue

                                #implicit-redundancy
                                txt1 = dcrt(sents[i], arg1, atype)
                                txt2 = dcrt(sents[j], arg2, atype)
                                
                                
                                txt = pred_data[doc_id]['event'].strip() + ' [EVENT] ' + fd1.get_doc(doc_idx[doc_id].strip(), txt1, txt2) + ' [DOC]'
                                x, _ = model.predict([txt])
                                if x == [1]: #1 -> redundant, 0 -> not redundant
                                    reject.append(arg1)
                
                #print('Rejected:..', reject)

                if len(reject) == len(val): #if all the arguments are deemed redundant, keep the most informative one.
                    data[doc_id][atype].append(val[-1]).strip()
                    continue
                for arg in val:
                    if arg.strip() not in reject: data[doc_id][atype].append(arg.strip())
                    data[doc_id][atype] = list(set(data[doc_id][atype]))

    #print(data)
    return data

if __name__ == '__main__':

    with open('../resources/aggr_data-pred.pickle', 'rb') as pkl_in:
        pred_data = pkl.load(pkl_in)

    '''with open('../resources/aggr_data.pickle', 'rb') as pkl_in:
        _ = pkl.load(pkl_in)
        _ = pkl.load(pkl_in)
        test = pkl.load(pkl_in)

    pred_data = test''' #if you want to test with gold argument mentions

    with open('../resources/ranked_pred_args.pkl', 'rb') as pkl_in:
        ranked_data = pkl.load(pkl_in)
        doc_idx = pkl.load(pkl_in)

    #separate docs with only single args per type and check their relevance
    print('--------Starting processing singles----------')
    singles_idx = list(doc_idx.keys() - ranked_data.keys())
    #single_docs = singles(pred_data)  #use if there are argument types with one arg in the doc
    print(singles_idx)
    
    #select non-redundant args for the non-singles
    print('--------Starting processing multiples----------')
    multiple_docs = select_from_redundant(ranked_data, doc_idx, pred_data)
    print('Total Length = ', len(multiple_docs))
    
    #merged_docs = {**single_docs, **multiple_docs}
    #assert len(merged_docs) == len(single_docs) + len(multiple_docs)
    #print('Total length, Length of single docs, multiple docs: ',  len(merged_docs), len(single_docs), len(multiple_docs))
    
    with open('../resources/merged_new_args.pkl', 'wb') as pkl_out:
         #pkl.dump(merged_docs, pkl_out)   
         #pkl.dump(single_docs, pkl_out)
         pkl.dump(multiple_docs, pkl_out)
    
    print('---------------Completed generating frames-------------')
