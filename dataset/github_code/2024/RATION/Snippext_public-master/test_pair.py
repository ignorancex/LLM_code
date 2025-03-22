import os
import argparse
import json
import itertools
from torch.utils import data
from snippext.baseline import initialize_and_train
from snippext.dataset import SnippextDataset
from transformers import BertTokenizer
from snippext.model import MultiTaskNet
import torch
from torch.utils import data
import spacy
import json
import numpy as np
def eval_classifier(model, iterator, vocab):
    """Evaluate a classification model state on a dev/test set.

    Args:
        model (MultiTaskNet): the model state
        iterator (DataLoader): a batch iterator of the dev/test set
        threshold (float, optional): the cut-off threshold for binary cls
        get_threshold (boolean, optional): return the selected threshold if True

    Returns:
        float: Precision (or accuracy if more than 2 classes)
        float: Recall (or accuracy if more than 2 classes)
        float: F1 (or macro F1 if more than 2 classes)
        float: The Loss
        float: The cut-off threshold
    """
    model.eval()

    Y = []
    Y_hat = []
    Y_prob = []
    total_size = 0
    words=[]
    for i in range(len(vocab)):
        if vocab[i]=='PAIR':
            pair_idx=i
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            word, x, _, _, _, y, _, taskname = batch
            taskname = taskname[0]
            logits, y1, y_hat = model(x, y, task=taskname)
            logits = logits.view(-1, logits.shape[-1])
            y1 = y1.view(-1)
            
            total_size += y.shape[0]

            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())
            Y_prob.extend(logits.softmax(dim=-1)[:,pair_idx].cpu().numpy().tolist())
            words.extend(word)
    Y_hat=[vocab[i] for i in Y_hat]
    return Y_hat,Y_prob,words
    # for glue
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="hotel_tagging")
    parser.add_argument("--lm", type=str, default="bert")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--bert_path", type=str, default=None)
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--output_path",type=str, default=None)
    parser.add_argument('--pair_thres',type=float,default=0.9)
    hp = parser.parse_args()

    # only a single task for baseline
    task = hp.task

    # create the tag of the run
    run_tag = 'baseline_task_%s_lm_%s_batch_size_%d_run_id_%d' % (task,
            hp.lm,
            hp.batch_size,
            hp.run_id)

    # load task configuration
    configs = json.load(open('configs.json'))
    configs = {conf['name'] : conf for conf in configs}
    config = configs[task]

    trainset = config['trainset']
    validset = config['validset']
    testset = config['testset']
    task_type = config['task_type']
    vocab = config['vocab']
    tasknames = [task]
    result={}
    # load train/dev/test sets
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MultiTaskNet([config],
                     device,
                     False,
                     lm=hp.lm,
                     bert_path=hp.bert_path)
    model.load_state_dict(torch.load(hp.bert_path,map_location=torch.device(device)))
    model=model.to(torch.device(device))
    model.eval()
    result={}
    with open(hp.input_path,'r') as file:
        a=json.load(file)
    for i in a.keys():
        result[i]={}
        sent=a[i]['sent']
        word=a[i]['word']
        tag=a[i]['tag']
        example=[]
        ress=[]
        sents=[]
        for j in range(len(tag)):
            sent_j=sent[j]
            word_j=word[j]
            tag_j=tag[j]
            #id_j=id1[j]
            aspect=[]
            opinion=[]
            counter=0
            while counter<len(tag_j):
                if tag_j[counter]=='B-AS':
                    aspect.append([counter])
                    counter+=1
                    while counter<len(tag_j) and tag_j[counter]=='I-AS':
                        aspect[-1].append(counter)
                        counter+=1
                else:
                    counter+=1
            for k in range(len(aspect)):
                media=[word_j[w] for w in aspect[k]]
                aspect[k]=[' '.join(media),aspect[k][0]]
            counter=0
            while counter<len(tag_j):
                if tag_j[counter]=='B-OP':
                    opinion.append([counter])
                    counter+=1
                    while counter<len(tag_j) and tag_j[counter]=='I-OP':
                        opinion[-1].append(counter)
                        counter+=1
                else:
                    counter+=1
            for k in range(len(opinion)):
                media=[word_j[w] for w in opinion[k]]
                opinion[k]=[' '.join(media),opinion[k][0]]
            #print(aspect)
            #print(opinion)
            res = list(itertools.product(aspect,opinion))
            res1=[]
            for k in res:
                if k[0][1]<k[1][1]:
                    res1.append(' '.join([k[0][0],k[1][0]]))
                else:
                    res1.append(' '.join([k[1][0],k[0][0]]))
            ress.extend([[k[0][0],k[1][0]] for k in res])
            for k in res1:
                example.append(' [SEP] '.join([sent_j,k]))
                sents.append(sent_j)
        test_dataset = SnippextDataset(example, vocab, task, lm=hp.lm)
        iterator = data.DataLoader(dataset=test_dataset,
                                   batch_size=hp.batch_size,
                                   shuffle=False,
                                   num_workers=1,
                                   collate_fn=SnippextDataset.pad)
        Y_hat,Y_prob,word=eval_classifier(model,iterator,test_dataset.idx2tag)
        for j in range(len(Y_hat)):
            if sents[j] not in result[i].keys():
                result[i][sents[j]]=[]
            if Y_prob[j]>=hp.pair_thres:
                result[i][sents[j]].append(ress[j])
    with open(hp.output_path,'w') as file:
        json.dump(result,file)
        
