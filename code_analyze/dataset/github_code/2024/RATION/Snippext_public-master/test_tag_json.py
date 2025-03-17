import os
import argparse
import json

from torch.utils import data
from snippext.baseline import initialize_and_train
from snippext.dataset import SnippextDataset
from transformers import BertTokenizer
from snippext.model import MultiTaskNet
import torch
from torch.utils import data
import spacy
import json
def eval_tagging(model, iterator, idx2tag):
    """Evaluate a tagging model state on a dev/test set.

    Args:
        model (MultiTaskNet): the model state
        iterator (DataLoader): a batch iterator of the dev/test set
        idx2tag (dict): a mapping from tag indices to tag names

    Returns:
        float: precision
        float: recall
        float: f1
        float: loss
    """
    model.eval()

    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        total_size = 0
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, mask, y, seqlens, taskname = batch
            #print(words)
            taskname = taskname[0]
            batch_size = y.shape[0]

            logits, y, y_hat = model(x, y, task=taskname)  # y_hat: (N, T)

            logits = logits.view(-1, logits.shape[-1])
            y = y.view(-1)
            total_size += batch_size

            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.cpu().numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    # gets results and save
    word_results=[]
    tag_results=[]
    for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
        y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
        preds = [idx2tag[hat] for hat in y_hat]
        word_result=[]
        tag_result=[]
        if len(preds)==len(words.split())==len(tags.split()):
            for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
                if p == '<PAD>':
                    p = 'O'
                if t == '<PAD>':
                    p = t = 'O'
                word_result.append(w)
                tag_result.append(p)
        word_results.append(word_result)
        tag_results.append(tag_result)
                
    ## calc metric
    return word_results,tag_results
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="hotel_tagging")
    parser.add_argument("--lm", type=str, default="bert")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--bert_path", type=str, default=None)
    parser.add_argument("--summary_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
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
    nlp=spacy.load('en_core_web_sm')
    with open(hp.summary_path,'r') as file:
        data1=json.load(file)
    for i in data1:
        media_key=i['entity_id']
        
        result[media_key]={}
        sents1=[]
        for j in i['reviews']:
            sents1.extend(j['sentences'])
        sents1=[j for j in sents1 if len(j)>1]
        sents=list(nlp.pipe(sents1))
        sents=[[str(w) for w in s] for s in sents]
        test_dataset = SnippextDataset(sents, vocab, task,
                                       lm=hp.lm)
        iterator = data.DataLoader(dataset=test_dataset,
                                   batch_size=hp.batch_size,
                                   shuffle=False,
                                   num_workers=1,
                                   collate_fn=SnippextDataset.pad)
        word_result,tag_result=eval_tagging(model,iterator,test_dataset.idx2tag)
        result[media_key]['word']=word_result
        result[media_key]['sent']=sents1
        result[media_key]['tag']=tag_result
        assert len(sents)==len(word_result)
    with open(hp.output_path,'w') as file:
        json.dump(result,file)
