import logging
import torch
from scipy.stats import spearmanr
import pandas as pd

def get_logger(log_file=None, log_level=logging.INFO, stream=True):

    logger = logging.getLogger(__name__)
    handlers = []
    if stream:
        stream_handler = logging.StreamHandler()
        handlers.append(stream_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(str(log_file), 'w')
        handlers.append(file_handler)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    logger.setLevel(log_level)

    return logger

def SentInx2TokInx(tokenizer, sent, sentInx):
    blocks = [tokenizer.encode(i, add_special_tokens=False) for i in sent.split()]
    start_inx = len([j for ix,i in enumerate(blocks) for j in i if ix < sentInx])
    tok_inx = list(range(start_inx + 1, start_inx+len(blocks[sentInx])+1 )) # +1 is because bos token
    assert [tokenizer.encode(sent)[i] for i in tok_inx] == tokenizer.encode(sent.split()[sentInx], add_special_tokens=False), (sent, sentInx)
    return tok_inx

def extract_sentence(txt_path, gold_path, mark, prev=0,pos=None):
    df = pd.read_csv(txt_path, header=None, delimiter="\t")
    df.columns = ["target", "pos", "index", "sent1", "sent2"]
    gt = [int(i.strip()=="T") for i in open(gold_path, "r").readlines()]
    df['GT'] = gt
    if not pos is None:
        df = df[df["pos"] == pos]
    GT_list = df['GT'].tolist()
    if mark == "base":
        sents_list_1 = df['sent1'].tolist()
        sents_list_2 = df['sent2'].tolist()
        inx_list = df['index'].tolist()
    elif mark == "prompt1":
        target_list = df['target']
        sents_list_1 = ["The \"%s\" in this sentence \"%s\" means in one word : \"" % (i,j) for (i,j) in zip(target_list, df['sent1'].tolist())]
        sents_list_2 = ["The \"%s\" in this sentence:\"%s\" means in one word : \"" % (i,j) for (i,j) in zip(target_list, df['sent2'].tolist())]
        inx_list = ["%d-%d"%(len(i.split())-1, len(j.split())-1) for i,j in zip(sents_list_1, sents_list_2) ]
    elif mark == "prompt2":
        target_list = df['target']
        sents_list_1 = ["In this sentence \"%s\", \"%s\" means in one word : \"" % (j,i) for (i,j) in zip(target_list, df['sent1'].tolist())]
        sents_list_2 = ["In this sentence:\"%s\", \"%s\" means in one word : \"" % (j,i) for (i,j) in zip(target_list, df['sent2'].tolist())]
        inx_list = ["%d-%d"%(len(i.split())-1, len(j.split())-1) for i,j in zip(sents_list_1, sents_list_2) ]
    elif mark == "repeat": # TEST later
        sents_list_1 = ["%s %s" % (i,i) for i in df['sent1'].tolist()]
        sents_list_2 = ["%s %s" % (i,i) for i in df['sent2'].tolist()]
        inx_list = ["%d-%d"%(len(i.split())+int(inx.split("-")[0])-prev, len(j.split())+int(inx.split("-")[1])-prev) for i,j,inx in zip(df['sent1'], df['sent2'], df['index']) ]
    elif mark == "repeat_hint":
        sents_list_1 = ["Rewrite the sentence: %s , rewritten sentence: %s" % (i,i) for i in df['sent1'].tolist()]
        sents_list_2 = ["Rewrite the sentence: %s , rewritten sentence: %s" % (i,i) for i in df['sent2'].tolist()]
        inx_list = ["%d-%d"%(len(i.split())+int(inx.split("-")[0])-prev+6, len(j.split())+int(inx.split("-")[1])-prev+6) for i,j,inx in zip(df['sent1'], df['sent2'], df['index']) ]
    return sents_list_1, sents_list_2, GT_list, inx_list

def get_inx_raw(csv_path):
    '''
    From raw-c format into WiC format
    '''
    df = pd.read_csv(csv_path)
    sents_list_1 = [i[:-1]+" "+i[-1] for i in df['sentence1']]
    sents_list_2 = [i[:-1]+" "+i[-1] for i in df['sentence2']]
    inx_1 = [s.split().index(w) if w in s.split() else -1 for w,s in zip(df['word'], sents_list_1)]
    inx_2 = [s.split().index(w) if w in s.split() else -1 for w,s in zip(df['word'], sents_list_2)]
    inx = [str(i1)+"-"+str(i2) for i1,i2 in zip(inx_1,inx_2)]
    df.insert(0, "inx", inx)
    df.to_csv("/home/liuzhu/LLM_LexSem/data/RAW/raw-c_inx.csv", index=False)


def extract_sentence_raw(csv_path, mark):
    df = pd.read_csv(csv_path)
    sents_list_1 = [i[:-1]+" "+i[-1] for i in df['sentence1']]
    sents_list_2 = [i[:-1]+" "+i[-1] for i in df['sentence2']]
    GT_list = df['mean_relatedness'].tolist()
    inx_list = df['inx'].tolist()

    if mark == "prompt1":
        sents_list_1 = ["The \"%s\" in this sentence: \"%s\" means in one word : \"" % (i,j) for (i,j) in zip(df['word'], sents_list_1)]
        sents_list_2 = ["The \"%s\" in this sentence: \"%s\" means in one word : \"" % (i,j) for (i,j) in zip(df['word'], sents_list_2)]
        inx_list = ["%d-%d"%(len(i.split())-1, len(j.split())-1) for i,j in zip(sents_list_1, sents_list_2) ]

    elif mark == "prompt2":
        sents_list_1 = ["In this sentence \"%s\", \"%s\" means in one word : \"" % (j,i) for (i,j) in zip(df['word'], sents_list_1)]
        sents_list_2 = ["In this sentence:\"%s\", \"%s\" means in one word : \"" % (j,i) for (i,j) in zip(df['word'], sents_list_2)]
        inx_list = ["%d-%d"%(len(i.split())-1, len(j.split())-1) for i,j in zip(sents_list_1, sents_list_2) ]

    elif mark == "repeat":
        inx_list = [str(len(s1.split())+int(ix.split("-")[0]))+"-"+str(len(s2.split())+int(ix.split("-")[1])) for s1,s2,ix in zip(sents_list_1,sents_list_2,inx_list)]
        sents_list_1 = ["%s %s" % (i,i) for i in sents_list_1]
        sents_list_2 = ["%s %s" % (i,i) for i in sents_list_2]

    elif mark == "repeat_prev":
        inx_list = [str(len(s1.split())+int(ix.split("-")[0])-1)+"-"+str(len(s2.split())+int(ix.split("-")[1])-1) for s1,s2,ix in zip(sents_list_1,sents_list_2,inx_list)]
        sents_list_1 = ["%s %s" % (i,i) for i in sents_list_1]
        sents_list_2 = ["%s %s" % (i,i) for i in sents_list_2]

    return sents_list_1, sents_list_2, GT_list, inx_list

def get_batch_data(tokenizer, sents_list, inx_list, batch_size=50, prompt=None):
    batch_list = []
    tar_inx_list = []
    for i in range(len(sents_list)//batch_size+1):
        if i*batch_size == len(sents_list):
            break
        sent = sents_list[i*batch_size:min(len(sents_list),(i+1)*batch_size)]
        inx = inx_list[i*batch_size:min(len(sents_list),(i+1)*batch_size)]
        tar_inx = [SentInx2TokInx(tokenizer, s, ix) for s,ix in zip(sent, inx)]
        encoded_sentence = tokenizer(sent, return_tensors="pt", padding=True)
        batch_list.append(encoded_sentence)
        tar_inx_list.append(tar_inx)
    return batch_list, tar_inx_list

def get_layer_representations(model, encoded_sentences, tar_inx, device):
    encoded_sentences.to(device)
    model.eval()
    with torch.no_grad():
        output = model(**encoded_sentences, output_hidden_states=True)
    all_layer_representations = output.hidden_states # the first is embedding layer
    
    represents = torch.zeros(all_layer_representations[0].shape[0], len(all_layer_representations), all_layer_representations[0].shape[2])
    for ix,i in enumerate(all_layer_representations):
        for j in range(i.shape[0]):
            represents[j, ix] = i[j, tar_inx[j]].mean(dim=0)

    return represents

def evaluation(pred, GT, type="acc"):
    assert len(pred) == len(GT), print(len(pred), len(GT))
    if type == "acc":
        result = sum([i==j for i,j in zip(pred, GT)]) / len(pred)
    elif type == "corr":
        result, _ = spearmanr(pred, GT)

    return result

def anisotropy_removal(embedding):
    return (embedding - embedding.mean(dim=0)) / (embedding.std(dim=0) + 1e-6)