import pandas as pd
import numpy as np
# from ..utils.ckbp_utils import ckbp2_relations
import random

from nltk.translate.bleu_score import sentence_bleu
from nltk import word_tokenize
from tqdm import tqdm


year = 2023
random.seed(year)
ckbp2_relations = 'xWant, oWant, xEffect, oEffect, xReact, oReact, '+ \
    'xAttr, xIntent, xNeed, Causes, xReason, isBefore, isAfter, HinderedBy, HasSubEvent'
ckbp2_relations = ckbp2_relations.split(', ')


def compute_self_bleu(serialize_mode='ht', parallel=False, num_proc=32):
  '''
  serialize_mode: 'hrt' = '{head} {relation} {tail}', 'ht' = '{head} {tail}'
  parallel: bool, indicates if we use multiprocessing to compute self bleu or not. Recommend to use it :)
  '''

  eval_data = pd.read_csv('annotation/ckbp2.0_raw_agg.csv')
  # eval_data = eval_data.sample(n=30, random_state=year)
  
  train_data = pd.read_csv('data/ckbp_csv/emnlp2021/train.csv')
  train_data.fillna('NaN')
  sample_per_rel = 10000
  train_subset = []
  for rel in ckbp2_relations:
    print(rel)
    temp = train_data[train_data['relation'] == rel]
    if sample_per_rel < len(temp):
      temp = temp.sample(n=sample_per_rel, random_state=year)
    if len(train_subset) == 0:
      train_subset = temp
    else:
      train_subset = train_subset.merge(temp, how='outer')

  if serialize_mode == 'hrt':
    train_sentences = list(train_subset['head'] + ' ' + train_subset['relation'] + ' ' + train_subset['tail'])
    eval_sentences = list(eval_data['head'] + ' ' + eval_data['relation'] + ' ' + eval_data['tail'])
  else:
    train_sentences = list(train_subset['head'] + ' ' + train_subset['tail'])
    eval_sentences = list(eval_data['head'] + ' ' + eval_data['tail'])

  num_eval_samples = len(eval_sentences)
  bleu_scores = []
  sentences_tokenized = {}
  for sent in set(eval_sentences + train_sentences):
    try:
      sentences_tokenized[sent] = word_tokenize(sent)
    except:
      print(sent)
      sentences_tokenized[sent] = ['NaN']
  print(len(eval_sentences + train_sentences), len(set(eval_sentences + train_sentences)), len(sentences_tokenized))

  if parallel:
    from multiprocessing import Process, Manager
    from math import floor

    def bleu_fn(idx, bleu_scores):
      for i in tqdm(idx):
        bleu = sentence_bleu(
          references=[sentences_tokenized[sent] for j, sent in enumerate(eval_sentences + train_sentences) if j != i],
          hypothesis=sentences_tokenized[eval_sentences[i]],
          weights=(0, 1))
        bleu_scores[i] = round(bleu, 4)

    bleu_scores = Manager().list([0]*num_eval_samples)
    cutoff = [floor(num_eval_samples*(i/num_proc)) for i in range(num_proc+1)]
    process = []
    for i in range(num_proc):
      p = Process(target=bleu_fn, args=(range(cutoff[i], cutoff[i+1]), bleu_scores,))
      p.start()
      process.append(p)
    for p in process:
      p.join()

    bleu_scores = list(bleu_scores)
    print(len(bleu_scores))

  else:
    for i in tqdm(range(num_eval_samples)):
      bleu = sentence_bleu(
        references=[sentences_tokenized[sent] for j, sent in enumerate(eval_sentences + train_sentences) if j != i],
        hypothesis=sentences_tokenized[eval_sentences[i]],
        weights=(0, 1))
      bleu_scores.append(round(bleu, 4))

  eval_data['self_bleu'] = bleu_scores
  eval_data.to_csv(f'annotation/ckbp2.0_raw_agg_with_self_bleu_{sample_per_rel}.csv', index=False)
  print(np.mean(bleu_scores))


# balancing the dev/test set
# for each relation
# sample dev/test with the sample 1/0 ratio, and but #dev = 1/5 #total
def split_eval_data(dev_total_ratio=0.2, *, 
  take_harder_set=False, clss_portion=[0.05, 0.05, 0.05, 0.85], harder_total_ratio=0.2, keep_rel_ratio=True):
  '''
  dev_total_ratio: as the named said, it's #dev_instances/#eval_instances ratio.
  take_harder_set: bool, indicate if we want to produce harder eval set
  harder_total_ratio: only effective when take_harder_set = True,
    it's top k% of the smallest self-bleu score within one strata (i.e when keep_rel_ratio set as True)
    or w.r.t the whole eval set (i.e i.e when keep_rel_ratio set as False)
  keep_rel_ratio: only effective when take_harder_set = True, indicate if we need to keep ratio of #instances between rel
  clss_portion: as the name said, the order is test_set, cs_head, all_head, adv. 
    If we don't set it, the hard version of ckbp2.0 seems .. not hard! Thus, need to increase the portion of adv samples
  '''
  if take_harder_set:
    df = pd.read_csv('annotation/ckbp2.0_raw_agg_with_self_bleu_10000.csv')
    df.sort_values(by=['self_bleu'], inplace=True)
    if not keep_rel_ratio:
      df = df.head(int(harder_total_ratio*len(df)))
  else:
    df = pd.read_csv('annotation/ckbp2.0_raw_agg.csv')
  df['split'] = 0 # 0 for dev, 1 for test
  del df['index']
  total_num_eval_samples = len(df)
  print(len(df))

  # prepare idx
  rel_portion, rel_idx, clss_idx, lb_idx = {}, {}, {}, {}
  for rel in ckbp2_relations:
    rel_idx[rel] = df['relation'] == rel
    rel_portion[rel] = sum(rel_idx[rel])/total_num_eval_samples
  for clss in ['test_set', 'cs_head', 'all_head', 'adv']:
    clss_idx[clss] = df['class'] == clss
  for l in [0,1]:
    lb_idx[l] = df['label'] == l

  # random sample
  for i, clss in enumerate(['test_set', 'cs_head', 'all_head', 'adv']):
    if keep_rel_ratio:
      for rel in ckbp2_relations:
        if rel == 'xReason':
          pos_neg_ratio = 1
        else:
          pos_neg_ratio = sum(rel_idx[rel]*clss_idx[clss]*lb_idx[1])/sum(rel_idx[rel]*clss_idx[clss]*lb_idx[0])
        pos_total_ratio = pos_neg_ratio/(1+pos_neg_ratio)
        for l in [0,1]:
          idx = rel_idx[rel]*clss_idx[clss]*lb_idx[l]
          if clss_portion:
            num = total_num_eval_samples*clss_portion[i]*rel_portion[rel]* \
              (pos_total_ratio if l == 1 else 1-pos_total_ratio)
            if num > sum(idx):
              num = sum(idx)
          else:
            num = sum(idx)
          if take_harder_set:
            num = int(num*harder_total_ratio)
          num_dev = int(num*dev_total_ratio)
          dev_test_assignment = [0]*num_dev + [1]*(num-num_dev)

          random.shuffle(dev_test_assignment)
          remove = [-1]*(sum(idx) - num)
          dev_test_assignment = dev_test_assignment + remove
          df['split'][idx] = dev_test_assignment
    else:
      raise NotImplementedError

  # finalize
  # sorting: df.sort_values
  if take_harder_set:
    df = df[df['split'] != -1]
    df['split'] = df['split'].apply(lambda x: 'dev' if x==0 else 'tst')
    print(len(df))
    df.to_csv(f'annotation/ckbp2.0_hard_{clss_portion}.csv', index=False)
  else:
    df['split'] = df['split'].apply(lambda x: 'dev' if x==0 else 'tst')
    df.to_csv('annotation/ckbp2.0.csv', index=False)
    print(len(df))


def remove_general_relation():
  train = pd.read_csv('data/ckbp_csv/emnlp2021/train.csv').fillna('')
  train = train[train['relation'].apply(lambda x: 'general' not in x)]
  train = train[train['tail'] != '']
  train.to_csv('data/ckbp_csv/emnlp2023/train.csv', index=False)


if __name__ == '__main__':
  # compute_self_bleu('ht', True)
  # split_eval_data(take_harder_set=False, clss_portion=None)
  remove_general_relation()
