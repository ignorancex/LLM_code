import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support


ckbp2_relations = 'xWant, oWant, xEffect, oEffect, xReact, oReact, '+ \
    'xAttr, xIntent, xNeed, Causes, xReason, isBefore, isAfter, HinderedBy, HasSubEvent'
ckbp2_relations = ckbp2_relations.split(', ')


def data_statistics():
  df = pd.read_csv('annotation/ckbp2.0.csv')
  df['class'][df['class'] == 'all_head'] = 'cs_head'
  idx_pos = df['label'] == 1

  # split
  for st, v in zip(['Dev', 'Test'], ['dev', 'tst']):
    idx = df['split'] == v
    num_triples = sum(idx)
    portion_pos = round(sum(idx*idx_pos)/num_triples, 4)*100
    print(st, '&', num_triples, '&', portion_pos, '\\\\')

  # instance type
  print()
  for st, v in zip(['ID', 'OOD', 'Adv.'], ['test_set', 'cs_head', 'adv']):
    idx = df['class'] == v
    num_triples = sum(idx)
    portion_pos = round(sum(idx*idx_pos)/num_triples, 4)*100
    print(st, '&', num_triples, '&', portion_pos, '\\\\')

  # relation
  print()
  for r in ckbp2_relations:
    idx = df['relation'] == r
    num_triples = sum(idx)
    portion_pos = round(sum(idx*idx_pos)/num_triples, 4)*100
    print(r, '&', num_triples, '&', portion_pos, '\\\\')


def portion_unseen_node():
  eval_data = pd.read_csv('annotation/ckbp2.0.csv')
  classes = eval_data['class'].copy()
  classes[classes == 'test_set'] = 'id'
  classes[classes == 'cs_head'] = 'ood'
  classes[classes == 'all_head'] = 'ood'
  eval_data['class'] = classes
  # eval_data = eval_data[eval_data['class'] != 'adv']
  # eval_data = pd.read_csv('data/ckbp_csv/emnlp2021/evaluation_set.csv')
  
  train_data = pd.read_csv('data/ckbp_csv/emnlp2021/train.csv').fillna('')
  list_train_nodes = train_data['head'].tolist() + train_data['tail'].tolist()
  set_train_nodes = set(list_train_nodes)

  for v in ['dev', 'tst']:
    df = eval_data[eval_data['split'] == v]
    list_eval_nodes = df['head'].tolist() + df['tail'].tolist()
    set_eval_nodes = set(list_eval_nodes)
    
    unseen_nodes = set_eval_nodes.difference(set_train_nodes)
    portion_unseen = len(unseen_nodes)/len(set_eval_nodes)
    print(portion_unseen)

  print()
  for v in ['id', 'ood', 'adv']:
    df = eval_data[eval_data['class'] == v]
    list_eval_nodes = df['head'].tolist() + df['tail'].tolist()
    set_eval_nodes = set(list_eval_nodes)
    
    unseen_nodes = set_eval_nodes.difference(set_train_nodes)
    portion_unseen = len(unseen_nodes)/len(set_eval_nodes)
    print(portion_unseen)

  print()
  for v in ckbp2_relations:
    df = eval_data[eval_data['relation'] == v]
    list_eval_nodes = df['head'].tolist() + df['tail'].tolist()
    set_eval_nodes = set(list_eval_nodes)
    
    unseen_nodes = set_eval_nodes.difference(set_train_nodes)
    portion_unseen = len(unseen_nodes)/len(set_eval_nodes)
    print(portion_unseen)


def fleiss_kappa(M): # not suitable in our case
  """
  source: https://gist.github.com/skylander86/65c442356377367e27e79ef1fed4adee
  See `Fleiss' Kappa <https://en.wikipedia.org/wiki/Fleiss%27_kappa>`_.
  :param M: a matrix of shape (:attr:`N`, :attr:`k`) where `N` is the number of subjects and `k` is the number of categories into which assignments are made. `M[i, j]` represent the number of raters who assigned the `i`th subject to the `j`th category.
  :type M: numpy matrix
  """
  N, k = M.shape  # N is no. of items, k is no. of categories
  n_annotators = float(np.sum(M[0, :]))  # no. of annotators
  p = np.sum(M, axis=0) / (N * n_annotators)
  P = (np.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
  Pbar = np.sum(P) / N
  PbarE = np.sum(p * p)
  kappa = (Pbar - PbarE) / (1 - PbarE)
  print('Fleiss =', kappa) 


def annotation_agreement():
  # IAA implementation based on the equation here
  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html
  # this version is adapted to the definition of disagreement in our case

  # TODO: not yet calculate pairwise agreement, seem to be more complicated, 
  # and also need to align with what describe in the report
  eval_data = pd.read_csv('annotation/ckbp2.0_raw_agg_w_expert_labels.csv')
  label1 = eval_data['label1'].to_numpy()
  label2 = eval_data['label2'].to_numpy()
  n = len(eval_data)
  p0 = 1- sum(abs(label1 - label2) >= 1)/n # only count label pairs (1, 0) and (0, 1) as disagreement
  pe = 1 - (sum(label1 == 1)*sum(label2 == 0) + sum(label1 == 0)*sum(label2 == 1))/(n**2)
  kappa = (p0 - pe)/(1 - pe)
  print('IAA =', kappa) # 0.9055


##########
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

CS_RELATIONS_2NL = {
    "AtLocation": "located or found at or in or on",
    "CapableOf": "is or are capable of",
    "Causes" : "causes",
    "CausesDesire": "makes someone want",
    "CreatedBy": " is created by",
    "Desires": "desires",
    "HasA": "has, possesses, or contains",
    "HasFirstSubevent": "begins with the event or action",
    "HasLastSubevent": "ends with the event or action",
    "HasPrerequisite": "to do this, one requires",
    "HasProperty": "can be characterized by being or having",
    "HasSubEvent" : "includes the event or action",
    "HinderedBy" : "can be hindered by",
    "InstanceOf" : " is an example or instance of",
    "isAfter" : "happens after",
    "isBefore" : "happens before",
    "isFilledBy" : "blank can be filled by",
    "MadeOf": "is made of",
    "MadeUpOf": "made up of",
    "MotivatedByGoal": "is a step towards accomplishing the goal",
    "NotDesires": "do not desire",
    "ObjectUse": "used for",
    "UsedFor": "used for",
    "oEffect" : "as a result, PersonY or others will",
    "oReact" : "as a result, PersonY or others feel",
    "oWant" : "as a result, PersonY or others want to",
    "PartOf" : "is a part of",
    "ReceivesAction" : "can receive or be affected by the action",
    "xAttr" : "PersonX is seen as",
    "xEffect" : "as a result, PersonX will",
    "xReact" : "as a result, PersonX feels",
    "xWant" : "as a result, PersonX wants to",
    "xNeed" : "but before, PersonX needed",
    "xIntent" : "because PersonX wanted",
    "xReason" : "because",
    "general Effect" : "as a result, other people or things will",
    "general Want" : "as a result, other people or things want to",
    "general React" : "as a result, other people or things feel",
    "gEffect" : "as a result, other people or things will",
    "gWant" : "as a result, other people or things want to",
    "gReact" : "as a result, other people or things feel",
}

def bow_lr_baseline(use_nl_rel=False):
  # load data
  train_data = pd.read_csv('data/ckbp_csv/emnlp2021/train.csv').fillna('')
  eval_data = pd.read_csv('annotation/ckbp2.0.csv').fillna('')
  if use_nl_rel:
    train_data['relation'] = train_data['relation'].apply(lambda r:CS_RELATIONS_2NL[r])
    eval_data['relation'] = eval_data['relation'].apply(lambda r:CS_RELATIONS_2NL[r])

  valid_data = eval_data[eval_data['split'] == 'dev']
  test_data = eval_data[eval_data['split'] == 'tst']

  x_train = train_data['head'] + ' ' + train_data['relation'] + ' ' + train_data['tail']
  y_train = train_data['label']
  x_valid = valid_data['head'] + ' ' + valid_data['relation'] + ' ' + valid_data['tail']
  y_valid = valid_data['label']
  x_test = test_data['head'] + ' ' + test_data['relation'] + ' ' + test_data['tail']
  y_test = test_data['label']

  # transform data
  tfidf = TfidfVectorizer(ngram_range=(1,1), max_df=0.9, min_df=5, max_features=30000)
  x_train_embeds = tfidf.fit_transform(x_train)
  x_valid_embeds = tfidf.transform(x_valid)
  x_test_embeds = tfidf.transform(x_test)

  # load model and train
  mlp = MLPClassifier(hidden_layer_sizes=(), activation='relu', random_state=2023)

  for i in range(2):
    mlp.partial_fit(x_train_embeds, y_train, [0,1])
    if i%1 == 0:
      # validate on the validation set
      print('After training {} iterations'.format(i))
      y_pred = mlp.predict(x_valid_embeds)
      print('valid:', 'auc', roc_auc_score(y_valid, y_pred), 'f1', f1_score(y_valid, y_pred))
      y_pred = mlp.predict(x_test_embeds)
      print('test:', 'auc', roc_auc_score(y_test, y_pred), 'f1', f1_score(y_test, y_pred))

# Best setting among (using/not using nl relation, max_features 10k 30k, using max_df + min_df or not)
# valid: auc 0.5275523595264877 f1 0.3161004431314623
# test: auc 0.5475082655554826 f1 0.35290148448043185 -> bad haha


def check_artifacts_like_CREAK():
  # formula is from https://arxiv.org/pdf/2104.08646.pdf, 3.1, y = 1, x = 1
  data = pd.read_csv('annotation/ckbp2.0.csv').fillna('') # Num of artifacts 83 over 3852 words
  # data = pd.read_csv('data/evaluation_set.csv').fillna('') # Num of artifacts 590 over 9660 words
  # data = pd.read_csv('data/ckbp_csv/emnlp2021/train.csv').fillna('')

  # we don't count the relation token, as it's type marker, and of different distribution
  # also remove PersonX, PersonY
  instance_list = ' ' + data['head'] + ' ' + data['tail'] + ' '
  idx_pos = data['label'] == 1
  vocab_list = sorted(list(set([x for node in instance_list for x in node.split(' ')])))
  for w in ['', 'PersonX', 'PersonY', 'PersonZ', 'PeopleX']:
    vocab_list.remove(w)
  vocab_size = len(vocab_list)

  z_ = 1 - 0.01/vocab_size
  p0_pos = sum(data['label'] == 1)/len(data)
  count_threshold = 20
  vocab_n = []
  vocab_prob_pos = []
  vocab_label_or_color = []
  vocab_annotation = []
  for v in vocab_list:
    idx_v = instance_list.apply(lambda x: f' {v} ' in x)
    n = sum(idx_v)
    vocab_n.append(n)
    prob_pos_hat_star = z_/np.sqrt(n/(p0_pos*(1-p0_pos))) + p0_pos
    prob_neg_hat_star = z_/np.sqrt(n/(p0_pos*(1-p0_pos))) + (1-p0_pos)
    pos = sum(idx_v*idx_pos)
    prob_pos = round(pos/n, 4)
    vocab_prob_pos.append(prob_pos)
    if n < count_threshold:
      vocab_label_or_color.append('NONE')
    elif prob_pos > prob_pos_hat_star:
      vocab_label_or_color.append('Plausible')
    elif 1-prob_pos > prob_neg_hat_star:
      vocab_label_or_color.append('Implausible')
    else:
      vocab_label_or_color.append('NONE')

  plot_data = pd.DataFrame({
    'n': vocab_n, 
    'prob_pos': vocab_prob_pos, 
    'group': vocab_label_or_color, 
    'word': vocab_list,
  })
  print('Num of artifacts', sum(plot_data['group'] != 'NONE'), 'over', len(vocab_list), 'words')

  # plot
  z_line_n = np.logspace(start=0, stop=4, base=10, num=41)
  z_line_prob_pos = np.array([z_/np.sqrt(n/(p0_pos*(1-p0_pos))) + p0_pos for n in z_line_n])
  z_line_prob_neg = np.array([z_/np.sqrt(n/(p0_pos*(1-p0_pos))) + (1-p0_pos) for n in z_line_n])

  fontsize = 16
  fig = plt.figure(figsize=(10,8))
  ax = fig.add_subplot(1,1,1) # (nrows, ncols, index)

  ax.set_title('Artifacts on CKBP v2', fontsize=fontsize+2)
  ax.set_xscale('log')
  ax.set_ybound(0, 1)
  ax.set_ylabel('$\\hat{p}(y|x)$', fontsize=fontsize)
  ax.set_xlabel('n', fontsize=fontsize)
  ax.plot([count_threshold, count_threshold], [-0.2, 1.2], label=f'n = {count_threshold}', alpha=0.2, color='grey')
  ax.plot(z_line_n, z_line_prob_pos, label=f'$\\alpha = 0.01/{vocab_size}$, Plausible', color='lime')
  ax.plot(z_line_n, z_line_prob_neg, label=f'$\\alpha = 0.01/{vocab_size}$, Implausible', color='tomato')

  for g in ['Plausible', 'Implausible', 'NONE']:
    temp = plot_data[plot_data['group']==g]
    if g == 'Plausible':
      ax.plot(temp['n'], temp['prob_pos'], 's', label='Plausible', alpha=0.8, color='green')
      ax.plot(temp['n'], 1-temp['prob_pos'], 'o', label=None, alpha=0.2, color='grey')
    elif g == 'Implausible':
      ax.plot(temp['n'], temp['prob_pos'], 's', label=None, alpha=0.2, color='grey')
      ax.plot(temp['n'], 1-temp['prob_pos'], 'o', label='Implausible', alpha=0.8, color='red')
    else:
      ax.plot(temp['n'], temp['prob_pos'], 's', label=None, alpha=0.2, color='grey')
      ax.plot(temp['n'], 1-temp['prob_pos'], 'o', label=None, alpha=0.2, color='grey')

  temp = plot_data[plot_data['group'] != 'NONE']
  annotation_data = temp[temp['n'] >= temp['n'].quantile(1-10/len(temp['n']))]
  for _, row in annotation_data.iterrows():
    plt.annotate(row.word, # this is the text
                 (row.n, row.prob_pos if row.group == 'Plausible' else 1-row.prob_pos), # coordinates
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center', # horizontal alignment can be left, right or center
                 fontsize=fontsize)

  plt.legend(fontsize=fontsize-2)
  plt.show()


def calculate_human_performance():
  df = pd.read_csv('annotation/ckbp2.0_raw_agg_w_expert_labels.csv')
  
  labels = df['label']
  classes = df['class'].copy()
  classes[classes == 'test_set'] = 'id'
  classes[classes == 'cs_head'] = 'ood'
  classes[classes == 'all_head'] = 'ood'

  scores = [[0]*8, [0]*8]
  for i in range(2):
    predicted_scores = df[f'label{i+1}']
    predicted_labels = predicted_scores = predicted_scores.apply(lambda x: 1 if x > 0.5 else 0)
    scores[i][0] = roc_auc_score(labels, predicted_scores)
    scores[i][4] = f1_score(labels, predicted_labels)

    for _, clss in enumerate(['id', 'ood', 'adv'], 1):
      idx = classes == clss
      scores[i][_] = roc_auc_score(labels[idx], predicted_scores[idx])
      scores[i][4+_] = f1_score(labels[idx], predicted_labels[idx])

    print(scores[i])

  final_score = [str(round((scores[0][i]+scores[1][i])*50, 1)) for i in range(8)]
  print('Human & ' + ' & '.join(final_score))


def error_analysis():
  df = pd.read_csv('annotation/ckbp2.0.csv')
  df['logit'] = pd.read_csv('annotation/ckbp2.0_kgbert_score.csv')['score']
  df['pred'] = df['logit'].apply(lambda x: 1 if x > 0.5 else 0)
  df[df['label'] != df['pred']].to_csv('annotation/ckbp2.0_error_analysis.csv')

  classes = df['class'].copy()
  classes[classes == 'test_set'] = 'id'
  classes[classes == 'cs_head'] = 'ood'
  classes[classes == 'all_head'] = 'ood'
  df['class'] = classes

  for v in ['id', 'ood', 'adv']:
    temp = df[df['class'] == v]
    print(v, precision_recall_fscore_support(temp['label'], temp['pred'], average=None))

''' sample
1515  PersonX flirt with PersonY  xIntent PersonX like PersonY  test_set  tst 0 0.990567863 1 -> inverse relation
2343  PersonX wake up at 7  Causes  it be dark  cs_head tst 0 0.516859591 1 -> related but wrong relation
2389  it be there Causes  PersonX find it test_set  tst 0 0.952884257 1 -> too confident while no context
2344  PersonY condition be go to be worse Causes  PersonY do not cooperate with PersonX all_head  tst 1 5.17E-05  0 -> misjudge the correct answer

4346  PersonX take PersonX dog to vet xNeed PersonX be a good owner adv tst 1 0.17104663  0 -> trace back to training data for similar assertion
PersonX take it to vet,PersonX be responsible,xAttr,atomic,1
PersonX take PersonX cat to vet,PersonX be responsible,xAttr,atomic,1
PersonX take the dog to vet,PersonX care for pet,xIntent,atomic,1
PersonX take the dog to vet,PersonX be responsible,xAttr,atomic,1
'''

if __name__ == '__main__':
  error_analysis()
