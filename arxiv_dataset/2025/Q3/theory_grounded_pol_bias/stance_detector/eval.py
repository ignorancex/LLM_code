
import pandas as pd
from transformers import pipeline, BartForSequenceClassification, AutoTokenizer
from datasets import load_from_disk
from sklearn.metrics import f1_score, classification_report
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec



def get_labels(example, labels, pipe): 
    r = pipe(example, labels, num_labels=4)
    return r['labels'][np.argmax(r['scores'])], np.max(r['scores'])


def compute_f1(conf_level, testset, confidence_var, pred_var):
    subset= testset[testset[confidence_var]>=conf_level]
    return f1_score(subset['label'], subset[pred_var], average='macro'), subset.shape[0]

def confidence_intervals(stats):
    lower, upper = np.percentile(stats, [2.5, 97.5])
    return lower, upper


testset = load_from_disk('../data/stance_detector_eval/test/test_set').to_pandas()

zero_shot_model =  BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
model = BartForSequenceClassification.from_pretrained('bart_main')
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')


labels = ['disagree', 'agree', 'neutral', 'unrelated']
pipe = pipeline('zero-shot-classification', model=model, tokenizer=tokenizer, template = "The stance of the statement is {}", device=0)
zeroshot_pipe = pipeline('zero-shot-classification', model=zero_shot_model, tokenizer=tokenizer, template = "The stance of the statement is {}", device=0)
testset[['pred', 'conf']] = testset['text'].apply(lambda x: pd.Series(get_labels(x, labels, pipe)))
testset[['zero_shot_pred', 'zero_shot_conf']] = testset['text'].apply(lambda x: pd.Series(get_labels(x, labels, zeroshot_pipe)))




conf_levels = np.round(np.linspace(0.0, 0.9, 10), 2)
f1_list, n_list, zero_f1_list, zero_n_list = [], [], [], []
stat_names = ['n', 'f1', 'zero_n', 'zero_f1']

stats = {i: {x: [] for x in stat_names} for i in conf_levels}
n_iterations = 100

for c in conf_levels:
    f1, n = compute_f1(c, testset, 'conf', 'pred')
    f1_list.append(f1)
    n_list.append(n)
    zero_f1, zero_n = compute_f1(c, testset, 'zero_shot_conf', 'zero_shot_pred')
    zero_f1_list.append(zero_f1)
    zero_n_list.append(zero_n)
    for i in range(n_iterations):
        sample = testset.sample(frac=1.0, replace=True)
        zero_f1, zero_n = compute_f1(c, sample, 'zero_shot_conf', 'zero_shot_pred')
        f1, n = compute_f1(c, sample, 'conf', 'pred')
        stats[c]['f1'].append(f1)
        stats[c]['n'].append(n)
        stats[c]['zero_f1'].append(zero_f1)
        stats[c]['zero_n'].append(zero_n)


with open('stats.json', 'w') as outfile: json.dump(stats, outfile)

eval_df = pd.DataFrame({'f1': f1_list, 'n': n_list, 'zeroshot_f1':zero_f1_list, 'zeroshot_n':zero_n_list, 'conf': conf_levels})

eval_df.to_csv('eval/eval_df.csv')


high_conf = testset[testset['conf']>=0.9]

report = pd.DataFrame(classification_report(high_conf['label'], high_conf['pred'], output_dict=True))

report.to_csv('eval/main_classifier_report.csv')
