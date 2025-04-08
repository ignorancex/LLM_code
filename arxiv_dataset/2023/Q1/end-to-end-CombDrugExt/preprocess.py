# running this file to create different kinds of preprocessed dataset
import os
os.chdir('drug-combo-extraction')
import jsonlines
import sys
import argparse
from sklearn.model_selection import train_test_split

sys.path.extend(["..", "."])

parser = argparse.ArgumentParser()
parser.add_argument('--window_size', type=int, default=1,
                    help="choose how many sentences around the relation bearing sentence")
parser.add_argument('--combine_train_vali', type=bool, default=True,
                    help="whether combine train and vali")

args = parser.parse_args()

# loading raw data files
training_data_raw = list(jsonlines.open('data/final_train_set.jsonl'))
test_data_raw = list(jsonlines.open('data/final_test_set.jsonl'))

# prepare generation target to the format of [drug1 @DRUG@ drug2 @DRUG@ @REL@]
def preprocess(data_raw):
    sentences = []
    relations = []
    for example in data_raw:
        spans = []
        sentence = example['sentence']
        spans_dic = example['spans']
        for span_dic in spans_dic:
            spans.append(span_dic['text'])
        rels = example['rels']
        if not rels:
            relation = ' @DRUG@ '.join(spans)
            relation = relation + ' @DRUG@ @NOCOMB@ '
        else:
            relation = ''
            for rel in rels:
                cls = rel['class']
                if cls == 'NEG':
                    cls = 'COMB'           
                    
#                 if cls == 'POS':
#                     cls = 'COMB'
#                 if cls != 'POS':
#                     cls = 'OTHER'

                curr_spans = []
                for i in rel['spans']:
                    curr_spans.append(spans[i])
                relation += ' @DRUG@ '.join(curr_spans) + ' @DRUG@ ' + f'@{cls}@ '
        if relation != '':
            sentences.append(sentence)
            relations.append(relation[:-1])
    return sentences, relations
# make directory for n-ary dataset
if not os.path.exists('n-ary'):
    os.mkdir('n-ary')
    
sentences, relations = preprocess(training_data_raw)
# split train data file to training and validation
train_sentences, valid_sentences, train_relations, valid_relations = train_test_split(sentences, relations, test_size=0.1)
if args.combine_train_vali:
    train_sentences = sentences
    train_relations = relations
# dump train
lines = []
for i in range(len(train_sentences)):
    line = train_sentences[i] + '\t' + train_relations[i] + '\n'
    lines.append(line)
with open('n-ary/train.txt', 'w') as f:
    for line in lines:
        f.write(line)
# dump valid
lines = []
for i in range(len(valid_sentences)):
    line = valid_sentences[i] + '\t' + valid_relations[i] + '\n'
    lines.append(line)
with open('n-ary/valid.txt', 'w') as f:
    for line in lines:
        f.write(line)
# dump test
sentences, relations = preprocess(test_data_raw)
lines = []
for i in range(len(sentences)):
    line = sentences[i] + '\t' + relations[i] + '\n'
    lines.append(line)
with open('n-ary/test.txt', 'w') as f:
    for line in lines:
        f.write(line)

# for splitting paragraph to sentences
from nltk.tokenize import sent_tokenize

# prepare source text with longer context
def preprocess_longer_context(data_raw, window_size=0):
    sentences = []
    relations = []
    for example in data_raw:
        paragraph = example['paragraph']
        sentence = '[SEP] ' + example['sentence'] + ' [SEP]'
        splitted_sentences = sent_tokenize(paragraph)
        for i, text in enumerate(splitted_sentences):
            if text == example['sentence']:
                if i > window_size - 1:
                    presen = ' '.join(splitted_sentences[i - window_size:i])
                else:
                    presen = ' '.join(splitted_sentences[:i])
                if i < len(splitted_sentences) - window_size:
                    aftersen = ' '.join(splitted_sentences[i + 1:i + window_size + 1])
                else:
                    aftersen = ' '.join(splitted_sentences[i + 1:])

        sentences.append(presen + ' ' + sentence + ' ' + aftersen)
        spans_dic = example['spans']
        spans = []
        for span_dic in spans_dic:
            span = span_dic['text']
            spans.append(span)
        rels = example['rels']
        if rels == []:
            relation = ' @DRUG@ '.join(spans)
            relation = relation + ' @DRUG@ @NOCOMB@'
            relations.append(relation)
        else:
            relation = ''
            for rel in rels:
                cls = rel['class']
                if cls == 'NEG':
                    cls = 'COMB'
                
#                 if cls == 'POS':
#                     cls = 'COMB'
#                 if cls != 'POS':
#                     cls = 'OTHER'
                
                curr_spans = []
                for i in rel['spans']:
                    curr_spans.append(spans[i])
                relation = relation + ' @DRUG@ '.join(curr_spans) + ' @DRUG@ ' + f' @{cls}@ '
            relations.append(relation[:-1])
    return sentences, relations
# make directory for longer context dataset
if not os.path.exists('longer-context-n-ary'):
    os.mkdir('longer-context-n-ary')
    
sentences, relations = preprocess_longer_context(training_data_raw, window_size=args.window_size)
# split train data file to training and validation
train_sentences, valid_sentences, train_relations, valid_relations = train_test_split(sentences, relations, test_size=0.1)
if args.combine_train_vali:
    train_sentences = sentences
    train_relations = relations
# dump train
lines = []
for i in range(len(train_sentences)):
    line = train_sentences[i] + '\t' + train_relations[i] + '\n'
    lines.append(line)
with open('longer-context-n-ary/train.txt', 'w') as f:
    for line in lines:
        f.write(line)
# dump valid
lines = []
for i in range(len(valid_sentences)):
    line = valid_sentences[i] + '\t' + valid_relations[i] + '\n'
    lines.append(line)
with open('longer-context-n-ary/valid.txt', 'w') as f:
    for line in lines:
        f.write(line)
# dump test
sentences, relations = preprocess_longer_context(test_data_raw, window_size=args.window_size)
lines = []
for i in range(len(sentences)):
    line = sentences[i] + '\t' + relations[i] + '\n'
    lines.append(line)
with open('longer-context-n-ary/test.txt', 'w') as f:
    for line in lines:
        f.write(line)

# prepare the target sequence with NER step
def preprocess_with_ner(data_raw):
    sentences = []
    relations = []
    for example in data_raw:
        sentence = example['sentence']
        sentences.append(sentence)
        spans_dic = example['spans']
        spans = []
        for span_dic in spans_dic:
            span = span_dic['text']
            spans.append(span)
        rels = example['rels']
        if rels == []:
            relation = ' ; '.join(spans)
            relation = relation + ' @NER@'
            relations.append(relation)
        else:
            relation = ' ; '.join(spans)
            relation = relation + ' @NER@ '
            for span_dic in spans_dic:
                span = span_dic['text']
                spans.append(span)
            for rel in rels:
                cls = rel['class']
                if cls == 'NEG':
                    cls = 'COMB'
                curr_spans = []
                for i in rel['spans']:
                    curr_spans.append(spans[i])
                relation = relation + ' ; '.join(curr_spans) + f' @{cls}@ '
            relations.append(relation[:-1])
    return (sentences, relations)
# make directory for NER dataset
if not os.path.exists('ner-n-ary'):
    os.mkdir('ner-n-ary')
sentences, relations = preprocess_with_ner(training_data_raw)
# split train data file to training and validation
train_sentences, valid_sentences, train_relations, valid_relations = train_test_split(sentences, relations, test_size=0.1)
if args.combine_train_vali:
    train_sentences = sentences
    train_relations = relations
# dump train
lines = []
for i in range(len(train_sentences)):
    line = train_sentences[i] + '\t' + train_relations[i] + '\n'
    lines.append(line)
with open('ner-n-ary/train.txt', 'w') as f:
    for line in lines:
        f.write(line)
# dump valid
lines = []
for i in range(len(valid_sentences)):
    line = valid_sentences[i] + '\t' + valid_relations[i] + '\n'
    lines.append(line)
with open('ner-n-ary/valid.txt', 'w') as f:
    for line in lines:
        f.write(line)
# dump test
sentences, relations = preprocess_with_ner(test_data_raw)
lines = []
for i in range(len(sentences)):
    line = sentences[i] + '\t' + relations[i] + '\n'
    lines.append(line)
with open('ner-n-ary/test.txt', 'w') as f:
    for line in lines:
        f.write(line)
