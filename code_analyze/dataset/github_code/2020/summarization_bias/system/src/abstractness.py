# Abstractness of the system

import os
import string
import nltk
from nltk import word_tokenize,WordNetLemmatizer,PorterStemmer,RegexpTokenizer
from collections import defaultdict
from nltk.util import ngrams

topic_path = 'source'
ref_path = 'references'

source_topic = 'sample_data/source.txt'
reference = 'sample_data/ref_summary.txt'

topic_path_text = open(source_topic,'r').read()
reference_text = open(reference,'r').read()
topic_path = []
reference = []

topic_path.append(topic_path_text)
reference.append(reference_text)


regex= RegexpTokenizer(r'\w+')

def abstract(topic,summary,gram_count):
    
    sentences = []
    for i in topic:
        doc_string = i
        doc_sentences = doc_string.split('.')
        for d in range(len(doc_sentences)):
            doc_sentences[d] = doc_sentences[d].lower()
        sentences.extend(doc_sentences)


    doc_ngrams = []
    
    for i in sentences:
            tokens = regex.tokenize(i)
            ngram = list(ngrams(tokens, 1))
            doc_ngrams.extend(ngram)

    summary = summary[0:len(summary)]
    reference_sentences = summary.split(".")
    reference_ngrams = []


    for i in reference_sentences:
        tokens = regex.tokenize(i)
        ngram = list(ngrams(tokens,1))
        reference_ngrams.extend(ngram)

    count = 0
    for i in reference_ngrams:
        if i in doc_ngrams:
            count += 1
    try:
        abstract_val.append(count/len(reference_ngrams))
    except Exception as e:
        pass

# Abstractness count
abstract_val = 0
counter_abs = 0
for f in range(len(source_topic)):
    doc_file = source_topic[f]
    ref_file = reference[f]
    abstract_val += len(list(set(reference[f].split(' ')) - set(source_topic[f].split(' '))))
    counter_abs += len(set(reference[f].split(' ')))

print('Abstractness of the system is ',abstract_val/counter_abs)