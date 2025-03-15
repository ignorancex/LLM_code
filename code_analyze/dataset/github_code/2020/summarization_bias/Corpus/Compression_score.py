
# coding: utf-8

# In[1]:


import os
import string
import nltk
from nltk import word_tokenize,WordNetLemmatizer,PorterStemmer,RegexpTokenizer
from collections import defaultdict
from nltk.util import ngrams


# In[2]:


topic_path = ''
ref_path = ''

regex= RegexpTokenizer(r'\w+')


# In[3]:


def abstract(topic,summary):
    
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
        ngram = list(ngrams(tokens, 1))
        reference_ngrams.extend(ngram)


    cs.append(len(doc_ngrams)/len(reference_ngrams))


# In[4]:


doc_files = os.listdir(topic_path)
cs = []
for f in doc_files:
    doc_file = open(topic_path + '/'+ f, 'r', encoding='utf-8')
    ref_file = open(ref_path + '/'+ f, 'r', encoding='utf-8')
    doc_string = doc_file.read()
    docs = doc_string.split("\n\n")
    ref_string = ref_file.read()
    abstract(docs,ref_string)


# In[5]:


print(sum(cs)/len(cs))

