import argparse
import logging
import math
import os
import yaml
from PIL import Image
import numpy as np
import clip
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader 
from torchvision.datasets import ImageNet
from tqdm import tqdm, trange
from CLIP_linear_probe import LinearProbeClipModel, linear_probe_train
import pandas as pd
import re 
import string 
import nltk 
import spacy
from nltk.corpus import words
#from english_words import english_words_set
from nltk.corpus import wordnet as wn
import json


from spacy.matcher import Matcher 
from spacy.tokens import Span 
from spacy import displacy 

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description='Create a CSV of CLIP encodes images and traget values')

parser.add_argument('-b', '--batch_size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 32), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
 

def main():
    args = parser.parse_args()
    root_dir = os.path.dirname(os.path.realpath(__file__))    
    cur_dir = os.path.realpath(os.curdir)

    batch_size = args.batch_size
    """ num_epochs = args.epochs
    print(f'num_epochs: {args.epochs}') """
    print(f'args.batch_size: {args.batch_size}')
    
    # load spaCy model
    nlp = spacy.load("en_core_web_sm")
    #words_list = list(nlp.vocab.strings)
    #words=[]
    #for x in nlp.vocab.strings:
    #  words.append(x)
    #nltk.download()
    #word_list = words.words()
    # prints 236736
    
    
    #print(f'\nlen(words_list): {len(words_list)}, words_list =\n{words_list}')
    #print(f'\nlen(word_list): {len(word_list)}, \nword_list =\n{word_list}')
    
    #print(f'\nlen(english_words_set): {len(english_words_set)}, \english_words_set =\n{english_words_set}')

    # disable everything except the tagger
    #other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "tagger"]
    #nlp.disable_pipes(*other_pipes)

    #test = list(english_words_set)
    

    # use nlp.pipe() instead of nlp() to process multiple texts more efficiently
    #for doc in nlp.pipe(test):
    #  if len(doc) > 0:
    #    print(doc[0].text, doc[0].tag_)

    #adjectives = []
    #for synset in list(wn.all_synsets('a')):
    #  adjectives.extend([str(lemma.name()) for lemma in synset.lemmas()])
      
    #adjectives = list(set(adjectives))

    #adjectives = [word for word in adjectives if word.isalpha() ]

    adjectives = {x.name().split(".", 1)[0] for x in wn.all_synsets("a")}
    adjectives = list(set(adjectives)) 
    
    with open("full_adjectives2.json", "w") as fp:
        json.dump(adjectives, fp)
       
        
if __name__ == "__main__":
    main()


    


