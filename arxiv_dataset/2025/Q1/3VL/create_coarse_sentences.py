import argparse
import logging
import math
import os
import yaml
from PIL import Image
import numpy as np
import string
#import clip
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader 
from torchvision.datasets import ImageNet
from tqdm import tqdm, trange
from CLIP_linear_probe import LinearProbeClipModel, linear_probe_train
import pandas as pd
import spacy
import json
#import _pickle as cPickle
import pickle
import random
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import igraph
from igraph import Graph, EdgeSeq, plot
import plotly.graph_objects as go
from spellchecker import SpellChecker
from VLI_SDRO.SISP_Transformations.gen_si import get_cfs_for_pos_statement


os.environ['TRANSFORMERS_CACHE'] = '/mnt5/nir/transformers/cache/'
#Disabling transformers parallelism to avoid deadlock with Dataloader parallelism
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration

nlp = spacy.load("en_core_web_sm")
spell = SpellChecker()
lemmatizer = WordNetLemmatizer()

#device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t5_word_dict = dict()

#model = None
#tokenizer = None

spatial_negate = {
    # Adjacency
    "adjacent to": "nonadjacent to", 
    "alongside": "away from", 
    "at the side of": "away from", 
    "at the right side of": "at the left side of", 
    "at the left side of": "at the right side of",
    "attached to": "disconnect from", 
    "at the back of": "at the front of", 
    "ahead of": "behind", 
    "against": "away from", 
    "at the edge of": "far from the edge of", 
    # Directional
    "off": "on", 
    "past": "before", 
    "toward": "away from", 
    "down": "up", 
    "away from": "not away from", 
    "along": "not along", 
    "around": "not around", 
    "into": "not into", 
    "across": "not accross",
    "across from": "not across from", 
    "down from": "up from", 
    # Orientation
    "facing": "facing away from", 
    "facing away from": "facing", 
    "parallel to": "perpendicular to", 
    "perpendicular to": "parallel to", 
    # Proximity
    "by": "far away from", 
    "close to": "far from", 
    "near": "far from", 
    "far from": "close to", 
    "far away from": "by", 
    # Topological
    "connected to": "detached from", 
    "detached from": "connected to", 
    "has as a part": "does not have a part", 
    "part of": "not part of", 
    "contains": "does not contain", 
    "within": "outside", 
    "at": "not at", 
    "on": "not on", 
    "in": "not in",
    "with": "not with", 
    "surrounding": "not surrounding", 
    "among": "not among", 
    "consists of": "does not consists of", 
    "out of": "not out of", 
    "between": "not between", 
    "inside": "outside", 
    "outside": "inside", 
    "touching": "not touching",
    # Unallocated
    "beyond": "inside",
    "next to": "far from", 
    "opposite to": "same as", 
    "enclosed by": "not enclosed by", 
    # missing
    "above": "below",
    "below": "above",
    "behind": "infront",
    "on top of": "not on top of",
    "under": "over",
    "over": "under",
    "left of": "right of",
    "right of": "left of",
    "in front of": "behind",
    "beneath": "not beneath",
    "beside": "not beside",
    "in the middle of": "not in the middle of",
    "congruent": "incongruent",
}

color_list=['teal','brown','green','black','silver','white','yellow','purple','gray','blue','orange','red','blond','concrete','cream','beige','tan','pink','maroon',
'olive','violet','charcoal','bronze','gold','navy','coral','burgundy','mauve','peach','rust','cyan','clay','ruby','amber']

material_list= ["rubber", "metal", "denim", "wooden", "cloth", "silk", "plastic", "bamboo ","stone" ,"wicker","brick","smooth","steel","iron","silver","wool","gold",
"glass", "helium", "hydrogen", "ice", "lace", "lead", "alloy", "aluminum", "asbestos", "ash", "brass", "bronze", "carbon dioxide", "cardboard", "cement",
"chalk", "charcoal", "clay", "coal", "copper", "cotton", "dust", "fiberglass", "gas", "leather", "linen", "magnesium",
"man-made fibers", "marble", "mercury", "mortar", "mud", "nickel", "nitrogen", "nylon", "oil", "oxygen", "paper", "paraffin",
"petrol", "plaster", "platinum", "polyester", "sand", "slate", "smoke", "soil", "steam", "straw", "tin", "uranium", "water",
"wood", "zinc",]

size_list= ["long", "old", "extended", "light", "hefty", "scraggy", "heavy", "scanty", "broad", "little", "stout", "curvy",
"miniature", "thickset", "emaciated", "minute", "tiny", "illimitable", "sizable", "bulky", "mammoth", "strapping", "enormous",
"obese", "towering", "fleshy", "petite", "underweight", "compact", "measly", "teensy", "grand", "puny", "expansive", "oversize",
"trim", "beefy", "lanky", "slender", "gaunt", "pocket-size", "wee", "cubby", "mini", "thick", "full-size", "pint-size",
"unlimited", "elfin", "minuscule", "thin", "epic", "outsized", "trifling", "huge", "scrawny", "giant", "portly", "wide",
"brawny", "limitless", "stocky", "big", "large", "slim", "immense", "skimpy", "immeasurable", "skeletal", "colossal", "meager",
"tall", "gigantic", "pudgy", "extensive", "overweight", "tubby", "cosmic", "microscopic", "teeny", "boundless", "life-size", 
"squat", "fat", "paltry", "undersized", "bony", "lean", "small", "chunky", "massive", "sturdy", "great", "rotund", "endless",
"narrow", "titanic", "hulking", "short", "infinitesimal", "skinny", "gargantuan", "plump", "vast"]

state_list= ["tinted", "dust", "dirty", "flip", "calm", "burnt", "sad", "dry", "sunny", "old", "young", "broken", "stormy",
"overcast", "unhappy", "cloudless", "adult", "wet", "foggy", "curly", "stained", "rough", "pale", "white", "fresh", "stained"
"good", "new", "first", "last", "long", "great", "little", "own", "other", "right", "big", "high", "different", "small", "large",
"next", "early", "important", "few", "public", "bad", "same", "able"]

action_list =["above","across","against","along","attached to","behind","belonging to","between","carrying","covered in",
"covering", "eating", "flying in", "for", "from", "growing on", "hanging from", "has", "holding", "in", "in front of",
"laying on", "looking at", "lying on", "made of", "mounted on", "near", "of", "on", "on back of", "over", "painted on",
"parked on", "part of", "playing", "riding", "says", "sitting on", "standing on", "under", "using", "walking in",
"walking on", "watching", "wearing", "wears", "with" ]

def is_valid_word(word):
    return (word == spell.correction(word))
    

def get_t5_model_and_tokenizer():
  #tokenizer = T5Tokenizer.from_pretrained("t5-3b", model_max_length=30)
  #tokenizer = T5Tokenizer.from_pretrained("t5-3b", model_max_length=5)
  #model = T5ForConditionalGeneration.from_pretrained("t5-3b", device_map="auto")

  tokenizer = T5Tokenizer.from_pretrained("t5-large", model_max_length=50)
  #model = T5ForConditionalGeneration.from_pretrained("t5-large", device_map="auto")
  model = T5ForConditionalGeneration.from_pretrained("t5-large", device_map="auto")


  #model = model.to(device)
  
  #model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
  #tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

  return model, tokenizer

def get_flan_t5_model_and_tokenizer():
  #tokenizer = T5Tokenizer.from_pretrained("t5-3b", model_max_length=30)
  #model = T5ForConditionalGeneration.from_pretrained("t5-3b", device_map="auto")
  
  # model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large", device_map="auto")
  model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to(device)
  # model = model.to(device)
  tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

  """
  return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)
  NotImplementedError: Cannot copy out of meta tensor; no data!
  """
  #model = model.to(device)

  return model, tokenizer


#model, tokenizer = get_t5_model_and_tokenizer()


parser = argparse.ArgumentParser(description='Create a CSV of CLIP encodes images and traget values')

parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 32), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--checkpoint_file', default='', type=str,
                    help='filename to save checkpoint to')


parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:2375', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for ei ther single node or '
                         'multi node data parallel training')
                                     


class Node():
    def __init__(self, node_num=0, prompt=None, parent=None, prompt_encoding=None):
        ''' constructor ''' 
        
        self.node_num = node_num
        self.prompt = prompt
        self.prompt_encoding = prompt_encoding
        self.parent = parent
        self.children = dict()


with open('full_adjectives2.json', 'rb') as fp:
  adjectives = json.load(fp)

with open('full_nouns.json', 'rb') as fp:
  nouns = json.load(fp)

with open('full_verbs.json', 'rb') as fp:
  verbs = json.load(fp)



prepositions_list = ['about', 'above', 'across', 'after', 'against', 'along', 'among', 'around',
                      'at', 'before',	'behind',	'below', 'between', 'beyond', 'but', 'by', 'concerning',
                      'despite', 'down', 'during', 'except', 'following', 'for', 'from', 'in',
                      'including', 'into', 'like', 'near', 'of', 'off', 'on', 'onto', 'out',
                      'over',	'past', 'plus', 'since', 'throughout', 'to', 'towards', 'under',
                      'until', 'up', 'upon', 'up to', 'with', 'within',	'without'
                    ]

prepositions_list_2 = ['about', 'above', 'across', 'after', 'against', 'along', 'among', 'around',
                      'at', 'before',	'behind',	'below', 'between', 'beyond', 'by', 
                      'down', 'following', 'right', 'left', 'in',
                      'next', 'into', 'near', 'off', 'on', 'onto', 'out',
                      'over',	'under', 'up'
                    ]

#spatial_list = ['above', 'after', 'before',	'behind',	'below', 'between', 'by', 
#                      'in', 'near', 'off', 'on', 'out', 'over',	'under',
#                      'with', 'without'
#                    ]

spatial_list = ['above', 'after', 'before',	'behind',	'below', 'between', 'by', 
                      'in', 'near', 'off', 'on', 'out', 'over',	'under'                      
                    ]


noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']
adjective_tags = ['JJ', 'JJR', 'JJS']


def get_rand_spatial_word(word):
  spatial_words = get_rand_spatial_word_list(word)
  
  #seed_everything()
  #random.sample(spatial_words, num_words)
  return random.choice(spatial_words)



def get_rand_spatial_word_list(word):
  spatial_words = spatial_list.copy()
  spatial_words = set(spatial_words)
  #spatial_words.difference_update(set(words))
  spatial_words.discard(word)
  spatial_words = list(spatial_words)
  
  if spatial_words:
    spatial_words.sort()
  
  return spatial_words



def get_rand_adposition_2(word):
  adpositions = get_rand_adposition_list_2(word)
  
  #seed_everything()
  return random.choice(adpositions)

def get_rand_adposition_list_2(word):
  adpositions = prepositions_list_2.copy()
  adpositions = set(adpositions)
  adpositions.discard(word)
  adpositions = list(adpositions)
  
  if adpositions:
    adpositions.sort()
  
  return adpositions

def get_rand_adposition(word):
  adpositions = get_rand_adposition_list(word)
  
  #seed_everything()
  return random.choice(adpositions)



def get_rand_adposition_list(word):
  adpositions = prepositions_list.copy()
  adpositions = set(adpositions)
  adpositions.discard(word)
  adpositions = list(adpositions)
  
  if adpositions:
    adpositions.sort()
  
  return adpositions

  
def get_antonym(word):
  #print(f'\nget_antonym word: {word}','\n')
  antonyms = []

  for syn in wn.synsets(word):
    #print(f'syn: {syn}','\n')
    for l in syn.lemmas():
      #print(f'lemma: {l}','\n')
      if l.antonyms():
        #print(f'l.antonyms(): {l.antonyms()}','\n')
        antonyms.append(l.antonyms()[0].name())
  
  
  antonyms = set(antonyms)
  antonyms.discard(word)
  antonyms = list(antonyms)
  antonyms_iter = filter(is_valid_word,antonyms)
  antonyms = list(antonyms_iter)
  #sort the list to make it deterministic (with random.seed())
  antonyms.sort()
  #print(f'\nantonyms: \n\n{antonyms}\n') 

  if not antonyms:
    #print(f'no antonyms found for word: {word}\n')
    return ''
  
  #seed_everything()
  return random.choice(antonyms)



def get_antonym_2(word):
  #print(f'\nget_antonym word: {word}','\n')
  antonyms = []

  words_to_discard = [word]

  for syn in wn.synsets(word):
    words_to_discard.extend([l.name() for l in syn.lemmas()])
    #print(f'syn: {syn}','\n')
    for l in syn.lemmas():
      #print(f'lemma: {l}','\n')
      if l.antonyms():
        #print(f'l.antonyms(): {l.antonyms()}','\n')
        antonyms.append(l.antonyms()[0].name())
  
  
  antonyms = set(antonyms)
  #antonyms.discard(word)
  antonyms.difference_update(words_to_discard)
  antonyms = list(antonyms)
  antonyms_iter = filter(is_valid_word,antonyms)
  antonyms = list(antonyms_iter)
  #sort the list to make it deterministic (with random.seed())
  antonyms.sort()
  #print(f'\nantonyms: \n\n{antonyms}\n') 

  if not antonyms:
    #print(f'no antonyms found for word: {word}\n')
    return ''
  
  #seed_everything()
  return random.choice(antonyms)


def get_co_hyponym(word):
  co_hyponyms = get_co_hyponym_list(word)
  #print(f'\nword: {word}, co_hyponyms: \n\n{co_hyponyms}\n')

  if not co_hyponyms:
    #print(f'no co_hyponyms found for word: {word}\n')
    return ''

  
  #seed_everything()
  return random.choice(co_hyponyms)

def get_co_hyponym_list(word):
  #print(f'get_co_hyponym word: {word}','\n')
  co_hyponyms = []

  for syn in wn.synsets(word):
    #print(f'syn: {syn}','\n')
    hypernym_lst = syn.hypernyms()
    #print(f'hypernym_lst: {hypernym_lst}','\n')

    if hypernym_lst:
      hypernym = hypernym_lst[0]
      hyponyms = hypernym.hyponyms()
      #print(f'hyponyms: {hyponyms}','\n')

      for hypo in hyponyms:
        hypo_names = [l.name() for l in hypo.lemmas()]
        #print(f'hypo_names: {hypo_names}','\n')
        co_hyponyms.extend(hypo_names)
  
  
  co_hyponyms = set(co_hyponyms)
  co_hyponyms.discard(word)
  co_hyponyms = list(co_hyponyms)
  co_hyponyms_iter = filter(is_valid_word,co_hyponyms)
  co_hyponyms = list(co_hyponyms_iter)
  #sort the list to make it deterministic (with random.seed())
  co_hyponyms.sort() 
  #print(f'\nco_hyponyms: {co_hyponyms}\n')

  return co_hyponyms

def get_co_hyponym_2(word):
  co_hyponyms = get_co_hyponym_list_2(word)
  #print(f'\nword: {word}, co_hyponyms: \n\n{co_hyponyms}\n')

  if not co_hyponyms:
    #print(f'no co_hyponyms found for word: {word}\n')
    return ''

  
  #seed_everything()
  return random.choice(co_hyponyms)

def get_co_hyponym_list_2(word):
  #print(f'get_co_hyponym word: {word}','\n')
  co_hyponyms = []

  words_to_discard = [word]

  for syn in wn.synsets(word):
    #print(f'syn: {syn}','\n')
    words_to_discard.extend([l.name() for l in syn.lemmas()])
    hypernym_lst = syn.hypernyms()
    #print(f'hypernym_lst: {hypernym_lst}','\n')

    if hypernym_lst:
      hypernym = hypernym_lst[0]
      hyponyms = hypernym.hyponyms()
      #print(f'hyponyms: {hyponyms}','\n')

      for hypo in hyponyms:
        hypo_names = [l.name() for l in hypo.lemmas()]
        #print(f'hypo_names: {hypo_names}','\n')
        co_hyponyms.extend(hypo_names)
  
  
  co_hyponyms = set(co_hyponyms)
  #co_hyponyms.discard(word)
  co_hyponyms.difference_update(words_to_discard)
  co_hyponyms = list(co_hyponyms)
  co_hyponyms_iter = filter(is_valid_word,co_hyponyms)
  co_hyponyms = list(co_hyponyms_iter)
  #sort the list to make it deterministic (with random.seed())
  co_hyponyms.sort() 
  #print(f'\nco_hyponyms: {co_hyponyms}\n')

  return co_hyponyms


def get_rand_word(token_text, token_pos):
  if not token_pos in ['NOUN', 'VERB', 'ADJ', 'ADP']:
    print(f'get_rand_word token_pos ({token_pos}) of token: {token_text} is not in [NOUN, VERB, ADJ, ADP]')
    return ''
  
  if token_pos in ['NOUN', 'VERB']:
    return get_co_hyponym(token_text)
  
  if token_pos == 'ADP':
    return get_rand_adposition(token_text)

  return get_antonym(token_text)

"""
def replace_root_verb(doc):
  print(f'\nreplace_root_verb orig_text: {doc}','\n')

  orig_text_lst = doc.text.split()

  for token in doc:
    print(f'token.text: {token.text}, index: {token.i}, dep_: {token.dep_}, pos_: {token.pos_}, head: {token.head.text}\n')
    #if token.pos_ == 'VERB' and token.dep_ == 'ROOT':
    if token.pos_ == 'VERB':
      rand_verb = get_co_hyponym(token.text)
      print(f'rand_verb: {rand_verb}')
"""

def get_lemma_names(word):
  lemma_names = [word]

  for syn in wn.synsets(word):
    lemma_names.extend([l.name() for l in syn.lemmas()])

  lemmatizations = []
  for name in lemma_names:
    lemmatizations.append(lemmatizer.lemmatize(name))

  
  lemma_names.extend(lemmatizations)
  
  return lemma_names


def shuffle_nouns(caption):
  shuff = shuffle_caption(caption, shuffle_nouns=True, shuffle_adjectives=False)
  if (shuff==caption):
    return "" 
  
  return shuff

def shuffle_adjectives(caption):

  shuff = shuffle_caption(caption, shuffle_nouns=False, shuffle_adjectives=True)
   
  if (shuff==caption):
    return "" 
  
  return shuff


def shuffle_nouns_adjectives(caption):
    
    shuff = shuffle_caption(caption, shuffle_nouns=True, shuffle_adjectives=True)
    
    if (shuff==caption):
      return "" 

    return shuff

def permute_arr(arr):
  
  if arr.shape[0] < 2:
    #nothing to shuffle
    return arr
 
  shuffled = np.random.permutation(arr)
  cnt = 0

  while (shuffled==arr).all() and cnt < 3:
    shuffled = np.random.permutation(arr)
    cnt += 1

  return shuffled

#shuffle functions taken from https://github.com/mertyg/vision-language-models-are-bows/blob/main/dataset_zoo/perturbations.py
def shuffle_caption(caption, shuffle_nouns=True, shuffle_adjectives=False):
  
  doc = nlp(caption)
  
  text_lst = [tok.text for tok in doc]
  text_arr = np.array(text_lst)

  if shuffle_nouns:
    noun_indices = [i for i, tok in enumerate(doc) if tok.tag_ in noun_tags]
    shuffled = permute_arr(text_arr[noun_indices])
    text_arr[noun_indices] = shuffled

  if shuffle_adjectives:
    adjective_indices = [i for i, tok in enumerate(doc) if tok.tag_ in adjective_tags]
    shuffled = permute_arr(text_arr[adjective_indices])
    text_arr[adjective_indices] = shuffled
  shuffled_text = " ".join(text_arr)     

  return shuffled_text

def shuffle_allbut_nouns_and_adj(caption):
  
  doc = nlp(caption)
  
  text_lst = [tok.text for tok in doc]
  text_arr = np.array(text_lst)
  noun_adj_tags = noun_tags + adjective_tags

  noun_adj_indices = [i for i, token in enumerate(doc) if token.tag_ in noun_adj_tags]

  else_idx = np.ones(text_arr.shape[0])
  else_idx[noun_adj_indices] = 0

  else_idx = else_idx.astype(bool)

  ## Shuffle everything that is not noun or adjective
  text_arr[else_idx] = np.random.permutation(text_arr[else_idx])
  return " ".join(text_arr)


def get_trigrams(caption):
  # Taken from https://github.com/lingo-mit/context-ablations/blob/478fb18a9f9680321f0d37dc999ea444e9287cc0/code/transformers/src/transformers/data/data_augmentation.py
  trigrams = []
  trigram = []
  for i in range(len(caption)):
      trigram.append(caption[i])
      if i % 3 == 2:
          trigrams.append(trigram[:])
          trigram = []
  if trigram:
      trigrams.append(trigram)
  return trigrams

def trigram_shuffle(caption):
  trigrams = get_trigrams(caption)
  for trigram in trigrams:
      random.shuffle(trigram)
  return " ".join([" ".join(trigram) for trigram in trigrams])


def shuffle_within_trigrams(caption):
  import nltk
  tokens = nltk.word_tokenize(caption)
  shuffled = trigram_shuffle(tokens)
  return shuffled


def shuffle_trigrams(caption):
  import nltk
  tokens = nltk.word_tokenize(caption)
  trigrams = get_trigrams(tokens)
  random.shuffle(trigrams)
  shuffled = " ".join([" ".join(trigram) for trigram in trigrams])
  return shuffled

def get_all_shuffles_sentences(caption):

  res = []
  # shuffle_funcs = [shuffle_nouns_adjectives, shuffle_allbut_nouns_and_adj, shuffle_trigrams, shuffle_within_trigrams]
  shuffle_funcs = [shuffle_nouns, shuffle_adjectives, shuffle_allbut_nouns_and_adj, shuffle_trigrams, shuffle_within_trigrams]

  for func in shuffle_funcs:
    shuffled = func(caption)
    if shuffled and shuffled != caption and shuffled not in res:
      res.append(shuffled)

  return res

def get_all_shuffles_random_sentence(caption):
  all_shuffles = get_all_shuffles_sentences(caption)
  
  if all_shuffles:
    shuff = random.choice(all_shuffles)
    return [shuff]

  return []

def get_shuffled_nouns_adjectives_sentences(caption):
  
  res = []
  #shuffle_funcs = [shuffle_nouns, shuffle_adjectives, shuffle_nouns_adjectives]
  #for func in shuffle_funcs:
  #  res.append(func(caption))
  
  #need to check we got a shuffled sentence (there might be zero or one nouns or adjectives)

  nouns_shuff = shuffle_nouns(caption)
  if nouns_shuff:
    res.append(nouns_shuff)
  
  adj_shuff = shuffle_adjectives(caption)
  if adj_shuff:
    res.append(adj_shuff)

  if nouns_shuff and adj_shuff:
    both_shuff = shuffle_nouns_adjectives(caption)
    if both_shuff and both_shuff not in [nouns_shuff, adj_shuff]:
      res.append(both_shuff)
    
  return res

def get_t5_rand_word2_lemmas(token_text, token_pos):
  if not token_pos in ['NOUN', 'VERB', 'ADJ', 'ADP']:
    print(f'get_t5_rand_word2 token_pos ({token_pos}) of token: {token_text} is not in [NOUN, VERB, ADJ, ADP]')
    return ''
  
  lower_txt = token_text.lower()

  token_lemma_names = get_lemma_names(lower_txt)

  if lower_txt in t5_word_dict:
    return t5_word_dict[lower_txt]

  opposites = get_t5_opposite(lower_txt)
  #print(f'get_t5_rand_word2 lower_txt: {lower_txt}, opposites: {opposites}')

  if not opposites:
    t5_word_dict[lower_txt] = ''
    return ''

  for opp in opposites:
    lower_opp = opp.lower()
    opp_lemma_names = get_lemma_names(lower_opp)
    if is_valid_word(lower_opp):
      
      #do not return the same word
      #check that lemmas intersection is empty
      if not list(set(token_lemma_names) & set(opp_lemma_names)):
        if lower_txt != lower_opp and (lower_txt+"\'s") != lower_opp and lower_opp != 'na':
          t5_word_dict[lower_txt] = opp
          return opp

        #allow for the word to appear if it has a negation prefix
        if 'un' in lower_opp or 'non' in lower_opp:
          t5_word_dict[lower_txt] = opp
          return opp
    
    if token_pos == 'ADP':
      t5_word_dict[lower_txt] = get_rand_adposition(lower_txt)
      return t5_word_dict[lower_txt]

  #TODO: return a similar word from T5 ? or fill masked word ?
  t5_word_dict[lower_txt] = get_co_hyponym_2(token_text)
  return t5_word_dict[lower_txt]


def get_rb_rand_word(token_text, token_pos):
  
  res = ''
  
  if not token_pos in ['NOUN', 'VERB', 'ADJ', 'ADP']:
    print(f'get_rb_rand_word token_pos ({token_pos}) of token: {token_text} is not in [NOUN, VERB, ADJ, ADP]')
    return res
  

  lower_txt = token_text.lower() 

  
  #TODO: add lower_txt lemmas to exclude from list
  token_lemma_names = get_lemma_names(lower_txt)

  # if token_pos == 'VERB':
  #   if lower_txt in action_list:
  #   negatives = list(set(action_list) - set(token_lemma_names))
  #   res = random.choice(negatives)
  #   print(f'get_rb_rand_word: action: {lower_txt} is replaced by action: {res}. token_lemma_names are: {token_lemma_names}')
  

  if lower_txt in color_list:
    # negatives = list(set(color_list) - set([lower_txt]))
    negatives = list(set(color_list) - set(token_lemma_names))
    res = random.choice(negatives)
  #   # print(f'get_rb_rand_word: color: {lower_txt} is replaced by color: {res}')

  if lower_txt in size_list:
    negatives = list(set(size_list) - set(token_lemma_names))
    res = random.choice(negatives)
    # print(f'get_rb_rand_word: size: {lower_txt} is replaced by size: {res}. token_lemma_names are: {token_lemma_names}')

  if lower_txt in material_list:
    negatives = list(set(material_list) - set(token_lemma_names))
    res = random.choice(negatives)
    # print(f'get_rb_rand_word: material: {lower_txt} is replaced by material: {res}. token_lemma_names are: {token_lemma_names}')
  
  
  return res



def get_t5_rand_word2(token_text, token_pos):
  if not token_pos in ['NOUN', 'VERB', 'ADJ', 'ADP']:
    print(f'get_t5_rand_word2 token_pos ({token_pos}) of token: {token_text} is not in [NOUN, VERB, ADJ, ADP]')
    return ''
  
  lower_txt = token_text.lower() 

  if lower_txt in t5_word_dict:
    return t5_word_dict[lower_txt]

  lower_txt = token_text.lower() 
  opposites = get_t5_opposite(lower_txt)
  #print(f'get_t5_rand_word2 lower_txt: {lower_txt}, opposites: {opposites}')

  if not opposites:
    t5_word_dict[lower_txt] = ''
    return ''

  for opp in opposites:
    lower_opp = opp.lower()
    if is_valid_word(lower_opp):
      #do not return the same word
      if lower_txt != lower_opp and (lower_txt+"\'s") != lower_opp and lower_opp != 'na':
        t5_word_dict[lower_txt] = opp
        return opp

      #allow for the word to appear if it has a negation prefix
      if 'un' in lower_opp or 'non' in lower_opp:
        t5_word_dict[lower_txt] = opp
        return opp
    
    if token_pos == 'ADP':
      t5_word_dict[lower_txt] = get_rand_adposition(lower_txt)
      return t5_word_dict[lower_txt]

  #TODO: return a similar word from T5 ? or fill masked word ?
  t5_word_dict[lower_txt] = get_co_hyponym(token_text)
  return t5_word_dict[lower_txt]

def get_t5_rand_word2_1(token_text, token_pos):
  if not token_pos in ['NOUN', 'VERB', 'ADJ', 'ADP']:
    print(f'get_t5_rand_word2_1 token_pos ({token_pos}) of token: {token_text} is not in [NOUN, VERB, ADJ, ADP]')
    return ''
  
  lower_txt = token_text.lower() 

  check_opposite = True
  if lower_txt in t5_word_dict:
    
    if t5_word_dict[lower_txt]:
      return t5_word_dict[lower_txt]
    
    #if opposite exits (is not "") return it, otherwise no need to check for opposite again
    check_opposite = False

  opposites = []
  if check_opposite:
    opposites = get_t5_opposite(lower_txt)

  if not opposites:
    t5_word_dict[lower_txt] = ''
    opposites = ['na']

  for opp in opposites:
    lower_opp = opp.lower()
    if is_valid_word(lower_opp):
      #do not return the same word
      if lower_txt != lower_opp and (lower_txt+"\'s") != lower_opp and lower_opp != 'na':
        t5_word_dict[lower_txt] = opp
        return opp

      #allow for the word to appear if it has a negation prefix
      if 'un' in lower_opp or 'non' in lower_opp:
        t5_word_dict[lower_txt] = opp
        return opp
    
    if token_pos == 'ADP': 
      if lower_txt in spatial_list:
        return get_rand_spatial_word(lower_txt)
      return "" #prevent returning random co_hyponym for 'ADP'

  #TODO: return a similar word from T5 ? or fill masked word ?
  t5_word_dict[lower_txt] = get_co_hyponym(token_text)
  return t5_word_dict[lower_txt]
  

def get_t5_rand_word2_2(token_text, token_pos):
  if not token_pos in ['NOUN', 'VERB', 'ADJ', 'ADP']:
    print(f'get_t5_rand_word2_1 token_pos ({token_pos}) of token: {token_text} is not in [NOUN, VERB, ADJ, ADP]')
    return ''
  
  lower_txt = token_text.lower() 

  check_opposite = True
  if lower_txt in t5_word_dict:
    
    if t5_word_dict[lower_txt]:
      return t5_word_dict[lower_txt]
    
    #if opposite exits (is not "") return it, otherwise no need to check for opposite again
    check_opposite = False

  opposites = []
  if check_opposite:
    opposites = get_t5_opposite(lower_txt)

  if not opposites:
    t5_word_dict[lower_txt] = ''
    opposites = ['na']

  for opp in opposites:
    lower_opp = opp.lower()
    if is_valid_word(lower_opp):
      #do not return the same word
      if lower_txt != lower_opp and (lower_txt+"\'s") != lower_opp and lower_opp != 'na':
        t5_word_dict[lower_txt] = opp
        return opp

      #allow for the word to appear if it has a negation prefix
      if 'un' in lower_opp or 'non' in lower_opp:
        t5_word_dict[lower_txt] = opp
        return opp
    
    if token_pos == 'ADP': 
      if lower_txt in spatial_list:
        return get_rand_spatial_word(lower_txt)
      #return "" #prevent returning random co_hyponym for 'ADP'
      #return random spatial word is word is in spatial_list but DO NOT prevent returning random co_hyponym for other 'ADP' 

  #TODO: return a similar word from T5 ? or fill masked word ?
  #t5_word_dict[lower_txt] = get_co_hyponym(token_text)
  #return t5_word_dict[lower_txt]
  return get_co_hyponym(token_text) #do not insert to cache to get random word each time
    
#get opposite for spatial adpositions, fill mask for nouns
#def get_t5_rand_word3(token_text, token_pos):
def get_t5_rand_word3(full_doc, tok_to_replace):
   
  if not tok_to_replace.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADP']:
    print(f'get_t5_rand_word3 token_pos ({tok_to_replace.pos_}) of token: {tok_to_replace.text} is not in [NOUN, VERB, ADJ, ADP]')
    return ''
  
  if tok_to_replace.pos_ in ['NOUN']:
     #rand_word = get_t5_rand_mask_fill(masked_text, masked_token)
     rand_word = get_t5_rand_mask_fill(full_doc, tok_to_replace.i, num_words=3)
     if rand_word:
      return rand_word

  lower_txt = tok_to_replace.text.lower() 

  check_opposite = True
  if lower_txt in t5_word_dict:
    
    if t5_word_dict[lower_txt]:
      return t5_word_dict[lower_txt]
    
    #if opposite exits (is not "") return it, otherwise no need to check for opposite again
    check_opposite = False

  opposites = []
  if check_opposite:
    opposites = get_t5_opposite(lower_txt)
    #print(f'get_t5_rand_word2 lower_txt: {lower_txt}, opposites: {opposites}')

  if not opposites:
    t5_word_dict[lower_txt] = ''
    opposites = ['na']

  for opp in opposites:
    lower_opp = opp.lower()
    if is_valid_word(lower_opp):
      #do not return the same word
      if lower_opp != 'na' and lower_txt != lower_opp and (lower_txt+"\'s") != lower_opp:
        t5_word_dict[lower_txt] = opp
        return opp

      #allow for the word to appear if it has a negation prefix
      #if 'un' in lower_opp or 'non' in lower_opp:
      #  t5_word_dict[lower_txt] = opp
      #  return opp
    
    if tok_to_replace.pos_ == 'ADP':
      #if first opp in opposites is not valid then there exists no opposite for this adposition (e.g. for, of, ...)
      t5_word_dict[lower_txt] = ''
      
      if lower_txt in spatial_list:
        return get_rand_spatial_word(lower_txt)  


  #get opposite if exists, otherwise if 'ADP' random spatial adpositions, else fill mask 
def get_t5_rand_word4(full_doc, tok_to_replace):
   
  if not tok_to_replace.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADP']:
    print(f'get_t5_rand_word3 token_pos ({tok_to_replace.pos_}) of token: {tok_to_replace.text} is not in [NOUN, VERB, ADJ, ADP]')
    return ''
  
  #if tok_to_replace.pos_ in ['NOUN']:
  #   rand_word = get_t5_rand_mask_fill(full_doc, tok_to_replace.i, num_words=3)
  #   if rand_word:
  #    return rand_word

  lower_txt = tok_to_replace.text.lower() 

  check_opposite = True
  if lower_txt in t5_word_dict:
    
    if t5_word_dict[lower_txt]:
      return t5_word_dict[lower_txt]
    
    #if opposite is exits (not "" return it, otherwise no need to check for opposite again)
    check_opposite = False

  opposites = []
  if check_opposite:
    opposites = get_t5_opposite(lower_txt)
    #print(f'get_t5_rand_word2 lower_txt: {lower_txt}, opposites: {opposites}')

  if not opposites:
    opposites = ['na']
    t5_word_dict[lower_txt] = ''

  for opp in opposites:
    lower_opp = opp.lower()
    if is_valid_word(lower_opp):
      #do not return the same word
      if lower_opp != 'na' and lower_txt != lower_opp and (lower_txt+"\'s") != lower_opp:
        t5_word_dict[lower_txt] = opp
        return opp

      #allow for the word to appear if it has a negation prefix
      #if 'un' in lower_opp or 'non' in lower_opp:
      #  t5_word_dict[lower_txt] = opp
      #  return opp
    
    if tok_to_replace.pos_ == 'ADP':
      #if first opp in opposites is not valid then there exists no opposite for this adposition (e.g. for, of, ...)
      t5_word_dict[lower_txt] = ''
      
      if lower_txt in spatial_list:
        return get_rand_spatial_word(lower_txt)   


  #get_t5_rand_mask_fill can change with the context of the full sentence, get_co_hyponym returns random co_hyponym
  #so don't save in cache. just mark it as having no opposite
  t5_word_dict[lower_txt] = ''

  rand_word = get_t5_rand_mask_fill(full_doc, tok_to_replace.i, num_words=3)
  if rand_word:
    return rand_word

  return get_co_hyponym(lower_txt)
    

"""
if token_pos == 'ADP':
      t5_word_dict[lower_txt] = get_rand_adposition(lower_txt)
      return t5_word_dict[lower_txt]
"""
def get_t5_rand_word4_1(full_doc, tok_to_replace):
   
  if not tok_to_replace.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADP']:
    print(f'get_t5_rand_word3 token_pos ({tok_to_replace.pos_}) of token: {tok_to_replace.text} is not in [NOUN, VERB, ADJ, ADP]')
    return ''
  
  #if tok_to_replace.pos_ in ['NOUN']:
  #   rand_word = get_t5_rand_mask_fill(full_doc, tok_to_replace.i, num_words=3)
  #   if rand_word:
  #    return rand_word

  lower_txt = tok_to_replace.text.lower() 

  check_opposite = True
  if lower_txt in t5_word_dict:
    
    if t5_word_dict[lower_txt]:
      return t5_word_dict[lower_txt]
    
    #if opposite is exits (not "" return it, otherwise no need to check for opposite again)
    check_opposite = False

  opposites = []
  if check_opposite:
    opposites = get_t5_opposite(lower_txt)
    #print(f'get_t5_rand_word2 lower_txt: {lower_txt}, opposites: {opposites}')

  if not opposites:
    opposites = ['na']
    t5_word_dict[lower_txt] = ''

  for opp in opposites:
    lower_opp = opp.lower()
    if is_valid_word(lower_opp):
      #do not return the same word
      if lower_opp != 'na' and lower_txt != lower_opp and (lower_txt+"\'s") != lower_opp:
        t5_word_dict[lower_txt] = opp
        return opp

      #allow for the word to appear if it has a negation prefix
      #if 'un' in lower_opp or 'non' in lower_opp:
      #  t5_word_dict[lower_txt] = opp
      #  return opp
    
    if tok_to_replace.pos_ == 'ADP':
      #if first opp in opposites is not valid then there exists no opposite for this adposition (e.g. for, of, ...)
      t5_word_dict[lower_txt] = ''
      
      if lower_txt in prepositions_list_2: 
        return get_rand_adposition_2(lower_txt) 


  #get_t5_rand_mask_fill can change with the context of the full sentence, get_co_hyponym returns random co_hyponym
  #so don't save in cache. just mark it as having no opposite
  t5_word_dict[lower_txt] = ''

  rand_word = get_t5_rand_mask_fill(full_doc, tok_to_replace.i, num_words=3)
  if rand_word and rand_word.lower() != lower_txt:
    return rand_word

  return get_co_hyponym(lower_txt)

def get_new_sentences(orig_text, pos_to_replace=['NOUN', 'ADJ']):
  #print(f'orig_text: {orig_text}','\n')
  #doc = nlp(orig_text)
  #orig_text_lst = orig_text.text.split()
  orig_text_tokens_lst = [token.text for token in orig_text]
  
  new_sentences = []
  token_index = 0
  #new_text_lst = []
  for token in orig_text:
    #print(f'token.text: {token.text}, index: {token_index}, dep_: {token.dep_}, pos_: {token.pos_}, head: {token.head.text}\n')
    #print(f'token.text: {token.text}, token_index: {token_index}, token.i: {token.i}, pos_: {token.pos_}\n')
    #new_text_lst.append(token.text)
    if token.pos_ in pos_to_replace:
      
      #print(f'token {token.text} index: {token_index} is {token.pos_}\n')
      #print(f'orig_text_lst: {orig_text_lst}\n')
      
      rand_word = get_rand_word(token.text, token.pos_)
      #print(f'rand_word: {rand_word}')
      if rand_word != '':
        #new_text_lst = orig_text_lst.copy()
        new_text_lst = orig_text_tokens_lst.copy()
        new_text_lst[token_index] = rand_word
        #new_text_lst[-1] = rand_word
        #print(f'new_text_lst: {new_text_lst}')
        new_text = ' '.join(new_text_lst)
        #print(f'new_text: {new_text}')
        new_sentences.append(new_text)
    
    token_index += 1

  #print(f'new_sentences: {new_sentences}\n')
  return new_sentences



def get_possible_mask_fill_list(orig_doc, mask_idx, num_words=3):
  #print(f'get_possible_mask_fill_list: orig_doc text: {orig_doc.text}')
  orig_text_tokens_lst = [token.text for token in orig_doc]
  masked_word = orig_text_tokens_lst[mask_idx]
  
  new_sentences = []
  #new_text_lst = []
  mask_token = '<extra_id_0>'
 
  new_text_lst = orig_text_tokens_lst.copy()
  new_text_lst[mask_idx] = mask_token
  masked_text = ' '.join(new_text_lst)

  #print(f'get_possible_mask_fill_list: masked_word: {masked_word}, masked_text: {masked_text}')


  rand_words = get_t5_rand_mask_fill_list(masked_text, num_words)
  #print(f'get_possible_mask_fill_list: rand_words: {rand_words}')
  rand_words = [w.translate(str.maketrans('', '', string.punctuation)) for w in rand_words]
  rand_words = set(rand_words)
  #rand_words.discard(masked_word)
  #rand_words.discard('s')
  rand_words.difference_update([masked_word, 's', 'and', 'or', '.', 'for', ''])
  rand_words = list(rand_words)
  #print(f'get_possible_mask_fill_list: rand_words: {rand_words}')

  return rand_words

#def get_t5_new_sentences(model, tokenizer, orig_doc, pos_to_replace=['NOUN', 'ADJ']):
def get_t5_new_sentences(orig_doc, pos_to_replace=['NOUN', 'ADJ']):
  #print(f'orig_text: {orig_text}','\n')
  #doc = nlp(orig_text)
  #orig_text_lst = orig_text.text.split()
  orig_text_tokens_lst = [token.text for token in orig_doc]
  
  new_sentences = []
  token_index = 0
  #new_text_lst = []
  mask_token = '<extra_id_0>'
  #TODO: can encode the entire orig_doc.text once
  for token in orig_doc:
    #print(f'token.text: {token.text}, index: {token_index}, dep_: {token.dep_}, pos_: {token.pos_}, head: {token.head.text}\n')
    #print(f'token.text: {token.text}, token_index: {token_index}, token.i: {token.i}, pos_: {token.pos_}\n')
    #new_text_lst.append(token.text)
    if token.pos_ in pos_to_replace:
      
      #print(f'token {token.text} index: {token_index} is {token.pos_}\n')
      #print(f'orig_text_lst: {orig_text_lst}\n')
      new_text_lst = orig_text_tokens_lst.copy()
      new_text_lst[token_index] = mask_token
      masked_text = ' '.join(new_text_lst)

      rand_words = get_t5_rand_word(masked_text, token)
      #print(f'rand_word: {rand_word}')
      if rand_words != '':
        
        new_text_lst[token_index] = rand_words
        
        #print(f'new_text_lst: {new_text_lst}')
        new_text = ' '.join(new_text_lst)
        #print(f'new_text: {new_text}')
        new_sentences.append(new_text)
    
    token_index += 1

  #print(f'new_sentences: {new_sentences}\n')
  return new_sentences



def get_t5_new_sentences2_lemmas(orig_text, pos_to_replace=['NOUN', 'ADJ']):
  #print(f'orig_text: {orig_text}','\n')
  #doc = nlp(orig_text)
  #orig_text_lst = orig_text.text.split()
  orig_text_tokens_lst = [token.text for token in orig_text]
  #print(f'get_t5_new_sentences2 orig_text.text: {orig_text.text}')
  new_sentences = []
  token_index = 0
  #new_text_lst = []
  for token in orig_text:
    #print(f'token.text: {token.text}, index: {token_index}, dep_: {token.dep_}, pos_: {token.pos_}, head: {token.head.text}\n')
    #print(f'token.text: {token.text}, token_index: {token_index}, token.i: {token.i}, pos_: {token.pos_}\n')
    #new_text_lst.append(token.text)
    if token.pos_ in pos_to_replace:
      
      #print(f'token {token.text} index: {token_index} is {token.pos_}\n')
      #print(f'orig_text_lst: {orig_text_lst}\n')
      
      rand_word = get_t5_rand_word2_lemmas(token.text, token.pos_)
      #print(f'rand_word: {rand_word}')
      if rand_word != '':
        #new_text_lst = orig_text_lst.copy()
        new_text_lst = orig_text_tokens_lst.copy()
        new_text_lst[token_index] = rand_word
        #new_text_lst[-1] = rand_word
        #print(f'new_text_lst: {new_text_lst}')
        new_text = ' '.join(new_text_lst)
        #print(f'new_text: {new_text}')
        new_sentences.append(new_text)
    
    token_index += 1

  #print(f'get_t5_new_sentences2 new_sentences: {new_sentences}\n')
  return new_sentences


def get_t5_new_sentences2_spatial_negate(orig_doc, pos_to_replace=['NOUN', 'ADJ']):
  #print(f'orig_text: {orig_text}','\n')
  #doc = nlp(orig_text)
  #orig_text_lst = orig_text.text.split()
  orig_text_tokens_lst = [token.text for token in orig_doc]
  #print(f'get_t5_new_sentences2 orig_text.text: {orig_text.text}')
  new_sentences = []

  orig_text = orig_doc.text
  shuffled_keys = list(spatial_negate.keys())
  random.shuffle(shuffled_keys)
  for k in shuffled_keys:
    if k in orig_text:
      new_text = orig_text.replace(k, spatial_negate[k])
      new_sentences.append(new_text)
      break

  token_index = 0
  #new_text_lst = []
  for token in orig_doc:
    #print(f'token.text: {token.text}, index: {token_index}, dep_: {token.dep_}, pos_: {token.pos_}, head: {token.head.text}\n')
    #print(f'token.text: {token.text}, token_index: {token_index}, token.i: {token.i}, pos_: {token.pos_}\n')
    #new_text_lst.append(token.text)
    if token.pos_ in pos_to_replace:
      
      #print(f'token {token.text} index: {token_index} is {token.pos_}\n')
      #print(f'orig_text_lst: {orig_text_lst}\n')
      
      # rand_word = get_rb_rand_word(token.text, token.pos_)
      rand_word = None

      if not rand_word:
        rand_word = get_t5_rand_word2(token.text, token.pos_)
      #print(f'rand_word: {rand_word}')
      if rand_word != '':
        #new_text_lst = orig_text_lst.copy()
        new_text_lst = orig_text_tokens_lst.copy()
        new_text_lst[token_index] = rand_word
        #new_text_lst[-1] = rand_word
        #print(f'new_text_lst: {new_text_lst}')
        new_text = ' '.join(new_text_lst)
        #print(f'new_text: {new_text}')
        new_sentences.append(new_text)
    
    token_index += 1

  #print(f'get_t5_new_sentences2 new_sentences: {new_sentences}\n')
  return new_sentences


# get_t5_new_sentences2_max_neg(orig_text, pos_to_replace=['NOUN', 'ADJ'], curr_neg=0, max_neg=math.inf):
#def get_t5_new_sentences2(orig_text, pos_to_replace=['NOUN', 'ADJ'], num_neg=0, max_neg=math.inf):
def get_t5_new_sentences2(orig_text, pos_to_replace=['NOUN', 'ADJ']):
  #print(f'orig_text: {orig_text}','\n')
  #doc = nlp(orig_text)
  #orig_text_lst = orig_text.text.split()
  orig_text_tokens_lst = [token.text for token in orig_text]
  #print(f'get_t5_new_sentences2 orig_text.text: {orig_text.text}')
  new_sentences = []
  # token_index = 0
  #new_text_lst = []
  token_list = [(i, t) for i, t in enumerate(orig_text)]
  random.shuffle(token_list)
  # for token in orig_text:
  for token_pair in token_list:
    #print(f'token.text: {token.text}, index: {token_index}, dep_: {token.dep_}, pos_: {token.pos_}, head: {token.head.text}\n')
    #print(f'token.text: {token.text}, token_index: {token_index}, token.i: {token.i}, pos_: {token.pos_}\n')
    #new_text_lst.append(token.text)
    #if num_neg >= max_neg:
     # return new_sentences

    if token_pair[1].pos_ in pos_to_replace:
      
      #print(f'token {token.text} index: {token_index} is {token.pos_}\n')
      #print(f'orig_text_lst: {orig_text_lst}\n')
      
      # rand_word = get_rb_rand_word(token.text, token.pos_)
      rand_word = None

      if not rand_word:
        rand_word = get_t5_rand_word2(token_pair[1].text, token_pair[1].pos_)
      #print(f'rand_word: {rand_word}')
      if rand_word != '':
        #new_text_lst = orig_text_lst.copy()
        new_text_lst = orig_text_tokens_lst.copy()
        new_text_lst[token_pair[0]] = rand_word
        #new_text_lst[-1] = rand_word
        #print(f'new_text_lst: {new_text_lst}')
        new_text = ' '.join(new_text_lst)
        #print(f'new_text: {new_text}')
        new_sentences.append(new_text)
        #num_neg += 1
    
    # token_index += 1

  #print(f'get_t5_new_sentences2 new_sentences: {new_sentences}\n')
  return new_sentences



def get_t5_new_sentences2_1(orig_text, pos_to_replace=['NOUN', 'ADJ']):
  #print(f'orig_text: {orig_text}','\n')
  #doc = nlp(orig_text)
  #orig_text_lst = orig_text.text.split()
  orig_text_tokens_lst = [token.text for token in orig_text]
  #print(f'get_t5_new_sentences2_1 orig_text.text: {orig_text.text}')
  new_sentences = []
  token_index = 0
  #new_text_lst = []
  for token in orig_text:
    #print(f'token.text: {token.text}, index: {token_index}, dep_: {token.dep_}, pos_: {token.pos_}, head: {token.head.text}\n')
    #print(f'token.text: {token.text}, token_index: {token_index}, token.i: {token.i}, pos_: {token.pos_}\n')
    #new_text_lst.append(token.text)
    if token.pos_ in pos_to_replace:
      
      #print(f'token {token.text} index: {token_index} is {token.pos_}\n')
      #print(f'orig_text_lst: {orig_text_lst}\n')
      
      rand_word = get_t5_rand_word2_1(token.text, token.pos_)
      #print(f'rand_word: {rand_word}')
      if rand_word != '':
        #new_text_lst = orig_text_lst.copy()
        new_text_lst = orig_text_tokens_lst.copy()
        new_text_lst[token_index] = rand_word
        #new_text_lst[-1] = rand_word
        #print(f'new_text_lst: {new_text_lst}')
        new_text = ' '.join(new_text_lst)
        #print(f'new_text: {new_text}')
        new_sentences.append(new_text)
    
    token_index += 1

  #print(f'get_t5_new_sentences2 new_sentences: {new_sentences}\n')
  return new_sentences


def get_t5_new_sentences2_2(orig_text, pos_to_replace=['NOUN', 'ADJ']):
  #print(f'orig_text: {orig_text}','\n')
  #doc = nlp(orig_text)
  #orig_text_lst = orig_text.text.split()
  orig_text_tokens_lst = [token.text for token in orig_text]
  #print(f'get_t5_new_sentences2_1 orig_text.text: {orig_text.text}')
  new_sentences = []
  token_index = 0
  #new_text_lst = []
  for token in orig_text:
    #print(f'token.text: {token.text}, index: {token_index}, dep_: {token.dep_}, pos_: {token.pos_}, head: {token.head.text}\n')
    #print(f'token.text: {token.text}, token_index: {token_index}, token.i: {token.i}, pos_: {token.pos_}\n')
    #new_text_lst.append(token.text)
    if token.pos_ in pos_to_replace:
      
      #print(f'token {token.text} index: {token_index} is {token.pos_}\n')
      #print(f'orig_text_lst: {orig_text_lst}\n')
      
      rand_word = get_t5_rand_word2_2(token.text, token.pos_)
      #print(f'rand_word: {rand_word}')
      if rand_word != '':
        #new_text_lst = orig_text_lst.copy()
        new_text_lst = orig_text_tokens_lst.copy()
        new_text_lst[token_index] = rand_word
        #new_text_lst[-1] = rand_word
        #print(f'new_text_lst: {new_text_lst}')
        new_text = ' '.join(new_text_lst)
        #print(f'new_text: {new_text}')
        new_sentences.append(new_text)
    
    token_index += 1

  #print(f'get_t5_new_sentences2 new_sentences: {new_sentences}\n')
  return new_sentences

def get_t5_new_sentences3(full_doc, doc_to_mod, pos_to_replace=['NOUN', 'ADJ']):
  #print(f'orig_text: {orig_text}','\n')
  #doc = nlp(orig_text)
  #orig_text_lst = orig_text.text.split()
  orig_text_tokens_lst = [token.text for token in doc_to_mod]
  
  new_sentences = []
  token_index = 0
  #new_text_lst = []
  for token in doc_to_mod:
    
    if token.pos_ in pos_to_replace:  
      
      rand_word = get_t5_rand_word3(full_doc, token)
      #print(f'rand_word: {rand_word}')
      if rand_word != '':
        #new_text_lst = orig_text_lst.copy()
        new_text_lst = orig_text_tokens_lst.copy()
        new_text_lst[token_index] = rand_word
        #new_text_lst[-1] = rand_word
        #print(f'new_text_lst: {new_text_lst}')
        new_text = ' '.join(new_text_lst)
        #print(f'new_text: {new_text}')
        new_sentences.append(new_text)
    
    token_index += 1

  #print(f'new_sentences: {new_sentences}\n')
  return new_sentences


def get_t5_new_sentences4(full_doc, doc_to_mod, pos_to_replace=['NOUN', 'ADJ']):
  #print(f'orig_text: {orig_text}','\n')
  #doc = nlp(orig_text)
  #orig_text_lst = orig_text.text.split()
  orig_text_tokens_lst = [token.text for token in doc_to_mod]
  
  new_sentences = []
  token_index = 0
  #new_text_lst = []
  for token in doc_to_mod:
    
    if token.pos_ in pos_to_replace:  
      
      rand_word = get_t5_rand_word4(full_doc, token)
      #print(f'rand_word: {rand_word}')
      if rand_word != '':
        #new_text_lst = orig_text_lst.copy()
        new_text_lst = orig_text_tokens_lst.copy()
        new_text_lst[token_index] = rand_word
        #new_text_lst[-1] = rand_word
        #print(f'new_text_lst: {new_text_lst}')
        new_text = ' '.join(new_text_lst)
        #print(f'new_text: {new_text}')
        new_sentences.append(new_text)
    
    token_index += 1

  #print(f'new_sentences: {new_sentences}\n')
  return new_sentences




def get_t5_new_sentences4_1(full_doc, doc_to_mod, pos_to_replace=['NOUN', 'ADJ']):
  #print(f'orig_text: {orig_text}','\n')
  #doc = nlp(orig_text)
  #orig_text_lst = orig_text.text.split()
  orig_text_tokens_lst = [token.text for token in doc_to_mod]
  
  new_sentences = []
  token_index = 0
  #new_text_lst = []
  for token in doc_to_mod:
    
    if token.pos_ in pos_to_replace:  
      
      rand_word = get_t5_rand_word4_1(full_doc, token)
      #print(f'rand_word: {rand_word}')
      if rand_word != '':
        #new_text_lst = orig_text_lst.copy()
        new_text_lst = orig_text_tokens_lst.copy()
        new_text_lst[token_index] = rand_word
        #new_text_lst[-1] = rand_word
        #print(f'new_text_lst: {new_text_lst}')
        new_text = ' '.join(new_text_lst)
        #print(f'new_text: {new_text}')
        new_sentences.append(new_text)
    
    token_index += 1

  #print(f'new_sentences: {new_sentences}\n')
  return new_sentences


def get_new_noun_phrases(noun_phrase):
  #print(f'get_new_noun_phrases')
  return get_new_sentences(noun_phrase, pos_to_replace=['NOUN', 'ADJ', 'ADP', 'VERB'])

def get_new_verb_phrases(verb_phrase):
  #print(f'get_new_verb_phrases')
  return get_new_sentences(verb_phrase, pos_to_replace=['VERB'])


def expand_token(token, num_expansaions=3):
  #print(f'\nexpand_token: {token.text}, pos: {token.pos_}\n')
  
  if token.pos_ == 'ADP':
    adpositions = get_rand_spatial_word_list(token.text)
    
    num_words = min(len(adpositions),num_expansaions)
    #seed_everything()
    return random.sample(adpositions, num_words)


  res = []
  nltk_pos = None
  if token.pos_ == 'ADJ':
    nltk_pos = wn.ADJ
    #nltk_pos = 'a'
  elif token.pos_ == 'NOUN':
    nltk_pos = wn.NOUN
    #nltk_pos = 'n'
  elif token.pos_ == 'VERB':
    nltk_pos = wn.VERB
    #nltk_pos = 'v'
  #print(f'nltk_pos: {nltk_pos}\n')
  new_token_captions = []
  #TODO: get synset with the correct POS 
  #TODO: get synset with the correct POS also in training get co_hyponyn and antonym
  #for syn in wn.synsets(token.text):
  #syn = wn.synsets(token.text, pos=token.pos_)
  #if sys[]
  #for syn in wn.synsets(token.text, pos=[nltk_pos]):
  
  #syn_list = wn.synsets(token.text, pos=[nltk_pos])
  syn_list = wn.synsets(token.text, pos=nltk_pos)
  if syn_list:

    syn = syn_list[0]
    """
    print(f'syn: {syn}','\n')
    print(f'syn.lexname(): {syn.lexname()}','\n')
    split = syn.lexname().split('.')
    print(f'syn.lexname().split(): {split}','\n')
    supersense = split[1]
    print(f'supersense: {supersense}','\n')
    supersense_syn = wn.synsets(supersense, pos=[nltk_pos])
    print(f'supersense_syn: {supersense_syn}','\n')
    if supersense_syn:
      print(f'supersense_syn[0].hyponyms(): {supersense_syn[0].hyponyms()}','\n')

    
    print(f'syn.definition(): {syn.definition()}','\n')
    print(f'syn.examples(): {syn.examples()}','\n')
    print(f'syn.member_holonyms(): {syn.member_holonyms()}','\n')
    print(f'syn.substance_holonyms(): {syn.substance_holonyms()}','\n')
    print(f'syn.part_holonyms(): {syn.part_holonyms()}','\n')
    print(f'syn.part_holonyms(): {syn.part_holonyms()}','\n')
    print(f'syn.member_meronyms(): {syn.member_meronyms()}','\n')
    print(f'syn.substance_meronyms(): {syn.substance_meronyms()}','\n')
    print(f'syn.part_meronyms(): {syn.part_meronyms()}','\n')
    print(f'syn.topic_domains(): {syn.topic_domains()}','\n')
    print(f'syn.in_topic_domains(): {syn.in_topic_domains()}','\n')
    print(f'syn.region_domains(): {syn.region_domains()}','\n')
    print(f'syn.in_region_domains(): {syn.in_region_domains()}','\n')
    print(f'syn.usage_domains(): {syn.usage_domains()}','\n')
    print(f'syn.in_usage_domains(): {syn.in_usage_domains()}','\n')
    print(f'syn.attributes(): {syn.attributes()}','\n')
    print(f'syn.entailments(): {syn.entailments()}','\n')
    print(f'syn.causes(): {syn.causes()}','\n')
    print(f'syn.also_sees(): {syn.also_sees()}','\n')
    print(f'syn.verb_groups(): {syn.verb_groups()}','\n')
    print(f'syn.similar_tos(): {syn.similar_tos()}','\n')
    print(f'syn.root_hypernyms(): {syn.root_hypernyms()}','\n')
    """
    ########################
    ########################
    if token.pos_ in ['NOUN', 'VERB']:
      co_hyponyms = get_co_hyponym_list(token.text)
      #TODO: add co co_hyponyms ? (all grandsons of grand father), add more relations?

      if not co_hyponyms:
        #print(f'no co_hyponyms found for word: {token.text}\n')
        return res

      num_words = min(len(co_hyponyms),num_expansaions)
      #seed_everything()
      return random.sample(co_hyponyms, num_words)

      #print(f'result: {res}')
    ########################
    ########################

    if token.pos_ == 'ADJ':
      similar_to_names = []
      similar_to = syn.similar_tos()

      for sim in similar_to:
        similar_to_names.extend( [l.name() for l in sim.lemmas()] )
      
      #print(f'similar_to_names: {similar_to_names}','\n')
  
      similar_to_names = set(similar_to_names)
      similar_to_names.discard(token.text)
      similar_to_names = list(similar_to_names)

      if not similar_to_names:
        #print(f'no similar_to_names found for word: {token.text}\n')
        return res

      similar_to_names.sort()
      num_words = min(len(similar_to_names),num_expansaions)
      #seed_everything()
      return random.sample(similar_to_names, num_words)
      
      #attributes = syn.attributes()
      #if attributes:
      #  attributes_names = []
      #  for attr in attributes:
      #    attributes_names.extend( [l.name() for l in attr.lemmas()] )
        
        #seed_everything()
      #  attr = random.choice(attributes_names)
        #print(f'attr: {attr}')
      #  res = [w if attr in w else f'{w} {attr}' for w in res]

      #print(f'result: {res}')
    
    return res

def expand_t5_token(orig_doc, mask_idx, num_expansaions=3):
  #print(f'\nexpand_token: {token.text}, pos: {token.pos_}\n')
  
  possible_words = get_possible_mask_fill_list(orig_doc, mask_idx, num_words=num_expansaions)

  if token.pos_ == 'ADP':
    adpositions = get_rand_spatial_word_list(token.text)
    
    num_words = min(len(adpositions),num_expansaions)
    #seed_everything()
    return random.sample(adpositions, num_words)


  res = []
  nltk_pos = None
  if token.pos_ == 'ADJ':
    nltk_pos = wn.ADJ
    #nltk_pos = 'a'
  elif token.pos_ == 'NOUN':
    nltk_pos = wn.NOUN
    #nltk_pos = 'n'
  elif token.pos_ == 'VERB':
    nltk_pos = wn.VERB
    #nltk_pos = 'v'
  #print(f'nltk_pos: {nltk_pos}\n')
  new_token_captions = []
  #TODO: get synset with the correct POS 
  #TODO: get synset with the correct POS also in training get co_hyponyn and antonym
  #for syn in wn.synsets(token.text):
  #syn = wn.synsets(token.text, pos=token.pos_)
  #if sys[]
  #for syn in wn.synsets(token.text, pos=[nltk_pos]):
  
  #syn_list = wn.synsets(token.text, pos=[nltk_pos])
  syn_list = wn.synsets(token.text, pos=nltk_pos)
  if syn_list:

    syn = syn_list[0]
    
    ########################
    ########################
    if token.pos_ in ['NOUN', 'VERB']:
      co_hyponyms = get_co_hyponym_list(token.text)
      #TODO: add co co_hyponyms ? (all grandsons of grand father), add more relations?

      if not co_hyponyms:
        #print(f'no co_hyponyms found for word: {token.text}\n')
        return res

      num_words = min(len(co_hyponyms),num_expansaions)
      #seed_everything()
      return random.sample(co_hyponyms, num_words)

      #print(f'result: {res}')
    ########################
    ########################

    if token.pos_ == 'ADJ':
      similar_to_names = []
      similar_to = syn.similar_tos()

      for sim in similar_to:
        similar_to_names.extend( [l.name() for l in sim.lemmas()] )
      
      #print(f'similar_to_names: {similar_to_names}','\n')
  
      similar_to_names = set(similar_to_names)
      similar_to_names.discard(token.text)
      similar_to_names = list(similar_to_names)

      if not similar_to_names:
        #print(f'no similar_to_names found for word: {token.text}\n')
        return res

      similar_to_names.sort()
      num_words = min(len(similar_to_names),num_expansaions)
      #seed_everything()
      return random.sample(similar_to_names, num_words)
      
      #attributes = syn.attributes()
      #if attributes:
      #  attributes_names = []
      #  for attr in attributes:
      #    attributes_names.extend( [l.name() for l in attr.lemmas()] )
        
        #seed_everything()
      #  attr = random.choice(attributes_names)
        #print(f'attr: {attr}')
      #  res = [w if attr in w else f'{w} {attr}' for w in res]

      #print(f'result: {res}')
    
    return res

def get_more_caption_tokens(predicted_token, correct_token):
  #print(f'get_more_caption_tokens')
  #def common_hypernyms(self, other)
  #def lowest_common_hypernyms(self, other, simulate_root=False, use_min_depth=False):
  #def path_similarity(self, other, verbose=False, simulate_root=True):
  #def res_similarity(self, other, ic, verbose=False):
  #def jcn_similarity(self, other, ic, verbose=False):
  #def lin_similarity(self, other, ic, verbose=False):
  #new_tokens_texts = []
  predicted_expansion = expand_token(predicted_token)
  correct_expansion = expand_token(correct_token)
  #new_tokens_texts.extend(predicted_expansion)
  #new_tokens_texts.extend(correct_expansion)
  return predicted_expansion, correct_expansion

def get_new_texts_list(orig_text, tok_index, new_words):
  res = []
  if not new_words:
    return res
  orig_text_lst = [token.text for token in orig_text]

  for word in new_words:
    new_text_lst = orig_text_lst.copy()
    new_text_lst[tok_index] = word
    new_text = ' '.join(new_text_lst)
    res.append(new_text)

  return res

def get_nouns_before_and_after_token(token_index, doc):
  before_index = -1
  after_index = -1
  noun_before = ""
  noun_after = ""

  for tok in doc:
    if tok.pos_ == 'NOUN':
      if tok.i < token_index:
        noun_before = tok.text
      elif tok.i > token_index:
        #this is the closest noun after the token
        noun_after = tok.text
        return noun_before, noun_after
  
  return noun_before, noun_after

def get_containing_noun_chunk(token_index, doc):
  chunk_text_lst = []
  word_index = -1
  
  for chunk in doc.noun_chunks:
    #should be enough to check only token_index_ < chunk.end 
    if token_index >= chunk.start and token_index < chunk.end:
      
      chunk_text_lst = [token.text for token in chunk]
      word_index = token_index - chunk.start #index in chunk_text_lst
      return chunk_text_lst, word_index 
  
  return chunk_text_lst, word_index
    
#output only parts of the current node caption without including previous nodes captions
def expand_caption2(predicted_caption, correct_caption):
  predicted_caption = predicted_caption.lower()
  correct_caption = correct_caption.lower()
  
  predicted_doc = nlp(predicted_caption)
  correct_doc = nlp(correct_caption)
  correct_doc_len = len(correct_doc)
  #full_doc = nlp(full_caption)
  correct_doc_start_token_index = -1 

  #orig_predicted_text_tokens_lst = [token.text for token in predicted_caption]
  #orig_correct_text_tokens_lst = [token.text for token in correct_caption]
  token_index = 0
  new_captions = []
  possible_expansion = []
  #new_predicted_captions = []
  #new_correct_captions = []
  for predicted_token, correct_token in zip(predicted_doc, correct_doc):
    #print(f'\npredicted_token.text: {predicted_token.text}, correct_token.text: {correct_token.text}')
    #print(f'\npredicted_token.pos_: {predicted_token.pos_}, correct_token.pos_: {correct_token.pos_}')
    if predicted_token.text != correct_token.text:
      
      #predicted_expansion = get_possible_mask_fill_list(predicted_doc, token_index, predicted_token)
      #correct_expansion = get_possible_mask_fill_list(correct_doc, token_index, correct_token)
      mask_fills = get_possible_mask_fill_list(correct_doc, token_index)
      new_predicted_tokens_texts, new_correct_tokens_texts = get_more_caption_tokens(predicted_token, correct_token)

      possible_expansion.extend(mask_fills)
      if new_predicted_tokens_texts:
        possible_expansion.extend(new_predicted_tokens_texts)
      if new_correct_tokens_texts:
        possible_expansion.extend(new_correct_tokens_texts)
      possible_expansion.append(predicted_token.text)
      

      possible_expansion = set(possible_expansion)
      possible_expansion = list(possible_expansion)
      if possible_expansion and correct_token.text not in possible_expansion: #make sure correct token is last in the list
        possible_expansion.append(correct_token.text)


      if not possible_expansion or correct_token.pos_ == 'NOUN':
        #enough to return "a photo of <noun>"
        return possible_expansion

      
      if correct_token.pos_ == 'ADJ':
        #need to return "a photo of <adj> <noun>" (use the original noun chunk of the token)
        #alternatively "a photo of <adj> object"
        #return [f"{exp} {noun_after}" for exp in possible_expansion]

        chunk_text_lst, word_index = get_containing_noun_chunk(token_index, correct_doc)
        
        res = []
        for word in possible_expansion:
          if word:
            new_text_lst = chunk_text_lst.copy()
            try:
              new_text_lst[word_index] = word
              new_text = ' '.join(new_text_lst)
              res.append(new_text)
  
            except:
              print(f'possible_expansion index out of range - new_text_lst: {new_text_lst}, word: {word}, token_index: {token_index}, word_index: {word_index}')
              print(f'possible_expansion: {possible_expansion}')
            
        return res            
      
      noun_before, noun_after = get_nouns_before_and_after_token(token_index, correct_doc)

      if correct_token.pos_ == 'VERB':
        #need to return "a photo of <noun> <verb>" (use the noun before the verb)
        #alternatively "a photo of object <verb>"
        #TODO: some verbs need <noun> <verb> <noun> (e.g. a <man> <doing> <something>)
        return [f"{noun_before} {exp}" for exp in possible_expansion]

      if correct_token.pos_ == 'ADP':
        #need to return "a photo of <noun> <adp> <noun>" (use the noun before and after the adposition)
        #alternatively "a photo of object <adp> object"
        return [f"{noun_before} {exp} {noun_after}" for exp in possible_expansion]

    token_index += 1

  return possible_expansion
  
def find_different_tokens(correct_caption, incorrect_caption):
  #print(f'find_different_tokens. correct_caption: {correct_caption}, incorrect_caption: {incorrect_caption}')
  incorrect_caption = incorrect_caption.lower()
  correct_caption = correct_caption.lower()
  
  incorrect_doc = nlp(incorrect_caption)
  correct_doc = nlp(correct_caption)

  token_index = 0

  for incorrect_token, correct_token in zip(incorrect_doc, correct_doc):

    if incorrect_token.text != correct_token.text:
      return correct_token, incorrect_token, token_index

    token_index += 1
  
  return None, None, -1

def find_word_difference(correct_caption, incorrect_caption):

  correct_token, incorrect_token, token_index = find_different_tokens(correct_caption, incorrect_caption)
  if token_index == -1:
    return correct_token, incorrect_token, token_index
  else:
    return correct_token.text, incorrect_token.text, correct_token.pos_


def expand_caption(predicted_caption, correct_caption):
  predicted_caption = predicted_caption.lower()
  correct_caption = correct_caption.lower()
  
  predicted_doc = nlp(predicted_caption)
  correct_doc = nlp(correct_caption)

  #orig_predicted_text_tokens_lst = [token.text for token in predicted_caption]
  #orig_correct_text_tokens_lst = [token.text for token in correct_caption]
  token_index = 0
  new_captions = []
  #new_predicted_captions = []
  #new_correct_captions = []
  for predicted_token, correct_token in zip(predicted_doc, correct_doc):
    #print(f'\npredicted_token.text: {predicted_token.text}, correct_token.text: {correct_token.text}')
    #print(f'\npredicted_token.pos_: {predicted_token.pos_}, correct_token.pos_: {correct_token.pos_}')
    if predicted_token.text != correct_token.text:
      new_predicted_tokens_texts, new_correct_tokens_texts = get_more_caption_tokens(predicted_token, correct_token)
      
      new_predicted_captions = get_new_texts_list(predicted_doc, token_index, new_predicted_tokens_texts)
      new_correct_captions = get_new_texts_list(correct_doc, token_index, new_correct_tokens_texts)
      new_captions.extend(new_predicted_captions)
      new_captions.extend(new_correct_captions)

    token_index += 1

  new_captions.append(predicted_caption)
  new_captions.append(correct_caption)
  new_captions = list(set(new_captions))
  return new_captions


def make_annotations(x_pos , y_pos, texts, font_size=10, font_color='rgb(250,250,250)'):
    L=len(x_pos)
    if len(texts)!=L:
        raise ValueError('The lists x_pos and texts must have the same len')
    annotations = []
    for k in range(L):
        annotations.append(
            dict(
                text=texts[k], 
                x=x_pos, y=y_pos,
                xref='x1', yref='y1',
                font=dict(color=font_color, size=font_size),
                showarrow=False)
        )
    return annotations

def get_box_text(text, max_line_width=20, break_every=3):
  #TODO use textwrap.TextWrapper(width=50) ??
  words = text.split(' ')
  ###
  box_text = ""
  word_count = 0
  line_width = 0
  for word in words:
    word_width = len(word)

    if line_width + word_width > max_line_width:
      #box_text += "\n"
      box_text += "<br>"
      line_width = 0
    
    box_text += word + " "
    line_width += word_width + 1
    #word_count += 1
    #if break_every == word_count:
    #  #box_text += "\n"
    #  box_text += "<br>"
    #  word_count = 0

  #print(f'\n\nget_box_text - \ntext: \n{text}, \nbox_text: \n{box_text}')
  return box_text


def get_caption_tree(original_caption):
  #print(f'\noriginal_sentence: {original_caption}\n')
  original_caption = original_caption.lower()
  doc = nlp(original_caption)
  #sub_sentences = []

  #for token in doc:
  #  print(token.text, token.dep_, token.head.text, token.pos_)

  #class Node():
  #  def __init__(self, node_num=0, prompt=None, prompt_encoding=None):

  node_index = 0
  root = Node(node_num=node_index, prompt='entity')
  node_index += 1
  true_label_path = []
  all_tree_nodes = []
  node_texts = []
  edges = []
  all_tree_nodes.append(root)
  
  node_texts.append("")
  curr_path_node = root
  next_path_node = None
  for chunk in doc.noun_chunks:
    #print(f'chunk.start: {chunk.start}, chunk.end: {chunk.end}')
    #print(f'chunk.text: {chunk.text}, root: {chunk.root.text}, root.dep_: {chunk.root.dep_}, root.head: {chunk.root.head.text},  root.head.dep_: {chunk.root.head.dep_}\n')
    #TODO: f'a photo of {chunk.text}'
    #sub_sentences.append(chunk.text)
    #add child node under curr_node
    true_label_path.append(node_index)
    edges.append([curr_path_node.node_num, node_index])
    next_path_node = Node(node_num=node_index, prompt=chunk.text, parent=curr_path_node)
    node_texts.append(get_box_text(chunk.text))
    all_tree_nodes.append(next_path_node)
    
    curr_path_node.children[node_index] = next_path_node
    node_index += 1

    new_noun_phrases = get_new_noun_phrases(chunk)
    for new_text in new_noun_phrases:
      tmp_node = Node(node_num=node_index, prompt=new_text, parent=curr_path_node)
      node_texts.append(get_box_text(new_text))
      all_tree_nodes.append(tmp_node)
      curr_path_node.children[node_index] = tmp_node
      edges.append([curr_path_node.node_num, node_index])
      node_index += 1

    curr_path_node = next_path_node
    #if  chunk.root.dep_ == 'nsubj':
    #  print(chunk.root.head.text,  chunk.root.head.dep_, chunk.root.head.pos_)
    #  print(f'chunk.root.head.lefts: {[token.text for token in chunk.root.head.lefts]}')
    #  print(f'chunk.root.head.rights: {[token.text for token in chunk.root.head.rights]}')
  
  #sub_sentences.append(original_sentence)
  true_label_path.append(node_index)
  next_path_node = Node(node_num=node_index, prompt=original_caption, parent=curr_path_node)
  node_texts.append(get_box_text(original_caption))
  all_tree_nodes.append(next_path_node)
  curr_path_node.children[node_index] = next_path_node
  edges.append([curr_path_node.node_num, node_index])
  node_index += 1
  
  new_sentences = get_new_verb_phrases(doc)
  for new_sentence in new_sentences:
      tmp_node = Node(node_num=node_index, prompt=new_sentence, parent=curr_path_node)
      #node_texts.append(new_sentence)
      node_texts.append(get_box_text(new_sentence))
      ###
      all_tree_nodes.append(tmp_node)
      curr_path_node.children[node_index] = tmp_node
      edges.append([curr_path_node.node_num, node_index])
      node_index += 1
  
  #######################
  #######################
  #plotly_plot_tree(all_tree_nodes, edges, original_caption, node_texts)
 
  return root, true_label_path, all_tree_nodes, edges ,node_texts

def get_caption_tree2(original_caption):
  #print(f'\noriginal_sentence: {original_caption}\n')
  original_caption = original_caption.translate(str.maketrans('', '', string.punctuation))
  original_caption = original_caption.lower()
  doc = nlp(original_caption)
  #sub_sentences = []

  #for token in doc:
  #  print(token.text, token.dep_, token.head.text, token.pos_)

  #class Node():
  #  def __init__(self, node_num=0, prompt=None, prompt_encoding=None):

  node_index = 0
  root = Node(node_num=node_index, prompt='entity')
  node_index += 1
  true_label_path = []
  all_tree_nodes = []
  node_texts = []
  edges = []
  all_tree_nodes.append(root)
  
  prev_text = ""
  add_prev_text = False
  node_texts.append("")
  next_path_node = root
  curr_path_node = None
  
  last_chunk_end_index = 0
  #print(f'original_caption : {original_caption}')
  for chunk in doc.noun_chunks:
    #print(f'chunk.text: {chunk.text}, root: {chunk.root.text}, root.dep_: {chunk.root.dep_}, root.head: {chunk.root.head.text},  root.head.dep_: {chunk.root.head.dep_}\n')
    #TODO: f'a photo of {chunk.text}'
    #sub_sentences.append(chunk.text)
    #add child node under curr_node
    curr_path_node = next_path_node
    true_label_path.append(node_index)
    edges.append([curr_path_node.node_num, node_index])
    last_chunk_end_index = chunk.end
    #chunk_last_idx = chunk[-1].i
    #text = get_box_text(doc[0:chunk_last_idx].text)
    if chunk.start != 0:
      prev_text = doc[0:chunk.start].text + " "
      #print(f'\nprev_text : {prev_text}, chunk.text : {chunk.text}')

    text = prev_text + chunk.text
    #print(f'chunk.start : {chunk.start}, chunk.end : {chunk.end}, text : {text}')
    next_path_node = Node(node_num=node_index, prompt=text, parent=curr_path_node)
    
    
    node_texts.append(get_box_text(text))
    all_tree_nodes.append(next_path_node)
    
    curr_path_node.children[node_index] = next_path_node
    node_index += 1

    
    #TODO: replace added adpositions (Prepositions and postpositions e.g. (in, under, towards, before), (of, for).)
    #TODO: replace all words that don't exist in previous node's text
    new_noun_phrases = get_new_noun_phrases(chunk)
    for mod_text in new_noun_phrases:
      new_text = prev_text + mod_text
      #print(f'\nmod_text : {mod_text}, \nnew_text : {new_text}')
      tmp_node = Node(node_num=node_index, prompt=new_text, parent=curr_path_node)
      node_texts.append(get_box_text(new_text))
      all_tree_nodes.append(tmp_node)
      curr_path_node.children[node_index] = tmp_node
      edges.append([curr_path_node.node_num, node_index])
      node_index += 1

  
  #print(f'\nlast_chunk_end_index: {last_chunk_end_index}, len(doc): {len(doc)}')

  #add original sentence only if not added before with last noun_chunk
  if last_chunk_end_index != len(doc):
    #print(f'last_chunk_end_index != len(doc)')
    curr_path_node = next_path_node
    true_label_path.append(node_index)
    next_path_node = Node(node_num=node_index, prompt=original_caption, parent=curr_path_node)
    node_texts.append(get_box_text(original_caption))
    all_tree_nodes.append(next_path_node)
    curr_path_node.children[node_index] = next_path_node
    edges.append([curr_path_node.node_num, node_index])
    node_index += 1
  
  new_sentences = get_new_verb_phrases(doc)
  for new_sentence in new_sentences:
      tmp_node = Node(node_num=node_index, prompt=new_sentence, parent=curr_path_node)
      #node_texts.append(new_sentence)
      node_texts.append(get_box_text(new_sentence))
      ###
      all_tree_nodes.append(tmp_node)
      curr_path_node.children[node_index] = tmp_node
      edges.append([curr_path_node.node_num, node_index])
      node_index += 1
  
  #######################
  #######################
  #plotly_plot_tree(all_tree_nodes, edges, original_caption, node_texts)
 
  return root, true_label_path, all_tree_nodes, edges ,node_texts


def get_caption_tree3(original_caption):
  #print(f'\noriginal_sentence: {original_caption}\n')
  original_caption = original_caption.translate(str.maketrans('', '', string.punctuation))
  original_caption = original_caption.lower()
  doc = nlp(original_caption)
  #sub_sentences = []

  #for token in doc:
  #  print(token.text, token.dep_, token.head.text, token.pos_)

  #class Node():
  #  def __init__(self, node_num=0, prompt=None, prompt_encoding=None):

  node_index = 0
  root = Node(node_num=node_index, prompt='entity')
  node_index += 1
  true_label_path = []
  all_tree_nodes = []
  node_texts = []
  edges = []
  all_tree_nodes.append(root)
  
  prev_text = ""
  add_prev_text = False
  node_texts.append("")
  next_path_node = root
  curr_path_node = None
  curr_doc = None
  prev_chunk_end_index = -1
  
  last_chunk_end_index = 0
  #print(f'original_caption : {original_caption}')
  for chunk in doc.noun_chunks:
    
    curr_doc = chunk
    curr_path_node = next_path_node
    true_label_path.append(node_index)
    edges.append([curr_path_node.node_num, node_index])
    last_chunk_end_index = chunk.end
    #chunk_last_idx = chunk[-1].i
    #text = get_box_text(doc[0:chunk_last_idx].text)
    if chunk.start != 0:
      #TODO: change to doc[0:prev_chunk_end_index].text + " and " ??
      prev_text = doc[0:chunk.start].text + " "
      if prev_chunk_end_index > -1:
        #TODO: change to doc[0:prev_chunk_end_index].text + " and " ??
        prev_text = doc[0:prev_chunk_end_index].text + " "
        curr_doc = doc[prev_chunk_end_index:chunk.end]
      #print(f'\nprev_text : {prev_text}, chunk.text : {chunk.text}')

    text = doc[0:chunk.end].text
    print(f'chunk.start : {chunk.start}, chunk.end : {chunk.end}, prev_chunk_end_index: {prev_chunk_end_index}, curr_doc.text : {curr_doc.text}, prev_text: {prev_text}')
    next_path_node = Node(node_num=node_index, prompt=text, parent=curr_path_node)
    
    
    node_texts.append(get_box_text(text))
    all_tree_nodes.append(next_path_node)
    
    curr_path_node.children[node_index] = next_path_node
    node_index += 1

    
    #TODO: replace added adpositions (Prepositions and postpositions e.g. (in, under, towards, before), (of, for).)
    #TODO: replace all words that don't exist in previous node's text
    new_noun_phrases = get_new_noun_phrases(curr_doc)
    for mod_text in new_noun_phrases:
      new_text = prev_text + mod_text
      #print(f'\nmod_text : {mod_text}, \nnew_text : {new_text}')
      tmp_node = Node(node_num=node_index, prompt=new_text, parent=curr_path_node)
      node_texts.append(get_box_text(new_text))
      all_tree_nodes.append(tmp_node)
      curr_path_node.children[node_index] = tmp_node
      edges.append([curr_path_node.node_num, node_index])
      node_index += 1

    prev_chunk_end_index = chunk.end

  
  #print(f'\nlast_chunk_end_index: {last_chunk_end_index}, len(doc): {len(doc)}')

  #add original sentence only if not added before with last noun_chunk
  if last_chunk_end_index != len(doc):
    #print(f'last_chunk_end_index != len(doc)')
    curr_path_node = next_path_node
    true_label_path.append(node_index)
    next_path_node = Node(node_num=node_index, prompt=original_caption, parent=curr_path_node)
    node_texts.append(get_box_text(original_caption))
    all_tree_nodes.append(next_path_node)
    curr_path_node.children[node_index] = next_path_node
    edges.append([curr_path_node.node_num, node_index])
    node_index += 1
  
  new_sentences = get_new_verb_phrases(doc)
  for new_sentence in new_sentences:
      tmp_node = Node(node_num=node_index, prompt=new_sentence, parent=curr_path_node)
      #node_texts.append(new_sentence)
      node_texts.append(get_box_text(new_sentence))
      ###
      all_tree_nodes.append(tmp_node)
      curr_path_node.children[node_index] = tmp_node
      edges.append([curr_path_node.node_num, node_index])
      node_index += 1
  
  #######################
  #######################
  #plotly_plot_tree(all_tree_nodes, edges, original_caption, node_texts)
 
  return root, true_label_path, all_tree_nodes, edges ,node_texts


def create_path_node(curr_index, paernt_node, text, true_label_path, edges, node_texts, all_tree_nodes):
  
  true_label_path.append(curr_index)

  return create_node(curr_index, paernt_node, text, edges, node_texts, all_tree_nodes)


def create_node(curr_index, paernt_node, text, edges, node_texts, all_tree_nodes):
  
  edges.append([paernt_node.node_num, curr_index])
  next_node = Node(node_num=curr_index, prompt=text, parent=paernt_node)

  node_texts.append(get_box_text(text))
  all_tree_nodes.append(next_node)
    
  paernt_node.children[curr_index] = next_node

  return next_node

def get_caption_tree4(original_caption):
  #print(f'\noriginal_sentence: {original_caption}\n')
  original_caption = original_caption.translate(str.maketrans('', '', string.punctuation))
  original_caption = original_caption.lower()
  doc = nlp(original_caption)
 
  node_index = 0
  root = Node(node_num=node_index, prompt='entity')
 
  node_index += 1
  true_label_path = []
  all_tree_nodes = []
  node_texts = []
  edges = []
  all_tree_nodes.append(root)
  
  prev_text = ""
  add_prev_text = False
  node_texts.append("")
  next_path_node = root
  prev_path_node = root
  #curr_path_node = None
  curr_doc = None
  prev_chunk_end_index = -1
  prev_noun_chunk = None
  
  last_chunk_end_index = 0
  #print(f'original_caption : {original_caption}')
  for chunk in doc.noun_chunks:
    
    text_to_modify_lst = get_text_to_modify(prev_noun_chunk, chunk, doc)
    prev_noun_chunk = chunk
    for (doc_mod, prefix, suffix) in text_to_modify_lst:
      
      prev_path_node = next_path_node
      #print(f'doc_mod.start : {doc_mod.start}, doc_mod.end : {doc_mod.end}, prefix: {prefix}, doc_mod.text: {doc_mod.text}, suffix: {suffix}')

      #curr_doc = chunk
      #prev_path_node = next_path_node
    
      #last_chunk_end_index = chunk.end
    
      #if chunk.start != 0:
        #TODO: change to doc[0:prev_chunk_end_index].text + " and " ??
        #prev_text = doc[0:chunk.start].text + " "
        #if prev_chunk_end_index > -1:
          #TODO: change to doc[0:prev_chunk_end_index].text + " and " ??
          #prev_text = doc[0:prev_chunk_end_index].text + " "
          #curr_doc = doc[prev_chunk_end_index:chunk.end]
        #print(f'\nprev_text : {prev_text}, chunk.text : {chunk.text}')

      #text = doc[0:chunk.end].text
      #print(f'chunk.start : {chunk.start}, chunk.end : {chunk.end}, prev_chunk_end_index: {prev_chunk_end_index}, curr_doc.text : {curr_doc.text}, prev_text: {prev_text}')

      text = prefix + doc_mod.text + suffix
      #print(text)
      next_path_node = create_path_node(node_index, prev_path_node, text, true_label_path, edges, node_texts, all_tree_nodes)
      node_index += 1

    
      new_noun_phrases = get_new_noun_phrases(doc_mod)
      for mod_text in new_noun_phrases:
        new_text = prefix + mod_text + suffix
      
        tmp_node = create_node(node_index, prev_path_node, new_text, edges, node_texts, all_tree_nodes)
        node_index += 1
      

      #prev_chunk_end_index = chunk.end

  
  #add original sentence only if not added before with last noun_chunk
  if prev_noun_chunk.end != len(doc):
    
    prev_path_node = next_path_node
    
    next_path_node = create_path_node(node_index, prev_path_node, original_caption, true_label_path, edges, node_texts, all_tree_nodes)
    node_index += 1
  
    #new_sentences = get_new_verb_phrases(doc)
    new_sentences = get_new_noun_phrases(doc[prev_noun_chunk.end:])
    for new_sentence in new_sentences:
      text = doc[0:prev_noun_chunk.end].text + " " + new_sentence
      tmp_node = create_node(node_index, prev_path_node, text, edges, node_texts, all_tree_nodes)
      node_index += 1
  
 
  return root, true_label_path, all_tree_nodes, edges ,node_texts




def get_caption_tree5(original_caption):
  #print(f'\noriginal_sentence: {original_caption}\n')
  original_caption = original_caption.translate(str.maketrans('', '', string.punctuation))
  original_caption = original_caption.lower()
  doc = nlp(original_caption)
 
  node_index = 0
  root = Node(node_num=node_index, prompt='entity')
 
  node_index += 1
  true_label_path = []
  all_tree_nodes = []
  node_texts = []
  edges = []
  all_tree_nodes.append(root)
  
  prev_text = ""
  add_prev_text = False
  node_texts.append("")
  next_path_node = root
  prev_path_node = root
  #curr_path_node = None
  curr_doc = None
  prev_chunk_end_index = -1
  prev_noun_chunk = None
  
  last_chunk_end_index = 0
  #print(f'original_caption : {original_caption}')
  for chunk in doc.noun_chunks:
    
    text_to_modify_lst = get_text_to_modify(prev_noun_chunk, chunk, doc)
    prev_noun_chunk = chunk
    for (doc_mod, prefix, suffix) in text_to_modify_lst:
      
      prev_path_node = next_path_node
      
      text = prefix + doc_mod.text + suffix
      #print(text)
      next_path_node = create_path_node(node_index, prev_path_node, text, true_label_path, edges, node_texts, all_tree_nodes)
      node_index += 1
   
      #new_noun_phrases = get_new_noun_phrases(doc_mod)
      new_noun_phrases = get_t5_new_sentences(doc_mod, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
      #get_t5_new_sentences(orig_doc, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
      for mod_text in new_noun_phrases:
        new_text = prefix + mod_text + suffix
      
        tmp_node = create_node(node_index, prev_path_node, new_text, edges, node_texts, all_tree_nodes)
        node_index += 1
       
  #add original sentence only if not added before with last noun_chunk
  if prev_noun_chunk.end != len(doc):
    
    prev_path_node = next_path_node
    
    next_path_node = create_path_node(node_index, prev_path_node, original_caption, true_label_path, edges, node_texts, all_tree_nodes)
    node_index += 1
  
    #new_sentences = get_new_verb_phrases(doc)
    #new_sentences = get_new_noun_phrases(doc[prev_noun_chunk.end:])
    new_sentences = get_t5_new_sentences(doc[prev_noun_chunk.end:], pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
    for new_sentence in new_sentences:
      text = doc[0:prev_noun_chunk.end].text + " " + new_sentence
      tmp_node = create_node(node_index, prev_path_node, text, edges, node_texts, all_tree_nodes)
      node_index += 1
  
 
  return root, true_label_path, all_tree_nodes, edges ,node_texts



def get_caption_tree6_shuffle_nouns_all_branches(original_caption):
  #print(f'\noriginal_sentence: {original_caption}\n')
  original_caption = original_caption.translate(str.maketrans('', '', string.punctuation))
  original_caption = original_caption.lower()
  doc = nlp(original_caption)
 
  node_index = 0
  root = Node(node_num=node_index, prompt='entity')
 
  node_index += 1
  true_label_path = []
  all_tree_nodes = []
  node_texts = []
  edges = []
  all_tree_nodes.append(root)
  
  prev_text = ""
  add_prev_text = False
  node_texts.append("")
  next_path_node = root
  prev_path_node = root
  #curr_path_node = None
  curr_doc = None
  prev_chunk_end_index = -1
  prev_noun_chunk = None
  #print(f'get_caption_tree6: original_caption: {original_caption}')
  last_chunk_end_index = 0
  #print(f'original_caption : {original_caption}')
  for chunk in doc.noun_chunks:
    #print(f'get_caption_tree6: chunk.text: {chunk.text}')
    text_to_modify_lst = get_text_to_modify(prev_noun_chunk, chunk, doc)
    prev_noun_chunk = chunk
    for (doc_mod, prefix, suffix) in text_to_modify_lst:
      #print(f'get_caption_tree6: doc_mod.text: {doc_mod.text}')
      prev_path_node = next_path_node
      
      text = prefix + doc_mod.text + suffix
      #print(text)
      next_path_node = create_path_node(node_index, prev_path_node, text, true_label_path, edges, node_texts, all_tree_nodes)
      node_index += 1
   
      #new_noun_phrases = get_new_noun_phrases(doc_mod)
      new_noun_phrases = get_t5_new_sentences2(doc_mod, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
      #print(f'new_noun_phrases: {new_noun_phrases}')
      #get_t5_new_sentences(orig_doc, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
      for mod_text in new_noun_phrases:
        new_text = prefix + mod_text + suffix
      
        tmp_node = create_node(node_index, prev_path_node, new_text, edges, node_texts, all_tree_nodes)
        node_index += 1
      
      ###########################
      #add reordering each branch
      ############################
      
      new_sentences =  get_shuffled_nouns_adjectives_sentences(text)
      if new_sentences:

        prev_path_node = next_path_node
        next_path_node = create_path_node(node_index, prev_path_node, text, true_label_path, edges, node_texts, all_tree_nodes)
        node_index += 1


        for new_text in new_sentences:
        
          tmp_node = create_node(node_index, prev_path_node, new_text, edges, node_texts, all_tree_nodes)
          node_index += 1

      ###########################
      #add reordering after each branch
      ############################
       
  #add original sentence only if not added before with last noun_chunk
  if prev_noun_chunk and prev_noun_chunk.end != len(doc):
    
    prev_path_node = next_path_node
    
    next_path_node = create_path_node(node_index, prev_path_node, original_caption, true_label_path, edges, node_texts, all_tree_nodes)
    node_index += 1
  
    #new_sentences = get_new_verb_phrases(doc)
    #new_sentences = get_new_noun_phrases(doc[prev_noun_chunk.end:])
    new_sentences = get_t5_new_sentences2(doc[prev_noun_chunk.end:], pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
    #shuffled_sentence = shuffle(original_caption)
    #new_sentences.aapend(shuffled_sentence)
    #print(f'new_sentences: {new_sentences}')
    for new_sentence in new_sentences:
      text = doc[0:prev_noun_chunk.end].text + " " + new_sentence
      #print(f'text: {text}')
      tmp_node = create_node(node_index, prev_path_node, text, edges, node_texts, all_tree_nodes)
      node_index += 1
  

  new_sentences =  get_shuffled_nouns_adjectives_sentences(original_caption)
  if new_sentences:

    prev_path_node = next_path_node
    next_path_node = create_path_node(node_index, prev_path_node, original_caption, true_label_path, edges, node_texts, all_tree_nodes)
    node_index += 1


    for text in new_sentences:
        #text = doc[0:prev_noun_chunk.end].text + " " + new_sentence
        #print(f'text: {text}')
        tmp_node = create_node(node_index, prev_path_node, text, edges, node_texts, all_tree_nodes)
        node_index += 1
  
  return root, true_label_path, all_tree_nodes, edges ,node_texts


def get_caption_tree6_with_all_shuffles(original_caption):
  #print(f'\noriginal_sentence: {original_caption}\n')
  original_caption = original_caption.translate(str.maketrans('', '', string.punctuation))
  original_caption = original_caption.lower()
  doc = nlp(original_caption)
 
  node_index = 0
  root = Node(node_num=node_index, prompt='entity')
 
  node_index += 1
  true_label_path = []
  all_tree_nodes = []
  node_texts = []
  edges = []
  all_tree_nodes.append(root)
  
  prev_text = ""
  add_prev_text = False
  node_texts.append("")
  next_path_node = root
  prev_path_node = root
  #curr_path_node = None
  curr_doc = None
  prev_chunk_end_index = -1
  prev_noun_chunk = None
  #print(f'get_caption_tree6: original_caption: {original_caption}')
  last_chunk_end_index = 0
  #print(f'original_caption : {original_caption}')
  for chunk in doc.noun_chunks:
    #print(f'get_caption_tree6: chunk.text: {chunk.text}')
    text_to_modify_lst = get_text_to_modify(prev_noun_chunk, chunk, doc)
    prev_noun_chunk = chunk
    for (doc_mod, prefix, suffix) in text_to_modify_lst:
      #print(f'get_caption_tree6: doc_mod.text: {doc_mod.text}')
      prev_path_node = next_path_node
      
      text = prefix + doc_mod.text + suffix
      #print(text)
      next_path_node = create_path_node(node_index, prev_path_node, text, true_label_path, edges, node_texts, all_tree_nodes)
      node_index += 1
   
      #new_noun_phrases = get_new_noun_phrases(doc_mod)
      new_noun_phrases = get_t5_new_sentences2(doc_mod, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
      #print(f'new_noun_phrases: {new_noun_phrases}')
      #get_t5_new_sentences(orig_doc, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
      for mod_text in new_noun_phrases:
        new_text = prefix + mod_text + suffix
      
        tmp_node = create_node(node_index, prev_path_node, new_text, edges, node_texts, all_tree_nodes)
        node_index += 1
      
      ######################################
      #add all shuffles reordering 
      #######################################
      new_sentences =  get_all_shuffles_sentences(text)

      for new_text in new_sentences:
         tmp_node = create_node(node_index, prev_path_node, new_text, edges, node_texts, all_tree_nodes)
         node_index += 1

            
  #add original sentence only if not added before with last noun_chunk
  if prev_noun_chunk and prev_noun_chunk.end != len(doc):
    
    prev_path_node = next_path_node
    
    next_path_node = create_path_node(node_index, prev_path_node, original_caption, true_label_path, edges, node_texts, all_tree_nodes)
    node_index += 1
  
    #new_sentences = get_new_verb_phrases(doc)
    #new_sentences = get_new_noun_phrases(doc[prev_noun_chunk.end:])
    new_sentences = get_t5_new_sentences2(doc[prev_noun_chunk.end:], pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
    #shuffled_sentence = shuffle(original_caption)
    #new_sentences.aapend(shuffled_sentence)
    #print(f'new_sentences: {new_sentences}')
    for new_sentence in new_sentences:
      text = doc[0:prev_noun_chunk.end].text + " " + new_sentence
      #print(f'text: {text}')
      tmp_node = create_node(node_index, prev_path_node, text, edges, node_texts, all_tree_nodes)
      node_index += 1
    
    random_reorder = get_all_shuffles_sentences(original_caption)
    for text in random_reorder:
        #text = doc[0:prev_noun_chunk.end].text + " " + new_sentence
        #print(f'text: {text}')
        tmp_node = create_node(node_index, prev_path_node, text, edges, node_texts, all_tree_nodes)
        node_index += 1
  
 
  return root, true_label_path, all_tree_nodes, edges ,node_texts

def get_caption_tree6_with_noun_adj_shuffle(original_caption):
  #print(f'\noriginal_sentence: {original_caption}\n')
  original_caption = original_caption.translate(str.maketrans('', '', string.punctuation))
  original_caption = original_caption.lower()
  doc = nlp(original_caption)
 
  node_index = 0
  root = Node(node_num=node_index, prompt='entity')
 
  node_index += 1
  true_label_path = []
  all_tree_nodes = []
  node_texts = []
  edges = []
  all_tree_nodes.append(root)
  
  prev_text = ""
  add_prev_text = False
  node_texts.append("")
  next_path_node = root
  prev_path_node = root
  #curr_path_node = None
  curr_doc = None
  prev_chunk_end_index = -1
  prev_noun_chunk = None
  #print(f'get_caption_tree6: original_caption: {original_caption}')
  last_chunk_end_index = 0
  #print(f'original_caption : {original_caption}')
  for chunk in doc.noun_chunks:
    #print(f'get_caption_tree6: chunk.text: {chunk.text}')
    text_to_modify_lst = get_text_to_modify(prev_noun_chunk, chunk, doc)
    prev_noun_chunk = chunk
    for (doc_mod, prefix, suffix) in text_to_modify_lst:
      #print(f'get_caption_tree6: doc_mod.text: {doc_mod.text}')
      prev_path_node = next_path_node
      
      text = prefix + doc_mod.text + suffix
      #print(text)
      next_path_node = create_path_node(node_index, prev_path_node, text, true_label_path, edges, node_texts, all_tree_nodes)
      node_index += 1
   
      #new_noun_phrases = get_new_noun_phrases(doc_mod)
      new_noun_phrases = get_t5_new_sentences2(doc_mod, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
      #print(f'new_noun_phrases: {new_noun_phrases}')
      #get_t5_new_sentences(orig_doc, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
      for mod_text in new_noun_phrases:
        new_text = prefix + mod_text + suffix
      
        tmp_node = create_node(node_index, prev_path_node, new_text, edges, node_texts, all_tree_nodes)
        node_index += 1
      
      ######################################
      #add nouns and adjectives reordering 
      #######################################
      new_sentences =  get_shuffled_nouns_adjectives_sentences(text)

      for new_text in new_sentences:
         tmp_node = create_node(node_index, prev_path_node, new_text, edges, node_texts, all_tree_nodes)
         node_index += 1

            
  #add original sentence only if not added before with last noun_chunk
  if prev_noun_chunk and prev_noun_chunk.end != len(doc):
    
    prev_path_node = next_path_node
    
    next_path_node = create_path_node(node_index, prev_path_node, original_caption, true_label_path, edges, node_texts, all_tree_nodes)
    node_index += 1
  
    #new_sentences = get_new_verb_phrases(doc)
    #new_sentences = get_new_noun_phrases(doc[prev_noun_chunk.end:])
    new_sentences = get_t5_new_sentences2(doc[prev_noun_chunk.end:], pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
    #shuffled_sentence = shuffle(original_caption)
    #new_sentences.aapend(shuffled_sentence)
    #print(f'new_sentences: {new_sentences}')
    for new_sentence in new_sentences:
      text = doc[0:prev_noun_chunk.end].text + " " + new_sentence
      #print(f'text: {text}')
      tmp_node = create_node(node_index, prev_path_node, text, edges, node_texts, all_tree_nodes)
      node_index += 1
    
    random_reorder = get_shuffled_nouns_adjectives_sentences(original_caption)
    for text in random_reorder:
        #text = doc[0:prev_noun_chunk.end].text + " " + new_sentence
        #print(f'text: {text}')
        tmp_node = create_node(node_index, prev_path_node, text, edges, node_texts, all_tree_nodes)
        node_index += 1
  
 
  return root, true_label_path, all_tree_nodes, edges ,node_texts

def get_caption_tree6_with_random_shuffle(original_caption):
  #print(f'\noriginal_sentence: {original_caption}\n')
  original_caption = original_caption.translate(str.maketrans('', '', string.punctuation))
  original_caption = original_caption.lower()
  doc = nlp(original_caption)
 
  node_index = 0
  root = Node(node_num=node_index, prompt='entity')
 
  node_index += 1
  true_label_path = []
  all_tree_nodes = []
  node_texts = []
  edges = []
  all_tree_nodes.append(root)
  
  prev_text = ""
  add_prev_text = False
  node_texts.append("")
  next_path_node = root
  prev_path_node = root
  #curr_path_node = None
  curr_doc = None
  prev_chunk_end_index = -1
  prev_noun_chunk = None
  #print(f'get_caption_tree6: original_caption: {original_caption}')
  last_chunk_end_index = 0
  #print(f'original_caption : {original_caption}')
  for chunk in doc.noun_chunks:
    #print(f'get_caption_tree6: chunk.text: {chunk.text}')
    text_to_modify_lst = get_text_to_modify(prev_noun_chunk, chunk, doc)
    prev_noun_chunk = chunk
    for (doc_mod, prefix, suffix) in text_to_modify_lst:
      #print(f'get_caption_tree6: doc_mod.text: {doc_mod.text}')
      prev_path_node = next_path_node
      
      text = prefix + doc_mod.text + suffix
      #print(text)
      next_path_node = create_path_node(node_index, prev_path_node, text, true_label_path, edges, node_texts, all_tree_nodes)
      node_index += 1
   
      #new_noun_phrases = get_new_noun_phrases(doc_mod)
      new_noun_phrases = get_t5_new_sentences2(doc_mod, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
      #print(f'new_noun_phrases: {new_noun_phrases}')
      #get_t5_new_sentences(orig_doc, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
      for mod_text in new_noun_phrases:
        new_text = prefix + mod_text + suffix
      
        tmp_node = create_node(node_index, prev_path_node, new_text, edges, node_texts, all_tree_nodes)
        node_index += 1
      
      ###########################
      #add random reordering 
      ############################
      new_sentences =  get_all_shuffles_random_sentence(text)

      for new_text in new_sentences:
         tmp_node = create_node(node_index, prev_path_node, new_text, edges, node_texts, all_tree_nodes)
         node_index += 1

            
  #add original sentence only if not added before with last noun_chunk
  if prev_noun_chunk and prev_noun_chunk.end != len(doc):
    
    prev_path_node = next_path_node
    
    next_path_node = create_path_node(node_index, prev_path_node, original_caption, true_label_path, edges, node_texts, all_tree_nodes)
    node_index += 1
  
    #new_sentences = get_new_verb_phrases(doc)
    #new_sentences = get_new_noun_phrases(doc[prev_noun_chunk.end:])
    new_sentences = get_t5_new_sentences2(doc[prev_noun_chunk.end:], pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
    #shuffled_sentence = shuffle(original_caption)
    #new_sentences.aapend(shuffled_sentence)
    #print(f'new_sentences: {new_sentences}')
    for new_sentence in new_sentences:
      text = doc[0:prev_noun_chunk.end].text + " " + new_sentence
      #print(f'text: {text}')
      tmp_node = create_node(node_index, prev_path_node, text, edges, node_texts, all_tree_nodes)
      node_index += 1
    
    random_reorder = get_all_shuffles_random_sentence(original_caption)
    for text in random_reorder:
        #text = doc[0:prev_noun_chunk.end].text + " " + new_sentence
        #print(f'text: {text}')
        tmp_node = create_node(node_index, prev_path_node, text, edges, node_texts, all_tree_nodes)
        node_index += 1
  
 
  return root, true_label_path, all_tree_nodes, edges ,node_texts


def get_caption_tree6_shuffle_random_all_branches(original_caption):
  #print(f'\noriginal_sentence: {original_caption}\n')
  original_caption = original_caption.translate(str.maketrans('', '', string.punctuation))
  original_caption = original_caption.lower()
  doc = nlp(original_caption)
 
  node_index = 0
  root = Node(node_num=node_index, prompt='entity')
 
  node_index += 1
  true_label_path = []
  all_tree_nodes = []
  node_texts = []
  edges = []
  all_tree_nodes.append(root)
  
  prev_text = ""
  add_prev_text = False
  node_texts.append("")
  next_path_node = root
  prev_path_node = root
  #curr_path_node = None
  curr_doc = None
  prev_chunk_end_index = -1
  prev_noun_chunk = None
  #print(f'get_caption_tree6: original_caption: {original_caption}')
  last_chunk_end_index = 0
  #print(f'original_caption : {original_caption}')
  for chunk in doc.noun_chunks:
    #print(f'get_caption_tree6: chunk.text: {chunk.text}')
    text_to_modify_lst = get_text_to_modify(prev_noun_chunk, chunk, doc)
    prev_noun_chunk = chunk
    for (doc_mod, prefix, suffix) in text_to_modify_lst:
      #print(f'get_caption_tree6: doc_mod.text: {doc_mod.text}')
      prev_path_node = next_path_node
      
      text = prefix + doc_mod.text + suffix
      #print(text)
      next_path_node = create_path_node(node_index, prev_path_node, text, true_label_path, edges, node_texts, all_tree_nodes)
      node_index += 1
   
      #new_noun_phrases = get_new_noun_phrases(doc_mod)
      new_noun_phrases = get_t5_new_sentences2(doc_mod, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
      #print(f'new_noun_phrases: {new_noun_phrases}')
      #get_t5_new_sentences(orig_doc, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
      for mod_text in new_noun_phrases:
        new_text = prefix + mod_text + suffix
      
        tmp_node = create_node(node_index, prev_path_node, new_text, edges, node_texts, all_tree_nodes)
        node_index += 1
      
      ###########################
      #add reordering each branch
      ############################
      
      new_sentences =  get_all_shuffles_random_sentence(text)
      if new_sentences:

        prev_path_node = next_path_node
        next_path_node = create_path_node(node_index, prev_path_node, text, true_label_path, edges, node_texts, all_tree_nodes)
        node_index += 1


        for new_text in new_sentences:
        
          tmp_node = create_node(node_index, prev_path_node, new_text, edges, node_texts, all_tree_nodes)
          node_index += 1

      ###########################
      #add reordering after each branch
      ############################
       
  #add original sentence only if not added before with last noun_chunk
  if prev_noun_chunk and prev_noun_chunk.end != len(doc):
    
    prev_path_node = next_path_node
    
    next_path_node = create_path_node(node_index, prev_path_node, original_caption, true_label_path, edges, node_texts, all_tree_nodes)
    node_index += 1
  
    #new_sentences = get_new_verb_phrases(doc)
    #new_sentences = get_new_noun_phrases(doc[prev_noun_chunk.end:])
    new_sentences = get_t5_new_sentences2(doc[prev_noun_chunk.end:], pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
    #shuffled_sentence = shuffle(original_caption)
    #new_sentences.aapend(shuffled_sentence)
    #print(f'new_sentences: {new_sentences}')
    for new_sentence in new_sentences:
      text = doc[0:prev_noun_chunk.end].text + " " + new_sentence
      #print(f'text: {text}')
      tmp_node = create_node(node_index, prev_path_node, text, edges, node_texts, all_tree_nodes)
      node_index += 1
  

  new_sentences =  get_all_shuffles_random_sentence(original_caption)
  if new_sentences:

    prev_path_node = next_path_node
    next_path_node = create_path_node(node_index, prev_path_node, original_caption, true_label_path, edges, node_texts, all_tree_nodes)
    node_index += 1


    for text in new_sentences:
        #text = doc[0:prev_noun_chunk.end].text + " " + new_sentence
        #print(f'text: {text}')
        tmp_node = create_node(node_index, prev_path_node, text, edges, node_texts, all_tree_nodes)
        node_index += 1
  
  return root, true_label_path, all_tree_nodes, edges ,node_texts

def get_caption_tree6_shuffle_all_branches(original_caption):
  #print(f'\noriginal_sentence: {original_caption}\n')
  original_caption = original_caption.translate(str.maketrans('', '', string.punctuation))
  original_caption = original_caption.lower()
  doc = nlp(original_caption)
 
  node_index = 0
  root = Node(node_num=node_index, prompt='entity')
 
  node_index += 1
  true_label_path = []
  all_tree_nodes = []
  node_texts = []
  edges = []
  all_tree_nodes.append(root)
  
  prev_text = ""
  add_prev_text = False
  node_texts.append("")
  next_path_node = root
  prev_path_node = root
  #curr_path_node = None
  curr_doc = None
  prev_chunk_end_index = -1
  prev_noun_chunk = None
  #print(f'get_caption_tree6: original_caption: {original_caption}')
  last_chunk_end_index = 0
  #print(f'original_caption : {original_caption}')
  for chunk in doc.noun_chunks:
    #print(f'get_caption_tree6: chunk.text: {chunk.text}')
    text_to_modify_lst = get_text_to_modify(prev_noun_chunk, chunk, doc)
    prev_noun_chunk = chunk
    for (doc_mod, prefix, suffix) in text_to_modify_lst:
      #print(f'get_caption_tree6: doc_mod.text: {doc_mod.text}')
      prev_path_node = next_path_node
      
      text = prefix + doc_mod.text + suffix
      #print(text)
      next_path_node = create_path_node(node_index, prev_path_node, text, true_label_path, edges, node_texts, all_tree_nodes)
      node_index += 1
   
      #new_noun_phrases = get_new_noun_phrases(doc_mod)
      new_noun_phrases = get_t5_new_sentences2(doc_mod, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
      #print(f'new_noun_phrases: {new_noun_phrases}')
      #get_t5_new_sentences(orig_doc, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
      for mod_text in new_noun_phrases:
        new_text = prefix + mod_text + suffix
      
        tmp_node = create_node(node_index, prev_path_node, new_text, edges, node_texts, all_tree_nodes)
        node_index += 1
      
      ###########################
      #add reordering after each branch
      ############################
      
      new_sentences =  get_all_shuffles_sentences(text)
      if new_sentences:

        prev_path_node = next_path_node
        next_path_node = create_path_node(node_index, prev_path_node, text, true_label_path, edges, node_texts, all_tree_nodes)
        node_index += 1


        for new_text in new_sentences:
        
          tmp_node = create_node(node_index, prev_path_node, new_text, edges, node_texts, all_tree_nodes)
          node_index += 1

      ###########################
      #add reordering after each branch
      ############################
       
  #add original sentence only if not added before with last noun_chunk
  if prev_noun_chunk and prev_noun_chunk.end != len(doc):
    
    prev_path_node = next_path_node
    
    next_path_node = create_path_node(node_index, prev_path_node, original_caption, true_label_path, edges, node_texts, all_tree_nodes)
    node_index += 1
  
    #new_sentences = get_new_verb_phrases(doc)
    #new_sentences = get_new_noun_phrases(doc[prev_noun_chunk.end:])
    new_sentences = get_t5_new_sentences2(doc[prev_noun_chunk.end:], pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
    #shuffled_sentence = shuffle(original_caption)
    #new_sentences.aapend(shuffled_sentence)
    #print(f'new_sentences: {new_sentences}')
    for new_sentence in new_sentences:
      text = doc[0:prev_noun_chunk.end].text + " " + new_sentence
      #print(f'text: {text}')
      tmp_node = create_node(node_index, prev_path_node, text, edges, node_texts, all_tree_nodes)
      node_index += 1
  

  new_sentences =  get_all_shuffles_sentences(original_caption)
  if new_sentences:

    prev_path_node = next_path_node
    next_path_node = create_path_node(node_index, prev_path_node, original_caption, true_label_path, edges, node_texts, all_tree_nodes)
    node_index += 1


    for text in new_sentences:
        #text = doc[0:prev_noun_chunk.end].text + " " + new_sentence
        #print(f'text: {text}')
        tmp_node = create_node(node_index, prev_path_node, text, edges, node_texts, all_tree_nodes)
        node_index += 1
  
  return root, true_label_path, all_tree_nodes, edges ,node_texts


def get_caption_tree6_narratives_shuffle_all_branches(original_caption):
  #print(f'\noriginal_sentence: {original_caption}\n')
  original_caption = original_caption.translate(str.maketrans('', '', string.punctuation))
  original_caption = original_caption.lower()
  doc = nlp(original_caption)
 
  node_index = 0
  root = Node(node_num=node_index, prompt='entity')
 
  node_index += 1
  true_label_path = []
  all_tree_nodes = []
  node_texts = []
  edges = []
  all_tree_nodes.append(root)
  
  prev_text = ""
  add_prev_text = False
  node_texts.append("")
  next_path_node = root
  prev_path_node = root
  #curr_path_node = None
  curr_doc = None
  prev_chunk_end_index = -1
  prev_noun_chunk = None
  #print(f'get_caption_tree6: original_caption: {original_caption}')
  last_chunk_end_index = 0
  #print(f'original_caption : {original_caption}')
  for chunk in doc.noun_chunks:
    #print(f'get_caption_tree6: chunk.text: {chunk.text}')
    text_to_modify_lst = get_narratives_text_to_modify(prev_noun_chunk, chunk, doc)
    prev_noun_chunk = chunk
    for (doc_mod, prefix, suffix) in text_to_modify_lst:
      #print(f'get_caption_tree6: doc_mod.text: {doc_mod.text}')
      prev_path_node = next_path_node
      
      text = prefix + doc_mod.text + suffix
      #print(text)
      next_path_node = create_path_node(node_index, prev_path_node, text, true_label_path, edges, node_texts, all_tree_nodes)
      node_index += 1
   
      #new_noun_phrases = get_new_noun_phrases(doc_mod)
      new_noun_phrases = get_t5_new_sentences2(doc_mod, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
      #print(f'new_noun_phrases: {new_noun_phrases}')
      #get_t5_new_sentences(orig_doc, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
      for mod_text in new_noun_phrases:
        new_text = prefix + mod_text + suffix
      
        tmp_node = create_node(node_index, prev_path_node, new_text, edges, node_texts, all_tree_nodes)
        node_index += 1
      
      ###########################
      #add reordering after each branch
      ############################
      
      new_sentences =  get_all_shuffles_sentences(text)
      if new_sentences:

        prev_path_node = next_path_node
        next_path_node = create_path_node(node_index, prev_path_node, text, true_label_path, edges, node_texts, all_tree_nodes)
        node_index += 1


        for new_text in new_sentences:
        
          tmp_node = create_node(node_index, prev_path_node, new_text, edges, node_texts, all_tree_nodes)
          node_index += 1

      ###########################
      #add reordering after each branch
      ############################
       
  #add original sentence only if not added before with last noun_chunk
  if prev_noun_chunk and prev_noun_chunk.end != len(doc):
    
    prev_path_node = next_path_node
    
    next_path_node = create_path_node(node_index, prev_path_node, original_caption, true_label_path, edges, node_texts, all_tree_nodes)
    node_index += 1
  
    #new_sentences = get_new_verb_phrases(doc)
    #new_sentences = get_new_noun_phrases(doc[prev_noun_chunk.end:])
    new_sentences = get_t5_new_sentences2(doc[prev_noun_chunk.end:], pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
    #shuffled_sentence = shuffle(original_caption)
    #new_sentences.aapend(shuffled_sentence)
    #print(f'new_sentences: {new_sentences}')
    for new_sentence in new_sentences:
      text = doc[0:prev_noun_chunk.end].text + " " + new_sentence
      #print(f'text: {text}')
      tmp_node = create_node(node_index, prev_path_node, text, edges, node_texts, all_tree_nodes)
      node_index += 1
  

  new_sentences =  get_all_shuffles_sentences(original_caption)
  if new_sentences:

    prev_path_node = next_path_node
    next_path_node = create_path_node(node_index, prev_path_node, original_caption, true_label_path, edges, node_texts, all_tree_nodes)
    node_index += 1


    for text in new_sentences:
        #text = doc[0:prev_noun_chunk.end].text + " " + new_sentence
        #print(f'text: {text}')
        tmp_node = create_node(node_index, prev_path_node, text, edges, node_texts, all_tree_nodes)
        node_index += 1
  
  return root, true_label_path, all_tree_nodes, edges ,node_texts


def get_caption_tree6_shuffle_all(original_caption):
  #print(f'\noriginal_sentence: {original_caption}\n')
  original_caption = original_caption.translate(str.maketrans('', '', string.punctuation))
  original_caption = original_caption.lower()
  doc = nlp(original_caption)
 
  node_index = 0
  root = Node(node_num=node_index, prompt='entity')
 
  node_index += 1
  true_label_path = []
  all_tree_nodes = []
  node_texts = []
  edges = []
  all_tree_nodes.append(root)
  
  prev_text = ""
  add_prev_text = False
  node_texts.append("")
  next_path_node = root
  prev_path_node = root
  #curr_path_node = None
  curr_doc = None
  prev_chunk_end_index = -1
  prev_noun_chunk = None
  #print(f'get_caption_tree6: original_caption: {original_caption}')
  last_chunk_end_index = 0
  #print(f'original_caption : {original_caption}')
  for chunk in doc.noun_chunks:
    #print(f'get_caption_tree6: chunk.text: {chunk.text}')
    text_to_modify_lst = get_text_to_modify(prev_noun_chunk, chunk, doc)
    prev_noun_chunk = chunk
    for (doc_mod, prefix, suffix) in text_to_modify_lst:
      #print(f'get_caption_tree6: doc_mod.text: {doc_mod.text}')
      prev_path_node = next_path_node
      
      text = prefix + doc_mod.text + suffix
      #print(text)
      next_path_node = create_path_node(node_index, prev_path_node, text, true_label_path, edges, node_texts, all_tree_nodes)
      node_index += 1
   
      #new_noun_phrases = get_new_noun_phrases(doc_mod)
      new_noun_phrases = get_t5_new_sentences2(doc_mod, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
      #print(f'new_noun_phrases: {new_noun_phrases}')
      #get_t5_new_sentences(orig_doc, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
      for mod_text in new_noun_phrases:
        new_text = prefix + mod_text + suffix
      
        tmp_node = create_node(node_index, prev_path_node, new_text, edges, node_texts, all_tree_nodes)
        node_index += 1
       
  #add original sentence only if not added before with last noun_chunk
  if prev_noun_chunk and prev_noun_chunk.end != len(doc):
    
    prev_path_node = next_path_node
    
    next_path_node = create_path_node(node_index, prev_path_node, original_caption, true_label_path, edges, node_texts, all_tree_nodes)
    node_index += 1
  
    #new_sentences = get_new_verb_phrases(doc)
    #new_sentences = get_new_noun_phrases(doc[prev_noun_chunk.end:])
    new_sentences = get_t5_new_sentences2(doc[prev_noun_chunk.end:], pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
    #shuffled_sentence = shuffle(original_caption)
    #new_sentences.aapend(shuffled_sentence)
    #print(f'new_sentences: {new_sentences}')
    for new_sentence in new_sentences:
      text = doc[0:prev_noun_chunk.end].text + " " + new_sentence
      #print(f'text: {text}')
      tmp_node = create_node(node_index, prev_path_node, text, edges, node_texts, all_tree_nodes)
      node_index += 1
  

  new_sentences =  get_all_shuffles_sentences(original_caption)
  if new_sentences:

    prev_path_node = next_path_node
    next_path_node = create_path_node(node_index, prev_path_node, original_caption, true_label_path, edges, node_texts, all_tree_nodes)
    node_index += 1


    for text in new_sentences:
        #text = doc[0:prev_noun_chunk.end].text + " " + new_sentence
        #print(f'text: {text}')
        tmp_node = create_node(node_index, prev_path_node, text, edges, node_texts, all_tree_nodes)
        node_index += 1
  
  return root, true_label_path, all_tree_nodes, edges ,node_texts

def get_caption_tree6_shuffled_nouns_adjectives(original_caption):
  #print(f'\noriginal_sentence: {original_caption}\n')
  original_caption = original_caption.translate(str.maketrans('', '', string.punctuation))
  original_caption = original_caption.lower()
  doc = nlp(original_caption)
 
  node_index = 0
  root = Node(node_num=node_index, prompt='entity')
 
  node_index += 1
  true_label_path = []
  all_tree_nodes = []
  node_texts = []
  edges = []
  all_tree_nodes.append(root)
  
  prev_text = ""
  add_prev_text = False
  node_texts.append("")
  next_path_node = root
  prev_path_node = root
  #curr_path_node = None
  curr_doc = None
  prev_chunk_end_index = -1
  prev_noun_chunk = None
  #print(f'get_caption_tree6: original_caption: {original_caption}')
  last_chunk_end_index = 0
  #print(f'original_caption : {original_caption}')
  for chunk in doc.noun_chunks:
    #print(f'get_caption_tree6: chunk.text: {chunk.text}')
    text_to_modify_lst = get_text_to_modify(prev_noun_chunk, chunk, doc)
    prev_noun_chunk = chunk
    for (doc_mod, prefix, suffix) in text_to_modify_lst:
      #print(f'get_caption_tree6: doc_mod.text: {doc_mod.text}')
      prev_path_node = next_path_node
      
      text = prefix + doc_mod.text + suffix
      #print(text)
      next_path_node = create_path_node(node_index, prev_path_node, text, true_label_path, edges, node_texts, all_tree_nodes)
      node_index += 1
   
      #new_noun_phrases = get_new_noun_phrases(doc_mod)
      new_noun_phrases = get_t5_new_sentences2(doc_mod, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
      #print(f'new_noun_phrases: {new_noun_phrases}')
      #get_t5_new_sentences(orig_doc, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
      for mod_text in new_noun_phrases:
        new_text = prefix + mod_text + suffix
      
        tmp_node = create_node(node_index, prev_path_node, new_text, edges, node_texts, all_tree_nodes)
        node_index += 1
       
  #add original sentence only if not added before with last noun_chunk
  if prev_noun_chunk and prev_noun_chunk.end != len(doc):
    
    prev_path_node = next_path_node
    
    next_path_node = create_path_node(node_index, prev_path_node, original_caption, true_label_path, edges, node_texts, all_tree_nodes)
    node_index += 1
  
    #new_sentences = get_new_verb_phrases(doc)
    #new_sentences = get_new_noun_phrases(doc[prev_noun_chunk.end:])
    new_sentences = get_t5_new_sentences2(doc[prev_noun_chunk.end:], pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
    #shuffled_sentence = shuffle(original_caption)
    #new_sentences.aapend(shuffled_sentence)
    #print(f'new_sentences: {new_sentences}')
    for new_sentence in new_sentences:
      text = doc[0:prev_noun_chunk.end].text + " " + new_sentence
      #print(f'text: {text}')
      tmp_node = create_node(node_index, prev_path_node, text, edges, node_texts, all_tree_nodes)
      node_index += 1
  

  new_sentences =  get_shuffled_nouns_adjectives_sentences(original_caption)
  if new_sentences:

    prev_path_node = next_path_node
    next_path_node = create_path_node(node_index, prev_path_node, original_caption, true_label_path, edges, node_texts, all_tree_nodes)
    node_index += 1


    for text in new_sentences:
        #text = doc[0:prev_noun_chunk.end].text + " " + new_sentence
        #print(f'text: {text}')
        tmp_node = create_node(node_index, prev_path_node, text, edges, node_texts, all_tree_nodes)
        node_index += 1
  
  return root, true_label_path, all_tree_nodes, edges ,node_texts


def get_caption_tree6_narratives(original_caption):
  #print(f'\noriginal_sentence: {original_caption}\n')
  # original_caption = original_caption.translate(str.maketrans('', '', string.punctuation))
  original_caption = original_caption.lower()
  
 
  node_index = 0
  root = Node(node_num=node_index, prompt='entity')
 
  node_index += 1
  true_label_path = []
  all_tree_nodes = []
  node_texts = []
  edges = []
  all_tree_nodes.append(root)
  
  prev_text = ""
  add_prev_text = False
  node_texts.append("")
  next_path_node = root
  prev_path_node = root
  #curr_path_node = None

  caption_sentences = original_caption.split('.')
  for capt in caption_sentences:
    capt = capt.translate(str.maketrans('', '', string.punctuation))

    doc = nlp(capt)
    curr_doc = None
    prev_chunk_end_index = -1
    prev_noun_chunk = None
    #print(f'get_caption_tree6_narratives: capt: {capt}')
    last_chunk_end_index = 0
    #print(f'capt : {capt}')
    for chunk in doc.noun_chunks:
      #print(f'get_caption_tree6: chunk.text: {chunk.text}')
      text_to_modify_lst = get_narratives_text_to_modify(prev_noun_chunk, chunk, doc)
      prev_noun_chunk = chunk
      for (doc_mod, prefix, suffix) in text_to_modify_lst:
        #print(f'get_caption_tree6: doc_mod.text: {doc_mod.text}')
        prev_path_node = next_path_node
        
        text = prefix + doc_mod.text + suffix
        #print(text)
        next_path_node = create_path_node(node_index, prev_path_node, text, true_label_path, edges, node_texts, all_tree_nodes)
        node_index += 1
    
        #new_noun_phrases = get_new_noun_phrases(doc_mod)
        new_noun_phrases = get_t5_new_sentences2(doc_mod, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
        #print(f'new_noun_phrases: {new_noun_phrases}')
        #get_t5_new_sentences(orig_doc, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
        for mod_text in new_noun_phrases:
          new_text = prefix + mod_text + suffix
        
          tmp_node = create_node(node_index, prev_path_node, new_text, edges, node_texts, all_tree_nodes)
          node_index += 1
       
    #add original sentence only if not added before with last noun_chunk
    if prev_noun_chunk and prev_noun_chunk.end != len(doc):
      
      prev_path_node = next_path_node
      
      next_path_node = create_path_node(node_index, prev_path_node, capt, true_label_path, edges, node_texts, all_tree_nodes)
      node_index += 1
    
      #new_sentences = get_new_verb_phrases(doc)
      #new_sentences = get_new_noun_phrases(doc[prev_noun_chunk.end:])
      new_sentences = get_t5_new_sentences2(doc[prev_noun_chunk.end:], pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
      #shuffled_sentence = shuffle(original_caption)
      #new_sentences.aapend(shuffled_sentence)
      #print(f'new_sentences: {new_sentences}')
      for new_sentence in new_sentences:
        text = doc[0:prev_noun_chunk.end].text + " " + new_sentence
        #print(f'text: {text}')
        tmp_node = create_node(node_index, prev_path_node, text, edges, node_texts, all_tree_nodes)
        node_index += 1
  
 
  return root, true_label_path, all_tree_nodes, edges ,node_texts

def get_caption_tree6(original_caption):
  #print(f'\noriginal_sentence: {original_caption}\n')
  original_caption = original_caption.translate(str.maketrans('', '', string.punctuation))
  original_caption = original_caption.lower()
  doc = nlp(original_caption)
 
  node_index = 0
  root = Node(node_num=node_index, prompt='entity')
 
  node_index += 1
  true_label_path = []
  all_tree_nodes = []
  node_texts = []
  edges = []
  all_tree_nodes.append(root)
  
  prev_text = ""
  add_prev_text = False
  node_texts.append("")
  next_path_node = root
  prev_path_node = root
  #curr_path_node = None
  curr_doc = None
  prev_chunk_end_index = -1
  prev_noun_chunk = None
  #print(f'get_caption_tree6: original_caption: {original_caption}')
  last_chunk_end_index = 0
  #print(f'original_caption : {original_caption}')
  for chunk in doc.noun_chunks:
    #print(f'get_caption_tree6: chunk.text: {chunk.text}')
    text_to_modify_lst = get_text_to_modify(prev_noun_chunk, chunk, doc)
    prev_noun_chunk = chunk
    for (doc_mod, prefix, suffix) in text_to_modify_lst:
      #print(f'get_caption_tree6: doc_mod.text: {doc_mod.text}')
      prev_path_node = next_path_node
      
      text = prefix + doc_mod.text + suffix
      #print(text)
      next_path_node = create_path_node(node_index, prev_path_node, text, true_label_path, edges, node_texts, all_tree_nodes)
      node_index += 1
   
      #new_noun_phrases = get_new_noun_phrases(doc_mod)
      new_noun_phrases = get_t5_new_sentences2(doc_mod, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
      #print(f'new_noun_phrases: {new_noun_phrases}')
      #get_t5_new_sentences(orig_doc, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
      for mod_text in new_noun_phrases:
        new_text = prefix + mod_text + suffix
      
        tmp_node = create_node(node_index, prev_path_node, new_text, edges, node_texts, all_tree_nodes)
        node_index += 1
       
  #add original sentence only if not added before with last noun_chunk
  if prev_noun_chunk and prev_noun_chunk.end != len(doc):
    
    prev_path_node = next_path_node
    
    next_path_node = create_path_node(node_index, prev_path_node, original_caption, true_label_path, edges, node_texts, all_tree_nodes)
    node_index += 1
  
    #new_sentences = get_new_verb_phrases(doc)
    #new_sentences = get_new_noun_phrases(doc[prev_noun_chunk.end:])
    new_sentences = get_t5_new_sentences2(doc[prev_noun_chunk.end:], pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
    #shuffled_sentence = shuffle(original_caption)
    #new_sentences.aapend(shuffled_sentence)
    #print(f'new_sentences: {new_sentences}')
    for new_sentence in new_sentences:
      text = doc[0:prev_noun_chunk.end].text + " " + new_sentence
      #print(f'text: {text}')
      tmp_node = create_node(node_index, prev_path_node, text, edges, node_texts, all_tree_nodes)
      node_index += 1
  
 
  return root, true_label_path, all_tree_nodes, edges ,node_texts


def get_caption_tree6_max_neg(original_caption, max_neg=math.inf):
  #print(f'\noriginal_sentence: {original_caption}\n')
  original_caption = original_caption.translate(str.maketrans('', '', string.punctuation))
  original_caption = original_caption.lower()
  doc = nlp(original_caption)
 
  node_index = 0
  num_neg = 0
  root = Node(node_num=node_index, prompt='entity')
 
  node_index += 1
  true_label_path = []
  all_tree_nodes = []
  node_texts = []
  edges = []
  all_tree_nodes.append(root)
  
  prev_text = ""
  add_prev_text = False
  node_texts.append("")
  next_path_node = root
  prev_path_node = root
  #curr_path_node = None
  curr_doc = None
  prev_chunk_end_index = -1
  prev_noun_chunk = None
  #print(f'get_caption_tree6: original_caption: {original_caption}')
  last_chunk_end_index = 0

  pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP']

  #print(f'original_caption : {original_caption}')
  for chunk in doc.noun_chunks:
    #print(f'get_caption_tree6: chunk.text: {chunk.text}')
    text_to_modify_lst = get_text_to_modify(prev_noun_chunk, chunk, doc)
    prev_noun_chunk = chunk
    for (doc_mod, prefix, suffix) in text_to_modify_lst:
      
      if num_neg >= max_neg:
        return root, true_label_path, all_tree_nodes, edges ,node_texts

      #print(f'get_caption_tree6: doc_mod.text: {doc_mod.text}')
      prev_path_node = next_path_node
      
      text = prefix + doc_mod.text + suffix
      #print(text)
      next_path_node = create_path_node(node_index, prev_path_node, text, true_label_path, edges, node_texts, all_tree_nodes)
      node_index += 1
   
      #new_noun_phrases = get_new_noun_phrases(doc_mod)
      
      new_noun_phrases = get_t5_new_sentences2(doc_mod, pos_to_replace, num_neg, max_neg)
      num_neg += len(new_noun_phrases)
      #print(f'new_noun_phrases: {new_noun_phrases}')
      #get_t5_new_sentences(orig_doc, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
      for mod_text in new_noun_phrases:
        new_text = prefix + mod_text + suffix
      
        tmp_node = create_node(node_index, prev_path_node, new_text, edges, node_texts, all_tree_nodes)
        node_index += 1
       
  #add original sentence only if not added before with last noun_chunk
  if prev_noun_chunk and num_neg < max_neg and prev_noun_chunk.end != len(doc):
    
    prev_path_node = next_path_node
    
    next_path_node = create_path_node(node_index, prev_path_node, original_caption, true_label_path, edges, node_texts, all_tree_nodes)
    node_index += 1
  
    #new_sentences = get_new_verb_phrases(doc)
    #new_sentences = get_new_noun_phrases(doc[prev_noun_chunk.end:])
    new_sentences = get_t5_new_sentences2(doc[prev_noun_chunk.end:], pos_to_replace, num_neg, max_neg)
    #shuffled_sentence = shuffle(original_caption)
    #new_sentences.aapend(shuffled_sentence)
    #print(f'new_sentences: {new_sentences}')
    for new_sentence in new_sentences:
      text = doc[0:prev_noun_chunk.end].text + " " + new_sentence
      #print(f'text: {text}')
      tmp_node = create_node(node_index, prev_path_node, text, edges, node_texts, all_tree_nodes)
      node_index += 1
  
 
  return root, true_label_path, all_tree_nodes, edges ,node_texts


def get_caption_tree6_spatial_negate(original_caption):
  #print(f'\noriginal_sentence: {original_caption}\n')
  original_caption = original_caption.translate(str.maketrans('', '', string.punctuation))
  original_caption = original_caption.lower()
  doc = nlp(original_caption)
 
  node_index = 0
  root = Node(node_num=node_index, prompt='entity')
 
  node_index += 1
  true_label_path = []
  all_tree_nodes = []
  node_texts = []
  edges = []
  all_tree_nodes.append(root)
  
  prev_text = ""
  add_prev_text = False
  node_texts.append("")
  next_path_node = root
  prev_path_node = root
  #curr_path_node = None
  curr_doc = None
  prev_chunk_end_index = -1
  prev_noun_chunk = None
  #print(f'get_caption_tree6: original_caption: {original_caption}')
  last_chunk_end_index = 0
  #print(f'original_caption : {original_caption}')
  for chunk in doc.noun_chunks:
    #print(f'get_caption_tree6: chunk.text: {chunk.text}')
    text_to_modify_lst = get_text_to_modify(prev_noun_chunk, chunk, doc)
    prev_noun_chunk = chunk
    for (doc_mod, prefix, suffix) in text_to_modify_lst:
      #print(f'get_caption_tree6: doc_mod.text: {doc_mod.text}')
      prev_path_node = next_path_node
      
      text = prefix + doc_mod.text + suffix
      #print(text)
      next_path_node = create_path_node(node_index, prev_path_node, text, true_label_path, edges, node_texts, all_tree_nodes)
      node_index += 1
   
      #new_noun_phrases = get_new_noun_phrases(doc_mod)
      new_noun_phrases = get_t5_new_sentences2_spatial_negate(doc_mod, pos_to_replace=['NOUN', 'ADJ', 'VERB'])
      #print(f'new_noun_phrases: {new_noun_phrases}')
      #get_t5_new_sentences(orig_doc, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
      for mod_text in new_noun_phrases:
        new_text = prefix + mod_text + suffix
      
        tmp_node = create_node(node_index, prev_path_node, new_text, edges, node_texts, all_tree_nodes)
        node_index += 1
       
  #add original sentence only if not added before with last noun_chunk
  if prev_noun_chunk and prev_noun_chunk.end != len(doc):
    
    prev_path_node = next_path_node
    
    next_path_node = create_path_node(node_index, prev_path_node, original_caption, true_label_path, edges, node_texts, all_tree_nodes)
    node_index += 1
  
    #new_sentences = get_new_verb_phrases(doc)
    #new_sentences = get_new_noun_phrases(doc[prev_noun_chunk.end:])
    new_sentences = get_t5_new_sentences2_spatial_negate(doc[prev_noun_chunk.end:], pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
    #shuffled_sentence = shuffle(original_caption)
    #new_sentences.aapend(shuffled_sentence)
    #print(f'new_sentences: {new_sentences}')
    for new_sentence in new_sentences:
      text = doc[0:prev_noun_chunk.end].text + " " + new_sentence
      #print(f'text: {text}')
      tmp_node = create_node(node_index, prev_path_node, text, edges, node_texts, all_tree_nodes)
      node_index += 1
  
 
  return root, true_label_path, all_tree_nodes, edges ,node_texts



def get_caption_tree6_gen_si(original_caption):
  #print(f'\noriginal_sentence: {original_caption}\n')
  original_caption = original_caption.translate(str.maketrans('', '', string.punctuation))
  original_caption = original_caption.lower()
  doc = nlp(original_caption)
 
  node_index = 0
  root = Node(node_num=node_index, prompt='entity')
 
  node_index += 1
  true_label_path = []
  all_tree_nodes = []
  node_texts = []
  edges = []
  all_tree_nodes.append(root)
  
  prev_text = ""
  add_prev_text = False
  node_texts.append("")
  next_path_node = root
  prev_path_node = root
  #curr_path_node = None
  curr_doc = None
  prev_chunk_end_index = -1
  prev_noun_chunk = None
  #print(f'get_caption_tree6: original_caption: {original_caption}')
  last_chunk_end_index = 0
  #print(f'original_caption : {original_caption}')
  for chunk in doc.noun_chunks:
    #print(f'get_caption_tree6: chunk.text: {chunk.text}')
    text_to_modify_lst = get_text_to_modify(prev_noun_chunk, chunk, doc)
    prev_noun_chunk = chunk
    for (doc_mod, prefix, suffix) in text_to_modify_lst:
      #print(f'get_caption_tree6: doc_mod.text: {doc_mod.text}')
      prev_path_node = next_path_node
      
      text = prefix + doc_mod.text + suffix
      #print(text)
      next_path_node = create_path_node(node_index, prev_path_node, text, true_label_path, edges, node_texts, all_tree_nodes)
      node_index += 1
   
      #new_noun_phrases = get_new_noun_phrases(doc_mod)
      # new_noun_phrases = get_t5_new_sentences2(doc_mod, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
      new_noun_phrases, cf_tag = get_cfs_for_pos_statement(doc_mod.text)
      #print(f'new_noun_phrases: {new_noun_phrases}')
      #get_t5_new_sentences(orig_doc, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
      for mod_text in new_noun_phrases:
        new_text = prefix + mod_text + suffix
      
        tmp_node = create_node(node_index, prev_path_node, new_text, edges, node_texts, all_tree_nodes)
        node_index += 1
       
  #add original sentence only if not added before with last noun_chunk
  if prev_noun_chunk and prev_noun_chunk.end != len(doc):
    
    prev_path_node = next_path_node
    
    next_path_node = create_path_node(node_index, prev_path_node, original_caption, true_label_path, edges, node_texts, all_tree_nodes)
    node_index += 1
  
    #new_sentences = get_new_verb_phrases(doc)
    #new_sentences = get_new_noun_phrases(doc[prev_noun_chunk.end:])
    # new_sentences = get_t5_new_sentences2(doc[prev_noun_chunk.end:], pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
    new_sentences, cf_tag = get_cfs_for_pos_statement(doc[prev_noun_chunk.end:].text)
    #shuffled_sentence = shuffle(original_caption)
    #new_sentences.aapend(shuffled_sentence)
    #print(f'new_sentences: {new_sentences}')
    for new_sentence in new_sentences:
      text = doc[0:prev_noun_chunk.end].text + " " + new_sentence
      #print(f'text: {text}')
      tmp_node = create_node(node_index, prev_path_node, text, edges, node_texts, all_tree_nodes)
      node_index += 1
  
 
  return root, true_label_path, all_tree_nodes, edges ,node_texts


def get_caption_tree6_depth1(original_caption):
  #print(f'\noriginal_sentence: {original_caption}\n')
  original_caption = original_caption.translate(str.maketrans('', '', string.punctuation))
  original_caption = original_caption.lower()
  doc = nlp(original_caption)
 
  node_index = 0
  root = Node(node_num=node_index, prompt='entity')
 
  node_index += 1
  true_label_path = []
  all_tree_nodes = []
  node_texts = []
  edges = []
  all_tree_nodes.append(root)
  
  prev_text = ""
  add_prev_text = False
  node_texts.append("")
  next_path_node = root
  prev_path_node = root
  #curr_path_node = None
  curr_doc = None
  prev_chunk_end_index = -1
  prev_noun_chunk = None
  #print(f'get_caption_tree6: original_caption: {original_caption}')
  last_chunk_end_index = 0
  #print(f'original_caption : {original_caption}')
    
  prev_path_node = next_path_node
  
  next_path_node = create_path_node(node_index, prev_path_node, original_caption, true_label_path, edges, node_texts, all_tree_nodes)
  node_index += 1

  #new_sentences = get_new_verb_phrases(doc)
  #new_sentences = get_new_noun_phrases(doc[prev_noun_chunk.end:])
  new_sentences = get_t5_new_sentences2(doc, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
  #shuffled_sentence = shuffle(original_caption)
  #new_sentences.aapend(shuffled_sentence)
  #print(f'new_sentences: {new_sentences}')
  for new_sentence in new_sentences:
    text = new_sentence
    #print(f'text: {text}')
    tmp_node = create_node(node_index, prev_path_node, text, edges, node_texts, all_tree_nodes)
    node_index += 1
  
 
  return root, true_label_path, all_tree_nodes, edges ,node_texts


def get_caption_tree6_max_depth(original_caption, max_depth=1):
  #print(f'\noriginal_sentence: {original_caption}\n')
  original_caption = original_caption.translate(str.maketrans('', '', string.punctuation))
  original_caption = original_caption.lower()
  doc = nlp(original_caption)

  node_index = 0
  root = Node(node_num=node_index, prompt='entity')

  node_index += 1
  true_label_path = []
  all_tree_nodes = []
  node_texts = []
  edges = []
  all_tree_nodes.append(root)
  
  prev_text = ""
  add_prev_text = False
  node_texts.append("")
  next_path_node = root
  prev_path_node = root
  #curr_path_node = None
  curr_doc = None
  prev_chunk_end_index = -1
  prev_noun_chunk = None
  #print(f'get_caption_tree6: original_caption: {original_caption}')
  last_chunk_end_index = 0
  #print(f'original_caption : {original_caption}')
  max_internal_nodes = max_depth - 1
  num_internal_nodes = 0
  for chunk in doc.noun_chunks:
    if num_internal_nodes >= max_internal_nodes:
      # print(f'new chunk num_internal_nodes: {num_internal_nodes} >= max_internal_nodes: {max_internal_nodes}')
      break
    #print(f'get_caption_tree6: chunk.text: {chunk.text}')
    text_to_modify_lst = get_text_to_modify(prev_noun_chunk, chunk, doc)
    prev_noun_chunk = chunk
    for (doc_mod, prefix, suffix) in text_to_modify_lst:
      if num_internal_nodes >= max_internal_nodes:
        # print(f'new doc_mod num_internal_nodes: {num_internal_nodes} >= max_internal_nodes: {max_internal_nodes}')
        break
      #print(f'get_caption_tree6: doc_mod.text: {doc_mod.text}')
      prev_path_node = next_path_node
      
      text = prefix + doc_mod.text + suffix
      #print(text)
      if num_internal_nodes < max_internal_nodes:
        num_internal_nodes += 1
        next_path_node = create_path_node(node_index, prev_path_node, text, true_label_path, edges, node_texts, all_tree_nodes)
        node_index += 1
    
        #new_noun_phrases = get_new_noun_phrases(doc_mod)
        new_noun_phrases = get_t5_new_sentences2(doc_mod, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
        #print(f'new_noun_phrases: {new_noun_phrases}')
        #get_t5_new_sentences(orig_doc, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
        for mod_text in new_noun_phrases:
          new_text = prefix + mod_text + suffix
        
          tmp_node = create_node(node_index, prev_path_node, new_text, edges, node_texts, all_tree_nodes)
          node_index += 1
      
  #add original sentence only if not added before with last noun_chunk
  if prev_noun_chunk and prev_noun_chunk.end != len(doc):
    
    prev_path_node = next_path_node
    
    next_path_node = create_path_node(node_index, prev_path_node, original_caption, true_label_path, edges, node_texts, all_tree_nodes)
    node_index += 1
  
    #new_sentences = get_new_verb_phrases(doc)
    #new_sentences = get_new_noun_phrases(doc[prev_noun_chunk.end:])
    new_sentences = get_t5_new_sentences2(doc[prev_noun_chunk.end:], pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
    #shuffled_sentence = shuffle(original_caption)
    #new_sentences.aapend(shuffled_sentence)
    #print(f'new_sentences: {new_sentences}')
    for new_sentence in new_sentences:
      text = doc[0:prev_noun_chunk.end].text + " " + new_sentence
      #print(f'text: {text}')
      tmp_node = create_node(node_index, prev_path_node, text, edges, node_texts, all_tree_nodes)
      node_index += 1
  

  return root, true_label_path, all_tree_nodes, edges ,node_texts


def get_caption_tree6_depth2(original_caption):
  return get_caption_tree6_max_depth(original_caption, max_depth=2)


def get_caption_tree6_depth3(original_caption):
    return get_caption_tree6_max_depth(original_caption, max_depth=3)


def get_caption_tree6_lemmas(original_caption):
  #print(f'\noriginal_sentence: {original_caption}\n')
  original_caption = original_caption.translate(str.maketrans('', '', string.punctuation))
  original_caption = original_caption.lower()
  doc = nlp(original_caption)
 
  node_index = 0
  root = Node(node_num=node_index, prompt='entity')
 
  node_index += 1
  true_label_path = []
  all_tree_nodes = []
  node_texts = []
  edges = []
  all_tree_nodes.append(root)
  
  prev_text = ""
  add_prev_text = False
  node_texts.append("")
  next_path_node = root
  prev_path_node = root
  #curr_path_node = None
  curr_doc = None
  prev_chunk_end_index = -1
  prev_noun_chunk = None
  #print(f'get_caption_tree6: original_caption: {original_caption}')
  last_chunk_end_index = 0
  #print(f'original_caption : {original_caption}')
  for chunk in doc.noun_chunks:
    #print(f'get_caption_tree6: chunk.text: {chunk.text}')
    text_to_modify_lst = get_text_to_modify(prev_noun_chunk, chunk, doc)
    prev_noun_chunk = chunk
    for (doc_mod, prefix, suffix) in text_to_modify_lst:
      #print(f'get_caption_tree6: doc_mod.text: {doc_mod.text}')
      prev_path_node = next_path_node
      
      text = prefix + doc_mod.text + suffix
      #print(text)
      next_path_node = create_path_node(node_index, prev_path_node, text, true_label_path, edges, node_texts, all_tree_nodes)
      node_index += 1
   
      #new_noun_phrases = get_new_noun_phrases(doc_mod)
      new_noun_phrases = get_t5_new_sentences2_lemmas(doc_mod, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
      #print(f'new_noun_phrases: {new_noun_phrases}')
      #get_t5_new_sentences(orig_doc, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
      for mod_text in new_noun_phrases:
        new_text = prefix + mod_text + suffix
      
        tmp_node = create_node(node_index, prev_path_node, new_text, edges, node_texts, all_tree_nodes)
        node_index += 1
       
  #add original sentence only if not added before with last noun_chunk
  if prev_noun_chunk and prev_noun_chunk.end != len(doc):
    
    prev_path_node = next_path_node
    
    next_path_node = create_path_node(node_index, prev_path_node, original_caption, true_label_path, edges, node_texts, all_tree_nodes)
    node_index += 1
  
    #new_sentences = get_new_verb_phrases(doc)
    #new_sentences = get_new_noun_phrases(doc[prev_noun_chunk.end:])
    new_sentences = get_t5_new_sentences2_lemmas(doc[prev_noun_chunk.end:], pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
    #print(f'new_sentences: {new_sentences}')
    for new_sentence in new_sentences:
      text = doc[0:prev_noun_chunk.end].text + " " + new_sentence
      #print(f'text: {text}')
      tmp_node = create_node(node_index, prev_path_node, text, edges, node_texts, all_tree_nodes)
      node_index += 1
  
 
  return root, true_label_path, all_tree_nodes, edges ,node_texts


def get_caption_tree6_1(original_caption):
  #print(f'\noriginal_sentence: {original_caption}\n')
  original_caption = original_caption.translate(str.maketrans('', '', string.punctuation))
  original_caption = original_caption.lower()
  doc = nlp(original_caption)
 
  node_index = 0
  root = Node(node_num=node_index, prompt='entity')
 
  node_index += 1
  true_label_path = []
  all_tree_nodes = []
  node_texts = []
  edges = []
  all_tree_nodes.append(root)
  
  prev_text = ""
  add_prev_text = False
  node_texts.append("")
  next_path_node = root
  prev_path_node = root
  #curr_path_node = None
  curr_doc = None
  prev_chunk_end_index = -1
  prev_noun_chunk = None
  #print(f'get_caption_tree6: original_caption: {original_caption}')
  last_chunk_end_index = 0
  #print(f'original_caption : {original_caption}')
  for chunk in doc.noun_chunks:
    #print(f'get_caption_tree6: chunk.text: {chunk.text}')
    text_to_modify_lst = get_text_to_modify(prev_noun_chunk, chunk, doc)
    prev_noun_chunk = chunk
    for (doc_mod, prefix, suffix) in text_to_modify_lst:
      #print(f'get_caption_tree6: doc_mod.text: {doc_mod.text}')
      prev_path_node = next_path_node
      
      text = prefix + doc_mod.text + suffix
      #print(text)
      next_path_node = create_path_node(node_index, prev_path_node, text, true_label_path, edges, node_texts, all_tree_nodes)
      node_index += 1
   
      #new_noun_phrases = get_new_noun_phrases(doc_mod)
      new_noun_phrases = get_t5_new_sentences2_1(doc_mod, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
      #print(f'new_noun_phrases: {new_noun_phrases}')
      #get_t5_new_sentences(orig_doc, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
      for mod_text in new_noun_phrases:
        new_text = prefix + mod_text + suffix
      
        tmp_node = create_node(node_index, prev_path_node, new_text, edges, node_texts, all_tree_nodes)
        node_index += 1
       
  #add original sentence only if not added before with last noun_chunk
  if prev_noun_chunk and prev_noun_chunk.end != len(doc):
    
    prev_path_node = next_path_node
    
    next_path_node = create_path_node(node_index, prev_path_node, original_caption, true_label_path, edges, node_texts, all_tree_nodes)
    node_index += 1
  
    #new_sentences = get_new_verb_phrases(doc)
    #new_sentences = get_new_noun_phrases(doc[prev_noun_chunk.end:])
    new_sentences = get_t5_new_sentences2_1(doc[prev_noun_chunk.end:], pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
    #print(f'new_sentences: {new_sentences}')
    for new_sentence in new_sentences:
      text = doc[0:prev_noun_chunk.end].text + " " + new_sentence
      #print(f'text: {text}')
      tmp_node = create_node(node_index, prev_path_node, text, edges, node_texts, all_tree_nodes)
      node_index += 1
  
 
  return root, true_label_path, all_tree_nodes, edges ,node_texts



def get_caption_tree6_2(original_caption):
  #print(f'\noriginal_sentence: {original_caption}\n')
  original_caption = original_caption.translate(str.maketrans('', '', string.punctuation))
  original_caption = original_caption.lower()
  doc = nlp(original_caption)
 
  node_index = 0
  root = Node(node_num=node_index, prompt='entity')
 
  node_index += 1
  true_label_path = []
  all_tree_nodes = []
  node_texts = []
  edges = []
  all_tree_nodes.append(root)
  
  prev_text = ""
  add_prev_text = False
  node_texts.append("")
  next_path_node = root
  prev_path_node = root
  #curr_path_node = None
  curr_doc = None
  prev_chunk_end_index = -1
  prev_noun_chunk = None
  #print(f'get_caption_tree6: original_caption: {original_caption}')
  last_chunk_end_index = 0
  #print(f'original_caption : {original_caption}')
  for chunk in doc.noun_chunks:
    #print(f'get_caption_tree6: chunk.text: {chunk.text}')
    text_to_modify_lst = get_text_to_modify(prev_noun_chunk, chunk, doc)
    prev_noun_chunk = chunk
    for (doc_mod, prefix, suffix) in text_to_modify_lst:
      #print(f'get_caption_tree6: doc_mod.text: {doc_mod.text}')
      prev_path_node = next_path_node
      
      text = prefix + doc_mod.text + suffix
      #print(text)
      next_path_node = create_path_node(node_index, prev_path_node, text, true_label_path, edges, node_texts, all_tree_nodes)
      node_index += 1
   
      #new_noun_phrases = get_new_noun_phrases(doc_mod)
      new_noun_phrases = get_t5_new_sentences2_2(doc_mod, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
      #print(f'new_noun_phrases: {new_noun_phrases}')
      #get_t5_new_sentences(orig_doc, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
      for mod_text in new_noun_phrases:
        new_text = prefix + mod_text + suffix
      
        tmp_node = create_node(node_index, prev_path_node, new_text, edges, node_texts, all_tree_nodes)
        node_index += 1
       
  #add original sentence only if not added before with last noun_chunk
  if prev_noun_chunk and prev_noun_chunk.end != len(doc):
    
    prev_path_node = next_path_node
    
    next_path_node = create_path_node(node_index, prev_path_node, original_caption, true_label_path, edges, node_texts, all_tree_nodes)
    node_index += 1
  
    #new_sentences = get_new_verb_phrases(doc)
    #new_sentences = get_new_noun_phrases(doc[prev_noun_chunk.end:])
    new_sentences = get_t5_new_sentences2_2(doc[prev_noun_chunk.end:], pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
    #print(f'new_sentences: {new_sentences}')
    for new_sentence in new_sentences:
      text = doc[0:prev_noun_chunk.end].text + " " + new_sentence
      #print(f'text: {text}')
      tmp_node = create_node(node_index, prev_path_node, text, edges, node_texts, all_tree_nodes)
      node_index += 1
  
 
  return root, true_label_path, all_tree_nodes, edges ,node_texts


def get_caption_tree7(original_caption):
  #print(f'\noriginal_sentence: {original_caption}\n')
  original_caption = original_caption.translate(str.maketrans('', '', string.punctuation))
  original_caption = original_caption.lower()
  doc = nlp(original_caption)
 
  node_index = 0
  root = Node(node_num=node_index, prompt='entity')
 
  node_index += 1
  true_label_path = []
  all_tree_nodes = []
  node_texts = []
  edges = []
  all_tree_nodes.append(root)
  
  prev_text = ""
  add_prev_text = False
  node_texts.append("")
  next_path_node = root
  prev_path_node = root
  #curr_path_node = None
  curr_doc = None
  prev_chunk_end_index = -1
  prev_noun_chunk = None
  
  last_chunk_end_index = 0
  #print(f'original_caption : {original_caption}')
  for chunk in doc.noun_chunks:
    
    text_to_modify_lst = get_text_to_modify(prev_noun_chunk, chunk, doc)
    prev_noun_chunk = chunk
    for (doc_mod, prefix, suffix) in text_to_modify_lst:
      
      prev_path_node = next_path_node
      
      text = prefix + doc_mod.text + suffix
      #print(text)
      next_path_node = create_path_node(node_index, prev_path_node, text, true_label_path, edges, node_texts, all_tree_nodes)
      node_index += 1
   
      #new_noun_phrases = get_new_noun_phrases(doc_mod)
      new_noun_phrases = get_t5_new_sentences3(doc, doc_mod, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
      #get_t5_new_sentences(orig_doc, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
      for mod_text in new_noun_phrases:
        new_text = prefix + mod_text + suffix
      
        tmp_node = create_node(node_index, prev_path_node, new_text, edges, node_texts, all_tree_nodes)
        node_index += 1
       
  #add original sentence only if not added before with last noun_chunk
  if prev_noun_chunk.end != len(doc):
    
    prev_path_node = next_path_node
    
    next_path_node = create_path_node(node_index, prev_path_node, original_caption, true_label_path, edges, node_texts, all_tree_nodes)
    node_index += 1
  
    #new_sentences = get_new_verb_phrases(doc)
    #new_sentences = get_new_noun_phrases(doc[prev_noun_chunk.end:])
    new_sentences = get_t5_new_sentences3(doc, doc[prev_noun_chunk.end:], pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
    for new_sentence in new_sentences:
      text = doc[0:prev_noun_chunk.end].text + " " + new_sentence
      tmp_node = create_node(node_index, prev_path_node, text, edges, node_texts, all_tree_nodes)
      node_index += 1
  
 
  return root, true_label_path, all_tree_nodes, edges ,node_texts

def get_caption_tree8(original_caption):
  #print(f'\noriginal_sentence: {original_caption}\n')
  original_caption = original_caption.translate(str.maketrans('', '', string.punctuation))
  original_caption = original_caption.lower()
  doc = nlp(original_caption)
 
  node_index = 0
  root = Node(node_num=node_index, prompt='entity')
 
  node_index += 1
  true_label_path = []
  all_tree_nodes = []
  node_texts = []
  edges = []
  all_tree_nodes.append(root)
  
  prev_text = ""
  add_prev_text = False
  node_texts.append("")
  next_path_node = root
  prev_path_node = root
  #curr_path_node = None
  curr_doc = None
  prev_chunk_end_index = -1
  prev_noun_chunk = None
  
  last_chunk_end_index = 0
  #print(f'original_caption : {original_caption}')
  for chunk in doc.noun_chunks:
    
    text_to_modify_lst = get_text_to_modify(prev_noun_chunk, chunk, doc)
    prev_noun_chunk = chunk
    for (doc_mod, prefix, suffix) in text_to_modify_lst:
      
      prev_path_node = next_path_node
      
      text = prefix + doc_mod.text + suffix
      #print(text)
      next_path_node = create_path_node(node_index, prev_path_node, text, true_label_path, edges, node_texts, all_tree_nodes)
      node_index += 1
   
      #new_noun_phrases = get_new_noun_phrases(doc_mod)
      new_noun_phrases = get_t5_new_sentences4(doc, doc_mod, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
      #get_t5_new_sentences(orig_doc, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
      for mod_text in new_noun_phrases:
        new_text = prefix + mod_text + suffix
      
        tmp_node = create_node(node_index, prev_path_node, new_text, edges, node_texts, all_tree_nodes)
        node_index += 1
       
  #add original sentence only if not added before with last noun_chunk
  if prev_noun_chunk and prev_noun_chunk.end != len(doc):
    
    prev_path_node = next_path_node
    
    next_path_node = create_path_node(node_index, prev_path_node, original_caption, true_label_path, edges, node_texts, all_tree_nodes)
    node_index += 1
  
    #new_sentences = get_new_verb_phrases(doc)
    #new_sentences = get_new_noun_phrases(doc[prev_noun_chunk.end:])
    new_sentences = get_t5_new_sentences4(doc, doc[prev_noun_chunk.end:], pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
    for new_sentence in new_sentences:
      text = doc[0:prev_noun_chunk.end].text + " " + new_sentence
      tmp_node = create_node(node_index, prev_path_node, text, edges, node_texts, all_tree_nodes)
      node_index += 1
  
 
  return root, true_label_path, all_tree_nodes, edges ,node_texts


def get_caption_tree8_1(original_caption):
  #print(f'\noriginal_sentence: {original_caption}\n')
  original_caption = original_caption.translate(str.maketrans('', '', string.punctuation))
  original_caption = original_caption.lower()
  doc = nlp(original_caption)
 
  node_index = 0
  root = Node(node_num=node_index, prompt='entity')
 
  node_index += 1
  true_label_path = []
  all_tree_nodes = []
  node_texts = []
  edges = []
  all_tree_nodes.append(root)
  
  prev_text = ""
  add_prev_text = False
  node_texts.append("")
  next_path_node = root
  prev_path_node = root
  #curr_path_node = None
  curr_doc = None
  prev_chunk_end_index = -1
  prev_noun_chunk = None
  
  last_chunk_end_index = 0
  #print(f'original_caption : {original_caption}')
  for chunk in doc.noun_chunks:
    
    text_to_modify_lst = get_text_to_modify(prev_noun_chunk, chunk, doc)
    prev_noun_chunk = chunk
    for (doc_mod, prefix, suffix) in text_to_modify_lst:
      
      prev_path_node = next_path_node
      
      text = prefix + doc_mod.text + suffix
      #print(text)
      next_path_node = create_path_node(node_index, prev_path_node, text, true_label_path, edges, node_texts, all_tree_nodes)
      node_index += 1
   
      #new_noun_phrases = get_new_noun_phrases(doc_mod)
      new_noun_phrases = get_t5_new_sentences4_1(doc, doc_mod, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
      #get_t5_new_sentences(orig_doc, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
      for mod_text in new_noun_phrases:
        new_text = prefix + mod_text + suffix
      
        tmp_node = create_node(node_index, prev_path_node, new_text, edges, node_texts, all_tree_nodes)
        node_index += 1
       
  #add original sentence only if not added before with last noun_chunk
  if prev_noun_chunk and prev_noun_chunk.end != len(doc):
    
    prev_path_node = next_path_node
    
    next_path_node = create_path_node(node_index, prev_path_node, original_caption, true_label_path, edges, node_texts, all_tree_nodes)
    node_index += 1
  
    #new_sentences = get_new_verb_phrases(doc)
    #new_sentences = get_new_noun_phrases(doc[prev_noun_chunk.end:])
    new_sentences = get_t5_new_sentences4_1(doc, doc[prev_noun_chunk.end:], pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
    for new_sentence in new_sentences:
      text = doc[0:prev_noun_chunk.end].text + " " + new_sentence
      tmp_node = create_node(node_index, prev_path_node, text, edges, node_texts, all_tree_nodes)
      node_index += 1
  
 
  return root, true_label_path, all_tree_nodes, edges ,node_texts


def get_narratives_text_to_modify(prev_noun_chunk, curr_noun_chunk, full_doc):
  """
  returns a list of tuples (doc_mod, prefix, suffix) 
  doc_mod - the part of the doc to modify 
  prefix and suffix strings to add before and after the prompt without modification
  """

  res = []
  
  if not prev_noun_chunk:
    #this is the first chunk - return the doc from the first index without prefix or suffix
    if not "image" in curr_noun_chunk.text.lower():
      res.append((full_doc[0:curr_noun_chunk.end], "", ""))
    return res
  
  #use the prefix from the first token to the end of prev_noun_chunk and concatenate 'and' to it
  # prefix = full_doc[0:prev_noun_chunk.end].text + ' '
  prefix = full_doc[0:curr_noun_chunk.start].text + ' '
  #modify curr_noun_chunk, and do not add suffix
  # res.append((curr_noun_chunk, prefix + 'and ', ''))
  if not "image" in curr_noun_chunk.text.lower():
    res.append((curr_noun_chunk, prefix, ''))

  #if prev_noun_chunk and curr_noun_chunk have verbs and adpositions connecting between them
  #modify verbs and adpositions, and add prev_noun_chunk and curr_noun_chunk as prefix and suffix
  if prev_noun_chunk.end != curr_noun_chunk.start:
    prefix = full_doc[0:prev_noun_chunk.end].text + ' '
    suffix = ' ' + curr_noun_chunk.text
    doc_mod = full_doc[prev_noun_chunk.end:curr_noun_chunk.start]
    res.append((doc_mod, prefix, suffix))   
  
  return res


def get_text_to_modify(prev_noun_chunk, curr_noun_chunk, full_doc):
  """
  returns a list of tuples (doc_mod, prefix, suffix) 
  doc_mod - the part of the doc to modify 
  prefix and suffix strings to add before and after the prompt without modification
  """

  res = []
  
  if not prev_noun_chunk:
    #this is the first chunk - return the doc from the first index without prefix or suffix
    res.append((full_doc[0:curr_noun_chunk.end], "", ""))
    return res
  
  #use the prefix from the first token to the end of prev_noun_chunk and concatenate 'and' to it
  prefix = full_doc[0:prev_noun_chunk.end].text + ' '
  #modify curr_noun_chunk, and do not add suffix
  res.append((curr_noun_chunk, prefix + 'and ', ''))

  #if prev_noun_chunk and curr_noun_chunk have verbs and adpositions connecting between them
  #modify verbs and adpositions, and add prev_noun_chunk and curr_noun_chunk as prefix and suffix
  if prev_noun_chunk.end != curr_noun_chunk.start:
    suffix = ' ' + curr_noun_chunk.text
    doc_mod = full_doc[prev_noun_chunk.end:curr_noun_chunk.start]
    res.append((doc_mod, prefix, suffix))   
  
  return res

"""
def add_caption_tree_level(prev_noun_chunk, curr_noun_chunk, full_doc):
  curr_doc = chunk
    curr_path_node = next_path_node
    true_label_path.append(node_index)
    edges.append([curr_path_node.node_num, node_index])
    last_chunk_end_index = chunk.end
    #chunk_last_idx = chunk[-1].i
    #text = get_box_text(doc[0:chunk_last_idx].text)
    if chunk.start != 0:
      #TODO: change to doc[0:prev_chunk_end_index].text + " and " ??
      prev_text = doc[0:chunk.start].text + " "
      if prev_chunk_end_index > -1:
        #TODO: change to doc[0:prev_chunk_end_index].text + " and " ??
        prev_text = doc[0:prev_chunk_end_index].text + " "
        curr_doc = doc[prev_chunk_end_index:chunk.end]
      #print(f'\nprev_text : {prev_text}, chunk.text : {chunk.text}')

    text = doc[0:chunk.end].text
    print(f'chunk.start : {chunk.start}, chunk.end : {chunk.end}, prev_chunk_end_index: {prev_chunk_end_index}, curr_doc.text : {curr_doc.text}, prev_text: {prev_text}')
    next_path_node = Node(node_num=node_index, prompt=text, parent=curr_path_node)
    
    
    node_texts.append(get_box_text(text))
    all_tree_nodes.append(next_path_node)
    
    curr_path_node.children[node_index] = next_path_node
    node_index += 1

    
    #TODO: replace added adpositions (Prepositions and postpositions e.g. (in, under, towards, before), (of, for).)
    #TODO: replace all words that don't exist in previous node's text
    new_noun_phrases = get_new_noun_phrases(curr_doc)
    for mod_text in new_noun_phrases:
      new_text = prev_text + mod_text
      #print(f'\nmod_text : {mod_text}, \nnew_text : {new_text}')
      tmp_node = Node(node_num=node_index, prompt=new_text, parent=curr_path_node)
      node_texts.append(get_box_text(new_text))
      all_tree_nodes.append(tmp_node)
      curr_path_node.children[node_index] = tmp_node
      edges.append([curr_path_node.node_num, node_index])
      node_index += 1

    prev_chunk_end_index = chunk.end
"""

def plotly_plot_tree(all_tree_nodes, edges, original_caption, node_texts, image_file_name=""):


  #######################
  #######################
  nr_vertices = len(all_tree_nodes)
  v_label = list(map(str, range(nr_vertices)))
  #G = Graph.Tree(nr_vertices, 2) # 2 stands for children number
  G = Graph(n=nr_vertices, edges=edges)
  #lay = G.layout('rt', [0])
  lay = G.layout_reingold_tilford(root=[0])
  #plot(G, layout=lay)

  
  position = {k: lay[k] for k in range(nr_vertices)}
  Y = [lay[k][1] for k in range(nr_vertices)]
  M = max(Y)

  es = EdgeSeq(G) # sequence of edges
  E = [e.tuple for e in G.es] # list of edges

  L = len(position)
  Xn = [position[k][0] for k in range(L)]
  Yn = [2*M-position[k][1] for k in range(L)]
  Xe = []
  Ye = []
  for edge in E:
    Xe+=[position[edge[0]][0],position[edge[1]][0], None]
    Ye+=[2*M-position[edge[0]][1],2*M-position[edge[1]][1], None]

  labels = v_label
  #######################
  #######################
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=Xe,
                   y=Ye,
                   mode='lines',
                   line=dict(color='rgb(210,210,210)', width=1),
                   hoverinfo='none'
                   ))
  fig.add_trace(go.Scatter(x=Xn,
                  y=Yn,
                  #mode='markers',
                  mode='text',
                  #mode='markers+text',
                  name=original_caption,
                  marker=dict(symbol='circle-dot',
                                #size=18,
                                size=12,
                                #color='#6175c1',    #'#DB4551',
                                color='#dfe3f2',    #'#DB4551',
                                line=dict(color='rgb(50,50,50)', width=1)
                                ),
                  #text=labels,
                  text=node_texts,
                  #textposition="bottom center",
                  #textposition="middle center",
                  textposition="top center",
                  hoverinfo='text',
                  opacity=0.8
                  ))
  #######################
  #######################
  axis = dict(showline=False, # hide axis line, grid, ticklabels and  title
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            )

  fig.update_layout(title=original_caption,
              annotations=make_annotations(Xn, Yn, node_texts),
              font_size=10,
              showlegend=False,
              xaxis=axis,
              yaxis=axis,
              width=1000,
              #margin=dict(l=50, r=40, b=85, t=100),
              #margin=dict(l=50, r=40, b=55, t=100),
              #margin=dict(l=50, r=40, b=35, t=100),
              margin=dict(l=50, r=40, b=35, t=50),
              #margin=dict(l=40, r=40, b=185, t=100),
              hovermode='closest',
              plot_bgcolor='rgb(248,248,248)'
  )
  fig.update_layout(title_text=original_caption)
  #fig.update_layout(title_xanchor="center")
  #fig.update_layout(title_xanchor="right")
  fig.update_layout(title_xanchor="left")
  fig.update_layout(title_x=0.3)
  #fig.update_layout(title_font_size=<VALUE>)
  fig.update_traces(automargin=True, selector=dict(type='pie'))

  #fig.update_annotations(width=1)

  #######################
  #######################
  
  image_dir = "plotly_images"

  if not os.path.exists(image_dir):
    os.mkdir(image_dir)
  
  if image_file_name:
    file_path = image_dir + "/" + image_file_name
  else:
    file_path = image_dir + "/" + "default_fig.png"

  fig.write_image(file_path)
  return file_path
  #fig.show()
  
  #######################
  #######################

def seed_everything(seed: int = 42):
  
  #print(f'seed_everythin with seed: {seed}')        
  random.seed(seed)
  #os.environ['PYTHONHASHSEED'] = str(seed)
  #np.random.seed(seed)
  #torch.manual_seed(seed)
  #torch.cuda.manual_seed(seed)
  #torch.backends.cudnn.deterministic = True
  #torch.backends.cudnn.benchmark = True




#def get_t5_rand_mask_fill(masked_text, masked_token, num_words=3):
def get_t5_rand_mask_fill(orig_doc, mask_idx, num_words=3):
  #rand_words = get_t5_rand_mask_fill_list(masked_text, num_words)
  rand_words = get_possible_mask_fill_list(orig_doc, mask_idx, num_words=3)
  
  if not rand_words:
    print('rand_words is empty')
    return ""
    
  rand_words.sort() #sort before shuffle in order to be deterministic when using same seed
  random.shuffle(rand_words)

  for word in rand_words:
    if is_valid_word(word):
      return word

  return ""


#def get_t5_rand_mask_fill_list(masked_text, masked_token, num_words=3):
def get_t5_rand_mask_fill_list(masked_text, num_words=3):

  model, tokenizer = get_t5_model_and_tokenizer()

  #input_ids = tokenizer(masked_text, add_special_tokens=True, return_tensors="pt").input_ids.to(device)
  input_ids = tokenizer(masked_text, return_tensors="pt").input_ids.to(device)
  #print(f'input_ids: \n{input_ids}')
  #print('21')

  #beams = 5
  #beams = 3
  beams = 5
  num_seq = min(beams, num_words+1)

  #sequence_ids = model.generate(input_ids, max_length=50)
  sequence_ids = model.generate(input_ids, max_new_tokens=3, num_beams=beams, num_return_sequences=num_seq)
  #print('22')
  sequences = tokenizer.batch_decode(sequence_ids)
  #print('23')
  #print(f'get_t5_rand_mask_fill_list sequences: {sequences}')


  def _filter(output, end_token='<extra_id_1>'):
    #The first token is <pad> and the second token is <extra_id_0> 
    _txt = tokenizer.decode(output[2:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if end_token in _txt:
        _end_token_index = _txt.index(end_token)
        return _txt[:_end_token_index]
    else:
        return _txt

  word_fills = list(map(_filter, sequence_ids))
  #print(f'get_t5_rand_mask_fill_list word_fills: {word_fills}')

  #print('24')
  #print(f'results: {results}')
  #print(f'results[0]: {results[0]}')

  #word_fills = set(word_fills)
  #word_fills.discard(masked_token.text)
  #word_fills = list(word_fills)

  return word_fills


def get_t5_opposite_sentence(sentence):
  
  #if not model:
  #  model, tokenizer = get_t5_model_and_tokenizer()
  model, tokenizer = get_flan_t5_model_and_tokenizer()
  prompt = f"find an opposite for the below sentence: \n{sentence}"
  #print(f'get_t5_opposite. prompt: {prompt}')
  inputs = tokenizer(prompt, return_tensors="pt").to(device)

  
  outputs = model.generate(**inputs, max_new_tokens=3, num_beams = 5, num_return_sequences=3)
  decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
  #print(f'get_t5_opposite. decoded_outputs: {decoded_outputs}')
  return decoded_outputs


def get_t5_opposite(word):
  
  #if not model:
  #  model, tokenizer = get_t5_model_and_tokenizer()
  model, tokenizer = get_flan_t5_model_and_tokenizer()
  prompt = f"find an opposite for the word: {word}"
  #print(f'get_t5_opposite. prompt: {prompt}')
  inputs = tokenizer(prompt, return_tensors="pt").to(device)

  
  outputs = model.generate(**inputs, max_new_tokens=3, num_beams = 5, num_return_sequences=3)
  decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
  #print(f'get_t5_opposite. decoded_outputs: {decoded_outputs}')
  return decoded_outputs

def complete_sentence():
  #model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
  #tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
  #model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")
  #tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")

  #tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
  #model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", device_map="auto")

  #tokenizer = T5Tokenizer.from_pretrained("t5-3b")
  #model = T5ForConditionalGeneration.from_pretrained("t5-3b", device_map="auto")

  #model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
  #tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")


  #### flan t5-large #####
  #inputs = tokenizer("A step by step recipe to make bolognese pasta:", return_tensors="pt")
  #inputs = tokenizer("A step by step recipe to make hot coaco:", return_tensors="pt")
  #inputs = tokenizer("what is the opposite of the word: friends?", return_tensors="pt")
  #inputs = tokenizer("what is the opposite of the word: a group?", return_tensors="pt")
  #inputs = tokenizer("what is the opposite of the word: a group of friends?", return_tensors="pt")
  #inputs = tokenizer("find a word close in meaning to: green", return_tensors="pt")
  #inputs = tokenizer("the word green is a kind of", return_tensors="pt")
  #inputs = tokenizer("does the word friends have an opposite?", return_tensors="pt") A: ['enemies']
  #inputs = tokenizer("does the word green have an opposite?", return_tensors="pt") A: ['red']
  #inputs = tokenizer("does the word blue have an opposite?", return_tensors="pt") A: ['red']
  #inputs = tokenizer("does the word yellow have an opposite?", return_tensors="pt") A: ['red']
  #inputs = tokenizer("does the word playing have an opposite?", return_tensors="pt") A: ['playing a game']
  #inputs = tokenizer("find an opposite for the word group in the sentence: A group of friends playing a motion controlled video game", return_tensors="pt") A: ['group of']
  #inputs = tokenizer("find a word to replace the word group in the sentence: A group of friends playing a motion controlled video game", return_tensors="pt") A: ['friends']
  #inputs = tokenizer("find a word to replace the word group in the sentence: A group of friends playing a motion controlled video game", return_tensors="pt")
  #inputs = tokenizer("Break the following text to a few phrases: A group of friends in a house playing a motion controlled video game.", return_tensors="pt").input_ids.to(device)
  #inputs = tokenizer("Break the following text to a few phrases: A group of friends in a house playing a motion controlled video game.", return_tensors="pt")
  #['a giraffe standing on a savanna']
  #inputs = tokenizer("A photo of a:", return_tensors="pt")
  #['a single']
  #inputs = tokenizer("find an opposite for: a group", return_tensors="pt")
  #['a group']
  #inputs = tokenizer("find an opposite for: a group of people", return_tensors="pt")
  #['a group']
  #inputs = tokenizer("find an opposite phrase for the next phrase: a group of people", return_tensors="pt")
  #['sy']
  #inputs = tokenizer("a group of ", return_tensors="pt")
  #['sand']
  #inputs = tokenizer("a group of", return_tensors="pt")
  #['whose brain']
  #inputs = tokenizer("find an opposite for people", return_tensors="pt")
  #['animals']
  #inputs = tokenizer("find an opposite for the word: people", return_tensors="pt")
  #['a single']  
  #inputs = tokenizer("corrext the following: a single animals", return_tensors="pt")
  #['enemies']
  #inputs = tokenizer("find an opposite for the word: friends", return_tensors="pt")
  #['barn'] -> (num_beams = 5, num_return_sequences=3) ['barn', 'house', 'yard']
  #inputs = tokenizer("find an opposite for the word: house", return_tensors="pt")
  #['barn']
  #inputs = tokenizer("a house is a kind of", return_tensors="pt")
  #['building']
  #inputs = tokenizer("a house is a type of", return_tensors="pt")
  #['entity']
  #inputs = tokenizer("a group is a type of", return_tensors="pt")
  #['a group']
  #inputs = tokenizer("a group looks like:", return_tensors="pt")
  #['group'] -> (num_beams = 5, num_return_sequences=3) 'group', 'group of people', 'a group']
  #inputs = tokenizer("a photo of a group looks like a photo of a:", return_tensors="pt")
  #['sy']
  #inputs = tokenizer("yellow is a type of:", return_tensors="pt")
  #['sy']
  #inputs = tokenizer("what is a different kind of the word: yellow", return_tensors="pt")
  #['green'] -> ['green', 'red', 'black']
  #inputs = tokenizer("find an opposite for the word: yellow", return_tensors="pt")
  #['blue']
  #inputs = tokenizer("find an opposite for the word: green", return_tensors="pt")
  #['playing'] -> ['playing', 'ignoring', 'playing a']
  #inputs = tokenizer("find an opposite for the word: playing", return_tensors="pt")
  #['ignoring', 'heeding', 'respecting']
  #inputs = tokenizer("find an opposite for the word: ignoring", return_tensors="pt")
  #['motion uncontrolled', 'uncontrolled', 'motionless']
  #inputs = tokenizer("find an opposite for the word: motion controlled", return_tensors="pt")
  #['board game', 'computer game', 'tv']
  #inputs = tokenizer("find an opposite for the word: video game", return_tensors="pt")
  #['below', 'under', 'beneath']
  #inputs = tokenizer("find an opposite for the word: above", return_tensors="pt")
  #['of', 'a', 'of -']
  #inputs = tokenizer("find an opposite for the word: of", return_tensors="pt")
  #['by', 'unselfish', 'aby']
  #inputs = tokenizer("find an opposite for the word: by", return_tensors="pt")
  #['ad', 'An a', 'The a']
  #inputs = tokenizer("give an example of an adposition word", return_tensors="pt")
  
  #inputs = tokenizer("find an opposite for the word: artistic", return_tensors="pt")

  inputs = tokenizer("find an opposite for the word: sandwich", return_tensors="pt")
  #inputs = tokenizer("find an opposite for the word: white", return_tensors="pt")
  #inputs = tokenizer("find an opposite for the word: plate", return_tensors="pt")



  outputs = model.generate(**inputs, max_new_tokens=3, num_beams = 5, num_return_sequences=3)
  #outputs = model.generate(**inputs, max_new_tokens=3)

  print(tokenizer.batch_decode(outputs, skip_special_tokens=True))


  #### t5-large #####

  #input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", add_special_tokens=True, return_tensors="pt").input_ids.to(device)
  #input_ids = tokenizer("what is the opposite of: friends?", add_special_tokens=True, return_tensors="pt").input_ids.to(device)
  #input_ids = tokenizer("what is the opposite of: a group?", add_special_tokens=True, return_tensors="pt").input_ids.to(device)

  
  
  #input_ids = tokenizer("Fill the masks: The <extra_id_0> walks in <extra_id_1> park", add_special_tokens=True, return_tensors="pt").input_ids.to(device)
  #input_ids = tokenizer("Fill the masks: The <mask_0> walks in <mask_1> park", add_special_tokens=True, return_tensors="pt").input_ids.to(device)
  #input_ids = tokenizer("summarize the following text: A group of friends in a house playing a motion controlled video game", add_special_tokens=True, return_tensors="pt").input_ids.to(device)
  #input_ids = tokenizer("A group of friends in a house playing a motion controlled video game. Q: who is playing?", add_special_tokens=True, return_tensors="pt").input_ids.to(device)
  #input_ids = tokenizer("Break the following text to a few phrases: A group of friends in a house playing a motion controlled video game.", add_special_tokens=True, return_tensors="pt").input_ids.to(device)
  #input_ids = tokenizer("A group of friends in a house playing a motion controlled video game. Q: What are all the verbs in the sentence?", add_special_tokens=True, return_tensors="pt").input_ids.to(device)
  #input_ids = tokenizer("A group of friends in a house playing a motion controlled video game", add_special_tokens=True, return_tensors="pt").input_ids.to(device)
  #['<pad><extra_id_0> group<extra_id_1> children<extra_id_2>.</s>']
  #input_ids = tokenizer("A <extra_id_0> of <extra_id_1> in a house playing a motion controlled video game", add_special_tokens=True, return_tensors="pt").input_ids.to(device)
  #['<pad><extra_id_0> action<extra_id_1> this<extra_id_2>.</s>']
  #input_ids = tokenizer("<extra_id_0> in <extra_id_1> motion controlled video game", add_special_tokens=True, return_tensors="pt").input_ids.to(device)
  #['<pad><extra_id_0>,<extra_id_1> this<extra_id_2>.</s>']
  #input_ids = tokenizer("<extra_id_0> in <extra_id_1> <extra_id_2> motion controlled video game", add_special_tokens=True, return_tensors="pt").input_ids.to(device)
  #['<pad><extra_id_0>.</s>']
  #input_ids = tokenizer("a photo of a <extra_id_0>", add_special_tokens=True, return_tensors="pt").input_ids.to(device)
  #['<pad><extra_id_0> group<extra_id_1>.</s>']
  #input_ids = tokenizer("a photo of a <extra_id_0> of friends", add_special_tokens=True, return_tensors="pt").input_ids.to(device)
  #['<pad> a photo of a</s>']
  #input_ids = tokenizer("a photo of a ", add_special_tokens=True, return_tensors="pt").input_ids.to(device)




  

  #sequence_ids = model.generate(input_ids, max_length=20)
  ##model.generate(input_ids=input_ids, num_beams=200, num_return_sequences=20, max_length=5)

  #sequences = tokenizer.batch_decode(sequence_ids)
  #print(sequences)


def main():
  seed_everything()
  #original_sentence = "flowers in the shape of wind turbines"
  #original_sentence = "neo soul artist speaks onstage during tv programme"
  #original_sentence = "a boxer dog leaping through the air on a snow filled country lane"
  #original_sentence = "with the right equipment , mowing the lawn is pretty much a game"
  #get_sub_sentences(original_sentence)
  #print(spacy.explain('acl'))

  #predicted_caption = "a big sheep"
  #correct_caption = "a small sheep"
  #expand_caption2: ['few sheep', 'cosmic sheep', 'flock of sheep', 'small sheep', 'lesser sheep', 
    #'walloping sheep', 'astronomical sheep', 'big sheep', 'undersized sheep', 'good sheep', 'littler sheep']
  predicted_caption = "full ski racks"
  correct_caption = "empty ski racks"
  #expand_caption2: [' ski', 'empty ski', 'looted ski', 'full ski', 'clean ski', 'for ski', 'chockablock ski', 
    #'empty-handed ski', 'fraught ski', 'or ski', 'air-filled ski', 'and ski']
    #[' ski racks', 'looted ski racks', 'empty-handed ski racks', 'for ski racks', 'empty ski racks', 'and ski racks', 
    #'full ski racks', 'or ski racks', 'chockablock ski racks', 'air-filled ski racks', 'clean ski racks', 'fraught ski racks']
    #['looted ski racks', 'empty-handed ski racks', 'clean ski racks', 'fraught ski racks', 'empty ski racks', 
    #'full ski racks', 'chockablock ski racks', 'air-filled ski racks']
  
  #predicted_caption = "A lunch shoebox"
  #correct_caption = "A lunch bowl"
  #expand_caption2: ['time', 'honeycomb', 'cup', 'glass', 'bowl', 'area', 'floor', 'catchment', 'shoebox']
  #predicted_caption = "A group of friends playing a motion mouse video game"
  #correct_caption = "A group of friends playing a motion controlled video game"
  #expand_caption2: ['motion Gemini', 'motion mouse', 'motion authority', 'motion sashay', 'motion intervene', 
    #'motion based', 'motion control', 'motion controlled', 'motion inhibit', 'motion manipulate']
  #predicted_caption = 'a gray cat sitting below a table'
  #correct_caption = 'a gray cat sitting on a table'
  #expand_caption2: ['cat at table', 'cat above table', 'cat below table', 'cat over table', 'cat behind table', 
  # 'cat on top table', 'cat on table', 'cat under table', 'cat out table', 'cat after table']
  ##correct_caption = "A group of friends in a house playing a motion controlled video game"
  #expanded_captions = expand_caption(predicted_caption, correct_caption)
  expanded_captions = expand_caption2(predicted_caption, correct_caption)
  print(f'\nexpanded_captions are:\n\n {expanded_captions}')

  
  #root, true_label_path, all_tree_nodes, edges ,node_texts = get_caption_tree5(correct_caption)
  #print(f'final node texts: \n')
  #print(node_texts)
  #get_antonym('for')
  #get_co_hyponym('for')

  #adp = get_rand_adposition('in')
  #print(f'get_rand_adposition(\'in\'): {adp}')
  #adp = get_rand_adposition('out')
  #print(f'get_rand_adposition(\'out\'): {adp}')
  #adp = get_rand_adposition('behind')
  #print(f'get_rand_adposition(\'behind\'): {adp}')
          
   
        
if __name__ == "__main__":
    
    # word = 'train'
    # print(f'word is: {word}')
    # for syn in wn.synsets(word):
    #   print(f'syn: {syn}, hypernyms: {syn.hypernyms()}\n')
    #   for l in syn.lemmas():
    #     print(f'lemma: {l.name()}, hypernyms: {l.hypernyms()}\n')
      
    # opp = get_t5_opposite(word)
    # print(f'get_t5_opposite: {opp}')

    # caption = "several people standing in a green field together while flying blue kites"
    # #shuffled = shuffle_nouns(caption)
    # #shuffled = shuffle_adjectives(caption)
    # #shuffled = shuffle_nouns_adjectives(caption)
    # shuffles = get_all_shuffles_sentences(caption)
    # print(f'original caption: \n{caption}')
    # print(f'shuffled captions: \n{shuffles}')
    

    # exit(0)
    # main()
    # sentence = "several people standing in a green field together while flying kites"
    # doc = nlp(sentence)

    # print(f'\nsentence: {sentence}, nouns_chunks:')
    # for chunk in doc.noun_chunks:
    #   print(f'\n{chunk.text}')

    #model, tokenizer = get_t5_model_and_tokenizer()
    #complete_sentence()
    
    #print('10')
    #masked_text = "A group of <extra_id_0> playing a motion controlled video game"
    #orig_text = "A group of friends playing a motion controlled video game"
    #orig_text = "A sandwich and sauce on a white plate."
    #orig_text = "Very artistic photograph of a house, a lake and a wooden boat."
    #orig_doc = nlp(orig_text)
    #print('20')
    #filled_mask = get_t5_rand_word(masked_text, model, tokenizer)
    #print('30')
    #print(f'masked_text: {masked_text}, filled_masked = {filled_mask}')
    #root, true_label_path, all_tree_nodes, edges ,node_texts = get_caption_tree6(orig_text)
    #print(node_texts)
    #token_text = 'sandwich'
    #token_text = 'sauce'
    #token_text = 'plate'
    #lower_txt = token_text.lower() 
    #opposites = get_t5_opposite(lower_txt)
    #get_t5_rand_word2 lower_txt: plate, opposites: ['plateless', 'uten', 'mug']
    #print(f'get_t5_rand_word2 lower_txt: {lower_txt}, opposites: {opposites}')

    #word = 'plate'
    #word = 'man'
    #word = 'kite'
    #rand_word = get_t5_rand_word2(word, 'NOUN')
    #print(f'get_t5_rand_word2 word: {word}, rand_word: {rand_word}')
    
    #mask_idx = 3
    #masked_token = orig_doc[mask_idx]
    #print(f'masked_token.text: {masked_token.text} masked_token.i: {masked_token.i}')
    #words = get_possible_mask_fill_list(orig_doc, mask_idx, num_words=3)
    #print(words)

    #for chunk in orig_doc.noun_chunks:
    #  print(f'chunk.text: {chunk.text}')
    #  new_noun_phrases = get_t5_new_sentences3(orig_doc, chunk, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
    #  print(f'new_noun_phrases: {new_noun_phrases}')

    # orig_text = "why play is so vital for learning !"
    # orig_text = "why play is so vital for learning"

    # orig_text = "This image is taken outdoors. In the bottom of the image there is a ground with grass on it. In the middle of the image a kid is playing baseball with a baseball bat. In the background there is a mesh and a board"

    #orig_text = "A group of friends playing a motion controlled video game"

    # orig_text = "a green field"
    # orig_text = "The two women are driving on the street with the convertible top down."
    orig_text = "The cow is ahead of the person"
    
    # root, true_label_path, all_tree_nodes, edges ,node_texts = get_caption_tree6_with_all_shuffles(orig_text)
    # root, true_label_path, all_tree_nodes, edges ,node_texts = get_caption_tree6(orig_text)
    # root, true_label_path, all_tree_nodes, edges ,node_texts = get_caption_tree6_narratives(orig_text)
    # root, true_label_path, all_tree_nodes, edges ,node_texts = get_caption_tree6_depth1(orig_text)

    # cf_stmt, cf_tag = get_cfs_for_pos_statement(orig_text)
    # print(f'\n\ncf_stmt: \n{cf_stmt}')

    opposite = get_t5_opposite_sentence(orig_text)
    print(f'orig_text: {orig_text} \nopposite: {opposite}')


    # root, true_label_path, all_tree_nodes, edges ,node_texts = get_caption_tree7(orig_text)
    # print(f'node_texts: {node_texts}')
    # print(f'true_label_path: {true_label_path}')

    #word = 'uten' #False
    #word = 'plateless' #False
    #word = 'mug' # True
    #is_valid = is_valid_word(word)
    #print(f'is_valid_word. word: {word} = {is_valid}')
    """
    get_t5_opposite. prompt: find an opposite for the word: sandwich
    get_t5_opposite. decoded_outputs: ['burger', 'sand', 'tuna']
    get_t5_rand_word2 lower_txt: sandwich, opposites: ['burger', 'sand', 'tuna']
    """


    """
    ['', 'a group ', 'a crowd ', 
    'a group and friends ', 'a group and <br>ritualist ', 
    'a group of friends ', 'a group above <br>friends ', 
    'a group of friends <br>and a motion <br>controlled video <br>game ', 
    'a group of friends <br>and a level <br>controlled video <br>game ', 
    'a group of friends <br>and a motion <br>forgather video game ', 
    'a group of friends <br>and a motion <br>controlled art game ', 
    'a group of friends <br>and a motion <br>controlled video <br>playoff ', 
    'a group of friends <br>playing a motion <br>controlled video <br>game ', 
    'a group of friends <br>forbiddance a motion <br>controlled video <br>game ']
    """
    #texts = get_t5_new_sentences(orig_doc, pos_to_replace=['NOUN', 'ADJ', 'VERB', 'ADP'])
    #print(texts)




    


