import os
import numpy as np
import pandas as pd
import json
import warnings
import logging
import gc
import random
import math
import pickle
import torch
import re
import ast
import torch
from tqdm import tqdm
from torch import nn
from typing import Optional
from datetime import datetime
import nltk
nltk.download('wordnet')
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score.rouge_scorer import RougeScorer
from sklearn.metrics import f1_score
from nltk import word_tokenize
from sklearn.metrics import accuracy_score
# from torchmetrics.text.bert import BERTScore
# from torchmetrics.functional.text.bert import bert_score
import os
import random


NewDataset_path = 'Multimodal_summary(with_intent).json'
Dataset = pd.DataFrame(read_json_data(NewDataset_path))

for i in range(0, len(Dataset)):
	temp = Dataset['Conversation'][i]
	