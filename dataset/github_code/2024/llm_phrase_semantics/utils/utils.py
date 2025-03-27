import os, sys
from typing import Dict

# allow importing from parent dir
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers import models
from .pooling import CustomPooling
from .glove_utils import get_word_emb, get_phrase_emb, init_glove_data
from config.model_path import MODEL_PATH, GLOVE_FILE_PATH


from torch import nn, Tensor


class TransposeLayer(nn.Module):
    def __init__(self):
        super(TransposeLayer, self).__init__()
    def forward(self, features: Dict[str, Tensor]):
        for k in features.keys():
            features[k] = features[k].T
        return features


def load_openai(model_name):
    pass


def load_model(model_path, pooling='mean'):
    if 'glove' in model_path:
        glove_dict_fname = os.path.join( model_path, 'glove_dict.pkl')
        average_emb_fname = os.path.join( model_path, 'default_value.npy')
        if not (os.path.exists(glove_dict_fname) and os.path.exists(average_emb_fname)) : 
            # initialize the glove dictionary if never used
            init_glove_data(GLOVE_FILE_PATH, model_path)
        # glove model is loaded in the form of a dictionary and a default value for oov
        with open( glove_dict_fname, 'rb') as f:
            word2coeff_dict = pickle.load(f)
        average_emb = np.load( average_emb_fname)
        return word2coeff_dict, average_emb

    if model_path.startswith('bert') or model_path.startswith('roberta') or 'phrase-bert' in model_path:
        word_embedding_model = models.Transformer(model_path)
        pooling_model = CustomPooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    elif 'sentence-transformers' in model_path.lower():
        if pooling == 'span':  # torch.cat([cls_token, last_emb])
            word_embedding_model = models.Transformer(model_path)
            pooling_model = CustomPooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode_span=True, pooling_mode_mean_tokens=False)
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        else:  # mean pooling
            word_embedding_model = models.Transformer(model_path)
            pooling_model = CustomPooling(word_embedding_model.get_word_embedding_dimension())
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    elif model_path.startswith('facebook/npm') or 'spanbert' in model_path.lower():
        # span pooling, torch.cat([cls_token, last_emb])
        word_embedding_model = models.Transformer(model_path)
        pooling_model = CustomPooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode_span=True, pooling_mode_mean_tokens=False)
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    elif model_path.startswith('facebook/dpr'):
        # CLS pooling
        word_embedding_model = models.Transformer(model_path)
        transpose = TransposeLayer()
        pooling_model = CustomPooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode_cls_token=True, pooling_mode_mean_tokens=False)
        model = SentenceTransformer(modules=[word_embedding_model, transpose, pooling_model])
    elif model_path.startswith('princeton-nlp/densephrase'):
        # CLS pooling
        # Not supported, they use 3 separate language models initialized from SpanBERT
        word_embedding_model = models.Transformer(model_path)
        pooling_model = CustomPooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode_cls_token=True, pooling_mode_mean_tokens=False)
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    else:
        model = SentenceTransformer(model_path)

    return model
