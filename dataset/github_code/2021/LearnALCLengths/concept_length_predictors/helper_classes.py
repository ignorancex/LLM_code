import torch, random
import sys, os
base_path = os.path.dirname(os.path.realpath(__file__)).split('concept_length_predictors')[0]
sys.path.append(base_path)
from concept_length_predictors.models import LengthLearner_LSTM, LengthLearner_GRU, LengthLearner_MLP, LengthLearner_MLP2, LengthLearner_CNN, LengthLearner_Reformer
from Embeddings.models import *
from helper_classes.dataloader import CLPDataLoader
from owlapy.model import OWLNamedIndividual
from typing import Set
from sklearn.utils import resample
import pandas as pd

class ConceptLengthPredictor:
    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.learner_name = kwargs['learner_name']
        self.length_predictor = self.get_length_learner()
        self.embedding_model = self.get_embedding_model(kwargs['emb_model_name'])
        self.dataloader = CLPDataLoader(kwargs)
        
    def get_embedding_model(self, name=""):
        if name == 'ConEx':
            return ConEx(self.kwargs)
        elif name == 'Complex':
            return Complex(self.kwargs)
        elif name == 'Distmult':
            return Distmult(self.kwargs)
        elif name == 'Tucker':
            return Tucker(self.kwargs)
        else:
            print('Wrong model name, will require pretrained embeddings in csv format')
    
    def get_embedding(self, embedding_model=None):
        if embedding_model:
            if embedding_model.name == 'ConEx':
                return (embedding_model.emb_ent_real, embedding_model.emb_ent_i)
            elif embedding_model.name == 'Complex':
                return (embedding_model.Er, embedding_model.Ei)
            elif embedding_model.name == 'Distmult':
                return (embedding_model.emb_ent_real,)
            elif embedding_model.name == 'Tucker':
                return (embedding_model.E,)
        return pd.read_csv(self.kwargs['path_to_csv_embeddings']).set_index('Unnamed: 0')
    
    def get_length_learner(self):
        if self.learner_name == 'GRU':
            return LengthLearner_GRU(self.kwargs)
        elif self.learner_name == 'LSTM':
            return LengthLearner_LSTM(self.kwargs)
        elif self.learner_name == 'CNN':
            return LengthLearner_CNN(self.kwargs)
        elif self.learner_name == "MLP":
            return LengthLearner_MLP(self.kwargs)
        elif self.learner_name == 'MLP2':
            return LengthLearner_MLP2(self.kwargs)
        elif self.learner_name == 'Reformer':
            return LengthLearner_Reformer(self.kwargs)
        else:
            print('Wrong concept learner name')
            raise ValueError
            
    def refresh(self):
        self.length_predictor = self.get_length_learner()
        
    def load_pretrained(self):
        assert self.kwargs['pretrained_length_learner'], 'No pretrained length learner'
        self.length_predictor = torch.load(self.kwargs['pretrained_length_learner'], map_location=torch.device('cpu'))
        if self.embedding_model:
            assert self.kwargs['pretrained_embedding_model'], 'No pretrained embedding model'
            self.embedding_model = torch.load(self.kwargs['pretrained_embedding_model'])
        
    def predict(self, pos: Set[OWLNamedIndividual], neg: Set[OWLNamedIndividual]):
        random.seed(1)
        p = [ind.get_iri().as_str().split("/")[-1] for ind in pos]
        n = [ind.get_iri().as_str().split("/")[-1] for ind in neg]
        if min(len(n),len(p)) >= self.kwargs['num_ex']//2:
            if len(p) > len(n):
                num_neg_ex = self.kwargs['num_ex']//2
                num_pos_ex = self.kwargs['num_ex']-num_neg_ex
            else:
                num_pos_ex = self.kwargs['num_ex']//2
                num_neg_ex = self.kwargs['num_ex']-num_pos_ex
        elif len(p) > len(n) and len(p) + len(n) >= self.kwargs['num_ex']:
            num_neg_ex = len(n)
            num_pos_ex = self.kwargs['num_ex']-num_neg_ex
        elif len(p) < len(n) and len(p) + len(n)>=self.kwargs['num_ex']:
            num_pos_ex = len(p)
            num_neg_ex = self.kwargs['num_ex']-num_pos_ex
        else:
            p = resample(p, replace=True, n_samples=self.kwargs['num_ex'], random_state=123)
            n = resample(n, replace=True, n_samples=self.kwargs['num_ex'], random_state=123)
            num_pos_ex = self.kwargs['num_ex']//2
            num_neg_ex = self.kwargs['num_ex']-num_pos_ex
        p = random.sample(p, k=num_pos_ex)
        n = random.sample(n, k=num_neg_ex)
        datapoint = [(" ", {"positive examples": p, "negative examples": n})]
        if self.embedding_model is not None:
            self.embedding_model.eval()
            x, _ = list(self.dataloader.load(self.get_embedding(self.embedding_model), datapoint, 1, False))[0]
        else:
            x, _ = self.dataloader.load(self.get_embedding(self.embedding_model), datapoint, 1, False)
        if self.learner_name == "MLP": x = x.mean(dim=1)
        self.length_predictor.eval()
        if self.length_predictor.as_classification:
            print("*** predicted length: {} ***".format(int(self.length_predictor(x).argmax(1))))
            print()
            return max(1,int(self.length_predictor(x).argmax(1)))
        print("*** predicted length: {} ***".format(int(torch.round(self.length_predictor(x)))))
        print()
        return max(1,int(torch.round(self.length_predictor(x))))
    
    
    