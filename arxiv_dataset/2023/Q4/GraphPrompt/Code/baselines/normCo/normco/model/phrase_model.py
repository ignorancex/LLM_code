import math
import random
import numpy as np
import torch as th
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset
from collections import defaultdict
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from .scoring import *
from ..utils.text_processing import *
# from .prepare_batch import load_text_batch

th.manual_seed(7)
np.random.seed(7)


class CoherenceModel(nn.Module):
    def __init__(self, rnn=nn.GRU, input_dim=200, output_dim=200, dropout_prob=0.3):
        super(CoherenceModel, self).__init__()

        self.input_dim = input_dim
        self.rnn_dim = output_dim // 2
        self.rnn = rnn(input_dim, self.rnn_dim, batch_first=True, bidirectional=True)

        self.output_dim = output_dim
        self.do = nn.Dropout(p=dropout_prob)

        self.reset_parameters()

    def reset_parameters(self):
        '''
        Initializers
        '''
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, input_mentions, lengths):
        '''
            input_mentions: b x sl x e
            lengths: b
        '''
        rnn_rep, _ = self.rnn(input_mentions)
        sl = rnn_rep.size()[1]

        output = rnn_rep

        return output.view(-1, sl, self.output_dim)


class SummationModel(nn.Module):
    '''
    Simple summation model
    '''

    def __init__(self, embeddings_init=None, vocab_size=10000, embedding_dim=200, sparse=False):
        super(SummationModel, self).__init__()

        # Create the word embeddings
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=sparse)
        if embeddings_init is not None:
            self.embeddings.weight.data.copy_(th.from_numpy(embeddings_init))

        self._vocab_size = vocab_size
        self._e_dim = embedding_dim

    def forward(self, inputs):
        # Forward pass for phrases
        # sz:length etc. are not used~
        examples = inputs['words']
        seq_len = examples.size()[1]
        word_len = examples.size()[2]

        # Mask the pad tokens
        nonzero = (examples != 0).type(th.FloatTensor).cuda().detach()
        embs = self.embeddings(examples.view(-1, word_len)) * nonzero.view(-1, word_len).unsqueeze(2)
        embs = embs.view(-1, seq_len, word_len, self._e_dim)

        # Sum the embeddings
        example_rep = th.sum(embs, dim=2)

        # b x sl x e
        return example_rep


class NormalizationModel(nn.Module):
    '''
    Top level normalization model measuring distance from text to concepts
    '''

    def __init__(self,num_diseases, disease_embeddings_init=None, phrase_embeddings_init=None,vocab_size=10000,
                 distfn=EuclideanDistance(), rnn=nn.GRU, embedding_dim=200, output_dim=200, dropout_prob=0.0,
                 sparse=False, use_features=False):
        super(NormalizationModel, self).__init__()

        # Phrase embedding model
        self.phrase_model = SummationModel(embeddings_init=phrase_embeddings_init,
                                           vocab_size=vocab_size,
                                           embedding_dim=embedding_dim,
                                           sparse=sparse).cuda()

        # Set the distance function
        self.distfn = distfn

        # Concept embeddings
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(num_diseases, embedding_dim, sparse=sparse, padding_idx=0)
        if disease_embeddings_init is not None:
            self.embeddings.weight.data.copy_(th.from_numpy(disease_embeddings_init))
        else:
            self.embeddings.state_dict()['weight'].uniform_(-1e-4, 1e-4)

            # Linear layer for phrase model
        self.L = nn.Linear(embedding_dim, output_dim)

        # Coherence model
        self.do = nn.Dropout(p=dropout_prob)
        self.coherence = CoherenceModel(rnn, input_dim=output_dim, output_dim=output_dim).cuda()

        # Parameter to combine models
        self.alpha = Parameter(th.FloatTensor(1))
        nn.init.constant_(self.alpha, 0.5)

        if use_features:
            self.feature_layer = nn.Linear(5, 5)
            self.f2 = nn.Tanh()
            self.score_layer = nn.Linear(5, 1, bias=False)

        self.output_dim = output_dim
        self.use_features = use_features

        self.reset_parameters()

    def reset_parameters(self):
        '''
        Initializers
        '''
        for name, param in self.L.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
                # nn.init.eye_(param)
        if self.use_features:
            for name, param in self.feature_layer.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)
            for name, param in self.score_layer.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)

    def forward(self, inputs, coherence, joint=True):
        '''
            disease_ids: b x sl x nneg
        '''
        # Get phrase, context, and concept representations
        batch_size = inputs['words'].size()[0]
        wseq_len = inputs['words'].size()[1]
        dseq_len = inputs['ids'].size()[1]
        nneg = inputs['ids'].size()[2]

        phrase_rep = self.phrase_model(inputs)
        disease_embs = self.embeddings(inputs['ids'].view(-1, nneg))

        disease_embs = disease_embs.view(-1, dseq_len, nneg,self.embedding_dim)

        linear_input = phrase_rep.view(-1, self.phrase_model._e_dim)

        # Embed phrase into concept space
        example_rep = self.L(linear_input).view(-1, wseq_len, self.output_dim)
        mention_rep = example_rep


        # Coherence vs mention only
        if coherence:
            coherence_rep = self.coherence(self.do(example_rep), inputs['seq_lens'])
            coherence_scores = self.distfn(coherence_rep.unsqueeze(2), disease_embs)

            # Joint vs separate training
            if joint:
                mention_scores = self.distfn(mention_rep.unsqueeze(2), disease_embs)
                alpha = nn.Sigmoid()(self.alpha.unsqueeze(0).unsqueeze(0))
                distance_scores = alpha * mention_scores + (1 - alpha) * coherence_scores
            else:
                distance_scores = coherence_scores
        else:
            mention_scores = self.distfn(mention_rep.unsqueeze(2), disease_embs)
            distance_scores = mention_scores
        return distance_scores
        # if self.use_features and coherence:
        #     scores = self.score_layer(self.f2(
        #         self.feature_layer(th.cat([-inputs['features'], distance_scores.unsqueeze(-1)], dim=-1).view(-1, 5))))
        #     return scores.view(-1, dseq_len, nneg)
        # else:
        #     return distance_scores