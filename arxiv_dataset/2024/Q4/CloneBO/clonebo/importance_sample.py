import torch
import gc
import numpy as np
from scipy.special import logsumexp
from .mcmc import energy_function, conditional_sample_single
from .seq_tools import remove_spaces
from copy import deepcopy
import time


class smc:
    def __init__(self, args, model, tokenizer, clones, labeled_seqs, labels,
                 batch_size=75):

        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.labeled_seqs = labeled_seqs
        self.labels = labels
        self.n_particles = len(clones)
        self.batch_size = 75
        
        self.seqs = clones

    def num_seqs(self, clone):
        if len(clone) == 0:
            return 0
        else:
            return len(clone.split(self.tokenizer.seq_sep_token))

    def sample_seq(self, n_steps=1):
        batch_slices = [np.s_[self.batch_size*i : self.batch_size*(i+1)]
                        for i in range(int(np.ceil(self.n_particles / self.batch_size)))]
        
        for slice_ in batch_slices:
            self.seqs[slice_] = conditional_sample_single(
                    self.args,  
                    self.model,
                    self.tokenizer,
                    self.seqs[slice_],
                    greedy=False
            )

    def importance_sample(self):
        energies, _, _ = energy_function(
                self.args, self.model, self.tokenizer, self.seqs, self.labeled_seqs, self.labels,
            )
    
        log_weights = - energies - logsumexp(- energies)
        sample = np.random.choice(self.seqs, p=np.exp(log_weights))
        return sample
