import os
import copy
import json
import time
import numpy as np
import pandas as pd

from xoa.commons.logger import *

from xoa.connectors.remote_space import RemoteParameterSpaceConnector
from .history import SearchHistory


class RemoteParameterSpace(SearchHistory):
    def __init__(self, space_url, cred):
        self.space = RemoteParameterSpaceConnector(space_url, credential=cred)
        self.name = "remote_{}".format(self.space.get_space_id())
        self.params_dim = None
        param_vectors = self.space.get_param_vectors('all')
        super(RemoteParameterSpace, self).__init__(len(param_vectors))

    def get_name(self):
        return self.name

    def get_params_dim(self):
        if self.params_dim == None:
            param_vectors = self.get_param_vectors('candidates')
            self.params_dim = param_vectors.shape[1]
        return self.params_dim
    
    def get_hp_config(self):
        return self.space.hp_config

    def get_param_vectors(self, type_or_index, min_epoch=0):
        return np.asarray(self.space.get_param_vectors(type_or_index, min_epoch))

    def get_hpv_dict(self, index):
            return self.space.get_hpv_dict(index)

    def get_hp_vectors(self):
            return self.space.get_hp_vectors()

    # For history
    def get_candidates(self):
        self.candidates = self.space.get_candidates()
        return self.candidates

    def get_completions(self, min_epoch=0):
        self.completions = self.space.get_completions(min_epoch)
        return self.completions

    def update_error(self, sample_index, test_error, num_epochs=None):
        return self.space.update_error(sample_index, test_error, num_epochs=num_epochs)

    def get_errors(self, type_or_id, error_type='test'):
        if error_type == 'test':
            self.test_errors, self.search_order = self.space.get_error(type_or_id, error_type=error_type)
            return self.test_errors 
        elif error_type == 'valid':
            self.valid_errors, self.search_order = self.space.get_error(type_or_id, error_type=error_type)
            return self.valid_errors

    def expand(self, hpv):
        return self.space.expand(hpv)
        