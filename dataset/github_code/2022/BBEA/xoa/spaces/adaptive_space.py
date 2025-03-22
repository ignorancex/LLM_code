import os
import copy
import json
import time
import numpy as np
from random import randint

from xoa.commons.logger import *
from .config_space import ConfigurationSpace


class AdaptiveConfigurationSpace(ConfigurationSpace):

    def __init__(self, name, hp_config_dict, hpv_list, 
                 space_setting={}):

        # Simple size diversification implementation
        self.resized_space = None
        self.resize = None
               
        if 'resize' in space_setting:
            self.resize = space_setting['resize']
            self.size_select = 'SEQ' 
        
            if 'select_size' in space_setting:
                self.size_select = space_setting['size_select']

        self.resize_start_index = 0
        self.size_selections = None

        # Simple candidate resampling implementation
        self.resampled_set = None

        super(AdaptiveConfigurationSpace, self).__init__(name, hp_config_dict, hpv_list, space_setting)

    def initialize(self):
        super(AdaptiveConfigurationSpace, self).initialize()
        self.resampled_set = None

        # space resizing
        if self.resize != None and type(self.resize) == list:
            all_cands = super(ConfigurationSpace, self).get_candidates()
            all_cands = np.array(all_cands)
            self.resized_space = {}
            self.size_selections = {}
            for s_size in self.resize:
                if type(s_size) == int:
                    cands = np.where(all_cands < s_size)[0]
                    self.resized_space[s_size] = cands
                elif type(s_size) == list:
                    i_start = s_size[0]
                    i_end = s_size[1]
                    cands = np.where(all_cands < i_end)[0]
                    cands = np.where(all_cands >= i_start)[0]
                    self.resized_space[i_end] = cands
            debug("Space resizing: {}".format(self.resized_space.keys()))

    def archive(self, run_index):

        if run_index == 0:
            k_hpv = "hpv"
            k_schemata = "schemata"
            k_gen_count = "gen_count"
        else:
            k_hpv = "hpv{}".format(run_index)
            k_schemata = "schemata{}".format(run_index)
            k_gen_count = "gen_count{}".format(run_index)

        self.backups[k_hpv] = copy.copy(self.hp_vectors)
        self.backups[k_schemata] = copy.copy(self.schemata)
        self.backups[k_gen_count] = copy.copy(self.gen_counts)

    def expand(self, indices):
        indices = super(AdaptiveConfigurationSpace, self).expand(indices)
        if self.resampled_set == None:
            return indices
        else:
            for i in indices:
                if type(i) == int:
                    self.resampled_set.append(i)
            return indices

    def remove(self, indices):
        if self.resampled_set == None:
            super(AdaptiveConfigurationSpace, self).remove(indices)
        else:
            candidates = np.setdiff1d(self.resampled_set, indices)
            self.resampled_set = candidates.tolist()

    def restore_candidates(self):
        self.resampled_set = None

    def set_candidates(self, cand_indices):
        if self.resampled_set and len(self.resampled_set) > 0 and \
            len(cand_indices) == 0 and \
            len(self.initial_hpv) <= min(self.resampled_set): # for expanded candidate only
            self.remove(self.resampled_set)
        #debug("Set candidates: {}".format(cand_indices))
        if isinstance(cand_indices, np.ndarray):
            cand_indices = cand_indices.tolist()
        
        done_indices = self.get_completions()
        new_indices = []
        for cand in cand_indices:
            if not cand in done_indices:
                new_indices.append(cand)
            else:
                warn('Candidate #{} already evaluated!'.format(cand))

        self.resampled_set = cand_indices

    def get_candidates(self):

        if self.resize != None:
            # very simple search space resizing strategy
            n_sizes = len(self.resize)
            n_done = len(self.get_completions())
            i = randint(0, n_sizes-1)
            if n_done == 0:
                self.resize_start_index = i
                #debug("Space diversification starts with {}.".format(self.resize_start_index))
            elif n_done in self.size_selections:
                cur_size = self.size_selections[n_done]
                return self.resized_space[cur_size]

            if self.size_select == 'SEQ':            
                if n_sizes > n_done:
                    i = n_done
                else:
                    i = n_done % n_sizes
            elif self.size_select == 'RANDOM':
                if n_done > 3:
                    i = randint(0, n_sizes-1)
                else:
                    i = self.resize_start_index
                    #debug("Space diversification is not started yet")
                
            cur_size = self.resize[i]
            if type(cur_size) == list:
                cur_size = cur_size[1]

            if cur_size in self.resized_space:
                debug("[SpaceDiv] # of done: {}, space id: {}".format(n_done, cur_size))
                self.size_selections[n_done] = cur_size                
                return self.resized_space[cur_size]
            else:
                raise ValueError("Invalid size: {}".format(cur_size))
        else:
            if type(self.resampled_set) == np.ndarray or type(self.resampled_set) == list:
                #debug("Use of resampled candidates: {}".format(self.resampled_set))
                return list(self.resampled_set)
            else:
                return self.candidates            

    def update_error(self, sample_index, error_value, num_epochs=None, error_type='test'):
        super(AdaptiveConfigurationSpace, self).update_error(sample_index, error_value, num_epochs, error_type)
    
        if self.resize != None:
            for k in self.resized_space:
                if sample_index in self.resized_space[k]:
                    self.resized_space[k] = np.setdiff1d(self.resized_space[k], sample_index)
                    #debug("Candidate #{} is removed in space size {}.".format(sample_index, k))

