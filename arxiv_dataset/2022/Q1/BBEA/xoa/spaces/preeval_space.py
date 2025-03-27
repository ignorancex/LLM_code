import os
import copy
import json
import time
from random import randrange
import numpy as np

from xoa.commons.logger import *

from xoa.spaces.config_space import ConfigurationSpace


class SurrogatesSpace(ConfigurationSpace):

    def __init__(self, lookup, space_setting={}):

        hpv_list = lookup.get_all_hyperparam_vectors()
        super(SurrogatesSpace, self).__init__(lookup.data_type,                                               
                                              lookup.hp_config.get_dict(),
                                              hpv_list)
        # preloaded results
        self.full_error_list = lookup.get_all_test_errors()
        self.exec_times = lookup.get_all_exec_times()
        self.lookup = lookup
        self.num_epochs = lookup.num_epochs

        # Simple candidate resampling implementation
        self.resampled_set = None

        # For fast nearest index finding
        self.pvl_list = None

    # For search history 
    def update_error(self, sample_index, test_error=None, num_epochs=None, error_type='test'):
        if test_error is None:
            test_error = self.test_errors[sample_index]
        if num_epochs is None:
            num_epochs = self.num_epochs
        super(ConfigurationSpace, self).update_error(sample_index, test_error, num_epochs, error_type)

    def get_errors(self, type_or_id, min_epoch=0, error_type='test'):
        error_list = []
        if type_or_id == "completions":
            for c in self.get_completions(min_epoch):
                err = self.full_error_list[c]
                error_list.append(err)
            return error_list
        elif type_or_id == "all":
            return self.error_list
        elif type_or_id < len(self.full_error_list):
            return self.full_error_list[type_or_id]
        else:
            raise ValueError('Invalid request to get_errors: {}'.format(type_or_id))

    def get_exec_time(self, index=None):
        if index != None:
            return self.exec_times[index]
        else:
            return self.exec_times

    def expand(self, hpv, schemata=None, gen_counts=None):
        s_t = time.time()
        '''
        # return approximated index instead of newly created index        
        idx, dist = self.hp_config.get_nearby_index(self.get_candidates(), 
                                                    self.hp_vectors, 
                                                    hpv)
        debug("Distance btw selection and surrogate: {}".format(dist))
        '''
        if type(hpv) == dict:
            hpv = self.hp_config.convert('dict', 'arr', hpv)
        hpv_list = hpv
        dim = len(np.array(hpv).shape)
        if dim == 1:
            hpv_list = [ hpv ]
        elif dim != 2:
            raise TypeError("Invalid hyperparameter vector: expand")
        
        
        indices = []

        ## FIXME:below is quick appending for SMAC of 20k dataset
        if len(hpv_list) > 1000:
            # resample from candidates
            indices = np.random.choice(self.candidates, len(hpv_list))
        else:
            # XXX: add to ingore completed indices
            for c in self.completions:
                indices.append(c)

            for hpv in hpv_list:
                if len(hpv) > 0:
                    index = self.find_nearby_index(hpv, indices)
                    indices.append(index)

            # XXX:remove completed indices in the indices
            for c in self.completions:
                indices.remove(c)
        
        if self.resampled_set == None:
            self.resampled_set = indices
        else:
            for i in indices:
                self.resampled_set.append(i)         
        debug("Finding neighbors takes {:.4f} sec.".format(time.time() - s_t))
        return indices

    def find_nearby_index(self, hpv, vetoers=[]):
        s_t = time.time()

        if self.pvl_list == None or len(self.pvl_list) == 0:
            # initialize vector length list to find a nearest index of new candidate fast
            pv_list = self.candidates # XXX: Use of orignal candidates!!

            self.pvl_list = []
            for i in range(len(pv_list)):
                ipv = self.get_param_vectors(i)
                ix = np.array(ipv)
                ipvl = np.sqrt(ix.dot(ix)) # much faster version than np.linalg.norm
                pv_dict = {'index': i, 'pv': ipv, 'length': ipvl}
                self.pvl_list.append(pv_dict)
            # sorted by pvl_list values
            self.pvl_list = sorted(self.pvl_list, key=lambda k: k['length'])

        pv = self.one_hot_encode(hpv)
        x = np.array(pv)
        pvl = np.sqrt(x.dot(x))
        
        # find nearest index of the pre-evaluated configurations
        size = len(self.pvl_list)
        #debug("Candidate size: {}".format(size))
        trace_list = []
        prev_idx = size - 1 # max index 
        idx = int(size / 2) # started at middle
        new_idx = idx
        while True:     
            cpvl = self.pvl_list[idx]['length']
            trace_list.append(idx)
            
            if pvl == cpvl:
                #debug("[impossible] index: {}, distance: {:.2f}".format(idx, abs(cpvl-pvl))) 
                return idx #XXX: almost impossible case
            elif pvl < cpvl:
                new_idx = idx - int(abs(idx - prev_idx) / 2)
            elif pvl > cpvl:
                new_idx = idx + int(abs(idx - prev_idx) / 2)
            
            if new_idx == idx:
                # no way to go further
                #debug("[nearest] index: {}, distance: {:.2f}".format(idx, abs(cpvl-pvl)))
                break

            elif new_idx < 0:
                new_idx = 0 # go to the start point
            elif new_idx >= size:                
                new_idx = size - 1 # go to the end point
            
            if new_idx in trace_list:
                # already visited
                #debug("[revisted] index: {}, distance: {:.2f}".format(idx, abs(cpvl-pvl)))
                break                                 
            
            prev_idx = idx
            idx = new_idx

        # check idx is in vetoers
        while self.pvl_list[idx]['index'] in vetoers:
            dir = randrange(size)- idx
            if dir > 0:
                idx += 1
            else:
                idx -= 1
        
        cpvl = self.pvl_list[idx]['length']
        index = self.pvl_list[idx]['index']
        #debug("Nearby distance: {:.7f}".format(abs(cpvl-pvl)))
        #debug("original: {}".format(hpv))
        #debug("replaced: {}".format(self.get_hpv(idx)))

        # remove selected item
        #for i in range(len(self.pvl_list)):
        #    if self.pvl_list[i]['index'] == index:
        #        del self.pvl_list[i]
        #        break

        #debug("Finding neighbor {} takes {:.4f} sec.".format(index, time.time() - s_t))
        return index

    def remove(self, indices):
        if self.resampled_set == None:
            super(SurrogatesSpace, self).remove(indices)
        else:
            candidates = np.setdiff1d(self.resampled_set, indices)
            self.resampled_set = candidates.tolist()

    def restore_candidates(self):
        self.resampled_set = None

    def set_candidates(self, cand_indices):
        # use of subset candidate from now
        # XXX:check candidate indices were evaluated
        done_indices = self.get_completions()
        new_indices = []
        for cand in cand_indices:
            if not cand in done_indices:
                new_indices.append(cand)
            else:
                warn('Candidate #{} already evaluated!'.format(cand))

        self.resampled_set = new_indices

    def get_candidates(self):
        try:
            if type(self.resampled_set) == np.ndarray and len(self.resampled_set) > 0:
                #debug("Using {} sampled candidates".format(len(self.resampled_set)))
                return self.resampled_set
            else:
                return self.candidates
        except:
            return self.candidates
        
