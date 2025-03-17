import numpy as np

from xoa.commons.logger import *


class SearchHistory(object):
    def __init__(self, num_samples):
        
        self.num_samples = num_samples         
        #self.initialize()
                
    def initialize(self):
        
        self.completions = [] #np.arange(0)
        self.candidates = [ i for i in range(self.num_samples) ] # np.setdiff1d(np.arange(num_samples), self.completions)
        debug("{} candidates from {} to {}".format(len(self.candidates), self.candidates[0], self.candidates[-1]))

        self.test_errors = {} #[ None for i in range(num_samples) ] # np.ones(num_samples)
        self.valid_errors = {} #[ None for i in range(num_samples) ] # np.ones(num_samples)
        self.train_epochs = {} #[ 0 for i in range(num_samples) ] # np.zeros(num_samples)
        self.val_loss_curves = {}
        self.test_loss_curves = {}

    def get_candidates(self):
        return self.candidates

    def get_completions(self, min_epoch=0):

        if min_epoch == 0:
            #debug("# of all completions: {}".format(len(self.completions))) 
            return self.completions
        else:
            completions = []
            for k in sorted(self.train_epochs.keys()):
                if self.train_epochs[k] >= min_epoch:
                    completions.append(k)
            #debug("# of all completions: {}, # over {} epochs: {}".format(len(self.completions), min_epoch, len(completions)))
            return completions
        
    def get_incumbent(self):
        completed = self.completions
        compl_errs = np.array(self.get_errors("completions"), dtype=np.float)
        try:            
            min_i = np.nanargmin(compl_errs)
            return completed[min_i]
        except Exception as ex:
            # FIX: when None value exists in compl_errs
            debug("No valid evaluation retrieved: {}".format(ex))
            return None

    def get_search_order(self, sample_index):
        if sample_index in self.completions:
            search_order = self.completions.tolist()
            return search_order.index(sample_index)
        else:
            return None

    def get_train_epoch(self, sample_index):
        return self.train_epochs[sample_index]
    
    def get_loss_curve(self, sample_index, loss_type='test'):
        loss_curves = None
        if loss_type == 'test':
            loss_curves = self.test_loss_curves
        elif loss_type == 'valid':
            loss_curves = self.val_loss_curves
        else:
            raise ValueError("Not supported loss type!")
        if not sample_index in loss_curves:
            raise ValueError("Sample index not found in the loss curves")

        return loss_curves[sample_index] # XXX:dict type returned

    def update_error(self, sample_index, error_value, num_epochs=None, error_type='test'):
        error_dict = None
        loss_curves = {}
        if error_type == 'test':
            error_dict = self.test_errors
            loss_curves = self.test_loss_curves
        elif error_type == 'valid':
            error_dict = self.valid_errors
            loss_curves = self.val_loss_curves
        else:
            raise ValueError("Not supported error type!")

        error_dict[sample_index] = error_value

        if not sample_index in self.completions:            
            self.completions.append(sample_index)
            #debug("{} is added from completions -> {}".format(sample_index, len(self.completions))) 

        if sample_index in self.candidates:
            self.candidates.remove(sample_index)
            #debug("Id#{} is ejected from candidates. remains: {}".format(sample_index, len(self.candidates)))
        else:
            #debug("Id#{} has been removed from candidates".format(sample_index))
            pass  

        if num_epochs != None:
            if not sample_index in self.train_epochs:
                self.train_epochs[sample_index] = num_epochs
            elif self.train_epochs[sample_index] < num_epochs:
                self.train_epochs[sample_index] = num_epochs
            
            if not sample_index in loss_curves:
                loss_curves[sample_index] = {}    
            elif not num_epochs in loss_curves[sample_index]:
                loss_curves[sample_index][num_epochs] = error_value

        #if num_epochs == None or num_epochs > 0:    
        #    debug("#{} is completed with {} error {:.4f}, total# {}".format(sample_index, error_type, error_value, len(self.completions)))

    def get_errors(self, type_or_id="completions", min_epoch=0, error_type='test'):
        error_dict = None
        if error_type == 'test':
            error_dict = self.test_errors
        elif error_type == 'valid':
            error_dict = self.valid_errors
        else:
            raise ValueError("Invalid error type: {}".format(error_type))
        if type_or_id == "completions":
            try:
                err_list = []
                cs = self.get_completions(min_epoch) 
                if len(cs) > 0:
                    for c in cs:
                        err_list.append(error_dict[c])
                    return err_list
                else:
                    return []
            except Exception as ex:
                warn("Exception on get errors: {}".format(ex))
                debug(traceback.format_exc())
                return []
        elif type_or_id == "all":
            return error_dict
        else:
            return error_dict[type_or_id]

    def expand(self, indices):
        for i in indices:
            self.candidates.append(i)

        return indices        

    def remove(self, indices):
        candidates = np.setdiff1d(self.candidates, indices)
        self.candidates = candidates.tolist()

