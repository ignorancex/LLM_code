import random
import numpy        as np
import numpy.random as npr
from .transform import *
from scipy.stats import rankdata, spearmanr

class BaseChooser(object):
    
    def __init__(self, space, acq_func='RANDOM', shaping_func=None):
        
        self.search_space = space
        if not type(acq_func) == list:
            self.acq_funcs = [acq_func]
        else:
            self.acq_funcs = acq_func
        
        self.est_eval_time = None
        self.time_penalties = []

        self.func_m = []
        self.func_v = []
        self.best_err = None
        self.model_spec = None

        try:
            self.shaping_func = str(shaping_func)
        except:
            self.shaping_func = 'no_shaping'

 

    def add_time_acq_func(self, time_penalties):
        if type(time_penalties) != list:
            time_penalties = [time_penalties]

        for time_penalty in time_penalties:
            self.time_penalties.append(time_penalty)

        for time_penalty in time_penalties:            
            self.acq_funcs.append(time_penalty['name'])

    def set_eval_time_penalty(self, est_eval_time):
        self.est_eval_time = est_eval_time

    def output_transform(self, errs):
        # transform errors for enhancing optimization performance
        func = eval('apply_{}'.format(self.shaping_func))
        v_func = np.vectorize(func)
        if self.shaping_func == "hybrid_log":
            errs = v_func(errs, threshold=self.alpha)
        elif self.shaping_func == "power_transform":
            # power transformation is closely implemented similar to HEBO
            if np.min(errs) <= 0.0:
                norm_errs = apply_power_transform(errs, np.std(errs), 'yeo-johnson')
            else:
                norm_errs = apply_power_transform(errs, np.std(errs), 'box-cox')
                if np.std(norm_errs) < 0.5:
                    norm_errs = apply_power_transform(norm_errs, np.std(norm_errs), 'yeo-johnson')            
            if np.std(norm_errs) < 0.5:
                warn('Power transformation failed due to high variance')
            else:
                errs = norm_errs        
        elif self.shaping_func == 'no_shaping':
            pass        
        else:
            errs = v_func(errs)
        return errs

    def is_modeled(self, candidates, completions):
        # check whether the probabilistic model has been built already.
        if len(self.func_m) == 0 or len(self.func_v) == 0:
            return False
        if self.model_spec == None:
            return False
        if len(self.model_spec['inputs']) != len(completions):
            return False

        if self.best_err == None:
            return False

        for c in completions:
            if not c in self.model_spec['inputs']:
                return False
        
        if len(self.model_spec['outputs']) != len(candidates):
            return False

        return True

    def get_valid_obs(self, min_epoch=0):
        comp = np.array(self.search_space.get_param_vectors("completions", min_epoch))
        errs = np.array(self.search_space.get_errors("completions", min_epoch, error_type='valid'), dtype=np.float64)

        try:
            none_indices = np.argwhere(np.isnan(errs)).flatten()
            
            if len(none_indices) > 0:
                #debug("# of failed evaluations: {}".format(len(none_indices)))
                errs = np.delete(errs, none_indices)            
                comp = np.delete(comp, none_indices, 0) # last 0 is very important!
                #debug("NaN results deleted: {}, {}".format(comp.shape, errs.shape))  
        except Exception as ex:
            warn("Exception raised when getting on valid observations: {}".format(ex))
        #debug('Completed indices: {}'.format(self.search_space.get_completions()))
        return comp, errs

    def next(self, af, min_epoch=0):
        
        candidates = np.array(self.search_space.get_candidates())
        if af == 'RANDOM':
            # Base chooser select uniformly at random 
            next_index = int(candidates[int(np.floor(candidates.shape[0]*npr.rand()))])
        elif af == 'SEQ':
            next_index = int(candidates[0])
        else:
            raise ValueError("Unsupported acquistion function: {}".format(af))
        return next_index, None

    def reset(self):
        pass
    
    def build_model(self, comp, cand, errs):
        raise NotImplementedError('build_model() is not implemented')

    def estimate(self, af, test_set_size=10, metric='top1'):
        raise NotImplementedError('estimate() is not implemented')

    def cross_validate(self, comp_vec, errs, acq_func, test_set_size=10, metric='top1'):
        k = int(len(errs) / test_set_size)
        if k > 5:
            k = 5 # limit up to 5-fold CV
        dataset_list = self.split_dataset(comp_vec, errs, k)
        score_list = []
        score = 0.0
        
        for dataset in dataset_list:
            score = self.train_valid(dataset, acq_func, test_set_size, metric)

            score_list.append(score)

        return score_list

    def train_valid(self, dataset, acq_func, test_set_size, metric):
        train_set = dataset['train']            
        train_X = []
        train_y = []
            
        for d in train_set:
            train_X.append(d['X'])
            train_y.append(d['y'])
            
        test_set = dataset['test']
        if len(test_set) > test_set_size:
            test_set = dataset['test'][test_set_size:]
            
        test_X = []
        test_y = []    
        for d in test_set:
            test_X.append(d['X'])
            test_y.append(d['y'])
        try:
            m, v = self.build_model(train_X, test_X, train_y)
            
            best_err = np.min(train_y)
            est_rank = rankdata([-1.0 * acq_v for acq_v in acq_func(best_err, m, v)], method='ordinal')
            true_rank = rankdata(test_y, method='ordinal')

            if metric == 'spearman':
                r = spearmanr(est_rank, true_rank)
                debug('The {}'.format(r))
                score = r[0]
                if r[1] > 0.05: # unreliable performance estimation treated as zero
                    score = 0.0
            elif metric == 'top1':
                est_best_i = est_rank.tolist().index(1)
                real_rank = true_rank[est_best_i]
                score = test_set_size - real_rank
        except Exception as ex:
            score = 0.0
        return score

    def split_dataset(self, comp_vec, errs, fold, order='sort'):
        
        # NOTE:larger error value has lower order. The order starts with 1.
        err_order = [ i + 1 for i in range(len(errs))]
        if order == 'sort':
            err_order = rankdata(errs, method='ordinal') 
        elif order == 'shuffle':
            random.shuffle(err_order)

        dataset_list = []
        
        for k in range(fold):
            train_set = []
            test_set = []
            for i in range(len(err_order)):
                j = err_order[i] - 1 # j will be index of comp_vec and errs
                if i % fold == k:
                    test_set.append({'X': comp_vec[j], 'y': errs[j] })
                else:
                    train_set.append({'X': comp_vec[j], 'y': errs[j] })
            dataset_list.append({ 'train': train_set, 'test': test_set })
        
        return dataset_list


