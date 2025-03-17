import os
import copy
import json
import time
import numpy as np
import pandas as pd
from random import randint

from tqdm import tqdm

from xoa.commons.logger import *
from xoa.commons.hp_cfg import HyperparameterConfiguration

from .history import SearchHistory

class ConfigurationSpace(SearchHistory):

    def __init__(self, name, hp_config_dict, hpv_list, 
                 space_setting={}, resampled=False):

        self.name = name
        self.hp_config = HyperparameterConfiguration(hp_config_dict)
        
        self.prior_history = None
        if 'prior_history' in space_setting:
            self.prior_history = space_setting['prior_history']

        self.priors = []
        self.backups = {} 

        self.initial_hpv = hpv_list
        if 'duplicate_check' in space_setting and space_setting['duplicate_check'] == True:
            self.initial_hpv = self.validate(hpv_list)

        self.hp_vectors = {}
        self.param_vectors = {}                 
        
        self.schemata = {} 
        self.gen_counts = {} 

        super(ConfigurationSpace, self).__init__(len(self.initial_hpv))

    def validate(self, hpv_list):
        s_t = time.time()
        num_samples = len(hpv_list)
        info("Candidate validation will be performed...")
        # duplicate check (takes long time!!)      
        upv_list = [] # unique param vector list
        uhpv_list = []

        for i in range(len(hpv_list)):
            hpv = hpv_list[i]
            pv = self.one_hot_encode(hpv)
            
            is_duplicate = False

            #debug("# of unique items: {}".format(len(upv_list)))
            
            for dup_i in range(len(upv_list)):
                upv = upv_list[dup_i]
                if abs(pv[0] - upv[0]) <= 0.001 and abs(pv[-1] - upv[-1]) <= 0.001:                                  
                    dist = np.linalg.norm(np.array(pv) - np.array(upv))
                    if dist < 0.0001:
                        is_duplicate = True
                        #debug("#{} is a duplicated configuration".format(i))
                        break

            if is_duplicate == False:
                upv_list.append(pv)
                uhpv_list.append(hpv)

        if len(uhpv_list) < num_samples:
            n_reduced = num_samples - len(uhpv_list)            
            info("# of samples {} are reduced after duplicate check ({:.1f} sec.)".format(n_reduced, time.time() - s_t))

        return uhpv_list       

    def initialize(self):
        s_t = time.time()
        self.hp_vectors = {}
        self.param_vectors = {}

        for i in range(len(self.initial_hpv)):
            hpv = self.initial_hpv[i]
            pv = self.one_hot_encode(hpv)

            self.param_vectors[i] = pv
            self.hp_vectors[i] = hpv

        super(ConfigurationSpace, self).initialize()

        # load results from the previous HPO runs
        if self.prior_history != None:
            try:                
                if len(self.priors) == 0:
                    if self.prior_history.lower().endswith('.csv'):
                        self.priors = self.load_prior_from_table(self.prior_history)
                    else:
                        hp_vectors = self.load('spaces', self.prior_history)
                        if len(hp_vectors) == 0:
                            raise IOError("No space information retrieved: {}".format(self.prior_history))
                        self.priors = self.extract_prior(hp_vectors, 'results', self.prior_history)
                
                self.preset()
                debug("The # of prior observations: {}".format(len(self.completions)))

            except Exception as ex:
                warn("Use of prior history failed: {}".format(ex))
        info("Search space initialization complete. ({:.1f} sec.)".format(time.time()-s_t))
    
    def archive(self, run_index):
        
        if run_index == 0:
            k_hpv = "hpv"
            k_schemata = "schemata"
            k_gen_count = "gen_count"
            self.backups[k_hpv] = copy.copy(self.hp_vectors)
            self.backups[k_schemata] = copy.copy(self.schemata)
            self.backups[k_gen_count] = copy.copy(self.gen_counts)
          

    def save_history(self, run_index, to_dir='temp/'):
        # FIXME: below works stupidly because it refreshs from scratch.
        # save current experiment to csv format
        hpv_dict_list = []
        for c in self.completions:
            h = self.get_hpv_dict(c)
            e = self.get_errors(c)
            t = self.get_train_epoch(c)
            h['_error_'] = e
            h['_epoch_'] = t
            hpv_dict_list.append(h)

        try:
            if to_dir == 'results/':
                to_path = "./results/{}/".format(self.name)
                f_path = "{}R{}H.csv".format(to_path, run_index)
            else:
                to_path = to_dir
                f_path = "{}{}.csv".format(to_dir, self.name)

            if not os.path.isdir(to_path):
                os.mkdir(to_path)

            # create dictionary type results
            if len(hpv_dict_list) > 0:
                df = pd.DataFrame.from_dict(hpv_dict_list)
                csv = df.to_csv(f_path, index=False)
                debug("Optimization history saved at {}".format(f_path))
        except Exception as ex:
            warn("Fail to save history due to {}".format(ex))

    def save(self):
        # save hyperparameter vectors & schemata when no backup available
        if not "hpv" in self.backups:
            self.backups["hpv"] = copy.copy(self.hp_vectors)

        if not "schemata" in self.backups:
            self.backups["schemata"] = copy.copy(self.schemata)        
        
        if not "gen_count" in self.backups:
            self.backups["gen_count"] = copy.copy(self.gen_counts)

        try:
            to_path = "./results/{}/".format(self.name)
            if not os.path.isdir(to_path):
                os.mkdir(to_path)

            file_name = "{}search_space.npz".format(to_path)
            np.savez_compressed(file_name, **self.backups)
            debug("{} saved properly.".format(file_name))
        except Exception as ex:
            warn("Fail to save {} due to {}".format(file_name, ex))

    def load(self, space_folder, space_name):
        if space_folder[-1] != '/':
            space_folder += '/'
        if not os.path.isdir(space_folder):
            raise IOError("{} folder not existed".format(space_folder))
        json_file = "{}{}.json".format(space_folder, space_name)
        npz_file = "{}{}.npz".format(space_folder, space_name)
        hp_vectors = []
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                space = json.load(f)
                if 'hpv' in space:
                    hp_vectors = space['hpv']
                else:
                    warn("Invalid space format!")
        elif os.path.exists(npz_file):
            space = np.load(npz_file)
            if 'hpv' in space:
                hp_vectors = space['hpv']
            else:
                warn("Invalid space format!")           
            space.close()
        else:
            raise IOError("{}{} file not exist".format(space_folder, space_name))
        return hp_vectors

    def preset(self):
        try:
            for k in self.priors:
                c = self.priors[k]
                if 'hyperparams' in c:
                    hpv = self.hp_config.convert("arr", "list", c['hyperparams'])
                    indices = self.expand(hpv)
                    self.update_error(indices[0], c['observed_error'], c['train_epoch'], 'test')
                    self.update_error(indices[0], c['observed_error'], c['train_epoch'], 'valid')

        except Exception as ex:
            warn("Preset previous history failed:{}".format(ex))

    def extract_prior(self, hp_vectors, result_folder, result_name):
        completions = {}
        if result_folder[-1] != '/':
            result_folder += '/'
        result_path = "{}{}".format(result_folder, result_name)
        if not os.path.isdir(result_path):
            raise IOError("{} not found".format(result_path))

        for dirpath, dirnames, filenames in os.walk(result_path):
            for filename in [f for f in filenames if f.endswith(".json")]:
                result_file = os.path.join(dirpath, filename)
                debug("Priors will be from {}.".format(result_file))
                with open(result_file, 'r') as json_file:
                    results = json.load(json_file)
                    for k in results.keys():
                        r = results[k]
                        if 'model_idx' in r:
                            for i in range(len(r['model_idx'])):
                                idx = r['model_idx'][i]
                                if idx in hp_vectors:
                                    completions[idx] = {
                                        "hyperparams": hp_vectors[idx],
                                        "observed_error": r['test_error'][i],
                                        "train_epoch": r['train_epoch'][i]
                                    }
                                else:
                                    raise ValueError("No index in hpv dict: {}".format(idx))
                        else:
                            raise ValueError("Invalid prior result format: {}".format(result_file))
                            
        return completions

    def load_prior_from_table(self, csv_file, csv_dir='temp/'):
        completions = {}
        csv_path = csv_dir + csv_file
        try:            
            hist = pd.read_csv(csv_path)
            hp_params = self.hp_config.get_param_names()
            errors = hist['_error_'].tolist()
            epochs = None
            if '_epoch_' in hist:
                epochs = hist['_epoch_'].tolist()
            for i in range(len(errors)):
                hp_vector = hist[hp_params].iloc[i].tolist()
                train_epoch = 0
                if epochs != None:
                    train_epoch = epochs[i]
                completions[i] = {
                    "hyperparams": hp_vector,
                    "observed_error": errors[i],
                    "train_epoch": train_epoch 
                } 

        except Exception as ex:
            warn("Exception on loading prior history from table: {}".format(ex))
            raise ValueError("Invalid prior table file: {}".format(csv_path))

        return completions

    def get_size(self):
        return len(self.hp_vectors.keys())

    def get_name(self):
        return self.name

    def get_hp_config(self):
        return self.hp_config

    def get_schema(self, index):
        if not index in self.schemata:
            return np.zeros(self.get_hpv_dim())
        else:
            return self.schemata[index]

    def get_generation(self, index):
        if not index in self.gen_counts:
            return 0
        else:
            return self.gen_counts[index]

    def get_hpv_dim(self):
        return len(self.initial_hpv[0])

    def get_params_dim(self):
        return len(self.one_hot_encode(self.initial_hpv[0]))

    def get_param_vectors(self, type_or_index='all', min_epoch=0):
        p_list = [] 
        if type(type_or_index) == str: 
            if type_or_index == "completions":
                completions = self.get_completions(min_epoch) 
                for c in completions:
                    if not c in self.param_vectors:
                        raise ValueError("Invalid completion: {}".format(c))
                    pv = self.param_vectors[c]
                    p_list.append(pv)
            elif type_or_index == "candidates":
                #debug("getting parameters of candidates")                
                for c in self.get_candidates():
                    try:
                        if c in self.param_vectors:
                            pv = self.param_vectors[c]
                            p_list.append(pv)
                    except Exception as ex:                        
                        error("{} is invalid: {}".format(c, ex))

            elif type_or_index == 'all':
                for k in sorted(self.param_vectors.keys()):
                    p_list.append(self.param_vectors[k])
            else:
                raise ValueError("Not supported type: {}".format(type_or_index))           
            return np.array(p_list)           
        elif type_or_index in self.param_vectors:
            return self.param_vectors[type_or_index]
        else:
            raise ValueError("Invalid index: {}".format(type_or_index))
 
    def get_hp_vectors(self):        
        return self.hp_vectors  # return dict type 

    def is_existed(self, index):
        if index in self.hp_vectors:
            return True
        else:
            return False

    def get_hpv(self, index):
        if self.is_existed(index):
            return self.hp_vectors[index]
        else:
            raise ValueError("Invalid index: {}".format(index))
    
    def get_hpv_dict(self, index, k=None):
        
        hpv_dict = self.hp_vectors
        if k != None:
            if k == 0:
                if 'hpv' in self.backups: 
                    hpv_dict = self.backups['hpv']
            else:
                key = 'hpv{}'.format(k)
                if key in self.backups:
                    hpv_dict = self.backups[key]
                else:
                    error("No backup of hyperparamter vectors: {}".format(k))
        if index in hpv_dict:
            hp_arr = hpv_dict[index]
            hpv = self.hp_config.convert('arr', 'dict', hp_arr)
            return hpv # XXX:return dictionary value
        else:
            raise ValueError("Invalid index: {}".format(index))

    def set_schema(self, index, schema):
        if len(schema) != self.get_hpv_dim():
            raise ValueError("Invalid schema dimension: {} != {}".format(len(schema), self.get_hpv_dim()))

        if not index in self.schemata:
            raise ValueError("Invalid index: {}".format(index))
        else:
            # TODO:validate input
            self.schemata[index] = schema

    def get_indices(self, hpv_list): 
        
        # convert hpv_list to param_vector_list
        spv_list = []
        for h in hpv_list:
            spv = self.hp_config.convert('arr', 'one_hot', h)
            spv_list.append(np.array(spv))
        size = len(spv_list)
        indices = []
        check_list = [ False for i in range(size) ]
        # FIXME: below logic takes too much time!!
        for j in tqdm(range(len(self.candidates))):
            c = self.candidates[j]
            tpv = np.array(self.param_vectors[c])
            found = False
            if check_list.count(False) == 0:
                debug("All items have been compared completely.")
                break

            for i in range(size):
                if check_list[i] == False:
                    spv = spv_list[i]
                    if np.linalg.norm(tpv - spv) < 1e-5: # almost equal
                        indices.append(c)
                        found = True
                        check_list[i] = True
                        break
        
        not_existed_list = []    
        if check_list.count(False) > 0: # some configuration remained
            for j in range(len(check_list)):
                if check_list[j] == False:
                    not_existed_list.append(hpv_list[j])
            debug("Search space will be expanded for new {} configurations".format(len(not_existed_list)))
            indices += self.expand(not_existed_list)

        return indices
 
    def expand(self, hpv, schemata=[], gen_counts=[]):
        # check dimensions
        if type(hpv) == dict:
            hpv = self.hp_config.convert('dict', 'arr', hpv)
        hpv_list = hpv
        dim = len(np.array(hpv).shape)
        if dim == 1:
            hpv_list = [ hpv ]
        elif dim != 2:
            raise TypeError("Invalid hyperparameter vector: expand")

        vec_indices = []
        key_list = self.hp_vectors.keys()
        vec_index = max(key_list) + 1 # starts with the last model index

        for i in range(len(hpv_list)):
            h = hpv_list[i]
            if len(h) > 0: # HOTFIX: empty vector will not be added
                self.hp_vectors[vec_index] = h
                self.param_vectors[vec_index] = self.hp_config.convert('arr', 'one_hot', h)
                
                if len(schemata) > i and len(schemata[i]) > 0:
                    self.schemata[vec_index] = schemata[i]

                if len(gen_counts) > i:
                    self.gen_counts[vec_index] = gen_counts[i]

                vec_indices.append(vec_index)
                vec_index += 1
        
        return super(ConfigurationSpace, self).expand(vec_indices)

    def one_hot_encode(self, hpv):
        dim = len(np.array(hpv).shape)
        if dim == 1:
            e = self.hp_config.convert('arr', 'one_hot', hpv)
            return np.asarray(e)
        else:
            raise ValueError("Invalid HPV: {}".format(hpv)) 

    def get_incumbent(self):
        i = super(ConfigurationSpace, self).get_incumbent()
        if i != None:
            return {
                        "index": i,
                        "test_error": self.get_errors(i),
                        "valid_error": self.get_errors(i, error_type='valid'), 
                        "hpv": self.get_hpv(i), 
                        "schema": self.get_schema(i),
                        "gen": self.get_generation(i)
                    }
        else:
            return None

    def is_evaluated(self, hpv_to_check):
        if type(hpv_to_check) == dict:
            hpv_to_check = self.hp_config.convert('dict', 'arr', hpv_to_check)
        pvc = self.one_hot_encode(hpv_to_check)
        
        for i in self.get_completions():
            pv = self.get_param_vectors(i)
            dist = np.linalg.norm(pv - pvc)
            if dist < 1e-5: # XXX: ignore very small difference
                #debug("Already evaluated configuration: {}".format(hpv_to_check))
                return True
        return False

    def is_duplicated(self, hpvs, hpv_to_check):
        if type(hpv_to_check) == dict:
            hpv_to_check = self.hp_config.convert('dict', 'arr', hpv_to_check)
        pvc = self.one_hot_encode(hpv_to_check)

        for hpv in hpvs:
            pv = self.one_hot_encode(hpv)
            dist = np.linalg.norm(pv - pvc)
            if dist < 1e-5: # XXX: ignore very small difference
                #debug("Duplicated configuration: {}".format(hpv_to_check))
                return True
        return False

    def remove(self, indices):
        for i in indices:
            if not i in self.completions:
                self.hp_vectors.pop(i, None)
                self.param_vectors.pop(i, None)

        super(ConfigurationSpace, self).remove(indices)

    ''' CAUTION: analysis purpose only '''
    def compute_all(self, trainer, train_epoch=None):
        test_errs = []
        try:
            for i in self.completions:
                r = trainer.train(i, train_epoch, no_etr=True)
                if 'test_error' in r:
                    test_errs.append(r['test_error'])
                else:
                    debug("#{} returns invalid result: {}".format(i, r))
            for i in self.candidates:
                r = trainer.train(i, train_epoch, no_etr=True)
                if 'test_error' in r:
                    test_errs.append(r['test_error'])
                else:
                    debug("#{} returns invalid result: {}".format(i, r))
        except Exception as ex:
            warn("Error when retrieves all available samples: {}".format(ex))
        
        debug("# of samples that are retrieved: {}".format(len(test_errs)))
        return test_errs

