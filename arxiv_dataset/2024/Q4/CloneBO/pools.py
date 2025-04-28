import pandas as pd
import numpy as np
from collections import defaultdict
import torch
import signal
from tqdm import tqdm

from clonebo.mcmc_proposers import get_muts, add_spaces, remove_spaces, is_alph
from clonebo.oracles import LLMOracle, RandInitNeuralNetOracle

from abnumber import ChainParseError
from iglm import IgLM

class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError( key )
        else:
            ret = self[key] = self.default_factory(key)
            return ret

iglm = IgLM()
lik_dict = keydefaultdict(lambda key: iglm.log_likelihood(key, "[HEAVY]", "[HUMAN]"))

def get_oracle_and_pool(cfg):
    oracle_name = cfg['oracle_name']

    if oracle_name == 'SARSCoV1':
        from clonebo.covid_model import CovidNeutralizationModel
        fn = CovidNeutralizationModel()
        # All sequences in CoVAbDaB that are known to bind or not bind SARS-CoV1
        pool = pd.read_csv('oracle_data/CovAbDab_heavy_binds SARS-CoV1.csv')['VHorVHH'].to_numpy()[:-2]
        pool = pool[is_alph(pool)]
        # build predictor
        class oracle_class():
            def __init__(self, w):
                self.w = w
            def predict(self, seq):
                with torch.no_grad():
                    return fn.predict(seq, "")[0][0].cpu().numpy() + self.w * lik_dict[seq] # 0 for cov1 and 1 for cov2
        # noramlize std between predictor and IgLM
        muts = remove_spaces(get_muts([pool[1]], only_subs=True)[0])
        s1 = np.std([lik_dict[m] for m in muts])
        s2 = np.std([fn.predict(m, "")[0][0].detach().cpu().numpy() for m in muts])
        oracle = oracle_class(s2 / s1)
        oracle_sign = 1 #maximize binding
        only_cdrs = 3 #only cdr 3
    elif oracle_name == 'SARSCoV2Beta':
        from clonebo.covid_model import CovidNeutralizationModel
        fn = CovidNeutralizationModel()
        # All sequences in CoVAbDaB that are known to bind or not bind SARS-CoV2 beta
        pool = pd.read_csv('oracle_data/CovAbDab_heavy_binds SARS-CoV2_Beta.csv')['VHorVHH'].to_numpy()[:-2]
        pool = pool[is_alph(pool)]
        class oracle_class():
            def __init__(self, w):
                self.w = w
            def predict(self, seq):
                with torch.no_grad():
                    return fn.predict(seq, "")[0][1].cpu().numpy() + self.w * lik_dict[seq] # 0 for cov1 and 1 for cov2
        muts = remove_spaces(get_muts([pool[1]], only_subs=True)[0])
        s1 = np.std([lik_dict[m] for m in muts])
        s2 = np.std([fn.predict(m, "")[0][1].detach().cpu().numpy() for m in muts])
        oracle = oracle_class(s2 / s1)
        oracle_sign = 1 #maximize binding
        only_cdrs = 3 #only cdr 3
    elif oracle_name == 'clone':
        pool = pd.read_csv('oracle_data/clone_train_data.csv').to_numpy()[:, 1]
        oracle = LLMOracle()
        pool = pool[is_alph(pool)]
        oracle_sign = -1 #minimize loss
        only_cdrs = False
    elif 'rand' in oracle_name:
        pool = pd.read_csv('oracle_data/clone_train_data.csv').to_numpy()[:, 1]
        oracle = RandInitNeuralNetOracle(alpha=float(oracle_name.split('_')[1]))
        pool = pool[is_alph(pool)]
        oracle_sign = -1 #minimize loss
        only_cdrs = False

    ##### Get labels and function #####
    
    def timeout_handler(signum, frame):
        raise TimeoutError()
    
    def cost_func(seq):
        """ takes seqs with or without spaces """
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)  # Set alarm for 10 seconds
        
        try:
            s = ''.join(seq.strip().split(' '))
            with torch.no_grad():
                d = np.squeeze(np.array(oracle.predict(s)))
            result = oracle_sign * d
            signal.alarm(0)  # Cancel the alarm
            return result
        except (ChainParseError, IndexError, TimeoutError):
            return np.nan

    # get starting labels
    bs = 50
    label_s = remove_spaces(add_spaces(pool))
    labels = np.array([cost_func(s) for s in label_s])
    start_mean = labels.mean()
    start_std = labels.std()
    labels = labels.reshape(-1)

    # Make sure there are no more than n_labelled_mut seqs
    ind = cfg['start_ind']
    n_mut = cfg['n_labelled_mut']
    labeled_seqs = add_spaces(pool)[ind:ind+n_mut]
    labels = labels[ind:ind+n_mut]
    return cost_func, labeled_seqs, labels, (start_mean, start_std), only_cdrs
