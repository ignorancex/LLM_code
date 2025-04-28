import numpy as np
import torch
from scipy.stats import norm
from .mcmc import conditional_likelihoods
from . import mutate_seqs
from .seq_tools import get_ohe, get_str, get_ali_dist, alphabets_en, remove_spaces, add_spaces, is_alph
from abnumber import Chain

#################################### TOOLS ################################

from collections import defaultdict
class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError( key )
        else:
            ret = self[key] = self.default_factory(key)
            return ret

def get_cdr(seq):
    cdrs = Chain(remove_spaces([seq])[0], scheme='imgt')
    return [cdrs.cdr1_seq, cdrs.cdr2_seq, cdrs.cdr3_seq]

cdr_dict = keydefaultdict(lambda key: get_cdr(key))
def mut_in_cdr(seq, mut, only_cdrs=True):
    if only_cdrs == 3:
        cdrs = [cdr_dict[seq][2]]
        m_s = remove_spaces([mut])[0]
        return not np.all([s in m_s for s in cdrs])
    else:
        cdrs = cdr_dict[seq]
        m_s = remove_spaces([mut])[0]
        return not np.all([s in m_s for s in cdrs])

def get_muts(seqs, only_subs=True):
    seqs = [s+' ' for s in remove_spaces(seqs)] # need extra space for get mutations
    ohe = get_ohe(seqs, alphabet_name='prot')
    muts = mutate_seqs.get_mutations(ohe, only_subs=only_subs)[0]
    return [add_spaces(m) for m in get_str(muts, alphabet_name='prot')]


#################################### simple proposers ################################

def scorer_local(clones, seqs, labeled_seqs, labels,
                 args, model, tokenizer, return_liks=True):
    # calc liks
    if len(labels)>0 and return_liks:
        liks = conditional_likelihoods(args, model, tokenizer, clones, labeled_seqs, progress_bar=False)[0] # lik of first clone
        beta = sample_beta(liks, labels, args)
    
        mu_M = (labels - liks * beta).mean()
        std_M = args.label_noise_sigma/ np.sqrt(len(labels))
        M_sample = mu_M + np.random.randn() * std_M
    else:
        beta = 1
        M_sample = 0

    mut_liks = conditional_likelihoods(args, model, tokenizer, clones, seqs, progress_bar=False)[0] # lik of first clone
    return mut_liks * beta + M_sample


def proposer_local_multi(clones, labeled_seqs, labels,
                         args, model, tokenizer,
                         best_seq, only_cdr=False,
                         return_liks=False,
                         max_steps=1):
    best_lik = - 1e7
    step = 0
    while step < max_steps:
        # get unseen muts of best sequences
        muts = [get_muts([s], only_subs=True)[0] for s in best_seq]
        muts = [mu[np.logical_not(np.isin(mu, labeled_seqs))] for mu in muts]
        if only_cdr:
            muts = [mu[[mut_in_cdr(seq, m, only_cdr) for m in mu]] for seq, mu in zip(best_seq, muts)]
        muts = np.r_[*muts]
        # calc liks
        liks = scorer_local(clones, muts, labeled_seqs, labels, args, model, tokenizer, return_liks)
        if liks.max() > best_lik:
            prop = muts[np.argmax(liks)]
            best_seq = [prop]
            best_lik = liks.max()
            step += 1
        else:
            break
    return (prop,) + return_liks * (best_lik,)


def sample_beta(liks, ys, args):
    if liks.shape[-1] == 1:
        return 0 * liks[:, 0]
    sig = args.label_noise_sigma
    sig /= np.sqrt(args.n_cond)
    sig_beta_sq_inv = ((liks**2).mean(-1) - (liks.mean(-1) ** 2)) / (sig ** 2)
    sig_beta_sq = 1 / sig_beta_sq_inv
    mu_beta = ((liks * ys).mean(-1) - liks.mean(-1) * ys.mean(-1)) / (sig ** 2)
    mu_beta = mu_beta * sig_beta_sq
    # sample beta_1
    norm_t = -mu_beta / np.sqrt(sig_beta_sq)
    u_sample = norm.cdf(norm_t) + (1 - norm.cdf(norm_t)) * np.random.uniform(0, 1)
    if norm.cdf(norm_t) > 0.99:
        # print("t is too large -- cdf is", norm.cdf(norm_t))
        beta_1_sample = 0
    else:
        u_sample = norm.cdf(norm_t) + (1 - norm.cdf(norm_t)) * np.random.uniform(0, 1)
        beta_1_sample = norm.ppf(u_sample)
        beta_1_sample = beta_1_sample * np.sqrt(sig_beta_sq) + mu_beta
    # print("beta 1:", beta_1_sample)
    return beta_1_sample
   
