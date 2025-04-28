import torch
from torch.distributions.normal import Normal
import gc
import random
import argparse
import argparse
import datetime
from pathlib import Path
import numpy as np
import time
from tqdm import tqdm
from scipy.stats import norm
from matplotlib import pyplot as plt
import wandb

import transformers
from transformers import ( 
    GenerationConfig
)

from .data import MAX_LENGTH
from .seq_tools import remove_spaces

########## Stable normal CDF

def norm_cdf(x):
  return (1 + torch.erf(x/np.sqrt(2)))/2

def log_norm_cdf_helper(x):
  a = 0.344
  b = 5.334
  return ((1 - a)*x + a*x**2+b).sqrt()

def log_norm_cdf(x):
    # stable norm cdf from https://gist.github.com/chausies/011df759f167b17b5278264454fff379
      thresh = 3
      out = x*0
      l = x<-thresh
      g = x>thresh
      m = torch.logical_and(x>=-thresh, x<=thresh)
      out[m] = norm_cdf(x[m]).log()
      out[l] = -(
          (x[l]**2 + np.log(2*np.pi))/2 + 
          log_norm_cdf_helper(-x[l]).log()
          )
      out[g] = torch.log1p(-
          (-x[g]**2/2).exp()/np.sqrt(2*np.pi)/log_norm_cdf_helper(x[g])
          )
      return out

########## Sampling and liks

# sample a single sequence conditioned on a clone (batched)
def conditional_sample_single(
    args, 
    model,
    tokenizer,
    clones,
    greedy=False
):
    gen_config = GenerationConfig(
        max_length=1e7,
        do_sample=not greedy,
        eos_token_id=tokenizer.seq_sep_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    n_seqsm1 = len(clones[0].split(tokenizer.seq_sep_token))
    # tokenize and pad clones
    tokens = [[tokenizer.bos_token] + x.split(" ") + [tokenizer.seq_sep_token] for x in clones]
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokens]
    max_len = max([len(x) for x in input_ids])
    input_ids = [[tokenizer.pad_token_id] * (max_len - len(x)) + x for x in input_ids]
    input_ids = torch.tensor(input_ids)
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    # ignore padding
    attn_mask = (input_ids != tokenizer.pad_token_id).float()

    # make batchs
    bs = min(args.max_gen_batch_size, len(clones))
    chunks = [
        input_ids[bs*i:bs*(i+1)] for i in range(0, len(clones)//bs)
    ]
    if len(clones) % bs != 0:
        chunks.append(input_ids[bs*(len(clones)//bs):])
    
    samples = []
    for chunk in chunks:
        samples += model.generate(chunk, gen_config, attention_mask=attn_mask)
    
    samples = tokenizer.batch_decode(
        samples, skip_special_tokens=True
    )
    # throw out anything that was generated after n+1 sequences
    samples = [tokenizer.seq_sep_token.join(x.split(tokenizer.seq_sep_token)[:n_seqsm1+1]).rstrip() for x in samples]

    return samples

######## Likelihoods

def model_loglikelihood(
    model,
    tokenizer,
    clones,
    likelihood_mask=None,
    past_key_values=None,
):
    input_ids = [
        tokenizer.convert_tokens_to_ids([tokenizer.bos_token]*(past_key_values is None) + x.split(" "))
        for x in clones
    ]
    #right pad to make all input_ids the same length
    max_len = max([len(x) for x in input_ids])
    input_ids = [x + [tokenizer.pad_token_id] * (max_len - len(x)) for x in input_ids]

    input_ids = torch.tensor(input_ids)
    if torch.cuda.is_available():
        input_ids = input_ids.cuda() 
        if likelihood_mask is not None:
            likelihood_mask = likelihood_mask.cuda()
            
    with torch.no_grad():
        lm_logits = model(input_ids,
                          past_key_values=past_key_values
                         ).logits
    lm_logits -= torch.logsumexp(lm_logits, -1, keepdim=True)
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = torch.clone(input_ids[..., 1:].contiguous())
    # shift_labels[shift_labels == tokenizer.pad_token_id] = -100

    # cross_entropy computes the LOSS, i.e. NLL
    clone_ll = -torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(shift_labels.size())

    if likelihood_mask is not None:
        clone_ll *= likelihood_mask

    return np.copy(clone_ll.sum(dim=-1).detach().cpu().numpy())


def conditional_likelihoods(
    args, 
    model,
    tokenizer,
    clones,
    seqs,
    progress_bar=False
):
    clone_plus_seq_ll = []
    
    for clone in clones:
        # get clone keys
        clone_ids = torch.tensor([tokenizer.convert_tokens_to_ids([tokenizer.bos_token] + clone.split(" "))])
        if torch.cuda.is_available():
            clone_ids = clone_ids.cuda() 
        with torch.no_grad():
            past_key_values = model(clone_ids).past_key_values
        
        clone_plus_seq, likelihood_mask = [], []
        for seq in seqs:
            clone_plus_seq.append(tokenizer.seq_sep_token + ' ' + seq + ' ' + tokenizer.seq_sep_token)
            mask = torch.ones(len(seq.split(" ")) + 1, dtype=torch.float)
            likelihood_mask.append(mask)

        # split batch into chunks
        bs = min(args.max_gen_batch_size, len(clone_plus_seq))
        chunks = [
            (clone_plus_seq[bs*i:bs*(i+1)], likelihood_mask[bs*i:bs*(i+1)])
             for i in range(0, int(np.ceil(len(clone_plus_seq)/bs)))
        ]
    
        t0 = time.time()
        # calculate conditional for each chunk
        
        for cps, mask in (tqdm(chunks) if progress_bar else chunks):
            torch.cuda.empty_cache()
            gc.collect()
            #pad all masks to the same length
            max_len = max([len(x) for x in mask])
            for i in range(len(mask)):
                pad_len = max_len - len(mask[i])
                mask[i] = torch.cat([mask[i], torch.zeros(pad_len, dtype=torch.float)])
            mask = torch.stack(mask)

            # broadcast past keys and values
            # print("with kv save memory:", torch.cuda.mem_get_info()[0]/1e9, "GB")
            pkv = [[k[0] for k in kv] for kv in past_key_values]
            pkv = [[k.expand(len(cps), -1, -1, -1) for k in kv] for kv in past_key_values]
            clone_plus_seq_ll += model_loglikelihood(model, tokenizer, cps, mask,
                                                     past_key_values=pkv).tolist()
    
    clone_plus_seq_ll = np.array(clone_plus_seq_ll)
    # clone_plus_seq_ll = model_loglikelihood(model, tokenizer, clone_plus_seq, likelihood_mask)
    clone_plus_seq_ll = clone_plus_seq_ll.reshape(len(clones), len(seqs))

    return clone_plus_seq_ll

########## E func and bias

def likelihood(zs, ys, args):
    if zs.shape[-1] == 1:
        return 0 * zs[:, 0]
    sig = args.label_noise_sigma
    sig /= np.sqrt(args.n_cond)
    ys = torch.tensor(ys).to(zs.device).to(zs.dtype)
    sig_beta_sq_inv = ((zs**2).mean(-1) - (zs.mean(-1) ** 2)) / (sig ** 2)
    sig_beta_sq = 1 / sig_beta_sq_inv
    mu_beta = ((zs * ys).mean(-1) - zs.mean(-1) * ys.mean(-1)) / (sig ** 2)
    mu_beta = mu_beta * sig_beta_sq
    return (0.5 * torch.log(sig_beta_sq + 1e-7)
            + 0.5 * (mu_beta ** 2) / sig_beta_sq
            + log_norm_cdf(mu_beta / torch.sqrt(sig_beta_sq)))


def energy_function(
    args,
    model,
    tokenizer,
    clones,
    labeled_seqs, 
    labels,
    plot=True
):
    n = len(labeled_seqs)
    ys = np.array(labels)
    
    labeled_seq_likelihoods = conditional_likelihoods(args, model, tokenizer, clones, labeled_seqs)
    liks_vec = torch.tensor(labeled_seq_likelihoods).to('cuda')#.requires_grad_(True)

    # print("labelled seq memory:", torch.cuda.mem_get_info()[0]/1e9, "GB")
    def lik_func(ezs, inds, start_zs=liks_vec):
        return likelihood(start_zs[inds] + ezs, ys, args)
    lik_funcs = [lambda ezs: lik_func(ezs, [i]) for i in range(len(clones))]
    label_ll = lik_func(torch.tensor(0.), np.s_[:])
    # label_ll.backward()
    plot_vec = liks_vec[0].to('cpu').numpy()

    # grads = liks_vec.grad.detach().to('cpu').numpy()
    label_ll = label_ll.to('cpu').numpy()

    energies = - label_ll
    if plot:
        plt.figure(figsize=[4, 4])
        plt.plot(plot_vec, ys, '.', color='black', alpha=0.2)
        plt.xlabel("fitness")
        plt.ylabel("label")
        plt.tight_layout()
        if wandb.run is not None:
            wandb.log({"fitness_vs_label": wandb.Image(plt)})
        plt.show()
        plt.close()
    return energies, None, lik_funcs
