import collections
import math
import random

import numpy as np
import torch

from fairseq import checkpoint_utils, distributed_utils, options, progress_bar, tasks, utils
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default="checkpoints/")
parser.add_argument('--fn', type=str, default="checkpoint_best.pt")
parser.add_argument('--cuda', action='store_false')
parser.add_argument('--attn_type', type=str)
parser.add_argument('--dataset_name', type=str)
parser.add_argument('--half', action='store_true')
parser.add_argument('--use_entmax', action='store_true')
parser.add_argument('--using_old_model', action='store_true')
args = parser.parse_args()

#handle command line options 
use_cuda = args.cuda
path = args.path
fn = args.fn
attn_type = args.attn_type
dataset_name = args.dataset_name
half = args.half
use_entmax = args.use_entmax
using_old_model = args.using_old_model

checkpoint = torch.load(path + fn)
args = checkpoint['args']

if using_old_model: # support vanilla fairseq models
    args.PRUNE_BOOL = False
    args.PRUNE_ENC_SELF_ATTN = False
    args.PRUNE_DEC_SELF_ATTN = False
    args.PRUNE_ENC_DEC_ATTN = False
    args.TAU = 0
    args.USE_ENTMAX = False
    args.ENCODER_SELF_ATTN_PATH = None
    args.DECODER_SELF_ATTN_PATH = None
    args.ENCODER_DECODER_ATTN_PATH = None
    args.CUDA = True
    args.RANDOM_PRUNE = False

task = tasks.setup_task(args)
model = task.build_model(args).cuda()
model.load_state_dict(checkpoint['model']);

if attn_type == 'self-enc':
    n_heads = args.encoder_attention_heads
    n_layers = args.encoder_layers
if attn_type == 'self-dec' or attn_type == 'enc-dec':
    n_heads = args.decoder_attention_heads
    n_layers = args.decoder_layers


criterion = task.build_criterion(args)

task.load_dataset('train', combine=False, epoch=0)
dataset = task.dataset('train')
   
# Initialize data iterator
itr = task.get_batch_iterator(
    dataset=dataset,
    max_tokens=args.max_tokens,
    max_sentences=args.max_sentences,
    max_positions= task.max_positions(),
    ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
    required_batch_size_multiple=args.required_batch_size_multiple,
    seed=args.seed,
    num_workers=args.num_workers,
).next_epoch_itr(shuffle=False)

subset = "train"

progress = progress_bar.build_progress_bar(
    args, itr,
    prefix='valid on \'{}\' subset'.format(subset),
    no_progress_bar='simple'
)

n = max(args.max_source_positions, args.max_target_positions)
avg_attn = [np.zeros((1, n_heads, n, n)) for i in range(0, n_layers)]
counts   = [np.zeros((1, n_heads, n, n)) for i in range(0, n_layers)]

for i, sample in enumerate(progress):
    sample = utils.move_to_cuda(sample) if use_cuda else sample
    
    model(sample['net_input']['src_tokens'], sample['net_input']['src_lengths'], sample['net_input']['prev_output_tokens'])
    
    for level in range(0, n_layers):

        if attn_type == 'self-enc': attn_probs = model.encoder.layers[level].self_attn.attn_probs.cpu().detach().numpy() 
        if attn_type == 'enc-dec': attn_probs = model.decoder.layers[level].encoder_attn.attn_probs.cpu().detach().numpy() 
        if attn_type == 'self-dec' : attn_probs = model.decoder.layers[level].self_attn.attn_probs.cpu().detach().numpy() 
        assert attn_probs is not None 

        bsz = attn_probs.shape[0]
        attn_probs = np.mean(attn_probs, axis=0, keepdims=True)

        d1, d2, d3, d4 = attn_probs.shape
        p1 = (bsz           / (counts[level] + bsz)) [0:d1, 0:d2, 0:d3, 0:d4]
        p2 = (counts[level] / (counts[level] + bsz)) [0:d1, 0:d2, 0:d3, 0:d4]
        
        avg_attn[level][0:d1, 0:d2, 0:d3, 0:d4] = p1 * attn_probs + p2 * avg_attn[level][0:d1, 0:d2, 0:d3, 0:d4]
        counts[level][0:d1, 0:d2, 0:d3, 0:d4] += bsz 

        torch.cuda.empty_cache()


# final average attention pattern 
result  = []
for tensor in avg_attn:
    result.append(tensor)
result = np.array(result)
fn = "avg_attn_probabilities/{}_reverse_{}_attn_probs_half-{}_entmax-{}.npy".format(attn_type, dataset_name, half, use_entmax)

np.save (fn, result)


