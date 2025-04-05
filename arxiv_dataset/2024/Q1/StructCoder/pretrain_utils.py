import argparse
import pickle
import random
import numpy as np
import torch
from datetime import datetime
from pytz import timezone
from modeling_structcoder import StructCoderForConditionalGeneration
import os
from transformers import RobertaTokenizerFast
import itertools
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import average_precision_score
from bleu import _bleu
import sys
sys.path.append('CodeBLEU')
from calc_code_bleu import calc_code_bleu, calc_code_bleu_multilang
tqdm.pandas()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples_debug', type=int, default=None) # for debugging

    # max lengths
    parser.add_argument('--max_length_source_code', type=int, default=400) 
    parser.add_argument('--max_length_source_text', type=int, default=150) 
    parser.add_argument("--max_source_dfg_nodes", default=65, type=int)
    parser.add_argument("--max_source_ast_leaves", default=300, type=int)
    parser.add_argument("--max_ast_depth", default=17, type=int)
    parser.add_argument('--max_length_target_code', type=int, default=400) # for get_batch(), max_length_target_code<=max_length_source_code
    parser.add_argument('--max_length_target_text', type=int, default=150)
    
    # denoising hyperparameters
    parser.add_argument('--mask_frac', type=float, default=0.35)
    parser.add_argument('--poisson_lambda', type=float, default=12)
    
    # ablation
    parser.add_argument('--model_size', type=str, default='none') 
    parser.add_argument('--dfg_ip', type=int, default=1) 
    parser.add_argument('--ast_ip', type=int, default=1) 
    
    # optimization hyperparameters
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--clip_grad_max', type=float, default=2.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--train_batch_size', type=int, default=20)
    parser.add_argument('--text2code_loss_weight', type=float, default=1) # denoising weight is always 1
    parser.add_argument('--code2text_loss_weight', type=float, default=0.5) 
    parser.add_argument('--dfg_loss_weight', type=float, default=0.1) # LM loss weight is always 1
    parser.add_argument('--ast_loss_weight', type=float, default=0.1) 
    
    # testing hyperparameters
    parser.add_argument('--validate_every', type=int, default=5000) # validate every __ training batches
    parser.add_argument('--num_valid_samples', type=int, default=3000) # use last __ samples in data for validation
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--num_eval_batches_bleu', type=int, default=30)
    parser.add_argument('--num_beams', type=int, default=2)
    
    # logging and other hyperparameters
    parser.add_argument('--resume', type=int, default=0) # whether to continue training with the last ckpt for this config
    parser.add_argument('--print_train_loss_every', type=int, default=100) # pritn train loss every __ training batches
    parser.add_argument('--checkpoint_every', type=int, default=50000) # save best model weights for every __ training batches
    parser.add_argument('--patience', type=int, default=1000000) # no. of validation steps with no improvement after which to stop training
    parser.add_argument('--max_steps', type=int, default=1000000000)
    parser.add_argument('--seed', type=int, default=2022) # for RNGs
    parser.add_argument('--data_dir', type=str, default='data/pretrain/') 
    parser.add_argument('--output_dir', type=str, default=None) 
    # output_dir is directory to save log file, checkpoints, etc. Set to None to automatically set this in set_output_dir()

    args = parser.parse_args()
    return args


def get_curr_time():
    return datetime.now().astimezone(timezone('US/Pacific')).strftime("%d/%m/%Y %H:%M:%S")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count()>0:
        torch.cuda.manual_seed_all(seed)
        
        
def set_rng_states(random_rng_state, np_rng_state, torch_rng_state, torch_cuda_rng_state):
    random.setstate(random_rng_state)
    np.random.set_state(np_rng_state)
    torch.set_rng_state(torch_rng_state)
    assert (len(torch_cuda_rng_state)==torch.cuda.device_count())
    for i,s in enumerate(torch_cuda_rng_state):
        torch.cuda.set_rng_state(s, device='cuda:'+str(i))
        
        
def save_ckpt(save_path, model, optimizer, step, train_cycler, wait, best_val_metric):
    save_dict = {'model_weights': model.state_dict(), 
                 'optimizer_state': optimizer.state_dict(),
                 'step': step,
                 'train_cycler': train_cycler,
                 'wait': wait,
                 'best_val_metric': best_val_metric,
                 'random_rng_state': random.getstate(),
                 'np_rng_state': np.random.get_state(),
                 'torch_rng_state': torch.get_rng_state(),
                 'torch_cuda_rng_state': [torch.cuda.get_rng_state(device='cuda:'+str(i)) for i in range(torch.cuda.device_count())],
                }
    torch.save(save_dict, save_path)
    del save_dict
        
        
def set_output_dir(args):
    if args.output_dir is not None:
        return
    args.output_dir = 'saved_models/pretrain/'
    if args.model_size=='small':
        args.output_dir += 'small_'
        for argument in ['dfg_ip', 'ast_ip', 'dfg_loss_weight', 'ast_loss_weight']:
            args.output_dir += argument+'_'+str(getattr(args,argument))
        args.output_dir += '/'
    os.makedirs(args.output_dir, exist_ok=True)
    
    
class Logger(): # write message to both output_dir/filename.txt and terminal
    def __init__(self, output_dir=None, filename=None):
        if filename is not None:
            self.log = os.path.join(output_dir, filename)

    def write(self, message, show_time=True):
        message = str(message)
        if show_time:
            if message.startswith('\n'): # if message starts with \n, print the \n first before printing time
                message = '\n'+get_curr_time()+' >> '+message[1:]
            else:
                message = get_curr_time()+' >> '+message
        print (message)
        if hasattr(self, 'log'):
            with open(self.log, 'a') as f:
                f.write(message+'\n') 
                
                
class CycleIndex(): 
    def __init__(self, num_samples, batch_size, shuffle=True):
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.pointer = 0 # always in [0, num_samples-1]
        self.indices = np.arange(num_samples)
        if shuffle:
            np.random.shuffle(self.indices)
        self.shuffle = shuffle

    # new : last batch in an epoch also has self.batch_size samples
    def get_batch_ind(self):
        start, end = self.pointer, self.pointer+self.batch_size
        
        if end<=self.num_samples: # If we have a full batch, then return it.
            self.pointer = 0 if end==self.num_samples else end # Reset pointer to 0 if epoch is over.
            return self.indices[start:end]  
        
        # If you reached here, then end>self.num_samples and self.indices[start:] does not contain a full batch.
        last_batch_indices_incomplete = self.indices[start:].copy() 
        remaining = self.batch_size - (self.num_samples-start)
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.pointer = remaining
        return np.concatenate((last_batch_indices_incomplete, self.indices[:remaining]))  
                
        
def set_tokenizer(args):
    args.tokenizer = RobertaTokenizerFast.from_pretrained('Salesforce/codet5-base')
    args.task_tokens = {'text2code':{lang : args.tokenizer(' Generate '+lang).input_ids[1:-1]+[args.tokenizer.eos_token_id] \
                                             for lang in ['php', 'java', 'go', 'javascript', 'ruby', 'python']}, 
                        'code2text':args.tokenizer(' Generate text').input_ids[1:-1]+[args.tokenizer.eos_token_id], 
                        'denoise':args.tokenizer(' Denoise').input_ids[1:-1]+[args.tokenizer.eos_token_id]}
                  
        
def parse_list_of_lists(s, max_value=None, max_len=None, inner_max_len=None):
    list_of_lists = s[1:-2].split('], ')
    if max_len is not None:
        list_of_lists = list_of_lists[:max_len]
    list_of_lists = [np.array([int(i) for i in x[1:].split(', ') if i!='']) for x in list_of_lists]
    if inner_max_len is not None:
        list_of_lists = [l[:inner_max_len] for l in list_of_lists]
    if max_value is not None:
        list_of_lists = [l[l<=max_value] for l in list_of_lists]
    return list_of_lists
        
    
def format_dfg_edges(s, max_source_dfg_nodes):
    if s=='[]':
        return []
    s = s[2:-2].split('), (') 
    ret = []
    for x in s:
        x = x.split(', [')
        left = int(x[0])
        if left<max_source_dfg_nodes:
            rights = np.array([int(i) for i in x[1][:-1].split(', ')])
            rights = rights[rights<max_source_dfg_nodes]
            if len(rights)>0:
                ret.append((left, rights))
    return ret
    

def read_data(args):
    args.data = []
    num_samples_read = 0
    for filename in tqdm(os.listdir(args.data_dir)):
        if filename.startswith('from_'):
            args.data.append(pd.read_parquet(args.data_dir+filename, engine='fastparquet'))
            num_samples_read += len(args.data[-1])
            if (args.num_samples_debug is not None) and (num_samples_read>=args.num_samples_debug):
                break
    args.data = pd.concat(args.data)
    if args.num_samples_debug is not None:
        args.data = args.data.iloc[:args.num_samples_debug]
    # columns: lang, text_tokens, code_tokens, dfg_node_code_token_idxs, dfg_edges, ast_leaf_code_token_idxs, ll_sims, lr_paths_types
    
    for col, max_len in zip(['dfg_node_code_token_idxs', 'ast_leaf_code_token_idxs'], 
                            [args.max_source_dfg_nodes, args.max_source_ast_leaves]):
        args.data[col] = args.data[col].progress_apply(lambda s:parse_list_of_lists(s,args.max_length_source_code,max_len))
        
    args.data['lr_paths_types'] = args.data['lr_paths_types'].progress_apply(
                                        lambda s:parse_list_of_lists(s,None,args.max_source_ast_leaves,args.max_ast_depth))
        
    args.data['num_dfg_nodes'] = args.data['dfg_node_code_token_idxs'].progress_apply(len)
    
    args.data['dfg_edges'] = args.data['dfg_edges'].progress_apply(lambda s:format_dfg_edges(s,args.max_source_dfg_nodes))
    
    args.logger.write('Read '+str(len(args.data))+' samples.')
    args.num_node_types = len(pickle.load(open(args.data_dir+'all_node_types.pkl','rb')))
    

def count_parameters(model, only_trainable=False):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

    
def load_model(args):
    model = StructCoderForConditionalGeneration(args)
    # print #parameters, #trainable_parameters
    args.logger.write('# parameters : '+str(count_parameters(model, only_trainable=False)))
    args.logger.write('# trainable parameters : '+str(count_parameters(model, only_trainable=True)))
    # place model on gpu
    model.to(args.device)
    if torch.cuda.device_count()>0:
        model = torch.nn.DataParallel(model)
        args.logger.write('Using '+str(torch.cuda.device_count())+' GPUs.')
    return model
    
    
def pad_batch(input_ids, args, return_lens=False):
    input_lens = np.array([len(x) for x in input_ids])
    max_len = max(input_lens)
    batch_size = len(input_ids)
    mask = np.zeros((batch_size, max_len))
    input_ids_arr = np.ones((batch_size, max_len)) * args.tokenizer.pad_token_id
    for i in range(batch_size):
        mask[i, :input_lens[i]] = 1
        input_ids_arr[i, :input_lens[i]] = input_ids[i]
    ret = {'input_ids':torch.LongTensor(input_ids_arr).to(args.device),
            'attention_mask':torch.FloatTensor(mask).to(args.device)}
    if return_lens:
        ret['input_lens'] = input_lens
    return ret

def get_node_code_links(sub_data_col, max_code_len, device):
    bsz = len(sub_data_col)
    max_nodes = sub_data_col.apply(len).max()
    node_code_links = np.zeros((bsz, max_nodes, max_code_len))
    for bind,list_of_lists in enumerate(sub_data_col):
        for node_ind,code_inds in enumerate(list_of_lists):
            if len(code_inds)>0:
                node_code_links[bind,node_ind,code_inds] = 1
    return torch.LongTensor(node_code_links).to(device)

def get_dfg_dfg_links(sub_data_col, max_nodes, device):
    bsz = len(sub_data_col)
    dfg_dfg_links = np.zeros((bsz, max_nodes, max_nodes))
    for bind,edges in enumerate(sub_data_col):
        for left, rights in edges:
            dfg_dfg_links[bind,left,rights] = 1
    return torch.IntTensor(dfg_dfg_links).to(device)

def get_ast_lr_paths(sub_data_col, args):
    bsz = len(sub_data_col)
    max_num_leaves = sub_data_col.apply(len).max()
    P = -np.ones((bsz,max_num_leaves,args.max_ast_depth))
    for bind,paths in enumerate(sub_data_col):
        for leaf_ind,path in enumerate(paths):
            P[bind,leaf_ind,-len(path):] = path # place path at end so that all roots get same depth embedding
    return torch.LongTensor(P).to(args.device)

def parse_ast_ast_sims(s, max_num_leaves):
    rows = s.split(';', maxsplit=max_num_leaves)[:max_num_leaves-1] # nrows-1
    num_leaves = len(rows)+1
    sims = np.array([['0']*(i+1) + row.split(',',maxsplit=max_num_leaves)[:num_leaves-i-1] 
                     for i,row in enumerate(rows)]).astype(float) # num_leaves-1, num_leaves
    sims = np.concatenate((sims, np.zeros((1,num_leaves))), axis=0) # num_leaves, num_leaves
    sims = sims + sims.T + np.identity(num_leaves)
    return sims

def get_ast_ast_sims(sub_data_col, max_num_leaves, device):
    S = np.zeros((len(sub_data_col),max_num_leaves,max_num_leaves))
    for bind,s in enumerate(sub_data_col):
        sims = parse_ast_ast_sims(s, max_num_leaves)
        L = sims.shape[1]
        S[bind,:L,:L]=sims
    S = torch.FloatTensor(S).to(device)
    return torch.log(1+S)

def add_noise(code_inputs, args):
    max_len = code_inputs['input_ids'].size()[1]
    num_to_mask = int(np.round(max_len * args.mask_frac))
    mask_lengths = np.random.poisson(args.poisson_lambda, num_to_mask) 
    mask_lengths[mask_lengths==0] = 1
    
    # Trim to masking budget
    if mask_lengths.sum()>num_to_mask:
        cum_length = np.cumsum(mask_lengths)
        i = 0
        while cum_length[i] < num_to_mask:
            i += 1
        mask_lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
        num_to_mask = i+1 if (mask_lengths[i]>0) else i # no. of spans to mask
        mask_lengths = mask_lengths[:num_to_mask]
        
    # start indices for span masking, no <CLS> or <SEP>
    indices = np.sort(np.random.permutation(max_len-2)[:num_to_mask]+1) # 1 to max_len-2
    
    # delete, replace with random word, or mask
    del_rep_mask = np.random.randint(0,3,size=num_to_mask)
    
    # replace with MASK, rand tokens
    mask_token_id = args.tokenizer.mask_token_id
    ids = code_inputs['input_ids'].reshape(-1)
    ids = ids[torch.isin(ids.cpu(), torch.LongTensor([args.tokenizer.pad_token_id, args.tokenizer.cls_token_id, 
                                                args.tokenizer.sep_token_id, args.tokenizer.unk_token_id]), invert=True)]
    num_ids = len(ids)
    keep = np.ones(max_len, dtype=bool)
    for i,(start,mask_length,typ) in enumerate(zip(indices,mask_lengths,del_rep_mask)):
        if typ==0: # replace with mask
            code_inputs['input_ids'][:,start:start+mask_length] = mask_token_id
        elif typ==1: # replace with random token
            end = min(start+mask_length, max_len)
            code_inputs['input_ids'][:,start:end] = ids[torch.randint(high=num_ids,size=(end-start,))]
        else: # delete
            keep[start:start+mask_length] = False
            
    # can get an error if keep.sum()<3
    if keep.sum()<3:
        keep[-3:] = True
            
    # delete tokens
    code_inputs['input_ids'] = code_inputs['input_ids'][:, keep]
    code_inputs['attention_mask'] = code_inputs['attention_mask'][:,keep]
    code_inputs['dfg_code_links'] = code_inputs['dfg_code_links'][:,:,keep]
    code_inputs['ast_code_links'] = code_inputs['ast_code_links'][:,:,keep]
            
    # make last 4 tokens " denoising EOS".
    bsz = code_inputs['attention_mask'].size()[0]
    code_inputs['attention_mask'] = torch.cat((code_inputs['attention_mask'], 
                                               torch.zeros((bsz,1)).to(args.device)), dim=1) # bsz,L+1
    code_inputs['input_ids'] = torch.cat((code_inputs['input_ids'], code_inputs['input_ids'][:,-1:]), dim=1) # bsz,L+1
    L_dfg = code_inputs['dfg_code_links'].size()[1]
    code_inputs['dfg_code_links'] = torch.cat((code_inputs['dfg_code_links'], 
                                               torch.zeros((bsz,L_dfg,1)).to(args.device)), dim=2)
    L_ast = code_inputs['ast_code_links'].size()[1]
    code_inputs['ast_code_links'] = torch.cat((code_inputs['ast_code_links'], 
                                               torch.zeros((bsz,L_ast,1)).to(args.device)), dim=2)
    new_lens = code_inputs['attention_mask'].sum(dim=1).int() # bsz
    new_lens = torch.clip(new_lens, min=3) # can get an error later if length<3
    eos_token_id = args.tokenizer.eos_token_id
    task_tokens = torch.cuda.LongTensor(args.task_tokens['denoise'])
    for i,l in enumerate(new_lens):
        code_inputs['input_ids'][i,l-3:l+1] = task_tokens
        code_inputs['attention_mask'][i,l-3:l+1] = 1
        
    # delete some DFG nodes and AST leaves.
    max_num_nodes = code_inputs['dfg_code_links'].size()[1]
    dfg_keep = torch.rand(max_num_nodes)>args.mask_frac
    code_inputs['dfg_code_links'] = code_inputs['dfg_code_links'][:,dfg_keep,:]
    code_inputs['dfg_dfg_links'] = code_inputs['dfg_dfg_links'][:,dfg_keep,:]
    code_inputs['dfg_dfg_links'] = code_inputs['dfg_dfg_links'][:,:,dfg_keep]
    
    max_num_leaves = code_inputs['ast_code_links'].size()[1]
    ast_keep = torch.rand(max_num_leaves)>args.mask_frac
    code_inputs['ast_code_links'] = code_inputs['ast_code_links'][:,ast_keep,:]
    code_inputs['ast_ast_sims'] = code_inputs['ast_ast_sims'][:,ast_keep,:]
    code_inputs['ast_ast_sims'] = code_inputs['ast_ast_sims'][:,:,ast_keep]
    code_inputs['ast_paths'] = code_inputs['ast_paths'][:,ast_keep,:]
    
    # remove some anecestors in lr_paths.
    ast_mask = (torch.rand(code_inputs['ast_paths'].size(), device=code_inputs['ast_paths'].device)>args.mask_frac).int()
    code_inputs['ast_paths'] = code_inputs['ast_paths']*ast_mask - (1-ast_mask)
    
    del code_inputs['input_lens']
    
    return code_inputs # input_ids, attention_mask, dfg_code_links, ast_code_links, dfg_dfg_links, ast_paths

# prepare io for 3 tasks using a batch_ind
# normal code input, normal text input, corrupted code input
# normal text output, normal code output, normal code output
def get_batch(batch_ind, args):
    batch_io = {} 
    sub_data = args.data.iloc[batch_ind]
    
    # normal code input 
    # input_ids, attention_mask, num_dfg_nodes, dfg_code_links, dfg_dfg_links, ast_code_links, ast_ast_sims, ast_paths
    input_ids = list(sub_data['code_tokens'].apply(lambda s:[int(t) for t in s.split(',')]))
    input_ids2 = [l1[:args.max_length_source_code+1]+args.task_tokens['code2text'] for l1 in input_ids] 
    code_inputs = pad_batch(input_ids2, args, return_lens=True) # input_ids, attention_mask
    code_inputs['num_dfg_nodes'] = torch.IntTensor(list(sub_data['num_dfg_nodes'])).to(args.device)
    max_code_len = code_inputs['input_ids'].size()[1]
    code_inputs['dfg_code_links'] = get_node_code_links(sub_data['dfg_node_code_token_idxs'], max_code_len, args.device) #b,Ld,L
    code_inputs['ast_code_links'] = get_node_code_links(sub_data['ast_leaf_code_token_idxs'], max_code_len, args.device) # b, L_ast, L
    code_inputs['dfg_dfg_links'] = get_dfg_dfg_links(sub_data['dfg_edges'], code_inputs['dfg_code_links'].size()[1], args.device) #b,Ld,Ld
    code_inputs['ast_paths'] = get_ast_lr_paths(sub_data['lr_paths_types'], args) #b,L_ast,max_depth
    code_inputs['ast_ast_sims'] = get_ast_ast_sims(sub_data['ll_sims'], code_inputs['ast_paths'].size()[1], args.device)
    batch_io['code_inputs'] = code_inputs.copy()
    del batch_io['code_inputs']['input_lens']
    
    # normal code output  
    # input_ids, attention_mask, dfg_dfg_links, ast_paths
    input_ids2 = [l1[:args.max_length_target_code+1]+[args.tokenizer.eos_token_id] for l1 in input_ids] 
    code_outputs = pad_batch(input_ids2, args, return_lens=True) # input_ids, attention_mask
    dfg_code_links = (code_inputs['dfg_code_links'][:,:,:code_outputs['input_ids'].size()[1]]).float() #b,Ld,L
    code_outputs['dfg_dfg_links'] = (torch.bmm(torch.bmm(dfg_code_links.transpose(1,2), code_inputs['dfg_dfg_links'].float()),
                                               dfg_code_links)>0).int()  # b,L,L
    attention_mask = torch.clone(code_outputs['attention_mask'])
    attention_mask[:,0] = 0 # remove bos
    attention_mask[:,code_outputs['input_lens']-1] = 0 # remove eos
    del code_outputs['input_lens']
    neg_attention_mask = 1-attention_mask
    code_outputs['dfg_dfg_links'] = code_outputs['dfg_dfg_links']*attention_mask[:,:,None] - neg_attention_mask[:,:,None]
    code_outputs['dfg_dfg_links'] = code_outputs['dfg_dfg_links']*attention_mask[:,None,:] - neg_attention_mask[:,None,:]
    
    ast_code_links = code_inputs['ast_code_links'][:,:,:code_outputs['input_ids'].size()[1]] #b,La,L
    ast_code_links = ((torch.cumsum(ast_code_links,dim=1)==1).int() * ast_code_links).float() ## BUG: change to dim=0
    code_outputs['ast_paths'] = torch.bmm(ast_code_links.transpose(1,2), code_inputs['ast_paths'].float()).long() # b,L,max_depth
    code_outputs['ast_paths'] = code_outputs['ast_paths']*attention_mask[:,:,None] - neg_attention_mask[:,:,None]
    batch_io['code_outputs'] = code_outputs
    
    # corrupted code input
    batch_io['corrupt_code_inputs'] = add_noise(code_inputs, args)
    
    # normal text input and output
    input_ids = list(sub_data['text_tokens'].apply(lambda s:[int(t) for t in s.split(',')]))
    task_ids = list(sub_data['lang'].apply(lambda lang:args.task_tokens['text2code'][lang]))
    input_ids_input = [l1[:args.max_length_source_text+1]+l2 for l1,l2 in zip(input_ids,task_ids)] 
    batch_io['text_inputs'] = pad_batch(input_ids_input, args)
    input_ids_output = [l1[:args.max_length_target_text+1]+[args.tokenizer.eos_token_id] for l1 in input_ids] 
    batch_io['text_outputs'] = pad_batch(input_ids_output, args)
    
    # remove structure for ablation if specified
    if not(args.dfg_ip):
        batch_io['code_inputs'].update({'num_dfg_nodes':None, 'dfg_code_links':None, 'dfg_dfg_links':None})
        batch_io['corrupt_code_inputs'].update({'num_dfg_nodes':None, 'dfg_code_links':None, 'dfg_dfg_links':None})
    if not(args.ast_ip):
        batch_io['code_inputs'].update({'ast_paths':None, 'ast_code_links':None, 'ast_ast_sims':None})
        batch_io['corrupt_code_inputs'].update({'ast_paths':None, 'ast_code_links':None, 'ast_ast_sims':None})
    
    return batch_io


def decode_and_write_to_file(seqs, filepath, args, new_line_replacement='<NEWLINE>'):
    # seqs is a list
    if type(seqs[0])==str:
        seqs = [list(map(int, seq.split(','))) for seq in seqs]
    decoded_seqs = []
    for i in range(0, len(seqs), 1024):
        decoded_seqs += args.tokenizer.batch_decode(seqs[i:i+1024], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    decoded_seqs = [seq.replace('\n',new_line_replacement).strip()+'\n' for seq in decoded_seqs]
    decoded_seqs[-1] = decoded_seqs[-1][:-1]
    with open(filepath,'w') as f:
        f.writelines(decoded_seqs)
    
    
def prepare_io(inds, args):
    num_subset_samples = len(inds)
    device = torch.device('cpu')
    eval_batch_size = args.eval_batch_size
    io = [get_batch(inds[i:i+eval_batch_size], args) for i in range(0,len(inds),eval_batch_size)]
    sub_data = args.data[['lang','code_tokens','text_tokens']].iloc[inds[:args.num_eval_batches_bleu*args.eval_batch_size]]
    langs = list(sub_data['lang'])
    # save ground truths for blue and codebleu
    decode_and_write_to_file(list(sub_data['code_tokens']), args.output_dir+'code_output_true.txt', args)
    decode_and_write_to_file(list(sub_data['text_tokens']), args.output_dir+'text_output_true.txt', args)
    return io, langs
    
    
def update_dfg_metrics(dfg_logits, dfg_dfg_links, metrics):
    # dfg_logits : b,L-1,L-1
    # dfg_dfg_links: b,L,L
    true, pred = dfg_dfg_links[:,1:,1:].flatten(), torch.sigmoid(dfg_logits).flatten()
    ii = true!=-1
    metrics['true'].append((true[ii]).cpu())
    metrics['pred'].append((pred[ii]).cpu())
        
def update_ast_metrics(ast_logits, ast_paths, metrics):
    # ast_logits : b,L-1,max_depth,num_nodes
    # ast_paths: b,L,max_depth
    ast_paths = ast_paths[:,1:,:]
    pred_pos = (ast_paths>=0)
    ast_pred = torch.argmax(ast_logits,dim=-1)
    metrics['correct'] += ((ast_pred==ast_paths)*pred_pos).sum()
    metrics['total'] += pred_pos.sum()
    
    
def test(model, valid_io, valid_langs, args, train_step=None):
    # performance on ast and dfg tasks, bleu and codebleu for generation, 3 losses
    model.eval()
    pbar = tqdm(valid_io)
    pbar.set_description('Validation:')
    cum_loss, num_steps = 0, len(valid_io)
    dfg_metrics = {'denoising': {'true':[], 'pred':[]}, 'text2code': {'true':[], 'pred':[]}}
    ast_metrics = {'denoising':{'total':0, 'correct':0}, 'text2code':{'total':0, 'correct':0}}
    denoising_preds, text2code_preds, code2text_preds = [], [], []
    num_batches = 0
    with torch.no_grad():
        for batch_io in pbar:
            batch_io = {k:{k2:v2.to(args.device) for k2,v2 in v.items()} for k,v in batch_io.items()}
            
            # next token prediction
            ret_denoising = model(inputs=batch_io['corrupt_code_inputs'], outputs=batch_io['code_outputs']) # denoising
            ret_text2code = model(inputs=batch_io['text_inputs'], outputs=batch_io['code_outputs']) # text2code
            if args.code2text_loss_weight>0:
                ret_code2text = model(inputs=batch_io['code_inputs'], outputs=batch_io['text_outputs']) # code2text
            else:
                ret_code2text = {'lm_loss': torch.tensor(0.0)}
            losses = [[ret[k].mean().item() for k in ['lm_loss','dfg_loss','ast_loss']] for ret in [ret_denoising, ret_text2code]] \
                    + [[ret_code2text['lm_loss'].mean().item(), 0, 0]] 
            cum_loss += np.array(losses)
            
            update_dfg_metrics(ret_denoising.dfg_logits, batch_io['code_outputs']['dfg_dfg_links'], dfg_metrics['denoising'])
            update_dfg_metrics(ret_text2code.dfg_logits, batch_io['code_outputs']['dfg_dfg_links'], dfg_metrics['text2code'])
            
            update_ast_metrics(ret_denoising.ast_logits, batch_io['code_outputs']['ast_paths'], ast_metrics['denoising'])
            update_ast_metrics(ret_text2code.ast_logits, batch_io['code_outputs']['ast_paths'], ast_metrics['text2code'])
            
            # predict full output
            if num_batches<args.num_eval_batches_bleu:
                denoising_preds += list(model.module(inputs=batch_io['corrupt_code_inputs'], max_length=args.max_length_target_code+2, 
                                             num_beams=args.num_beams).cpu().numpy()) # denoising
                text2code_preds += list(model.module(inputs=batch_io['text_inputs'], max_length=args.max_length_target_code+2, 
                                             num_beams=args.num_beams).cpu().numpy()) # text2code
                if args.code2text_loss_weight>0:
                    code2text_preds += list(model.module(inputs=batch_io['code_inputs'], max_length=args.max_length_target_text+2, 
                                             num_beams=args.num_beams).cpu().numpy()) # code2text
                num_batches += 1
            
    # compute bleu and codebleu
    decode_and_write_to_file(denoising_preds, args.output_dir+'denoising_output.txt', args)
    decode_and_write_to_file(text2code_preds, args.output_dir+'text2code_output.txt', args)
    if args.code2text_loss_weight>0:
        decode_and_write_to_file(code2text_preds, args.output_dir+'code2text_output.txt', args)
    bleu_metrics = {'denoising':_bleu(args.output_dir+'code_output_true.txt', args.output_dir+'denoising_output.txt'),
                    'text2code':_bleu(args.output_dir+'code_output_true.txt', args.output_dir+'text2code_output.txt')}
    if args.code2text_loss_weight>0:
        bleu_metrics.update({'code2text':_bleu(args.output_dir+'text_output_true.txt', args.output_dir+'code2text_output.txt')})
        
    keywords_dir = 'CodeBLEU/keywords'
    codebleu_metrics = {'denoising':calc_code_bleu_multilang(args.output_dir+'code_output_true.txt', 
                                                             args.output_dir+'denoising_output.txt', valid_langs, keywords_dir),
                        'text2code':calc_code_bleu_multilang(args.output_dir+'code_output_true.txt', 
                                                             args.output_dir+'text2code_output.txt', valid_langs, keywords_dir)}
            
    if train_step is not None:
        args.logger.write('Valid-res at step '+str(train_step))
        args.logger.write('Losses:', show_time=False)
        args.logger.write(cum_loss/num_steps, show_time=False)
        args.logger.write('BLEU: '+str({k:round(v,4) for k,v in bleu_metrics.items()}), show_time=False)
        args.logger.write('CodeBLEU: ', show_time=False)
        for task, lang_dict in codebleu_metrics.items():
            args.logger.write(task+': '+str({k:[round(vi*100,4) for vi in v] for k,v in lang_dict.items()}), show_time=False)
        args.logger.write('Data Flow Prediction:', show_time=False)
        for task,mdict in dfg_metrics.items():
            mdict = {k:torch.cat(v) for k,v in mdict.items()}
            args.logger.write(task+': '+str({'AP':average_precision_score(mdict['true'], mdict['pred'])}), show_time=False)
        args.logger.write('AST Paths Prediction:', show_time=False)
        for task,mdict in ast_metrics.items():
            args.logger.write(task+': '+str({'accuracy':mdict['correct'].item()/mdict['total'].item()}), show_time=False)
        
    return {'CodeBLEU':codebleu_metrics, 'BLEU':bleu_metrics}



