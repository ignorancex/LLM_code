import argparse
import os
import pickle
import numpy as np
from pretrain_utils import count_parameters, update_dfg_metrics, update_ast_metrics, decode_and_write_to_file
from modeling_structcoder import StructCoderForConditionalGeneration
import torch
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from bleu import _bleu
import sys
sys.path.append('CodeBLEU')
from calc_code_bleu import calc_code_bleu


def parse_args():
    parser = argparse.ArgumentParser()
    
    # train or test or both
    parser.add_argument('--do_train', type=int, default=1)
    parser.add_argument('--do_test', type=int, default=1)
    
    # ablation
    parser.add_argument('--model_size', type=str, default='none') 
    parser.add_argument('--dfg_ip', type=int, default=1) 
    parser.add_argument('--ast_ip', type=int, default=1) 
    
    # pretrained weights if args.do_train, finetuned weights if not
    parser.add_argument("--load_model_path", default='saved_models/pretrain/checkpoint_best_at_175000.bin', type=str)
    
    # source and target
    parser.add_argument("--source_lang", default='java', type=str)  
    parser.add_argument("--target_lang", default='cs', type=str)  

    # max lengths
    parser.add_argument('--max_length_source_code', type=int, default=320) 
    parser.add_argument("--max_source_dfg_nodes", default=65, type=int)
    parser.add_argument("--max_source_ast_leaves", default=250, type=int)
    parser.add_argument("--max_ast_depth", default=17, type=int)
    parser.add_argument('--max_length_target_code', type=int, default=256) # for get_batch(), max_length_target_code<=max_length_source_code
    
    # optimization hyperparameters
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--train_batch_size', type=int, default=25)
    parser.add_argument('--dfg_loss_weight', type=float, default=0.1) # LM loss weight is always 1
    parser.add_argument('--ast_loss_weight', type=float, default=0.1) 
    
    # testing hyperparameters
    parser.add_argument('--validate_after', type=int, default=0) # validate only after __ training steps
    parser.add_argument('--validate_every', type=int, default=500) # validate every __ training batches
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--num_eval_batches_aux', type=int, default=10000)
    parser.add_argument('--num_eval_batches_bleu', type=int, default=10000)
    parser.add_argument('--num_beams', type=int, default=10)
    
    # logging and other hyperparameters
    parser.add_argument('--resume', type=int, default=0) # whether to continue training with the last ckpt for this config
    parser.add_argument('--print_train_loss_every', type=int, default=100) # print train loss every __ training batches
    parser.add_argument('--checkpoint_every', type=int, default=15000) # save best model weights for every __ training batches
    parser.add_argument('--patience', type=int, default=1000000) # no. of validation steps with no improvement after which to stop training
    parser.add_argument('--max_steps', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=2022) # for RNGs
    parser.add_argument('--output_dir', type=str, default=None) 
    # output_dir is directory to save log file, checkpoints, etc. Set to None to automatically set this in set_output_dir()

    args = parser.parse_args()
    return args


def set_output_dir(args):
    if args.output_dir is not None:
        return
    args.output_dir = 'saved_models/codexglue_translation/'+args.source_lang+'_'+args.target_lang+'/'
    for argument in ['lr', 'max_length_source_code', 'dfg_loss_weight', 'ast_loss_weight', 'num_beams', 'train_batch_size']:
        args.output_dir += argument+'_'+str(getattr(args,argument))
    args.output_dir += '/'
    os.makedirs(args.output_dir, exist_ok=True)
    
    
# linking matrix from list of lists
def get_link_mat_from_ll(ll, num_cols):
    m = np.zeros((len(ll), num_cols))
    for i,l in enumerate(ll):
        m[i,l] = 1
    return m
    
    
# truncate to max lengths, and save as arrays or lists
def prepare_code_inputs(data, args):
    input_ids = []
    for l in data[args.source_lang+'_tokens']:
        if len(l)>2+args.max_length_source_code:
            l = l[:1+args.max_length_source_code]+[l[-1]]
        input_ids.append(l)
    code_lens_with_ends = [len(l) for l in input_ids]
    
    # Links between DFG nodes (or AST leaves) and code tokens
    dfg_code_links = []
    ast_code_links = []
    for ll_dfg, ll_ast, code_len_with_ends in zip(data[args.source_lang+'_dfg_node_code_token_idxs'],
                                                  data[args.source_lang+'_ast_leaf_code_token_idxs'], code_lens_with_ends):
        ll = [[li for li in l if li<code_len_with_ends-1] for l in ll_dfg[:args.max_source_dfg_nodes]]
        dfg_code_links.append(get_link_mat_from_ll(ll, code_len_with_ends))
        ll = [[li for li in l if li<code_len_with_ends-1] for l in ll_ast[:args.max_source_ast_leaves]]
        ast_code_links.append(get_link_mat_from_ll(ll, code_len_with_ends))
    
    # DFG adjacency matrix
    num_dfg_nodes = [len(mat) for mat in dfg_code_links]
    dfg_dfg_links = []
    for lt, n_dfg in zip(data[args.source_lang+'_dfg_edges'], num_dfg_nodes):
        curr_mat = np.zeros((n_dfg,n_dfg))
        for left, rights in lt:
            if left<n_dfg:
                rights = np.array(rights)
                curr_mat[left, rights[rights<n_dfg]] = 1
        dfg_dfg_links.append(curr_mat)
        
    # AST leaf-leaf similarities
    num_ast_leaves = [len(mat) for mat in ast_code_links]
    ast_ast_sims = []
    for sims, n_leaves in zip(data[args.source_lang+'_ll_sims'], num_ast_leaves):
        ast_ast_sims.append(np.log(1+sims[:n_leaves,:n_leaves]))
    
    # AST leaf-root paths
    ast_paths = []
    for ll, n_leaves in zip(data[args.source_lang+'_lr_paths_types'], num_ast_leaves):
        ll = [path[:args.max_ast_depth] for path in ll[:n_leaves]]
        ast_paths.append(np.array([[-1]*(args.max_ast_depth-len(path))+path for path in ll]))
            
    code_inputs= {'input_ids':input_ids, 'dfg_code_links':dfg_code_links, 'ast_code_links':ast_code_links, 
                  'dfg_dfg_links':dfg_dfg_links, 'num_dfg_nodes':num_dfg_nodes, 'ast_ast_sims':ast_ast_sims, 'ast_paths':ast_paths}
    
    return code_inputs


def prepare_code_outputs(data, args):
    input_ids = []
    for l in data[args.target_lang+'_tokens']:
        if len(l)>2+args.max_length_target_code:
            l = l[:1+args.max_length_target_code]+[l[-1]]
        input_ids.append(l)
    code_lens_with_ends = [len(l) for l in input_ids]
    
    dfg_code_links = []
    ast_code_links = []
    for ll_dfg, ll_ast, code_len_with_ends in zip(data[args.target_lang+'_dfg_node_code_token_idxs'],
                                                  data[args.target_lang+'_ast_leaf_code_token_idxs'], code_lens_with_ends):
        ll = [[li for li in l if li<code_len_with_ends-1] for l in ll_dfg]
        dfg_code_links.append(get_link_mat_from_ll(ll, code_len_with_ends))
        ll = [[li for li in l if li<code_len_with_ends-1] for l in ll_ast]
        ast_code_links.append(get_link_mat_from_ll(ll, code_len_with_ends))
    
    # DFG-DFG links
    num_dfg_nodes = [len(mat) for mat in dfg_code_links]
    dfg_dfg_links = []
    for lt, n_dfg, dfg_code_mat in zip(data[args.target_lang+'_dfg_edges'], num_dfg_nodes, dfg_code_links):
        dfg_dfg_mat = np.zeros((n_dfg,n_dfg)) # L_dfg,L_dfg
        for left, rights in lt:
            dfg_dfg_mat[left, rights] = 1
        code_dfg_code_mat = (np.matmul(np.matmul(dfg_code_mat.T, dfg_dfg_mat), dfg_code_mat)>0).astype(int) # L,L
        code_dfg_code_mat[[0,-1],:] = -1
        code_dfg_code_mat[:,[0,-1]] = -1
        dfg_dfg_links.append(code_dfg_code_mat)

    # AST paths
    ast_paths = []
    for ll, ast_code_mat in zip(data[args.target_lang+'_lr_paths_types'], ast_code_links):
        ll = [path[:args.max_ast_depth] for path in ll]
        curr_paths = np.array([[-1]*(args.max_ast_depth-len(path))+path for path in ll]) # L_ast, max_depth
        ast_code_mat = (np.cumsum(ast_code_mat,axis=0)==1) * ast_code_mat # keep only first 1 in each col
        ast_paths.append(np.matmul(ast_code_mat.T, curr_paths)) # L, max_depth
    
    code_outputs = {'input_ids':input_ids, 'dfg_dfg_links':dfg_dfg_links, 'ast_paths':ast_paths}
    return code_outputs

    
def read_data(args):
    data_by_split = pickle.load(open('data/codexglue_translation/preprocessed_data_by_split.pkl', 'rb'))
    for split, data in data_by_split.items():
        data_by_split[split] = {'code_inputs':prepare_code_inputs(data, args), 'code_outputs':prepare_code_outputs(data, args)}
    return data_by_split
    
    
def load_model(args, ft_node_types_path):
    ft_node_types = pickle.load(open(ft_node_types_path,'rb'))
    args.num_node_types = len(ft_node_types)
    model = StructCoderForConditionalGeneration(args)
    
    # load pretrained weights or finetuned weights
    if args.do_train:
        if args.load_model_path!='none':
            args.logger.write('\nLoading model from '+args.load_model_path)
            pt_dict = torch.load(args.load_model_path, map_location=torch.device('cpu'))['model_weights']
            my_dict = model.state_dict()
            for k,v in pt_dict.items():
                if k not in ['module.ast_type_emb.weight', 'module.ast_path_head.weight']:
                    my_dict[k[len('module.'):]] = v
            pt_node_types = pickle.load(open('data/pretrain/all_node_types.pkl','rb'))
            pt_node_types = {k:i for i,k in enumerate(pt_node_types)}
            my_to_pt_node_types = [-1 for i in range(len(ft_node_types))]
            for i,k in enumerate(ft_node_types):
                if k in pt_node_types:
                    my_to_pt_node_types[i] = pt_node_types[k]
                    with torch.no_grad():
                        my_dict['ast_type_emb.weight'][i,:] = pt_dict['module.ast_type_emb.weight'][pt_node_types[k], :]
                        my_dict['ast_path_head.weight'][i::len(ft_node_types), :] = \
                                        pt_dict['module.ast_path_head.weight'][pt_node_types[k]::len(pt_node_types), :]
            args.logger.write('******* No. of new node types = '+str((np.array(my_to_pt_node_types)==-1).sum())+'/'+str(len(ft_node_types)))
            model.load_state_dict(my_dict)
    else:
        model.load_state_dict(torch.load(args.load_model_path, map_location=torch.device('cpu'))['model_weights'])
    
    # print #parameters, #trainable_parameters
    args.logger.write('# parameters : '+str(count_parameters(model, only_trainable=False)))
    args.logger.write('# trainable parameters : '+str(count_parameters(model, only_trainable=True)))
    # place model on gpu
    model.to(args.device)
    if torch.cuda.device_count()>0:
        model = torch.nn.DataParallel(model)
        args.logger.write('Using '+str(torch.cuda.device_count())+' GPUs.')
    return model
    
    
def get_batch(data, batch_ind, args):
    bsz = len(batch_ind)
    
    # 1. Code inputs
    code_inputs = {}
    for k in data['code_inputs']:
        code_inputs[k] = [data['code_inputs'][k][i] for i in batch_ind]
        
    # 1.1. input_ids and attention_mask
    code_lens = np.array([len(l) for l in code_inputs['input_ids']])
    max_code_len = code_lens.max()
    code_pad_lens = max_code_len - code_lens
    code_inputs['input_ids'] = torch.LongTensor([l+[args.tokenizer.pad_token_id]*p
                                                  for l,p in zip(code_inputs['input_ids'],code_pad_lens)]).to(args.device)
    code_inputs['attention_mask'] = torch.FloatTensor([[1]*c+[0]*p for c,p in zip(code_lens,code_pad_lens)]).to(args.device)
    
    # 1.2. dfg info
    dfg_lens = np.array([len(m) for m in code_inputs['dfg_code_links']])
    max_dfg_len = dfg_lens.max()
    dfg_code_links = np.zeros((bsz,max_dfg_len,max_code_len))
    dfg_dfg_links = np.zeros((bsz,max_dfg_len,max_dfg_len))
    for i,(dc,dd,code_len,dfg_len) in enumerate(zip(code_inputs['dfg_code_links'], code_inputs['dfg_dfg_links'], code_lens, dfg_lens)):
        dfg_code_links[i,:dfg_len,:code_len] = dc
        dfg_dfg_links[i,:dfg_len,:dfg_len] = dd
    code_inputs['dfg_code_links'] = torch.LongTensor(dfg_code_links).to(args.device)
    code_inputs['dfg_dfg_links'] = torch.IntTensor(dfg_dfg_links).to(args.device)
    code_inputs['num_dfg_nodes'] = torch.IntTensor(code_inputs['num_dfg_nodes']).to(args.device)
    
    # 1.3. ast info
    ast_lens = np.array([len(m) for m in code_inputs['ast_code_links']])
    max_ast_len = ast_lens.max()
    ast_code_links = np.zeros((bsz,max_ast_len,max_code_len))
    ast_ast_sims = np.zeros((bsz,max_ast_len,max_ast_len))
    ast_paths = -np.ones((bsz,max_ast_len,args.max_ast_depth))
    for i,(ac,aa,ap,code_len,ast_len) in enumerate(zip(code_inputs['ast_code_links'], code_inputs['ast_ast_sims'], 
                                                       code_inputs['ast_paths'], code_lens, ast_lens)):
        ast_code_links[i,:ast_len,:code_len] = ac
        ast_ast_sims[i,:ast_len,:ast_len] = aa
        ast_paths[i,:ast_len,:] = ap
    code_inputs['ast_code_links'] = torch.LongTensor(ast_code_links).to(args.device)
    code_inputs['ast_ast_sims'] = torch.FloatTensor(ast_ast_sims).to(args.device)
    code_inputs['ast_paths'] = torch.LongTensor(ast_paths).to(args.device)
    
    # 2. Code outputs
    code_outputs = {}
    for k in ['input_ids', 'dfg_dfg_links', 'ast_paths']:
        code_outputs[k] = [data['code_outputs'][k][i] for i in batch_ind]
        
    # 2.1. input_ids and attention_mask
    code_lens = np.array([len(l) for l in code_outputs['input_ids']])
    max_code_len = max(code_lens)
    pad_lens = max_code_len - code_lens
    code_outputs['input_ids'] = torch.LongTensor([l+[args.tokenizer.pad_token_id]*p
                                                  for l,p in zip(code_outputs['input_ids'],pad_lens)]).to(args.device)
    code_outputs['attention_mask'] = torch.FloatTensor([[1]*c+[0]*p for c,p in zip(code_lens,pad_lens)]).to(args.device)
    
    # 2.2. dfg_dfg_links
    dfg_dfg_links = -np.ones((bsz, max_code_len,max_code_len))
    for i,(m,c) in enumerate(zip(code_outputs['dfg_dfg_links'], code_lens)):
        dfg_dfg_links[i,:c,:c] = m
    code_outputs['dfg_dfg_links'] = torch.FloatTensor(dfg_dfg_links).to(args.device)
    
    # 2.3. ast_paths
    ast_paths = -np.ones((bsz, max_code_len, args.max_ast_depth))
    for i,(m,c) in enumerate(zip(code_outputs['ast_paths'], code_lens)):
        ast_paths[i,:c,:] = m
    code_outputs['ast_paths'] = torch.LongTensor(ast_paths).to(args.device)
    
    return {'code_inputs':code_inputs, 'code_outputs':code_outputs}


def calc_xmatch(true_filepath, pred_filepath):
    true_lines = open(true_filepath, 'r').readlines()
    pred_lines = open(pred_filepath, 'r').readlines()
    return np.mean([l1==l2 for l1,l2 in zip(true_lines, pred_lines)])
    

def test(model, data, args, train_step=None, split='test'):
    # performance on ast and dfg tasks, bleu and codebleu for generation, 3 losses
    args.logger.write('Results on '+split+' set at step '+str(train_step))
    model.eval()
    num_data = len(data['code_inputs']['input_ids'])
    cum_loss, num_batches = 0, 0
    dfg_metrics = {'true':[], 'pred':[]}
    ast_metrics = {'total':0, 'correct':0}
    preds = []
    
    with torch.no_grad():
        for start in tqdm(range(0, num_data, args.eval_batch_size)):
            batch_ind = list(range(start, min(start+args.eval_batch_size, num_data)))
            batch_io = get_batch(data, batch_ind, args)
   
            # next token prediction
            if num_batches<args.num_eval_batches_aux:
                ret  = model(inputs=batch_io['code_inputs'], outputs=batch_io['code_outputs'])
                cum_loss += np.array([ret.lm_loss.mean().item(), ret.dfg_loss.mean().item(), ret.ast_loss.mean().item()])

                update_dfg_metrics(ret.dfg_logits, batch_io['code_outputs']['dfg_dfg_links'], dfg_metrics)
                update_ast_metrics(ret.ast_logits, batch_io['code_outputs']['ast_paths'], ast_metrics)
            
            # predict full output
            if num_batches<args.num_eval_batches_bleu:
                preds += list(model.module(inputs=batch_io['code_inputs'], max_length=args.max_length_target_code+2, 
                                             num_beams=args.num_beams).cpu().numpy())
            num_batches += 1
            
    # compute bleu and codebleu
    true_filepath = 'data/codexglue_translation/'+split+'_'+args.target_lang+'.txt'
    decode_and_write_to_file(preds, args.output_dir+'output_pred.txt', args, ' ')
    bleu_metric = _bleu(true_filepath, args.output_dir+'output_pred.txt')
    xmatch_metric = calc_xmatch(true_filepath, args.output_dir+'output_pred.txt')
        
    keywords_dir = 'CodeBLEU/keywords'
    target_lang = 'c_sharp' if args.target_lang=='cs' else args.target_lang
    codebleu_metric = calc_code_bleu(true_filepath, args.output_dir+'output_pred.txt', target_lang, keywords_dir)
            
    if train_step is not None:
        args.logger.write('Losses:', show_time=False)
        args.logger.write(cum_loss/num_batches, show_time=False)
        args.logger.write('BLEU: '+str(round(bleu_metric,4)), show_time=False)
        args.logger.write('xMatch: '+str(round(xmatch_metric,4)), show_time=False)
        args.logger.write('CodeBLEU: '+str([round(vi*100,4) for vi in codebleu_metric]), show_time=False)
        args.logger.write('Data Flow Prediction: '+str({'AP':average_precision_score(torch.cat(dfg_metrics['true']), 
                                                                                     torch.cat(dfg_metrics['pred']))}), show_time=False)
        args.logger.write('AST Paths Prediction: '+str({'accuracy':ast_metrics['correct'].item()/ast_metrics['total'].item()}), 
                          show_time=False)
        
    return {'CodeBLEU':codebleu_metric, 'BLEU':bleu_metric, 'xMatch':xmatch_metric}






    