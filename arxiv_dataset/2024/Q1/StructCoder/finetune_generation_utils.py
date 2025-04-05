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
from finetune_translation_utils import prepare_code_outputs, load_model, calc_xmatch


def parse_args():
    parser = argparse.ArgumentParser()
    
    # train or test or both
    parser.add_argument('--do_train', type=int, default=1)
    parser.add_argument('--do_test', type=int, default=1)
    
    # pretrained weights if args.do_train, finetuned weights if not
    parser.add_argument("--load_model_path", default='saved_models/pretrain/checkpoint_best_at_175000.bin', type=str)
    
    # ablation
    parser.add_argument('--model_size', type=str, default='none') 
    
    # target language
    parser.add_argument("--target_lang", default='java', type=str)  

    # max lengths
    parser.add_argument('--max_length_source_text', type=int, default=320) 
    parser.add_argument("--max_ast_depth", default=17, type=int)
    parser.add_argument('--max_length_target_code', type=int, default=150) # for get_batch(), max_length_target_code<=max_length_source_code
    
    # optimization hyperparameters
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--dfg_loss_weight', type=float, default=0.1) # LM loss weight is always 1
    parser.add_argument('--ast_loss_weight', type=float, default=0.1) 
    
    # testing hyperparameters
    parser.add_argument('--validate_after', type=int, default=0) # validate only after __ training steps
    parser.add_argument('--validate_every', type=int, default=3000) # validate every __ training batches
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--num_eval_batches_aux', type=int, default=20000)
    parser.add_argument('--num_eval_batches_bleu', type=int, default=20000)
    parser.add_argument('--num_beams', type=int, default=10)
    
    # logging and other hyperparameters
    parser.add_argument('--resume', type=int, default=0) # whether to continue training with the last ckpt for this config
    parser.add_argument('--print_train_loss_every', type=int, default=100) # print train loss every __ training batches
    parser.add_argument('--checkpoint_every', type=int, default=15000) # save best model weights for every __ training batches
    parser.add_argument('--patience', type=int, default=500000) # no. of validation steps with no improvement after which to stop training
    parser.add_argument('--max_steps', type=int, default=300000)
    parser.add_argument('--seed', type=int, default=2022) # for RNGs
    parser.add_argument('--output_dir', type=str, default=None) 
    # output_dir is directory to save log file, checkpoints, etc. Set to None to automatically set this in set_output_dir()

    args = parser.parse_args()
    return args


def set_output_dir(args):
    if args.output_dir is not None:
        return
    args.output_dir = 'saved_models/codexglue_generation/'
    for argument in ['lr', 'max_length_source_text', 'max_length_target_code', 
                     'dfg_loss_weight', 'ast_loss_weight', 'num_beams', 'train_batch_size']:
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
def prepare_text_inputs(data, args):
    input_ids = []
    for l in data['nl_tokens']:
        if len(l)>2+args.max_length_source_text:
            l = l[:1+args.max_length_source_text]+[l[-1]]
        input_ids.append(l)
    text_inputs= {'input_ids':input_ids}
    return text_inputs

    
def read_data(args):
    data_by_split = pickle.load(open('data/codexglue_generation/preprocessed_data_by_split.pkl', 'rb'))
    for split, data in data_by_split.items():
        data_by_split[split] = {'text_inputs':prepare_text_inputs(data, args), 'code_outputs':prepare_code_outputs(data, args)}
    return data_by_split
    
    
def get_batch(data, batch_ind, args):
    bsz = len(batch_ind)
    
    # 1. Textputs
    text_inputs = {}
    for k in data['text_inputs']:
        text_inputs[k] = [data['text_inputs'][k][i] for i in batch_ind]
        
    # 1.1. input_ids and attention_mask
    text_lens = np.array([len(l) for l in text_inputs['input_ids']])
    max_text_len = text_lens.max()
    text_pad_lens = max_text_len - text_lens
    text_inputs['input_ids'] = torch.LongTensor([l+[args.tokenizer.pad_token_id]*p
                                                  for l,p in zip(text_inputs['input_ids'],text_pad_lens)]).to(args.device)
    text_inputs['attention_mask'] = torch.FloatTensor([[1]*c+[0]*p for c,p in zip(text_lens,text_pad_lens)]).to(args.device)
    if 'code_outputs' not in data:
        return {'text_inputs':text_inputs}
    
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
    
    return {'text_inputs':text_inputs, 'code_outputs':code_outputs}
    

def test(model, data, args, train_step=None, split='test'):
    # performance on ast and dfg tasks, bleu and codebleu for generation, 3 losses
    args.logger.write('Results on '+split+' set at step '+str(train_step))
    model.eval()
    num_data = len(data['text_inputs']['input_ids'])
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
                ret  = model(inputs=batch_io['text_inputs'], outputs=batch_io['code_outputs'])
                cum_loss += np.array([ret.lm_loss.mean().item(), ret.dfg_loss.mean().item(), ret.ast_loss.mean().item()])

                update_dfg_metrics(ret.dfg_logits, batch_io['code_outputs']['dfg_dfg_links'], dfg_metrics)
                update_ast_metrics(ret.ast_logits, batch_io['code_outputs']['ast_paths'], ast_metrics)
            
            # predict full output
            if num_batches<args.num_eval_batches_bleu:
                preds += list(model.module(inputs=batch_io['text_inputs'], max_length=args.max_length_target_code+2, 
                                             num_beams=args.num_beams).cpu().numpy())
            num_batches += 1
            
    # compute bleu and codebleu
    true_filepath = 'data/codexglue_generation/'+split+'_'+args.target_lang+'.txt'
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






    