import argparse
import os
import pickle
from finetune_translation_utils import prepare_code_outputs
from finetune_generation_utils import prepare_text_inputs, get_batch
import torch
from tqdm import tqdm
from pretrain_utils import decode_and_write_to_file


def parse_args():
    parser = argparse.ArgumentParser()
    
    # train or test or both
    parser.add_argument('--do_train', type=int, default=1)
    parser.add_argument('--do_test', type=int, default=1)
    
    # pretrained weights if args.do_train, finetuned weights if not
    parser.add_argument("--load_model_path", default='saved_models/pretrain/checkpoint_best_at_175000.bin', type=str)
    parser.add_argument('--model_size', type=str, default=None) 
    
    # target language
    parser.add_argument("--target_lang", default='python', type=str)  

    # max lengths
    parser.add_argument('--max_length_source_text', type=int, default=600) 
    parser.add_argument("--max_ast_depth", default=17, type=int)
    parser.add_argument('--max_length_target_code', type=int, default=512) # for get_batch(), max_length_target_code<=max_length_source_code
    
    # optimization hyperparameters
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--train_batch_size', type=int, default=20)
    parser.add_argument('--dfg_loss_weight', type=float, default=0.1) # LM loss weight is always 1
    parser.add_argument('--ast_loss_weight', type=float, default=0.1) 
    
    # testing hyperparameters
    parser.add_argument('--generate_after', type=int, default=15000) 
    parser.add_argument('--generate_every', type=int, default=10000) 
    parser.add_argument('--eval_batch_size', type=int, default=12)
    parser.add_argument('--num_generate_batches', type=int, default=2000000)
    parser.add_argument('--num_beams', type=int, default=5)
    
    # logging and other hyperparameters
    parser.add_argument('--resume', type=int, default=0) # whether to continue training with the last ckpt for this config
    parser.add_argument('--print_train_loss_every', type=int, default=100) # print train loss every __ training batches
    parser.add_argument('--checkpoint_every', type=int, default=30000) # save best model weights for every __ training batches
    parser.add_argument('--patience', type=int, default=50000) # no. of validation steps with no improvement after which to stop training
    parser.add_argument('--max_steps', type=int, default=40000)
    parser.add_argument('--seed', type=int, default=2022) # for RNGs
    parser.add_argument('--output_dir', type=str, default=None) 
    # output_dir is directory to save log file, checkpoints, etc. Set to None to automatically set this in set_output_dir()

    args = parser.parse_args()
    return args


def set_output_dir(args):
    if args.output_dir is not None:
        return
    args.output_dir = 'saved_models/apps_generation/'
    if args.load_model_path=='none':
        args.output_dir += 'codet5_'
    for argument in ['lr', 'max_length_source_text', 'max_length_target_code', 
                     'dfg_loss_weight', 'ast_loss_weight', 'num_beams', 'train_batch_size']:
        args.output_dir += argument+'_'+str(getattr(args,argument))
    args.output_dir += '/'
    os.makedirs(args.output_dir, exist_ok=True)
    
    
def read_data(args):
    data_by_split = pickle.load(open('data/apps_generation/preprocessed_data_by_split.pkl', 'rb'))
    for split, data in data_by_split.items():
        if split!='test':
            data_by_split[split] = {'text_inputs':prepare_text_inputs(data, args), 'code_outputs':prepare_code_outputs(data, args)}
        else:
            data_by_split[split] = {'text_inputs':prepare_text_inputs(data, args)}
    return data_by_split
    

def generate(model, data, args, train_step=None, split='test', outfile=None):
    args.logger.write('Generating on '+split+' set at step '+str(train_step))
    model.eval()
    num_data = len(data['text_inputs']['input_ids'])
    num_batches = 0
    preds = []
    
    with torch.no_grad():
        for start in tqdm(range(0, num_data, args.eval_batch_size)):
            batch_ind = list(range(start, min(start+args.eval_batch_size, num_data)))
            batch_io = get_batch(data, batch_ind, args)
   
            # predict full output
            if num_batches<args.num_generate_batches:
                preds += list(model.module(inputs=batch_io['text_inputs'], max_length=args.max_length_target_code+2, 
                                             num_beams=args.num_beams).cpu().numpy())
            num_batches += 1
    if outfile is None:
        outfile = args.output_dir+str(train_step)+'_output_pred.txt'
    decode_and_write_to_file(preds, outfile, args, '--<NEWLINE>--')





    







