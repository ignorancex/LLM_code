import torch
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from enc_dec import *
import random

class ReplayMemory:
  def __init__(self, capacity):
    self.memory = deque([], maxlen=capacity)

  def push(self, transition):
    self.memory.append(transition)

  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)

  def __len__(self):
    return len(self.memory)


range_key = "selected_range"

range_tensor_key = "selected_range_tensor"

def integrate_curr_program_with_prev_programs(trainer, next_program, curr_vec_ls, atom_ls, program_col_ls, next_program_str, curr_atom_str_ls, next_outbound_mask_ls):
        prev_prog_ids = atom_ls[prev_prog_key].cpu()
        curr_col_ids = atom_ls[col_key]
        outbound_mask = atom_ls[outbound_key]
        program = []
        outbound_mask_ls = []
        sample_ids = torch.arange(len(next_program[0]))
        # program length
        for pid in range(len(next_program)):
            program.append(torch.stack([next_program[pid][sample_ids, prev_prog_ids[:,k]] for k in range(prev_prog_ids.shape[-1])],dim=1))
            outbound_mask_ls.append(torch.stack([next_outbound_mask_ls[pid][sample_ids, prev_prog_ids[:,k]] for k in range(prev_prog_ids.shape[-1])], dim=-1))
        program.append(curr_vec_ls)
        outbound_mask_ls.append(outbound_mask)
        new_program_col_ls = []
        new_program_str = []
        for idx in range(len(program_col_ls)):
            curr_sample_new_program_col_ls = []
            curr_sample_new_program_str = []
            for k in range(trainer.topk_act):
                curr_new_program_col_ls = []
                curr_new_program_str = []
                # for pid in range(len(program_col_ls[idx])):
                
                #     curr_new_program_col_ls.append(program_col_ls[idx][prev_prog_ids[idx,k].item()][pid])
                #     # [k].append()
                #     curr_new_program_str.append(next_program_str[idx][prev_prog_ids[idx,k].item()][pid])
                curr_new_program_col_ls.extend(program_col_ls[idx][prev_prog_ids[idx,k].item()])
                curr_new_program_str.extend(next_program_str[idx][prev_prog_ids[idx,k].item()])
                
                
                curr_new_program_col_ls.append(curr_col_ids[idx][k])
                curr_new_program_str.append(curr_atom_str_ls[idx][k])
                curr_sample_new_program_col_ls.append(curr_new_program_col_ls)
                curr_sample_new_program_str.append(curr_new_program_str)
            new_program_col_ls.append(curr_sample_new_program_col_ls)
            new_program_str.append(curr_sample_new_program_str)
        return program, new_program_col_ls, new_program_str, outbound_mask_ls



def process_curr_atoms0(trainer, atom_ls, program, program_str, all_other_pats_ls, program_col_ls, X_pd_ls, outbound_mask_ls, other_keys=None):
    # process = psutil.Process()
    # if not trainer.do_medical:    
    curr_atom_str_ls = trainer.lang.atom_to_str_ls_full(X_pd_ls, atom_ls, col_key, op_key, pred_v_key, trainer.feat_range_mappings, trainer.train_dataset.cat_id_unique_vals_mappings, other_keys=other_keys)
    # else:
    #     curr_atom_str_ls = trainer.lang.atom_to_str_ls_full_medical(atom_ls, col_key, range_key, trainer.feat_range_mappings)
    # outbound_mask_ls = atom_ls[outbound_key]
    
    next_program = program.copy()
    
    next_outbound_mask_ls=outbound_mask_ls.copy()
    
    next_program_str = program_str.copy()
    
    curr_vec_ls = trainer.dqn.atom_to_vector_ls0(atom_ls)

    if len(program) > 0:            
        next_program, program_col_ls, next_program_str, next_outbound_mask_ls = integrate_curr_program_with_prev_programs(trainer, next_program, curr_vec_ls, atom_ls, program_col_ls, next_program_str, curr_atom_str_ls, next_outbound_mask_ls)
    else:
        next_program.append(curr_vec_ls)
        next_outbound_mask_ls.append(atom_ls[outbound_key])
        program_col_ls = []
        for vec_idx in range(len(curr_vec_ls)):
            # vec = curr_vec_ls[vec_idx]
            atom_str = curr_atom_str_ls[vec_idx]
            program_sub_col_ls = []
            next_program_str_sub_ls = []
            for k in range(len(atom_ls[col_key][vec_idx])):
                # program_col_ls[vec_idx][k].append(atom_ls[col_key][vec_idx][k])
                program_sub_col_ls.append([atom_ls[col_key][vec_idx][k]])
                next_program_str_sub_ls.append([atom_str[k]])
            next_program_str.append(next_program_str_sub_ls)
            program_col_ls.append(program_sub_col_ls)
    # if not trainer.do_medical:
        # print(process.memory_info().rss/(1024*1024*1024))
    next_all_other_pats_ls,_ = trainer.lang.evaluate_atom_ls_ls_on_dataset_full_multi(atom_ls, all_other_pats_ls, col_key, op_key, pred_v_key, other_keys=other_keys)
        # print(process.memory_info().rss/(1024*1024*1024))
    # else:
    #     next_all_other_pats_ls = trainer.lang.evaluate_atom_ls_ls_on_dataset_full_multi_medicine(atom_ls, all_other_pats_ls, col_id_key, range_key)

    return next_program, next_program_str, next_all_other_pats_ls, program_col_ls, next_outbound_mask_ls
