# Transition = namedtuple("Transition", ("features", "program", "action", "next_program", "reward"))
import torch
import numpy as np
import torch.nn as nn

from collections import namedtuple, deque

Transition = namedtuple("Transition", ("features", "data", "program", "action", "next_program", "reward"))

min_Q_val = -1.01

col_key = "col"

col_id_key = "col_id"

col_Q_key = "col_Q"

pred_Q_key = "pred_Q"

op_Q_key = "op_Q"

col_probs_key = "col_probs"

pred_probs_key = "pred_probs"

op_probs_key = "op_probs"

pred_v_key = "pred_v"

op_id_key = "op_id"
        
op_key = "op"

select_num_feat_key = "select_num_feat_key"



prev_prog_key = "prev_prog"

outbound_key = "outbound"

further_sel_mask_key = "further_sel_mask"

# if torch.cuda.is_available():
#     DEVICE = "cuda"
# else:
#     DEVICE = "cpu"


def atom_to_vector_ls0_main(net, atom_ls):
    ret_tensor_ls = []
    pred_v_arr = atom_ls[pred_v_key]
    
    col_id_tensor = atom_ls[col_id_key]
    
    op_id_tensor = atom_ls[op_id_key]
    
    
    
    ret_tensor_ls = torch.zeros([len(pred_v_arr), net.topk_act, net.ATOM_VEC_LENGTH])
            
    sample_id_tensor = torch.arange(len(ret_tensor_ls),device=net.device)
    
    for k in range(net.topk_act):
        ret_tensor_ls[sample_id_tensor,k, net.num_start_pos + col_id_tensor[:,k]]=1
        ret_tensor_ls[(op_id_tensor[:,k] < 2),k, net.op_start_pos + op_id_tensor[(op_id_tensor[:,k] < 2),k]]=1
        # if torch.sum(op_id_tensor[:,k] == 2) > 0:
        #     print()
        
        ret_tensor_ls[(op_id_tensor[:,k] == 2),k, net.op_start_pos]=1
        ret_tensor_ls[(op_id_tensor[:,k] == 2),k, net.op_start_pos + 1]=1
        ret_tensor_ls[(op_id_tensor[:,k] < 2).cpu(), k, net.ATOM_VEC_LENGTH-1] = torch.from_numpy(pred_v_arr)[(op_id_tensor[:,k] < 2).cpu(), k]
    # ret_tensor_ls[:, :, net.ATOM_VEC_LENGTH-1] = torch.from_numpy(pred_v_arr)

    return ret_tensor_ls

def mask_atom_representation1(X_pd_full, topk_act, num_feat_len, op_start_pos, program_ls, outbound_mask_ls, feat_pred_logit, device, init=False):
    
    op1_feat_occur_mat = torch.zeros([program_ls[0].shape[0], topk_act, num_feat_len], device = device)
    op2_feat_occur_mat = torch.zeros([program_ls[0].shape[0], topk_act, num_feat_len], device = device)
    
    
    for idx in range(len(program_ls)):
        program = program_ls[idx]
        if not init:
            outbound_mask = outbound_mask_ls[idx].type(torch.float).unsqueeze(-1)
            op1_feat_occur_mat += ((program[:,:, op_start_pos:op_start_pos+1] + outbound_mask)>=1).type(torch.float).to(device)*program[:,:,op_start_pos+2:-1].to(device)
            op2_feat_occur_mat += ((program[:,:, op_start_pos+1:op_start_pos+2] + outbound_mask)>=1).type(torch.float).to(device)*program[:,:,op_start_pos+2:-1].to(device)
        # op1_feat_occur_mat, op2_feat_occur_mat = mask_feat_by_nan_time_range(X_pd_full, op1_feat_occur_mat, op2_feat_occur_mat)
    
    feat_occur_count_mat = op1_feat_occur_mat + op2_feat_occur_mat
    
    feat_pred_probs = torch.softmax(feat_pred_logit, dim=-1) + 1e-6

    feat_pred_Q = torch.tanh(feat_pred_logit)

    if not init:
        feat_pred_probs = feat_pred_probs*(feat_occur_count_mat < 2).float()

        feat_pred_Q = feat_pred_Q*(feat_occur_count_mat < 2).float() + (min_Q_val)*(feat_occur_count_mat >= 2).float()
    else:
        feat_pred_probs = feat_pred_probs*(feat_occur_count_mat[:,0] < 2).float()

        feat_pred_Q = feat_pred_Q*(feat_occur_count_mat[:,0] < 2).float() + (min_Q_val)*(feat_occur_count_mat[:,0] >= 2).float()
    
    further_selection_masks = torch.ones(feat_occur_count_mat.shape[0]).bool()#(torch.sum(feat_occur_count_mat.view(feat_occur_count_mat.shape[0], -1) < 2, dim=-1) >= topk_act)
    return feat_pred_probs, feat_pred_Q, op1_feat_occur_mat, op2_feat_occur_mat, further_selection_masks


def forward_main0(net, hx, eval, epsilon, program, outbound_mask_ls, atom, pat_count, X_pd_full, init=False, train=False, X_pd_full2=None):
    if not eval:
        if np.random.rand() < epsilon: # and not is_ppo:
            if net.feat_group_names is None:
                output_num = net.num_feats
            else:
                output_num = net.feat_group_num

            if init:
                selected_feat_logit = torch.rand([pat_count, output_num], device=net.device)
            else:
                selected_feat_logit = torch.rand([pat_count,net.topk_act, output_num], device=net.device)

        else:
            selected_feat_logit = net.feat_selector(hx)
    
    else:
        selected_feat_logit = net.feat_selector(hx)

    if net.feat_group_names is None:
        selected_feat_probs, selected_Q_feat, op1_feat_occur_mat, op2_feat_occur_mat = mask_atom_representation1(net.topk_act, net.num_feats, net.op_start_pos, program, outbound_mask_ls, selected_feat_logit, device=net.device, init=init)
    else:
        selected_feat_probs, selected_Q_feat, op1_feat_occur_mat, op2_feat_occur_mat = mask_atom_representation1(net.topk_act, net.feat_group_num, net.op_start_pos, program, outbound_mask_ls, selected_feat_logit, device=net.device, init=init)

    # if len(net.removed_feat_ls) > 0:
    #     selected_Q_feat, selected_feat_probs = down_weight_removed_feats(net, selected_Q_feat, selected_feat_probs, net.removed_feat_ls)

    if not eval:
        if init:
            _, selected_feat_col = torch.topk(selected_feat_probs, k=net.topk_act, dim=-1)
        else:
            # selected_feat_col = torch.argmax(selected_feat_probs, dim=-1)
            _,selected_feat_col = torch.topk(selected_feat_probs.view(len(selected_feat_probs),-1), k=net.topk_act, dim=-1)
        # else:
        #     if train:
        #         selected_feat_col = torch.multinomial(selected_feat_probs.view(len(selected_feat_probs),-1), net.topk_act, replacement=False)
        #     else:
        #         # selected_feat_col = torch.argmax(selected_feat_probs, dim=-1)
        #         _,selected_feat_col = torch.topk(selected_feat_probs.view(len(selected_feat_probs),-1), k=net.topk_act, dim=-1)

    else:
        selected_feat_col = atom[col_id_key]

    selected_feat_col_onehot = torch.zeros_like(selected_feat_probs)
    prev_program_ids = None
    curr_selected_feat_col = None

    if init:
        selected_feat_col_onehot = selected_feat_col_onehot.unsqueeze(1).repeat(1, net.topk_act, 1)
        for k in range(net.topk_act):
            selected_feat_col_onehot[torch.arange(len(selected_feat_col_onehot)), k, selected_feat_col[:,k]]=1
        hx = torch.cat([hx.unsqueeze(1).repeat(1, net.topk_act, 1), selected_feat_probs.unsqueeze(1)*selected_feat_col_onehot], dim=-1)
    else:
        if not eval:
            if net.feat_group_names is None:
                prev_program_ids = torch.div(selected_feat_col, net.num_feats, rounding_mode='floor')
                curr_selected_feat_col = selected_feat_col%net.num_feats
            else:
                prev_program_ids = torch.div(selected_feat_col, net.feat_group_num, rounding_mode='floor')
                curr_selected_feat_col = selected_feat_col%net.feat_group_num
        else:
            prev_program_ids = atom[prev_prog_key]
            curr_selected_feat_col = atom[col_id_key]
        new_hx = []
        seq_ids = torch.arange(pat_count)
        for k in range(net.topk_act):
            selected_feat_col_onehot[seq_ids, prev_program_ids[:,k], curr_selected_feat_col[:,k]]=1
            new_hx.append(torch.cat([hx[seq_ids, prev_program_ids[:,k]], selected_feat_probs[seq_ids,prev_program_ids[:,k]]*selected_feat_col_onehot[seq_ids, prev_program_ids[:,k]]],dim=-1))

        hx = torch.stack(new_hx, dim=1)    

    if not eval:
        if np.random.rand() < epsilon:# and not is_ppo:
            pred = torch.rand([pat_count, net.topk_act, net.discretize_feat_value_count], device=net.device)
        else:
            pred = net.token_nets["constant"](hx)
    else:
        pred = net.token_nets["constant"](hx)

    # if not eval:
    #     if np.random.rand() < epsilon and not is_ppo:
    #     else:
    #         selected_op_logit = net.op_selector(hx)    
    # else:
    #     selected_op_logit = net.op_selector(hx)
    
    selected_col_ls = []
    selected_feat_point_ls = []
    selected_num_col_boolean_ls = []
    if init:
        selected_feat_col_ls = selected_feat_col.cpu().tolist()

        

        for idx in range(len(selected_feat_col_ls)):
            if net.feat_group_names is None:
                selected_col_ls.append([net.grammar_num_to_token_val['num_feat'][selected_feat_col_ls[idx][k]] if selected_feat_col_ls[idx][k]< net.num_feat_len else net.grammar_num_to_token_val['cat_feat'][selected_feat_col_ls[idx][k] - net.num_feat_len]  for k in range(len(selected_feat_col_ls[idx]))])
                selected_num_col_boolean_ls.append([1 if selected_feat_col_ls[idx][k] < net.num_feat_len else 0 for k in range(len(selected_feat_col_ls[idx]))])
                if net.feat_bound_point_ls is not None:
                    selected_feat_point_ls.append([net.feat_bound_point_ls[col] if col in net.grammar_token_val_to_num['num_feat'] else [0]*len(list(net.feat_bound_point_ls.values())[0]) for col in selected_col_ls[-1]])
            else:
                selected_col_ls.append([net.feat_group_names[selected_feat_col_ls[idx][k]][0] for k in range(len(selected_feat_col_ls[idx]))])
                if net.feat_bound_point_ls is not None:
                    selected_feat_point_ls.append([net.feat_bound_point_ls[col] for col in selected_col_ls[-1]])
    else:
        selected_feat_col_ls = curr_selected_feat_col.cpu().tolist()
        for idx in range(len(prev_program_ids)):
            if net.feat_group_names is None:
                selected_col_ls.append([net.grammar_num_to_token_val['num_feat'][selected_feat_col_ls[idx][k]] if selected_feat_col_ls[idx][k] < net.num_feat_len else net.grammar_num_to_token_val['cat_feat'][selected_feat_col_ls[idx][k] - net.num_feat_len] for k in range(len(selected_feat_col_ls[idx]))])
                selected_num_col_boolean_ls.append([1 if selected_feat_col_ls[idx][k] < net.num_feat_len else 0 for k in range(len(selected_feat_col_ls[idx]))])
                if net.feat_bound_point_ls is not None:
                    selected_feat_point_ls.append([net.feat_bound_point_ls[col] if col in net.grammar_token_val_to_num['num_feat'] else [0]*len(list(net.feat_bound_point_ls.values())[0]) for col in selected_col_ls[-1]])
            else:
                selected_col_ls.append([net.feat_group_names[selected_feat_col_ls[idx][k]][0] for k in range(len(selected_feat_col_ls[idx]))])
                if net.feat_bound_point_ls is not None:
                    selected_feat_point_ls.append([net.feat_bound_point_ls[col] for col in selected_col_ls[-1]])
    
    selected_num_feat_tensors_bool = torch.tensor(selected_num_col_boolean_ls).to(net.device)            
    if net.feat_bound_point_ls is not None:
        selected_feat_point_tensors = torch.tensor(selected_feat_point_ls,dtype=torch.float).to(net.device)
        
        selected_feat_point_tensors_min = selected_feat_point_tensors[:,:,0]
        selected_feat_point_tensors_max = selected_feat_point_tensors[:,:,-1]
    else:
        selected_feat_point_tensors = None
        selected_feat_point_tensors_min = None
        selected_feat_point_tensors_max = None
    feat_val = []
    for idx in range(len(selected_col_ls)):
        curr_feat_val = torch.tensor([X_pd_full2[idx][selected_feat_col_ls[idx][k]].item() for k in range(len(selected_col_ls[idx]))], dtype=torch.float).to(net.device)
        feat_val.append(curr_feat_val)

    feat_val = torch.stack(feat_val, dim=0)


    if not eval:
        pred_out_mask =  mask_atom_representation_for_op0(net, net.topk_act, op1_feat_occur_mat, op2_feat_occur_mat, selected_feat_col, pred, prev_program_ids, curr_selected_feat_col, feat_val, selected_feat_point_tensors, selected_feat_point_tensors_min, selected_feat_point_tensors_max, init=init)
    else:
        pred_out_mask =  mask_atom_representation_for_op0(net, net.topk_act, op1_feat_occur_mat, op2_feat_occur_mat, selected_feat_col, pred, atom[prev_prog_key], atom[col_id_key], feat_val,selected_feat_point_tensors, selected_feat_point_tensors_min, selected_feat_point_tensors_max, init=init)

    pred_probs_vals = (torch.softmax(pred,dim=-1) + 1e-5)*pred_out_mask
    pred_Q_vals = torch.tanh(pred)*pred_out_mask + (min_Q_val)*(1 - pred_out_mask)
    # if net.prefer_smaller_range:
        
    #     regularized_coeff = [torch.exp(-net.prefer_smaller_range_coeff*(feat_val[:,k].view(-1,1) - net.selected_vals.view(1,-1))**2) for k in range(net.topk_act)]
    #     regularized_coeff = torch.stack(regularized_coeff, dim=1)
    #     pred_probs_vals = pred_probs_vals*regularized_coeff
    #     pred_Q_vals = torch.tanh(pred)*pred_out_mask*regularized_coeff + (min_Q_val)*(1 - pred_out_mask*regularized_coeff)
    # pred_Q_vals = pred_Q_vals*selected_num_feat_tensors_bool + 0*(1-selected_num_feat_tensors_bool)
    # if not eval:
    #     if not is_ppo:
    #         selected_op = torch.argmax(selected_op_probs, dim=-1)
    #     else:
    #         if train:
    #             dist = torch.distributions.Categorical(selected_op_probs)
    #             selected_op = dist.sample()
    #         else:
    #             selected_op = torch.argmax(selected_op_probs, dim=-1)
    # else:
    #     selected_op = atom[op_id_key]
    
    # selected_op_onehot = torch.zeros_like(selected_op_probs)
    # for k in range(net.topk_act):
    #     selected_op_onehot[torch.arange(len(selected_op_probs)), k, selected_op[:,k]]=1
    # hx = torch.cat([hx, selected_op_probs*selected_op_onehot], dim=-1)
     
    ret = {}
    

    

        # pred_lb = pred_lb*(range_max - range_min) + range_min
    if not eval:
        


        # selected_op_probs = selected_op_probs/torch.sum(selected_op_probs, dim=-1).unsqueeze(-1)

        # op_val = (selected_op_probs > 0.5).data.cpu().numpy().astype(float)
        
        
        # if not is_ppo:
        argmax = torch.argmax(pred_probs_vals,dim=-1)
        
        
        if net.feat_bound_point_ls is None:
            pred_v = argmax/(net.discretize_feat_value_count-1)
        else:
            pred_v = torch.stack([selected_feat_point_tensors[torch.arange(len(selected_feat_point_tensors)),k,argmax[:,k]] for k in range(argmax.shape[-1])], dim=1)
            
        # pred_v = pred_v*selected_cat_feat_tensors + feat_val*(1-selected_cat_feat_tensors)
        # # __ge__
        # pred_v1 = (feat_val)*(argmax/(discretize_feat_value_count-1))
        # # __le__
        # pred_v2 = (1 - feat_val)*(argmax/(discretize_feat_value_count-1)) + feat_val
        # else:
        #     if not net.continue_act:
        #         if train:
        #             dist = torch.distributions.Categorical(torch.softmax(pred, dim=-1))
        #             argmax = dist.sample()
        #         else:
        #             argmax = torch.argmax(pred, dim=-1)

        #         # argmax = argmax.cpu().numpy()
        #         if net.feat_bound_point_ls is None:
        #             pred_v = argmax/(net.discretize_feat_value_count-1)
        #         else:
        #             pred_v = torch.stack([selected_feat_point_tensors[torch.arange(len(selected_feat_point_tensors)),k,argmax[:,k]] for k in range(argmax.shape[-1])], dim=1)
        #         # # __ge__
        #         # pred_v1 = (feat_val)*(argmax/(discretize_feat_value_count-1))
        #         # # __le__
        #         # pred_v2 = (1 - feat_val)*(argmax/(discretize_feat_value_count-1)) + feat_val
        #     else:
        #         pred = torch.clamp(pred, min=1e-6, max=1)
        #         if train:
                    
        #             dist = torch.distributions.normal.Normal(pred[:,:,0], 1e-3)

        #             # dist = torch.distributions.Categorical(torch.softmax(pred, dim=-1))
        #             argmax = torch.clamp(dist.sample(), min=0, max=1)
        #         else:
        #             argmax = pred[:,:,0]
        #         # argmax = argmax.cpu().numpy()
        #         if net.feat_bound_point_ls is None:
        #             pred_v = argmax/(net.discretize_feat_value_count-1)
        #         else:
        #             pred_v = torch.stack([selected_feat_point_tensors[torch.arange(len(selected_feat_point_tensors)),k,argmax[:,k]] for k in range(argmax.shape[-1])], dim=1)

        #         # # __ge__
        #         # pred_v1 = (feat_val)*(argmax)
        #         # # __le__
        #         # pred_v2 = (1 - feat_val)*(argmax) + feat_val
                
        if net.feat_bound_point_ls is None:
            outbound_mask = (feat_val >= 1)
            outbound_mask = torch.logical_or(outbound_mask, (feat_val <= 0))
            # pred_v[(pred_v >= 1) & (feat_val >= 1)] = feat_val[(pred_v >= 1) & (feat_val >= 1)] + 1e-5
            # pred_v[(pred_v <= 0) & (feat_val <= 0)] = feat_val[(pred_v <= 0) & (feat_val <= 0)] - 1e-5
        else:
            outbound_mask = (feat_val >= selected_feat_point_tensors_max)
            outbound_mask = torch.logical_or(outbound_mask, (feat_val <= selected_feat_point_tensors_min))
            
            # pred_v[(pred_v >= selected_feat_point_tensors_max) & (feat_val >= selected_feat_point_tensors_max)] = feat_val[(pred_v >= selected_feat_point_tensors_max) & (feat_val >= selected_feat_point_tensors_max)] + 1e-5
            # pred_v[(pred_v <= selected_feat_point_tensors_min) & (feat_val <= selected_feat_point_tensors_min)] = feat_val[(pred_v <= selected_feat_point_tensors_min) & (feat_val <= selected_feat_point_tensors_min)] - 1e-5
        
        # if len(torch.nonzero(selected_num_feat_tensors_bool == 0)) > 0:
        #     print()
        
        pred_v = pred_v*selected_num_feat_tensors_bool + feat_val*(1-selected_num_feat_tensors_bool)
        
        selected_op = (pred_v <= feat_val).type(torch.long)*selected_num_feat_tensors_bool + 2*(1-selected_num_feat_tensors_bool)

        selected_op_ls = []
            
        selected_op_id_ls = selected_op.cpu().tolist()

        for idx in range(len(selected_op_id_ls)):
            selected_op_ls.append([net.grammar_num_to_token_val['num_op'][selected_op_id_ls[idx][k]] if selected_op_id_ls[idx][k] < 2 else net.grammar_num_to_token_val['cat_op'][0] for k in range(len(selected_op_id_ls[idx]))])


        # if net.lang.precomputed is not None:
        #     pred_v1, pred_v2 = find_nearest_thres_vals(net.lang.precomputed, selected_col_ls, pred_v1, pred_v2)
        # if self.op_list[0] == operator.__ge__:     
        
            
        # pred_v = pred_v1*op_val[:,:, 1] + pred_v2*op_val[:,:, 0]
    
    if init:
        ret[col_id_key] = selected_feat_col
    else:
        ret[col_id_key] = curr_selected_feat_col

    ret[select_num_feat_key] = selected_num_feat_tensors_bool

    if eval:
        ret[pred_Q_key] = pred_Q_vals# torch.tanh(pred)
        ret[col_Q_key] = selected_Q_feat
        # ret[op_Q_key] = selected_Q_op
        ret[prev_prog_key] = prev_program_ids
        
        ret[col_probs_key] = selected_feat_probs
        # ret[op_probs_key] = selected_op_probs
        # if net.continue_act and is_ppo:
        #     ret[pred_probs_key] = pred_probs_vals
        # else:
        ret[pred_probs_key] = pred_probs_vals#torch.softmax(pred, dim=-1)
    else:
        ret[pred_Q_key] = pred_Q_vals.data#torch.tanh(pred).data
        ret[col_Q_key] = selected_Q_feat.data
        # ret[op_Q_key] = selected_Q_op.data
        
        ret[col_probs_key] = selected_feat_probs.data
        # ret[op_probs_key] = selected_op_probs.data
        # if net.continue_act and is_ppo:
        #     ret[pred_probs_key] = pred.data
        # else:
        ret[pred_probs_key] = torch.softmax(pred, dim=-1).data

        ret[pred_v_key] = pred_v.data.cpu().numpy()
    
        ret[op_key] = selected_op_ls        
        
        ret[op_id_key] = selected_op.data
        
        ret[col_key] = selected_col_ls
        ret[prev_prog_key] = prev_program_ids

        ret[outbound_key] = outbound_mask.cpu()
    
    return ret

def mask_atom_representation_for_op0(net, topk_act, op1_feat_occur_mat, op2_feat_occur_mat, selected_feat_idx, pred, prev_program_ids, curr_selected_feat_col, feat_val, selected_feat_point_tensors, selected_feat_point_tensors_min, selected_feat_point_tensors_max, init=False):
        
    assert len(torch.unique(op1_feat_occur_mat)) <= 2
    assert len(torch.unique(op2_feat_occur_mat)) <= 2
    
    
    # op_pred_probs = torch.softmax(op_pred_logic ,dim=-1) + 1e-6

    # op_pred_Q = torch.tanh(op_pred_logic)
    pred_out_mask = torch.ones_like(pred)
    

    if not init:
        mask_ls = []

        for k in range(topk_act):
            mask_ls.append(torch.stack([op1_feat_occur_mat[torch.arange(len(op1_feat_occur_mat), device=net.device),prev_program_ids[:,k],curr_selected_feat_col[:,k]], op2_feat_occur_mat[torch.arange(len(op2_feat_occur_mat), device=net.device),prev_program_ids[:,k],curr_selected_feat_col[:,k]]],dim=-1))

    # mask = mask.unsqueeze(1).repeat(1,self.topk_act, 1)
        mask = torch.stack(mask_ls, dim=1)

        # pred_vals = pred

        # op_pred_probs = op_pred_probs*(1-mask)
        pred_out_mask = determine_pred_ids_by_val(net, feat_val, pred, mask[:,:,0], mask[:,:,1], selected_feat_point_tensors, selected_feat_point_tensors_min, selected_feat_point_tensors_max)
        


        # op_pred_Q = op_pred_Q*(1-mask) + (min_Q_val)*mask
    return pred_out_mask

def determine_pred_ids_by_val(net, feat_val, pred, op1_mask, op2_mask, selected_feat_point_tensors, selected_feat_point_tensors_min, selected_feat_point_tensors_max):
    
    if selected_feat_point_tensors is None:
        pred_candidate_vals = torch.zeros_like(pred)
        for k in range(net.discretize_feat_value_count):
            pred_candidate_vals[:, :, k] = 1/(net.discretize_feat_value_count-1)*k
    else:
        pred_candidate_vals = selected_feat_point_tensors

    pred_out_mask = (pred_candidate_vals > feat_val.unsqueeze(-1))*(1-op1_mask.unsqueeze(-1)) + (pred_candidate_vals <= feat_val.unsqueeze(-1))*(1-op2_mask.unsqueeze(-1))

    return (pred_out_mask >= 1).type(torch.float)#, torch.logical_or(feat_val < selected_feat_point_tensors_min, feat_val > selected_feat_point_tensors_max)


def forward_main0_opt(net, hx, eval, epsilon, program, outbound_mask_ls, atom, pat_count, X_pd_full, init=False, train=False, abnormal_info=None):
    if abnormal_info is not None:
        abnormal_feature_indicator, activated_indicator = abnormal_info
        activated_indicator = activated_indicator.to(net.device)
        abnormal_feature_indicator = abnormal_feature_indicator.to(net.device)
        
    else:
        abnormal_feature_indicator, activated_indicator = None, None
    
    if not eval:
        if np.random.rand() < epsilon:# and not is_ppo:
            # pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[col])], device=net.device)
            if net.feat_group_names is None:
                output_num = net.num_feats
            else:
                output_num = net.feat_group_num

            if init:
                selected_feat_logit = torch.rand([pat_count, output_num], device=net.device)
            else:
                selected_feat_logit = torch.rand([pat_count,net.topk_act, output_num], device=net.device)

        else:
            selected_feat_logit = net.feat_selector(hx)
    
    else:
        selected_feat_logit = net.feat_selector(hx)

    if net.feat_group_names is None:
        selected_feat_probs, selected_Q_feat, op1_feat_occur_mat, op2_feat_occur_mat, further_selection_masks = mask_atom_representation1(X_pd_full, net.topk_act, net.num_feats, net.op_start_pos, program, outbound_mask_ls, selected_feat_logit, init=init, device=net.device)
    else:
        selected_feat_probs, selected_Q_feat, op1_feat_occur_mat, op2_feat_occur_mat, further_selection_masks = mask_atom_representation1(X_pd_full, net.topk_act, net.feat_group_num, net.op_start_pos, program, outbound_mask_ls, selected_feat_logit, init=init)

    # if len(net.removed_feat_ls) > 0:
    #     selected_Q_feat, selected_feat_probs = down_weight_removed_feats(net, selected_Q_feat, selected_feat_probs, net.removed_feat_ls)
    # if abnormal_feature_indicator is not None:
    #     if init:
    #         selected_feat_probs = down_weight_features_not_abnormal(selected_feat_probs, abnormal_feature_indicator)
    #     else:
    #         selected_feat_probs = down_weight_features_not_abnormal(selected_feat_probs, abnormal_feature_indicator.unsqueeze(1).repeat(1, net.topk_act, 1))
    
    if not eval:
        selected_feat_col = torch.ones([len(selected_feat_probs), net.topk_act], dtype=torch.long, device=net.device)*(-1)
        # if not is_ppo:
        if init:
            _, sub_selected_feat_col = torch.topk(selected_feat_probs[further_selection_masks], k=net.topk_act, dim=-1)
        else:
            # selected_feat_col = torch.argmax(selected_feat_probs, dim=-1)
            _,sub_selected_feat_col = torch.topk(selected_feat_probs[further_selection_masks].view(len(selected_feat_probs[further_selection_masks]),-1), k=net.topk_act, dim=-1)
        # else:
        #     if train:
        #         sub_selected_feat_col = torch.multinomial(selected_feat_probs[further_selection_masks].view(len(selected_feat_probs[further_selection_masks]),-1), net.topk_act, replacement=False)
        #     else:
        #         # selected_feat_col = torch.argmax(selected_feat_probs, dim=-1)
        #         _,sub_selected_feat_col = torch.topk(selected_feat_probs[further_selection_masks].view(len(selected_feat_probs[further_selection_masks]),-1), k=net.topk_act, dim=-1)
        selected_feat_col[further_selection_masks] = sub_selected_feat_col
    else:
        selected_feat_col = atom[col_id_key]
        further_selection_masks = atom[further_sel_mask_key]

    selected_feat_col_onehot = torch.zeros_like(selected_feat_probs)
    prev_program_ids = None
    curr_selected_feat_col = None

    if init:
        selected_feat_col_onehot = selected_feat_col_onehot.unsqueeze(1).repeat(1, net.topk_act, 1)
        for k in range(net.topk_act):
            selected_feat_col_onehot[torch.arange(len(selected_feat_col_onehot)), k, selected_feat_col[:,k]]=1
        hx = torch.cat([hx.unsqueeze(1).repeat(1, net.topk_act, 1), selected_feat_probs.unsqueeze(1)*selected_feat_col_onehot], dim=-1)
    else:
        if not eval:
            curr_selected_feat_col = torch.zeros_like(selected_feat_col)
            if net.feat_group_names is None:
                prev_program_ids = torch.div(selected_feat_col[further_selection_masks], net.num_feats, rounding_mode='floor')
                sub_curr_selected_feat_col = selected_feat_col[further_selection_masks]%net.num_feats
            else:
                prev_program_ids = torch.div(selected_feat_col[further_selection_masks], net.feat_group_num, rounding_mode='floor')
                sub_curr_selected_feat_col = selected_feat_col[further_selection_masks]%net.feat_group_num
            curr_selected_feat_col[further_selection_masks] = sub_curr_selected_feat_col
        else:
            prev_program_ids = atom[prev_prog_key]
            curr_selected_feat_col = atom[col_id_key]
        new_hx = []
        seq_ids = torch.arange(pat_count)
        for k in range(net.topk_act):
            # selected_feat_col_onehot[seq_ids, prev_program_ids[:,k], curr_selected_feat_col[:,k]]=1
            selected_feat_col_onehot[further_selection_masks, prev_program_ids[:,k], curr_selected_feat_col[:,k]]=1
            curr_added_hx = selected_feat_probs[further_selection_masks,prev_program_ids[:,k]]*selected_feat_col_onehot[further_selection_masks, prev_program_ids[:,k]]
            curr_new_hx = torch.zeros([len(selected_feat_probs), curr_added_hx.shape[-1]], dtype=torch.float, device=net.device)
            curr_new_hx[further_selection_masks] = curr_added_hx
            new_hx.append(torch.cat([hx[seq_ids, prev_program_ids[:,k]], curr_new_hx],dim=-1))

        hx = torch.stack(new_hx, dim=1)    

    if not eval:
        if np.random.rand() < epsilon: # and not is_ppo:
            # pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[col])], device=net.device)
            pred = torch.rand([pat_count, net.topk_act, net.discretize_feat_value_count], device=net.device)
        else:
            pred = net.token_nets["constant"](hx)
    else:
        pred = net.token_nets["constant"](hx)

    # if not eval:
    #     if np.random.rand() < epsilon and not is_ppo:
    #         # pred = torch.rand([pat_count, len(self.grammar_num_to_token_val[col])], device=net.device)
    #         selected_op_logit = torch.rand([pat_count, net.topk_act, net.op_num], device=net.device)
    #     else:
    #         selected_op_logit = net.op_selector(hx)    
    # else:
    #     selected_op_logit = net.op_selector(hx)
    
    selected_col_ls = []
    selected_feat_point_ls = []
    selected_num_col_boolean_ls = []
    if init:
        selected_feat_col_ls = selected_feat_col.cpu().tolist()

        

        for idx in range(len(selected_feat_col_ls)):
            if further_selection_masks[idx]:
                if net.feat_group_names is None:
                    selected_col_ls.append([net.grammar_num_to_token_val['num_feat'][selected_feat_col_ls[idx][k]] if selected_feat_col_ls[idx][k]< net.num_feat_len else net.grammar_num_to_token_val['cat_feat'][selected_feat_col_ls[idx][k] - net.num_feat_len]  for k in range(len(selected_feat_col_ls[idx]))])
                    selected_num_col_boolean_ls.append([1 if selected_feat_col_ls[idx][k] < net.num_feat_len else 0 for k in range(len(selected_feat_col_ls[idx]))])
                    if net.feat_bound_point_ls is not None:
                        selected_feat_point_ls.append([net.feat_bound_point_ls[col] if col in net.grammar_token_val_to_num['num_feat'] else [0]*len(list(net.feat_bound_point_ls.values())[0]) for col in selected_col_ls[-1]])
                else:
                    selected_col_ls.append([net.feat_group_names[selected_feat_col_ls[idx][k]][0] for k in range(len(selected_feat_col_ls[idx]))])
                    if net.feat_bound_point_ls is not None:
                        selected_feat_point_ls.append([net.feat_bound_point_ls[col] for col in selected_col_ls[-1]])
            else:
                selected_col_ls.append([])
                selected_num_col_boolean_ls.append([])
                selected_feat_point_ls.append([])
    else:
        selected_feat_col_ls = curr_selected_feat_col.cpu().tolist()
        for idx in range(len(prev_program_ids)):
            if further_selection_masks[idx]:
                if net.feat_group_names is None:
                    selected_col_ls.append([net.grammar_num_to_token_val['num_feat'][selected_feat_col_ls[idx][k]] if selected_feat_col_ls[idx][k] < net.num_feat_len else net.grammar_num_to_token_val['cat_feat'][selected_feat_col_ls[idx][k] - net.num_feat_len] for k in range(len(selected_feat_col_ls[idx]))])
                    selected_num_col_boolean_ls.append([1 if selected_feat_col_ls[idx][k] < net.num_feat_len else 0 for k in range(len(selected_feat_col_ls[idx]))])
                    if net.feat_bound_point_ls is not None:
                        selected_feat_point_ls.append([net.feat_bound_point_ls[col] if col in net.grammar_token_val_to_num['num_feat'] else [0]*len(list(net.feat_bound_point_ls.values())[0]) for col in selected_col_ls[-1]])
                else:
                    selected_col_ls.append([net.feat_group_names[selected_feat_col_ls[idx][k]][0] for k in range(len(selected_feat_col_ls[idx]))])
                    if net.feat_bound_point_ls is not None:
                        selected_feat_point_ls.append([net.feat_bound_point_ls[col] for col in selected_col_ls[-1]])
            else:
                selected_col_ls.append([])
                selected_num_col_boolean_ls.append([])
                selected_feat_point_ls.append([])
    
    selected_num_feat_tensors_bool = torch.tensor(selected_num_col_boolean_ls).to(net.device)            
    if net.feat_bound_point_ls is not None:
        selected_feat_point_tensors = torch.tensor(selected_feat_point_ls,dtype=torch.float).to(net.device)
        
        selected_feat_point_tensors_min = selected_feat_point_tensors[:,:,0]
        selected_feat_point_tensors_max = selected_feat_point_tensors[:,:,-1]
    else:
        selected_feat_point_tensors = None
        selected_feat_point_tensors_min = None
        selected_feat_point_tensors_max = None
    feat_val = []
    for idx in range(len(selected_col_ls)):
        # feat_val.append(torch.tensor([X_pd_full[idx][selected_col_ls[idx][k]] for k in range(len(selected_col_ls[idx]))], dtype=torch.float).to(net.device))
        if further_selection_masks[idx]:
            feat_val.append(torch.tensor([X_pd_full[idx][selected_feat_col_ls[idx][k]] for k in range(len(selected_feat_col_ls[idx]))], dtype=torch.float).to(net.device))
        else:
            feat_val.append(torch.tensor([-1 for _ in range(net.topk_act)], dtype=torch.float).to(net.device))

    feat_val = torch.stack(feat_val, dim=0)


    if not eval:
        pred_out_mask =  mask_atom_representation_for_op0(net, net.topk_act, op1_feat_occur_mat, op2_feat_occur_mat, selected_feat_col, pred, prev_program_ids, curr_selected_feat_col, feat_val, selected_feat_point_tensors, selected_feat_point_tensors_min, selected_feat_point_tensors_max, init=init)
    else:
        pred_out_mask =  mask_atom_representation_for_op0(net, net.topk_act, op1_feat_occur_mat, op2_feat_occur_mat, selected_feat_col, pred, atom[prev_prog_key], atom[col_id_key], feat_val,selected_feat_point_tensors, selected_feat_point_tensors_min, selected_feat_point_tensors_max, init=init)

    pred_probs_vals = (torch.softmax(pred,dim=-1) + 1e-5)*pred_out_mask
    pred_Q_vals = torch.tanh(pred)*pred_out_mask + (min_Q_val)*(1 - pred_out_mask)
    # if net.prefer_smaller_range:
        
    #     regularized_coeff = [torch.exp(-net.prefer_smaller_range_coeff*(feat_val[:,k].view(-1,1) - net.selected_vals.view(1,-1))**2) for k in range(net.topk_act)]
    #     regularized_coeff = torch.stack(regularized_coeff, dim=1)
    #     pred_probs_vals = pred_probs_vals*regularized_coeff
    #     pred_Q_vals = torch.tanh(pred)*pred_out_mask*regularized_coeff + (min_Q_val)*(1 - pred_out_mask*regularized_coeff)
    
    # if abnormal_feature_indicator is not None and activated_indicator is not None:
    #     if not init:
    #         activated_indicator_given_cols = select_sub_tensors_for_each_conjunction(net, activated_indicator, curr_selected_feat_col)
    #     else:
    #         activated_indicator_given_cols = select_sub_tensors_for_each_conjunction(net, activated_indicator, selected_feat_col)
    #     # activated_indicator_given_cols = activated_indicator[torch.arange(len(activated_indicator)), selected_feat_col.view(-1)]
    #     # activated_indicator_given_cols = activated_indicator_given_cols.unsqueeze(1).repeat(1, net.topk_act, 1)
    #     pred_probs_vals = down_weight_features_not_abnormal(pred_probs_vals, activated_indicator_given_cols)
        # pred_probs_vals = pred_probs_vals*activated_indicator + 1e-5*(1 - activated_indicator)
        
    # pred_Q_vals = pred_Q_vals*selected_num_feat_tensors_bool + 0*(1-selected_num_feat_tensors_bool)
    # if not eval:
    #     if not is_ppo:
    #         selected_op = torch.argmax(selected_op_probs, dim=-1)
    #     else:
    #         if train:
    #             dist = torch.distributions.Categorical(selected_op_probs)
    #             selected_op = dist.sample()
    #         else:
    #             selected_op = torch.argmax(selected_op_probs, dim=-1)
    # else:
    #     selected_op = atom[op_id_key]
    
    # selected_op_onehot = torch.zeros_like(selected_op_probs)
    # for k in range(net.topk_act):
    #     selected_op_onehot[torch.arange(len(selected_op_probs)), k, selected_op[:,k]]=1
    # hx = torch.cat([hx, selected_op_probs*selected_op_onehot], dim=-1)
     
    ret = {}
    

    

        # pred_lb = pred_lb*(range_max - range_min) + range_min
    if not eval:
        


        # selected_op_probs = selected_op_probs/torch.sum(selected_op_probs, dim=-1).unsqueeze(-1)

        # op_val = (selected_op_probs > 0.5).data.cpu().numpy().astype(float)
        
        
        # if not is_ppo:
        argmax = torch.argmax(pred_probs_vals,dim=-1)
        
        
        if net.feat_bound_point_ls is None:
            pred_v = argmax/(net.discretize_feat_value_count-1)
        else:
            pred_v = torch.stack([selected_feat_point_tensors[torch.arange(len(selected_feat_point_tensors)),k,argmax[:,k]] for k in range(argmax.shape[-1])], dim=1)
                
            # pred_v = pred_v*selected_cat_feat_tensors + feat_val*(1-selected_cat_feat_tensors)
            # # __ge__
            # pred_v1 = (feat_val)*(argmax/(discretize_feat_value_count-1))
            # # __le__
            # pred_v2 = (1 - feat_val)*(argmax/(discretize_feat_value_count-1)) + feat_val
        # else:
        #     if not net.continue_act:
        #         if train:
        #             dist = torch.distributions.Categorical(torch.softmax(pred, dim=-1))
        #             argmax = dist.sample()
        #         else:
        #             argmax = torch.argmax(pred, dim=-1)

        #         # argmax = argmax.cpu().numpy()
        #         if net.feat_bound_point_ls is None:
        #             pred_v = argmax/(net.discretize_feat_value_count-1)
        #         else:
        #             pred_v = torch.stack([selected_feat_point_tensors[torch.arange(len(selected_feat_point_tensors)),k,argmax[:,k]] for k in range(argmax.shape[-1])], dim=1)
        #         # # __ge__
        #         # pred_v1 = (feat_val)*(argmax/(discretize_feat_value_count-1))
        #         # # __le__
        #         # pred_v2 = (1 - feat_val)*(argmax/(discretize_feat_value_count-1)) + feat_val
        #     else:
        #         pred = torch.clamp(pred, min=1e-6, max=1)
        #         if train:
                    
        #             dist = torch.distributions.normal.Normal(pred[:,:,0], 1e-3)

        #             # dist = torch.distributions.Categorical(torch.softmax(pred, dim=-1))
        #             argmax = torch.clamp(dist.sample(), min=0, max=1)
        #         else:
        #             argmax = pred[:,:,0]
        #         # argmax = argmax.cpu().numpy()
        #         if net.feat_bound_point_ls is None:
        #             pred_v = argmax/(net.discretize_feat_value_count-1)
        #         else:
        #             pred_v = torch.stack([selected_feat_point_tensors[torch.arange(len(selected_feat_point_tensors)),k,argmax[:,k]] for k in range(argmax.shape[-1])], dim=1)

                # # __ge__
                # pred_v1 = (feat_val)*(argmax)
                # # __le__
                # pred_v2 = (1 - feat_val)*(argmax) + feat_val
                
        if net.feat_bound_point_ls is None:
            outbound_mask = (feat_val >= 1)
            outbound_mask = torch.logical_or(outbound_mask, (feat_val <= 0))
            # pred_v[(pred_v >= 1) & (feat_val >= 1)] = feat_val[(pred_v >= 1) & (feat_val >= 1)] + 1e-5
            # pred_v[(pred_v <= 0) & (feat_val <= 0)] = feat_val[(pred_v <= 0) & (feat_val <= 0)] - 1e-5
        else:
            outbound_mask = (feat_val >= selected_feat_point_tensors_max)
            outbound_mask = torch.logical_or(outbound_mask, (feat_val <= selected_feat_point_tensors_min))
            
            # pred_v[(pred_v >= selected_feat_point_tensors_max) & (feat_val >= selected_feat_point_tensors_max)] = feat_val[(pred_v >= selected_feat_point_tensors_max) & (feat_val >= selected_feat_point_tensors_max)] + 1e-5
            # pred_v[(pred_v <= selected_feat_point_tensors_min) & (feat_val <= selected_feat_point_tensors_min)] = feat_val[(pred_v <= selected_feat_point_tensors_min) & (feat_val <= selected_feat_point_tensors_min)] - 1e-5
        
        # if len(torch.nonzero(selected_num_feat_tensors_bool == 0)) > 0:
        #     print()
        
        pred_v = pred_v*selected_num_feat_tensors_bool + feat_val*(1-selected_num_feat_tensors_bool)
        
        selected_op = (pred_v <= feat_val).type(torch.long)*selected_num_feat_tensors_bool + 2*(1-selected_num_feat_tensors_bool)

        selected_op_ls = []
            
        selected_op_id_ls = selected_op.cpu().tolist()

        for idx in range(len(selected_op_id_ls)):
            selected_op_ls.append([net.grammar_num_to_token_val['num_op'][selected_op_id_ls[idx][k]] if selected_op_id_ls[idx][k] < 2 else net.grammar_num_to_token_val['cat_op'][0] for k in range(len(selected_op_id_ls[idx]))])


        # if net.lang.precomputed is not None:
        #     pred_v1, pred_v2 = find_nearest_thres_vals(net.lang.precomputed, selected_col_ls, pred_v1, pred_v2)
        # if self.op_list[0] == operator.__ge__:     
        
            
        # pred_v = pred_v1*op_val[:,:, 1] + pred_v2*op_val[:,:, 0]
    
    if init:
        ret[col_id_key] = selected_feat_col
    else:
        ret[col_id_key] = curr_selected_feat_col

    ret[select_num_feat_key] = selected_num_feat_tensors_bool.cpu()

    if eval:
        ret[pred_Q_key] = pred_Q_vals# torch.tanh(pred)
        ret[col_Q_key] = selected_Q_feat
        # ret[op_Q_key] = selected_Q_op
        ret[prev_prog_key] = prev_program_ids
        
        ret[col_probs_key] = selected_feat_probs
        
        ret[further_sel_mask_key] = further_selection_masks
        # ret[op_probs_key] = selected_op_probs
        # if net.continue_act and is_ppo:
        #     ret[pred_probs_key] = pred_probs_vals
        # else:
        ret[pred_probs_key] = pred_probs_vals#torch.softmax(pred, dim=-1)
    else:
        ret[pred_Q_key] = pred_Q_vals.data.cpu()#torch.tanh(pred).data
        ret[col_Q_key] = selected_Q_feat.data.cpu()
        # ret[op_Q_key] = selected_Q_op.data
        
        ret[col_probs_key] = selected_feat_probs.data.cpu()
        # ret[op_probs_key] = selected_op_probs.data
        # if net.continue_act and is_ppo:
        #     ret[pred_probs_key] = pred.data.cpu()
        # else:
        ret[pred_probs_key] = torch.softmax(pred, dim=-1).data.cpu()

        ret[pred_v_key] = pred_v.data.cpu().numpy()
    
        ret[op_key] = selected_op_ls        
        
        ret[op_id_key] = selected_op.data.cpu()
        
        ret[col_key] = selected_col_ls
        if prev_program_ids is not None:
            ret[prev_prog_key] = prev_program_ids.cpu()
        else:
            ret[prev_prog_key] = prev_program_ids

        ret[outbound_key] = outbound_mask.cpu()
        
        ret[further_sel_mask_key] = further_selection_masks.cpu()
    
    return ret

class InvariantModel(nn.Module):
    def __init__(self, phi: nn.Module, rho: nn.Module):
        super().__init__()
        self.phi = phi
        self.rho = rho

    def forward(self, x):
        # compute the representation for each data point
        x = self.phi.forward(x)

        # sum up the representations
        # here I have assumed that x is 2D and the each row is representation of an input, so the following operation
        # will reduce the number of rows to 1, but it will keep the tensor as a 2D tensor.
        x = torch.sum(x, dim=1, keepdim=True)

        # compute the output
        out = self.rho.forward(x)

        return out


class TokenNetwork3(nn.Module):
    def __init__(self, input_size, latent_size, output_size):
        super(TokenNetwork3, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, int(output_size)),
            # nn.ReLU(),
            # nn.Linear(int(latent_size), num_output_classes),
            # nn.Softmax(dim=-1),
        )
        # self.to(device=DEVICE)

    def forward(self, x):
        return self.linear_relu_stack(x)

class TokenNetwork2(nn.Module):
    def __init__(self, input_size, latent_size):
        super(TokenNetwork2, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, latent_size),
            nn.ReLU(),
            # nn.Linear(latent_size, int(latent_size)),
            # nn.ReLU(),
            # nn.Linear(int(latent_size), num_output_classes),
            # nn.Softmax(dim=-1),
        )
        # self.to(device=DEVICE)

    def forward(self, x):
        return self.linear_relu_stack(x)

def create_deep_set_net_for_programs(net, ATOM_VEC_LENGTH, latent_size):
    encoder = TokenNetwork3(ATOM_VEC_LENGTH, latent_size, latent_size).to(net.device)
    # decoder = torch.nn.Linear(latent_size, ATOM_VEC_LENGTH)
    decoder = torch.nn.Identity(latent_size).to(net.device)
    net.program_net = InvariantModel(encoder, decoder).to(net.device)