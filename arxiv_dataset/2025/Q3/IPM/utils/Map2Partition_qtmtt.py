'''
Form partition map to partition structure
Input: partition map
Output: partition appearance + QT depth map + direction map in the .txt form
(the partition data form in this code version is not the best form for encoder implementation, \
    a little complicated but you can focus on the legalization process of output partition map)
Author: Aolin Feng, Xinmin Feng
'''

import numpy as np
import torch.nn.functional as F
from random import choice
import torch
import random

def check_square_unity(mat):  # input 4*4 tensor
    num0 = len(torch.where(mat == 0)[0])
    if num0 >= 0 and num0 <= 12:  # 0 in the minority
        mat = torch.where(mat == 0, torch.full_like(mat, 1).cuda(), mat)
        # process 4 sub-mats
        for i in [0, 2]:
            for j in [0, 2]:
                sum_sub_mat = torch.sum(mat[i:i + 2, j:j + 2])
                if sum_sub_mat <= 10 and sum_sub_mat >= 5: # 1 and 2 or 3 mixed
                    sub_num1 = len(torch.where(mat[i:i + 2, j:j + 2] == 1)[0])
                    if sub_num1 < 3:
                        mat[i:i + 2, j:j + 2] = torch.where(mat[i:i + 2, j:j + 2] == 1, (torch.ones((2, 2)) * 2).cuda(), mat[i:i + 2, j:j + 2])
                    else:
                        mat[i:i + 2, j:j + 2] = torch.ones((2, 2)).cuda()
    elif num0 > 12 and num0 < 16:
        mat = torch.zeros((4, 4)).cuda()
    return mat

def eli_structual_error(out_batch):
    N = out_batch.shape[0]
    pooled_batch = torch.clamp(torch.round(F.max_pool2d(out_batch, 2)), min=0, max=3)
    for num in range(N):
        pooled_batch[num][0] = check_square_unity(pooled_batch[num][0])
    post_batch = F.interpolate(pooled_batch, scale_factor=2)
    del pooled_batch
    return post_batch

def th_round(input_batch, thd):
    input_batch = np.where(input_batch >= thd, np.full_like(input_batch, 1), input_batch)
    input_batch = np.where(input_batch <= -thd, np.full_like(input_batch, -1), input_batch)
    input_batch = np.where((input_batch > -thd) & (input_batch < thd), np.full_like(input_batch, 0),
                                 input_batch)
    return input_batch

def th_round2(input_batch, thd_1, thd_2):
    # for qt map, 0 1 [2] 3, thd=0.2
    input_batch = np.where((input_batch >= (1 + thd_2)) * (input_batch  <= (3 - thd_2)), np.full_like(input_batch, 2), input_batch)
    input_batch = np.where((input_batch >= (1 - thd_1)) * (input_batch  <= (1 + thd_1)), np.full_like(input_batch, 1), input_batch)
    input_batch = np.round(input_batch)
    return input_batch


def th_round3(input_batch, thd):
    # for mtt depth map, region for 2
    input_batch = np.where((input_batch >= (1 + thd)), np.full_like(input_batch, 2), input_batch)
    input_batch = np.round(input_batch)
    return input_batch

def th_round4(input_batch, thd_1, thd_2):
    # for mtt depth map, region for 1
    input_batch = np.where((input_batch <= (1 + thd_2)) * (input_batch >= (1 - thd_1)), np.full_like(input_batch, 1), input_batch)
    input_batch = np.where((input_batch >= (1 + thd_2)) * (input_batch <= 2.5), np.full_like(input_batch, 2), input_batch)
    input_batch = np.where((input_batch <= (1 - thd_1)), np.full_like(input_batch, 0), input_batch)
    input_batch = np.round(input_batch)
    return input_batch

def delete_tree(root):
    if len(root.children) == 0:  # no child
        del root
    else:
        children = root.children
        for child in children:
            delete_tree(child)

# Split_Mode and Search are set for generate combination modes
# example: input [[1,2], [3,4,5]]; output [[1,3],[1,4],[1,5],[2,3],[2,4],[2,5]]

class Split_Node():
    def __init__(self, split_type):
        self.split_type = split_type
        self.children = []

class Search():
    def __init__(self, cus_candidate_mode_list):
        self.split_root = Split_Node(0)
        self.parent_list = []
        self.cus_mode_list = cus_candidate_mode_list
        self.partition_modes = []
        self.cus_modes = []

    def get_cus_mode_tree(self):
        self.parent_list = [self.split_root]
        while len(self.cus_mode_list) != 0:  # transverse every cu
            parent_temp = []
            for parent in self.parent_list:
                for split_type in self.cus_mode_list[0]:
                    child = Split_Node(split_type)
                    parent.children.append(child)
                    parent_temp.append(child)
            self.parent_list = parent_temp
            self.cus_mode_list.pop(0)

    def bfs(self, node):
        self.cus_modes.append(node.split_type)
        if len(node.children) == 0:
            temp = self.cus_modes[1:]
            self.partition_modes.append(temp)
            self.cus_modes.pop(-1)
        else:
            for child in node.children:
                self.bfs(child)
            self.cus_modes.pop(-1)

    def get_partition_modes(self):
        self.get_cus_mode_tree()
        self.bfs(self.split_root)
        return self.partition_modes

class Map_Node():
    def __init__(self, qt_map, bt_map, dire_map, cus, depth, bt_depth, out_mt_map, out_dire_map, parent=None, early_skip=True):
        self.qt_map = qt_map
        self.bt_map = bt_map
        self.dire_map = dire_map
        self.depth = depth
        self.cus = cus  # [x, y, h, w] list
        self.children = []
        self.parent = parent
        self.bt_depth = bt_depth
        self.out_mt_map = out_mt_map
        self.out_dire_map = out_dire_map
        if early_skip:
            self.early_skip_list = []

def find_max_key(dictionary):
    max_value = max(dictionary.values())
    for key, value in dictionary.items():
        if value == max_value:
            return key
        

class Map_to_Partition():
    """Convert Partition maps to Split flags to Partition vectors
    
    Args:
        acc_level: options {0,1,2,3}, accuracy level for partition depth
    """
    def __init__(self, qt_map, msbt_map, msdire_map, chroma_factor, lamb1=0.85, lamb2=[0.2, 0.2], lamb3=1.5, lamb4=0.3, \
                 lamb5=0.7, lamb6=0.3, lamb7=[0.5, 0.5, 0.5], lamb8=[0.3, 0.2, 0.1], block_size=64, no_dir=False, early_skip=False, debug_mode=False, acc_level=3):
        self.early_skip = early_skip  # Whether successor partitions affect previous partitions
        self.no_dir = no_dir  # Whether to consider direction map
        
        # Round maps for easier processing
        self.qt_map = th_round2(qt_map, thd_1=lamb7[0], thd_2=lamb7[1])
        self.qt_map = self.qt_map.clip(max=3)
        self.msbt_map = th_round4(msbt_map, thd_1=lamb7[1], thd_2=lamb7[2])
        self.ori_qt_map = qt_map
        self.ori_msbt_map = msbt_map
        self.msdire_map = th_round(msdire_map, thd=0.5)
        self.ori_msdire_map = msdire_map
        
        self.block_ratio = block_size // 64
        self.chroma_factor = chroma_factor  # luma=1, chroma=2
        self.par_vec = np.zeros((2, self.block_ratio * 16 + 1, self.block_ratio * 16 + 1), dtype=np.uint8)
        self.out_msdire_map = np.zeros((3, 16 * self.block_ratio, 16 * self.block_ratio), dtype=np.int8)
        self.out_qt_map = np.zeros((8 * self.block_ratio, 8 * self.block_ratio))
        self.out_bt_map = np.zeros_like(msbt_map)
        self.debug_mode = debug_mode
        self.cur_leaf_nodes = []  # Store leaf nodes of Map Tree

        # Threshold parameters
        self.lamb1 = lamb1  # Control no partition based on depth map
        self.lamb2 = lamb2  # QT partition decision
        self.lamb3 = lamb3  # Control horizontal or vertical
        self.lamb4 = lamb4  # Control number of minus
        self.lamb5 = lamb5  # Control number of zero
        self.lamb6 = lamb6  # Judge whether QT or not early
        
        self.lamb8 = lamb8  # Excludes non-split modes
        self.time = 0
        self.acc_level = acc_level
        
    def split_cur_map(self, x, y, h, w, split_type):
        # Split current CU [x,y,h,w]
        # split_type: 0=no split, 1=BTH, 2=BTV, 3=TTH, 4=TTV, 5=QT
        if split_type == 0:
            return [[x, y, h, w]]
        elif split_type == 1:  # BTH
            return [[x, y, h//2, w], [x+h//2, y, h//2, w]]
        elif split_type == 2:  # BTV
            return [[x, y, h, w//2], [x, y+w//2, h, w//2]]
        elif split_type == 3:  # TTH
            return [[x, y, h//4, w], [x+h//4, y, h//2, w], [x+(h*3)//4, y, h//4, w]]
        elif split_type == 4:  # TTV
            return [[x, y, h, w//4], [x, y+w//4, h, w//2], [x, y+(w*3)//4, h, w//4]]
        elif split_type == 5:  # QT
            return [[x, y, h//2, w//2], [x, y+w//2, h//2, w//2], [x+h//2, y, h//2, w//2], [x+h//2, y+w//2, h//2, w//2]]
        else:
            print("Unknown split type!")

    def early_skip_chk(self, x, y, h, w, split_type):
        if split_type == 0:
            return [0]
        elif split_type == 1 or split_type == 2:  # BTH/BTV
            return [1, 1]
        elif split_type == 3 or split_type == 4:  # TTH/TTV
            return [1, 1, 1]
        elif split_type == 5:  # QT
            return [1, 1, 1, 1]
        else:
            print("Unknown split type!")

    def can_split_mode_list(self, x, y, h, w, cur_bt_map, depth, cur_qt_map, cur_bt_depth):
        """Output candidate split type list for current CU"""
        if h <= 1 and w <= 1:
            return [0]
        if cur_bt_depth >= 3:
            return [0]
            
        comp_map_qt = self.qt_map[x//2:(x+h)//2, y//2:(y+w)//2] - cur_qt_map[x//2:(x+h)//2, y//2:(y+w)//2]
        comp_map_mtt = self.msbt_map[cur_bt_depth, x:x+h, y:y+w] - cur_bt_map[x:x+h, y:y+w]
        count_zero_mtt = len(np.where((comp_map_mtt).round() == 0)[0])
        count_zero_qt = len(np.where((comp_map_qt).round() == 0)[0])
        
        if (count_zero_qt >= self.lamb1 * h * w // 4) and (count_zero_mtt >= self.lamb1 * h * w):
            return [0]

        qt_terminal = False
        if count_zero_qt >= self.lamb1 * h * w // 4:
            qt_terminal = True
        
        # Complete QT prediction
        if cur_qt_map[x//2, y//2] == 0:
            ratio = self.lamb2[0]
        else:
            ratio = self.lamb2[1]
            
        if (np.sum(self.qt_map[x//2:(x+h)//2, y//2:(y+w)//2] > cur_qt_map[x//2:(x+h)//2, y//2:(y+w)//2]) >= ratio * h * w // 4) and depth <= 2:
            res_map = depth + 1 - self.qt_map[x//2:(x+h)//2, y//2:(y+w)//2].clip(min=0, max=depth+1)
            self.qt_map[x//2:(x+h)//2, y//2:(y+w)//2] += res_map
            res_map = depth + 1 - self.ori_qt_map[x//2:(x+h)//2, y//2:(y+w)//2].clip(min=0, max=depth+1)
            self.ori_qt_map[x//2:(x+h)//2, y//2:(y+w)//2] += res_map
            return [5]

        if self.acc_level == 0:
            return [0]
        elif cur_bt_depth >= self.acc_level:
            return [0]
            
        direction = 0  # 0=Unknown, 1=Horizontal, 2=Vertical
        count_dire_nonzero = 0
        if not self.no_dir:
            count_hor = len(np.where(self.msdire_map[cur_bt_depth, x:x+h, y:y+w] == 1)[0])
            count_ver = len(np.where(self.msdire_map[cur_bt_depth, x:x+h, y:y+w] == -1)[0])
            count_dire_nonzero = (count_hor + count_ver) / (h * w)
            direction = 1 if count_hor > self.lamb3 * count_ver else 2
            
        if (len(np.where((comp_map_mtt).round() != 0)[0]) > self.lamb8[cur_bt_depth] * h * w):
            exclude_non_split = True
        else:
            exclude_non_split = False
            
        initial_split_list = []
        for split_mode in [1, 2, 3, 4, 5]:
            if (split_mode == 1 or split_mode == 5) and (h // (2*self.chroma_factor) == 0 or h % (2*self.chroma_factor) != 0):
                continue
            if (split_mode == 2 or split_mode == 5) and (w // (2*self.chroma_factor) == 0 or w % (2*self.chroma_factor) != 0):
                continue
            if split_mode == 3 and (h // (4*self.chroma_factor) == 0 or h % (4*self.chroma_factor) != 0):
                continue
            if split_mode == 4 and (w // (4*self.chroma_factor) == 0 or w % (4*self.chroma_factor) != 0):
                continue
            if (split_mode == 1 or split_mode == 3) and direction == 2:
                continue
            if (split_mode == 2 or split_mode == 4) and direction == 1:
                continue
            if (split_mode == 3 or split_mode == 4) and depth == 0:
                continue
            if split_mode == 5 and (depth >= 3 or direction != 0 or cur_bt_depth != 0 or qt_terminal or exclude_non_split):
                continue  
            initial_split_list.append(split_mode)
        
        if cur_bt_depth == 0:
            previous_msbt = 0
        else:
            previous_msbt = self.msbt_map[cur_bt_depth - 1, x:x+h, y:y+w]
            
        if (self.msbt_map[cur_bt_depth, x:x+h, y:y+w] - previous_msbt).max() == 2 and (3 in initial_split_list or 4 in initial_split_list):
            if 0 in initial_split_list:
                initial_split_list.remove(0)
            if 1 in initial_split_list:
                initial_split_list.remove(1)
            if 2 in initial_split_list:
                initial_split_list.remove(2)
            if 5 in initial_split_list:
                initial_split_list.remove(5)
            tt_flag = True
        else:
            tt_flag = False
        
        temp_cost_dict = {}
        candidate_mode_list = []  
        for split_mode in initial_split_list:
            sub_map_xyhw = self.split_cur_map(x, y, h, w, split_mode)
            map_temp = np.zeros_like(cur_bt_map, dtype=np.int8)
            map_temp[:, :] = cur_bt_map[:, :]
            map_temp_qt = np.zeros_like(cur_qt_map, dtype=np.int8)
            map_temp_qt[:, :] = cur_qt_map[:, :] 
            split_thres = 0
            temp_cost, score_list = 0, []
            count_one_ratio = []
            
            for sub_map_id in range(len(sub_map_xyhw)):
                [sub_x, sub_y, sub_h, sub_w] = sub_map_xyhw[sub_map_id]
                if split_mode == 5:
                    map_temp_qt[sub_x // 2:(sub_x + sub_h) // 2, sub_y // 2:(sub_y + sub_w) // 2] += 1
                    comp_map = self.qt_map[sub_x // 2:(sub_x + sub_h) // 2, sub_y//2:(sub_y + sub_w) // 2] - map_temp_qt[sub_x // 2:(sub_x + sub_h) // 2, sub_y // 2:(sub_y + sub_w) // 2]
                    num_pixel = sub_h * sub_w // 4
                else:
                    map_temp[sub_x:sub_x + sub_h, sub_y:sub_y + sub_w] += 1
                    if (split_mode == 3 or split_mode == 4) and (sub_map_id != 1):
                        map_temp[sub_x:sub_x + sub_h, sub_y:sub_y + sub_w] += 1
                    if tt_flag and direction != 0:
                        msbt_res_map = map_temp[sub_x:sub_x + sub_h, sub_y:sub_y + sub_w] - self.msbt_map[cur_bt_depth, sub_x:sub_x + sub_h, sub_y:sub_y + sub_w]
                        for temp_d in range(cur_bt_depth, 3):
                            self.msbt_map[temp_d, sub_x:sub_x + sub_h, sub_y:sub_y + sub_w] += msbt_res_map
                    comp_map = self.msbt_map[cur_bt_depth, sub_x:sub_x + sub_h, sub_y:sub_y + sub_w] - map_temp[sub_x:sub_x + sub_h, sub_y:sub_y + sub_w]
                    num_pixel = sub_h * sub_w
                    
                count_minus = len(np.where(comp_map < 0)[0])
                count_zero = len(np.where(comp_map >= 0)[0])
                count_one_ratio.append(len(np.where(comp_map >= 1)[0]) / num_pixel)
                
                if isinstance(self.lamb5, list):
                    lamb5_weight = self.lamb5[cur_bt_depth]
                else:
                    lamb5_weight = self.lamb5
                    
                if count_zero >= num_pixel * lamb5_weight:
                    split_thres += 1
                    temp_cost += count_zero
                score_list.append(count_zero / num_pixel)
                
            temp_cost_dict[str(split_mode)] = temp_cost
            if sum(count_one_ratio) / len(count_one_ratio) >= 0.4:
                if split_mode in [3, 4]:
                    candidate_mode_list.append(split_mode)
            elif (split_thres >= len(sub_map_xyhw) // 2 + 1) \
                or (sum(score_list) / len(score_list) >= 0.6 and split_thres >= len(sub_map_xyhw) // 2):
                candidate_mode_list.append(split_mode)
            elif tt_flag:
                candidate_mode_list.append(split_mode)
        
        if len(candidate_mode_list) == 0:
            if exclude_non_split and len(temp_cost_dict.keys()) != 0 and temp_cost_dict[find_max_key(temp_cost_dict)] >= self.lamb8[cur_bt_depth] * h * w:
                candidate_mode_list = [int(find_max_key(temp_cost_dict))]
            else:
                candidate_mode_list = [0]
                
        if self.no_dir and direction == 0:
            if 1 in candidate_mode_list and 2 in candidate_mode_list:
                candidate_mode_list = list(set(candidate_mode_list) - set([choice([1,2]),]))
            if 3 in candidate_mode_list and 4 in candidate_mode_list:
                if temp_cost_dict['3'] >= temp_cost_dict['4']:
                    candidate_mode_list = list(set(candidate_mode_list) - set([4]))
                else:
                    candidate_mode_list = list(set(candidate_mode_list) - set([3]))

        return candidate_mode_list

    def get_candidate_map_tree(self, map_node):
        if map_node.depth >= 7:
            return
        cur_cus = map_node.cus
        cu_num = len(cur_cus)
        cur_bt_map = map_node.bt_map
        cur_dire_map = map_node.dire_map
        cur_depth = map_node.depth
        cur_qt_map = map_node.qt_map
        cur_bt_depth_list = map_node.bt_depth
        cur_out_dire_map = map_node.out_dire_map
        cur_out_mt_map = map_node.out_mt_map
        if self.early_skip:
            early_skip_list = map_node.early_skip_list
            
        cus_candidate_mode_list = []
        for i in range(cu_num):
            cus_candidate_mode_list.append([])
            
        for cu_id in range(cu_num):
            if self.early_skip and early_skip_list[cu_id] == 0:
                cus_candidate_mode_list[cu_id] = [0]
                continue
            [cu_x, cu_y, cu_h, cu_w] = cur_cus[cu_id]
            candidate_mode_list = self.can_split_mode_list(cu_x, cu_y, cu_h, cu_w, cur_bt_map, cur_depth, cur_qt_map, cur_bt_depth_list[cu_id])
            if len(candidate_mode_list) == 0:
                return
            cus_candidate_mode_list[cu_id] += candidate_mode_list
            
        s = Search(cus_candidate_mode_list)
        partition_modes = s.get_partition_modes()

        if False not in [(set(modes) == set([0])) for modes in partition_modes]:
            return
        
        if len(partition_modes) >= 1024:
            if cu_num <= 32:
                partition_modes = random.sample(partition_modes, 256)
            else:
                partition_modes = random.sample(partition_modes, 8)

        for cus_modes in partition_modes:
            child_qt_map = np.zeros_like(cur_qt_map, dtype=np.int8)
            child_bt_map = np.zeros_like(cur_bt_map, dtype=np.int8)
            child_dire_map = np.zeros_like(cur_dire_map, dtype=np.int8)
            child_bt_map[:, :] = cur_bt_map                
            child_qt_map[:, :] = cur_qt_map
            child_out_mt_map = np.zeros_like(cur_out_mt_map)
            child_out_mt_map[:,:,:] = cur_out_mt_map[:,:,:]
            child_out_dire_map = np.zeros_like(cur_out_dire_map)
            child_out_dire_map[:,:,:] = cur_out_dire_map[:,:,:]
            child_cus = []
            child_bt_depths = []
            if self.early_skip:
                early_skip_list = []
                
            for cu_id in range(cu_num):
                [cu_x, cu_y, cu_h, cu_w] = cur_cus[cu_id]
                cu_mode = cus_modes[cu_id]
                child_map_xyhw = self.split_cur_map(cu_x, cu_y, cu_h, cu_w, cu_mode)
                child_cus += child_map_xyhw
                cu_bt_depth = cur_bt_depth_list[cu_id]
                child_bt_depth = [cu_bt_depth + 1 if cu_mode in [1,2,3,4] else cu_bt_depth for i in range(len(child_map_xyhw))]
                child_bt_depths += child_bt_depth
                
                if self.early_skip:
                    early_skip_list += self.early_skip_chk(cu_x, cu_y, cu_h, cu_w, cu_mode)
                    
                if cu_mode == 0:
                    child_dire_map[cu_x:cu_x+cu_h, cu_y:cu_y+cu_w] = 0
                    if cu_bt_depth <= 2:
                        child_out_dire_map[cu_bt_depth, cu_x:cu_x+cu_h, cu_y:cu_y+cu_w] = 0
                    continue
                elif cu_mode == 1 or cu_mode == 3:
                    child_dire_map[cu_x:cu_x + cu_h, cu_y:cu_y + cu_w] = 1
                    child_out_dire_map[cu_bt_depth, cu_x:cu_x+cu_h, cu_y:cu_y+cu_w] = 1
                elif cu_mode == 2 or cu_mode == 4:
                    child_dire_map[cu_x:cu_x + cu_h, cu_y:cu_y + cu_w] = -1
                    child_out_dire_map[cu_bt_depth, cu_x:cu_x+cu_h, cu_y:cu_y+cu_w] = -1

                for sub_block_id in range(len(child_map_xyhw)):
                    [sub_x, sub_y, sub_h, sub_w] = child_map_xyhw[sub_block_id]
                    if cu_mode == 5:
                        child_qt_map[sub_x//2:(sub_x + sub_h)//2, sub_y//2:(sub_y + sub_w)//2] += 1
                    else:
                        child_bt_map[sub_x:sub_x + sub_h, sub_y:sub_y + sub_w] += 1
                        child_out_mt_map[cu_bt_depth, sub_x:sub_x + sub_h, sub_y:sub_y + sub_w] = 1
                    if (cu_mode == 3 or cu_mode == 4) and (sub_block_id != 1):
                        child_bt_map[sub_x:sub_x + sub_h, sub_y:sub_y + sub_w] += 1
                        child_out_mt_map[cu_bt_depth, sub_x:sub_x + sub_h, sub_y:sub_y + sub_w] += 1
                
            child_map_node = Map_Node(bt_map=child_bt_map, dire_map=child_dire_map, \
                    cus=child_cus, parent=map_node, depth=cur_depth+1, qt_map=child_qt_map, \
                        early_skip=self.early_skip, bt_depth=child_bt_depths, out_mt_map=child_out_mt_map, out_dire_map=child_out_dire_map)
            if self.early_skip:
                child_map_node.early_skip_list = early_skip_list
            self.get_candidate_map_tree(child_map_node)
            map_node.children.append(child_map_node)

    def get_leaf_nodes(self, map_node):
        if len(map_node.children) == 0:
            self.cur_leaf_nodes.append(map_node)
        else:
            for child_node in map_node.children:
                self.get_leaf_nodes(child_node)

    def print_tree(self, map_node, depth):
        print('**********************')
        print('node', depth)
        print(map_node.mtt_depth)
        print(map_node.bt_map)
        print(map_node.cus)
        print(len(map_node.children))
        print('**********************')
        if len(map_node.children) != 0:
            for child_node in map_node.children:
                self.print_tree(child_node, depth+1)

    def set_bt_partition_vector(self, x, y, h, w):
        init_qt_map = np.zeros((8 * self.block_ratio, 8 * self.block_ratio), dtype=np.int8)
        init_bt_map = np.zeros((16 * self.block_ratio, 16 * self.block_ratio), dtype=np.int8)
        init_dire_map = np.zeros((16 * self.block_ratio, 16 * self.block_ratio), dtype=np.int8)
        init_out_mt_map = np.zeros((3, 16 * self.block_ratio, 16 * self.block_ratio), dtype=np.int8)
        init_out_dire_map = np.zeros((3, 16 * self.block_ratio, 16 * self.block_ratio), dtype=np.int8)
        map_root = Map_Node(qt_map=init_qt_map, bt_map=init_bt_map, dire_map=init_dire_map, depth=0, cus=[[x, y, h, w]], early_skip=self.early_skip, bt_depth=[0], out_mt_map=init_out_mt_map, out_dire_map=init_out_dire_map)
        if self.early_skip:
            map_root.early_skip_list = [1]
        self.get_candidate_map_tree(map_root)

        self.cur_leaf_nodes = []
        self.get_leaf_nodes(map_root)
        
        self.ori_msbt_map = self.msbt_map
        self.ori_msdire_map = self.msdire_map
        error_list = []
        
        if len(self.cur_leaf_nodes) > 1:
            for node4 in self.cur_leaf_nodes:
                qt_map = node4.qt_map
                bt_map = node4.out_mt_map
                dire_map = node4.out_dire_map
                error = np.sum(np.abs(qt_map - self.ori_qt_map)) + \
                        np.sum(np.abs(bt_map - self.ori_msbt_map))
                error_list.append(error)
            min_index = error_list.index(min(error_list))
        else:
            min_index = 0
        best_node4 = self.cur_leaf_nodes[min_index]
        
        self.out_qt_map = best_node4.qt_map
        self.out_msdire_map = best_node4.out_dire_map
        self.out_bt_map = best_node4.out_mt_map
        
        best_cus = self.cur_leaf_nodes[min_index].cus
        delete_tree(map_root)
        for cu in best_cus:
            [cu_x, cu_y, cu_h, cu_w] = cu
            for i_w in range(cu_w):
                self.par_vec[0, cu_x, cu_y + i_w] = 1
                self.par_vec[0, cu_x + cu_h, cu_y + i_w] = 1
            for i_h in range(cu_h):
                self.par_vec[1, cu_x + i_h, cu_y] = 1
                self.par_vec[1, cu_x + i_h, cu_y + cu_w] = 1

    def get_partition(self):
        self.set_bt_partition_vector(0, 0, 32, 32)
        return self.par_vec, self.out_msdire_map, self.out_qt_map


def map_to_partition_qtmtt(qt_map, bt_map, dire_map, qp, chroma_factor, block_size=128, early_skip=True, debug_mode=False, no_dir=False, acc_level=3, frm_id=None, block_x=None, block_y=None):
    """Convert partition maps to partition vectors
    
    Args:
        early_skip: Whether successor partitions affect previous partitions
        debug_mode: Whether to output optimized qt_map, bt_map and dir_map
        no_dir: Whether to use direction map for prediction
    """
    block_ratio = block_size // 64
    
    # Parameter settings based on QP
    if qp == 37:
        lamb_params = {'lamb1': 0.85, 'lamb2': [0.7, 0.85], 'lamb3': 1, 'lamb7': [0.5, 0.5, 0.4, 0.5], 'lamb5': [0.3, 0.2, 0.1], 'lamb8': [0.3, 0.2, 0.1]}
    elif qp == 32:
        lamb_params = {'lamb1': 0.85, 'lamb2': [0.4, 0.2], 'lamb3': 1, 'lamb7': [0.5, 0.2, 0.5, 0.5], 'lamb5': [0.3, 0.2, 0.1], 'lamb8': [0.3, 0.2, 0.1]}
    elif qp == 27:
        lamb_params = {'lamb1': 0.85, 'lamb2': [0.4, 0.9], 'lamb3': 1, 'lamb7': [0.3, 0.2, 0.5, 0.5], 'lamb5': [0.3, 0.2, 0.1], 'lamb8': [0.3, 0.2, 0.1]}
    elif qp == 22:
        lamb_params = {'lamb1': 0.85, 'lamb2': [0.4, 0.9], 'lamb3': 1, 'lamb7': [0.3, 0.2, 0.5, 0.5], 'lamb5': [0.3, 0.2, 0.1], 'lamb8': [0.3, 0.2, 0.1]}
    
    partition = Map_to_Partition(qt_map, bt_map, dire_map, chroma_factor, block_size=128, early_skip=early_skip, debug_mode=debug_mode, no_dir=no_dir, acc_level=acc_level, **lamb_params)
    p, d, q = partition.get_partition()
    
    if debug_mode:
        return partition.out_qt_map, partition.out_bt_map, partition.out_msdire_map
    else:
        if frm_id is None:
            return p[0][:16*block_ratio, :16*block_ratio], p[1][:16*block_ratio, :16*block_ratio], d, q
        else:
            return p[0][:16*block_ratio, :16*block_ratio], p[1][:16*block_ratio, :16*block_ratio], d, q, frm_id, block_x, block_y