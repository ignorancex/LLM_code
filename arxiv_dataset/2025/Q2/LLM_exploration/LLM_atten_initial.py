from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import torch
import copy
import argparse
import numpy as np
import json
import scipy
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, LlamaTokenizer, AdamW, get_linear_schedule_with_warmup,AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, LoraModel, PeftConfig, PeftModel
import os
import pickle
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss
import json
import random
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import pickle
from proj import FP
import random

def get_total_grad_norm(parameters):
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)
    
def get_first_and_second_order_neighbors(data, input_ids):
    # 将邻接矩阵转换为 SparseTensor，这里直接使用 data.adj_t，因为它已经是 SparseTensor 类型
    adj_matrix = data.adj_t
    
    # 创建一个从原始节点 ID 到新节点 ID 的映射
    id_mapping = {old_id.item(): new_id for new_id, old_id in enumerate(data.n_id)}
    
    # 创建一个空字典来存储结果
    neighbors_dict = {}
    
    # 对于每一个输入节点
    for node_id in input_ids:
        # 将原始节点 ID 映射到新的节点 ID
        new_node_id = id_mapping[node_id.item()]
        
        # 获取该节点的所有邻居
        first_order_neighbor_ids = adj_matrix.storage.col()[adj_matrix.storage.row() == new_node_id]
        
        # 获取一阶邻居的原始 ID
        first_order_neighbors = {data.n_id[i].item(): [] for i in first_order_neighbor_ids.tolist()}
        
        # 为每个一阶邻居获取二阶邻居
        for first_order_neighbor in first_order_neighbor_ids:
            # 获取该一阶邻居的所有邻居
            second_order_neighbor_ids = adj_matrix.storage.col()[adj_matrix.storage.row() == first_order_neighbor]
            
            # 过滤掉自己作为一阶邻居的情况
            second_order_neighbor_ids = second_order_neighbor_ids[second_order_neighbor_ids != new_node_id]
            
            # 获取二阶邻居的原始 ID
            second_order_neighbors = [data.n_id[i].item() for i in second_order_neighbor_ids.tolist()]
            
            # 添加二阶邻居到对应的一阶邻居下
            first_order_neighbors[data.n_id[first_order_neighbor].item()].extend(second_order_neighbors)
        
        # 将邻居列表添加到字典中
        neighbors_dict[node_id.item()] = first_order_neighbors
    return neighbors_dict

def random_get_first_and_second_order_neighbors_I(data, input_ids):
    # 将邻接矩阵转换为 SparseTensor，这里直接使用 data.adj_t，因为它已经是 SparseTensor 类型
    adj_matrix = data.adj_t
    
    # 创建一个从原始节点 ID 到新节点 ID 的映射
    id_mapping = {old_id.item(): new_id for new_id, old_id in enumerate(data.n_id)}
    
    # 创建一个空字典来存储结果
    neighbors_dict = {}
    
    # 对于每一个输入节点
    for node_id in input_ids:
        # 将原始节点 ID 映射到新的节点 ID
        new_node_id = id_mapping[node_id.item()]
        
        # 获取该节点的所有邻居
        first_order_neighbor_ids = adj_matrix.storage.col()[adj_matrix.storage.row() == new_node_id]
        
        # 获取一阶邻居的原始 ID
        first_order_neighbors = {data.n_id[i].item(): [] for i in first_order_neighbor_ids.tolist()}
        
        # 为每个一阶邻居获取二阶邻居
        for first_order_neighbor in first_order_neighbor_ids:
            # 获取该一阶邻居的所有邻居
            second_order_neighbor_ids = adj_matrix.storage.col()[adj_matrix.storage.row() == first_order_neighbor]
            
            # 过滤掉自己作为一阶邻居的情况
            second_order_neighbor_ids = second_order_neighbor_ids[second_order_neighbor_ids != new_node_id]
            
            # 获取二阶邻居的原始 ID
            second_order_neighbors = [data.n_id[i].item() for i in second_order_neighbor_ids.tolist()]
            
            # 添加二阶邻居到对应的一阶邻居下
            first_order_neighbors[data.n_id[first_order_neighbor].item()].extend(second_order_neighbors)
        
        # 在完成邻居获取后，尝试添加扰动
        if len(first_order_neighbors) > 1:
            # 随机选择两个一阶邻居
            selected_neighbors = random.sample(list(first_order_neighbors.keys()), 2)
            
            # 交换它们的二阶邻居列表
            first_order_neighbors[selected_neighbors[0]], first_order_neighbors[selected_neighbors[1]] = \
                first_order_neighbors[selected_neighbors[1]].copy(), first_order_neighbors[selected_neighbors[0]].copy()
        
        # 将邻居列表添加到字典中
        neighbors_dict[node_id.item()] = first_order_neighbors
    
    return neighbors_dict

def random_get_first_and_second_order_neighbors_II(data, input_ids):
    adj_matrix = data.adj_t
    id_mapping = {old_id.item(): new_id for new_id, old_id in enumerate(data.n_id)}
    neighbors_dict = {}
    
    for node_id in input_ids:
        new_node_id = id_mapping[node_id.item()]
        first_order_neighbor_ids = adj_matrix.storage.col()[adj_matrix.storage.row() == new_node_id]
        first_order_neighbors = {data.n_id[i].item(): [] for i in first_order_neighbor_ids.tolist()}
        
        for first_order_neighbor in first_order_neighbor_ids:
            second_order_neighbor_ids = adj_matrix.storage.col()[adj_matrix.storage.row() == first_order_neighbor]
            second_order_neighbor_ids = second_order_neighbor_ids[second_order_neighbor_ids != new_node_id]
            second_order_neighbors = [data.n_id[i].item() for i in second_order_neighbor_ids.tolist()]
            first_order_neighbors[data.n_id[first_order_neighbor].item()].extend(second_order_neighbors)
        
        iter_num=0
        while iter_num<=10:
            if len(first_order_neighbors) > 1:
                # 随机选择两个一阶邻居
                selected_neighbors = random.sample(list(first_order_neighbors.keys()), 2)
                # 为这两个邻居随机选择一些二阶邻居进行交换
                swap_size = min(len(first_order_neighbors[selected_neighbors[0]]), len(first_order_neighbors[selected_neighbors[1]]))
                if swap_size > 0:
                    swap_indices = random.sample(range(swap_size), swap_size)
                    for idx in swap_indices:
                        first_order_neighbors[selected_neighbors[0]][idx], first_order_neighbors[selected_neighbors[1]][idx] = \
                            first_order_neighbors[selected_neighbors[1]][idx], first_order_neighbors[selected_neighbors[0]][idx]
            iter_num+=1
        
        neighbors_dict[node_id.item()] = first_order_neighbors
    
    return neighbors_dict

def random_get_first_and_second_order_neighbors_III(data, input_ids):
    adj_matrix = data.adj_t
    id_mapping = {old_id.item(): new_id for new_id, old_id in enumerate(data.n_id)}
    all_nodes = list(id_mapping.values())
    
    neighbors_dict = {}
    
    for node_id in input_ids:
        new_node_id = id_mapping[node_id.item()]
        first_order_neighbor_ids = adj_matrix.storage.col()[adj_matrix.storage.row() == new_node_id]
        first_order_neighbors = {data.n_id[i].item(): [] for i in first_order_neighbor_ids.tolist()}
        
        for first_order_neighbor in first_order_neighbor_ids:
            second_order_neighbor_ids = adj_matrix.storage.col()[adj_matrix.storage.row() == first_order_neighbor]
            second_order_neighbor_ids = second_order_neighbor_ids[second_order_neighbor_ids != new_node_id]
            second_order_neighbors = [data.n_id[i].item() for i in second_order_neighbor_ids.tolist()]
            first_order_neighbors[data.n_id[first_order_neighbor].item()].extend(second_order_neighbors)
        
        # 将所有的一阶和二阶节点收集到一个列表中
        all_neighbors = list(first_order_neighbors.keys()) + [neighbor for sublist in first_order_neighbors.values() for neighbor in sublist]
        
        print(first_order_neighbors)
        
        # 打乱所有节点
        np.random.shuffle(all_neighbors)
        
        # 重新分配一阶和二阶节点
        new_first_order_neighbors = all_neighbors[:len(first_order_neighbors)]
        new_second_order_neighbors = all_neighbors[len(first_order_neighbors):]
        
        # 构建新的邻居字典
        new_neighbors_dict = {node: [] for node in new_first_order_neighbors}
        
        for i, node in enumerate(new_first_order_neighbors):
            new_neighbors_dict[node] = new_second_order_neighbors[i * len(first_order_neighbors):(i + 1) * len(first_order_neighbors)]
        
        neighbors_dict[node_id.item()] = new_neighbors_dict
    
    return neighbors_dict


def get_all_neighbors(data, input_ids):
    # 获取所有输入节点的一阶和二阶邻居
    all_neighbors = {}
    adj_matrix = data.adj_t
    id_mapping = {old_id.item(): new_id for new_id, old_id in enumerate(data.n_id)}
    
    for node_id in input_ids:
        new_node_id = id_mapping[node_id.item()]
        first_order_neighbor_ids = adj_matrix.storage.col()[adj_matrix.storage.row() == new_node_id]
        first_order_neighbors = {data.n_id[i].item(): [] for i in first_order_neighbor_ids.tolist()}
        
        for first_order_neighbor in first_order_neighbor_ids:
            second_order_neighbor_ids = adj_matrix.storage.col()[adj_matrix.storage.row() == first_order_neighbor]
            second_order_neighbor_ids = second_order_neighbor_ids[second_order_neighbor_ids != new_node_id]
            second_order_neighbors = [data.n_id[i].item() for i in second_order_neighbor_ids.tolist()]
            first_order_neighbors[data.n_id[first_order_neighbor].item()].extend(second_order_neighbors)
        
        all_neighbors[node_id.item()] = first_order_neighbors
    
    return all_neighbors

def random_get_first_and_second_order_neighbors_IV(data, input_ids):
    all_neighbors = get_all_neighbors(data, input_ids)  # 获取整个 batch 的邻居信息
    neighbors_dict = {}

    for center_node_id in input_ids:
        temp_neighbors = copy.deepcopy(all_neighbors[center_node_id.item()])
        other_center_nodes = set(input_ids) - {center_node_id}
        
        # 收集其他中心节点的所有一阶和二阶邻居作为不相关节点池
        unrelated_nodes_pool = []
        for other_node_id in other_center_nodes:
            unrelated_nodes_pool.extend(list(all_neighbors[other_node_id.item()].keys()))
            for second_order_neighbors in all_neighbors[other_node_id.item()].values():
                unrelated_nodes_pool.extend(second_order_neighbors)
        
        # 确保不相关节点池中的节点不在当前中心节点的邻居列表中
        unrelated_nodes_pool = list(set(unrelated_nodes_pool) - set(temp_neighbors.keys()) - set([node for sublist in temp_neighbors.values() for node in sublist]))
        
        iter_num=0
        while iter_num<=10:
            if unrelated_nodes_pool:  # 如果存在不相关节点，则进行替换
                # 随机选择一定数量的不相关节点
                num_unrelated_to_replace = min(len(unrelated_nodes_pool), len(temp_neighbors)) // 2  # 可根据需要调整比例

                unrelated_nodes_sample = random.sample(unrelated_nodes_pool, num_unrelated_to_replace)

                # 对于每一个选中的不相关节点，随机决定是替换一阶还是二阶邻居
                for unrelated_node in unrelated_nodes_sample:
                    if random.choice([True, False]):  # 替换一阶邻居
                        replaced_node = random.choice(list(temp_neighbors.keys()))
                        temp_neighbors[unrelated_node] = temp_neighbors.pop(replaced_node)
                    else:  # 替换二阶邻居
                        if temp_neighbors:
                            first_order_node, second_order_neighbors = random.choice(list(temp_neighbors.items()))
                            if second_order_neighbors:
                                replaced_node = random.choice(second_order_neighbors)
                                second_order_neighbors.remove(replaced_node)
                                second_order_neighbors.append(unrelated_node)
            iter_num+=1
        neighbors_dict[center_node_id.item()] = temp_neighbors
    
    return neighbors_dict



def get_args():
    parser = argparse.ArgumentParser(description="PyTorch PYG implementation")
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    # CPU/GPU
    parser.add_argument("--num_workers", default=6, type=int)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    
    """LLM Config"""
    parser.add_argument('--backbone', type=str, default='./llama2-7b-hf')
    parser.add_argument('--tokenizer', type=str, default='AutoTokenizer')
    parser.add_argument('--max_text_length', type=int, default=4096)
    parser.add_argument('--lora_r', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=128)
    parser.add_argument('--lora_dropout', type=int, default=0.05)

    
    """LLM Training"""
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=0.005)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--adam_eps', type=float, default=1e-8)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument(
        "--num_neighbors",
        type=str,
        default="8,8",
        help="Number of samples for each layer in SAGE. Length = num_layers",
    )
    parser.add_argument(
        "--perturbation",
        type=int,
        default=0,
        help="perturbation type",
    )
    
    """Dataset"""
    parser.add_argument('--do_lower_case', action='store_true')
    parser.add_argument("--dataset", type=str, default="amazon_ratings", help="Dataset")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to data")
    parser.add_argument("--num_nodes", type=int, default="24492", help="the number of nodes")

    """Global """
    parser.add_argument("--train", type=bool, default="True", help="training ")
    parser.add_argument("--test", type=bool, default="False", help="testing ")
    args = parser.parse_args(args=[])

    
    return args

def pre_data(args):
    if args.dataset=='ogbn-arxiv':
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./arxiv/', transform=T.ToSparseTensor())
        data=dataset[0]
        data.adj_t = data.adj_t.to_symmetric()
        split_idx = dataset.get_idx_split()
        train_loader = NeighborLoader(data, input_nodes=split_idx["train"],
                                       num_neighbors=[eval(num_neighbors) for num_neighbors in args.num_neighbors.split(",")]
                                      ,batch_size=args.batch_size, 
                                      shuffle=True,num_workers=args.num_workers,
                                      pin_memory=True)
        valid_loader = NeighborLoader(copy.copy(data), input_nodes=split_idx["valid"],
                                      batch_size=args.batch_size,
                                         num_neighbors=[eval(num_neighbors) for num_neighbors in args.num_neighbors.split(",")]
                                      , shuffle=False,num_workers=args.num_workers)
        test_loader = NeighborLoader(copy.copy(data), input_nodes=split_idx["test"],
                                     batch_size=args.batch_size,
                                    num_neighbors=[eval(num_neighbors) for num_neighbors in args.num_neighbors.split(",")]
                                     , shuffle=False,num_workers=args.num_workers)
        
    if args.dataset=='deezer-europe':
        deezer = scipy.io.loadmat(f'./deezer_europe/deezer-europe.mat')
        adj_t= SparseTensor(row=torch.tensor(deezer['A'].tocoo().row).to(torch.long), col=torch.tensor(deezer['A'].tocoo().col).to(torch.long),sparse_sizes=(len(deezer['label'][0]), len(deezer['label'][0])))
        data= Data(x=torch.tensor(deezer['features'].toarray()).to(torch.float32), adj_t=adj_t,y=torch.tensor(deezer['label']).squeeze())
        data.adj_t = data.adj_t.to_symmetric()
        # 获取节点总数
        num_nodes = len(data.y)
        # 定义数据集划分比例
        train_ratio = 0.5
        val_ratio = 0.25
        test_ratio = 0.25
        # 计算每种数据集包含的节点数
        num_train = int(num_nodes * train_ratio)
        num_val = int(num_nodes * val_ratio)
        num_test = num_nodes - num_train - num_val
        # 随机排列节点索引
        node_indices = torch.randperm(num_nodes)
        # 切分索引
        train_indices = node_indices[:num_train]
        val_indices = node_indices[num_train:num_train + num_val]
        test_indices = node_indices[num_train + num_val:]

        train_loader = NeighborLoader(data, input_nodes=train_indices,
        num_neighbors=[eval(num_neighbors) for num_neighbors in args.num_neighbors.split(",")], 
                                      batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers)
        
        valid_loader = NeighborLoader(copy.copy(data), input_nodes=val_indices,
        num_neighbors=[eval(num_neighbors) for num_neighbors in args.num_neighbors.split(",")], 
                                      batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)
        
        test_loader = NeighborLoader(copy.copy(data), input_nodes=test_indices,
        num_neighbors=[eval(num_neighbors) for num_neighbors in args.num_neighbors.split(",")], 
                                      batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)    
    if args.dataset in ['roman_empire','amazon_ratings','questions']:
        
        file_path = f'./{args.dataset}/{args.dataset}_right.npz'
        if args.dataset=='amazon_ratings':
            file_path = f'./{args.dataset}/{args.dataset}_right_10.npz'
        
        data = np.load(file_path)
        
        # 切分索引
        train_indices = np.where(data['train_masks'][0])[0]
        val_indices = np.where(data['val_masks'][0])[0]
        test_indices = np.where(data['test_masks'][0])[0]
        
        
        # data = np.load('./roman_empire/roman_empire.npz')
        adj_t= SparseTensor(row=torch.tensor(data['edges']).t()[0].to(torch.long), col=torch.tensor(data['edges']).t()[1].to(torch.long),sparse_sizes=(len(data['node_labels']),len(data['node_labels']) ))
        data= Data(x=torch.tensor(data['node_features']), adj_t=adj_t,y=torch.tensor(data['node_labels']))
        data.adj_t = data.adj_t.to_symmetric()


        train_loader = NeighborLoader(data, input_nodes=train_indices,
                                       num_neighbors=[eval(num_neighbors) for num_neighbors in args.num_neighbors.split(",")], 
                                      batch_size=args.batch_size,
                                      shuffle=True,num_workers=args.num_workers,
                                      pin_memory=True)
        
        valid_loader = NeighborLoader(copy.copy(data), input_nodes=val_indices,
        num_neighbors=[eval(num_neighbors) for num_neighbors in args.num_neighbors.split(",")], 
                                      batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)
        
        test_loader = NeighborLoader(copy.copy(data), input_nodes=test_indices,
        num_neighbors=[eval(num_neighbors) for num_neighbors in args.num_neighbors.split(",")], 
                batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)
    if args.dataset in ['pubmed']:
        file_path = f'./{args.dataset}/data.pt'
        data = torch.load(file_path)
        
        data.adj_t = data.adj_t.to_symmetric()
        
        data.y=torch.tensor(data.y)

        train_indices = data.train_id
        val_indices = data.val_id
        test_indices = data.test_id

            
        train_loader = NeighborLoader(data, input_nodes=train_indices,
                                       num_neighbors=[eval(num_neighbors) for num_neighbors in args.num_neighbors.split(",")], 
                                      batch_size=args.batch_size,
                                      shuffle=True,num_workers=args.num_workers,
                                      pin_memory=True)
        valid_loader = NeighborLoader(copy.copy(data), input_nodes=val_indices,
        num_neighbors=[eval(num_neighbors) for num_neighbors in args.num_neighbors.split(",")], 
                                      batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)
        
        test_loader = NeighborLoader(copy.copy(data), input_nodes=test_indices,
        num_neighbors=[eval(num_neighbors) for num_neighbors in args.num_neighbors.split(",")], 
                batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)
        
    if args.dataset in ['wikics']:
        file_path = f'./{args.dataset}/data_token_right_10.pt'
        
        data = torch.load(file_path)
        
        node_id = np.arange(data.num_nodes)
        
        
        random.shuffle(node_id)
        
        train_indices = np.sort(node_id[:int(data.num_nodes * 0.6)])
        val_indices = np.sort(
            node_id[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)])
        test_indices = np.sort(node_id[int(data.num_nodes * 0.8):])
        
        train_loader = NeighborLoader(data, input_nodes=train_indices,
                                       num_neighbors=[eval(num_neighbors) for num_neighbors in args.num_neighbors.split(",")], 
                                      batch_size=args.batch_size,
                                      shuffle=True,num_workers=args.num_workers,
                                      pin_memory=True)
        
        valid_loader = NeighborLoader(copy.copy(data), input_nodes=val_indices,
        num_neighbors=[eval(num_neighbors) for num_neighbors in args.num_neighbors.split(",")], 
                                      batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)
        
        test_loader = NeighborLoader(copy.copy(data), input_nodes=test_indices,
        num_neighbors=[eval(num_neighbors) for num_neighbors in args.num_neighbors.split(",")], 
                batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)

    
    return train_loader,valid_loader,test_loader,data

class Trainer():
    def __init__(self,args):
        self.args=args
        
        if args.dataset=='wikics':
            template={}
            template['train']="<User>: In paper dataset, papers that cite each other form a linkage relationship. Based on the linkage relationships among papers, the research directions of papers can be predicted. Given that a paper {} that connect {}, What is the category of the paper {}? <Assistant>: {}"
            template['test']="<User>: In paper dataset, papers that cite each other form a linkage relationship. Based on the linkage relationships among papers, the research directions of papers can be predicted. Given that a paper {} that connect {}, What is the category of the paper {}? <Assistant>:"
        if args.dataset=='roman_empire':
            template={}
            template['train']="<User>: In an article, words that have dependency relationships (where one word depends on another) are connected, forming a dependency graph. Based on the connections between words, determine the syntactic role of each word. Given that a word {} that connect {}, what is the word {} syntactic role? <Assistant>: {}"
            template['test']="<User>: In an article, words that have dependency relationships (where one word depends on another) are connected, forming a dependency graph. Based on the connections between words, determine the syntactic role of each word. Given that a word {} that connect {}, what is the word {} syntactic role? <Assistant>:"
        if args.dataset=='amazon_ratings':
            template={}
            template['train']="<User>: In a product graph dataset, edges connect products that are frequently purchased together. Based on the connections between products (books, music CDs, DVDs, VHS tapes), predict the average rating given by reviewers for the products. Given that a product {} that connect {}, what is the product {} rating? <Assistant>: {}"
            template['test']="<User>: In a product graph dataset, edges connect products that are frequently purchased together. Based on the connections between products (books, music CDs, DVDs, VHS tapes), predict the average rating given by reviewers for the products. Given that a product {} that connect {}, what is the product {} rating? <Assistant>:"
        if args.dataset=='pubmed':
            template={}
            template['train']="<User>: In medical paper dataset, papers that cite each other form a linkage relationship. Based on the linkage relationships among papers, the research directions of medical papers can be predicted. Given that a paper {} that connect {}, What is the category of the paper {}? <Assistant>: {}"
            template['test']="<User>: In medical paper dataset, papers that cite each other form a linkage relationship. Based on the linkage relationships among papers, the research directions of medical papers can be predicted. Given that a paper {} that connect {}, What is the category of the paper {}? <Assistant>:"
        
        self.template=template
        
        self.tokenizer = self.get_tokenizer()
        self.train_loader, self.valid_loader, self.test_loader,self.data=pre_data(self.args)
        
        self.model= self.get_model()
        
        self.optimizer, self.lr_scheduler=self.get_optimizer()
        
    def get_tokenizer(self):
        tokenizer = LlamaTokenizer.from_pretrained(self.args.backbone, max_length=self.args.max_text_length,do_lower_case=self.args.do_lower_case)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.unk_token
        
        new_tokens=[ 'node_'+str(i) for i in range(self.args.num_nodes)]
        
        tokenizer.add_tokens(new_tokens)
        
        return tokenizer
    def get_optimizer(self):
        
        batch_per_epoch = len(self.train_loader)
        t_total = batch_per_epoch // self.args.gradient_accumulation_steps * self.args.epoch
        warmup_ratio = self.args.warmup_ratio
        warmup_iters = int(t_total * warmup_ratio)
        
        print("Batch per epoch: %d" % batch_per_epoch)
        print("Total Iters: %d" % t_total)
        print('Warmup ratio:', warmup_ratio)
        print("Warm up Iters: %d" % warmup_iters)
        
        
        if self.args.dataset in ['pubmed','amazon_ratings']:
            for param in self.model.model.model.embed_tokens.parameters():
                param.requires_grad = True
            
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": self.args.weight_decay,
                    'lr': self.args.lr,
                },
                # 这个组包含了bias和LayerNorm的所有参数，不应用权重衰减
                {
                    "params":[p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": 0.0,
                    'lr': self.args.lr,
                }

        ]
        optim = AdamW(optimizer_grouped_parameters,
                          lr=self.args.lr, eps=self.args.adam_eps)
        lr_scheduler = get_linear_schedule_with_warmup(optim, warmup_iters, t_total)
        
        return optim, lr_scheduler
    
    def get_model(self):
        model = AutoModelForCausalLM.from_pretrained(
                                    self.args.backbone,
                                    load_in_8bit=True,
                                    torch_dtype=torch.float16,
                                    use_safetensors=False,
                                    device_map='cuda:0'
                                )
        
        # model_embed=model.model.embed_tokens.weight.data
        
        model.resize_token_embeddings(32000+self.args.num_nodes)
        
        model.model.embed_tokens.weight.data[-self.args.num_nodes:]=self.data.x
        
        model = prepare_model_for_kbit_training(model)
        
        lora_config = LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            # target_modules=['q_proj','k_proj']
            target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj']
            
        )
        
        model= get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        
        return model
    
    def get_prompt(self,batch,is_training=True):
        #将label又数字id形式转化为文字
        if self.args.dataset=='ogbn-arxiv':
            dict_labelid2categeory=load_pickle('dict_labelid2arxivcategeory.pkl')
        if self.args.dataset=='deezer-europe':
            dict_labelid2categeory={}
            dict_labelid2categeory[0]='female'
            dict_labelid2categeory[1]='male'
        if self.args.dataset=='roman_empire':
            dict_labelid2categeory={}
            
            dict_labelid2categeory = {1: 'prepositional object',2: 'preposition',3: 'determiner',4: 'adjectival',5: 
                                    'conjunct',6: 'nominal subject',7: 'coordinating conjunction',0: 'root',
                                    8: 'direct object',9: 'adverbial',10: 'compound',11: 'auxiliary',
                                    12: 'appositional',13: 'passive auxiliary',14: 'passive nominal subject',15:
                                    'possession',16: 'relative clause',17: 'other'}
            
        if self.args.dataset=='amazon_ratings':
            
            dict_labelid2categeory={}
            dict_labelid2categeory[0]='Very Positive'
            dict_labelid2categeory[1]='Positive'
            dict_labelid2categeory[2]='Neutral'
            dict_labelid2categeory[3]='Negative'
            dict_labelid2categeory[4]='Very Negative'
            
        if self.args.dataset=='questions':
            dict_labelid2categeory={}
            dict_labelid2categeory[0]='activate'
            dict_labelid2categeory[1]='no'
        
        if self.args.dataset=='wikics':
            dict_labelid2categeory={
            0: 'Computational linguistics',
            1: 'Databases',
            2: 'Operating systems',
            3: 'Computer architecture',
            4: 'Computer security',
            5: 'Internet protocols',
            6: 'Computer file systems',
            7: 'Distributed computing architecture',
            8: 'Web technology',
            9: 'Programming language topics'}
        if self.args.dataset=='pubmed':
            dict_labelid2categeory={
            0: 'Diabetes Mellitus, Experimental',
            1: 'Diabetes Mellitus Type 1',
            2: 'Diabetes Mellitus Type 2'}
        if self.args.perturbation==0:
            neighbors_dict=get_first_and_second_order_neighbors(batch,batch.n_id[:batch.batch_size])
        if self.args.perturbation==1:
            neighbors_dict=random_get_first_and_second_order_neighbors_I(batch,batch.n_id[:batch.batch_size])
        if self.args.perturbation==2:
            neighbors_dict=random_get_first_and_second_order_neighbors_II(batch,batch.n_id[:batch.batch_size])
        if self.args.perturbation==3:
            neighbors_dict=random_get_first_and_second_order_neighbors_III(batch,batch.n_id[:batch.batch_size])
        if self.args.perturbation==4:
            neighbors_dict=random_get_first_and_second_order_neighbors_IV(batch,batch.n_id[:batch.batch_size])
            
            
        batch_text=[]
        labels=[]
        for i,label in zip(neighbors_dict.keys(),batch.y[:batch.batch_size]):
            label=dict_labelid2categeory[label.item()]
            connect_text='['
            text=''
            for j in neighbors_dict[i].keys():
                connect_text+='node_'+str(j) + ' is connected [' + ','.join('node_'+str(item) for item in neighbors_dict[i][j]) + ' ] ,'
            connect_text=connect_text[:-1]+']'
            if is_training :
                text = self.template['train'].format('node_'+str(i),connect_text,'node_'+str(i),label)+'</s>'
            else:
                text = self.template['test'].format('node_'+str(i),connect_text,'node_'+str(i))
            batch_text.append(text)
            labels.append(label+'</s>')
        input_ids=self.tokenizer(batch_text,padding='longest',
                                 max_length=self.args.max_text_length,return_tensors="pt")['input_ids']
        attention_mask=self.tokenizer(batch_text,padding='longest',
                                      max_length=self.args.max_text_length,return_tensors="pt")['attention_mask']
        
        #去掉开头的字符
        label_ids=self.tokenizer(labels,padding='longest',
                                 max_length=self.args.max_text_length,return_tensors="pt")['input_ids']
        
        
        
        if is_training:
            
            label_ids[label_ids.eq(self.tokenizer.pad_token_id)]=-100
            label_ids[:,-1]=2
            label_ids[label_ids.eq(1)]=-100
            label_ids=torch.cat((torch.full((label_ids.size(0), input_ids.size(-1)-label_ids.size(-1)), -100),
                              label_ids),dim=-1)
        else:
            # 测试阶段可能不需要生成标签
            label_ids = label_ids
        
        return input_ids, attention_mask, label_ids,neighbors_dict
    def load_checkpoint(self, ckpt_path,proj_path):
        results = self.model.load_state_dict(torch.load(ckpt_path), strict=True)  
        results = self.proj_model.load_state_dict(torch.load(proj_path), strict=True)
        if self.verbose:
            print('Model loaded from ', ckpt_path)
            print('Model loaded from ', proj_path)
            print(results)
    def train(self):
        
        self.model.train()
        
        pbar = tqdm(total=len(self.train_loader), ncols=275)
        for epoch in range(self.args.epoch):
            
            for step_i, batch in enumerate(self.train_loader):
                
                input_ids, attention_mask, labels, neighbors_dict=self.get_prompt(batch,True)
                
                attention_mask=attention_mask.to(self.args.device)
                
                labels=labels.to(self.args.device)
                
                input_ids=input_ids.to(self.args.device)
                
                output= self.model(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True)
                
                loss = output['loss']/ self.args.gradient_accumulation_steps
                
                
                loss.backward()
                
                
                if step_i % self.args.gradient_accumulation_steps == 0:
                    # 在训练循环中调用
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                    
                    self.optimizer.step()  # Update
                    self.lr_scheduler.step()
                    for param in self.model.parameters():
                        param.grad = None
                if step_i % 1 == 0:
                    lr = self.lr_scheduler.get_lr()[0]
                    desc_str = f'Epoch {epoch} | LR {lr:.6f} |'
                    desc_str += f' Loss:{loss:.3f}'
                    pbar.set_description(desc_str)
                    pbar.update(1)
        pbar.close()
            
        torch.save(self.model.state_dict(),"llmcom_{}_end_{}.pth".format(self.args.epoch,self.args.dataset))
                                         
    def test(self):
        for epoch in range(1):
            ckpt_path = "llmcom_1_end_{}.pth".format(self.args.dataset)

            self.model.load_state_dict(torch.load(ckpt_path), strict=True)  
            
            self.model.eval()
            acc_list=[]
            for time in range(4):
                with torch.no_grad():
                    print('len of val_loader is {}'.format(len(self.test_loader)))
                    acc=0
                    for step_i, batch in tqdm(enumerate(self.test_loader)):

                        input_ids, attention_mask, labels,neighbors_dict=self.get_prompt(batch,False)

                        attention_mask=attention_mask.to(self.args.device)

                        input_ids=input_ids.to(self.args.device)

                        embeds=self.model.model.model.embed_tokens(input_ids).to(self.args.device)

                        output= self.model.generate(inputs_embeds=embeds,
                                                    attention_mask=attention_mask,max_new_tokens=20,num_beams=2)
                        output=self.tokenizer.batch_decode(output,skip_special_tokens=True)

                        labels=self.tokenizer.batch_decode(labels,skip_special_tokens=True)
                        # print(output)
                        print(labels)
                        for i in range(len(output)):
                            if labels[i] == output[i]:
                               acc+=1
                        print(acc)
                    acc_list.append(acc)
            print(acc_list)
    def test_perturbation(self):
        for epoch in range(1):
            ckpt_path = "llmcom_1_end_{}.pth".format(self.args.dataset)

            self.model.load_state_dict(torch.load(ckpt_path), strict=True)  
            
            self.model.eval()
            
            acc_list_perbation=[]
            for perturbation in range(1,5):
                print('test_perturbation is {}'.format(perturbation))
                self.args.perturbation=perturbation
                acc_list=[]
                for time in range(4):
                    with torch.no_grad():
                        print('len of val_loader is {}'.format(len(self.test_loader)))
                        acc=0
                        for step_i, batch in tqdm(enumerate(self.test_loader)):

                            input_ids, attention_mask, labels,neighbors_dict=self.get_prompt(batch,False)

                            attention_mask=attention_mask.to(self.args.device)

                            input_ids=input_ids.to(self.args.device)

                            embeds=self.model.model.model.embed_tokens(input_ids).to(self.args.device)

                            output= self.model.generate(inputs_embeds=embeds,
                                                        attention_mask=attention_mask,max_new_tokens=20,num_beams=2)
                            output=self.tokenizer.batch_decode(output,skip_special_tokens=True)

                            labels=self.tokenizer.batch_decode(labels,skip_special_tokens=True)
                            # print(output)
                            print(labels)
                            for i in range(len(output)):
                                if labels[i] == output[i]:
                                   acc+=1
                            print(acc)
                        acc_list.append(acc)
                print(acc_list)
                acc_list_perbation.append(acc_list)
            print(acc_list_perbation)    
def set_random_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    args=get_args()

    seed_value = 42
    
    set_random_seed(seed_value)
    trainer=Trainer(args)
    if args.train==True:
        trainer.train()
        
    # set_random_seed(seed_value)
    # trainer=Trainer(args)
    # if args.test==True:   
    #     trainer.test()
    
    # set_random_seed(seed_value)
    # trainer=Trainer(args)
    # if args.test==True:   
    #     trainer.test_perturbation()

if __name__=='__main__':
    main()
