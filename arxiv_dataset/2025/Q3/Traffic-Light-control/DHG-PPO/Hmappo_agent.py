import torch
import math
import pickle
from agent import Agent
import random
import psutil
import os
import time
import numpy as np
import torch.nn as nn

import torch.nn.functional as F

"""
Model for HMAPPO
"""



class Predictor(nn.Module):

    def __init__(self, in_dim, n_cats):
        super(Predictor, self).__init__()
        self.predict = nn.Linear(in_dim, n_cats)
        torch.nn.init.xavier_uniform_(self.predict.weight)
        self.sigma = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, node_emb):
        n_cat = self.predict(node_emb)
        return n_cat


class Readout(nn.Module):

    def __init__(self, in_dim, method="mean"):
        super(Readout, self).__init__()
        self.method = method
        self.linear = nn.Linear(2*in_dim, in_dim)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, node_fea, raw):
        if self.method == "mean":
            return self.linear(torch.cat([node_fea, raw],dim=-1))  




class SpatialHyperedge(nn.Module):
    
    def __init__(self,in_dim,l2_lamda = 0.001,recons_lamda = 0.2,num_node = 16,lb_th = 0):
        super(SpatialHyperedge, self).__init__()
        self.l2_lamda = l2_lamda
        self.fea_dim = in_dim
        self.recons_lamda = recons_lamda
        
        self.num_node = num_node
        self.r_proj = nn.Parameter(torch.nn.init.xavier_uniform_(torch.rand(in_dim,in_dim)))  
        
        self.incidence_m = nn.Parameter(torch.nn.init.xavier_uniform_(torch.rand(num_node, num_node-1)))  
        self.lb_th = lb_th  

    def forward(self, X):
        
        self.incidence_all = torch.zeros(self.num_node, self.num_node)  
        self_node = torch.eye(self.num_node)
        self.recon_loss = 0
        batch = len(X)
        X = X.detach()  
        for node_idx in range(self.num_node):
            master_node_fea = X[:,node_idx,:]  
            master_node_proj_fea = torch.matmul(master_node_fea, self.r_proj).reshape(-1, self.fea_dim)  
            slave_node_idx = [i for i in range(self.num_node) if i != node_idx]  
            node_linear_comb = self.incidence_m[node_idx].unsqueeze(0)  
            node_linear_comb = torch.clamp(node_linear_comb, min=self.lb_th)  

            node_linear_comb_mask = node_linear_comb > self.lb_th  
            node_linear_comb = node_linear_comb.masked_fill(~node_linear_comb_mask, value=torch.tensor(0))  
            neigh_recon_fea = torch.matmul(node_linear_comb, X[:,slave_node_idx,:]).reshape(batch,-1)  
            self.incidence_all[node_idx][slave_node_idx] = node_linear_comb  
            
            linear_comb_l1 = torch.max(torch.sum(torch.abs(node_linear_comb),dim=0))

            
            linear_comb_l2 = torch.sqrt(torch.sum(node_linear_comb.pow(2)))

            
            recon_error = torch.sqrt(torch.sum((master_node_proj_fea - neigh_recon_fea).pow(2),dim=-1)).reshape(batch,-1) 
            
            node_recons_loss = recon_error * self.recons_lamda + linear_comb_l1 + self.l2_lamda * linear_comb_l2  
                
            self.recon_loss += node_recons_loss
        self.incidence_all = self.incidence_all + self_node  
        return self.recon_loss, self.incidence_all  



class TemporalHyperedge(nn.Module):

    def __init__(self,in_dim,l2_lamda = 0.001,recons_lamda = 0.2, num_node = 16, lb_th = 0):
        super(TemporalHyperedge, self).__init__()
        self.l2_lamda = l2_lamda
        self.recons_lamda = recons_lamda
        
        self.num_node = num_node
        self.r_proj = nn.Parameter(torch.nn.init.xavier_uniform_(torch.rand(in_dim, in_dim)))  
        
        self.incidence_m = nn.Parameter(torch.nn.init.xavier_uniform_(torch.rand(num_node, num_node)))  
        self.lb_th = lb_th

    def forward(self, cur, pre):
        
        
        self_node = torch.eye(self.num_node)
        self.recon_loss = 0
        batch = len(cur)
        cur = cur.detach()
        pre = pre.detach()
        for node_idx in range(self.num_node):
            master_node_fea = cur[:,node_idx,:]  
            master_node_proj_fea = torch.matmul(master_node_fea, self.r_proj).reshape(batch, -1)  
            slave_node_idx = [i for i in range(self.num_node)]  
            node_linear_comb = self.incidence_m[node_idx].unsqueeze(0)  
            node_linear_comb = torch.clamp(node_linear_comb, min=self.lb_th)

            node_linear_comb_mask = node_linear_comb > self.lb_th
            node_linear_comb = node_linear_comb.masked_fill(~node_linear_comb_mask, value=torch.tensor(0))
            neigh_recon_fea = torch.matmul(node_linear_comb, pre).reshape(batch,-1)  
            
            
            linear_comb_l1 = torch.max(torch.sum(torch.abs(node_linear_comb), dim=0))
            

            linear_comb_l2 = torch.sqrt(torch.sum(node_linear_comb.pow(2)))
            

            recon_error = torch.sqrt(torch.sum((master_node_proj_fea - neigh_recon_fea).pow(2),dim=-1)).reshape(batch,-1)  
            
                
                
                
            node_recons_loss = recon_error * self.recons_lamda + linear_comb_l1 + self.l2_lamda * linear_comb_l2
                
            self.recon_loss += node_recons_loss
        
        return self.recon_loss, self.incidence_m  



class SpatialHyperedgeMP(nn.Module):

    def __init__(self, num_node):
        super(SpatialHyperedgeMP, self).__init__()
        

    def get_head_tail(self, edge, num_node):
        edge_num = len(edge)
        head = torch.zeros((edge_num, num_node))
        for i, idx in enumerate(edge):
            for j, jdx in enumerate(idx):
                head[i][jdx] = 1
        return head

    def construct_directed(self, incidence_m):
        
        num_node = len(incidence_m)
        incidence_mask = incidence_m > 0
        
        head_edge = []
        tail_edge = []
        directed_edge = {}
        marked_node = []
        for node_idx in range(num_node):

            tail_edge_temp = []
            if node_idx in marked_node:
                continue
            marked_node.append(node_idx)
            tail_edge_temp.append(node_idx)
            head_edge_temp = (incidence_mask[node_idx] == True).nonzero().view(-1)
            for slave_node_idx in range(num_node):
                if slave_node_idx in marked_node or slave_node_idx == node_idx:
                    continue
                if incidence_mask[node_idx].equal(incidence_mask[slave_node_idx]):
                    tail_edge_temp.append(slave_node_idx)
                    marked_node.append(slave_node_idx)
            head_edge.append(head_edge_temp)
            tail_edge.append(tail_edge_temp)

        

        head = self.get_head_tail(head_edge, num_node)  
        tail = self.get_head_tail(tail_edge, num_node)
        return head

    def forward(self, cur, incidence_m):
        
        head = self.construct_directed(incidence_m)
        incidence_m += F.normalize(torch.tensor(head), dim=-1)
        batch = len(cur)
        edge_fea = torch.matmul(incidence_m, cur)  
        edge_degree = torch.sum(incidence_m, dim=-1).reshape(-1, 1)  
        edge_fea_normed = torch.div(edge_fea, edge_degree)
        return edge_fea_normed  



class TemporalHyperedgeMP(nn.Module):

    def __init__(self):
        super(TemporalHyperedgeMP, self).__init__()

    def forward(self, cur, pre, incidence_m):
        
        edge_fea = torch.matmul(incidence_m, pre) + cur  
        self_degree = torch.ones(incidence_m.shape[0], 1)  
        edge_degree = torch.sum(incidence_m, dim=1).reshape(-1, 1) + self_degree
        edge_fea_normed = torch.div(edge_fea, edge_degree)  
        return edge_fea_normed  




class HHNodeMP(nn.Module):

    def __init__(self, in_dim = 32, num_node = 16, drop_rate = 0.3):
        super(HHNodeMP, self).__init__()
        self.node_proj = nn.Parameter(torch.nn.init.xavier_uniform_(torch.rand(in_dim, in_dim)))  
        self.spatial_edge_proj = nn.Parameter(torch.nn.init.xavier_uniform_(torch.rand(in_dim, in_dim)))
        self.temporal_edge_proj = nn.Parameter(torch.nn.init.xavier_uniform_(torch.rand(in_dim, in_dim)))
        self.num_node = num_node
        self.act_1 = nn.Softmax(dim = 0)
        self.in_dim = in_dim
        self.drop = nn.Dropout(drop_rate)
        self.act = nn.ReLU(inplace=True)
        self.theta = nn.Linear(in_dim, in_dim, bias=True)
        torch.nn.init.xavier_uniform_(self.theta.weight)

    def forward(self, cur, spatial_hyperedge_emb, temporal_hyperedge_emb):
        
        rlt = []
        batch = len(cur)
        for node_idx in range(self.num_node):
            node_fea = cur[:,node_idx,:]  
            node_fea = (torch.matmul(node_fea, self.node_proj)).reshape(batch,-1,1)  

            spatial_hyperedge_fea = spatial_hyperedge_emb[:,node_idx,:]  
            temporal_hyperedge_fea = temporal_hyperedge_emb[:,node_idx,:]  
            spatial_hyperedge_fea = torch.matmul(spatial_hyperedge_fea, self.spatial_edge_proj)  
            temporal_hyperedge_fea = torch.matmul(temporal_hyperedge_fea, self.temporal_edge_proj)  

            
            
            hyperedge = torch.cat((spatial_hyperedge_fea[:,None,:],temporal_hyperedge_fea[:,None,:]),dim=1)  

            atten = self.act_1(torch.matmul(hyperedge, node_fea)/math.sqrt(self.in_dim)).reshape(batch,-1, 1)  
            rlt.append(torch.sum(torch.mul(atten, hyperedge), dim=1).reshape(batch,1, -1))  
        concatenated = torch.stack(rlt, dim=1).reshape(batch,-1,32)  
        return self.drop(self.act(self.theta(concatenated)))  


class TimeBlock(nn.Module):
    def __init__(self, in_dim=32, num_node=16):
        super(TimeBlock, self).__init__()
        self.spatial_hyperedge = SpatialHyperedge(in_dim)
        self.temporal_hyperedge = TemporalHyperedge(in_dim)
        self.spatial_hyperedge_MP = SpatialHyperedgeMP(num_node)
        self.temporal_hyperedge_MP = TemporalHyperedgeMP()
        self.node_mp = HHNodeMP()

    def forward(self, pre, cur):  
        """
        :param cur: (?-1)*N * d
        :param pred: (?-1)*N * d
        :return: (?-1)*N * d
        """
        spatial_hyperedge_loss, spatial_hyperedge_incidence = self.spatial_hyperedge(cur)  
        temporal_hyperedge_loss, temporal_hyperedge_incidence = self.temporal_hyperedge(cur, pre)
        spatial_hyperedge_emb = self.spatial_hyperedge_MP(cur, spatial_hyperedge_incidence)  
        temporal_hyperedge_emb = self.temporal_hyperedge_MP(cur, pre, temporal_hyperedge_incidence)  
        node_emb = self.node_mp(cur, spatial_hyperedge_emb, temporal_hyperedge_emb) 
        return node_emb, temporal_hyperedge_loss+spatial_hyperedge_loss  


class HyperModule(nn.Module):
    """
    multi-timestamp training
    """
    def __init__(self, win_size, h_dim=32, n_cats=4, recons_lambda=0.1):
        super(HyperModule, self).__init__()
        self.win_size = win_size
        self.time_cursor = TimeBlock()
        self.predictor = Predictor(h_dim, n_cats=n_cats)
        self.readout = Readout(h_dim)
        self.recons_lambda = recons_lambda

    def forward(self, node_fea):
        recon_loss = 0
        
        for i in range(self.win_size-1):  
            if i == 0:
                pre_node = node_fea[:-1,:,:]  
                cur_node = node_fea[1:,:,:]
                cur_node_emb, r_loss = self.time_cursor(pre_node, cur_node)  
                recon_loss += r_loss
            else:
                cur_node = node_fea[i + 1].contiguous()
                pre_node = cur_node_emb.contiguous()
                cur_node_emb,r_loss = self.time_cursor(pre_node, cur_node)
                recon_loss += r_loss
        graph_emb = self.readout(cur_node_emb, node_fea[1:,:,:])  
        logits = self.predictor(graph_emb)  
        
        return logits, recon_loss*self.recons_lambda  



class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim=20, hidden_dim=8, action_dim=4):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):  
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim= -1)  


class QValueNet(torch.nn.Module):
    
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        
        self.HyperModule = HyperModule(win_size=2)
        self.fc2 = torch.nn.Linear(action_dim, 1)

    def forward(self, x_pair):  
        x = F.relu(self.fc1(x_pair))  
        total_logits, total_r_loss = self.HyperModule(x)  
        total_logits = self.fc2(total_logits)  
        
        return total_logits, total_r_loss


class HMAPPONet(nn.Module):
    def __init__(self, state_dim=20, hidden_dim=8, action_dim=4, actor_lr=1e-3, critic_lr=1e-2):
        super(HMAPPONet,self).__init__()
        
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim)
        
        self.critic = QValueNet(state_dim, hidden_dim, action_dim)


        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = 0.91
        self.lmbda = 0.86
        self.epochs = 20  
        self.eps = 0.3  
        self.alpha_hyper = 0.001 


class HMaPPOAgent(Agent):
    def __init__(self,
                 dic_agent_conf=None,
                 dic_traffic_env_conf=None,
                 dic_path=None,
                 cnt_round=None,
                 best_round=None,
                 intersection_id="0",
                 bar_round=None
                 ):

        super(HMaPPOAgent, self).__init__(
            dic_agent_conf, dic_traffic_env_conf, dic_path, intersection_id)  

        
        self.num_agents = dic_traffic_env_conf['NUM_INTERSECTIONS']
        self.num_actions = len(
            self.dic_traffic_env_conf["PHASE"][self.dic_traffic_env_conf['SIMULATOR_TYPE']])  
        self.num_lanes = np.sum(np.array(list(self.dic_traffic_env_conf["LANE_NUM"].values())))  
        self.memory = self.build_memory()  
        self.round_cur = cnt_round
        
        state_dim = 20
        hidden_dim = 32
        action_dim = 4
        actor_lr = 3e-4
        critic_lr = 3e-4

        if cnt_round == 0:
            
            self.HmaPPONet = HMAPPONet(state_dim, hidden_dim, action_dim, actor_lr, critic_lr)  

            if os.listdir(self.dic_path["PATH_TO_MODEL"]):  
                self.HmaPPONet.load_state_dict(torch.load(
                    os.path.join(self.dic_path["PATH_TO_MODEL"], "round_0_inter_{0}.pth".format(intersection_id))))  
            
        else:
            
            try:
                if best_round:
                    pass
                else:
                    self.load_network("round_{0}_inter_{1}".format(max(cnt_round - 1,0), self.intersection_id))  
            except:
                print("fail to load network, current round: {0}".format(cnt_round))

        
        """
        "EPSILON": 0.8,
        "EPSILON_DECAY": 0.95,
        "MIN_EPSILON": 0.2,
        """
        if os.path.exists(os.path.join(self.dic_path["PATH_TO_MODEL"],"round_-1_inter_{0}.pth".format(intersection_id))):
            
            self.dic_agent_conf["EPSILON"] = self.dic_agent_conf["MIN_EPSILON"]
            print('round%d, EPSILON:%.4f' % (cnt_round, self.dic_agent_conf["EPSILON"]))
        else:
            decayed_epsilon = self.dic_agent_conf["EPSILON"] * pow(self.dic_agent_conf["EPSILON_DECAY"],
                                                                   cnt_round)  
            self.dic_agent_conf["EPSILON"] = max(decayed_epsilon, self.dic_agent_conf["MIN_EPSILON"])  


    
    def action_att_predict(self, state, total_features=[],  bar=False):
        
        batch_size = len(state)  
        
        if total_features == []:
            total_features = list()
            for i in range(batch_size):
                feature = [] 
                for j in range(self.num_agents):
                    observation = []  
                    for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                        if 'adjacency' in feature_name:
                            continue
                        if feature_name == "cur_phase":
                            if len(state[i][j][feature_name]) == 1:
                                
                                observation.extend(
                                    self.dic_traffic_env_conf['PHASE'][self.dic_traffic_env_conf['SIMULATOR_TYPE']]
                                    [state[i][j][feature_name][0]])
                            else:
                                observation.extend(state[i][j][feature_name])
                        elif feature_name == "lane_num_vehicle":
                            observation.extend(state[i][j][feature_name])
                    feature.append(observation)  
                total_features.append(feature)  
            
            total_features = np.reshape(np.array(total_features), [batch_size, self.num_agents, -1])  
            
        total_features = torch.tensor(total_features, dtype=torch.float)
        all_output = self.HmaPPONet.actor(total_features)  
        action = all_output

        
        if len(action) > 1:
            
            
            return total_features 

        
        action = action.detach().numpy()  
        max_action = np.expand_dims(np.argmax(action, axis=-1), axis=-1)  
        random_action = np.reshape(np.random.randint(self.num_actions, size=1 * self.num_agents),
                                   (1, self.num_agents, 1))
        
        possible_action = np.concatenate([max_action, random_action], axis=-1)  
        selection = np.random.choice(
            [0, 1],
            size=batch_size * self.num_agents,
            p=[1 - self.dic_agent_conf["EPSILON"], self.dic_agent_conf["EPSILON"]])
        act = possible_action.reshape((batch_size * self.num_agents, 2))[
            np.arange(batch_size * self.num_agents), selection]
        act = np.reshape(act, (batch_size, self.num_agents))
        return act 


    
    def choose_action(self, count, state):
        """
        input: state:[batch_size,num_agent,feature]，默认这里的batch_size是1
        output: out:[batch_size,num_agent,action]
        """
        act = self.action_att_predict([state])  
        return act[0]  


    
    def prepare_Xs_Y(self, memory, dic_exp_conf):
        
        ind_end = len(memory)  
        print("memory size before forget: {0}".format(ind_end))
        
        if dic_exp_conf["PRETRAIN"] or dic_exp_conf["AGGREGATE"]:
            sample_slice = memory  
        
        else:
            
            ind_sta = max(0, ind_end - 1000)  
            memory_after_forget = memory[ind_sta: ind_end]  
            print("memory size after forget:", len(memory_after_forget))

            
            
            sample_size = min(50, len(memory_after_forget))  
            start = random.randint(0,len(memory_after_forget)-sample_size)
            sample_slice = memory_after_forget[start:start + sample_size]
            
            print("memory samples number:", sample_size)

        
        _state = []
        _next_state = []
        _action = []
        _reward = []

        
        for i in range(len(sample_slice)):
            _state.append([])
            _next_state.append([])
            _action.append([])
            _reward.append([])
            for j in range(self.num_agents):
                state, action, next_state, reward, _ = sample_slice[i][j]
                _state[i].append(state)
                _next_state[i].append(next_state)
                _action[i].append(action)
                _reward[i].append(reward)


        
        _features = self.action_att_predict(_state)
        _next_features = self.action_att_predict(_next_state)  

        
        self.transition_dict = {
            'states':_features,
            'actions':_action,
            'next_states':_next_features,
            'rewards':_reward,
            'dones':None
        }
        return


    
    def build_memory(self):

        return []

    
    def load_network(self, file_name, file_path=None):
        if file_path == None:
            file_path = self.dic_path["PATH_TO_MODEL"]

        state_dim = 20
        hidden_dim = 32
        action_dim = 4
        actor_lr = 3e-4
        critic_lr = 3e-4
        self.HmaPPONet = HMAPPONet(state_dim, hidden_dim, action_dim, actor_lr, critic_lr)  
        
        
        self.HmaPPONet.load_state_dict(torch.load(os.path.join(file_path, "%s.pth" % file_name)))
        self.HmaPPONet.eval()
        print("succeed in loading model %s" % file_name)

    def save_network(self, file_name):
        
        torch.save(self.HmaPPONet.state_dict(), os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.pth" % file_name))


    def train_network(self, dic_exp_conf):

            if dic_exp_conf["PRETRAIN"] or dic_exp_conf["AGGREGATE"]:
                epochs = 1000
            else:
                epochs = self.dic_agent_conf["EPOCHS"]  
            
            
            
            for i in range(60):
                self.update(self.transition_dict,i)
            

    
    def compute_advantage(self, gamma, lmbda, td_delta):  
        
        td_delta = td_delta.detach().numpy()  
        advantage_list = []  
        advantage = 0.0
        for delta in td_delta[::-1]:  
            
            advantage = gamma*lmbda*advantage + delta
            advantage_list.append(advantage) 
        advantage_list.reverse()
        
        advantage_list = torch.tensor(advantage_list, dtype=torch.float)
        advantage_list = ((advantage_list - torch.mean(advantage_list, dim=0, keepdim=True)) / (torch.std(advantage_list, dim=0,keepdim=True) + 1e-5))
        return advantage_list



    
    def update(self, transition_dict, i):
        states = torch.tensor(transition_dict['states'], dtype=torch.float)  
        actions = torch.tensor(transition_dict['actions']).view(-1, self.num_agents ,1) 
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, self.num_agents, 1)  
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float)  
        

        
        critic_ns_value, _ = self.HmaPPONet.critic(next_states) 
        rewards = rewards[1:]  
        td_target =  rewards + self.HmaPPONet.gamma * critic_ns_value  

        critic_value, _ = self.HmaPPONet.critic(states)  
        td_delta = td_target - critic_value  
        
        advantage = self.compute_advantage(self.HmaPPONet.gamma, self.HmaPPONet.lmbda, td_delta)  
        old_q_a = self.HmaPPONet.actor(states).gather(-1, actions)
        old_log_probs = torch.log(old_q_a+1e-8).detach()  

        old_log_probs = old_log_probs[1:]  
        for _ in range(self.HmaPPONet.epochs):
            q_a = self.HmaPPONet.actor(states).gather(-1, actions)
            log_probs = torch.log(q_a + 1e-8)  
            log_probs = log_probs[1:]  

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage  
            surr2 = torch.clamp(ratio, 1 - self.HmaPPONet.eps, 1 + self.HmaPPONet.eps) * advantage  
            min_surr = -torch.min(surr1, surr2).reshape(len(surr1),-1)
            actor_loss = torch.mean(min_surr) 
            critic_value_epoch, critic_recon_loss_epoch = self.HmaPPONet.critic(states)  
            critic_value_epoch.reshape(len(surr1),-1)
            td_target.reshape(len(surr1),-1)
            critic_loss_recon = self.HmaPPONet.alpha_hyper*torch.mean(critic_recon_loss_epoch)
            critic_loss_self = (1-self.HmaPPONet.alpha_hyper)*torch.mean(F.mse_loss(critic_value_epoch, td_target.detach()))
            critic_loss = critic_loss_recon + critic_loss_self
            self.HmaPPONet.critic_optimizer.zero_grad()
            self.HmaPPONet.actor_optimizer.zero_grad()
            critic_loss.backward()
            actor_loss.backward()
            self.HmaPPONet.critic_optimizer.step()
            self.HmaPPONet.actor_optimizer.step()

        print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
        print('round:' + str(self.round_cur) + ' Epoch:' + str(i + 1) + ' critic_1_loss:' + str(critic_loss))
        print('round:' + str(self.round_cur) + ' Epoch:' + str(i + 1) + ' actor_loss:' + str(actor_loss))


