import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from Aggregator import MeanAggregator
from utils import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


class DArtNet(nn.Module):
    def __init__(self,
                 num_nodes,
                 h_dim,
                 num_rels,
                 dropout=0,
                 model=0,
                 seq_len=10,
                 num_k=10,
                 gamma=1):
        super(DArtNet, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.num_rels = num_rels
        self.model = model
        self.seq_len = seq_len
        self.num_k = num_k
        self.gamma = gamma
        self.rel_embeds = nn.Parameter(torch.Tensor(num_rels, h_dim))
        nn.init.xavier_uniform_(self.rel_embeds,
                                gain=nn.init.calculate_gain('relu'))

        self.ent_embeds = nn.Parameter(torch.Tensor(num_nodes, h_dim))
        nn.init.xavier_uniform_(self.ent_embeds,
                                gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

        self.f1 = nn.Linear(1 * self.h_dim, 1)
        self.f2 = nn.Linear(2 * h_dim, num_nodes)

        # For recording history in inference

        self.entity_s_his_test = None
        self.att_s_his_test = None
        self.rel_s_his_test = None
        self.self_att_s_his_test = None

        self.entity_s_his_cache = None
        self.att_s_his_cache = None
        self.rel_s_his_cache = None
        self.self_att_s_his_cache = None

        self.att_s_dict = {}

        self.latest_time = 0

        self.criterion = nn.CrossEntropyLoss()
        self.att_criterion = nn.MSELoss()

    """
    Prediction function in training. 
    This should be different from testing because in testing we don't use ground-truth history.
    """
    def forward(self,
                triplets,
                s_hist,
                rel_s_hist,
                att_s_hist,
                self_att_s_hist,
                o_hist,
                rel_o_hist,
                att_o_hist,
                self_att_o_hist,
                predict_both=True):
        # print('here')
        s = triplets[:, 0].type(torch.cuda.LongTensor)
        r = triplets[:, 1].type(torch.cuda.LongTensor)
        o = triplets[:, 2].type(torch.cuda.LongTensor)
        a_s = triplets[:, 3].type(torch.cuda.FloatTensor)
        a_o = triplets[:, 4].type(torch.cuda.FloatTensor)

        batch_size = len(s)

        s_hist_len = torch.LongTensor(list(map(len, s_hist))).cuda()
        s_len, s_idx = s_hist_len.sort(0, descending=True)

        if predict_both:

            ob_pred = self.f2(
                self.dropout(
                    torch.cat(
                        (self.ent_embeds[s[s_idx]], self.rel_embeds[r[s_idx]]),
                        dim=1)))
            loss_sub = self.criterion(ob_pred, o[s_idx])
        else:
            ob_pred = None
            loss_sub = 0

        sub_att_pred = self.f1(self.dropout(
            self.ent_embeds[s[s_idx]])).squeeze()

        loss_att_sub = self.att_criterion(sub_att_pred, a_s[s_idx])

        loss = loss_sub + self.gamma * loss_att_sub

        return loss, loss_att_sub, ob_pred, sub_att_pred, s_idx

    def init_history(self):
        self.entity_s_his_test = [[] for _ in range(self.num_nodes)]
        self.att_s_his_test = [[] for _ in range(self.num_nodes)]
        self.rel_s_his_test = [[] for _ in range(self.num_nodes)]
        self.self_att_s_his_test = [[] for _ in range(self.num_nodes)]

        self.entity_s_his_cache = [[] for _ in range(self.num_nodes)]
        self.att_s_his_cache = [[] for _ in range(self.num_nodes)]
        self.rel_s_his_cache = [[] for _ in range(self.num_nodes)]
        self.self_att_s_his_cache = [[] for _ in range(self.num_nodes)]

    def get_loss(self, triplets, s_hist, rel_s_hist, att_s_hist,
                 self_att_s_hist, o_hist, rel_o_hist, att_o_hist,
                 self_att_o_hist):
        loss, loss_att_sub, _, _, _ = self.forward(triplets, s_hist,
                                                   rel_s_hist, att_s_hist,
                                                   self_att_s_hist, o_hist,
                                                   rel_o_hist, att_o_hist,
                                                   self_att_o_hist)
        return loss, loss_att_sub

    """
    Prediction function in testing
    """
    def predict(self, triplets, s_hist, rel_s_hist, att_s_hist,
                self_att_s_hist, o_hist, rel_o_hist, att_o_hist,
                self_att_o_hist):

        self.att_s_dict = {}
        # self.att_o_dict = {}
        self.att_residual_dict = {}

        _, loss_att_sub, _, sub_att_pred, s_idx = self.forward(
            triplets, s_hist, rel_s_hist, att_s_hist, self_att_s_hist, o_hist,
            rel_o_hist, att_o_hist, self_att_o_hist, False)
        # print(triplets[:, 0])
        # print(s_hist)
        # print(sub_att_pred)
        indices = {}
        for i in range(len(triplets)):
            s = triplets[s_idx[i], 0].type(torch.LongTensor).item()
            o = triplets[s_idx[i], 2].type(torch.LongTensor).item()
            t = triplets[s_idx[i], 5].type(torch.LongTensor).item()
            s_att = sub_att_pred[i].cpu().item()

            if s not in self.att_s_dict:
                self.att_s_dict[s] = s_att
                indices[s] = i
            else:
                assert (self.att_s_dict[s] == s_att)

        for i in range(self.num_nodes):
            if i not in self.att_s_dict:  # and i not in self.att_o_dict:
                sub_att_pred = self.f1(self.ent_embeds[[i]]).squeeze()
                self.att_residual_dict[i] = sub_att_pred

        return loss_att_sub

    def predict_single(self, triplet, s_hist, rel_s_hist, att_s_hist,
                       self_att_s_hist, o_hist, rel_o_hist, att_o_hist,
                       self_att_o_hist):
        # print(triplet)
        s = triplet[0].type(torch.cuda.LongTensor)
        r = triplet[1].type(torch.cuda.LongTensor)
        o = triplet[2].type(torch.cuda.LongTensor)

        t = triplet[5].cpu()
        # print('here')
        if self.latest_time != t:

            for ee in range(self.num_nodes):
                if len(self.entity_s_his_cache[ee]) != 0:
                    if len(self.entity_s_his_test[ee]) >= self.seq_len:
                        self.entity_s_his_test[ee].pop(0)
                        self.att_s_his_test[ee].pop(0)
                        self.self_att_s_his_test[ee].pop(0)
                        self.rel_s_his_test[ee].pop(0)

                    self.entity_s_his_test[ee].append(
                        self.entity_s_his_cache[ee].clone())
                    self.att_s_his_test[ee].append(
                        self.att_s_his_cache[ee].clone())
                    self.self_att_s_his_test[ee].append(
                        self.self_att_s_his_cache[ee])
                    self.rel_s_his_test[ee].append(
                        self.rel_s_his_cache[ee].clone())

                    self.entity_s_his_cache[ee] = []
                    self.att_s_his_cache[ee] = []
                    self.self_att_s_his_cache[ee] = []
                    self.rel_s_his_cache[ee] = []

            self.latest_time = t

        if len(s_hist) == 0:
            s_h = torch.zeros(self.h_dim).cuda()

        else:
            if len(self.entity_s_his_test[s]) == 0:
                self.entity_s_his_test[s] = s_hist.copy()
                self.rel_s_his_test[s] = rel_s_hist.copy()
                self.att_s_his_test[s] = att_s_hist.copy()
                self.self_att_s_his_test[s] = self_att_s_hist

            s_history = self.entity_s_his_test[s]
            rel_s_history = self.rel_s_his_test[s]
            att_s_history = self.att_s_his_test[s]
            self_att_s_history = self.self_att_s_his_test[s]

        ob_pred = self.f2(
            torch.cat((self.ent_embeds[s], self.rel_embeds[r]), dim=0))

        _, o_candidate = torch.topk(ob_pred, self.num_k)

        self.entity_s_his_cache[s], self.rel_s_his_cache[
            s], self.att_s_his_cache[s], self.self_att_s_his_cache[
                s] = self.update_cache(self.entity_s_his_cache[s],
                                       self.rel_s_his_cache[s],
                                       self.att_s_his_cache[s],
                                       self.self_att_s_his_cache[s], s.cpu(),
                                       r.cpu(), o_candidate.cpu())

        return ob_pred

    def evaluate_filter(self, triplet, s_hist, rel_s_hist, att_s_hist,
                        self_att_s_hist, o_hist, rel_o_hist, att_o_hist,
                        self_att_o_hist, all_triplets):
        s = triplet[0].type(torch.cuda.LongTensor)
        r = triplet[1].type(torch.cuda.LongTensor)
        o = triplet[2].type(torch.cuda.LongTensor)
        ob_pred = self.predict_single(triplet, s_hist, rel_s_hist, att_s_hist,
                                      self_att_s_hist, o_hist, rel_o_hist,
                                      att_o_hist, self_att_o_hist)
        o_label = o
        s_label = s
        # sub_pred = torch.sigmoid(sub_pred)
        ob_pred = torch.sigmoid(ob_pred)

        ground = ob_pred[o].clone()

        s_id = torch.nonzero(
            all_triplets[:, 0].type(torch.cuda.LongTensor) == s).view(-1)
        idx = torch.nonzero(
            all_triplets[s_id, 1].type(torch.cuda.LongTensor) == r).view(-1)
        idx = s_id[idx]
        idx = all_triplets[idx, 2].type(torch.cuda.LongTensor)
        ob_pred[idx] = 0
        ob_pred[o_label] = ground

        ob_pred_comp1 = (ob_pred > ground).data.cpu().numpy()
        ob_pred_comp2 = (ob_pred == ground).data.cpu().numpy()
        rank_ob = np.sum(ob_pred_comp1) + (
            (np.sum(ob_pred_comp2) - 1.0) / 2) + 1

        return np.array([rank_ob])

    def update_cache(self, s_his_cache, r_his_cache, att_his_cache,
                     self_att_his_cache, s, r, o_candidate):
        if len(s_his_cache) == 0:
            s_his_cache = o_candidate.view(-1)
            r_his_cache = r.repeat(len(o_candidate), 1).view(-1)
            att_his_cache = []
            for key in s_his_cache:
                k = key.item()
                if k in self.att_s_dict:
                    att_his_cache.append(self.att_s_dict[k])
                # elif k in self.att_o_dict:
                #     att_his_cache.append(self.att_o_dict[k])
                else:
                    att_his_cache.append(self.att_residual_dict[k])

            if s.item() in self.att_s_dict:
                self_att_his_cache = self.att_s_dict[s.item()]
            # elif s.item() in self.att_o_dict:
            #     self_att_his_cache = self.att_o_dict[s.item()]
            else:
                self_att_his_cache = self.att_residual_dict[s.item()]

            if type(att_his_cache) != torch.Tensor:
                att_his_cache = torch.FloatTensor(att_his_cache)
        else:
            ent_list = s_his_cache[torch.nonzero(r_his_cache == r).view(-1)]
            tem = []
            for i in range(len(o_candidate)):
                if o_candidate[i] not in ent_list:
                    tem.append(i)

            if len(tem) != 0:
                forward = o_candidate[torch.LongTensor(tem)].view(-1)
                forward2 = r.repeat(len(tem), 1).view(-1)

                s_his_cache = torch.cat(
                    (torch.LongTensor(s_his_cache), forward), dim=0)
                r_his_cache = torch.cat(
                    (torch.LongTensor(r_his_cache), forward2), dim=0)
                att_his_cache = torch.cat((torch.FloatTensor(att_his_cache),
                                           forward2.type(torch.FloatTensor)),
                                          dim=0)
                # self_att_his_cache = torch.cat((self_att_his_cache, forward2),
                #                                dim=0)
                # print('---------------no')
                for i in range(len(s_his_cache)):
                    if s_his_cache[i] in ent_list:
                        # print('-------------------yes')
                        if s_his_cache[i].item() in self.att_s_dict:
                            att_his_cache[i] = self.att_s_dict[
                                s_his_cache[i].item()]
                        # elif s_his_cache[i].item() in self.att_o_dict:
                        #     att_his_cache[i] = self.att_o_dict[
                        #         s_his_cache[i].item()]
                        else:
                            att_his_cache[i] = self.att_residual_dict[
                                s_his_cache[i].item()]

                if s.item() in self.att_s_dict:
                    self_att_his_cache = self.att_s_dict[s.item()]
                # elif s.item() in self.att_o_dict:
                #     self_att_his_cache = self.att_o_dict[s.item()]
                else:
                    self_att_his_cache = self.att_residual_dict[s.item()]

        return s_his_cache, r_his_cache, att_his_cache, self_att_his_cache
