import os
import sys
sys.path.append(os.getcwd())

from src.argument_parse import args
import dgl
import dgl.nn as dglnn
import dgl.function as dglfn
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import heapq
import numpy as np
import time
from utils import seed_setting
from src.douban_data_preprocess_v1 import DataSet

if args.model == 'multi_weight_rgcn':
    from src.new_graph_encoder import GraphEncoder
elif args.model == 'hybrid_rgcn':
    from src.hybrid_graph_encoder import GraphEncoder
else:
    from src.graph_encoder import GraphEncoder

def main():
    print(args)
    seed_setting(args.random_seed)
    model = Model(args)
    model.run()

class Model():
    def __init__(self, args):
        self.data_dict = {}
        self.dataName_A = args.data_domains[0]
        self.dataName_B = args.data_domains[1]
        self.dataName_C = args.data_domains[2]

        self.dataSet_A = DataSet(self.dataName_A, None)
        self.dataSet_B = DataSet(self.dataName_B, None)
        self.dataSet_C = DataSet(self.dataName_C, None)
        self.shape_A = self.dataSet_A.shape
        self.maxRate_A = self.dataSet_A.maxRate
        self.shape_B = self.dataSet_B.shape
        self.maxRate_B = self.dataSet_B.maxRate
        self.shape_C = self.dataSet_C.shape
        self.maxRate_C = self.dataSet_C.maxRate
        self.train_A = self.dataSet_A.train
        self.test_A = self.dataSet_A.test
        self.train_B = self.dataSet_B.train
        self.test_B = self.dataSet_B.test
        self.train_C = self.dataSet_C.train
        self.test_C = self.dataSet_C.test

        self.test_A_all, self.test_A_pos, self.test_A_neg = self.dataSet_A.getTestNeg(self.test_A, 99)
        self.test_B_all, self.test_B_pos, self.test_B_neg = self.dataSet_B.getTestNeg(self.test_B, 99)
        self.test_C_all, self.test_C_pos, self.test_C_neg = self.dataSet_C.getTestNeg(self.test_C, 99)

        # self.domains = self.get_domains(args.domains, args.filter_domains)

        self.train_graph = self.generate_graph("train")
        self.train_dataloader = self.generate_graph_dataloader(self.train_graph, args.n_layer)

        self.graph_encoder = GraphEncoder(self.train_graph, args.train_and_eval_domains, args.projection_on)
        if args.device == 'gpu':
            self.graph_encoder.to(torch.device("cuda:0"))
        self.optimizer = torch.optim.Adam(self.graph_encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # def get_domains(self, domains, filter_domains):
    #     domains = []
    #     for domain in args.domains:
    #         if domain not in args.filter_domains:
    #             domains.append(domain)
    #     return domains

    def get_data_set(self, domain, task):
        if domain == self.dataName_A:
            if task == 'train':
                return self.train_A
            elif task == 'test_pos':
                return self.test_A_pos
            elif task == 'test_neg':
                return self.test_A_neg
        elif domain == self.dataName_B:
            if task == 'train':
                return self.train_B
            elif task == 'test_pos':
                return self.test_B_pos
            elif task == 'test_neg':
                return self.test_B_neg
        elif domain == self.dataName_C:
            if task == 'train':
                return self.train_C
            elif task == 'test_pos':
                return self.test_C_pos
            elif task == 'test_neg':
                return self.test_C_neg
        return None

    def get_num_nodes(self):
        num_nodes_dict = {'user': max(self.shape_A[0], self.shape_B[0], self.shape_C[0])}
        for domain in args.graph_domains:
            if domain == self.dataName_A:
                num_nodes_dict[domain] = self.shape_A[1]
            elif domain == self.dataName_B:
                num_nodes_dict[domain] = self.shape_B[1]
            elif domain == self.dataName_C:
                num_nodes_dict[domain] = self.shape_C[1]
        return num_nodes_dict

    def generate_graph(self, task='train'):
        edge_dict = {}
        for each_domain in args.graph_domains:
            edge_dict[('user', 'user_click_' + each_domain, each_domain)] = [[], []]
            edge_dict[(each_domain, each_domain + '_click_by_user', 'user')] = [[], []]

            dataset = self.get_data_set(each_domain, task)

            for line in dataset:
                userid, itemid, rating = line[0], line[1], line[2]
                edge_dict[('user', 'user_click_' + each_domain, each_domain)][0].append(userid)
                edge_dict[('user', 'user_click_' + each_domain, each_domain)][1].append(itemid)

                edge_dict[(each_domain, each_domain + '_click_by_user', 'user')][0].append(itemid)
                edge_dict[(each_domain, each_domain + '_click_by_user', 'user')][1].append(userid)

            edge_dict[('user', 'user_click_' + each_domain, each_domain)] = (
                list(map(int, edge_dict[('user', 'user_click_' + each_domain, each_domain)][0])),
                list(map(int, edge_dict[('user', 'user_click_' + each_domain, each_domain)][1]))
            )
            edge_dict[(each_domain, each_domain + '_click_by_user', 'user')] = (
                list(map(int, edge_dict[(each_domain, each_domain + '_click_by_user', 'user')][0])),
                list(map(int, edge_dict[(each_domain, each_domain + '_click_by_user', 'user')][1]))
            )

        if args.add_self:
            for type in self.get_num_nodes():
                if type in args.self_loop_filter:
                    print('self loop, filter ' + type)
                    continue
                edge_dict[(type, type + '_self_loop', type)] = (
                    list(range(self.get_num_nodes()[type])), list(range(self.get_num_nodes()[type]))
                )

        graph_format = dgl.heterograph(edge_dict, num_nodes_dict=self.get_num_nodes())
        # for each_node_type in args.graph_domains + ['user']:
        #     tensor = torch.Tensor(list(range(graph_format.num_nodes(each_node_type))))
        #     graph_format.nodes[each_node_type].data['feature'] = tensor
        return graph_format

    def generate_graph_dataloader(self, mother_graph, layer_num):
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(layer_num)
        eid_dict = {}
        for domain in args.train_and_eval_domains:
            etype = ('user', 'user_click_' + domain, domain)
            eid_dict[etype] = list(range(mother_graph.number_of_edges(etype)))

        reverse_etypes = {}
        for domain in args.train_and_eval_domains:
            etype = ('user', 'user_click_' + domain, domain)
            rtype = (domain, domain + '_click_by_user', 'user')
            reverse_etypes[etype] = rtype
            reverse_etypes[rtype] = etype

        dataloader = dgl.dataloading.EdgeDataLoader(
            g=mother_graph, eids=eid_dict, block_sampler=sampler, exclude='reverse_types', reverse_etypes=reverse_etypes,
            negative_sampler=dgl.dataloading.negative_sampler.Uniform(args.k_negative_num),
            batch_size=args.batch_size, shuffle=True, drop_last=True
        )
        return dataloader

    def eval(self, all_emb, test_data, domain):
        if domain not in args.train_and_eval_domains:
            return -1.0, -1.0
        def getHitRatio(ranklist, targetItem):
            for item in ranklist:
                if item == targetItem:
                    return 1
            return 0

        def getNDCG(ranklist, targetItem):
            for i in range(len(ranklist)):
                item = ranklist[i]
                if item == targetItem:
                    return math.log(2) / math.log(i + 2)
            return 0
        test_user = test_data[0]
        test_item = test_data[1]
        hr_list = []
        ndcg_list = []
        input_hr_list = []
        input_ndcg_list = []
        for i in range(len(test_user)):
            target_item = test_item[i][0]
            user_emb = all_emb['layer_' + str(args.n_layer) + '_feats']['output'][domain + '_user_output'][test_user[i]]
            item_emb = all_emb['layer_' + str(args.n_layer) + '_feats']['output'][domain + '_item_output'][test_item[i]]
            score = torch.sum(user_emb * item_emb, dim=1)
            item_score_dict = {}
            for j in range(len(test_item[i])):
                item_score_dict[test_item[i][j]] = score[j]
            ranklist = heapq.nlargest(10, item_score_dict, key=item_score_dict.get)
            tmp_hr = getHitRatio(ranklist, target_item)
            tmp_ndcg = getNDCG(ranklist, target_item)
            if tmp_hr == 0:
                target_user = test_user[i][0]
                sorted_list = sorted(item_score_dict.items(), key = lambda x: x[1], reverse=True)
                sorted_list = [(x[0], round(x[1].item(), 6)) for x in sorted_list]
                for k in range(len(sorted_list)):
                    if sorted_list[k][0] == target_item:
                        target_item_index = k
                        break
                A_len = self.dataSet_A.userLenDict[target_user] if target_user in self.dataSet_A.userLenDict else 0
                B_len = self.dataSet_B.userLenDict[target_user] if target_user in self.dataSet_B.userLenDict else 0
                C_len = self.dataSet_C.userLenDict[target_user] if target_user in self.dataSet_C.userLenDict else 0
                # print("domain:{},user:{},item:{},rank:{},A_len:{},B_len:{},C_len:{}".format(
                #     domain, target_user, target_item, target_item_index, A_len, B_len, C_len))

            hr_list.append(tmp_hr)
            ndcg_list.append(tmp_ndcg)
        #临时测试下最底层emb的效果
        for i in range(len(test_user)):
            target_item = test_item[i][0]
            user_emb = all_emb['input_feats']['user'][test_user[i]]
            item_emb = all_emb['input_feats'][domain][test_item[i]]
            score = torch.sum(user_emb * item_emb, dim=1)
            item_score_dict = {}
            for j in range(len(test_item[i])):
                item_score_dict[test_item[i][j]] = score[j]
            ranklist = heapq.nlargest(10, item_score_dict, key=item_score_dict.get)
            tmp_hr = getHitRatio(ranklist, target_item)
            tmp_ndcg = getNDCG(ranklist, target_item)
            input_hr_list.append(tmp_hr)
            input_ndcg_list.append(tmp_ndcg)
        print('domain:{},hr:{}, ndcg:{}, input_hr:{}, input_ndcg:{}'.format(domain, np.mean(hr_list), np.mean(ndcg_list), np.mean(input_hr_list), np.mean(input_ndcg_list)))
        return np.mean(hr_list), np.mean(ndcg_list)

    def run(self):
        best_hr_A, best_hr_B, best_hr_C = -1, -1, -1
        best_ndcg_A, best_ndcg_B, best_ndcg_C = -1, -1, -1

        for epoch in range(args.n_epochs):
            # train
            edge_num = sum([self.train_graph.num_edges(('user', 'user_click_' + domain, domain)) for domain in args.train_and_eval_domains])
            print('epochs: ', args.n_epochs, 'edge_num: ', edge_num, 'batch_num: ', edge_num // (args.batch_size))
            loss_list = []
            count = 0
            t0 = time.time()
            for input_nodes, positive_graph, negative_graph, blocks in self.train_dataloader:
                t1 = time.time()
                if args.device == 'gpu':
                    blocks = [b.to(torch.device('cuda')) for b in blocks]
                    positive_graph = positive_graph.to(torch.device('cuda'))
                    negative_graph = negative_graph.to(torch.device('cuda'))
                    input_nodes = {k: v.cuda() for k, v in input_nodes.items()}

                # for etype in positive_graph.etypes:
                #     print(positive_graph.edges(etype=etype))
                #     print(negative_graph.edges(etype=etype))
                emb_result = self.graph_encoder.get_graph_encoder(blocks, flag="block", train=True)
                main_loss = self.graph_encoder.get_loss(emb_result, positive_graph, negative_graph)
                # contrastive_loss = self.graph_encoder.get_contrastive_loss(emb_result)
                contrastive_loss = torch.tensor(0.0)
                # loss = main_loss + 0.01*contrastive_loss
                loss = main_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_list.append(loss.item())
                count += 1
                print('epoch: {}, count: {}, main_loss: {}, contrastive_loss: {}, loss: {}, mean_loss: {}, time:{:.6f}'.format(epoch, count, main_loss.item(), contrastive_loss.item(), loss.item(), np.mean(loss_list), time.time() - t1))
            print('epoch: {}, mean_loss: {}; time:{:.6f}'.format(epoch, np.mean(loss_list), time.time() - t0))

            # eval on test data
            with torch.no_grad():
                if args.device == 'gpu':
                    self.train_graph = self.train_graph.to(torch.device('cuda'))
                all_emb = self.graph_encoder.get_graph_encoder(self.train_graph, flag = 'graph', train = False)
                self.train_graph = self.train_graph.cpu()

                hr_A, ndcg_A = self.eval(all_emb, self.test_A_all, self.dataName_A)
                best_hr_A = max(hr_A, best_hr_A); best_ndcg_A = max(ndcg_A, best_ndcg_A)
                hr_B, ndcg_B = self.eval(all_emb, self.test_B_all, self.dataName_B)
                best_hr_B = max(hr_B, best_hr_B); best_ndcg_B = max(ndcg_B, best_ndcg_B)
                hr_C, ndcg_C = self.eval(all_emb, self.test_C_all, self.dataName_C)
                best_hr_C = max(hr_C, best_hr_C); best_ndcg_C = max(ndcg_C, best_ndcg_C)
                print('epoch: {}, hr_A: {:.6f}, ndcg_A: {:.6f}, , hr_B: {:.6f}, ndcg_B: {:.6f}, , hr_C: {:.6f}, ndcg_C: {:.6f}'.format(epoch, hr_A, ndcg_A, hr_B, ndcg_B, hr_C, ndcg_C))
        print('epochs:{}, best hr_A: {:.6f}, best ndcg_A: {:.6f}, best hr_B: {:.6f}, best ndcg_B: {:.6f}, best hr_C: {:.6f}, best ndcg_C: {:.6f}'.format(args.n_epochs, best_hr_A, best_ndcg_A, best_hr_B, best_ndcg_B, best_hr_C, best_ndcg_C))




if __name__ == '__main__':
    main()
