#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: jjzhou012
@contact: jjzhou012@163.com
@file: REGA.py
@time: 2020/7/22 9:22
@desc:
'''

import math
from utils.graph_op import *
from utils.load_op import *
from utils.communityDetection_op import *
import time
import random
import csv

t1 = time.time()


class robustECD_GA():
    def __init__(self, bmname, cdm, popsize, Pc, Pm, recombine_rate, sample_add_ratio, sample_del_ratio, iterNum, output_file, varlen=False):

        self.count = 0
        self.dataset = bmname
        self.cdm = cdm

        self.popsize = popsize
        self.Pc = Pc
        self.Pm = Pm
        self.recombine_rate = recombine_rate

        self.iterNum = iterNum
        self.varlen = varlen

        self.file = 'data/{}/{}.gml'.format(bmname, bmname)
        self.output_file = output_file

        self.g = nx.read_gml(self.file, label='id')
        self.labels_true = list(nx.get_node_attributes(self.g, 'commID').values())
        self.g = graph_to_undirected_unweighted(self.g)
        self.G = convert_networkx_to_igraph(self.g)
        # upper threshold
        self.sample_add_upper = int(math.ceil(sample_add_ratio * nx.number_of_edges(self.g)))
        self.sample_del_upper = int(math.ceil(sample_del_ratio * nx.number_of_edges(self.g)))

        # self.ADJ = igraph_to_sparse_matrix(self.G)

        self.labels_true = list(nx.get_node_attributes(self.g, 'commID').values())
        self.commNum_true = len(set(self.labels_true))

        self.initCommunities = community_method_dict[cdm].__wrapped__(self.labels_true, self.G)
        self.orig_Q, orig_nmi, orig_ARI, orig_f1 = evaluate_results(self.g, self.initCommunities, self.labels_true)

        self.initNumComm = len(self.initCommunities)

        # self.orig_Q = community_method_dict[cdm](labels_true, self.G, 'Q')

        self.init_pop()

    def init_pop(self):
        # pop init
        self.pop = {}
        self.nodeList = self.g.nodes()
        # print('init pop......')
        # initalize length of dna
        if self.varlen:
            if self.sample_del_upper != 0 and self.sample_add_upper != 0:
                self.addEdgeLen_list = np.random.randint(low=0, high=self.sample_add_upper, size=self.popsize).tolist()
                self.delEdgeLen_list = np.random.randint(low=0, high=self.sample_del_upper, size=self.popsize).tolist()
            elif self.sample_del_upper == 0 and self.sample_add_upper != 0:
                self.addEdgeLen_list = np.random.randint(low=0, high=self.sample_add_upper, size=self.popsize).tolist()
                self.delEdgeLen_list = np.zeros(shape=self.popsize).tolist()
            elif self.sample_add_upper == 0 and self.sample_del_upper != 0:
                self.addEdgeLen_list = np.zeros(shape=self.popsize).tolist()
                self.delEdgeLen_list = np.random.randint(low=0, high=self.sample_del_upper, size=self.popsize).tolist()
        else:
            self.addEdgeLen_list = [self.sample_add_upper] * self.popsize
            self.delEdgeLen_list = [self.sample_del_upper] * self.popsize

        allInterEdges = get_interComm_edges(self.initCommunities)
        allIntraEdges = get_intraComm_edges(self.initCommunities)

        intraEdgeAdds = set(nx.non_edges(self.g)) & set(allIntraEdges)
        interEdgeDels = set(nx.edges(self.g)) & set(allInterEdges)

        interEdgeAdds = set(nx.non_edges(self.g)) & set(allInterEdges)
        intraEdgeDels = set(nx.edges(self.g)) & set(allIntraEdges)

        # assert (interEdgeDels | intraEdgeDels) == set(nx.edges(self.g))
        # assert (interEdgeAdds | intraEdgeAdds) == set(nx.non_edges(self.g))


        if self.initNumComm > self.commNum_true:
            self.candidate_addEdges = list(interEdgeAdds | intraEdgeAdds)
            self.candidate_delEdges = list(interEdgeDels)
        elif self.initNumComm < self.commNum_true:
            self.candidate_addEdges = list(intraEdgeAdds)
            self.candidate_delEdges = list(interEdgeDels | intraEdgeDels)
        else:
            self.candidate_addEdges = list(interEdgeAdds | intraEdgeAdds)
            self.candidate_delEdges = list(interEdgeDels | intraEdgeDels)

        # generate DNA
        for i in range(self.popsize):
            # a individuals as a DNA
            DNA = {"del_edge": None,  # del edge list: [(source, target),...]
                   "add_edge": None,  # add edge list: [(source, target),...]
                   "rewired_graph": None,  # rewired graph
                   "Q": float,  # modularity
                   "fitness": float,
                   "commNum": int}

            # adj = nx.adjacency_matrix(self.g)
            # del
            # candidate_delEdges = nx.edges(self.g)
            if self.delEdgeLen_list[i] == 0:
                DNA['del_edge'] = []
            else:
                try:
                    del_i = np.random.choice(np.arange(len(self.candidate_delEdges)), self.delEdgeLen_list[i], replace=False)
                    DNA['del_edge'] = [self.candidate_delEdges[ind] for ind in del_i]
                except:
                    # print('error: too large edge to remove!')
                    DNA['del_edge'] = copy.deepcopy(self.candidate_delEdges)

            # add
            # candidate_addEdges = nx.non_edges(self.g)
            if self.addEdgeLen_list[i] == 0:
                DNA['add_edge'] = []
            else:
                try:
                    add_i = np.random.choice(np.arange(len(self.candidate_addEdges)), self.addEdgeLen_list[i], replace=False)
                    DNA['add_edge'] = [self.candidate_addEdges[ind] for ind in add_i]
                except:
                # print('error: too large edge to add!')
                    DNA['add_edge'] = copy.deepcopy(self.candidate_addEdges)

            self.pop[i] = copy.deepcopy(DNA)
            del DNA
        # print(self.pop)
        self.cal_fitness()

    def cal_fitness(self):

        for ind, DNA in self.pop.items():
            # rewire graph
            rewire_g = copy.deepcopy(self.G)
            rewire_g = rewire_graph(rewire_g, add_edges=DNA['add_edge'], del_edges=DNA['del_edge'])
            DNA['rewired_graph'] = rewire_g

            # community detection
            communities = community_method_dict[self.cdm].__wrapped__(self.labels_true, rewire_g)
            commNum_pred = len(communities)
            # cal modularity
            partition = community_2_clusterDict(communities)
            Q = modularity(partition, self.g)

            DNA['Q'] = Q
            DNA['commNum'] = commNum_pred
            # DNA['nmi'] = nmi
            # DNA['ari'] = ari
            DNA['fitness'] = 1 / math.exp(abs(self.commNum_true - commNum_pred)) * abs(Q)

    def select(self):
        # print('select individuals......')
        #  parents consist of the selected individuals
        self.selected_pop = {}
        scoreSum = sum([DNA['fitness'] for DNA in self.pop.values()])
        DNA_index = np.arange(self.popsize)
        scores = [self.pop[i]['fitness'] / scoreSum for i in DNA_index]
        # print(scores)
        for i in range(self.popsize):
            self.selected_pop[i] = copy.deepcopy(self.pop[np.random.choice(DNA_index, 1, False, scores)[0]])
        # print('selected pop: {}'.format(self.selected_pop))

    def crossover(self):
        # print('crossover parents......')
        for i in range(self.popsize // 2):
            parent_1 = self.selected_pop.popitem()[1]
            parent_2 = self.selected_pop.popitem()[1]
            # print('crossover: {}'.format(i))
            crossRateRand = np.random.random()
            if crossRateRand <= self.Pc:
                self.pop[i * 2], self.pop[i * 2 + 1] = self.dna_crossover(parent_1, parent_2)
                self.pop[i * 2]['rewired_graph'] = rewire_graph(copy.deepcopy(self.G), add_edges=self.pop[i * 2]['add_edge'],
                                                                del_edges=self.pop[i * 2]['del_edge'])
                self.pop[i * 2 + 1]['rewired_graph'] = rewire_graph(copy.deepcopy(self.G), add_edges=self.pop[i * 2 + 1]['add_edge'],
                                                                    del_edges=self.pop[i * 2 + 1]['del_edge'])
            else:
                self.pop[i * 2], self.pop[i * 2 + 1] = copy.deepcopy(parent_1), copy.deepcopy(parent_2)
        # print(self.pop)

    def dna_crossover(self, parent_1, parent_2):

        # 简化大网络交叉运算
        # if self.dataset == 'polblogs':

        # crossover add-edge
        DNA_1_edge_add = copy.deepcopy(parent_1['add_edge'])
        DNA_2_edge_add = copy.deepcopy(parent_2['add_edge'])

        dup_edge_add = set(DNA_1_edge_add) & set(DNA_2_edge_add)
        nodup_DNA_1_edge_add = set(DNA_1_edge_add) - dup_edge_add
        nodup_DNA_2_edge_add = set(DNA_2_edge_add) - dup_edge_add
        if self.varlen:
            if len(nodup_DNA_1_edge_add) != 0:
                cross_num_add_1 = np.random.randint(len(nodup_DNA_1_edge_add))
                seq_add_1 = random.sample(nodup_DNA_1_edge_add, cross_num_add_1)
            else:
                seq_add_1 = []
            if len(nodup_DNA_2_edge_add) != 0:
                cross_num_add_2 = np.random.randint(len(nodup_DNA_2_edge_add))
                seq_add_2 = random.sample(nodup_DNA_2_edge_add, cross_num_add_2)
            else:
                seq_add_2 = []
        else:
            assert len(nodup_DNA_1_edge_add) == len(nodup_DNA_2_edge_add)
            if len(nodup_DNA_1_edge_add) != 0:
                cross_num_add_1 = np.random.randint(len(nodup_DNA_1_edge_add))
                cross_num_add_2 = cross_num_add_1
                seq_add_1 = random.sample(nodup_DNA_1_edge_add, cross_num_add_1)
                seq_add_2 = random.sample(nodup_DNA_2_edge_add, cross_num_add_2)
            else:
                seq_add_1 = []
                seq_add_2 = []

        new_DNA_1_edge_add = dup_edge_add | set(seq_add_2) | (nodup_DNA_1_edge_add - set(seq_add_1))
        new_DNA_2_edge_add = dup_edge_add | set(seq_add_1) | (nodup_DNA_2_edge_add - set(seq_add_2))

        # crossover del-edge
        DNA_1_edge_del = copy.deepcopy(parent_1['del_edge'])
        DNA_2_edge_del = copy.deepcopy(parent_2['del_edge'])
        # print(DNA_1_edge_del)
        # print(DNA_2_edge_del)
        dup_edge_del = set(DNA_1_edge_del) & set(DNA_2_edge_del)
        nodup_DNA_1_edge_del = set(DNA_1_edge_del) - dup_edge_del
        nodup_DNA_2_edge_del = set(DNA_2_edge_del) - dup_edge_del

        if self.varlen:
            if len(nodup_DNA_1_edge_del) != 0:
                cross_num_del_1 = np.random.randint(len(nodup_DNA_1_edge_del))
                seq_del_1 = random.sample(nodup_DNA_1_edge_del, cross_num_del_1)
            else:
                seq_del_1 = []
            if len(nodup_DNA_2_edge_del) != 0:
                cross_num_del_2 = np.random.randint(len(nodup_DNA_2_edge_del))
                seq_del_2 = random.sample(nodup_DNA_2_edge_del, cross_num_del_2)
            else:
                seq_del_2 = []
        else:
            assert len(nodup_DNA_1_edge_del) == len(nodup_DNA_2_edge_del)
            if len(nodup_DNA_1_edge_del) != 0:
                cross_num_del_1 = np.random.randint(len(nodup_DNA_1_edge_del))
                cross_num_del_2 = cross_num_del_1
                seq_del_1 = random.sample(nodup_DNA_1_edge_del, cross_num_del_1)
                seq_del_2 = random.sample(nodup_DNA_2_edge_del, cross_num_del_2)
            else:
                seq_del_1 = []
                seq_del_2 = []

        new_DNA_1_edge_del = dup_edge_del | set(seq_del_2) | (nodup_DNA_1_edge_del - set(seq_del_1))
        new_DNA_2_edge_del = dup_edge_del | set(seq_del_1) | (nodup_DNA_2_edge_del - set(seq_del_2))

        parent_1['add_edge'] = list(new_DNA_1_edge_add)
        parent_1['del_edge'] = list(new_DNA_1_edge_del)
        parent_1['rewired_graph'] = None
        parent_2['add_edge'] = list(new_DNA_2_edge_add)
        parent_2['del_edge'] = list(new_DNA_2_edge_del)
        parent_2['rewired_graph'] = None

        return parent_1, parent_2

    def mutation(self):
        # print('individual mutation......')

        for ind, DNA in self.pop.items():
            # print('mutation: {}'.format(ind))
            # array of mutation rate
            mutation_rate_arr = sp.csr_matrix(np.random.rand(self.G.vcount(), self.G.vcount()))
            # rewired adj
            adj = igraph_to_sparse_matrix(DNA['rewired_graph']).tocsr()
            # enhance scheme
            scheme = adj - nx.adjacency_matrix(self.g)

            mutation_arr = sp.triu(mutation_rate_arr.multiply(scheme), k=1).toarray()

            # no mutation edge index
            x_add_fixed, y_add_fixed = np.where(mutation_arr > self.Pm)
            x_del_fixed, y_del_fixed = np.where(mutation_arr < -self.Pm)

            # mutation edge index
            x_add_m, y_add_m = np.where((mutation_arr < self.Pm) & (mutation_arr > 0))
            x_del_m, y_del_m = np.where((mutation_arr > -self.Pm) & (mutation_arr < 0))

            edge_fixed_add = set(zip(x_add_fixed, y_add_fixed))
            edge_fixed_del = set(zip(x_del_fixed, y_del_fixed))

            # orig_zero_x, orig_zero_y = sp.triu(np.ones(shape=self.ADJ.shape) - self.ADJ.toarray() - np.eye(self.G.vcount()), k=1).nonzero()
            # edge_add_cand = [(x, orig_zero_y[i]) for i, x in enumerate(orig_zero_x)]
            # edge_del_cand = self.G.get_edgelist()

            # add mutation
            for i in range(len(x_add_m)):
                while True:
                    add_edge = random.choice(self.candidate_addEdges)
                    if add_edge not in edge_fixed_add and (add_edge[1], add_edge[0]) not in edge_fixed_add:
                        if add_edge != (x_add_m[i], y_add_m[i]) and add_edge != (y_add_m[i], x_add_m[i]):
                            edge_fixed_add.add(add_edge)
                            # print(add_edge)
                        break
            # del mutation
            for i in range(len(x_del_m)):
                while True:
                    del_edge = random.choice(self.candidate_delEdges)
                    if del_edge not in edge_fixed_del and (del_edge[1], del_edge[0]) not in edge_fixed_del:
                        if del_edge != (x_del_m[i], y_del_m[i]) and del_edge != (y_del_m[i], x_del_m[i]):
                            edge_fixed_del.add(del_edge)
                            # print(del_edge)
                        break

            DNA['add_edge'] = list(edge_fixed_add)
            DNA['del_edge'] = list(edge_fixed_del)
            # print(self.pop[ind])
            # break

    def pop_recombine(self):
        # 父代排序
        last_fitness_list = sorted(self.last_pop.items(), key=lambda item: item[1]["fitness"], reverse=True)  # last_pop: 按fitness从大到小排序
        # 子代排序
        now_fitness_list = sorted(self.pop.items(), key=lambda item: item[1]["fitness"], reverse=False)  # pop: 按fitness从小到大排序
        # print("last_pop_best_Q: ", last_fitness_list[0][1]["Q"])
        # print("now_pop_best_Q: ", now_fitness_list[-1][1]["Q"])
        # 重组长度
        recombine_size = int(math.ceil(self.popsize * self.recombine_rate))
        # 重组
        recombine_pop = copy.deepcopy(last_fitness_list[:recombine_size]) + copy.deepcopy(now_fitness_list[recombine_size:])

        self.pop = {}
        # 因为重组后索引可能重复，不能直接转列表，需要重新赋予索引
        for i in range(len(recombine_pop)):
            self.pop[i] = copy.deepcopy(recombine_pop[i][1])

    def output_info(self, now_iter_num, iter_num):
        now_fitness_list = sorted(self.pop.items(), key=lambda item: item[1]["fitness"], reverse=True)

        best_enhance = copy.deepcopy(now_fitness_list[0])

        best_QValue = best_enhance[1]["Q"]
        best_fitness = best_enhance[1]["fitness"]
        best_del = best_enhance[1]["del_edge"]
        best_add = best_enhance[1]["add_edge"]
        best_commNum = best_enhance[1]["commNum"]

        rg = best_enhance[1]["rewired_graph"]
        comm = community_method_dict[self.cdm].__wrapped__(self.labels_true, rg)
        Q, nmi, ari, f1 = evaluate_results(self.file, comm, self.labels_true)
        # print("beat_add: ", best_add)
        # print("best_del: ", best_del)
        # print("best_fitness: ", best_fitness)
        # print("best_QValue: ", best_QValue)
        # print('cluster num: ', best_enhance[1]["commNum"])

        self.output_list = [now_iter_num, best_fitness, best_commNum, best_QValue, nmi, ari, best_del, best_add]

        return best_fitness, best_add, best_del, best_commNum

    def run(self):
        """
        进化迭代， 1.轮盘赌选择 ---> 2.交叉  ---> 3.变异  ---> 4.种群重组
        :param iter_num:
        :return:
        """
        with open(self.output_file, "w", newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["iter_num", "best_fitness", "average_fitness", "best_QValue", "average_QValue", "del_edge", "add_edge"])

            fitness = 0
            for i in range(self.iterNum):
                # print(i, self.iterNum)
                # print("iter num:" + str(i + 1) + ".................................")
                # 保存上一代种群信息
                self.last_pop = copy.deepcopy(self.pop)
                # 轮盘赌选择DNA组成pop
                self.select()  # return self.selected_pop
                # 交叉
                self.crossover()  # 更新 self.pop  中的 del_edge、add_edge
                # 变异
                self.mutation()  # 更新 self.pop  中的 del_edge、add_edge
                # 计算fitness
                self.cal_fitness()  # 更新 self.pop 中的 Q、fitness
                # 种群重组
                self.pop_recombine()  # 更新 self.pop
                # 打印保存每代最优
                best_fitness, best_add, best_del, best_commNum = self.output_info(now_iter_num=i + 1,
                                                                                  iter_num=self.iterNum)  # return self.output_list
                writer.writerow(self.output_list)

                # 如果fitness 200次没增长，默认结束
                if best_fitness == fitness:
                    self.count += 1
                else:
                    self.count = 0
                fitness = best_fitness
                if self.count == 50 or i == self.iterNum - 1:
                    # save update graph
                    # print(self.output_file)
                    save_path = self.output_file[:-7] + 'enhance_g.gml'
                    # print(save_path)
                    self.enhance_graph = rewire_graph(copy.deepcopy(self.G), add_edges=best_add, del_edges=best_del, save_path=save_path)
                    break

        csv_file.close()
        return community_method_dict[self.cdm].__wrapped__(self.labels_true, self.enhance_graph)


def get_interComm_edges(communities):
    edges = []
    for i in range(len(communities)):
        for j in range(i + 1, len(communities)):
            comm1 = communities[i]
            comm2 = communities[j]
            edges.extend([(min(u, v), max(u, v)) for u in comm1 for v in comm2])
    return edges

def get_intraComm_edges(communities):
    edges = []
    for comm in communities:
        edges.extend([(min(u, v), max(u, v)) for u in comm for v in comm if u != v])
    return edges



def main():
    dataset = 'karate'
    cdm = 'LU'
    popsize = 10
    Pc = 0.8
    Pm = 0.2
    recombine_rate = 0.2
    sample_add_ratio = 0.1
    sample_del_ratio = 0.1
    varlen = bool(1)
    iterNum = 10

    rega = robustECD_GA(bmname=dataset,
                        cdm=cdm,
                        popsize=popsize,
                        Pc=Pc, Pm=Pm,
                        recombine_rate=recombine_rate,
                        sample_del_ratio=sample_del_ratio,
                        sample_add_ratio=sample_add_ratio,
                        iterNum=iterNum,
                        varlen=varlen,
                        output_file='1.csv').run()

if __name__ == '__main__':
    main()
