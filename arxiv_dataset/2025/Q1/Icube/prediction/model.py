import torch
import torch.nn as nn
import networkx as nx
import numpy as np
import random
import json
import time
import dgl
import dgl.nn as dnn
import dgl.function as dfn
import copy
from tqdm import *
import math
from dgl.nn import GATConv, SumPooling, AvgPooling, MaxPooling
from pyproj import Geod
from shapely.geometry import Point, LineString
from pypower.api import ppoption, runpf
from train import *
from colorama import init
from colorama import Fore, Back, Style

init()

# device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')
device = "cuda:0"
# device = "cpu"



class SAGE(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.conv1 = dnn.SAGEConv(in_dim, hid_dim, 'mean')
        self.conv2 = dnn.SAGEConv(hid_dim, out_dim, 'mean')
        self.relu = nn.ReLU()

    def forward(self, graph, input):
        output = self.conv1(graph, input)
        output = self.relu(output)
        output = self.conv2(graph, output)

        return output


class Innerproduct(nn.Module):
    def forward(self, graph, feat):
        with graph.local_scope():
            graph.ndata['feat'] = feat
            graph.apply_edges(dfn.u_dot_v('feat', 'feat', 'score'))
            return graph.edata['score']


class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.sage = SAGE(in_dim, hid_dim, out_dim)
        self.pred = Innerproduct()

    def forward(self, graph, neg_graph, feat):
        feat = self.sage(graph, feat)
        return self.pred(graph, feat), self.pred(neg_graph, feat)


def construct_negative_graph(graph, k):
    src, dst = graph.edges()

    neg_src = src.repeat_interleave(k).cpu()
    neg_dst = torch.randint(0, graph.num_nodes(), (len(src) * k,)).cpu()
    return dgl.graph((neg_src, neg_dst), num_nodes=graph.num_nodes())


def compute_loss(pos_score, neg_score):
    n_edges = pos_score.shape[0]
    return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()


class Graph():
    def __init__(self):
        self.graph = None
        self.feat = None
        self.node_list = None

    @property
    def node_num(self):
        return self.nxgraph.number_of_nodes()

    @property
    def egde_num(self):
        return self.nxgraph.number_of_edges()

    def build_graph(self):
        pass

    def build_feat(self, embed_dim, hid_dim, feat_dim,
                   k, epochs,
                   pt_path):

        print('training features ...')
        # self.graph = self.graph.to(device)
        # embedding = nn.Embedding(self.node_num, embed_dim, max_norm=1).to(device)
        embedding = nn.Embedding(self.node_num, embed_dim, max_norm=1)
        self.graph.ndata['feat'] = embedding.weight

        # gcn = GCN(embed_dim, hid_dim, feat_dim).to(device)
        gcn = GCN(embed_dim, hid_dim, feat_dim)
        optimizer = torch.optim.Adam(gcn.parameters())
        optimizer.zero_grad()

        for epoch in range(epochs):
            t = time.time()
            # negative_graph = construct_negative_graph(self.graph, k).to(device)
            negative_graph = construct_negative_graph(self.graph, k)
            pos_score, neg_score = gcn(self.graph, negative_graph, self.graph.ndata['feat'])
            loss = compute_loss(pos_score, neg_score)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 5 == 0:
                print("Epoch:", '%03d' % (epoch + 1), " train_loss = ", "{:.5f} ".format(loss.item()),
                      " time=", "{:.4f}s".format(time.time() - t)
                      )

        feat = gcn.sage(self.graph, self.graph.ndata['feat'])
        try:
            torch.save(feat, pt_path)
            print("saving features sucess")
        except:
            print("saving features failed")
        return feat


class ElecGraph(Graph):
    def __init__(self, file, embed_dim, hid_dim, feat_dim, khop, epochs, pt_path):
        print(Fore.RED, Back.YELLOW)
        print('Electricity network construction!')
        print(Style.RESET_ALL)
        self.node_list, self.nxgraph, self.graph = self.build_graph(file)
        # self.graph = self.graph.to(device)
        self.graph = self.graph
        self.degree = dict(nx.degree(self.nxgraph))
        self.CI = self.build_CI()

        try:
            feat = torch.load(pt_path)
            print('Elec features loaded.')
            self.feat = feat
        except:
            self.feat = self.build_feat(embed_dim, hid_dim, feat_dim,
                                        khop, epochs,
                                        pt_path)

    def build_graph(self, file):

        print('building elec graph ...')
        try:
            elec_graph = nx.read_gpickle(file)
        except:
            with open(file, 'r') as f:
                data = json.load(f)
            elec_graph = nx.Graph()
            for key, facility in data.items():
                for node_id in facility.keys():
                    node = facility[node_id]
                    for neighbor in node['relation']:
                        if int(node_id) < 6e8 and neighbor < 6e8:
                            elec_graph.add_edge(int(node_id), neighbor)

        node_list: dict = {i: j for i, j in enumerate(list(elec_graph.nodes()))}
        print('electric graph builded.')
        return node_list, elec_graph, dgl.from_networkx(elec_graph)

    def build_CI(self):
        CI = []
        d = self.degree
        for node in d:
            ci = 0
            neighbors = list(self.nxgraph.neighbors(node))
            for neighbor in neighbors:
                ci += (d[neighbor] - 1)
            CI.append((node, ci * (d[node] - 1)))

        return CI


class TraGraph(Graph):
    def __init__(self, file1, file2, file3,
                 embed_dim, hid_dim, feat_dim, r_type,
                 khop, epochs, pt_path):
        print(Fore.RED, Back.YELLOW)
        print('Traffice network construction!')
        print(Style.RESET_ALL)
        self.node_list, self.nxgraph, self.graph = self.build_graph(file1, file2, file3, r_type)
        self.degree = dict(nx.degree(self.nxgraph))
        self.CI = self.build_CI()
        try:
            feat = torch.load(pt_path)
            print('Traffic features loaded.')
            self.feat = feat
        except:
            self.feat = self.build_feat(embed_dim, hid_dim, feat_dim,
                                        khop, epochs,
                                        pt_path)

    def build_graph(self, file1, file2, file3, r_type):

        print('building traffic graph ...')
        graph = nx.Graph()
        with open(file1, 'r') as f:
            data = json.load(f)
        with open(file2, 'r') as f:
            road_type = json.load(f)
        with open(file3, 'r') as f:
            tl_id_road2elec_map = json.load(f)
        for road, junc in data.items():
            if len(junc) == 2 and road_type[road] == r_type:
                graph.add_edge(tl_id_road2elec_map[str(junc[0])], tl_id_road2elec_map[str(junc[1])])

        node_list: dict = {i: j for i, j in enumerate(list(graph.nodes()))}
        print('traffic graph builded.')
        return node_list, graph, dgl.from_networkx(graph)

    def build_CI(self):
        CI = []
        d = self.degree
        for node in d:
            ci = 0
            neighbors = list(self.nxgraph.neighbors(node))
            for neighbor in neighbors:
                ci += (d[neighbor] - 1)
            CI.append((node, ci * (d[node] - 1)))

        return CI


class BSGraph(Graph):
    def __init__(self, file, embed_dim, hid_dim, feat_dim,
                 khop, epochs, pt_path):
        print(Fore.RED, Back.YELLOW)
        print('BaseStation network construction!')
        print(Style.RESET_ALL)
        self.node_list, self.nxgraph, self.graph = self.build_graph(file)
        self.degree = dict(nx.degree(self.nxgraph))
        self.CI = self.build_CI()
        try:
            feat = torch.load(pt_path)
            print('Traffic features loaded.')
            self.feat = feat
        except:
            self.feat = self.build_feat(embed_dim, hid_dim, feat_dim,
                                        khop, epochs,
                                        pt_path)

    def build_graph(self, file):

        print('building basestation graph ...')
        graph = nx.Graph()
        with open(file, 'r') as f:
            data = json.load(f)
        for key, facility in tqdm(data.items(), total=len(data)):
            for relation in facility['relation']:
                node = int(relation)
                graph.add_edge(int(node), int(key))
        node_list: dict = {i: j for i, j in enumerate(list(graph.nodes()))}
        print('basestation graph builded.')
        return node_list, graph, dgl.from_networkx(graph)

    def build_CI(self):
        CI = []
        d = self.degree
        for node in d:
            ci = 0
            neighbors = list(self.nxgraph.neighbors(node))
            for neighbor in neighbors:
                ci += (d[neighbor] - 1)
            CI.append((node, ci * (d[node] - 1)))

        return CI


class AOIGraph(Graph):
    def __init__(self, file, embed_dim, hid_dim, feat_dim,
                 khop, epochs, pt_path):
        print(Fore.RED, Back.YELLOW)
        print('BaseStation network construction!')
        print(Style.RESET_ALL)
        self.node_list, self.nxgraph, self.graph = self.build_graph(file)
        # try:
        #     feat = torch.load(pt_path)
        #     print('Traffic features loaded.')
        #     self.feat = feat
        # except:
        #     self.feat = self.build_feat(embed_dim, hid_dim, feat_dim,
        #                                 khop, epochs,
        #                                 pt_path)

    def build_graph(self, file):
        print('building basestation graph ...')
        graph = nx.Graph()
        with open(file, 'r') as f:
            data = json.load(f)
        for key, facility in tqdm(data.items(), total=len(data)):
            graph.add_node(int(key))
        node_list: dict = {i: j for i, j in enumerate(list(graph.nodes()))}
        print('basestation graph builded.')
        return node_list, graph, dgl.from_networkx(graph)


class RGCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, rel_names):
        super().__init__()
        self.conv1 = dnn.HeteroGraphConv({
            rel: dnn.GraphConv(in_dim, hid_dim)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dnn.HeteroGraphConv({
            rel: dnn.GraphConv(hid_dim, out_dim)
            for rel in rel_names}, aggregate='sum')

        self.relu = nn.ReLU()

    def forward(self, graph, input):
        output = self.conv1(graph, input)
        output = {k: self.relu(v) for k, v in output.items()}
        output = self.conv2(graph, output)
        return output


class GAERGCN(nn.Module):
    def __init__(self, in_dim, hid_dim, rel_names):
        super().__init__()
        self.conv1 = dnn.HeteroGraphConv({
            rel: dnn.GraphConv(in_dim, hid_dim)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dnn.HeteroGraphConv({
            rel: dnn.GraphConv(hid_dim, hid_dim)
            for rel in rel_names}, aggregate='sum')

        self.relu = nn.ReLU()

    def forward(self, graph, input):
        output = self.conv1(graph, input)
        output = {k: self.relu(v) for k, v in output.items()}
        output = self.conv2(graph, output)
        return output


class HeteroInnerProduct(nn.Module):
    def forward(self, graph, feat, etype):
        with graph.local_scope():
            graph.ndata['feat'] = feat
            graph.apply_edges(dfn.u_dot_v('feat', 'feat', 'score'))
            return graph.edges[etype].data['score']


class HeteroGCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, rel_names):
        super().__init__()
        self.layer = RGCN(in_dim, hid_dim, out_dim, rel_names)
        self.pred = HeteroInnerProduct()

    def forward(self, graph, neg_graph, feat, etype):
        feat = self.layer(graph, feat)
        return self.pred(graph, feat, etype), self.pred(neg_graph, feat, etype)


def numbers_to_etypes(num):
    switcher = {
        0: ('power', 'elec', 'power'),
        1: ('power', 'eleced-by', 'power'),
        2: ('junc', 'tran', 'junc'),
        3: ('junc', 'traned-by', 'junc'),
        4: ('bs', 'communicate', 'bs'),
        5: ('bs', 'communicate-by', 'bs'),
        6: ('junc', 'tran-supp-elec', 'power'),
        7: ('power', 'elec-suppd-by-tran', 'junc'),
        8: ('bs', 'bs-supp-elec', 'power'),
        9: ('power', 'elec-suppd-by-bs', 'bs'),
        10: ('power', 'elec-suppd-aoi', 'aoi'),
        11: ('aoi', 'aoi-suppd-by-elec', 'power'),
        12: ('bs', 'bs-supp-aoi', 'aoi'),
        13: ('aoi', 'aoi-suppd-by-bs', 'bs')
    }

    return switcher.get(num, "wrong!")


def construct_negative_graph_with_type(graph, k, etype):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,))
    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes})


def build_hetero(nxgraph, embed_dim, hid_dim, feat_dim, subgraph, pt_path):
    bsgraph, egraph, tgraph, aoigraph = subgraph

    power_embedding = torch.load(pt_path[0])
    junc_embedding = torch.load(pt_path[1])
    bs_embedding = torch.load(pt_path[2])
    aoi_embedding = torch.load(pt_path[3])

    n_power = egraph.node_num
    n_junc = tgraph.node_num
    n_bs = bsgraph.node_num
    n_aoi = aoigraph.node_num

    power_idx = {v: k for k, v in egraph.node_list.items()}
    junc_idx = {v: k - n_power for k, v in tgraph.node_list.items()}
    bs_idx = {v: k - n_power - n_junc for k, v in bsgraph.node_list.items()}
    aoi_idx = {v: k - n_power - n_junc - n_bs for k, v in aoigraph.node_list.items()}

    edge_list = nxgraph.edges()
    elec_edge = [(u, v) for (u, v) in edge_list if u // 1e8 < 6 and v // 1e8 < 6]
    tran_edge = [(u, v) for (u, v) in edge_list if u // 1e8 == 9 and v // 1e8 == 9]
    bs_edge = [(u, v) for (u, v) in edge_list if u // 1e8 == 7 and v // 1e8 == 7]

    ele_bs_edge = [(u, v) for (u, v) in edge_list
                   if (u // 1e8 < 6 and v // 1e8 == 7) or (v // 1e8 < 6 and u // 1e8 == 7)]

    ele_tran_edge = [(u, v) for (u, v) in edge_list
                     if (u // 1e8 < 6 and v // 1e8 == 9) or (v // 1e8 < 6 and u // 1e8 == 9)]

    ele_aoi_edge = [(u, v) for (u, v) in edge_list
                    if (u // 1e8 < 6 and v // 1e8 == 11) or (v // 1e8 < 6 and u // 1e8 == 11)]

    bs_aoi_edge = [(u, v) for (u, v) in edge_list
                   if (u // 1e8 == 7 and v // 1e8 == 11)]

    elec_src, elec_dst = np.array([power_idx[u] for (u, _) in elec_edge]), np.array(
        [power_idx[v] for (_, v) in elec_edge])
    tran_src, tran_dst = np.array([junc_idx[u] for (u, _) in tran_edge]), np.array(
        [junc_idx[v] for (_, v) in tran_edge])
    bs_src, bs_dst = np.array([bs_idx[u] for (u, _) in bs_edge]), np.array(
        [bs_idx[v] for (_, v) in bs_edge])
    ele_bs_src, ele_bs_dst = np.array([power_idx[u] for (u, _) in ele_bs_edge]), np.array(
        [bs_idx[v] for (_, v) in ele_bs_edge])
    ele_tran_src, ele_tran_dst = np.array([junc_idx[u] for (u, _) in ele_tran_edge]), np.array(
        [power_idx[v] for (_, v) in ele_tran_edge])
    ele_aoi_src, ele_aoi_dst = np.array([power_idx[u] for (u, _) in ele_aoi_edge]), np.array(
        [aoi_idx[v] for (_, v) in ele_aoi_edge])
    bs_aoi_src, bs_aoi_dst = np.array([bs_idx[u] for (u, _) in bs_aoi_edge]), np.array(
        [aoi_idx[v] for (_, v) in bs_aoi_edge])

    hetero_graph = dgl.heterograph({
        ('power', 'elec', 'power'): (elec_src, elec_dst),
        ('power', 'eleced-by', 'power'): (elec_dst, elec_src),

        ('junc', 'tran', 'junc'): (tran_src, tran_dst),
        ('junc', 'traned-by', 'junc'): (tran_dst, tran_src),

        ('bs', 'communicate', 'bs'): (bs_src, bs_dst),
        ('bs', 'communicate-by', 'bs'): (bs_src, bs_dst),

        ('junc', 'tran-supp-elec', 'power'): (ele_tran_src, ele_tran_dst),
        ('power', 'elec-suppd-by-tran', 'junc'): (ele_tran_dst, ele_tran_src),

        ('bs', 'bs-supp-elec', 'power'): (ele_bs_dst, ele_bs_src),
        ('power', 'elec-suppd-by-bs', 'bs'): (ele_bs_src, ele_bs_dst),

        ('power', 'elec-suppd-aoi', 'aoi'): (ele_aoi_src, ele_aoi_dst),
        ('aoi', 'aoi-suppd-by-elec', 'power'): (ele_aoi_dst, ele_aoi_src),

        ('bs', 'bs-supp-aoi', 'aoi'): (bs_aoi_src, bs_aoi_dst),
        ('aoi', 'aoi-suppd-by-bs', 'bs'): (bs_aoi_dst, bs_aoi_src)
    })

    hetero_graph.nodes['power'].data['feature'] = power_embedding
    hetero_graph.nodes['junc'].data['feature'] = junc_embedding
    hetero_graph.nodes['bs'].data['feature'] = bs_embedding
    hetero_graph.nodes['aoi'].data['feature'] = aoi_embedding

    return hetero_graph


class Bigraph(Graph):
    def __init__(self, efile, tfile1, tfile2, tfile3, file, bsfile, ele2bsfile, aoifile,
                 embed_dim, hid_dim, feat_dim,
                 r_type, subgraph,
                 khop, epochs, pt_path):
        print(Fore.RED, Back.YELLOW)
        print('Bigraph network construction!')
        print(Style.RESET_ALL)
        self.nxgraph = self.build_graph(efile, tfile1, tfile2, tfile3, bsfile, ele2bsfile, aoifile, r_type)
        bsgraph, egraph, tgraph, aoigraph = subgraph
        self.node_list = egraph.node_list
        tgraph.node_list = {k + egraph.node_num: v for k, v in tgraph.node_list.items()}
        self.node_list.update(tgraph.node_list)
        bsgraph.node_list = {k + egraph.node_num + tgraph.node_num: v for k, v in bsgraph.node_list.items()}
        self.node_list.update(bsgraph.node_list)
        aoigraph.node_list = {k + egraph.node_num + tgraph.node_num + bsgraph.node_num: v for k, v in
                              aoigraph.node_list.items()}
        self.node_list.update(aoigraph.node_list)
        with open(file, 'r') as f:
            self.elec2road = json.load(f)
        with open(ele2bsfile, 'r') as f:
            self.elec2bs = json.load(f)
        '''
        node list : {node index:  node id}
        0     -- 10886:     elec node
        10887 -- 15711:     road node
        15712 -- xxx        bs node
        '''
        try:
            feat = {}
            feat['power'] = torch.load(pt_path[0])
            feat['junc'] = torch.load(pt_path[1])
            feat['bs'] = torch.load(pt_path[2])
            feat['aoi'] = torch.load(pt_path[3])
            print('Bigraph features loaded.')
            self.feat = feat
        except:
            self.feat = self.build_feat(embed_dim, hid_dim, feat_dim,
                                        subgraph,
                                        khop, epochs,
                                        pt_path)

    def build_graph(self, efile, tfile1, tfile2, tfile3, bsfile, ele2bsfile, aoifile, r_type):

        print('building bigraph ...')

        graph = nx.Graph()
        with open(tfile1, 'r') as f:
            data = json.load(f)
        with open(tfile2, 'r') as f:
            road_type = json.load(f)
        with open(tfile3, 'r') as f:
            tl_id_road2elec_map = json.load(f)
        for road, junc in data.items():
            if len(junc) == 2 and road_type[road] == r_type:
                graph.add_edge(tl_id_road2elec_map[str(junc[0])], tl_id_road2elec_map[str(junc[1])], id=int(road))

        with open(efile, 'r') as f:
            data = json.load(f)
        for key, facility in data.items():
            for node_id in facility.keys():
                node = facility[node_id]
                for neighbor in node['relation']:
                    if int(node_id) < 6e8 and (neighbor < 6e8):
                        graph.add_edge(int(node_id), neighbor)
        for tl_id, value in data['tl'].items():
            if int(tl_id) in list(graph.nodes()):
                for neighbor in value['relation']:
                    graph.add_edge(neighbor, int(tl_id))

        with open(bsfile, 'r') as f:
            data = json.load(f)
        for node_id in tqdm(data.keys()):
            node = data[node_id]
            for neighbor in node['relation']:
                graph.add_edge(int(node_id), int(neighbor))

        with open(ele2bsfile, 'r') as f:
            data = json.load(f)
        for ele, value in tqdm(data.items()):
            if int(ele) in list(graph.nodes()):
                for neighbor in value:
                    graph.add_edge(int(neighbor), int(ele))

        with open(aoifile, 'r') as f:
            data = json.load(f)
        for aoi, value in tqdm(data.items()):
            relation = np.array(value['relation']).astype(int)
            relation = np.intersect1d(relation, np.array(list(graph.nodes)))
            for neighbor in relation:
                graph.add_edge(int(neighbor), int(aoi))

        print('bigraph builded.')
        return graph

    def build_feat(self, embed_dim, hid_dim, feat_dim, subgraph, k, epochs, pt_path):
        bsgraph, egraph, tgraph, aoigraph = subgraph

        n_power = egraph.node_num
        n_junc = tgraph.node_num
        n_bs = bsgraph.node_num
        n_aoi = aoigraph.node_num

        power_idx = {v: k for k, v in egraph.node_list.items()}
        junc_idx = {v: k - n_power for k, v in tgraph.node_list.items()}
        bs_idx = {v: k - n_power - n_junc for k, v in bsgraph.node_list.items()}
        aoi_idx = {v: k - n_power - n_junc - n_bs for k, v in aoigraph.node_list.items()}

        edge_list = self.nxgraph.edges()
        elec_edge = [(u, v) for (u, v) in edge_list if u // 1e8 < 6 and v // 1e8 < 6]
        tran_edge = [(u, v) for (u, v) in edge_list if u // 1e8 == 9 and v // 1e8 == 9]
        bs_edge = [(u, v) for (u, v) in edge_list if u // 1e8 == 7 and v // 1e8 == 7]

        ele_bs_edge = [(u, v) for (u, v) in edge_list
                       if (u // 1e8 < 6 and v // 1e8 == 7) or (v // 1e8 < 6 and u // 1e8 == 7)]

        ele_tran_edge = [(u, v) for (u, v) in edge_list
                         if (u // 1e8 < 6 and v // 1e8 == 9) or (v // 1e8 < 6 and u // 1e8 == 9)]

        ele_aoi_edge = [(u, v) for (u, v) in edge_list
                        if (u // 1e8 < 6 and v // 1e8 == 11) or (v // 1e8 < 6 and u // 1e8 == 11)]

        bs_aoi_edge = [(u, v) for (u, v) in edge_list
                       if (u // 1e8 == 7 and v // 1e8 == 11)]

        elec_src, elec_dst = np.array([power_idx[u] for (u, _) in elec_edge]), np.array(
            [power_idx[v] for (_, v) in elec_edge])
        tran_src, tran_dst = np.array([junc_idx[u] for (u, _) in tran_edge]), np.array(
            [junc_idx[v] for (_, v) in tran_edge])
        bs_src, bs_dst = np.array([bs_idx[u] for (u, _) in bs_edge]), np.array(
            [bs_idx[v] for (_, v) in bs_edge])
        ele_bs_src, ele_bs_dst = np.array([power_idx[u] for (u, _) in ele_bs_edge]), np.array(
            [bs_idx[v] for (_, v) in ele_bs_edge])
        ele_tran_src, ele_tran_dst = np.array([junc_idx[u] for (u, _) in ele_tran_edge]), np.array(
            [power_idx[v] for (_, v) in ele_tran_edge])
        ele_aoi_src, ele_aoi_dst = np.array([power_idx[u] for (u, _) in ele_aoi_edge]), np.array(
            [aoi_idx[v] for (_, v) in ele_aoi_edge])
        bs_aoi_src, bs_aoi_dst = np.array([bs_idx[u] for (u, _) in bs_aoi_edge]), np.array(
            [aoi_idx[v] for (_, v) in bs_aoi_edge])

        hetero_graph = dgl.heterograph({
            ('power', 'elec', 'power'): (elec_src, elec_dst),
            ('power', 'eleced-by', 'power'): (elec_dst, elec_src),

            ('junc', 'tran', 'junc'): (tran_src, tran_dst),
            ('junc', 'traned-by', 'junc'): (tran_dst, tran_src),

            ('bs', 'communicate', 'bs'): (bs_src, bs_dst),
            ('bs', 'communicate-by', 'bs'): (bs_src, bs_dst),

            ('junc', 'tran-supp-elec', 'power'): (ele_tran_src, ele_tran_dst),
            ('power', 'elec-suppd-by-tran', 'junc'): (ele_tran_dst, ele_tran_src),

            ('bs', 'bs-supp-elec', 'power'): (ele_bs_dst, ele_bs_src),
            ('power', 'elec-suppd-by-bs', 'bs'): (ele_bs_src, ele_bs_dst),

            ('power', 'elec-suppd-aoi', 'aoi'): (ele_aoi_src, ele_aoi_dst),
            ('aoi', 'aoi-suppd-by-elec', 'power'): (ele_aoi_dst, ele_aoi_src),

            ('bs', 'bs-supp-aoi', 'aoi'): (bs_aoi_src, bs_aoi_dst),
            ('aoi', 'aoi-suppd-by-bs', 'bs'): (bs_aoi_dst, bs_aoi_src)
        })

        hetero_graph.nodes['power'].data['feature'] = torch.nn.Embedding(n_power, embed_dim, max_norm=1).weight
        hetero_graph.nodes['junc'].data['feature'] = torch.nn.Embedding(n_junc, embed_dim, max_norm=1).weight
        hetero_graph.nodes['bs'].data['feature'] = torch.nn.Embedding(n_bs, embed_dim, max_norm=1).weight
        hetero_graph.nodes['aoi'].data['feature'] = torch.nn.Embedding(n_aoi, embed_dim, max_norm=1).weight

        hgcn = HeteroGCN(embed_dim, hid_dim, feat_dim, hetero_graph.etypes)

        bifeatures = {
            'junc': hetero_graph.nodes['junc'].data['feature'],
            'power': hetero_graph.nodes['power'].data['feature'],
            'bs': hetero_graph.nodes['bs'].data['feature'],
            'aoi': hetero_graph.nodes['aoi'].data['feature']
        }

        optimizer = torch.optim.Adam(hgcn.parameters())
        print('training features ...')
        for epoch in range(epochs):
            num = epoch % 14
            etype = numbers_to_etypes(num)
            t = time.time()
            negative_graph = construct_negative_graph_with_type(hetero_graph, k, etype)
            pos_score, neg_score = hgcn(hetero_graph, negative_graph, bifeatures, etype)
            loss = compute_loss(pos_score, neg_score)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("Epoch:", '%03d' % (epoch + 1), " train_loss = ", "{:.5f} ".format(loss.item()),
                  " time=", "{:.4f}s".format(time.time() - t)
                  )
            del loss
        feat = hgcn.layer(hetero_graph, bifeatures)
        try:
            torch.save(feat['power'], pt_path[0])
            torch.save(feat['junc'], pt_path[1])
            torch.save(feat['bs'], pt_path[2])
            torch.save(feat['aoi'], pt_path[3])
            print("saving features sucess")
        except:
            print("saving features failed")

        return feat


BASE = 100000000
geod = Geod(ellps="WGS84")




class Net(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.hidden = nn.Linear(in_dim, hid_dim)
        self.output = nn.Linear(hid_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        hidden = self.hidden(inputs)
        outputs = self.relu(hidden)
        outputs = self.output(outputs)

        return outputs


class GAEEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(GAEEncoder, self).__init__()

        self.encoder = SAGE(in_dim, hidden_dim, hidden_dim)

    def forward(self, g, x):
        z = self.encoder(g, x)

        return z


class GAEDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(GAEDecoder, self).__init__()

        self.decoder = SAGE(in_dim, hidden_dim, hidden_dim)

    def forward(self, g, z):
        z = self.decoder(g, z)

        return z


class InnerProductDecoder(nn.Module):
    def forward(self, g, z):
        adj_rec = torch.mm(z, z.t())
        return adj_rec



def txt_to_tensor(fpath):
    with open(fpath, 'r') as f:
        data = f.readline()
        data = torch.tensor([float(data.split('\t')[0]), float(data.split('\t')[1])], dtype=torch.float32)
    return data



class GAE_GMNN_RGCN_DP_delta(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, etypes):
        super(GAE_GMNN_RGCN_DP_delta, self).__init__()
        self.CoupledEncoder = GAERGCN(in_dim + 16 + 2, hidden_dim, etypes)
        self.ElecEncoder = GAEEncoder(in_dim + 16 + 2, hidden_dim)
        self.JuncEncoder = GAEEncoder(in_dim + 16 + 2, hidden_dim)
        self.BsEncoder = GAEEncoder(in_dim + 16 + 2, hidden_dim)
        self.Layer1 = GAERGCN(2 * hidden_dim, hidden_dim, etypes)
        self.CoupledDecoder = GAERGCN(hidden_dim, hidden_dim, etypes)
        self.ElecDncoder = GAEDecoder(hidden_dim, hidden_dim)
        self.JuncDncoder = GAEDecoder(hidden_dim, hidden_dim)
        self.BsDncoder = GAEDecoder(hidden_dim, hidden_dim)
        self.Layer2 = GAERGCN(2 * hidden_dim, hidden_dim, etypes)
        self.Layer3 = GAERGCN(hidden_dim, 2, etypes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.th_elec = nn.Linear(2, 1)
        self.th_junc = nn.Linear(2, 1)
        self.th_bs = nn.Linear(2, 1)
        self.th_aoi = nn.Linear(2, 1)

    def get_embedding(self, fpath):
        with open(fpath, 'r') as f:
            data = f.readlines()
        embedding = []
        for item in data:
            item = item.split('\t')[-1]
            item = float(item.strip('\n'))
            embedding.append(item)
        embedding = torch.tensor(embedding).unsqueeze(-1)
        return embedding

    def forward(self, graphs: list, inputs: list, case_num):
        n_elec = graphs[1].num_nodes()
        n_junc = graphs[2].num_nodes()
        n_bs = graphs[3].num_nodes()
        n_aoi = graphs[0].num_nodes() - n_elec - n_junc - n_bs
        fpath = './GMNN_data/embedding/case_{0}.pt'.format(case_num)
        GMNN_emb = torch.load(fpath)
        fpath = '../Diffpool_test_single/diffpool_path_delta_coupled/{0}.txt'.format(case_num)
        DP_Coupled_emb = self.get_embedding(fpath)
        fpath = '../Diffpool_test_single/diffpool_path_delta_elec/{0}.txt'.format(case_num)
        DP_elec_emb = self.get_embedding(fpath)
        fpath = '../Diffpool_test_single/diffpool_path_delta_junc/{0}.txt'.format(case_num)
        DP_junc_emb = self.get_embedding(fpath)
        fpath = '../Diffpool_test_single/diffpool_path_delta_bs/{0}.txt'.format(case_num)
        DP_bs_emb = self.get_embedding(fpath)
        inputs['power'] = torch.cat([inputs['power'].to(device),
                                     GMNN_emb[0:n_elec].to(device),
                                     DP_Coupled_emb[0:n_elec].to(device),
                                     DP_elec_emb.to(device)], dim=1)
        inputs['junc'] = torch.cat([inputs['junc'].to(device),
                                    GMNN_emb[n_elec:n_elec + n_junc].to(device),
                                    DP_Coupled_emb[n_elec:n_elec + n_junc].to(device),
                                    DP_junc_emb.to(device)], dim=1)
        inputs['bs'] = torch.cat([inputs['bs'].to(device),
                                  GMNN_emb[n_elec + n_junc:n_elec + n_junc + n_bs].to(device),
                                  DP_Coupled_emb[n_elec + n_junc:n_elec + n_junc + n_bs].to(device),
                                  DP_bs_emb.to(device)], dim=1)
        inputs['aoi'] = torch.cat([inputs['aoi'].to(device),
                                   GMNN_emb[n_elec + n_junc + n_bs:].to(device),
                                   DP_Coupled_emb[n_elec + n_junc + n_bs:].to(device),
                                   torch.zeros([n_aoi, 1]).to(device)], dim=1)
        Coupled_Z = self.CoupledEncoder(graphs[0], inputs)
        Elec_Z = self.ElecEncoder(graphs[1], inputs['power'])
        Junc_Z = self.JuncEncoder(graphs[2], inputs['junc'])
        Bs_Z = self.BsEncoder(graphs[3], inputs['bs'])
        Coupled_Z['aoi'] = torch.cat((Coupled_Z['aoi'],
                                      torch.zeros((Coupled_Z['aoi'].shape[0], 20), requires_grad=True, device=device)),
                                     dim=1)
        Coupled_Z['power'] = torch.cat((Coupled_Z['power'], Elec_Z), dim=1)
        Coupled_Z['junc'] = torch.cat((Coupled_Z['junc'], Junc_Z), dim=1)
        Coupled_Z['bs'] = torch.cat((Coupled_Z['bs'], Bs_Z), dim=1)
        Coupled_Z = self.Layer1(graphs[0], Coupled_Z)
        Decoupled_Z = self.CoupledDecoder(graphs[0], Coupled_Z)
        Decoupled_Elec_Z = self.ElecDncoder(graphs[1], Coupled_Z['power'])
        Decoupled_Junc_Z = self.JuncDncoder(graphs[2], Coupled_Z['junc'])
        Decoupled_Bs_Z = self.BsDncoder(graphs[3], Coupled_Z['bs'])
        Decoupled_Z['power'] = torch.cat((Decoupled_Z['power'], Decoupled_Elec_Z), dim=1)
        Decoupled_Z['junc'] = torch.cat((Decoupled_Z['junc'], Decoupled_Junc_Z), dim=1)
        Decoupled_Z['bs'] = torch.cat((Decoupled_Z['bs'], Decoupled_Bs_Z), dim=1)
        Decoupled_Z['aoi'] = torch.cat((Decoupled_Z['aoi'],
                                        torch.zeros((Decoupled_Z['aoi'].shape[0], 20),
                                                    requires_grad=True,
                                                    device=device)), dim=1)
        Decoupled_Z = self.Layer2(graphs[0], Decoupled_Z)
        Decoupled_Z = {k: self.relu(v) for k, v in Decoupled_Z.items()}
        Decoupled_Z = self.Layer3(graphs[0], Decoupled_Z)
        Decoupled_Z = {k: self.softmax(v) for k, v in Decoupled_Z.items()}
        return Decoupled_Z


