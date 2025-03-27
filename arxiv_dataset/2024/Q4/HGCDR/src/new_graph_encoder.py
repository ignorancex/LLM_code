from argument_parse import args
import dgl
import dgl.nn as dglnn
import dgl.function as dglfn
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.keras import layers

class EmbeddingEncoder(nn.Module):
    def __init__(self, graph):
        super(EmbeddingEncoder, self).__init__()
        self.emb_dict = {}
        ntypes = graph.ntypes
        for tt in ntypes:
            node_num = graph.num_nodes(tt)
            if tt == 'user':
                user_emb = nn.Embedding(node_num, args.input_dim)
                nn.init.normal_(user_emb.weight, std=0.01)
                for domain in args.graph_domains:
                    new_tt = tt + '_' + domain
                    self.emb_dict[new_tt] = user_emb
            else:
                self.emb_dict[tt] = nn.Embedding(node_num, args.input_dim)
                nn.init.normal_(self.emb_dict[tt].weight, std=0.01)
        self.emb_dict = nn.ModuleDict(self.emb_dict)

    def forward(self, batch_graph, flag = 'block'):
        if flag == 'block':
            node_ids_dict = batch_graph.ndata[dgl.NID]
        else:
            node_ids_dict = {}
            for each_node_type in batch_graph.ntypes:
                nodes = batch_graph.nodes(each_node_type)
                if args.device == 'gpu':
                    nodes = nodes.to(torch.device('cuda'))
                node_ids_dict[each_node_type] = nodes
        embedding_result = {}
        for node_type, node_id in node_ids_dict.items():
            if node_type == 'user':
                for domain in args.graph_domains:
                    embedding_result[node_type + '_' + domain] = self.emb_dict[node_type + '_' + domain](node_id)
            else:
                embedding_layer = self.emb_dict[node_type]
                embedding_result[node_type] = embedding_layer(node_id)
        return embedding_result

class MyHeroGraphConvTest(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MyHeroGraphConvTest, self).__init__()
        self.conv_dict = {}

        for domain1 in args.graph_domains:
            self.conv_dict[domain1 + '_item_to_' + '_user'] = dglnn.GraphConv(in_dim, out_dim)
            self.conv_dict[domain1 + '_user_to_' + domain1 + '_item'] = dglnn.GraphConv(in_dim, out_dim)
        self.conv_dict = nn.ModuleDict(self.conv_dict)

        for _, v in self.conv_dict.items():
            set_allow_zero_in_degree_fn = getattr(v, 'set_allow_zero_in_degree', None)
            if callable(set_allow_zero_in_degree_fn):
                set_allow_zero_in_degree_fn(True)

    def forward(self, graph, feature_dict):
        output_dict = {}
        for key in feature_dict:
            output_dict[key] = []
        for stype, etype, dtype in graph.canonical_etypes:
            rel_graph = graph[stype, etype, dtype]
            if dtype == 'user':
                for d_domain in args.graph_domains:
                    mod = self.conv_dict[stype + '_item_to_' + '_' + dtype]
                    output = mod(rel_graph, feature_dict[stype])
                    output_dict['user_' + d_domain].append(output)
            else:
                output_dict[dtype] = []
                mod = self.conv_dict[dtype + '_user_to_' + dtype + '_item']
                output = mod(rel_graph, feature_dict['user_' + dtype])
                output_dict[dtype].append(output)

        for dtype in output_dict.keys():
            cat_output = torch.stack(output_dict[dtype], dim = 0)
            output_dict[dtype] = torch.sum(cat_output, dim = 0)
        return output_dict

class MyHeroGraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MyHeroGraphConv, self).__init__()
        self.conv_dict = {}

        for domain1 in args.graph_domains:
            for domain2 in args.graph_domains:
                self.conv_dict[domain1 + '_item_to_' + domain2 + '_user'] = dglnn.GraphConv(in_dim, out_dim)
            self.conv_dict[domain1 + '_user_to_' + domain1 + '_item'] = dglnn.GraphConv(in_dim, out_dim)
        self.conv_dict = nn.ModuleDict(self.conv_dict)

        for _, v in self.conv_dict.items():
            set_allow_zero_in_degree_fn = getattr(v, 'set_allow_zero_in_degree', None)
            if callable(set_allow_zero_in_degree_fn):
                set_allow_zero_in_degree_fn(True)

    def forward(self, graph, feature_dict):
        output_dict = {}
        for key in feature_dict:
            output_dict[key] = []
        for stype, etype, dtype in graph.canonical_etypes:
            rel_graph = graph[stype, etype, dtype]
            if dtype == 'user':
                for d_domain in args.graph_domains:
                    mod = self.conv_dict[stype + '_item_to_' + d_domain + '_' + dtype]
                    output = mod(rel_graph, feature_dict[stype])
                    output_dict['user_' + d_domain].append(output)
            else:
                output_dict[dtype] = []
                mod = self.conv_dict[dtype + '_user_to_' + dtype + '_item']
                output = mod(rel_graph, feature_dict['user_' + dtype])
                output_dict[dtype].append(output)

        for dtype in output_dict.keys():
            cat_output = torch.stack(output_dict[dtype], dim = 0)
            output_dict[dtype] = torch.sum(cat_output, dim = 0)
        return output_dict

class GraphEncoder(nn.Module):
    def __init__(self, graph, domains, projection_on):
        super(GraphEncoder, self).__init__()
        self.domains = domains
        self.emb = EmbeddingEncoder(graph)
        self.input_layer = MyHeroGraphConv(args.input_dim, args.hidden_dim)
        self.hidden_layer = MyHeroGraphConv(args.hidden_dim, args.hidden_dim)
        self.output_layer = MyHeroGraphConv(args.hidden_dim, args.output_dim)
        self.layers = nn.ModuleList()
        self.layers.append(self.input_layer)
        for i in range(args.n_layer - 2):
            self.layers.append(self.hidden_layer)
        self.layers.append(self.output_layer)

        self.projection_on = projection_on
        if self.projection_on:
            self.output_weight_dict = {}
            for domain in self.domains:
                self.output_weight_dict[domain] = nn.Parameter(nn.init.normal_(torch.randn((args.output_dim, args.output_dim))), requires_grad=True)
                self.register_parameter(domain, self.output_weight_dict[domain])

        for k, v in self.named_parameters():
            print(k)
        # for k in self.parameters():
        #     print(k)

    #v3: predict no dropout
    def get_graph_encoder(self, blocks, flag = 'graph', train = True):
        if flag == 'graph':
            feats = self.emb(blocks, "graph")
        else:
            feats = self.emb(blocks[0], "block")

        for i, layer in enumerate(self.layers):
            if flag == 'graph':
                feats = layer(blocks, feats)
            else:
                feats = layer(blocks[i], feats)
            if i != len(self.layers) - 1:
                feats = {k: F.relu(v) for k, v in feats.items()}
                feats = {k: F.dropout(v, p=args.dropout, training=train) for k, v in feats.items()}

        result = dict()
        for domain in args.train_and_eval_domains:
            user_emb = feats['user_' + domain]
            if  self.projection_on:
                user_emb = torch.mm(user_emb, self.output_weight_dict[domain])
            result[domain + '_user_output'] = user_emb
            item_emb = feats[domain]
            result[domain + '_item_output'] = item_emb
        return result

    def get_score(self, h, graph):
        score_dict = {}
        with graph.local_scope():
            for domain in self.domains:
                graph.nodes['user'].data['h'] = h[domain + '_user_output']
                graph.nodes[domain].data['h'] = h[domain + '_item_output']
                graph.apply_edges(dglfn.u_dot_v('h', 'h', 'score'), etype=('user', 'user_click_' + domain , domain))
                score = graph.edges['user_click_' + domain].data['score']
                score_dict[domain] = score
        return score_dict

    def get_loss(self, h, positive_graph, negative_graph):
        pos_score = self.get_score(h, positive_graph)
        neg_score = self.get_score(h, negative_graph)

        pos_score = torch.cat([pos_score[domain] for domain in self.domains])
        pos_score = torch.repeat_interleave(pos_score.squeeze(), 5)
        neg_score = torch.cat([neg_score[domain] for domain in self.domains])

        loss = torch.mean(nn.functional.softplus(neg_score - pos_score))
        return loss