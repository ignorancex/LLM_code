from argument_parse import args
import dgl
import dgl.nn as dglnn
import dgl.function as dglfn
import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingEncoder(nn.Module):
    def __init__(self, graph):
        super(EmbeddingEncoder, self).__init__()
        self.emb_dict = {}
        ntypes = graph.ntypes
        for tt in ntypes:
            node_num = graph.num_nodes(tt)
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
            embedding_layer = self.emb_dict[node_type]
            embedding_result[node_type] = embedding_layer(node_id)
        return embedding_result

class GraphEncoder(nn.Module):
    def __init__(self, graph, domains, projection_on):
        super(GraphEncoder, self).__init__()
        self.domains = domains
        self.emb = EmbeddingEncoder(graph)
        rel_names = graph.etypes
        self.input_layer = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(args.input_dim, args.hidden_dim) for rel in rel_names
        }, aggregate='sum')
        self.hidden_layer = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(args.hidden_dim, args.hidden_dim) for rel in rel_names
        }, aggregate='sum')
        self.output_layer = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(args.hidden_dim, args.output_dim) for rel in rel_names
        }, aggregate='sum')
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

    #v1
    # def get_graph_encoder(self, blocks, flag = 'graph', train = True):
    #     if flag == 'graph':
    #         feats = self.emb(blocks, "graph")
    #     else:
    #         feats = self.emb(blocks[0], "block")
    #
    #     for i, layer in enumerate(self.layers):
    #         if i != 0:
    #             feats = {k: self.dropout(v) for k, v in feats.items()}
    #         if flag == 'graph':
    #             feats = layer(blocks, feats)
    #         else:
    #             feats = layer(blocks[i], feats)
    #
    #         feats = {k: F.relu(v) for k, v in feats.items()}
    #
    #     return feats

    #v2: last layer no relu
    # def get_graph_encoder(self, blocks, flag = 'graph', train = True):
    #     if flag == 'graph':
    #         feats = self.emb(blocks, "graph")
    #     else:
    #         feats = self.emb(blocks[0], "block")
    #
    #     for i, layer in enumerate(self.layers):
    #         if i != 0:
    #             feats = {k: self.dropout(v) for k, v in feats.items()}
    #         if flag == 'graph':
    #             feats = layer(blocks, feats)
    #         else:
    #             feats = layer(blocks[i], feats)
    #         if i != len(self.layers) - 1:
    #             feats = {k: F.relu(v) for k, v in feats.items()}
    #
    #     return feats

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
        for domain in self.domains:
            user_emb = feats['user']
            if self.projection_on:
                user_emb = torch.matmul(user_emb, self.output_weight_dict[domain])
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