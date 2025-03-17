from argument_parse import args
import dgl
import dgl.nn as dglnn
import dgl.function as dglfn
import torch
import torch.nn as nn
import torch.nn.functional as F
#from tensorflow.keras import layers
import numpy as np
from info_nce import InfoNCE

class EmbeddingEncoder(nn.Module):
    def __init__(self, graph):
        super(EmbeddingEncoder, self).__init__()
        self.emb_dict = {}
        self.bias_emb_dict = {}
        ntypes = graph.ntypes

        def create_nn_embedding(node_num, input_dim):
            emb = nn.Embedding(node_num, input_dim)
            nn.init.normal_(emb.weight, std=0.01)
            return emb

        for tt in ntypes:
            node_num = graph.num_nodes(tt)
            emb = nn.Embedding(node_num, args.input_dim)
            nn.init.normal_(emb.weight, std=0.01)
            if tt == 'user':
                for domain in args.graph_domains:
                    self.emb_dict['inner_' + domain + '_user'] = emb
                    self.bias_emb_dict['inner_' + domain + '_user_bias'] = create_nn_embedding(node_num, args.input_dim)
                self.emb_dict['outer_user'] = emb
                self.bias_emb_dict['outer_user_bias'] = create_nn_embedding(node_num, args.input_dim)
            else:
                self.emb_dict['inner_' + tt + '_item'] = emb
                self.emb_dict['outer_' + tt] = emb
                self.bias_emb_dict['outer_' + tt + '_bias'] = create_nn_embedding(node_num, args.input_dim)
        self.emb_dict = nn.ModuleDict(self.emb_dict)
        self.bias_emb_dict = nn.ModuleDict(self.bias_emb_dict)

    def forward(self, batch_graph):
        if batch_graph.is_block:
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
                    embedding_result['inner_' + domain + '_user'] = self.emb_dict['inner_' + domain + '_user'](node_id)
                embedding_result['outer_user'] = self.emb_dict['outer_user'](node_id)
            else:
                embedding_result['inner_' + node_type + '_item'] = self.emb_dict['inner_' + node_type + '_item'](node_id)
                embedding_result['outer_' + node_type] = self.emb_dict['outer_' + node_type](node_id)
        return embedding_result

    def get_shallow_emb(self, batch_graph):
        if batch_graph.is_block:
            node_ids_dict = batch_graph.dstdata[dgl.NID]
        else:
            node_ids_dict = {}
            for each_node_type in batch_graph.ntypes:
                nodes = batch_graph.dstnodes(each_node_type)
                if args.device == 'gpu':
                    nodes = nodes.to(torch.device('cuda'))
                node_ids_dict[each_node_type] = nodes
        embedding_result = {}
        for node_type, node_id in node_ids_dict.items():
            embedding_result['outer_' + node_type + '_bias'] = self.bias_emb_dict['outer_' + node_type + '_bias'](node_id)
            if node_type == 'user':
                for domain in args.graph_domains:
                    embedding_result['inner_' + domain + '_user_bias'] = self.bias_emb_dict['inner_' + domain + '_user_bias'](node_id)
        return embedding_result

class ShareEmbeddingEncoder(nn.Module):
    def __init__(self, graph):
        super(ShareEmbeddingEncoder, self).__init__()
        self.emb_dict = {}
        self.bias_emb_dict = {}

        ntypes = graph.ntypes
        input_dim = args.input_dim

        self.small_domains = []  #小域emb减小输入emb维度，输入到图中时，用权重矩阵变换到相同维度
        self.domain_inner_outer_differ_input = False #输入层每个域的inner和outer的emb分开，每个域通过独自的变换矩阵来分开
        self.inner_outer_differ_input = False #inner和outer的emb分开，通过一个统一的变换矩阵控制分开
        self.bias_input = False #是否存在浅层bias emb

        def create_nn_embedding(node_num, input_dim):
            emb = nn.Embedding(node_num, input_dim)
            nn.init.normal_(emb.weight, std=0.01)
            return emb

        for tt in ntypes:
            node_num = graph.num_nodes(tt)
            if tt in self.small_domains:
                emb = create_nn_embedding(node_num, int(input_dim/2))
            else:
                emb = create_nn_embedding(node_num, input_dim)
            self.emb_dict[tt] = emb
        self.emb_dict = nn.ModuleDict(self.emb_dict)

        if self.bias_input:
            for tt in ntypes:
                node_num = graph.num_nodes(tt)
                if tt in self.small_domains:
                    bias_emb = create_nn_embedding(node_num, int(input_dim / 2))
                else:
                    bias_emb = create_nn_embedding(node_num, input_dim)
                self.bias_emb_dict[tt] = bias_emb
            self.bias_emb_dict = nn.ModuleDict(self.bias_emb_dict)

        self.parameter_dict = {}

        for domain in self.small_domains:
            self.parameter_dict[domain + '_input_expand'] = nn.Parameter(
                nn.init.xavier_uniform_(torch.randn((int(input_dim/2), input_dim))), requires_grad=True)
            self.register_parameter(domain + '_input_expand', self.parameter_dict[domain + '_input_expand'])

        if self.inner_outer_differ_input:
            self.parameter_dict['inner_weight'] = nn.Parameter(nn.init.xavier_uniform_(torch.randn((input_dim, input_dim))), requires_grad=True)
            self.register_parameter('inner_weight', self.parameter_dict['inner_weight'])

        if self.domain_inner_outer_differ_input:
            for tt in ntypes:
                if tt == 'user':
                    for domain in args.graph_domains:
                        self.parameter_dict['inner_' + domain + '_user'] = nn.Parameter(nn.init.xavier_uniform_(torch.randn((input_dim, input_dim))), requires_grad=True)
                        self.register_parameter('inner_' + domain + '_user', self.parameter_dict['inner_' + domain + '_user'])
                    self.parameter_dict['outer_user'] = nn.Parameter(nn.init.xavier_uniform_(torch.randn((input_dim, input_dim))), requires_grad=True)
                    self.register_parameter('outer_user', self.parameter_dict['outer_user'])
                else:
                    self.parameter_dict['inner_' + tt + '_item'] = nn.Parameter(nn.init.xavier_uniform_(torch.randn((input_dim, input_dim))), requires_grad=True)
                    self.register_parameter('inner_' + tt + '_item', self.parameter_dict['inner_' + tt + '_item'])
                    self.parameter_dict['outer_' + tt] = nn.Parameter(nn.init.xavier_uniform_(torch.randn((input_dim, input_dim))), requires_grad=True)
                    self.register_parameter('outer_' + tt, self.parameter_dict['outer_' + tt])

    def forward(self, batch_graph):
        if batch_graph.is_block:
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
            emb = self.emb_dict[node_type](node_id)

            if node_type in self.small_domains:
                emb = torch.matmul(emb, self.parameter_dict[node_type + '_input_expand'])

            if self.inner_outer_differ_input:
                if node_type == 'user':
                    for domain in args.graph_domains:
                        embedding_result['inner_' + domain + '_user'] = torch.matmul(emb, self.parameter_dict['inner_weight'])
                    embedding_result['outer_user'] = emb
                else:
                    embedding_result['inner_' + node_type + '_item'] = torch.matmul(emb, self.parameter_dict['inner_weight'])
                    embedding_result['outer_' + node_type] = emb
            elif self.domain_inner_outer_differ_input:
                if node_type == 'user':
                    for domain in args.graph_domains:
                        embedding_result['inner_' + domain + '_user'] = torch.matmul(emb, self.parameter_dict['inner_' + domain + '_user'])
                    embedding_result['outer_user'] = torch.matmul(emb, self.parameter_dict['outer_user'])
                else:
                    embedding_result['inner_' + node_type + '_item'] = torch.matmul(emb, self.parameter_dict['inner_' + node_type + '_item'])
                    embedding_result['outer_' + node_type] = torch.matmul(emb, self.parameter_dict['outer_' + node_type])
            else:
                if node_type == 'user':
                    for domain in args.graph_domains:
                        embedding_result['inner_' + domain + '_user'] = emb
                    embedding_result['outer_user'] = emb
                else:
                    embedding_result['inner_' + node_type + '_item'] = emb
                    embedding_result['outer_' + node_type] = emb
        return embedding_result

    def get_shallow_emb(self, batch_graph):
        if batch_graph.is_block:
            node_ids_dict = batch_graph.dstdata[dgl.NID]
        else:
            node_ids_dict = {}
            for each_node_type in batch_graph.ntypes:
                nodes = batch_graph.dstnodes(each_node_type)
                if args.device == 'gpu':
                    nodes = nodes.to(torch.device('cuda'))
                node_ids_dict[each_node_type] = nodes
        result = {}
        result['input_feats'] = {}
        for node_type, node_id in node_ids_dict.items():
            if node_type in self.small_domains:
                result['input_feats'][node_type] = torch.matmul(self.emb_dict[node_type](node_id), self.parameter_dict[node_type + '_input_expand'])
            else:
                result['input_feats'][node_type] = self.emb_dict[node_type](node_id)
        return result

class MyHeroGraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, graph):
        super(MyHeroGraphConv, self).__init__()
        self.conv_dict = {}

        rel_names = graph.etypes
        for rel in rel_names:
            self.conv_dict[rel] = dglnn.GraphConv(in_dim, out_dim)
            # self.conv_dict['inner_' + rel] = dglnn.GraphConv(in_dim, out_dim)
        self.conv_dict = nn.ModuleDict(self.conv_dict)

        for _, v in self.conv_dict.items():
            set_allow_zero_in_degree_fn = getattr(v, 'set_allow_zero_in_degree', None)
            if callable(set_allow_zero_in_degree_fn):
                set_allow_zero_in_degree_fn(True)

    def get_innner_key(self, stype, etype, dtype):
        if dtype == 'user' and stype in args.graph_domains:
            return [('inner_' + stype + '_item', 'inner_' + stype + '_user')]
        elif dtype in args.graph_domains and stype == 'user':
            return [('inner_' + dtype + '_user', 'inner_' + dtype + '_item')]
        elif stype == dtype:
            if stype == 'user':
                ret = []
                for domain in args.graph_domains:
                    ret.append(( 'inner_' + domain + '_user', 'inner_' + domain + '_user'))
                return ret
            if dtype in args.graph_domains:
                return [('inner_' + dtype + '_item', 'inner_' + dtype + '_item')]

    def forward(self, graph, feature_dict):
        output_dict = {}
        for key in feature_dict:
            if key == 'input_feats': continue
            output_dict[key] = []
        for stype, etype, dtype in graph.canonical_etypes:
            rel_graph = graph[stype, etype, dtype]
            mod = self.conv_dict[etype]
            output = mod(rel_graph, feature_dict['outer_' + stype])
            output_dict['outer_' + dtype].append(output)

            # mod = self.conv_dict['inner_' + etype]
            inner_key = self.get_innner_key(stype, etype, dtype)
            # print('edges:{},{},{}; inner_key:{}'.format(stype, etype, dtype, inner_key))
            for key in inner_key:
                output = mod(rel_graph, feature_dict[key[0]])
                output_dict[key[1]].append(output)

        for dtype in output_dict.keys():
            cat_output = torch.stack(output_dict[dtype], dim = 0)
            output_dict[dtype] = torch.sum(cat_output, dim = 0)
        return output_dict

class GraphEncoder(nn.Module):
    def __init__(self, graph, domains, projection_on):
        super(GraphEncoder, self).__init__()
        self.output_mode = 'add2'
        self.domains = domains
        self.emb = ShareEmbeddingEncoder(graph)
        # self.emb = EmbeddingEncoder(graph)

        self.input_layer = MyHeroGraphConv(args.input_dim, args.hidden_dim, graph)
        self.hidden_layer = MyHeroGraphConv(args.hidden_dim, args.hidden_dim, graph)
        self.output_layer = MyHeroGraphConv(args.hidden_dim, args.output_dim, graph)
        self.layers = nn.ModuleList()
        self.layers.append(self.input_layer)
        for i in range(args.n_layer - 2):
            self.layers.append(self.hidden_layer)
        self.layers.append(self.output_layer)

        self.loss_weight = 0.0

        self.projection_on = projection_on
        if self.projection_on:
            self.output_weight_dict = {}
            for domain in self.domains:
                dims = args.output_dim
                if self.output_mode == 'cat2':
                    dims = 2 * dims
                self.output_weight_dict[domain] = nn.Parameter(nn.init.normal_(torch.randn((dims, dims))), requires_grad=True)
                self.register_parameter(domain, self.output_weight_dict[domain])
                if self.output_mode == 'add2_weight':   #inner和outer权重不一致
                    self.output_weight_dict['inner_' + domain] = nn.Parameter(nn.init.normal_(torch.randn((dims, dims))), requires_grad=True)
                    self.register_parameter('inner_' + domain, self.output_weight_dict['inner_' + domain])

        for k, v in self.named_parameters():
            print(k)
        for k in self.parameters():
            print(k)

    def get_output(self, feats, shallow_feats):
        result = {}
        for domain in self.domains:
            if self.output_mode == 'single_domain_bpr':
                user_emb = shallow_feats[domain + 'user']
                item_emb = shallow_feats[domain]
                self.projection_on = False
            elif self.output_mode == 'cross_domain_bpr':
                user_emb = shallow_feats['user']
                item_emb = shallow_feats[domain]
                self.projection_on = False
            elif self.output_mode == 'cat2':
                user_emb = torch.cat([feats['outer_user'], feats['inner_' + domain + '_user']], axis=-1)
                item_emb = torch.cat([feats['outer_' + domain], feats['inner_' + domain + '_item']], axis=-1)
            elif self.output_mode == 'add2':
                user_emb = torch.add(feats['outer_user'], feats['inner_' + domain + '_user'])
                item_emb = torch.add(feats['outer_' + domain], feats['inner_' + domain + '_item'])
            elif self.output_mode == 'add2_weight':
                outer_user_emb = torch.matmul(feats['outer_user'], self.output_weight_dict[domain])
                inter_user_emb = torch.matmul(feats['inner_' + domain + '_user'], self.output_weight_dict['inner_' + domain])
                user_emb = torch.add(outer_user_emb, inter_user_emb)
                item_emb = torch.add(feats['outer_' + domain], feats['inner_' + domain + '_item'])
                self.projection_on = False
            elif self.output_mode == 'add3':
                user_emb = torch.add(torch.add(feats['outer_user'], feats['inner_' + domain + '_user']), shallow_feats['input_feats']['user'])
                item_emb = torch.add(torch.add(feats['outer_' + domain], feats['inner_' + domain + '_item']), shallow_feats['input_feats'][domain])
            elif self.output_mode == 'add4':
                user_emb = torch.add(feats['outer_user'], feats['inner_' + domain + '_user']) + shallow_feats['outer_user_bias'] + shallow_feats['inner_' + domain + '_user_bias']
                item_emb = torch.add(feats['outer_' + domain], feats['inner_' + domain + '_item']) + shallow_feats['outer_' + domain + '_bias']
            else:
                user_emb = feats['outer_user']
                item_emb = feats['outer_' + domain]

            if self.projection_on:
                user_emb = torch.matmul(user_emb, self.output_weight_dict[domain])

            result[domain + '_user_output'] = user_emb
            result[domain + '_item_output'] = item_emb
        return result

    def get_graph_encoder(self, blocks, flag = 'graph', train = True):
        result = dict()
        if flag == 'graph':
            feats = self.emb(blocks)
            shallow_feats = self.emb.get_shallow_emb(blocks)
        else:
            feats = self.emb(blocks[0])
            shallow_feats = self.emb.get_shallow_emb(blocks[-1])

        result['input_feats'] = shallow_feats['input_feats']
        result['layer_0_feats'] = feats

        for i, layer in enumerate(self.layers):
            if flag == 'graph':
                feats = layer(blocks, feats)
            else:
                feats = layer(blocks[i], feats)
            result['layer_' + str(i+1) + '_feats'] = feats
            if i != len(self.layers) - 1:
                feats = {k: F.relu(v) for k, v in feats.items()}
                feats = {k: F.dropout(v, p=args.dropout, training=train) for k, v in feats.items()}

        result['layer_2_feats']['output'] = self.get_output(result['layer_2_feats'], shallow_feats)

        return result

    def get_score(self, h, graph, flag = 0):
        score_dict = {}
        with graph.local_scope():
            for domain in self.domains:
                if flag == 1:
                    graph.ndata['h'] = h
                else:
                    graph.nodes['user'].data['h'] = h[domain + '_user_output']
                    graph.nodes[domain].data['h'] = h[domain + '_item_output']
                graph.apply_edges(dglfn.u_dot_v('h', 'h', 'score'), etype=('user', 'user_click_' + domain , domain))
                score = graph.edges['user_click_' + domain].data['score']
                score_dict[domain] = score
        return score_dict

    def get_contrastive_loss(self, h):
        # for key in h:
        #     if "user_output" in key:
        #         user_num = h[key].shape[0]
        #         break
        #
        # contrastive_train_data = np.random.choice(user_num, size = (user_num, 2), replace = True)
        # emb0 = h['feats']['inner_movie_user'][contrastive_train_data[:, 0]]
        # emb1 = h['feats']['inner_music_user'][contrastive_train_data[:, 0]]
        # emb2 = h['feats']['inner_music_user'][contrastive_train_data[:, 1]]
        # pos_score = torch.sum(emb0 * emb1, axis = -1)
        # neg_score = torch.sum(emb0 * emb2, axis = -1)
        # loss = torch.mean(nn.functional.softplus(neg_score - pos_score))
        loss = InfoNCE()
        # v8.12
        # query = h['graph_feats']['inner_movie_user']
        # positive_key = h['graph_feats']['inner_music_user']
        # v8.13
        # query = h['graph_feats']['outer_user']
        # positive_key = h['graph_feats']['inner_music_user']
        # v8.14
        query = h['graph_feats']['outer_user']
        positive_key = h['input_feats']['user']
        output = loss(query, positive_key)
        return output

    def get_loss(self, emb_dict, positive_graph, negative_graph):
        loss_list = set(['default', 'cluster_loss_mean_reg'])

        h = emb_dict['layer_2_feats']['output']
        pos_score = self.get_score(h, positive_graph)
        neg_score = self.get_score(h, negative_graph)

        pos_score = torch.cat([pos_score[domain] for domain in self.domains])
        pos_score = torch.repeat_interleave(pos_score.squeeze(), 5).reshape([-1, 1])
        neg_score = torch.cat([neg_score[domain] for domain in self.domains])

        # print('cmp score:', torch.cat((pos_score, neg_score), axis = -1))

        main_loss = torch.mean(nn.functional.softplus(neg_score - pos_score))
        loss = main_loss

        if "input_loss" in loss_list:
            h = emb_dict['input_feats']
            pos_score = self.get_score(h, positive_graph, flag = 1)
            neg_score = self.get_score(h, negative_graph, flag = 1)

            pos_score = torch.cat([pos_score[domain] for domain in self.domains])
            pos_score = torch.repeat_interleave(pos_score.squeeze(), 5).reshape([-1, 1])
            neg_score = torch.cat([neg_score[domain] for domain in self.domains])

            input_loss = torch.mean(nn.functional.softplus(neg_score - pos_score))
            loss += 2.0 * input_loss
            print("loss_weight:", self.loss_weight, ", loss:", loss.item(), ", main_loss:", main_loss.item(), ", input_loss:", input_loss.item())

        if "cluster_loss_mean" in loss_list:
            mse_loss = nn.MSELoss()
            cluster_mean_loss = mse_loss(
                emb_dict['layer_2_feats']['inner_music_user'] + emb_dict['layer_2_feats']['inner_movie_user'] +
                emb_dict['layer_2_feats']['inner_book_user'],
                3.0 * emb_dict['layer_2_feats']['outer_user'])

            loss += 0.1 * cluster_mean_loss
            print("loss_weight:", self.loss_weight, ", loss:", loss.item(), ", main_loss:", main_loss.item(), ", cluster_mean_loss:", cluster_mean_loss.item())

        if "cluster_loss_mean_reg" in loss_list:
            mse_loss = nn.MSELoss()
            emb1 = (emb_dict['layer_2_feats']['inner_music_user'] + emb_dict['layer_2_feats']['inner_movie_user'] + emb_dict['layer_2_feats']['inner_book_user'])/3.0
            emb2 = emb_dict['layer_2_feats']['outer_user']
            emb_reg = (emb1 * 3.0 + emb2 * 1.0)/4.0
            cluster_mean_loss = torch.mean(torch.sum((emb1-emb2)**2, axis = -1)/ torch.sum(emb_reg**2, axis = -1))

            loss += 0.1 * torch.mean(cluster_mean_loss)
            print("loss_weight:", self.loss_weight, ", loss:", loss.item(), ", main_loss:", main_loss.item(), ", cluster_mean_loss:", cluster_mean_loss.item())

        if "cluster_loss_mean_reg1" in loss_list:
            mse_loss = nn.MSELoss()
            emb1 = (emb_dict['layer_2_feats']['inner_music_user'] + emb_dict['layer_2_feats']['inner_movie_user'] + emb_dict['layer_2_feats']['inner_book_user'])/3.0
            emb2 = emb_dict['layer_2_feats']['outer_user']
            cluster_mean_loss = torch.sum((emb1-emb2)**2, axis = -1)/ torch.sqrt(torch.sum(emb1 ** 2, axis = -1) * torch.sum(emb2 ** 2, axis = -1))
            cluster_mean_loss = torch.mean(cluster_mean_loss)
            loss += 0.1 * torch.mean(cluster_mean_loss)
            print("loss_weight:", self.loss_weight, ", loss:", loss.item(), ", main_loss:", main_loss.item(), ", cluster_mean_loss:", cluster_mean_loss.item())

        if "cluster_loss_cosin" in loss_list:
            cluster_loss_cosin = -torch.mean(torch.nn.functional.cosine_similarity(emb_dict['layer_2_feats']['outer_user'],
                                                                           emb_dict['layer_2_feats'][
                                                                               'inner_movie_user']) \
                                     + torch.nn.functional.cosine_similarity(emb_dict['layer_2_feats']['outer_user'],
                                                                             emb_dict['layer_2_feats'][
                                                                                 'inner_music_user']) \
                                     + torch.nn.functional.cosine_similarity(emb_dict['layer_2_feats']['outer_user'],
                                                                             emb_dict['layer_2_feats'][
                                                                                 'inner_book_user']), axis=-1)

            loss += main_loss + 0.1 * cluster_loss_cosin
            print("loss_weight:", self.loss_weight, ", loss:", loss.item(), ", main_loss:", main_loss.item(),
                  ", cluster_loss_cosin:", cluster_loss_cosin.item())

        return loss

    def get_domain_loss(self, emb_dict, positive_graph, negative_graph):

        h = emb_dict['layer_2_feats']['output']
        pos_score = self.get_score(h, positive_graph)
        neg_score = self.get_score(h, negative_graph)

        all_loss = 0
        loss_dict = {}
        for domain in self.domains:
            domain_pos_score = torch.repeat_interleave(pos_score[domain].squeeze(), 5).reshape([-1, 1])
            domain_neg_score = neg_score[domain]
            domain_loss = torch.mean(nn.functional.softplus(domain_neg_score - domain_pos_score))
            all_loss += domain_loss
            loss_dict[domain] = domain_loss

        print("loss_movie:{},loss_music:{},loss_book;:{}".format(loss_dict['movie'].item(), loss_dict['music'].item(), loss_dict['book'].item()))

        return all_loss
