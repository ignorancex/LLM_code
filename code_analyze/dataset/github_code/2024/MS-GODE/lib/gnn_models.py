import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax,add_remaining_self_loops
import math
import lib.utils as utils
import torch.autograd as autograd
from torch_geometric.nn.inits import glorot
from torch_scatter import scatter_add
class GetSubnet_topk(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k=0.5):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())
        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1
        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        a = 0
        return g, None

class GetSubnet_fast(autograd.Function):
    @staticmethod
    #def forward(ctx, scores, k=0.5):
    def forward(ctx, scores):
        return (scores >= 0).float()

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g

def mask_init(module,range=0.5):
    scores = torch.Tensor(module.weight.size())
    nn.init.kaiming_uniform_(scores, a=math.sqrt(range))
    return scores

def signed_constant(module):
    fan = nn.init._calculate_correct_fan(module.weight, 'fan_in')
    gain = nn.init.calculate_gain('relu')
    std = gain / math.sqrt(fan)
    module.weight.data = module.weight.data.sign() * std

class maskLinear(nn.Linear):
    def __init__(self, *args, num_tasks=10, bias=False, thresholding='topk', **kwargs):
        super().__init__(*args, **kwargs)
        self.num_tasks = num_tasks
        self.scores = nn.ParameterList([nn.Parameter(mask_init(self)) for _ in range(num_tasks)])
        self.use_bias = bias

        # Keep weights untrained
        self.weight.requires_grad = False
        signed_constant(self)
        self.get_subnet = GetSubnet_topk if 'topk' in thresholding else GetSubnet_fast
        self.k = float(thresholding.replace('topk','')) if 'topk' in thresholding else -1

    @torch.no_grad()
    def cache_masks(self):
        self.register_buffer(
            "stacked",
            torch.stack(
                [
                    self.get_subnet_wrap(self.scores[j],k=self.k) # return binarized values by self.get_subnet, seems to be same as .forward(None, scores), but base class Function requires calling apply instead of forward. see https://discuss.pytorch.org/t/difference-between-apply-an-call-for-an-autograd-function/13845
                    #self.get_subnet.apply(self.scores[j],k=self.k) # return binarized values by self.get_subnet, seems to be same as .forward(None, scores), but base class Function requires calling apply instead of forward. see https://discuss.pytorch.org/t/difference-between-apply-an-call-for-an-autograd-function/13845
                    #self.get_subnet.apply(self.scores[j]) # return binarized values by self.get_subnet, seems to be same as .forward(None, scores), but base class Function requires calling apply instead of forward. see https://discuss.pytorch.org/t/difference-between-apply-an-call-for-an-autograd-function/13845
                    for j in range(self.num_tasks)
                ]
            ),
        )

    def get_subnet_wrap(self, scores, k):
        if self.k==-1:
            return self.get_subnet.apply(scores)
        else:
            return self.get_subnet.apply(scores, k)

    def forward(self, x):
        if self.task < 0:
            # Superimposed forward pass
            alpha_weights = self.alphas[: self.num_tasks_learned]
            idxs = (alpha_weights > 0).squeeze().view(self.num_tasks_learned)
            if len(idxs.shape) == 0: # if only one non-zero entry exists
                idxs = idxs.view(1)
            subnet = (alpha_weights[idxs]
                    * self.stacked[: self.num_tasks_learned][idxs]
            ).sum(dim=0)
        else:
            # Subnet forward pass (given task info in self.task)
            subnet = self.get_subnet_wrap(self.scores[self.task],self.k)
            #subnet = self.get_subnet.apply(self.scores[self.task])
        w = self.weight * subnet
        x = F.linear(x, w, self.bias) if self.use_bias else F.linear(x, w)
        return x

    def __repr__(self):
        #return f"MultitaskMaskLinear({self.in_dims}, {self.out_dims})"
        return f"maskLinear()"

def set_model_task(model, task, verbose=True):
    for n, m in model.named_modules():
        if isinstance(m, maskLinear):
            if verbose:
                print(f"=> Set task of {n} to {task}")
            m.task = task
        if hasattr(m, 'layer_norm'):
            m.layer_norm = m.layer_norms[task] if task!=-1 else m.layer_norms[model.num_tasks_learned - 1] # -1 denotes unknown task ids, use the finally learnt layer_norm

def get_model_scores(model):
    scores = []
    for n, m in model.named_modules():
        if isinstance(m, maskLinear):
            ss = m.scores
            scores.append(ss)
    return scores

class maskTemporalEncoding(nn.Module):

    def __init__(self, d_hid):
        super().__init__()
        self.d_hid = d_hid
        self.div_term = torch.FloatTensor([1 / np.power(10000, 2 * (hid_j // 2) / self.d_hid) for hid_j in range(self.d_hid)]) #[20]
        self.div_term = torch.reshape(self.div_term,(1,-1))
        self.div_term = nn.Parameter(self.div_term,requires_grad=False)

    def forward(self, t):
        '''
        :param t: [n,1]
        :return:
        '''
        t = t.view(-1,1)
        t = t *200  # scale from [0,1] --> [0,200], align with 'attention is all you need'
        position_term = torch.matmul(t,self.div_term)
        position_term[:,0::2] = torch.sin(position_term[:,0::2])
        position_term[:,1::2] = torch.cos(position_term[:,1::2])

        return position_term


class maskGTrans(MessagePassing):

    def __init__(self, n_heads=2,d_input=6, d_k=6,dropout = 0.1, thresholding='topk', n_tasks = 10,**kwargs):
        super().__init__(aggr='add', **kwargs)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

        self.d_input = d_input
        self.d_k = d_k//n_heads
        self.d_q = d_k//n_heads
        self.d_e = d_k//n_heads
        self.d_sqrt = math.sqrt(d_k//n_heads)

        #Attention Layer Initialization
        self.w_k_list_same = nn.ModuleList([maskLinear(self.d_input, self.d_k, bias=False,thresholding=thresholding) for i in range(self.n_heads)])
        self.w_k_list_diff = nn.ModuleList([maskLinear(self.d_input, self.d_k, bias=False,thresholding=thresholding) for i in range(self.n_heads)])
        self.w_q_list = nn.ModuleList([maskLinear(self.d_input, self.d_q, bias=False,thresholding=thresholding) for i in range(self.n_heads)])
        self.w_v_list_same = nn.ModuleList([maskLinear(self.d_input, self.d_e, bias=False,thresholding=thresholding) for i in range(self.n_heads)])
        self.w_v_list_diff = nn.ModuleList([maskLinear(self.d_input, self.d_k, bias=False,thresholding=thresholding) for i in range(self.n_heads)])

        #self.w_transfer = nn.ModuleList([nn.Linear(self.d_input*2, self.d_k, bias=False) for i in range(self.n_heads)])
        self.w_transfer = nn.ModuleList([maskLinear(self.d_input +1, self.d_k, bias=False,thresholding=thresholding) for i in range(self.n_heads)])

        #initiallization
        utils.init_network_weights(self.w_k_list_same)
        utils.init_network_weights(self.w_k_list_diff)
        utils.init_network_weights(self.w_q_list)
        utils.init_network_weights(self.w_v_list_same)
        utils.init_network_weights(self.w_v_list_diff)
        utils.init_network_weights(self.w_transfer)

        #Temporal Layer
        self.temporal_net = maskTemporalEncoding(d_input)

        #Normalization
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_input) for i in range(n_tasks)])
        self.layer_norm = self.layer_norms[0]

    def forward(self, x, edge_index, edge_value,time_nodes,edge_same):

        residual = x
        x = self.layer_norm(x)

        return self.propagate(edge_index, x=x, edges_temporal=edge_value, edge_same=edge_same, residual=residual)

    def message(self, x_j,x_i,edge_index_i, edges_temporal,edge_same):
        '''

           :param x_j: [num_edge, d] sender
           :param x_i: [num_edge,d]  receiver
           :param edge_index_i:  receiver node list [num_edge]
           :param edges_temporal: [num_edge,d]
           :return:
        '''
        messages = []
        edge_same = edge_same.view(-1,1)
        for i in range(self.n_heads):
            k_linear_same = self.w_k_list_same[i]
            k_linear_diff = self.w_k_list_diff[i]
            q_linear = self.w_q_list[i]
            v_linear_same = self.w_v_list_same[i]
            v_linear_diff = self.w_v_list_diff[i]
            w_transfer = self.w_transfer[i]

            edge_temporal_true = self.temporal_net(edges_temporal)
            edges_temporal = edges_temporal.view(-1,1)
            x_j_transfer = F.gelu(w_transfer(torch.cat((x_j, edges_temporal), dim=1))) + edge_temporal_true

            attention = self.each_head_attention(x_j_transfer,k_linear_same,k_linear_diff,q_linear,x_i,edge_same) #[4,1]
            attention = torch.div(attention,self.d_sqrt)
            attention_norm = softmax(attention,edge_index_i) #[4,1]
            sender_same = edge_same * v_linear_same(x_j_transfer)
            sender_diff = (1-edge_same) * v_linear_diff(x_j_transfer)
            sender = sender_same + sender_diff

            message  = attention_norm * sender #[4,3]
            messages.append(message)

        message_all_head  = torch.cat(messages,1)

        return message_all_head

    def each_head_attention(self,x_j_transfer,w_k_same,w_k_diff,w_q,x_i,edge_same):
        x_i = w_q(x_i) #receiver #[num_edge,d*heads]

        # wraping k

        sender_same = edge_same * w_k_same(x_j_transfer)
        sender_diff = (1 - edge_same) * w_k_diff(x_j_transfer)
        sender = sender_same + sender_diff #[num_edge,d]

       # Calculate attention score
        attention = torch.bmm(torch.unsqueeze(sender,1),torch.unsqueeze(x_i,2))

        return torch.squeeze(attention,1)

    def update(self, aggr_out,residual):
        x_new = residual + F.gelu(aggr_out)

        return self.dropout(x_new)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)


class maskNRIConv(nn.Module):
    """MLP decoder module."""

    def __init__(self, in_channels, out_channels, dropout=0., skip_first=False,thresholding='topk'):
        super().__init__()

        self.edge_types = 2
        self.msg_fc1 = nn.ModuleList(
            [maskLinear(2 * in_channels, out_channels, bias=False,thresholding=thresholding) for _ in range(self.edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [maskLinear(out_channels, out_channels, bias=False,thresholding=thresholding) for _ in range(self.edge_types)])
        self.msg_out_shape = out_channels
        self.skip_first_edge_type = skip_first

        self.out_fc1 = maskLinear(in_channels + out_channels, out_channels, bias=False,thresholding=thresholding)
        self.out_fc2 = maskLinear(out_channels, out_channels, bias=False,thresholding=thresholding)
        self.dropout = nn.Dropout(dropout)


        #input data
        self.rel_type = None
        self.rel_rec = None
        self.rel_send = None


    def forward(self, inputs, pred_steps=1):
        # NOTE: Assumes that we have the same graph across all samples.
        '''

        :param inputs: [b,n_ball,feat]
        :param rel_type: [b,20,2]
        :param rel_rec:  [20,5] : [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4]
        :param rel_send: [20,5]: [1,2,3,4,0,2,3,4,0,1,3,4,0,1,2,4,0,1,2,3]
        :param pred_steps:10
        :return:
        '''
        rel_type = self.rel_type
        rel_rec = self.rel_rec
        rel_send = self.rel_send

        # Node2edge
        receivers = torch.matmul(rel_rec, inputs)  # [b,20,256], 20edges, receiver features: [20,4]
        senders = torch.matmul(rel_send, inputs)  # [b,20,256], 20edges, receiver_features: [20,4]
        pre_msg = torch.cat([senders, receivers], dim=-1)  # 【b,20,256*2]

        all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1),self.msg_out_shape)  # [b,20,256]

        if inputs.is_cuda:
            all_msgs = all_msgs.cuda(inputs.get_device())

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = self.dropout(msg)
            msg = F.relu(self.msg_fc2[i](msg))  # 【b,20,256]
            msg = msg * rel_type[:, :, i:i + 1] # [b,20,256]
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1) #[b,5,256]

        # Skip connection
        aug_inputs = torch.cat([inputs, agg_msgs], dim=-1)

        # Output MLP
        pred = self.dropout(F.relu(self.out_fc1(aug_inputs)))
        pred = self.dropout(F.relu(self.out_fc2(pred)))

        # Predict position/velocity difference
        return inputs + pred





class maskGNN(nn.Module):
    '''
    wrap up multiple layers
    '''

    def __init__(self, in_dim, n_hid, out_dim, n_heads, n_layers, dropout=0.2, conv_name='GTrans', aggregate="add",thresholding='topk',n_tasks=10):
        super().__init__()
        # for mask selection
        self.alphas = []
        self.gcs = nn.ModuleList()
        self.in_dim = in_dim
        self.n_hid = n_hid
        self.drop = nn.Dropout(dropout)
        self.adapt_ws = maskLinear(in_dim, n_hid, bias=False,thresholding=thresholding)
        self.sequence_w = maskLinear(n_hid, n_hid, bias=False,thresholding=thresholding)  # for encoder
        self.out_w_ode = maskLinear(n_hid, out_dim, bias=False,thresholding=thresholding)
        self.out_w_encoder = maskLinear(n_hid, out_dim * 2, bias=False,thresholding=thresholding)

        # initialization
        for w in [self.adapt_ws,self.sequence_w,self.out_w_ode,self.out_w_encoder]:
            utils.init_network_weights(w)

        # Normalization
        self.layer_norms = nn.ModuleList([nn.LayerNorm(n_hid) for i in range(n_tasks)])
        self.layer_norm = self.layer_norms[0]
        self.aggregate = aggregate
        for l in range(n_layers):
            self.gcs.append(maskGeneralConv(conv_name, n_hid, n_hid, n_heads, dropout, thresholding))

        if conv_name == 'GTrans':
            self.temporal_net = TemporalEncoding(n_hid)
            # self.w_transfer = nn.Linear(self.n_hid * 2, self.n_hid, bias=False)
            self.w_transfer = maskLinear(self.n_hid + 1, self.n_hid, bias=False,thresholding=thresholding)
            utils.init_network_weights(self.w_transfer)

    def forward(self, x, edge_time=None, edge_index=None, x_time=None, edge_same=None, batch=None,
                batch_y=None):  # aggregation part
        h_0 = F.relu(self.adapt_ws(x))
        h_t = self.drop(h_0)
        h_t = self.layer_norm(h_t)

        for gc in self.gcs:
            h_t = gc(h_t, edge_index, edge_time, x_time, edge_same)  # [num_nodes,d]

        ### Output
        if batch != None:  ## for encoder
            batch_new = self.rewrite_batch(batch, batch_y)  # group by balls
            if self.aggregate == "add":
                h_ball = global_mean_pool(h_t, batch_new)  # [num_ball,d], without activation

            elif self.aggregate == "attention":
                # h_t = F.gelu(self.w_transfer(torch.cat((h_t, edges_temporal), dim=1))) + edges_temporal
                x_time = x_time.view(-1, 1)
                h_t = F.gelu(self.w_transfer(torch.cat((h_t, x_time), dim=1))) + self.temporal_net(x_time)
                attention_vector = F.relu(self.sequence_w(
                    global_mean_pool(h_t, batch_new)))  # [num_ball,d] ,graph vector with activation Relu
                attention_vector_expanded = self.attention_expand(attention_vector, batch, batch_y)  # [num_nodes,d]
                attention_nodes = torch.sigmoid(torch.squeeze(
                    torch.bmm(torch.unsqueeze(attention_vector_expanded, 1), torch.unsqueeze(h_t, 2)))).view(-1,
                                                                                                             1)  # [num_nodes]
                nodes_attention = attention_nodes * h_t  # [num_nodes,d]
                h_ball = global_mean_pool(nodes_attention, batch_new)  # [num_ball,d] without activation

            h_out = self.out_w_encoder(h_ball)  # [num_ball,2d]
            mean, mu = self.split_mean_mu(h_out)
            mu = mu.abs()
            return mean, mu

        else:  # for ODE
            # h_t [n_ball,d]
            h_out = self.out_w_ode(h_t)

        return h_out

    def rewrite_batch(self, batch, batch_y):
        assert (torch.sum(batch_y).item() == list(batch.size())[0])
        batch_new = torch.zeros_like(batch)
        group_num = 0
        current_index = 0
        for ball_time in batch_y:
            batch_new[current_index:current_index + ball_time] = group_num
            group_num += 1
            current_index += ball_time.item()

        return batch_new

    def attention_expand(self, attention_ball, batch, batch_y):
        '''

        :param attention_ball: [num_ball, d]
        :param batch: [num_nodes,d]
        :param batch_new: [num_ball,d]
        :return:
        '''
        node_size = batch.size()[0]
        dim = attention_ball.size()[1]
        new_attention = torch.ones(node_size, dim)
        if attention_ball.device != torch.device("cpu"):
            new_attention = new_attention.cuda(attention_ball.get_device())

        group_num = 0
        current_index = 0
        for ball_time in batch_y:
            new_attention[current_index:current_index + ball_time] = attention_ball[group_num]
            group_num += 1
            current_index += ball_time.item()

        return new_attention

    def split_mean_mu(self, h):
        last_dim = h.size()[-1] // 2
        res = h[:, :last_dim], h[:, last_dim:]
        return res


class TemporalEncoding(nn.Module):

    def __init__(self, d_hid):
        super(TemporalEncoding, self).__init__()
        self.d_hid = d_hid
        self.div_term = torch.FloatTensor([1 / np.power(10000, 2 * (hid_j // 2) / self.d_hid) for hid_j in range(self.d_hid)]) #[20]
        self.div_term = torch.reshape(self.div_term,(1,-1))
        self.div_term = nn.Parameter(self.div_term,requires_grad=False)

    def forward(self, t):
        '''

        :param t: [n,1]
        :return:
        '''
        t = t.view(-1,1)
        t = t *200  # scale from [0,1] --> [0,200], align with 'attention is all you need'
        position_term = torch.matmul(t,self.div_term)
        position_term[:,0::2] = torch.sin(position_term[:,0::2])
        position_term[:,1::2] = torch.cos(position_term[:,1::2])

        return position_term


class GTrans(MessagePassing):

    def __init__(self, n_heads=2,d_input=6, d_k=6,dropout = 0.1,**kwargs):
        super(GTrans, self).__init__(aggr='add', **kwargs)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

        self.d_input = d_input
        self.d_k = d_k//n_heads
        self.d_q = d_k//n_heads
        self.d_e = d_k//n_heads
        self.d_sqrt = math.sqrt(d_k//n_heads)

        #Attention Layer Initialization
        self.w_k_list_same = nn.ModuleList([nn.Linear(self.d_input, self.d_k, bias=True) for i in range(self.n_heads)])
        self.w_k_list_diff = nn.ModuleList([nn.Linear(self.d_input, self.d_k, bias=True) for i in range(self.n_heads)])
        self.w_q_list = nn.ModuleList([nn.Linear(self.d_input, self.d_q, bias=True) for i in range(self.n_heads)])
        self.w_v_list_same = nn.ModuleList([nn.Linear(self.d_input, self.d_e, bias=True) for i in range(self.n_heads)])
        self.w_v_list_diff = nn.ModuleList([nn.Linear(self.d_input, self.d_k, bias=True) for i in range(self.n_heads)])

        #self.w_transfer = nn.ModuleList([nn.Linear(self.d_input*2, self.d_k, bias=True) for i in range(self.n_heads)])
        self.w_transfer = nn.ModuleList([nn.Linear(self.d_input +1, self.d_k, bias=True) for i in range(self.n_heads)])

        #initiallization
        utils.init_network_weights(self.w_k_list_same)
        utils.init_network_weights(self.w_k_list_diff)
        utils.init_network_weights(self.w_q_list)
        utils.init_network_weights(self.w_v_list_same)
        utils.init_network_weights(self.w_v_list_diff)
        utils.init_network_weights(self.w_transfer)


        #Temporal Layer
        self.temporal_net = TemporalEncoding(d_input)

        #Normalization
        self.layer_norm = nn.LayerNorm(d_input)

    def forward(self, x, edge_index, edge_value,time_nodes,edge_same):

        residual = x
        x = self.layer_norm(x)

        return self.propagate(edge_index, x=x, edges_temporal=edge_value, edge_same=edge_same, residual=residual)

    def message(self, x_j,x_i,edge_index_i, edges_temporal,edge_same):
        '''

           :param x_j: [num_edge, d] sender
           :param x_i: [num_edge,d]  receiver
           :param edge_index_i:  receiver node list [num_edge]
           :param edges_temporal: [num_edge,d]
           :return:
        '''
        messages = []
        edge_same = edge_same.view(-1,1)
        for i in range(self.n_heads):
            k_linear_same = self.w_k_list_same[i]
            k_linear_diff = self.w_k_list_diff[i]
            q_linear = self.w_q_list[i]
            v_linear_same = self.w_v_list_same[i]
            v_linear_diff = self.w_v_list_diff[i]
            w_transfer = self.w_transfer[i]

            edge_temporal_true = self.temporal_net(edges_temporal)
            edges_temporal = edges_temporal.view(-1,1)
            x_j_transfer = F.gelu(w_transfer(torch.cat((x_j, edges_temporal), dim=1))) + edge_temporal_true

            attention = self.each_head_attention(x_j_transfer,k_linear_same,k_linear_diff,q_linear,x_i,edge_same) #[4,1]
            attention = torch.div(attention,self.d_sqrt)
            attention_norm = softmax(attention,edge_index_i) #[4,1]
            sender_same = edge_same * v_linear_same(x_j_transfer)
            sender_diff = (1-edge_same) * v_linear_diff(x_j_transfer)
            sender = sender_same + sender_diff

            message  = attention_norm * sender #[4,3]
            messages.append(message)

        message_all_head  = torch.cat(messages,1)

        return message_all_head

    def each_head_attention(self,x_j_transfer,w_k_same,w_k_diff,w_q,x_i,edge_same):
        x_i = w_q(x_i) #receiver #[num_edge,d*heads]

        # wraping k

        sender_same = edge_same * w_k_same(x_j_transfer)
        sender_diff = (1 - edge_same) * w_k_diff(x_j_transfer)
        sender = sender_same + sender_diff #[num_edge,d]

       # Calculate attention score
        attention = torch.bmm(torch.unsqueeze(sender,1),torch.unsqueeze(x_i,2))

        return torch.squeeze(attention,1)

    def update(self, aggr_out,residual):
        x_new = residual + F.gelu(aggr_out)

        return self.dropout(x_new)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)


class NRIConv(nn.Module):
    """MLP decoder module."""

    def __init__(self, in_channels, out_channels, dropout=0., skip_first=False):
        super(NRIConv, self).__init__()

        self.edge_types = 2
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * in_channels, out_channels) for _ in range(self.edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(out_channels, out_channels) for _ in range(self.edge_types)])
        self.msg_out_shape = out_channels
        self.skip_first_edge_type = skip_first

        self.out_fc1 = nn.Linear(in_channels + out_channels, out_channels)
        self.out_fc2 = nn.Linear(out_channels, out_channels)
        self.dropout = nn.Dropout(dropout)


        #input data
        self.rel_type = None
        self.rel_rec = None
        self.rel_send = None



    def forward(self, inputs, pred_steps=1):
        # NOTE: Assumes that we have the same graph across all samples.
        '''

        :param inputs: [b,n_ball,feat]
        :param rel_type: [b,20,2]
        :param rel_rec:  [20,5] : [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4]
        :param rel_send: [20,5]: [1,2,3,4,0,2,3,4,0,1,3,4,0,1,2,4,0,1,2,3]
        :param pred_steps:10
        :return:
        '''
        rel_type = self.rel_type
        rel_rec = self.rel_rec
        rel_send = self.rel_send

        # Node2edge
        receivers = torch.matmul(rel_rec, inputs)  # [b,20,256], 20edges, receiver features: [20,4]
        senders = torch.matmul(rel_send, inputs)  # [b,20,256], 20edges, receiver_features: [20,4]
        pre_msg = torch.cat([senders, receivers], dim=-1)  # 【b,20,256*2]

        all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1),self.msg_out_shape)  # [b,20,256]

        if inputs.is_cuda:
            all_msgs = all_msgs.cuda(inputs.get_device())

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = self.dropout(msg)
            msg = F.relu(self.msg_fc2[i](msg))  # 【b,20,256]
            msg = msg * rel_type[:, :, i:i + 1] # [b,20,256]
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1) #[b,5,256]

        # Skip connection
        aug_inputs = torch.cat([inputs, agg_msgs], dim=-1)

        # Output MLP
        pred = self.dropout(F.relu(self.out_fc1(aug_inputs)))
        pred = self.dropout(F.relu(self.out_fc2(pred)))

        # Predict position/velocity difference
        return inputs + pred


class maskGeneralConv(nn.Module):
    '''
    general layers
    '''
    def __init__(self, conv_name, in_hid, out_hid, n_heads, dropout, thresholding):
        super().__init__()
        self.conv_name = conv_name
        if self.conv_name in ['GTrans','maskGTrans']:
            self.base_conv = maskGTrans(n_heads,in_hid,out_hid,dropout, thresholding=thresholding)
        elif self.conv_name in ["NRI","maskNRI"]:
            self.base_conv = maskNRIConv(in_hid,out_hid,dropout, thresholding=thresholding)


    def forward(self, x, edge_index, edge_time, x_time,edge_same):
        if self.conv_name in ['GTrans','maskGTrans']:
            return self.base_conv(x, edge_index, edge_time, x_time,edge_same)
        elif self.conv_name in ["NRI","maskNRI"]:
            return self.base_conv(x)


class GeneralConv(nn.Module):
    '''
    general layers
    '''
    def __init__(self, conv_name, in_hid, out_hid, n_heads, dropout):
        super(GeneralConv, self).__init__()
        self.conv_name = conv_name
        if self.conv_name == 'GTrans':
            self.base_conv = GTrans(n_heads,in_hid,out_hid,dropout)
        elif self.conv_name == "NRI":
            self.base_conv = NRIConv(in_hid,out_hid,dropout)


    def forward(self, x, edge_index, edge_time, x_time,edge_same):
        if self.conv_name == 'GTrans':
            return self.base_conv(x, edge_index, edge_time, x_time,edge_same)
        elif self.conv_name =="NRI":
            return self.base_conv(x)

class GeneralConv_cgode(nn.Module):
    '''
    general layers
    '''
    def __init__(self, conv_name, in_hid, out_hid, n_heads, dropout,args):
        super(GeneralConv_cgode, self).__init__()
        self.conv_name = conv_name
        if self.conv_name == 'GTrans':
            self.base_conv = GTrans_cgode(n_heads, in_hid, out_hid, dropout)
        elif self.conv_name == "Node":
            self.base_conv = Node_GCN(in_hid,out_hid,args.n_balls,dropout)


    def forward(self, x, edge_index, edge_weight, x_time,edge_time):

        return self.base_conv(x, edge_index, edge_weight, x_time,edge_time)

class GNN(nn.Module):
    '''
    wrap up multiple layers
    '''
    def __init__(self, in_dim, n_hid,out_dim, n_heads, n_layers,
                 dropout = 0.2, conv_name = 'GTrans',aggregate = "add"):
        super(GNN, self).__init__()
        self.gcs = nn.ModuleList()
        self.in_dim    = in_dim
        self.n_hid     = n_hid
        self.drop = nn.Dropout(dropout)
        self.adapt_ws = nn.Linear(in_dim,n_hid)
        self.sequence_w = nn.Linear(n_hid,n_hid) # for encoder
        self.out_w_ode = nn.Linear(n_hid,out_dim)
        self.out_w_encoder = nn.Linear(n_hid,out_dim*2)

        #initialization
        utils.init_network_weights(self.adapt_ws)
        utils.init_network_weights(self.sequence_w)
        utils.init_network_weights(self.out_w_ode)
        utils.init_network_weights(self.out_w_encoder)

        # Normalization
        self.layer_norm = nn.LayerNorm(n_hid)
        self.aggregate = aggregate
        for l in range(n_layers):
            self.gcs.append(GeneralConv(conv_name, n_hid, n_hid,  n_heads, dropout))

        if conv_name == 'GTrans':
            self.temporal_net = TemporalEncoding(n_hid)
            #self.w_transfer = nn.Linear(self.n_hid * 2, self.n_hid, bias=True)
            self.w_transfer = nn.Linear(self.n_hid + 1, self.n_hid, bias=False)
            utils.init_network_weights(self.w_transfer)

    def forward(self, x,edge_time=None, edge_index=None, x_time=None, edge_same=None,batch= None, batch_y = None):  #aggregation part
        h_0 = F.relu(self.adapt_ws(x))
        h_t = self.drop(h_0)
        h_t = self.layer_norm(h_t)

        for gc in self.gcs:
            h_t = gc(h_t, edge_index, edge_time, x_time,edge_same)  #[num_nodes,d]

        ### Output
        if batch!= None:  ## for encoder
            batch_new = self.rewrite_batch(batch,batch_y) #group by balls
            if self.aggregate == "add":
                h_ball = global_mean_pool(h_t,batch_new) #[num_ball,d], without activation

            elif self.aggregate == "attention":


                #h_t = F.gelu(self.w_transfer(torch.cat((h_t, edges_temporal), dim=1))) + edges_temporal
                x_time = x_time.view(-1,1)
                h_t = F.gelu(self.w_transfer(torch.cat((h_t, x_time), dim=1))) + self.temporal_net(x_time)
                attention_vector = F.relu(self.sequence_w(global_mean_pool(h_t,batch_new))) #[num_ball,d] ,graph vector with activation Relu
                attention_vector_expanded = self.attention_expand(attention_vector,batch,batch_y) #[num_nodes,d]
                attention_nodes = torch.sigmoid(torch.squeeze(torch.bmm(torch.unsqueeze(attention_vector_expanded,1),torch.unsqueeze(h_t,2)))).view(-1,1) #[num_nodes]
                nodes_attention = attention_nodes * h_t #[num_nodes,d]
                h_ball = global_mean_pool(nodes_attention,batch_new) #[num_ball,d] without activation
          

            h_out = self.out_w_encoder(h_ball) #[num_ball,2d]
            mean,mu = self.split_mean_mu(h_out)
            mu = mu.abs()
            return mean,mu

        else:  # for ODE
            # h_t [n_ball,d]
            h_out = self.out_w_ode(h_t)

        return h_out

    def rewrite_batch(self,batch, batch_y):
        assert (torch.sum(batch_y).item() == list(batch.size())[0])
        batch_new = torch.zeros_like(batch)
        group_num = 0
        current_index = 0
        for ball_time in batch_y:
            batch_new[current_index:current_index + ball_time] = group_num
            group_num += 1
            current_index += ball_time.item()

        return batch_new

    def attention_expand(self,attention_ball, batch,batch_y):
        '''

        :param attention_ball: [num_ball, d]
        :param batch: [num_nodes,d]
        :param batch_new: [num_ball,d]
        :return:
        '''
        node_size = batch.size()[0]
        dim = attention_ball.size()[1]
        new_attention = torch.ones(node_size, dim)
        if attention_ball.device != torch.device("cpu"):
            new_attention = new_attention.cuda(attention_ball.get_device())

        group_num = 0
        current_index = 0
        for ball_time in batch_y:
            new_attention[current_index:current_index+ball_time] = attention_ball[group_num]
            group_num +=1
            current_index += ball_time.item()

        return new_attention

    def split_mean_mu(self,h):
        last_dim = h.size()[-1] //2
        res = h[:,:last_dim], h[:,last_dim:]
        return res

class GNN_cgode(nn.Module):
    '''
    wrap up multiple layers
    '''
    def __init__(self, in_dim, n_hid,out_dim, n_heads, n_layers,args, dropout = 0.2, conv_name = 'GTrans', is_encoder = False):
        super(GNN_cgode, self).__init__()
        self.gcs = nn.ModuleList()
        self.in_dim = in_dim
        self.n_hid = n_hid
        self.out_dim = out_dim
        self.drop = nn.Dropout(dropout)
        self.is_encoder = is_encoder


        if is_encoder:
            # If encoder, adding 1.) sequence_W 2.)transform_W ( to 2*z_dim).
            self.sequence_w = nn.Linear(n_hid,n_hid) # for encoder
            self.hidden_to_z0 = nn.Sequential(
		        nn.Linear(n_hid, n_hid//2),
		        nn.Tanh(),
		        nn.Linear(n_hid//2, out_dim))
            self.adapt_w = nn.Linear(in_dim,n_hid)
            utils.init_network_weights(self.sequence_w)
            utils.init_network_weights(self.hidden_to_z0)
            utils.init_network_weights(self.adapt_w)
        else: # ODE GNN
            assert self.in_dim == self.n_hid

        # first layer is input layer
        for l in range(0,n_layers):
            self.gcs.append(GeneralConv_cgode(conv_name, self.n_hid, self.n_hid,  n_heads, dropout,args))

        if conv_name in  ['GTrans'] :
            self.temporal_net = TemporalEncoding(n_hid)  #// Encoder, needs positional encoding for sequence aggregation.

    def forward(self, x, edge_weight=None, edge_index=None, x_time=None, edge_time=None,batch= None, batch_y = None):  #aggregation part

        if not self.is_encoder: #Encoder initial input node feature
            h_t = self.drop(x)
        else:
            h_t = self.drop(F.gelu(self.adapt_w(x)))  #initial input for encoder


        for gc in self.gcs:
            h_t = gc(h_t, edge_index, edge_weight, x_time,edge_time)  #[num_nodes,d]

        ### Output
        if batch!= None:  ## for encoder
            batch_new = self.rewrite_batch(batch,batch_y) #group by balls

            h_t += self.temporal_net(x_time)
            attention_vector = F.gelu(
                self.sequence_w(global_mean_pool(h_t, batch_new)))  # [num_ball,d] ,graph vector with activation Relu
            attention_vector_expanded = self.attention_expand(attention_vector, batch, batch_y)  # [num_nodes,d]
            attention_nodes = torch.sigmoid(
                torch.squeeze(torch.bmm(torch.unsqueeze(attention_vector_expanded, 1), torch.unsqueeze(h_t, 2)))).view(
                -1, 1)  # [num_nodes]
            nodes_attention = attention_nodes * h_t  # [num_nodes,d]
            h_ball = global_mean_pool(nodes_attention, batch_new)  # [num_ball,d] without activation

            h_out = self.hidden_to_z0(h_ball) #[num_ball,2*z_dim] Must ganrantee NO 0 ENTRIES!
            mean,mu = self.split_mean_mu(h_out)
            mu = mu.abs()
            return mean,mu

        else:  # for ODE
            h_out = h_t
            return h_out

    def rewrite_batch(self,batch, batch_y):
        assert (torch.sum(batch_y).item() == list(batch.size())[0])
        batch_new = torch.zeros_like(batch)
        group_num = 0
        current_index = 0
        for ball_time in batch_y:
            batch_new[current_index:current_index + ball_time] = group_num
            group_num += 1
            current_index += ball_time.item()

        return batch_new

    def attention_expand(self,attention_ball, batch,batch_y):
        '''

        :param attention_ball: [num_ball, d]
        :param batch: [num_nodes,d]
        :param batch_new: [num_ball,d]
        :return:
        '''
        node_size = batch.size()[0]
        dim = attention_ball.size()[1]
        new_attention = torch.ones(node_size, dim)
        if attention_ball.device != torch.device("cpu"):
            new_attention = new_attention.cuda()

        group_num = 0
        current_index = 0
        for ball_time in batch_y:
            new_attention[current_index:current_index+ball_time] = attention_ball[group_num]
            group_num +=1
            current_index += ball_time.item()

        return new_attention

    def split_mean_mu(self,h):
        last_dim = h.size()[-1] //2
        res = h[:,:last_dim], h[:,last_dim:]
        return res

class Node_GCN(nn.Module):
    """Node ODE function."""

    def __init__(self, in_dims, out_dims, num_atoms,dropout=0.):
        super(Node_GCN, self).__init__()

        self.w_node = nn.Parameter(torch.FloatTensor(in_dims, out_dims), requires_grad=True) #[D,D]
        self.num_atoms = num_atoms
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dims,elementwise_affine = False)

        glorot(self.w_node)


    def forward(self, inputs, edges,z_0):

        '''
        :param inputs: [K*N,D] (node attributes, H)
        :param edges: [K,N*N], after normalize
        :param z_0: [K*N,D],
        :return:
        '''
        inputs = self.layer_norm(inputs)

        num_feature = inputs.shape[-1]

        edges = edges.view(-1,self.num_atoms,self.num_atoms) #[K,N,N]
        inputs_transform = torch.matmul(inputs,self.w_node) #[K*N,D]
        inputs_transform = inputs_transform.view(-1,self.num_atoms,num_feature) #[K,N,D]

        x_hidden = torch.bmm(edges,inputs_transform) #[K,N,D]
        x_hidden = x_hidden.view(-1,num_feature) #[K*N,D]

        x_new = F.gelu(x_hidden) - inputs + z_0

        return self.dropout(x_new)

def normalize_graph_asymmetric(edge_index,edge_weight, num_nodes):
    '''

    :param edge_index: [num_edges,2], torch.LongTensor
    :param edge_weight: [num_edge]
    :param num_nodes:
    :return:
    '''
    assert (not torch.isnan(edge_weight).any())

    row, col = edge_index[0],edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)  # [K*N]
    deg_inv_sqrt = deg.pow_(-1)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    if torch.isnan(deg_inv_sqrt).any():
        assert (torch.sum(deg == 0) == 0)
        assert (torch.sum(deg < 0) == 0)


    edge_weight_normalized = deg_inv_sqrt[row] * edge_weight  # [num_edge]

    # Reshape back
    assert (not torch.isnan(edge_weight_normalized).any())

    return edge_weight_normalized
class GTrans_cgode(MessagePassing):
    '''
    Multiply attention by edgeweight
    '''

    def __init__(self, n_heads=1,d_input=6, d_output=6,dropout = 0.1,**kwargs):
        super(GTrans_cgode, self).__init__(aggr='add', **kwargs)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

        self.d_input = d_input
        self.d_k = d_output//n_heads
        self.d_q = d_output//n_heads
        self.d_e = d_output//n_heads
        self.d_sqrt = math.sqrt(d_output//n_heads)


        #Attention Layer Initialization
        self.w_k_list = nn.ModuleList([nn.Linear(self.d_input, self.d_k, bias=True) for _ in range(self.n_heads)])
        self.w_q_list = nn.ModuleList([nn.Linear(self.d_input, self.d_q, bias=True) for _ in range(self.n_heads)])
        self.w_v_list = nn.ModuleList([nn.Linear(self.d_input, self.d_e, bias=True) for _ in range(self.n_heads)])

        #initiallization
        utils.init_network_weights(self.w_k_list)
        utils.init_network_weights(self.w_q_list)
        utils.init_network_weights(self.w_v_list)


        #Temporal Layer
        self.temporal_net = TemporalEncoding(d_input)

        #Normalization
        self.layer_norm = nn.LayerNorm(d_input,elementwise_affine = False)

    def forward(self, x, edge_index, edge_weight,time_nodes,edge_time):
        '''

        :param x:
        :param edge_index:
        :param edge_wight: edge_weight
        :param time_nodes:
        :param edge_time: edge_time_attr
        :return:
        '''

        residual = x
        x = self.layer_norm(x)

        # Edge normalization if using multiplication
        print('edge_index',edge_index)
        print('edge_index shape',edge_index.shape)

        edge_weight = torch.ones([edge_index.shape[1]]).to(edge_index.get_device())
        print(' edge_weight shape', edge_weight.shape)
        edge_weight = normalize_graph_asymmetric(edge_index,edge_weight,time_nodes.shape[0])
        #assert (torch.sum(edge_weight<0)==0) and (torch.sum(edge_weight>1) == 0)

        return self.propagate(edge_index, x=x, edges_weight=edge_weight, edge_time=edge_time, residual=residual)

        #return self.propagate(edge_index, x=x, edges_weight=edge_weight, edge_same=edge_time, residual=residual) # from GTrans above

    def message(self, x_j,x_i,edge_index_i, edges_weight,edge_time):
        '''

           :param x_j: [num_edge, d] sender
           :param x_i: [num_edge,d]  receiver
           :param edge_index_i:  receiver node list [num_edge]
           :param edges_temporal: [num_edge,d]
           :param edge_time: [num_edge,d]
           :return:
        '''


        messages = []
        for i in range(self.n_heads):
            k_linear = self.w_k_list[i]
            q_linear = self.w_q_list[i]
            v_linear = self.w_v_list[i]

            edge_temporal_vector = self.temporal_net(edge_time) #[num_edge,d]
            edges_weight = edges_weight.view(-1, 1)
            x_j_transfer = x_j + edge_temporal_vector

            attention = self.each_head_attention(x_j_transfer,k_linear,q_linear,x_i) #[N_edge,1]
            attention = torch.div(attention,self.d_sqrt)

            # Need to multiply by original edge weight
            attention = attention * edges_weight

            attention_norm = softmax(attention,edge_index_i) #[N_neighbors_,1]
            sender = v_linear(x_j_transfer)

            message  = attention_norm * sender #[N_nodes,d]
            messages.append(message)

        message_all_head  = torch.cat(messages,1) #[N_nodes, k*d] ,assuming K head

        return message_all_head

    def each_head_attention(self,x_j_transfer,w_k,w_q,x_i):
        '''

        :param x_j_transfer: sender [N_edge,d]
        :param w_k:
        :param w_q:
        :param x_i: receiver
        :return:
        '''

        # Receiver #[num_edge,d*heads]
        x_i = w_q(x_i)
        # Sender
        sender = w_k(x_j_transfer)
        # Calculate attention score
        attention = torch.bmm(torch.unsqueeze(sender,1),torch.unsqueeze(x_i,2)) #[N,1]

        return torch.squeeze(attention,1)


    def update(self, aggr_out,residual):
        x_new = residual + F.gelu(aggr_out)

        return self.dropout(x_new)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)