
import torch.nn.functional as F

from torch.autograd import Variable
from longformer.sliding_chunks import sliding_chunks_matmul_qk,sliding_chunks_matmul_pv,sliding_chunks_no_overlap_matmul_qk,sliding_chunks_no_overlap_matmul_pv
from longformer.diagonaled_mm_tvm import diagonaled_mm as diagonaled_mm_tvm, mask_invalid_locations
from transformers.models.longformer.modeling_longformer import *
import numpy as np


class FixedPositionalEncoding(nn.Module):
    def __init__(self, dim_mid, max_length=None):
        super(FixedPositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(0.1)

        pe = torch.zeros(max_length, dim_mid)

        position = torch.arange(0, max_length).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, dim_mid, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / dim_mid)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(0)], requires_grad=True)

        return x


class VideoEmbedding(nn.Module):
    def __init__(self,dim_in,dim_mid):
        super(VideoEmbedding, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(dim_in, dim_mid),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
            nn.LayerNorm(dim_mid)
        )
    def forward(self,x):
        x = self.embedding(x)
        return x
    
def get_attn_pad_mask_video_de(seq_q, seq_k):
    len_q, emb_dim = seq_q.size(0),seq_q.size(1)
    len_k, emb_dim = seq_k.size(0),seq_q.size(1)
    pad_attn_mask = torch.zeros((1, len_q, len_k), dtype=torch.bool)

    return pad_attn_mask
def get_attn_pad_mask_video(T, seq_q, seq_k,max_length,global_idx):


    mask = torch.cat((torch.ones(T), torch.zeros(max_length - T)), dim=-1).cuda()
    global_idx = torch.tensor(global_idx//15).cuda()
    mask[global_idx]=-1
    mask = mask.unsqueeze(0)
    return mask


def get_attn_subsequent_mask(seq):
    seq = seq.unsqueeze(0)
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask

def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  
    return torch.FloatTensor(sinusoid_table)
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,dim_mid=64, heads=1,dff=2048):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = dim_mid
        self.d_ff = dff
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(self.d_ff, self.d_model, bias=False)
        )
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, inputs):
        residual = inputs 
        output = self.fc(inputs)
        output = output + residual
        output = self.layer_norm(output)
        return output

class LocalGlobalScaledDotProductAttention(nn.Module):
    def __init__(self, window_size=32, step_size=8, dim_mid=64, heads=8,attention_mode=None):
        super(LocalGlobalScaledDotProductAttention, self).__init__()
        self.attention_window = window_size
        self.dim_mid = dim_mid
        self.d_k = dim_mid // heads
        self.d_v = dim_mid // heads
        self.num_heads = heads
        self.head_dim =dim_mid // heads
        self.dropout = 0.1
        self.attention_dilation = 1
        self.attention_mode = attention_mode
        self.W_Q = nn.Linear(dim_mid, self.d_k * self.num_heads).cuda()
        self.W_K = nn.Linear(dim_mid, self.d_k * self.num_heads).cuda()
        self.W_V = nn.Linear(dim_mid, self.d_v * self.num_heads).cuda()

        self.W_Q_global = nn.Linear(dim_mid, self.d_k * self.num_heads).cuda()
        self.W_K_global = nn.Linear(dim_mid, self.d_k * self.num_heads).cuda()
        self.W_V_global = nn.Linear(dim_mid, self.d_v * self.num_heads).cuda()




    def forward(self, hidden_states,q,k,v,attn_mask):
        output_attentions = True



        attention_mask = attn_mask.clone().detach()
        pad_num =(attention_mask == 0).sum().item()
        # global_num  = (attention_mask == -1).sum().item()
        T = int(attention_mask.size(1))-pad_num

        if attention_mask is not None:
            # attention_mask = attention_mask.squeeze(dim=2).squeeze(dim=1)
            key_padding_mask = attention_mask < 0
            extra_attention_mask = attention_mask > 0
            remove_from_windowed_attention_mask = attention_mask != 0

            num_extra_indices_per_batch = extra_attention_mask.long().sum(dim=1)
            max_num_extra_indices_per_batch = num_extra_indices_per_batch.max()
            if max_num_extra_indices_per_batch <= 0:
                extra_attention_mask = None
            else:
                # To support the case of variable number of global attention in the rows of a batch,
                # we use the following three selection masks to select global attention embeddings
                # in a 3d tensor and pad it to `max_num_extra_indices_per_batch`
                # 1) selecting embeddings that correspond to global attention
                extra_attention_mask_nonzeros = extra_attention_mask.nonzero(as_tuple=True)
                zero_to_max_range = torch.arange(0, max_num_extra_indices_per_batch,
                                                 device=num_extra_indices_per_batch.device)
                # mask indicating which values are actually going to be padding
                selection_padding_mask = zero_to_max_range < num_extra_indices_per_batch.unsqueeze(dim=-1)
                # 2) location of the non-padding values in the selected global attention
                selection_padding_mask_nonzeros = selection_padding_mask.nonzero(as_tuple=True)
                # 3) location of the padding values in the selected global attention
                selection_padding_mask_zeros = (selection_padding_mask == 0).nonzero(as_tuple=True)
        else:
            remove_from_windowed_attention_mask = None
            extra_attention_mask = None
            key_padding_mask = None

        hidden_states = hidden_states.transpose(0, 1)
        seq_len, bsz, embed_dim = hidden_states.size()
        assert embed_dim == self.dim_mid


        q = self.W_Q(q)
        k = self.W_K(k)
        v = self.W_V(v)
        q /= math.sqrt(self.head_dim)

        q = q.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)


        if self.attention_mode == 'tvm':
            q = q.float().contiguous()
            k = k.float().contiguous()
            attn_weights = diagonaled_mm_tvm(q, k, self.attention_window, self.attention_dilation, False, 0, False)
        elif self.attention_mode == "sliding_chunks":
            attn_weights = sliding_chunks_matmul_qk(q, k, self.attention_window, padding_value=0)
        elif self.attention_mode == "sliding_chunks_no_overlap":
            attn_weights = sliding_chunks_no_overlap_matmul_qk(q, k, self.attention_window, padding_value=0)
        else:
            raise False


        mask_invalid_locations(attn_weights, self.attention_window, self.attention_dilation, False)
        if remove_from_windowed_attention_mask is not None:
            # This implementation is fast and takes very little memory because num_heads x hidden_size = 1
            # from (bsz x seq_len) to (bsz x seq_len x num_heads x hidden_size)
            remove_from_windowed_attention_mask = remove_from_windowed_attention_mask.unsqueeze(dim=-1).unsqueeze(
                dim=-1)
            # cast to float/half then replace 1's with -inf
            float_mask = remove_from_windowed_attention_mask.type_as(q).masked_fill(remove_from_windowed_attention_mask,
                                                                                    -10000.0)
            repeat_size = 1 if isinstance(self.attention_dilation, int) else len(self.attention_dilation)
            float_mask = float_mask.repeat(1, 1, repeat_size, 1)
            ones = float_mask.new_ones(size=float_mask.size())  # tensor of ones
            # diagonal mask with zeros everywhere and -inf inplace of padding
            if self.attention_mode == 'tvm':
                d_mask = diagonaled_mm_tvm(ones, float_mask, self.attention_window, self.attention_dilation, False, 0,
                                           False)
            elif self.attention_mode == "sliding_chunks":
                d_mask = sliding_chunks_matmul_qk(ones, float_mask, self.attention_window, padding_value=0)
            elif self.attention_mode == "sliding_chunks_no_overlap":
                d_mask = sliding_chunks_no_overlap_matmul_qk(ones, float_mask, self.attention_window, padding_value=0)

            attn_weights += d_mask


        assert list(attn_weights.size())[:3] == [bsz, seq_len, self.num_heads]
        assert attn_weights.size(dim=3) in [self.attention_window * 2 + 1, self.attention_window * 3]

        # the extra attention
        if extra_attention_mask is not None:
            selected_k = k.new_zeros(bsz, max_num_extra_indices_per_batch, self.num_heads, self.head_dim)
            selected_k[selection_padding_mask_nonzeros] = k[extra_attention_mask_nonzeros]
            # (bsz, seq_len, num_heads, max_num_extra_indices_per_batch)
            selected_attn_weights = torch.einsum('blhd,bshd->blhs', (q, selected_k))
            selected_attn_weights[selection_padding_mask_zeros[0], :, :, selection_padding_mask_zeros[1]] = -10000
            # concat to attn_weights
            # (bsz, seq_len, num_heads, extra attention count + 2*window+1)
            attn_weights = torch.cat((selected_attn_weights, attn_weights), dim=-1)



        attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)  # use fp32 for numerical stability
        if key_padding_mask is not None:
            # softmax sometimes inserts NaN if all positions are masked, replace them with 0
            attn_weights_float = torch.masked_fill(attn_weights_float, key_padding_mask.unsqueeze(-1).unsqueeze(-1),
                                                   0.0)
            #attn_weights_float = torch.masked_fill(attn_weights_float, key_padding_mask[0][0].unsqueeze(0).unsqueeze(-1).unsqueeze(-1),0.0)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)
        v = v.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        attn = 0
        if extra_attention_mask is not None:
            selected_attn_probs = attn_probs.narrow(-1, 0, max_num_extra_indices_per_batch)
            selected_v = v.new_zeros(bsz, max_num_extra_indices_per_batch, self.num_heads, self.head_dim)
            selected_v[selection_padding_mask_nonzeros] = v[extra_attention_mask_nonzeros]
            # use `matmul` because `einsum` crashes sometimes with fp16
            # attn = torch.einsum('blhs,bshd->blhd', (selected_attn_probs, selected_v))
            attn = torch.matmul(selected_attn_probs.transpose(1, 2),
                                selected_v.transpose(1, 2).type_as(selected_attn_probs)).transpose(1, 2)
            attn_probs = attn_probs.narrow(-1, max_num_extra_indices_per_batch,
                                           attn_probs.size(-1) - max_num_extra_indices_per_batch).contiguous()

        if self.attention_mode == 'tvm':
            v = v.float().contiguous()
            attn += diagonaled_mm_tvm(attn_probs, v, self.attention_window, self.attention_dilation, True, 0, False)
        elif self.attention_mode == "sliding_chunks":
            attn += sliding_chunks_matmul_pv(attn_probs, v, self.attention_window)
        elif self.attention_mode == "sliding_chunks_no_overlap":
            attn += sliding_chunks_no_overlap_matmul_pv(attn_probs, v, self.attention_window)
        else:
            raise False

        attn = attn.type_as(hidden_states)
        assert list(attn.size()) == [bsz, seq_len, self.num_heads, self.head_dim]
        attn = attn.transpose(0, 1).reshape(seq_len, bsz, embed_dim).contiguous()

        # For this case, we'll just recompute the attention for these indices
        # and overwrite the attn tensor.
        if extra_attention_mask is not None:
            selected_hidden_states = hidden_states.new_zeros(max_num_extra_indices_per_batch, bsz, embed_dim)
            selected_hidden_states[selection_padding_mask_nonzeros[::-1]] = hidden_states[
                extra_attention_mask_nonzeros[::-1]]

            q = self.W_Q_global(selected_hidden_states)
            k = self.W_K_global(hidden_states)
            v = self.W_V_global(hidden_states)
            q /= math.sqrt(self.head_dim)

            q = q.contiguous().view(max_num_extra_indices_per_batch, bsz * self.num_heads, self.head_dim).transpose(0,
                                                                                                                    1)  # (bsz*self.num_heads, max_num_extra_indices_per_batch, head_dim)
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0,
                                                                                       1)  # bsz * self.num_heads, seq_len, head_dim)
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0,
                                                                                       1)  # bsz * self.num_heads, seq_len, head_dim)
            attn_weights = torch.bmm(q, k.transpose(1, 2))
            assert list(attn_weights.size()) == [bsz * self.num_heads, max_num_extra_indices_per_batch, seq_len]


            attn_weights = attn_weights.view(bsz, self.num_heads, max_num_extra_indices_per_batch, seq_len)
            attn_weights_show4 = attn_weights.clone().detach().cpu().numpy()
            attn_weights[selection_padding_mask_zeros[0], :, selection_padding_mask_zeros[1], :] = -10000.0
            if key_padding_mask is not None:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    -10000.0,
                )
            attn_weights = attn_weights.view(bsz * self.num_heads, max_num_extra_indices_per_batch, seq_len)
            attn_weights_float = F.softmax(attn_weights, dim=-1,
                                           dtype=torch.float32)  # use fp32 for numerical stability
            attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)
            selected_attn = torch.bmm(attn_probs, v)
            assert list(selected_attn.size()) == [bsz * self.num_heads, max_num_extra_indices_per_batch, self.head_dim]

            selected_attn_4d = selected_attn.view(bsz, self.num_heads, max_num_extra_indices_per_batch, self.head_dim)

            attn_weights_show5 = attn_weights.clone().detach().cpu().numpy()

            nonzero_selected_attn = selected_attn_4d[selection_padding_mask_nonzeros[0], :,
                                    selection_padding_mask_nonzeros[1]]
            attn[extra_attention_mask_nonzeros[::-1]] = nonzero_selected_attn.view(
                len(selection_padding_mask_nonzeros[0]), -1).type_as(hidden_states)

        context_layer = attn.transpose(0, 1)
        if output_attentions:
            if extra_attention_mask is not None:

                attn_weights = attn_weights.view(bsz, self.num_heads, max_num_extra_indices_per_batch, seq_len)
            else:

                attn_weights = attn_weights.permute(0, 2, 1, 3)

        attn_weights_show = attn_weights.clone().detach().cpu().numpy()  

        return context_layer, attn_weights_show

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim_mid=64, heads=8):
        super(ScaledDotProductAttention, self).__init__()
        self.dim_mid = dim_mid

        self.d_k = dim_mid // heads
    def forward(self, Q, K, V, attn_mask):

        attn_mask = attn_mask.to(torch.bool)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores.masked_fill_(attn_mask.cuda(), -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

#
class LocalGlobalMultiHeadAttention(nn.Module):
    def __init__(self, dim_mid=64, heads=8, window_size=10, stride=5,attention_mode = None):
        super(LocalGlobalMultiHeadAttention, self).__init__()
        self.dim_mid = dim_mid
        self.d_k = dim_mid // heads
        self.d_v = dim_mid // heads
        self.n_heads = heads
        self.window_size = window_size
        self.stride = stride
        self.W_Q = nn.Linear(dim_mid, self.d_k * self.n_heads)
        self.W_K = nn.Linear(dim_mid, self.d_k * self.n_heads)
        self.W_V = nn.Linear(dim_mid, self.d_v * self.n_heads)
        self.linear = nn.Linear(self.n_heads * self.d_v, dim_mid)
        self.layer_norm = nn.LayerNorm(dim_mid)
        self.attention_mode = attention_mode

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size,seq_len = Q, Q.size(0),Q.size(1)
        hidden_states = Q

        q_s = Q
        k_s = K
        v_s = V

        attention_mask = attn_mask.clone().detach()
        pad_num =(attention_mask == 0).sum().item()

        T = int(attention_mask.size(1))-pad_num
        attention_mask[0, :pad_num] = torch.where(attention_mask[0, :pad_num] == 1,torch.tensor(0),attention_mask[0, :pad_num])
        attention_mask[0, :pad_num] = torch.where(attention_mask[0, :pad_num] == -1,torch.tensor(1),attention_mask[0, :pad_num])
        attention_mask[0, pad_num:] = -1

        context, attn = LocalGlobalScaledDotProductAttention(window_size=self.window_size,
                                                       step_size=self.stride,
                                                       dim_mid=self.dim_mid,
                                                       heads = self.n_heads,
                                                       attention_mode=self.attention_mode)(hidden_states,q_s, k_s, v_s,attention_mask)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            self.n_heads * self.d_v)
        output = self.linear(context)
        return self.layer_norm(output + residual), attn


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_mid=64, heads=8):
        super(MultiHeadAttention, self).__init__()
        self.dim_mid = dim_mid

        self.d_k = dim_mid // heads
        self.d_v = dim_mid // heads
        self.n_heads = heads


        self.W_Q = nn.Linear(dim_mid, self.d_k * self.n_heads)
        self.W_K = nn.Linear(dim_mid, self.d_k * self.n_heads)
        self.W_V = nn.Linear(dim_mid, self.d_v * self.n_heads)
        self.linear = nn.Linear(self.n_heads * self.d_v, dim_mid)
        self.layer_norm = nn.LayerNorm(dim_mid)

    def forward(self, Q, K, V, attn_mask):

        residual, batch_size,seq_len = Q, Q.size(0), Q.size(1)

        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]


        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)



        context = context.squeeze(0).transpose(0, 1).contiguous().view(seq_len,-1, self.n_heads * self.d_v)
        # context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context).transpose(0, 1)

        return self.layer_norm(output + residual), attn,context
class SparseEncoderLayer(nn.Module):
    def __init__(self,window_size, stride,dim_mid,attention_mode,dff):
        super(SparseEncoderLayer, self).__init__()
        self.enc_self_attn = LocalGlobalMultiHeadAttention(dim_mid=dim_mid,window_size=window_size,stride= stride,attention_mode = attention_mode)
        self.pos_ffn = PoswiseFeedForwardNet(dim_mid=dim_mid,dff=dff)
    def forward(self, enc_inputs, enc_self_attn_mask):

        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)

        return enc_outputs, attn

class DecoderLayer(nn.Module):
    def __init__(self,window_size, stride,dim_mid,heads,dff):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(dim_mid=dim_mid,heads=heads)
        self.dec_enc_attn = MultiHeadAttention(dim_mid=dim_mid,heads=heads)
        self.pos_ffn = PoswiseFeedForwardNet(dim_mid=dim_mid,dff=dff)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn, _  = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn, context = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn, context
class Encoder(nn.Module):
    def __init__(self,dim_in,dim_mid,enlayers,length,window_size,stride,heads,attention_mode,dff):
        super(Encoder, self).__init__()

        self.src_emb = VideoEmbedding(dim_in,dim_mid)

        self.max_length = length
        self.nlayers = enlayers
        self.position_embeddings = FixedPositionalEncoding(dim_mid, max_length=length)

        self.local_layers =nn.ModuleList([SparseEncoderLayer(window_size,stride,dim_mid,attention_mode,dff) for _ in range(self.nlayers)])


    def forward(self, T,enc_inputs,global_idx):


        enc_outputs = enc_inputs + self.position_embeddings(enc_inputs)
        enc_self_attn_mask_lga = get_attn_pad_mask_video(T,enc_inputs, enc_inputs,self.max_length,global_idx)

        enc_self_attns = []

        for layer_local in self.local_layers:
            enc_outputs, enc_self_attn  = layer_local(enc_outputs, enc_self_attn_mask_lga)
            enc_self_attns.append(enc_self_attn)


        return enc_outputs.squeeze(0), enc_self_attns
class Decoder(nn.Module):
    def __init__(self,dim_mid,delayers,length,window_size,stride,heads,dff):
        super(Decoder, self).__init__()
        self.d_model = dim_mid
        self.delayers = delayers

        self.position_embeddings = FixedPositionalEncoding(dim_mid, max_length=length)
        self.layers = nn.ModuleList([DecoderLayer(window_size,stride,dim_mid,heads,dff) for _ in range(self.delayers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs): 
        dec_outputs = dec_inputs + self.position_embeddings(dec_inputs)
        dec_self_attn_pad_mask = get_attn_pad_mask_video_de(dec_inputs, dec_inputs).cuda()
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs).cuda()
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        dec_enc_attn_mask = get_attn_pad_mask_video_de(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []

        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn, context = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)


        return dec_outputs, dec_self_attns, dec_enc_attns   


class Transformer(nn.Module):

    def __init__(self,T, dim_in, heads,enlayers,delayers,dim_mid,length, window_size, stride,
                 attention_mode,dff):
        super(Transformer,self).__init__()

        self.dim_mid = dim_mid
        self.d_k = dim_mid / heads
        self.d_v = dim_mid / heads
        self.n_heads = heads
        self.length = length


        self.embedding = VideoEmbedding(dim_in,dim_mid)
        self.encoder = Encoder(dim_in,dim_mid,enlayers,length,window_size,stride,heads,attention_mode,dff)
        self.decoder = Decoder(dim_mid,delayers,length,window_size,stride,heads,dff)

        self.projection = nn.Linear(self.dim_mid, self.length, bias=False)


    def forward(self,x,target,global_idx):

        T = x.shape[1]
        pad = torch.zeros((1,self.length - T,1024)).cuda()
        x = torch.cat((x,pad), dim=1)

        enc_inputs = self.embedding(x)
        dec_inputs = self.embedding(target)


        enc_outputs, enc_self_attns= self.encoder(T,enc_inputs,global_idx)
        
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)


        dec_logits = self.projection(dec_outputs)
        dec_logits = dec_logits.view(-1, dec_logits.size(-1))
        dec_logits = dec_logits[:,0:T]
        generator_dec_output = F.softmax(dec_logits, dim=-1)

        return generator_dec_output, enc_self_attns, dec_self_attns, dec_enc_attns
