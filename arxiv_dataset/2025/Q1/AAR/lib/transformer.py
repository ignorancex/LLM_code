import torch
import torch.nn as nn
import copy
from einops import reduce

class TransformerEncoderLayer(nn.Module):

    def __init__(self, embed_dim=1936, nhead=4, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, input_key_padding_mask):
        # local attention
        src2, local_attention_weights = self.self_attn(src, src, src, key_padding_mask=input_key_padding_mask)

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(nn.functional.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src, local_attention_weights


class TransformerDecoderLayer(nn.Module):

    def __init__(self, embed_dim=1936, nhead=4, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        self.multihead2 = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)


        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, query=None, key=None, value=None, key_padding_mask=None):

        tgt2, global_attention_weights = self.multihead2(query=query, key=key,
                                                         value=value, key_padding_mask=key_padding_mask)

        tgt = query + self.dropout2(tgt2)
        tgt = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(nn.functional.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt, global_attention_weights


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        # get no. of encoder layers by a simple copy op
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, input, input_key_padding_mask):
        output = input
        weights = torch.zeros([self.num_layers, output.shape[1], output.shape[0], output.shape[0]]).to(output.device)

        for i, layer in enumerate(self.layers):
            output, local_attention_weights = layer(output, input_key_padding_mask)
            weights[i] = local_attention_weights
        if self.num_layers > 0:
            return output, weights
        else:
            return output, None


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, embed_dim):
        super().__init__()
        # get no. of decoder layers by a simple copy op
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers


    # def forward(self, global_input, input_key_padding_mask, position_embed):
    def forward(self, query=None, key=None, value=None, key_padding_mask=None):

        output = query
        weights = torch.zeros([self.num_layers, output.shape[1], output.shape[0], output.shape[0]]).to(output.device)

        for i, layer in enumerate(self.layers):
            output, global_attention_weights = layer(output, key, value, key_padding_mask)
            weights[i] = global_attention_weights

        if self.num_layers>0:
            return output, weights
        else:
            return output, None


class transformer(nn.Module):
    ''' Spatial Temporal Transformer
        local_attention: spatial encoder
        global_attention: temporal decoder
        position_embedding: frame encoding (window_size*dim)
        mode: both--use the features from both frames in the window
              latter--use the features from the latter frame in the window
    '''
    # don't forget i changed 1936 -> 1536
    def __init__(self, enc_layer_num=1, dec_layer_num=3, embed_dim=1936, nhead=8, dim_feedforward=2048,
                 dropout=0.1, mode=None, cross_attention=True):
        super(transformer, self).__init__()
        self.mode = mode
        self.cross_attention = cross_attention

        if self.cross_attention:
            embed_dim = 1424

        encoder_layer = TransformerEncoderLayer(embed_dim=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward,
                                                dropout=dropout)
        self.local_attention = TransformerEncoder(encoder_layer, enc_layer_num)

        decoder_layer = TransformerDecoderLayer(embed_dim=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward,
                                                dropout=dropout)

        self.global_attention = TransformerDecoder(decoder_layer, dec_layer_num, embed_dim)

        if self.cross_attention:
            self.linear_projection = nn.Linear(512, 1424)

    def forward(self, features, im_idx, pool_type, split=False):

        if not self.cross_attention:
            rel_idx = torch.arange(im_idx.shape[0])

            l = torch.sum(im_idx == torch.mode(im_idx)[0])  # the highest box number in the single frame - to allow similar length inputs to the transformer
            b = int(im_idx[-1] + 1)
            rel_input = torch.zeros([l, b, features.shape[1]]).to(features.device) # size similar to input of transformer
            masks = torch.zeros([b, l], dtype=torch.uint8).to(features.device) # byte tensor

            for i in range(b):
                # put the sub-obj relations depending on the no. of it for each frame
                rel_input[:torch.sum(im_idx == i), i, :] = features[im_idx == i]
                # if the max is 3 bbox, but the current frame only has 2 bbox, mask out the third
                # as masks is a byte tensor, transformer will not attend to non-zero values.
                # https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer.forward
                masks[i, torch.sum(im_idx == i):] = 1

            local_output, local_attention_weights = self.local_attention(rel_input, masks)
            # decoder
            global_output, global_attention_weights = self.global_attention(query=local_output, key=local_output, value=local_output, key_padding_mask=masks)

            if not split:

                global_output = (global_output.permute(1, 0, 2)).contiguous().view(-1, features.shape[1])[masks.view(-1) == 0]
                output = torch.zeros([b, global_output.shape[1]]).to(global_output.device)

                for i in range(b):
                    if pool_type == 'max':
                        output[i, :] = reduce(global_output[im_idx == i], 'n d -> d', 'max') # n represents no. of features
                    elif pool_type == 'avg':
                        output[i, :] = reduce(global_output[im_idx == i], 'n d -> d', 'mean')

                return output, global_attention_weights, local_attention_weights
            else:
                global_output = (global_output.permute(1, 0, 2)).contiguous().view(-1, features.shape[1])[masks.view(-1) == 0]
                return global_output, global_attention_weights, local_attention_weights

        else:
            # print("Using cross-attention")

            # decoder input
            decoder_query = features[0]
            # encoder self-attention
            encoder_query = features[1] 
            encoder_key = features[1]
            encoder_value = features[1]

            rel_idx = torch.arange(im_idx.shape[0])

            l = torch.sum(im_idx == torch.mode(im_idx)[0])  # the highest box number in the single frame - to allow similar length inputs to the transformer
            b = int(im_idx[-1] + 1)
            rel_input = torch.zeros([l, b, encoder_query.shape[1]]).to(encoder_query.device) # size similar to input of transformer
            masks = torch.zeros([b, l], dtype=torch.uint8).to(encoder_query.device) # byte tensor

            for i in range(b):
                # put the sub-obj relations depending on the no. of it for each frame
                rel_input[:torch.sum(im_idx == i), i, :] = encoder_query[im_idx == i]
                # if the max is 3 bbox, but the current frame only has 2 bbox, mask out the third
                # as masks is a byte tensor, transformer will not attend to non-zero values.
                # https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer.forward
                masks[i, torch.sum(im_idx == i):] = 1

            local_output, local_attention_weights = self.local_attention(rel_input, masks)

            proj_decoder_query = self.linear_projection(decoder_query)

            decoder_input = torch.zeros([l, b, proj_decoder_query.shape[1]]).to(proj_decoder_query.device) # size similar to input of transformer

            for i in range(b):
                # put the sub-obj relations depending on the no. of it for each frame
                decoder_input[:torch.sum(im_idx == i), i, :] = proj_decoder_query[im_idx == i]
                # if the max is 3 bbox, but the current frame only has 2 bbox, mask out the third
                # as masks is a byte tensor, transformer will not attend to non-zero values.
                # https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer.forward
                # masks[i, torch.sum(im_idx == i):] = 1

            global_output, global_attention_weights = self.global_attention(query=decoder_input, key=local_output, value=local_output, key_padding_mask=masks)

            if not split:

                global_output = (global_output.permute(1, 0, 2)).contiguous().view(-1, encoder_query.shape[1])[masks.view(-1) == 0]
                output = torch.zeros([b, global_output.shape[1]]).to(global_output.device)

                for i in range(b):
                    if pool_type == 'max':
                        output[i, :] = reduce(global_output[im_idx == i], 'n d -> d', 'max') # n represents no. of features
                    elif pool_type == 'avg':
                        output[i, :] = reduce(global_output[im_idx == i], 'n d -> d', 'mean')

                return output, global_attention_weights, local_attention_weights
            else:
                global_output = (global_output.permute(1, 0, 2)).contiguous().view(-1, encoder_query.shape[1])[masks.view(-1) == 0]
                return global_output, global_attention_weights, local_attention_weights

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


