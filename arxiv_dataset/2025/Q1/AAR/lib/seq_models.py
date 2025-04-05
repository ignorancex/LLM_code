import torch
import torch.nn as nn

class GRUDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.1):
        super(GRUDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if num_layers >= 2:
            self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout)
        else:
            self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.fc = nn.Sequential(nn.Linear(hidden_size, 1024), # 1936 - 1024
                                nn.ReLU(),
                                nn.Dropout(),
                                nn.Linear(1024, num_classes))

    def forward(self, input, hidden):
        # action_distribution # seq_length, bs, input_size
        outputs, hidden = self.gru(input, hidden)
        outputs = self.fc(outputs)
        return outputs, hidden
    
    def init_hidden(self, encoder_features):
        # Convert the encoder features into a tensor with the desired shape
        return encoder_features.unsqueeze(0).repeat(self.num_layers, 1, 1) # num_layers, bs, hidden_size
        # return hidden


class SeqTransformerDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_mlp_layers, num_heads, num_classes, dropout=0.1):
        super(SeqTransformerDecoder, self).__init__()

        # Create a single Transformer Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(input_size, num_heads, dim_feedforward=hidden_size, dropout=0.1)

        # Stack multiple Transformer Decoder Layers to form the full Decoder
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # Define the output linear layer
        if num_mlp_layers == 2:
            self.linear = nn.Sequential(nn.Linear(hidden_size, 1024), # 1936 - 1024
                                        nn.ReLU(),
                                        nn.Dropout(),
                                        nn.Linear(1024, num_classes))
        elif num_mlp_layers == 1:
            self.linear = nn.Linear(hidden_size, num_classes)


    def forward(self, entry, conf):
        im_idx = entry["im_idx"]
        tgt = entry["relational_feats"] # query - pooled features

        if tgt.ndim != 3:
            tgt = tgt.unsqueeze(0)

        if conf.model_type == 'RBP' or conf.model_type == 'BiGED':
            # for RBP and BiGED models
            memory, memory_key_padding_mask = entry["unpooled_relational_feats"]
        else:
            # for the rest of the models
            features = entry["unpooled_relational_feats"] # memory - key-value
            rel_idx = torch.arange(im_idx.shape[0])

            l = torch.sum(im_idx == torch.mode(im_idx)[0])  # the highest box number in the single frame
            b = int(im_idx[-1] + 1)

            memory = torch.zeros([l, b, features.shape[1]]).to(features.device)

            memory_key_padding_mask = torch.zeros([b, l], dtype=torch.uint8).to(features.device)

            for i in range(b):
                memory[:torch.sum(im_idx == i), i, :] = features[im_idx == i]
                memory_key_padding_mask[i, torch.sum(im_idx == i):] = 1


        # Pass the input sequence (features) and the encoder output (memory) through the Transformer Decoder
        decoder_output = self.decoder(tgt, memory, memory_key_padding_mask=memory_key_padding_mask)

        # Apply a linear layer to the decoder output to generate the final output sequence
        action_scores = self.linear(decoder_output)

        return decoder_output, action_scores