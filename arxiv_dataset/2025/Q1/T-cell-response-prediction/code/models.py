import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TcellMLP(nn.Module):
    def __init__(self, params, data):
        super().__init__()
        input_dim = data.get_input_dimension()
        self.params = params
        self.dropout = nn.Dropout(p=params['dropout'])
        self.linear_1 = nn.Linear(input_dim, params['hidden_dim'])
        self.linear = nn.Linear(params['hidden_dim'], 1)

    def forward(self, batch):
        x = batch[self.params['feature_selector']]
        h = F.relu(self.linear_1(self.dropout(x)))
        prediction = self.linear(self.dropout(h))
        return prediction, prediction, h


class TcellLogit(nn.Module):
    def __init__(self, params, data):
        super().__init__()
        input_dim = data.get_input_dimension()
        self.params = params
        self.dropout = nn.Dropout(p=params['dropout'])
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, batch):
        x = batch[self.params['feature_selector']]
        prediction = self.linear(self.dropout(x))
        return prediction


class MHC_presentation(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.dropout = nn.Dropout(p=params['dropout'])
        self.scaling = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, batch):
        x = batch['mhc_presentation']
        h = torch.sigmoid(self.scaling * x + self.bias)
        return h


class Adversary(nn.Module):
    def __init__(self, params, input_dim, num_classes, baseline_pred_pos, baseline_pred_neg):
        super().__init__()
        self.params = params
        self.dropout = nn.Dropout(p=params['dropout'])
        if params['adv_tcell_label_input']:
            input_dim += 1
        self.linear_input = nn.Linear(input_dim, params['adversary_hidden_dim'])
        # self.linear_middle = nn.Linear(params['adversary_hidden_dim'], params['adversary_hidden_dim'])
        self.linear = nn.Linear(params['adversary_hidden_dim'], num_classes)
        # self.batch_norm = nn.BatchNorm1d(params['adversary_hidden_dim'])
        # self.layer_norm = nn.LayerNorm(params['adversary_hidden_dim'])
        self.baseline_pred_pos = torch.log(0.00001 + baseline_pred_pos/(1-baseline_pred_pos+0.00001))  # torch.log(baseline_pred_pos)
        self.baseline_pred_neg = torch.log(0.00001 + baseline_pred_neg/(1-baseline_pred_neg+0.00001))  # torch.log(baseline_pred_neg)
        self.latent_dim = params['adversary_hidden_dim']
        if params['adv_tcell_label_input']:  # and not params['adv_grad_reversal_loss'] in ['negate_gradient', 'invert_labels']:
            self.latent_dim += baseline_pred_pos.size(0)

    def forward(self, adv_input, labels):
        # prediction = self.linear(network_weights)
        # prediction = self.linear(self.dropout(network_activations))
        if self.params['adv_tcell_label_input']:
            input = torch.cat((labels[:, None], adv_input), dim=-1)
        else:
            input = adv_input
        h = F.relu(self.linear_input(F.relu(self.dropout(input))))
        # h2 = F.relu(self.linear_middle(self.dropout(h)))
        # h = self.layer_norm(h)
        prediction = self.linear(self.dropout(h))
        if self.params['adv_tcell_label_input']:  # and not self.params['adv_grad_reversal_loss'] in ['negate_gradient', 'invert_labels']:
            baseline_pred_pos = self.baseline_pred_pos[None, :].repeat(prediction.size(0), 1) * labels[:, None]
            baseline_pred_neg = self.baseline_pred_neg[None, :].repeat(prediction.size(0), 1) * (1-labels[:, None])
            # if torch.sum(prediction[0] - prediction[1]) == 0:
            #     print(prediction[:2])
            #     print(baseline_pred_pos[:2] + baseline_pred_neg[:2])
            #     print('')
            h = torch.cat((h, baseline_pred_pos + baseline_pred_neg), dim=-1)
            return baseline_pred_pos + baseline_pred_neg + prediction, h
        else:
            return prediction, h


class TransformerEmbedding(nn.Module):
    """
    Embeddings for the Transformers that use one-hot encoded sequences.
    """
    def __init__(self, data, params, embedding_dim):
        super().__init__()
        # embedding_dim = int(params['embedding_dim']/2) if params['concat_positional_encoding'] and params['positional_encoding'] else params['embedding_dim']
        self.params = params
        self.dropout = nn.Dropout(p=params['dropout'])
        self.embedding = nn.Embedding(
            num_embeddings=len(data.amino_acid_to_id) + 4,  # amino acids and padding and prediction tokens
            embedding_dim=embedding_dim,
            padding_idx=data.padding_token_idx)  # max_norm=1.0, norm_type=2.0

    def forward(self, x):
        return self.dropout(self.embedding(x))


class CategoricalEmbedding(nn.Module):
    """
    This allele embedding can be used instead of the allele pseudo sequences.
    """
    def __init__(self, num_categories, embedding_dim, params):
        super().__init__()
        self.dropout = nn.Dropout(p=params['dropout'])
        self.embedding = nn.Embedding(
            num_embeddings=num_categories,
            embedding_dim=embedding_dim
        )  # max_norm=1.0, norm_type=2.0

    def forward(self, x):
        return self.dropout(self.embedding(x))


class LearnedPosEmbedding(nn.Module):
    def __init__(self, embedding_dim, max_seq_len, params):
        super().__init__()
        self.dropout = nn.Dropout(p=params['dropout'])
        self.pos_embedding = nn.Parameter(torch.randn(max_seq_len, embedding_dim))

    def forward(self, x):
        return self.dropout(self.pos_embedding)[None, :x.size(1), :].repeat(x.size(0), 1, 1)


class PeptideTransformer(nn.Module):
    """
    """
    def __init__(self, params, data):
        super().__init__()
        self.params = params
        self.source_mhc_params = nn.ModuleList()
        actual_embedding_dim = int(params['embedding_dim']/2)

        self.embedding = TransformerEmbedding(data, params, actual_embedding_dim)

        self.fixed_pos_encoder = PositionalEncoding(d_model=actual_embedding_dim, dropout=params['dropout'], max_len=30)
        self.learned_pos_encoder = LearnedPosEmbedding(embedding_dim=actual_embedding_dim, max_seq_len=30, params=params)

        self.embedding_norm = nn.LayerNorm(actual_embedding_dim, eps=1e-5)
        self.embedding_pos_norm = nn.LayerNorm(params['embedding_dim'], eps=1e-5)

        if self.params['apply_transformer']:
            dim_feedforward = int(params['embedding_dim']/2) if params['half_feedforward_dim'] else params['embedding_dim']
            encoder_layer = nn.TransformerEncoderLayer(d_model=params['embedding_dim'], nhead=params['attention_heads'],
                                                       dim_feedforward=dim_feedforward, dropout=params['dropout'])
            encoder_norm = nn.LayerNorm(params['embedding_dim']) if self.params['layer_norm_transformer'] else None
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, params['num_attention_layers'], encoder_norm)

        mhc_presentation_dim = 1 if params['mhc_presentation_feature'] else 0

        if self.params['output_transform'] == 'linear':
            self.linear = nn.Linear(params['embedding_dim'] + mhc_presentation_dim, 1)
        else:
            self.linear1 = nn.Linear(params['embedding_dim'] + mhc_presentation_dim, params['embedding_dim'] + mhc_presentation_dim)
            self.linear = nn.Linear(params['embedding_dim'] + mhc_presentation_dim, 1)

        self.dropout = nn.Dropout(p=params['dropout'])

        if self.params['mhc_presentation_feature']:
            self.mhc_presentation = MHC_presentation(params)

        final_activations_dim = params['embedding_dim']
        transformer_output_dim = params['embedding_dim']
        if self.params['adversary_input'] == 'final_activations':
            self.adversary_input_dim = final_activations_dim
        elif self.params['adversary_input'] == 'final_activations_pred':
            self.adversary_input_dim = final_activations_dim + 1
        elif self.params['adversary_input'] == 'transformer_output':
            self.adversary_input_dim = transformer_output_dim
        elif self.params['adversary_input'] == 'transformer_output_pred':
            self.adversary_input_dim = transformer_output_dim + 1
        elif self.params['adversary_input'] == 'both':
            self.adversary_input_dim = transformer_output_dim + final_activations_dim
        elif self.params['adversary_input'] == 'both_pred':
            self.adversary_input_dim = transformer_output_dim + final_activations_dim + 1

    def forward(self, batch):
        padding_mask = batch['padding_mask']

        x = batch['peptide']
        embedding = self.embedding(x)

        if self.params['embedding_norm'] == 'amino_acid':
            embedding = self.embedding_norm(embedding)

        positional_encoding = torch.zeros(embedding.size())
        if self.params['positional_encoding'] == 'both':
            positional_encoding = self.fixed_pos_encoder(positional_encoding) + self.learned_pos_encoder(positional_encoding)
        elif self.params['positional_encoding'] == 'learned':
            positional_encoding = self.learned_pos_encoder(positional_encoding)
        else:  # fixed
            positional_encoding = self.fixed_pos_encoder(positional_encoding)

        embedding = torch.cat((embedding, positional_encoding), 2)

        embedding_t = torch.transpose(embedding, 1, 0)  # batch_first is not supported in PyTorch 1.8

        if self.params['apply_transformer']:
            output_all_t = self.transformer_encoder(embedding_t, mask=None, src_key_padding_mask=padding_mask)
        else:
            output_all_t = embedding_t

        output_all = torch.transpose(output_all_t, 1, 0)
        output_padding = 1 - padding_mask.long()[:, :, None].repeat(1, 1, output_all.size(2))
        prediction_output = torch.sum(output_all * output_padding, dim=1) / (batch['peptide_lengths'][:, None])

        adversary_input = prediction_output

        vis_input = prediction_output

        if self.params['mhc_presentation_feature']:
            presentation_scores = self.mhc_presentation(batch)
            prediction_output = torch.cat((prediction_output, presentation_scores[:, None]), dim=1)

        if self.params['output_transform'] == 'linear':
            prediction = self.linear(F.relu(self.dropout(prediction_output)))
        else:
            h = self.linear1(F.relu(self.dropout(prediction_output)))
            prediction = self.linear(F.relu(self.dropout(h)))
            if self.params['adversary_input'] == 'both':
                adversary_input = torch.cat((h, adversary_input), dim=-1)
            if self.params['adversary_input'] == 'both_pred':
                adversary_input = torch.cat((prediction, h, adversary_input), dim=-1)
            if self.params['adversary_input'] == 'transformer_output_pred':
                adversary_input = torch.cat((prediction, adversary_input), dim=-1)
            if self.params['adversary_input'] == 'final_activations_pred':
                adversary_input = torch.cat((prediction, h), dim=-1)
            if self.params['adversary_input'] == 'final_activations':
                adversary_input = h

        return prediction, vis_input, adversary_input

class PositionalEncoding(nn.Module):
    """
    Positional encoding implementation
    from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    Can be removed as soon as it is integrated into torch.nn
    """
    def __init__(self, d_model, dropout, max_len=100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(100.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]  # x.size(0) is the batch dimension
        return self.dropout(x)
