import torch.nn as nn
import torch
from scipy.stats import spearmanr
import numpy as np
from einops import rearrange, repeat, reduce, pack

from src.TPP.model.utils import L1_distance_across_events
from src.TPP.model.ifib.nonneg import NonNegLinear


TA = {
    'tanh': nn.Tanh
}


class IFIB(nn.Module):
    def __init__(self, d_history, d_intensity, num_events, dropout, history_module, history_module_layers,
                 mlp_layers, nonlinear, event_toggle, denominator_shift, pretrain, alpha, beta, device):
        super(IFIB, self).__init__()
        self.device = device
        self.num_events = num_events
        self.event_toggle = event_toggle
        self.denominator_shift = denominator_shift

        if self.event_toggle:
            self.events = nn.Embedding(num_events + 1, d_history, padding_idx = num_events, device = device)
        else:
            self.events = None

        try:
            self.his_encoder = getattr(nn, history_module)(input_size = d_history + 1, hidden_size = d_history, num_layers = history_module_layers,\
                        batch_first = True, dropout = dropout, device = device)
        except:
            raise Exception(f'Unknown history module {history_module}.')

        '''
        Map the time number into a vector.
        '''
        self.weight_for_t = nn.Parameter(torch.zeros((self.num_events, d_intensity), device = self.device, requires_grad = True))
        nn.init.xavier_uniform_(self.weight_for_t)

        '''
        Map history and time embeddings into the same hidden space.
        '''
        self.history_mapper = nn.Linear(d_history, d_intensity, bias = True, device = device)
        self.time_mapper = NonNegLinear(d_intensity, d_intensity, device = self.device)

        '''
        IEM module featuring non-negative fully connected layers.
        '''
        self.mlp = nn.ModuleList([
            NonNegLinear(d_intensity, d_intensity, bias = True, device = device) for _ in range(mlp_layers)
        ])

        self.aggregate = NonNegLinear(d_intensity, 1, bias = True, device = device)
        self.layer_activation = TA[nonlinear]()

        '''
        These two scalars controls the residual connection in IEM.
        '''
        if pretrain:
            # alpha
            self.output_factor = nn.Parameter(torch.tensor(alpha,  device = self.device, requires_grad = True))
            # beta
            self.residual_factor = nn.Parameter(torch.tensor(beta, device = self.device, requires_grad = True))
        else:
            # alpha
            self.output_factor = torch.tensor(alpha,  device = self.device)
            # beta
            self.residual_factor = torch.tensor(beta, device = self.device)

        self.nonneg_activation = nn.Softplus()
        self.nonneg_factor = nn.ReLU()
        self.nonneg_integral = nn.Sigmoid()


    def forward(self, events_history, time_history, time_next, mean, var):
        '''
        The forwardpropagation function of IFIB-C, triggered by pytorch.
        '''

        '''
        Obtain historical embeddings.
        '''
        time_history = (time_history - mean) / var                             # [batch_size, seq_len]

        if self.event_toggle:
            events_embeddings = self.events(events_history)                    # [batch_size, seq_len, d_history]
            history, history_ps = pack([events_embeddings, time_history], 'b s *')
                                                                               # [batch_size, seq_len, d_history + 1]
        else:
            history = rearrange(time_history, '... -> ... 1')                  # [batch_size, seq_len, 1]
        
        # Reshape hidden output for full connection layers.
        hidden_history, (_, _) = self.his_encoder(history)                     # [batch_size, seq_len, d_history]

        if self.event_toggle:
            hidden_history = repeat(hidden_history, 'b s dh -> b s ne dh', ne = self.num_events)
                                                                               # [batch_size, seq_len, num_events, d_history]

        hidden_history = self.history_mapper(hidden_history)                   # [batch_size, seq_len, num_events, d_intensity] if we need events else [batch_size, seq_len, d_intensity]

        '''
        Obtain timestamp embeddings.
        '''
        time_next = (time_next - mean) / var                                   # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        time_next_zero = torch.ones_like(time_next) * (-mean / var)            # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]

        time_embedding = time_next.unsqueeze(dim = -1) * self.nonneg_activation(self.weight_for_t)
                                                                               # [batch_size, seq_len, num_events, d_intensity] if we need events else [batch_size, seq_len, d_intensity]
        time_zero_embedding = time_next_zero.unsqueeze(dim = -1) * self.nonneg_activation(self.weight_for_t)
                                                                               # [batch_size, seq_len, num_events, d_intensity] if we need events else [batch_size, seq_len, d_intensity]
        
        time_embedding = self.time_mapper(time_embedding)                      # [batch_size, seq_len, num_events, d_intensity] if we need events else [batch_size, seq_len, d_intensity]
        time_zero_embedding = self.time_mapper(time_zero_embedding)            # [batch_size, seq_len, num_events, d_intensity] if we need events else [batch_size, seq_len, d_intensity]
        
        output = time_embedding + hidden_history                               # [batch_size, seq_len, num_events, d_intensity] if we need events else [batch_size, seq_len, d_intensity]
        output_zero = time_zero_embedding + hidden_history                     # [batch_size, seq_len, num_events, d_intensity] if we need events else [batch_size, seq_len, d_intensity]

        for layer in self.mlp:
            residual = output                                                  # [batch_size, seq_len, num_events, d_intensity] if we need events else [batch_size, seq_len, d_intensity]
            output = layer(output)                                             # [batch_size, seq_len, num_events, d_intensity] if we need events else [batch_size, seq_len, d_intensity]
            output = self.layer_activation(output)                             # [batch_size, seq_len, num_events, d_intensity] if we need events else [batch_size, seq_len, d_intensity]
            output = self.nonneg_factor(self.residual_factor) * residual + self.nonneg_factor(self.output_factor) * output
                                                                               # [batch_size, seq_len, num_events, d_intensity] if we need events else [batch_size, seq_len, d_intensity]

            residual_zero = output_zero                                        # [batch_size, seq_len, num_events, d_intensity] if we need events else [batch_size, seq_len, d_intensity]
            output_zero = layer(output_zero)                                   # [batch_size, seq_len, num_events, d_intensity] if we need events else [batch_size, seq_len, d_intensity]
            output_zero = self.layer_activation(output_zero)                   # [batch_size, seq_len, num_events, d_intensity] if we need events else [batch_size, seq_len, d_intensity]
            output_zero = self.nonneg_factor(self.residual_factor) * residual_zero + self.nonneg_factor(self.output_factor) * output_zero
                                                                               # [batch_size, seq_len, num_events, d_intensity] if we need events else [batch_size, seq_len, d_intensity]

        probability_integral_from_t_to_inf = self.nonneg_integral(-self.aggregate(output))
                                                                               # [batch_size, seq_len, num_events, 1] if we need events else [batch_size, seq_len, 1]
        probability_integral_from_tl_to_inf = self.nonneg_integral(-self.aggregate(output_zero))
                                                                               # [batch_size, seq_len, num_events, 1] if we need events else [batch_size, seq_len, 1]

        if self.event_toggle:
            probability_integral_from_t_to_inf = rearrange(probability_integral_from_t_to_inf, '... 1 -> ...')
                                                                               # [batch_size, seq_len, num_events]
            probability_integral_from_tl_to_inf = reduce(probability_integral_from_tl_to_inf, '... ne 1 -> ... ()', 'sum')
                                                                               # [batch_size, seq_len, 1]
        else:
            probability_integral_from_t_to_inf = rearrange(probability_integral_from_t_to_inf, '... 1 -> ...')
                                                                               # [batch_size, seq_len]
            probability_integral_from_tl_to_inf = reduce(probability_integral_from_tl_to_inf, '... 1 -> ...', 'sum')
                                                                               # [batch_size, seq_len]

        return probability_integral_from_t_to_inf / (probability_integral_from_tl_to_inf + self.denominator_shift)


    def probability(self, events_history, time_history, time_next, resolution, mean, var):
        '''
        probability distribution prober. This function returns values of learned probability distribution
        $ p^*(m, t) $ at given times.

        The function name contains time_next_2d because the shape of time_next is [batch_size, seq_len].
        '''

        '''
        History embeddings
        '''
        time_history = (time_history - mean) / var                             # [batch_size, seq_len]

        if self.event_toggle:
            events_embeddings = self.events(events_history)                    # [batch_size, seq_len, d_history]
            history, history_ps = pack([events_embeddings, time_history], 'b s *')
                                                                               # [batch_size, seq_len, d_history + 1]
        else:
            history = rearrange(time_history, '... -> ... 1')                  # [batch_size, seq_len, 1]

        hidden_history, (_, _) = self.his_encoder(history)                     # [batch_size, seq_len, d_history]
        hidden_history = self.history_mapper(hidden_history)                   # [batch_size, seq_len, d_intensity]

        if self.event_toggle:
            hidden_history = repeat(hidden_history, 'b s di -> b s r ne di', r = resolution, ne = self.num_events)
                                                                               # [batch_size, seq_len, resolution, num_events, d_intensity]
        else:
            hidden_history = repeat(hidden_history, 'b s di -> b s r di', r = resolution)
                                                                               # [batch_size, seq_len, resolution, d_intensity]


        '''
        Expanded time embedding 
        '''
        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
                                                                               # [resolution]
        original_time_expand = time_multiplier * time_next.unsqueeze(dim = -1) # [batch_size, seq_len, resolution]
        time_expand = original_time_expand.clone()                             # [batch_size, seq_len, resolution]
        if self.event_toggle:
            time_expand = repeat(time_expand, 'b s r -> b s r ne', ne = self.num_events)
                                                                               # [batch_size, seq_len, resolution, num_events]

        time_expand.requires_grad = True
        time_expand_norm = (time_expand - mean) / var                          # [batch_size, seq_len, resolution, num_events] is we need events else [batch_size, seq_len, resolution]

        emb_time_expand = time_expand_norm.unsqueeze(dim = -1) * self.nonneg_activation(self.weight_for_t)
                                                                               # [batch_size, seq_len, resolution, num_events, d_intensity] is we need events else [batch_size, seq_len, resolution, d_intensity]

        emb_time_expand = self.time_mapper(emb_time_expand)                    # [batch_size, seq_len, resolution, num_events, d_intensity] if we need events else [batch_size, seq_len, resolution, d_intensity]
        output = emb_time_expand + hidden_history                              # [batch_size, seq_len, resolution, num_events, d_intensity] if we need events else [batch_size, seq_len, resolution, d_intensity]

        for layer in self.mlp:
            residual = output                                                  # [batch_size, seq_len, resolution, num_events, d_intensity] if we need events else [batch_size, seq_len, resolution, d_intensity]
            output = layer(output)                                             # [batch_size, seq_len, resolution, num_events, d_intensity] if we need events else [batch_size, seq_len, resolution, d_intensity]
            output = self.layer_activation(output)                             # [batch_size, seq_len, resolution, num_events, d_intensity] if we need events else [batch_size, seq_len, resolution, d_intensity]
            output = self.nonneg_factor(self.residual_factor) * residual + self.nonneg_factor(self.output_factor) * output
                                                                               # [batch_size, seq_len, resolution, num_events, d_intensity] if we need events else [batch_size, seq_len, resolution, d_intensity]

        expand_integral = self.nonneg_integral(-self.aggregate(output))        # [batch_size, seq_len, resolution, num_events, 1] if we need events else [batch_size, seq_len, resolution, 1]
        
        if self.event_toggle:
            integral_from_zero_to_inf = expand_integral[:, :, 0, :, :].detach()# [batch_size, seq_len, num_events, 1]
            integral_sum = reduce(integral_from_zero_to_inf, 'b s ne 1 -> b s 1 1 1', 'sum')
                                                                               # [batch_size, seq_len, 1, 1, 1]
            expand_integral = expand_integral / (integral_sum + self.denominator_shift)
                                                                               # [batch_size, seq_len, resolution, num_events, 1]
        else:
            integral_from_zero_to_inf = expand_integral[:, :, 0, :].detach()   # [batch_size, seq_len, 1]
            integral_sum = rearrange(integral_from_zero_to_inf, 'b s 1 -> b s 1 1')
                                                                               # [batch_size, seq_len, 1, 1]
            expand_integral = expand_integral / (integral_sum + self.denominator_shift)
                                                                               # [batch_size, seq_len, resolution, 1]

        expand_probability = - torch.autograd.grad(
            outputs=expand_integral,
            inputs=time_expand,
            grad_outputs=torch.ones_like(expand_integral),
        )[0]                                                                   # [batch_size, seq_len, resolution, num_events] if we need events else [batch_size, seq_len, resolution]
        time_expand.requires_grad = False

        expand_probability = expand_probability.detach()                       # [batch_size, seq_len, resolution, num_events] if we need events else [batch_size, seq_len, resolution]

        '''
        Restore the original timestamp
        '''
        batch_size, seq_len = events_history.shape[0], events_history.shape[1]
        dummy_inception = torch.zeros((batch_size, seq_len, 1), device = self.device)
        timestamp, timestamp_ps = pack(
            [dummy_inception, original_time_expand.diff(dim = -1)],
            'b s *')                                                           # [batch_size, seq_len, resolution]

        return expand_probability, timestamp


    def model_probe_function(self, events_history, time_history, time_next, resolution, mean, var, mask):
        '''
        History embeddings
        '''
        time_history = (time_history - mean) / var                             # [batch_size, seq_len]

        if self.event_toggle:
            events_embeddings = self.events(events_history)                    # [batch_size, seq_len, d_history]
            history, history_ps = pack([events_embeddings, time_history], 'b s *')
                                                                               # [batch_size, seq_len, d_history + 1]
        else:
            history = rearrange(time_history, '... -> ... 1')                  # [batch_size, seq_len, 1]

        hidden_history, (_, _) = self.his_encoder(history)                     # [batch_size, seq_len, d_history]
        hidden_history = self.history_mapper(hidden_history)                   # [batch_size, seq_len, d_intensity]

        if self.event_toggle:
            hidden_history = repeat(hidden_history, 'b s di -> b s r ne di', r = resolution, ne = self.num_events)
                                                                               # [batch_size, seq_len, resolution, num_events, d_intensity]
        else:
            hidden_history = repeat(hidden_history, 'b s di -> b s r di', r = resolution)
                                                                               # [batch_size, seq_len, resolution, d_intensity]

        '''
        Expanded time embedding 
        '''
        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
                                                                               # [resolution]
        original_time_expand = time_multiplier * rearrange(time_next, '... -> ... 1')
                                                                               # [batch_size, seq_len, resolution]
        time_expand = original_time_expand.clone()                             # [batch_size, seq_len, resolution]
        if self.event_toggle:
            time_expand = repeat(original_time_expand, 'b s r -> b s r ne', ne = self.num_events)
                                                                               # [batch_size, seq_len, resolution, num_events]
        
        time_expand.requires_grad = True      
        time_expand_norm = (time_expand - mean) / var                          # [batch_size, seq_len, resolution, num_events] if we need events else [batch_size, seq_len, resolution]

        emb_time_expand = time_expand_norm.unsqueeze(dim = -1) * self.nonneg_activation(self.weight_for_t)
                                                                               # [batch_size, seq_len, resolution, num_events, d_intensity] is we need events else [batch_size, seq_len, resolution, d_intensity]

        emb_time_expand = self.time_mapper(emb_time_expand)                    # [batch_size, seq_len, resolution, num_events, d_intensity] if we need events else [batch_size, seq_len, resolution, d_intensity]
        output = emb_time_expand + hidden_history                              # [batch_size, seq_len, resolution, num_events, d_intensity] if we need events else [batch_size, seq_len, resolution, d_intensity]


        for layer in self.mlp:
            residual = output                                                  # [batch_size, seq_len, resolution, num_events, d_intensity] if we need events else [batch_size, seq_len, resolution, d_intensity]
            output = layer(output)                                             # [batch_size, seq_len, resolution, num_events, d_intensity] if we need events else [batch_size, seq_len, resolution, d_intensity]
            output = self.layer_activation(output)                             # [batch_size, seq_len, resolution, num_events, d_intensity] if we need events else [batch_size, seq_len, resolution, d_intensity]
            output = self.nonneg_factor(self.residual_factor) * residual + self.nonneg_factor(self.output_factor) * output
                                                                               # [batch_size, seq_len, resolution, num_events, d_intensity] if we need events else [batch_size, seq_len, resolution, d_intensity]

        expand_integral = self.nonneg_activation(-self.aggregate(output))      # [batch_size, seq_len, resolution, num_events, 1] if self.event_toggle else [batch_size, seq_len, resolution, 1]
        expand_integral = expand_integral.squeeze(dim = -1)                    # [batch_size, seq_len, resolution, num_events] if self.event_toggle else [batch_size, seq_len, resolution]

        if self.event_toggle:
            integral_from_zero_to_inf = expand_integral[:, :, 0, :].detach()   # [batch_size, seq_len, num_events]
            integral_sum = reduce(integral_from_zero_to_inf, 'b s ne -> b s ()', 'sum')
                                                                               # [batch_size, seq_len, 1]
            integral_sum = rearrange(integral_sum, 'b s 1 -> b s 1 1')         # [batch_size, seq_len, 1, 1]
            expand_integral = expand_integral / (integral_sum + self.denominator_shift)
                                                                               # [batch_size, seq_len, resolution, num_events]
        else:
            integral_from_zero_to_inf = expand_integral[:, :, 0].detach()      # [batch_size, seq_len]
            integral_sum = rearrange(integral_from_zero_to_inf, 'b s -> b s 1')# [batch_size, seq_len, 1, 1]
            expand_integral = expand_integral / (integral_sum + self.denominator_shift)
                                                                               # [batch_size, seq_len, resolution]

        # Gradient 1: Integral -> time
        events_probability_at_each_interpolated_timestamp = \
        - torch.autograd.grad(
            outputs=expand_integral,
            inputs=time_expand,
            grad_outputs=torch.ones_like(expand_integral),
            retain_graph=True
        )[0]                                                                   # [batch_size, seq_len, resolution, num_events] if we need events else [batch_size, seq_len, resolution]
                
        time_expand.requires_grad = False

        # Timestamp part
        batch_size, seq_len = hidden_history.shape[0], hidden_history.shape[1]
        zero_inception = torch.zeros((batch_size, seq_len, 1), device = self.device)
        timestamp, timstamp_ps = pack(
            [zero_inception, original_time_expand.diff(dim = -1)],
            'b s *')                                                           # [batch_size, seq_len, resolution]
        timestamp = rearrange(timestamp, 'b s r -> b (s r)')                   # [batch_size, seq_len * resolution]

        '''
        The data dict is defined here.
        This dict should pack all data required by plot().
        '''
        data = {}
        data['expand_probability_for_each_event'] = events_probability_at_each_interpolated_timestamp
                                                                               # [batch_size, seq_len, resolution, num_events] if self.event_toggle else [batch_size, seq_len, resolution]

        if self.event_toggle:
            probability_for_each_event = \
                rearrange(events_probability_at_each_interpolated_timestamp.detach().cpu(), 'b s r ne -> b (s r) ne')
                                                                               # [batch_size, seq_len * resolution, num_events]
            
            spearman_matrix = []
            pearson_matrix = []
            L1_matrix = []
            for idx, (expand_probability_per_seq, mask_per_seq, time_next_per_seq) in \
                                                  enumerate(zip(probability_for_each_event, mask, time_next)):
                seq_len = mask_per_seq.sum()
                # rho: spearman coefficient
                spearman_matrix_per_seq = spearmanr(expand_probability_per_seq[:seq_len * resolution])[0]
                if self.num_events == 2:
                    spearman_matrix_per_seq = np.array([[1, spearman_matrix_per_seq], [spearman_matrix_per_seq, 1]])

                # r: pearson coefficient
                pearson_matrix_per_seq = np.corrcoef(expand_probability_per_seq[:seq_len * resolution], rowvar = False)
                # L^1 metric
                L1_matrix_per_seq = L1_distance_across_events(expand_probability_per_seq[:seq_len * resolution], 
                                                resolution = resolution, num_events = self.num_events,
                                                time_next = time_next_per_seq[:seq_len])

                spearman_matrix.append(spearman_matrix_per_seq)
                pearson_matrix.append(pearson_matrix_per_seq)
                L1_matrix.append(L1_matrix_per_seq)

            data['spearman_matrix'] = spearman_matrix
            data['pearson_matrix'] = pearson_matrix
            data['L1_matrix'] = L1_matrix

        return data, timestamp