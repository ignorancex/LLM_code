import torch.nn as nn
import torch
from scipy.stats import spearmanr
import numpy as np
from einops import rearrange, repeat, pack

from src.TPP.model.fullynn.nonneg import NonNegLinear
from src.TPP.model.utils import L1_distance_across_events


TA = {
    'tanh': nn.Tanh
}


class FullyNN(nn.Module):
    def __init__(self, d_history, d_intensity, num_events, dropout, history_module, history_module_layers,
                 mlp_layers, nonlinear, event_toggle, zero_shift, device):
        super(FullyNN, self).__init__()
        self.device = device
        self.num_events = num_events
        self.event_toggle = event_toggle

        '''
        Should we force the model to start from 0?
        '''
        self.zero_shift = zero_shift

        '''
        Should we compress marker information into the history embedding?

        Caveat:
        FullyNN can not distinguish different markers because of computation graph overlap.
        It is expected that the original FullyNN achieves very inferior marker prediction performance in spite of the model size.
        '''
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
        self.weight_for_t = nn.Parameter(torch.zeros((1, d_intensity), device = self.device, requires_grad = True))
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
        self.layer_activation = TA[nonlinear]()
        self.aggregate = NonNegLinear(d_intensity, 1, bias = True, device = device)
        self.nonneg_activation = nn.Softplus()


    def forward(self, events_history, time_history, time_next, mean, var):
        '''
        The forwardpropagation function of FullyNN, triggered by pytorch.
        '''

        time_history = (time_history - mean) / var                             # [batch_size, seq_len]
        time_next = (time_next - mean) / var                                   # [batch_size, seq_len, num_events]
        
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
        
        time_embedding = time_next.unsqueeze(dim = -1) * self.nonneg_activation(self.weight_for_t)
                                                                               # [batch_size, seq_len, num_events, d_intensity] if self.event_toggle else [batch_size, seq_len, d_intensity]

        hidden_history = self.history_mapper(hidden_history)                   # [batch_size, seq_len, num_events, d_intensity] if self.event_toggle else [batch_size, seq_len, d_intensity]
        time_embedding = self.time_mapper(time_embedding)                      # [batch_size, seq_len, num_events, d_intensity] if self.event_toggle else [batch_size, seq_len, d_intensity]
        output = self.layer_activation(time_embedding + hidden_history)        # [batch_size, seq_len, num_events, d_intensity] if self.event_toggle else [batch_size, seq_len, d_intensity]

        for nonneg_layer in self.mlp:
            output = nonneg_layer(output)                                      # [batch_size, seq_len, num_events, d_intensity] if self.event_toggle else [batch_size, seq_len, d_intensity]
            output = self.layer_activation(output)                             # [batch_size, seq_len, num_events, d_intensity] if self.event_toggle else [batch_size, seq_len, d_intensity]

        integral = self.nonneg_activation(self.aggregate(output))              # [batch_size, seq_len, num_events, 1] if self.event_toggle else [batch_size, seq_len, 1]

        if self.zero_shift:
            zero = torch.ones_like(time_next, device = self.device) * ( - mean / var)
                                                                               # [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len]
            zero_time_embedding = zero.unsqueeze(dim = -1) * self.non_neg(self.weight_for_t)
                                                                               # [batch_size, seq_len, num_events, d_intensity] if self.event_toggle else [batch_size, seq_len, d_intensity]

            zero_time_embedding = self.time_mapper(zero_time_embedding)        # [batch_size, seq_len, num_events, d_intensity] if self.event_toggle else [batch_size, seq_len, d_intensity]
            zero_output = self.activate(zero_time_embedding + hidden_history)  # [batch_size, seq_len, num_events, d_intensity] if self.event_toggle else [batch_size, seq_len, d_intensity]
            for nonneg_layer in self.mlp:
                zero_output = nonneg_layer(zero_output)                        # [batch_size, seq_len, num_events, d_intensity] if self.event_toggle else [batch_size, seq_len, d_intensity]
                zero_output = self.activate(zero_output)                       # [batch_size, seq_len, num_events, d_intensity] if self.event_toggle else [batch_size, seq_len, d_intensity]
            
            zero_integral = self.nonneg_activation(self.aggregate(zero_output))# [batch_size, seq_len, num_events, 1] if self.event_toggle else [batch_size, seq_len, 1]
            integral = integral - zero_integral.detach()                       # [batch_size, seq_len, num_events, 1] if self.event_toggle else [batch_size, seq_len, 1]

        integral = integral.squeeze(dim = -1)                                  # [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len]

        return integral
    

    def integral_intensity_time_next_2d(self, events_history, time_history, time_next, resolution, mean, var):
        '''
        Intensity integral & intensity function prober. This function returns values of learned intensity function
        $ \lambda^*(m, t) $ and corresponding integral values $ \Lambda^*(m, t) $ at given times.

        The function name contains time_next_2d because the shape of time_next is [batch_size, seq_len].
        '''

        '''
        Prepare the history embedding.
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
        Prepare the time embedding.
        '''
        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
                                                                               # [resolution]
        original_time_expand = time_next.unsqueeze(dim = -1) * time_multiplier # [batch_size, seq_len, resolution]
        time_expand = original_time_expand.clone()                             # [batch_size, seq_len, resolution]
        if self.event_toggle:
            time_expand = repeat(original_time_expand, 'b s r -> b s r ne', ne = self.num_events)
                                                                               # [batch_size, seq_len, resolution, num_events]
        time_expand.requires_grad = True
        normed_time_expand = (time_expand - mean) / var                        # [batch_size, seq_len, resolution, num_events] is self.event_toggle else [batch_size, seq_len, resolution]

        emb_normed_time_expand = normed_time_expand.unsqueeze(dim = -1) * self.nonneg_activation(self.weight_for_t)
                                                                               # [batch_size, seq_len, resolution, num_events, d_intensity] is self.event_toggle else [batch_size, seq_len, resolution, d_intensity]

        emb_normed_time_expand = self.time_mapper(emb_normed_time_expand)      # [batch_size, seq_len, resolution, num_events, d_intensity] if self.event_toggle else [batch_size, seq_len, resolution, d_intensity]
        output = self.layer_activation(emb_normed_time_expand + hidden_history)# [batch_size, seq_len, resolution, num_events, d_intensity] if self.event_toggle else [batch_size, seq_len, resolution, d_intensity]

        '''
        Get intensity integrals.
        '''
        for nonneg_layer in self.mlp:
            output = nonneg_layer(output)                                      # [batch_size, seq_len, resolution, num_events, d_intensity] if self.event_toggle else [batch_size, seq_len, resolution, d_intensity]
            output = self.layer_activation(output)                             # [batch_size, seq_len, resolution, num_events, d_intensity] if self.event_toggle else [batch_size, seq_len, resolution, d_intensity]

        expand_integral = self.nonneg_activation(self.aggregate(output))       # [batch_size, seq_len, resolution, num_events, 1] if self.event_toggle else [batch_size, seq_len, resolution, 1]

        if self.zero_shift:
            if self.event_toggle:
                integral_at_zero = rearrange(expand_integral[:, :, 0, :, :].detach(), 'b s ne 1 -> b s 1 ne 1')
                expand_integral = expand_integral - integral_at_zero           # [batch_size, seq_len, 1, num_events, 1]
            else:
                integral_at_zero = rearrange(expand_integral[:, :, 0, :].detach(), 'b s 1 -> b s 1 1')
                expand_integral = expand_integral - integral_at_zero           # [batch_size, seq_len, 1, 1]

        '''
        Get intensity values at every sampled $ t $.
        '''
        expand_intensity = torch.autograd.grad(
            outputs=expand_integral,
            inputs=time_expand,
            grad_outputs=torch.ones_like(expand_integral),
        )[0]                                                                   # [batch_size, seq_len, resolution, num_events] if self.event_toggle else [batch_size, seq_len, resolution]
        time_expand.requires_grad = False

        expand_integral = expand_integral.squeeze(dim = -1).detach()           # [batch_size, seq_len, resolution, num_events] if self.event_toggle else [batch_size, seq_len, resolution]
        expand_intensity = expand_intensity.detach()                           # [batch_size, seq_len, resolution, num_events] if self.event_toggle else [batch_size, seq_len, resolution]

        '''
        Restore the original timestamp
        '''
        batch_size, seq_len = events_history.shape
        dummy_inception = torch.zeros((batch_size, seq_len, 1), device = self.device)
        timestamp, timestamp_ps = pack(
            [dummy_inception, original_time_expand.diff(dim = -1)],
            'b s *')                                                           # [batch_size, seq_len, resolution]

        return expand_integral, expand_intensity, timestamp


    def integral_intensity_time_next_3d(self, events_history, time_history, time_next, resolution, mean, var):
        '''
        Intensity integral & intensity function prober. This function returns values of learned intensity function
        $ \lambda^*(m, t) $ and corresponding integral values $ \Lambda^*(m, t) $ at given times.

        The function name contains time_next_3d because the shape of time_next is [batch_size, seq_len, num_events].
        '''

        '''
        Prepare the history embedding.
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

        hidden_history = repeat(hidden_history, 'b s di -> b s r ne ne1 di', r = resolution, ne = self.num_events, ne1 = self.num_events)
                                                                               # [batch_size, seq_len, resolution, num_events, num_events, d_intensity]

        '''
        Prepare the time embedding.
        '''
        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
                                                                               # [resolution]
        original_time_expand = time_next.unsqueeze(dim = -2) * rearrange(time_multiplier, 'r -> 1 1 r 1')
                                                                               # [batch_size, seq_len, resolution, num_events]
        time_expand = repeat(original_time_expand.clone(), '... -> ... ne', ne = self.num_events)                     
                                                                               # [batch_size, seq_len, resolution, num_events, num_events]
        time_expand.requires_grad = True
        normed_time_expand = (time_expand - mean) / var                        # [batch_size, seq_len, resolution, num_events, num_events]

        emb_normed_time_expand = normed_time_expand.unsqueeze(dim = -1) * self.nonneg_activation(self.weight_for_t)
                                                                               # [batch_size, seq_len, resolution, num_events, num_events, d_intensity]
        emb_normed_time_expand = self.time_mapper(emb_normed_time_expand)      # [batch_size, seq_len, resolution, num_events, num_events, d_intensity]
        output = self.layer_activation(emb_normed_time_expand + hidden_history)# [batch_size, seq_len, resolution, num_events, num_events, d_intensity]

        '''
        Get intensity integrals.
        '''
        for nonneg_layer in self.mlp:
            output = nonneg_layer(output)                                      # [batch_size, seq_len, resolution, num_events, num_events, d_intensity]
            output = self.layer_activation(output)                             # [batch_size, seq_len, resolution, num_events, num_events, d_intensity]

        expand_integral = self.nonneg_activation(self.aggregate(output))       # [batch_size, seq_len, resolution, num_events, num_events, 1]

        if self.zero_shift:
            integral_at_zero = rearrange(expand_integral[:, :, 0, :, :, :].detach(), 'b s ne ne1 1 -> b s 1 ne ne1 1')
            expand_integral = expand_integral - integral_at_zero               # [batch_size, seq_len, 1, num_events, num_events, 1]

        '''
        Get intensity values at every sampled $ t $.
        '''
        expand_intensity = torch.autograd.grad(
            outputs=expand_integral,
            inputs=time_expand,
            grad_outputs=torch.ones_like(expand_integral),
        )[0]                                                                   # [batch_size, seq_len, resolution, num_events, num_events]
        time_expand.requires_grad = False

        expand_integral = expand_integral.squeeze(dim = -1).detach()           # [batch_size, seq_len, resolution, num_events, num_events]
        expand_intensity = expand_intensity.detach()                           # [batch_size, seq_len, resolution, num_events, num_events]

        '''
        Restore the original timestamp
        '''
        batch_size, seq_len = events_history.shape
        dummy_inception = torch.zeros((batch_size, seq_len, 1, self.num_events), device = self.device)
        timestamp, timestamp_ps = pack(
            [dummy_inception, original_time_expand.diff(dim = -2)],
            'b s * ne')                                                        # [batch_size, seq_len, resolution, num_events]

        return expand_integral, expand_intensity, timestamp


    def model_probe_function(self, events_history, time_history, time_next, resolution, mean, var, mask_next):
        '''
        Prepare the history embedding.
        '''
        time_history = (time_history - mean) / var                             # [batch_size, seq_len]

        if self.event_toggle:
            events_embeddings = self.events(events_history)                    # [batch_size, seq_len, d_history]
            history, history_ps = pack(
                [events_embeddings, time_history],
                'b s *'
            )                                                                  # [batch_size, seq_len, d_history + 1]
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
        Prepare the time embedding.
        '''
        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
                                                                               # [resolution]
        original_time_expand = time_next.unsqueeze(dim = -1) * time_multiplier # [batch_size, seq_len, resolution]
        time_expand = original_time_expand.clone()                             # [batch_size, seq_len, resolution]
        if self.event_toggle:
            time_expand = repeat(original_time_expand, 'b s r -> b s r ne', ne = self.num_events)
                                                                               # [batch_size, seq_len, resolution, num_events]

        time_expand.requires_grad = True
        normed_time_expand = (time_expand - mean) / var                        # [batch_size, seq_len, resolution, num_events] if self.event_toggle else [batch_size, seq_len, resolution]
        
        emb_normed_time_expand = normed_time_expand.unsqueeze(dim = -1) * self.nonneg_activation(self.weight_for_t)
                                                                               # [batch_size, seq_len, resolution, num_events, d_intensity] if self.event_toggle else [batch_size, seq_len, resolution, d_intensity]

        emb_normed_time_expand = self.time_mapper(emb_normed_time_expand)      # [batch_size, seq_len, resolution, num_events, d_intensity] if self.event_toggle else [batch_size, seq_len, resolution, d_intensity]
        output = self.layer_activation(emb_normed_time_expand + hidden_history)# [batch_size, seq_len, resolution, num_events, d_intensity] if self.event_toggle else [batch_size, seq_len, resolution, d_intensity]

        '''
        Get intensity integrals.
        '''
        for nonneg_layer in self.mlp:
            output = nonneg_layer(output)                                      # [batch_size, seq_len, resolution, num_events, d_intensity] if self.event_toggle else [batch_size, seq_len, resolution, d_intensity]
            output = self.layer_activation(output)                             # [batch_size, seq_len, resolution, num_events, d_intensity] if self.event_toggle else [batch_size, seq_len, resolution, d_intensity]
        
        expand_integral = self.nonneg_activation(self.aggregate(output))       # [batch_size, seq_len, resolution, num_events, 1] if self.event_toggle else [batch_size, seq_len, resolution, 1]

        if self.zero_shift:
            if self.event_toggle:
                integral_at_zero = rearrange(expand_integral[:, :, 0, :, :].detach(), 'b s ne 1 -> b s 1 ne 1')
                expand_integral = expand_integral - integral_at_zero           # [batch_size, seq_len, 1, num_events, 1]
            else:
                integral_at_zero = rearrange(expand_integral[:, :, 0, :].detach(), 'b s 1 -> b s 1 1')
                expand_integral = expand_integral - integral_at_zero           # [batch_size, seq_len, 1, 1]

        expand_intensity = torch.autograd.grad(
            outputs = expand_integral,
            inputs = time_expand,
            grad_outputs = torch.ones_like(expand_integral),
            retain_graph = True
        )[0]                                                                   # [batch_size, seq_len, resolution, num_events] if self.event_toggle else [batch_size, seq_len, resolution]

        time_expand.requires_grad = False

        expand_integral = expand_integral.squeeze(dim = -1)                    # [batch_size, seq_len, resolution, num_events] if self.event_toggle else [batch_size, seq_len, resolution]

        '''
        Obtain timestamp here.
        '''
        batch_size, seq_len = time_history.shape
        zero_inception = torch.zeros((batch_size, seq_len, 1), device = self.device)
        timestamp, timstamp_ps = pack(
            [zero_inception, original_time_expand.diff(dim = -1)],
            'b s *')                                                           # [batch_size, seq_len, resolution]
        
        '''
        The data dict is defined here.
        This dict should pack all data required by plot().
        '''
        data = {}
        data['expand_intensity_for_each_event'] = expand_intensity             # [batch_size, seq_len, resolution, num_events] if self.event_toggle else [batch_size, seq_len, resolution]
        data['expand_integral_for_each_event'] = expand_integral               # [batch_size, seq_len, resolution, num_events] if self.event_toggle else [batch_size, seq_len, resolution]


        if self.event_toggle:
            expand_intensity = rearrange(expand_intensity.detach().cpu(), 'b s r ne -> b (s r) ne')
                                                                               # [batch_size, seq_len * resolution, num_event]
            expand_integral = rearrange(expand_integral.detach().cpu(), 'b s r ne -> b (s r) ne')
                                                                               # [batch_size, seq_len * resolution, num_event]
            
            spearman_matrix = []
            pearson_matrix = []
            L1_matrix = []
            for idx, (expand_intensity_per_seq, expand_integral_per_seq, mask_per_seq, time_next_per_seq) \
                in enumerate(zip(expand_intensity, expand_integral, mask_next, time_next)):
                seq_len = mask_per_seq.sum()

                probability_distribution = expand_intensity_per_seq * torch.exp(-expand_integral_per_seq)
                # rho: spearman coefficient
                spearman_matrix_per_seq = spearmanr(probability_distribution[:seq_len * resolution])[0]
                if self.num_events == 2:
                    spearman_matrix_per_seq = np.array([[1, spearman_matrix], [spearman_matrix, 1]])

                # r: pearson coefficient
                pearson_matrix_per_seq = np.corrcoef(probability_distribution[:seq_len * resolution], rowvar = False)
                # L^1 metric
                L1_matrix_per_seq = L1_distance_across_events(probability_distribution[:seq_len * resolution], 
                                                              resolution = resolution, num_events = self.num_events,
                                                              time_next = time_next_per_seq[:seq_len])
                spearman_matrix.append(spearman_matrix_per_seq)
                pearson_matrix.append(pearson_matrix_per_seq)
                L1_matrix.append(L1_matrix_per_seq)

            data['spearman_matrix'] = spearman_matrix
            data['pearson_matrix'] = pearson_matrix
            data['L1_matrix'] = L1_matrix

        return data, timestamp