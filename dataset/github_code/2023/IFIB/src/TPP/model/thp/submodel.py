import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat, pack
import numpy as np
from scipy.stats import spearmanr

from src.TPP.model.utils import L1_distance_across_events
from src.TPP.model.thp.utils import softplus_ext
from src.TPP.model.thp.transformers import TransformerTPP


class THP(nn.Module):
    def __init__(self, device, num_events, d_input, d_rnn, d_hidden, n_layers, n_head, d_qk,\
                 d_v, dropout, beta, monte_carlo_resolution):
        super(THP, self).__init__()
        self.device = device
        self.num_events = num_events
        self.monte_carlo_resolution = monte_carlo_resolution

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.ones((self.num_events), dtype = torch.float32, \
                                  device = self.device, requires_grad = True))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.ones((self.num_events), dtype = torch.float32, \
                                  device = self.device, requires_grad = True) * beta)
        
        # convert hidden vectors into valid intensity function values.
        self.linear = nn.Linear(d_input, num_events, device = self.device)

        # the history encoder
        self.history_encoder = TransformerTPP(num_events, device = self.device, d_input = d_input, \
                                              d_rnn = d_rnn, d_hidden = d_hidden, n_layers = n_layers, \
                                              n_head = n_head, d_qk = d_qk, d_v = d_v, dropout = dropout)


    def forward(self, time_history, time_next, events_history, mask_history, mask_next):
        '''
        The forwardpropagation function of FullyNN, triggered by pytorch.
        '''
        history = self.history_encoder(time_history, events_history, mask_history)
                                                                               # [batch_size, seq_len, d_input]

        aggregate_time = time_history.cumsum(dim = -1)                         # [batch_size, seq_len]
        # Avoid zero denominator
        aggregate_time = aggregate_time + 1.0                                  # [batch_size, seq_len]

        scaled_time = (time_next / aggregate_time).unsqueeze(dim = -1)         # [batch_size, seq_len, 1]
        intensity_all_events = softplus_ext(self.linear(history) + self.alpha * scaled_time, beta = F.softplus(self.beta))
                                                                               # [batch_size, seq_len, num_events]
        integral_all_events = self.compute_integral_monte_carlo(history, time_history, time_next, mask_next)
                                                                               # [batch_size, seq_len, num_events]
        
        return integral_all_events, intensity_all_events


    def compute_integral_monte_carlo(self, history, time_history, time_next, mask_next):
        """ Log-likelihood of non-events, using Monte Carlo integration. """
    
        diff_time = time_next * mask_next
        aggregate_time = rearrange(time_history.cumsum(dim = -1), '... -> ... 1 1')
                                                                               # [batch_size, seq_len, 1, 1]
        aggregate_time = aggregate_time + 1.0                                  # [batch_size, seq_len, 1, 1]

        temp_time = diff_time.unsqueeze(dim = -1) * \
                    torch.rand([*diff_time.size(), self.monte_carlo_resolution], device = self.device)
                                                                               # [batch_size, seq_len, resolution]
        temp_time = self.alpha * temp_time.unsqueeze(dim = -1) / aggregate_time# [batch_size, seq_len, resolution, num_events]

        intensity_all_events_pre_softplus = self.linear(history)               # [batch_size, seq_len, num_events]
        intensity_all_events_pre_softplus = repeat(intensity_all_events_pre_softplus, '... ne -> ... r ne', r = self.monte_carlo_resolution)
                                                                               # [batch_size, seq_len, resolution, num_events]
        all_lambda = softplus_ext(intensity_all_events_pre_softplus + temp_time, F.softplus(self.beta))
                                                                               # [batch_size, seq_len, resolution, num_events]
        lambda_mean = torch.mean(all_lambda, dim = -2)                         # [batch_size, seq_len, num_events]
    
        unbiased_integral = lambda_mean * diff_time.unsqueeze(dim = -1)        # [batch_size, seq_len, num_events]

        return unbiased_integral


    def integral_intensity_time_next_2d(self, events_history, time_history, time_next, mask_history, resolution, mean, var):
        '''
        Intensity integral & intensity function prober. This function returns values of learned intensity function
        $ \lambda^*(m, t) $ and corresponding integral values $ \Lambda^*(m, t) $ at given times.

        The function name contains time_next_2d because the shape of time_next is [batch_size, seq_len].
        '''
        assert len(time_next.shape) == 2, "Wrong input time tensor shape."

        history = self.history_encoder(time_history, events_history, mask_history)
                                                                               # [batch_size, seq_len, d_input]
        history = repeat(history, 'b s di -> b s 1 di')                        # [batch_size, seq_len, 1, d_input]

        aggregate_time = time_history.cumsum(dim = -1).unsqueeze(dim = -1)     # [batch_size, seq_len]
        # Avoid zero denominator
        aggregate_time = aggregate_time + 1.0                                  # [batch_size, seq_len]

        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
        expanded_time = time_next.unsqueeze(dim = -1) * time_multiplier        # [batch_size, seq_len, resolution]
        expanded_time_gap = torch.diff(expanded_time, dim = -1)                # [batch_size, seq_len, resolution - 1]
        zero_gap = torch.zeros_like(expanded_time_gap[:, :, 0])                # [batch_size, seq_len]
        expanded_time_gap, expanded_time_gap_ps = pack(
            (zero_gap, expanded_time_gap), 'b s *'
        )                                                                      # [batch_size, seq_len, resolution]

        scaled_time = (expanded_time / aggregate_time).unsqueeze(dim = -1)     # [batch_size, seq_len, resolution, 1]
        expanded_intensity_all_events = softplus_ext(self.linear(history) + self.alpha * scaled_time, beta = F.softplus(self.beta))
                                                                               # [batch_size, seq_len, resolution, num_events]
        expanded_intensity_all_events_for_monte_carlo, \
        expanded_intensity_all_events_for_monte_carlo_ps = pack(
            (expanded_intensity_all_events[:, :, 0, :], \
             expanded_intensity_all_events[:, :, :-1, :]), 'b s * ne'
        )                                                                      # [batch_size, seq_len, resolution, num_events]
        expanded_integral_all_events \
            = torch.cumsum(expanded_intensity_all_events_for_monte_carlo * expanded_time_gap.unsqueeze(dim = -1), dim = -2)
                                                                               # [batch_size, seq_len, resolution, num_events]

        # aggregated timestamp
        batch_size, seq_len, _ = expanded_time.shape
        timestamp = torch.cat(
            (torch.zeros((batch_size, seq_len, 1), device = self.device), expanded_time.diff(dim = -1)),
            dim = -1)                                                          # [batch_size, seq_len, resolution]
        
        return expanded_integral_all_events, expanded_intensity_all_events, timestamp
        

    def integral_intensity_time_next_3d(self, events_history, time_history, time_next, mask_history, resolution, mean, var):
        '''
        Intensity integral & intensity function prober. This function returns values of learned intensity function
        $ \lambda^*(m, t) $ and corresponding integral values $ \Lambda^*(m, t) $ at given times.

        The function name contains time_next_3d because the shape of time_next is [batch_size, seq_len, num_events].
        '''
        assert len(time_next.shape) == 3, "Wrong input time tensor shape."

        history = self.history_encoder(time_history, events_history, mask_history)
                                                                               # [batch_size, seq_len, d_input]

        # Intensity and integral estimation
        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
                                                                               # [resolution]
        original_expanded_time = time_next.unsqueeze(dim = -1) * time_multiplier
                                                                               # [batch_size, seq_len, num_event, resolution]
        expanded_time = original_expanded_time.unsqueeze(dim = -1)             # [batch_size, seq_len, num_event, resolution, 1]

        expanded_time_gap = torch.diff(expanded_time.squeeze(dim = -1), dim = -1)
                                                                               # [batch_size, seq_len, num_events, resolution - 1]
        zero_gap = torch.zeros_like(expanded_time_gap[:, :, :, 0])             # [batch_size, seq_len, num_events]
        expanded_time_gap, expanded_time_gap_ps = pack(
            (zero_gap, expanded_time_gap), 'b s ne *'
        )                                                                      # [batch_size, seq_len, num_events, resolution]

        aggregate_time = rearrange(torch.cumsum(time_history, dim = -1), 'b s -> b s 1 1 1')
                                                                               # [batch_size, seq_len, 1, 1, 1]
        aggregate_time = aggregate_time + 1.0                                  # [batch_size, seq_len, 1, 1, 1]
        scaled_expanded_time = expanded_time / aggregate_time                  # [batch_size, seq_len, num_event, resolution, 1]

        intensity_for_each_event = self.linear(history)                        # [batch_size, seq_len, num_events]
        intensity_for_each_event = rearrange(intensity_for_each_event, '... ne -> ... 1 1 ne')
                                                                               # [batch_size, seq_len, 1, 1, num_events]
        
        expanded_intensity_across_all_events = softplus_ext(self.alpha * scaled_expanded_time + intensity_for_each_event, F.softplus(self.beta))
                                                                               # [batch_size, seq_len, num_events, resolution, num_events]

        expanded_intensity_across_events, \
        expanded_intensity_across_events_ps = pack(
            (expanded_intensity_across_all_events[:, :, :, 0, :], \
             expanded_intensity_across_all_events[:, :, :, :-1, :]), 'b s ne * ne1'
        )                                                                      # [batch_size, seq_len, num_events, resolution, num_events]
        expanded_integral_across_events = torch.cumsum(expanded_intensity_across_events * expanded_time_gap.unsqueeze(dim = -1), dim = -2)
                                                                               # [batch_size, seq_len, num_events, resolution, num_events]
        
        # aggregated timestamp
        batch_size, seq_len, num_event, _ = original_expanded_time.shape
        timestamp = torch.cat(
            (torch.zeros((batch_size, seq_len, num_event, 1), device = self.device), original_expanded_time.diff(dim = -1)),
            dim = -1)                                                          # [batch_size, seq_len, num_events, resolution]
        
        return expanded_integral_across_events, expanded_intensity_across_events, timestamp
    

    def model_probe_function(self, events_history, time_history, time_next, mask_history, mask_next, resolution, mean, var):
        '''
        Prepare the history embedding.
        '''
        history = self.history_encoder(time_history, events_history, mask_history)
                                                                               # [batch_size, seq_len, d_input]
        history = repeat(history, 'b s di -> b s 1 di')                        # [batch_size, seq_len, 1, d_input]

        aggregate_time = time_history.cumsum(dim = -1).unsqueeze(dim = -1)     # [batch_size, seq_len]
        # Avoid zero denominator
        aggregate_time = aggregate_time + 1.0                                  # [batch_size, seq_len]

        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
        expanded_time = time_next.unsqueeze(dim = -1) * time_multiplier        # [batch_size, seq_len, resolution]
        expanded_time_gap = torch.diff(expanded_time, dim = -1)                # [batch_size, seq_len, resolution - 1]
        zero_gap = torch.zeros_like(expanded_time_gap[:, :, 0])                # [batch_size, seq_len]
        expanded_time_gap, expanded_time_gap_ps = pack(
            (zero_gap, expanded_time_gap), 'b s *'
        )                                                                      # [batch_size, seq_len, resolution]

        scaled_time = (expanded_time / aggregate_time).unsqueeze(dim = -1)     # [batch_size, seq_len, resolution, 1]
        expanded_intensity_all_events = softplus_ext(self.linear(history) + self.alpha * scaled_time, beta = F.softplus(self.beta))
                                                                               # [batch_size, seq_len, resolution, num_events]
        expanded_intensity_all_events_for_monte_carlo, \
        expanded_intensity_all_events_for_monte_carlo_ps = pack(
            (expanded_intensity_all_events[:, :, 0, :], \
             expanded_intensity_all_events[:, :, :-1, :]), 'b s * ne'
        )                                                                      # [batch_size, seq_len, resolution, num_events]
        expanded_integral_all_events \
            = torch.cumsum(expanded_intensity_all_events_for_monte_carlo * expanded_time_gap.unsqueeze(dim = -1), dim = -2)
                                                                               # [batch_size, seq_len, resolution, num_events]

        # aggregated timestamp
        batch_size, seq_len, _ = expanded_time.shape
        timestamp = torch.cat(
            (torch.zeros((batch_size, seq_len, 1), device = self.device), expanded_time.diff(dim = -1)),
            dim = -1)                                                          # [batch_size, seq_len, resolution]
        
        # construct the plot dict
        data = {}
        data['expand_intensity_for_each_event'] = expanded_intensity_all_events# [batch_size, seq_len, resolution, num_events]
        data['expand_integral_for_each_event'] = expanded_integral_all_events  # [batch_size, seq_len, resolution, num_events]

        # THP always assumes that the event information is present.
        # So model_probe_function() always provides spearman, pearson coefficient and L1 distance.

        expand_intensity = rearrange(expanded_intensity_all_events.detach().cpu(), 'b s r ne -> b (s r) ne')
                                                                               # [batch_size, seq_len * resolution, num_event]
        expand_integral = rearrange(expanded_integral_all_events.detach().cpu(), 'b s r ne -> b (s r) ne')
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