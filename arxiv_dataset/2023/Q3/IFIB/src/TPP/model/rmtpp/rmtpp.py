import torch
import torch.nn as nn

from einops import rearrange


class RMTPPModule(nn.Module):
    def __init__(self, input_size, hidden_size, history_encoder_layers, dropout, event_toggle, 
                 num_events, output_size, limited_history_norm, time_scalar_min, device):
        super(RMTPPModule, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_events = num_events
        self.event_toggle = event_toggle
        self.limited_history_norm = limited_history_norm
        self.time_scalar_min = time_scalar_min
        self.zero_shift_factor = 1e-12

        '''
        RMTPP's history encoder
        '''
        self.time_embedding = nn.Linear(1, input_size, device = self.device)
        self.rnn = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = history_encoder_layers, batch_first = True, \
                            dropout = dropout, device = self.device)
        self.project = nn.Linear(hidden_size, output_size, device = self.device)
        
        '''
        RMTPP's dedicated mark predictor.
        '''
        self.event_embedding = nn.Embedding(num_embeddings = num_events + 1, embedding_dim = input_size,\
                                            padding_idx = num_events, device = self.device)
        self.event_mapper = nn.Linear(output_size, self.num_events, device = self.device)
        self.event_decider = nn.Softmax(dim = -1)
        self.non_neg_activation = nn.Softplus()

        '''
        Parameters in Equation 11.

        self.intensity      -> \mathbf{v}^{t^{\top}}
        self.time_scalar    -> w^{t}
        self.base_intensity -> b^t
        '''
        self.intensity = nn.Linear(output_size, 1, device = self.device)
        self.time_scalar = nn.Linear(output_size, 1, device = self.device)
        self.base_intensity = nn.Linear(output_size, 1, device = self.device)


    def clamp_time_scalar(self, time_scalar):
        '''
        The integral of the intensity function contains \frac{1}{w^t}. Therefore, we must prevent w^t from becoming too small that explodes the integral.
        This function inflicts the absolute value of w^t not to be smaller than time_scalar_min.
        '''
        time_scalar_sign = (time_scalar >= 0).int() - (time_scalar < 0).int()  # [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len, 1]
        shifted_time_scalar_abs_value = torch.abs(time_scalar).clamp(min = self.time_scalar_min)
                                                                               # [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len, 1]
        time_scalar = shifted_time_scalar_abs_value * time_scalar_sign         # [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len, 1]
        return time_scalar


    def forward(self, events_history, time_history, time_next, mean, var):
        '''
        The forwardpropagation function of RMTPP, triggered by pytorch.
        '''

        time_history = (time_history) / var
        time_next = (time_next) / var

        time_history, time_next = time_history.unsqueeze(dim = -1), time_next.unsqueeze(dim = -1)
                                                                               # [batch_size, seq_len, 1]

        time_vec = self.time_embedding(time_history)                           # [batch_size, seq_len, input_size]
        if self.event_toggle:
            events_vec = self.event_embedding(events_history)                  # [batch_size, seq_len, input_size]
            input_vec = time_vec + events_vec
        else:
            input_vec = time_vec                                               # [batch_size, seq_len, input_size]

        hidden_history, (_, _) = self.rnn(input_vec)                           # [batch_size, seq_len, hidden_size]
        hidden_history = self.project(hidden_history)                          # [batch_size, seq_len, output_size]
        hidden_history = torch.relu(hidden_history)                            # [batch_size, seq_len, output_size]

        history_part = self.intensity(hidden_history)                          # [batch_size, seq_len, 1]

        if self.limited_history_norm:
            history_part = torch.tanh(history_part)                            # [batch_size, seq_len, 1]

        history_part = torch.exp(history_part)                                 # [batch_size, seq_len, 1]

        constant = history_part * torch.exp(self.base_intensity(hidden_history))
                                                                               # [batch_size, seq_len, 1]
        time_scalar = self.time_scalar(hidden_history)                         # [batch_size, seq_len, 1]

        # time_scalar can not be zero.
        time_scalar = self.clamp_time_scalar(time_scalar)                      # [batch_size, seq_len, 1]

        # Get the intensity function and corresponding integral.
        intensity = torch.exp(time_scalar * time_next) * constant              # [batch_size, seq_len, 1]
        integral = (intensity - constant) / time_scalar                        # [batch_size, seq_len, 1]

        mark = None
        if self.event_toggle:
            intensity, integral = intensity.sum(dim = -1), integral.sum(dim = -1)
                                                                               # [batch_size, seq_len]
            mark = self.event_decider(self.event_mapper(hidden_history))       # [batch_size, seq_length, num_events]


        return integral, intensity, mark, history_part


    def integral_intensity_time_next_2d(self, events_history, time_history, time_next, resolution, mean, var):
        '''
        Intensity integral & intensity function prober. This function returns values of learned intensity function
        $ \lambda^*(m, t) $ and corresponding integral values $ \Lambda^*(m, t) $ at given times.

        The function name contains time_next_2d because the shape of time_next is [batch_size, seq_len].
        '''

        time_history = ((time_history) / var).unsqueeze(dim = -1)

        time_vec = self.time_embedding(time_history)                           # [batch_size, seq_len, input_size]
        if self.num_events > 1:
            events_vec = self.event_embedding(events_history)                  # [batch_size, seq_len, input_size]
            input_vec = time_vec + events_vec
        else:
            input_vec = time_vec                                               # [batch_size, seq_len, input_size]

        output, (_, _) = self.rnn(input_vec)                                   # [batch_size, seq_len, hidden_size]
        history_output = self.project(output)                                  # [batch_size, seq_len, output_size]
        history_output = torch.relu(history_output)                            # [batch_size, seq_len, output_size]

        history_part = self.intensity(history_output)                          # [batch_size, seq_len, 1]
        if self.limited_history_norm:
            history_part = torch.tanh(history_part)                            # [batch_size, seq_len, 1]

        history_part = torch.exp(history_part)                                 # [batch_size, seq_len, 1]

        constant = (history_part * torch.exp(self.base_intensity(history_output))).unsqueeze(-1)
                                                                               # [batch_size, seq_len, 1, 1]

        time_scalar = self.time_scalar(history_output).unsqueeze(dim = -1)     # [batch_size, seq_len, 1, 1]

        # time_scalar can not be zero.
        time_scalar = self.clamp_time_scalar(time_scalar)                      # [batch_size, seq_len, 1, 1]


        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
        original_time_expand = time_next.unsqueeze(dim = -1) * time_multiplier # [batch_size, seq_len, resolution]
        original_time_expand_normed = original_time_expand / var               # [batch_size, seq_len, resolution]
        expanded_time = original_time_expand_normed.unsqueeze(dim = -2)        # [batch_size, seq_len, 1, resolution]
        
        intensity_events = torch.exp(time_scalar * expanded_time) * constant   # [batch_size, seq_len, 1, resolution]
        integral_events = (intensity_events - constant) / time_scalar          # [batch_size, seq_len, 1, resolution]

        # aggregated timestamp
        batch_size, seq_len, _ = original_time_expand.shape
        timestamp = torch.cat(
            (torch.zeros((batch_size, seq_len, 1), device = self.device), original_time_expand.diff(dim = -1)),
            dim = -1)                                                          # [batch_size, seq_len, resolution]
        
        intensity = rearrange(intensity_events, 'b s ne r -> b s r ne')        # [batch_size, seq_len, resolution, 1]
        integral = rearrange(integral_events, 'b s ne r -> b s r ne')          # [batch_size, seq_len, resolution, 1]

        return integral, intensity, timestamp


    def model_probe_function(self, events_history, time_history, time_next, resolution, mean, var, mask_next):
        time_history = ((time_history) / var).unsqueeze(dim = -1)

        time_vec = self.time_embedding(time_history)                           # [batch_size, seq_len, input_size]
        if self.num_events > 1:
            events_vec = self.event_embedding(events_history)                  # [batch_size, seq_len, input_size]
            input_vec = time_vec + events_vec
        else:
            input_vec = time_vec                                               # [batch_size, seq_len, input_size]

        output, (_, _) = self.rnn(input_vec)                                   # [batch_size, seq_len, hidden_size]
        history_output = self.project(output)                                  # [batch_size, seq_len, output_size]
        history_output = torch.relu(history_output)                            # [batch_size, seq_len, output_size]

        history_part = self.intensity(history_output)                          # [batch_size, seq_len, 1]
        if self.limited_history_norm:
            history_part = torch.tanh(history_part)                            # [batch_size, seq_len, 1]

        history_part = torch.exp(history_part)                                 # [batch_size, seq_len, 1]

        constant = (history_part * torch.exp(self.base_intensity(history_output))).unsqueeze(-1)
                                                                               # [batch_size, seq_len, 1, 1]

        time_scalar = self.time_scalar(history_output).unsqueeze(dim = -1)     # [batch_size, seq_len, 1, 1]

        # time_scalar can not be zero.
        time_scalar = self.clamp_time_scalar(time_scalar)                      # [batch_size, seq_len, 1, 1]


        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
        original_time_expand = time_next.unsqueeze(dim = -1) * time_multiplier # [batch_size, seq_len, resolution]
        original_time_expand_normed = original_time_expand / var               # [batch_size, seq_len, resolution]
        expanded_time = original_time_expand_normed.unsqueeze(dim = -2)        # [batch_size, seq_len, 1, resolution]
        
        intensity_events = torch.exp(time_scalar * expanded_time) * constant   # [batch_size, seq_len, 1, resolution]
        integral_events = (intensity_events - constant) / time_scalar          # [batch_size, seq_len, 1, resolution]

        # aggregated timestamp
        batch_size, seq_len, _ = original_time_expand.shape
        timestamp = torch.cat(
            (torch.zeros((batch_size, seq_len, 1), device = self.device), original_time_expand.diff(dim = -1)),
            dim = -1)                                                          # [batch_size, seq_len, resolution]
        
        intensity = rearrange(intensity_events, 'b s ne r -> b s r ne')        # [batch_size, seq_len, resolution, 1]
        integral = rearrange(integral_events, 'b s ne r -> b s r ne')          # [batch_size, seq_len, resolution, 1]

        '''
        Here we start constructing data dict.
        '''
        data = {}
        data['expand_intensity_for_each_event'] = intensity                    # [batch_size, seq_len, resolution, 1]
        data['expand_integral_for_each_event'] = integral                      # [batch_size, seq_len, resolution, 1]


        return data, timestamp