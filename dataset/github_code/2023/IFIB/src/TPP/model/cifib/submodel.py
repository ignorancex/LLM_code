import torch.nn.functional as F
import torch.nn as nn
import torch
from einops import rearrange, repeat, reduce, pack, unpack

from src.TPP.model.cifib.nonneg import NonNegLinear


class CIFIB(nn.Module):
    '''
    The CIFIB, defined on continuous time and marks.
    '''
    def __init__(self, d_history, d_expression, d_pro_integral, dim_events, dropout, history_module,
                 history_module_layers, mlp_layers, continuous_mark_upperbound, continuous_mark_lowerbound, 
                 denominator_shift, sample_resolution, pretrain, alpha, beta, device):
        super(CIFIB, self).__init__()
        self.device = device
        self.dim_events = dim_events
        self.denominator_shift = denominator_shift
        self.sample_resolution = sample_resolution
        self.continuous_mark_upperbound = torch.tensor(continuous_mark_upperbound, device = self.device)
        self.continuous_mark_lowerbound = torch.tensor(continuous_mark_lowerbound, device = self.device)
        self.mask_out_time = torch.tensor([1]*self.dim_events + [0], device = self.device)
                                                                               # [dim_events + 1]

        assert self.continuous_mark_upperbound.shape[0] == self.dim_events, 'unmatched continuous_mark_upperbound with dim_events!'
        assert self.continuous_mark_lowerbound.shape[0] == self.dim_events, 'unmatched continuous_mark_lowerbound with dim_events!'

        '''
        Here we prepare the embedding vectors for continuous markers.
        We consider time as a special continuous marker.
        Embedding continuous historical events into a hidden space by multiplying input values with
        learned weights. Nothing special here.
        '''
        self.embedding_matrix_history = nn.Parameter(torch.zeros(dim_events + 1, d_history, device = self.device))
                                                                               # [dim_events + 1, d_history]
        nn.init.xavier_uniform_(self.embedding_matrix_history)
        self.location_emb_to_single_emb_his = nn.Linear((dim_events + 1) * d_history, d_history, device = self.device)
        
        '''
        Here we expect the model to encode the expression of sub-intergals regarding each dimension into these vectors.
        As these expressions are, in fact, conditional probability distributions, sequential models may apply to these embeddings.
        '''
        self.embedding_base_expression = nn.Parameter(torch.zeros(dim_events + 1, d_expression, device = self.device))
                                                                               # [dim_events + 1, d_history]

        nn.init.xavier_uniform_(self.embedding_base_expression)
        '''
        We use this module to propagate expression infomation among different conditional probability distribution p(x_i|x_{i-1}, ..., x_1)
        '''
        self.let_expression_outer_variables_aware = nn.LSTM(
            input_size = d_expression, hidden_size = d_expression, num_layers = 1, batch_first = True, device = self.device)

        '''
        History embedding module.
        '''
        try:
            self.his_encoder = getattr(nn, history_module)(input_size = d_history, hidden_size = d_history, num_layers = history_module_layers,\
                        batch_first = True, dropout = dropout, device = device)
        except:
            raise Exception(f'Unknown history module {history_module}.')

        '''
        We map the history and current event embedding into the same hidden space.
        '''
        self.history_mapper = nn.Linear(d_history, d_pro_integral, device = self.device)
        self.time_mapper = NonNegLinear(d_expression, d_pro_integral, device = self.device)

        '''
        self.mlp -> self.mlp_time and self.mlp_event
        '''
        self.mlp_time = nn.ModuleList([
            NonNegLinear(d_pro_integral, d_pro_integral, bias = True, device = device) for _ in range(mlp_layers)
        ])

        self.mlp_event = nn.ModuleList([
            NonNegLinear(d_pro_integral, d_pro_integral, bias = True, device = device) for _ in range(mlp_layers)
        ])

        '''
        self.aggregate -> self.aggregate_time and self.aggregate_event
        '''
        self.aggregate_time = NonNegLinear(d_pro_integral, 1, bias = True, device = device)
        self.aggregate_events = NonNegLinear(d_pro_integral, 1, bias = True, device = device)

        self.layer_activation = nn.Tanh()

        # We might need these two factors to control the vector's norm.
        if pretrain:
            # alpha
            self.output_factor_time = nn.Parameter(torch.tensor(alpha, device = self.device, requires_grad = True))
            # beta
            self.residual_factor_time = nn.Parameter(torch.tensor(beta, device = self.device, requires_grad = True))
            # alpha
            self.output_factor_events = nn.Parameter(torch.tensor(alpha, device = self.device, requires_grad = True))
            # beta
            self.residual_factor_events = nn.Parameter(torch.tensor(beta, device = self.device, requires_grad = True))
        else:
            # alpha
            self.output_factor_time = torch.tensor(alpha,  device = self.device)
            # beta
            self.residual_factor_time = torch.tensor(beta, device = self.device)
            # alpha
            self.output_factor_events = torch.tensor(alpha,  device = self.device)
            # beta
            self.residual_factor_events = torch.tensor(beta, device = self.device)

        self.nonneg_activation = nn.Softplus()
        self.nonneg_factor = nn.ReLU()
        self.nonneg_integral = nn.Sigmoid()

    
    '''
    events_normalize() normalizes the input coordinates into [-1, 1]^{n}.
    events_restore() reverses this process.
    '''
    def events_normalize(self, events, mean_and_var_events):
        mean_events, var_events = mean_and_var_events
        events = (events - \
                  rearrange(torch.from_numpy(mean_events).to(self.device), 'de -> 1 1 de')) / \
                  rearrange(torch.from_numpy(var_events).to(self.device), 'de -> 1 1 de')
                                                                               # [batch_size, seq_len, dim_events]
        events = torch.tanh(events)                                            # [batch_size, seq_len, dim_events]
        
        return events


    def events_restore(self, events, mean_and_var_events):
        mean_events, var_events = mean_and_var_events
        restored_events = torch.atanh(events)                                  # [batch_size, seq_len, dim_events]

        restored_events = (restored_events * \
                  rearrange(torch.from_numpy(var_events).to(self.device), 'de -> 1 1 de')) +  \
                  rearrange(torch.from_numpy(mean_events).to(self.device), 'de -> 1 1 de')
                                                                               # [batch_size, seq_len, dim_events]
        
        return restored_events


    def embed_time_and_events_history(self, time, events, mean_and_var_events):
        normed_events = self.events_normalize(events, mean_and_var_events)     # [batch_size, seq_len, dim_events]
        normed_events_and_time, normed_events_and_time_ps = pack([normed_events, time], 'b s *')
                                                                               # [batch_size, seq_len, dim_events + 1]

        emb_normed_events_and_time = self.embedding_matrix_history * normed_events_and_time.unsqueeze(dim = -1)
                                                                               # [batch_size, seq_len, dim_events + 1, d_history]
        emb_normed_events_and_time \
            = self.location_emb_to_single_emb_his(rearrange(emb_normed_events_and_time, 'b s de dh -> b s (de dh)'))
                                                                               # [batch_size, seq_len, d_history]

        return emb_normed_events_and_time


    '''
    These functions combine time and event information into one tensor.
    You should call different embed functions depending on your purpose and the shape of tensor "time" and "events".
    '''
    def embed_time_and_events_nonneg(self, time, events, mean_and_var_events, no_norm = False):
        '''
        When should one use no_norm?
        Input events are the upperbound or lowerbound itself.
        '''
        if no_norm:
            normed_events = events
        else:
            normed_events = self.events_normalize(events, mean_and_var_events) # [batch_size, seq_len, dim_events]

        normed_events_and_time, normed_events_and_time_ps = pack([normed_events, time], 'b s *')
                                                                               # [batch_size, seq_len, dim_events + 1]
        
        fused_embedding_base_expression, _ = self.let_expression_outer_variables_aware(self.embedding_base_expression)
                                                                               # [dim_events + 1, d_expression]
        fused_embedding_base_expression = rearrange(fused_embedding_base_expression, 'd_ev d_ex -> 1 1 d_ev d_ex')
                                                                               # [1, 1, dim_events + 1, d_expression]

        emb_normed_events_and_time = F.softplus(fused_embedding_base_expression) * normed_events_and_time.unsqueeze(dim = -1)
                                                                               # [batch_size, seq_len, dim_events + 1, d_expression]

        return emb_normed_events_and_time


    def embed_time_and_events_nonneg_sample_on_time(self, time, events, mean_and_var_events, no_norm = False):
        '''
        time: [batch_size, seq_len, resolution]
        '''
        resolution = time.shape[-1]

        if no_norm:
            normed_events = events
        else:
            normed_events = self.events_normalize(events, mean_and_var_events) # [batch_size, seq_len, dim_events]

        normed_events = repeat(normed_events, 'b s de -> b s r de', r = resolution)
                                                                               # [batch_size, seq_len, resolution, dim_events]
        normed_events_and_time, normed_events_and_time_ps = pack([normed_events, time], 'b s r *')
                                                                               # [batch_size, seq_len, resolution, dim_events + 1]
        
        fused_embedding_base_expression, _ = self.let_expression_outer_variables_aware(self.embedding_base_expression)
                                                                               # [dim_events + 1, d_expression]
        fused_embedding_base_expression = rearrange(fused_embedding_base_expression, 'd_ev d_ex -> 1 1 d_ev d_ex')
                                                                               # [1, 1, dim_events + 1, d_expression]

        emb_normed_events_and_time = F.softplus(fused_embedding_base_expression) * normed_events_and_time.unsqueeze(dim = -1)
                                                                               # [batch_size, seq_len, dim_events + 1, d_expression]

        return emb_normed_events_and_time


    def embed_time_and_events_nonneg_sample(self, time, events):
        '''
        The input events of this function always get normed elsewhere beforehand.
        '''

        normed_events_and_time, normed_events_and_time_ps = pack([events, time], 'b s sample *')
                                                                               # [batch_size, seq_len, resolution ** dim_events, dim_events + 1]
        
        fused_embedding_base_expression, _ = self.let_expression_outer_variables_aware(self.embedding_base_expression)
                                                                               # [dim_events + 1, d_expression]
        fused_embedding_base_expression = rearrange(fused_embedding_base_expression, 'd_ev d_ex -> 1 1 1 d_ev d_ex')
                                                                               # [1, 1, 1, dim_events + 1, d_expression]

        emb_normed_events_and_time = F.softplus(fused_embedding_base_expression) * normed_events_and_time.unsqueeze(dim = -1)
                                                                               # [batch_size, seq_len, resolution ** dim_events, dim_events + 1, d_expression]

        return emb_normed_events_and_time


    def go_through_nonneg_mlps(self, input_vecs):
        '''
        Toolset function for mapping vectors into nonnegative numbers.
        '''
        '''
        Changes: We separate mlp as mlp_time and mlp_events. Credits to MFullyNN and DHP by Okawa et al.
        '''
        input_vecs_events, input_vecs_time = torch.split(input_vecs, (self.dim_events, 1), dim = -2)
                                                                               # [batch_size, seq_len, dim_events, d_pro_integral] + [batch_size, seq_len, 1, d_pro_integral]
        for layer in self.mlp_time:
            residual_input_vecs_time = input_vecs_time                         # [batch_size, seq_len, 1, d_pro_integral]
            input_vecs_time = layer(input_vecs_time)                           # [batch_size, seq_len, 1, d_pro_integral]
            input_vecs_time = self.layer_activation(input_vecs_time)           # [batch_size, seq_len, 1, d_pro_integral]
            input_vecs_time = self.nonneg_factor(self.residual_factor_time) * residual_input_vecs_time + \
                              self.nonneg_factor(self.output_factor_time) * input_vecs_time
                                                                               # [batch_size, seq_len, 1, d_pro_integral]

        for layer in self.mlp_event:
            residual_input_vecs_events = input_vecs_events                     # [batch_size, seq_len, dim_events, d_pro_integral]
            input_vecs_events = layer(input_vecs_events)                       # [batch_size, seq_len, dim_events, d_pro_integral]
            input_vecs_events = self.layer_activation(input_vecs_events)       # [batch_size, seq_len, dim_events, d_pro_integral]
            input_vecs_events = self.nonneg_factor(self.residual_factor_events) * residual_input_vecs_events + \
                                self.nonneg_factor(self.output_factor_events) * input_vecs_events
                                                                               # [batch_size, seq_len, dim_events, d_pro_integral]
        
        unnormalised_probability_integral_time = self.nonneg_integral(-self.aggregate_time(input_vecs_time))
                                                                               # [batch_size, seq_len, 1, 1]
        unnormalised_probability_integral_events = self.nonneg_integral(-self.aggregate_events(input_vecs_events))
                                                                               # [batch_size, seq_len, dim_events, 1]
        
        unnormalised_probability_integral = torch.cat(
            (unnormalised_probability_integral_events, unnormalised_probability_integral_time),
            dim = -2)                                                          # [batch_size, seq_len, dim_events + 1, 1]
        
        return unnormalised_probability_integral


    def forward(self, events_history_set, events_next_set, time_history, time_next, mean_and_var_events, mean_and_var_time):
        '''
        The forwardpropagation function of IFIB-N.

        Functions whose n rank derivative is always positive always have the output explosion issue with larger inputs.
        Maybe we should create CIBIF as the composition of several IBIFs.

        New thoughts:
        we could decompose the multiple integral into the multiplication of several sub-integrals that could be handled
        by IFIB.
        For example:
        \int_{a_1}^{b_1}{\int_{a_2}^{b_2}{\cdots \int_{a_n}^{b_n}{p(x_1, x_2, x_3, \cdots, x_n)dx_1dx_2dx_3\cdots d_x_n}}}
        = \int_{a_1}^{b_1}{p(x_1)dx_1}\int_{a_2}^{b_2}{p(x_2|x_1)dx_2}\int_{a_3}^{b_3}{p(x_3|x_1, x_2)dx_3}\cdots\int_{a_n}^{b_n}{p(x_n|x_1, x_2, \cdots, x_{n-1})dx_n}

        All continuous dimensions EXCEPT time should be normalized.
        self.mask_out_time is used to exclude time from the explicit normalization.
        '''
        
        '''
        Obtain historical embeddings.
        All history information is embedded into a historical hidden space.
        '''
        events_history, _ = pack(events_history_set, 'b s *')                  # [batch_size, seq_len, dim_events]
        events_next, _ = pack(events_next_set, 'b s *')                        # [batch_size, seq_len, dim_events]

        mean_time, var_time = mean_and_var_time
        batch_size, seq_len = time_history.shape

        time_history = (time_history - mean_time) / var_time                   # [batch_size, seq_len]
        emb_history = self.embed_time_and_events_history(time_history, events_history, mean_and_var_events)
                                                                               # [batch_size, seq_len, d_history]
        
        # Reshape hidden output for full connection layers.
        hidden_history, (_, _) = self.his_encoder(emb_history)                 # [batch_size, seq_len, d_history]
        hidden_history = self.history_mapper(hidden_history)                   # [batch_size, seq_len, d_pro_integral]

        '''
        Obtain embeddings of time_next and event_next. This is where we introduce the multiple integration decomposition.

        Each module should provide a vector representing the expression of \int_{a_i}^{+infty}{p(x_i|x_1, x_2, \cdot, x_{i-1})dx_i}.
        The module has following properties:
        1. As a_i -> +infty, the value should be close to 0(If upper bound is present, As a_i -> (the upper bound), the value should be close to 0).
        2. As a_i -> a_{lower_bound}, the value should be close to 1.
        3. Every module should be monotonically increasing.
        4. In the stage where we aggregate these hidden expressions to obtain the final integral, neither activation functions
        nor fully-connected layers should participate in(avoid differentiating any activation functions or fully-connected layers n times).

        In essence, I just introduce dim_event sigmoid into this architecture. This trick avoids the annoying n-rank positive derivative requirement
        by turning "differentiating 1 function n times" to "differentiating n functions 1 time".
        The smallest function whose n-rank detivative is positive is a n-rank polynomial. Taking it as the activation inevitably leads to output explosion. 
        '''
        time_next_zero = (- mean_time / var_time) * torch.ones_like(time_next) # [batch_size, seq_len]
        time_next = (time_next - mean_time) / var_time                         # [batch_size, seq_len]
        
        '''
        Events embedding for events_next

        Different from time which starts from 0 and never ends, normal continuous marks always have lower and higher bounds.
        Therefore, we need to remove the meaningless integral from upper bound to infinity from emb_events_lower_anchor, otherwise
        CIFIB might introduce wrong normalization factors in the last step.
        '''
        emb_events_next = self.embed_time_and_events_nonneg(time_next, events_next, mean_and_var_events)
                                                                               # [batch_size, seq_len, dim_events + 1, d_expression]
        emb_events_upper = self.embed_time_and_events_nonneg(time_next, \
                                                       repeat(self.continuous_mark_upperbound, 'de -> b s de', b = batch_size, s = seq_len), \
                                                       mean_and_var_events, no_norm = True)
                                                                               # [batch_size, seq_len, dim_events + 1, d_expression]
        emb_events_lower_anchor = self.embed_time_and_events_nonneg(time_next_zero, \
                                                       repeat(self.continuous_mark_lowerbound, 'de -> b s de', b = batch_size, s = seq_len), \
                                                       mean_and_var_events, no_norm = True)
                                                                               # [batch_size, seq_len, dim_events + 1, d_expression]
        emb_events_upper_anchor = self.embed_time_and_events_nonneg(time_next_zero, \
                                                       repeat(self.continuous_mark_upperbound, 'de -> b s de', b = batch_size, s = seq_len), \
                                                       mean_and_var_events, no_norm = True)
                                                                               # [batch_size, seq_len, dim_events + 1, d_expression]

        emb_events_next = self.time_mapper(emb_events_next)                    # [batch_size, seq_len, dim_events + 1, d_pro_integral]
        emb_events_upper = self.time_mapper(emb_events_upper)                  # [batch_size, seq_len, dim_events + 1, d_pro_integral]
        emb_events_lower_anchor = self.time_mapper(emb_events_lower_anchor)    # [batch_size, seq_len, dim_events + 1, d_pro_integral]
        emb_events_upper_anchor = self.time_mapper(emb_events_upper_anchor)    # [batch_size, seq_len, dim_events + 1, d_pro_integral]

        output_events_next = emb_events_next + rearrange(hidden_history, 'b s di -> b s 1 di')
                                                                               # [batch_size, seq_len, dim_events + 1, d_pro_integral]
        output_events_upper = emb_events_upper + rearrange(hidden_history, 'b s di -> b s 1 di')
                                                                               # [batch_size, seq_len, dim_events + 1, d_pro_integral]
        output_events_lower_anchor = emb_events_lower_anchor + rearrange(hidden_history, 'b s di -> b s 1 di')
                                                                               # [batch_size, seq_len, dim_events + 1, d_pro_integral]
        output_events_upper_anchor = emb_events_upper_anchor + rearrange(hidden_history, 'b s di -> b s 1 di')
                                                                               # [batch_size, seq_len, dim_events + 1, d_pro_integral]
        
        unnormalised_probability_integral_output_events_next = self.go_through_nonneg_mlps(output_events_next).squeeze(dim = -1)
                                                                               # [batch_size, seq_len, dim_events + 1]
        unnormalised_probability_integral_output_events_upper = self.go_through_nonneg_mlps(output_events_upper).squeeze(dim = -1)
                                                                               # [batch_size, seq_len, dim_events + 1]
        unnormalised_probability_integral_output_events_lower_anchor = self.go_through_nonneg_mlps(output_events_lower_anchor).squeeze(dim = -1)
                                                                               # [batch_size, seq_len, dim_events + 1]
        unnormalised_probability_integral_output_events_upper_anchor = self.go_through_nonneg_mlps(output_events_upper_anchor).squeeze(dim = -1)
                                                                               # [batch_size, seq_len, dim_events + 1]
        
        unnormalised_probability_integral = unnormalised_probability_integral_output_events_next - unnormalised_probability_integral_output_events_upper * self.mask_out_time
                                                                               # [batch_size, seq_len, dim_events + 1]
        normalised_factor = unnormalised_probability_integral_output_events_lower_anchor - unnormalised_probability_integral_output_events_upper_anchor * self.mask_out_time
                                                                               # [batch_size, seq_len, dim_events + 1]
        
        normalised_subprobability_integrals = unnormalised_probability_integral / (normalised_factor + self.denominator_shift)
                                                                               # [batch_size, seq_len, dim_events + 1]

        normalised_probability_integral = normalised_subprobability_integrals.prod(dim = -1)
                                                                               # [batch_size, seq_len]

        return normalised_probability_integral


    def probability_integral_from_t_all_markers(self, events_history_set, time_history, tau, mean_and_var_events, mean_and_var_time):
        '''
        mean_absolute_error() uses this function to calculate the integral 
        P_m(t) = \int_{0}^{t}{\int_{R}{p(t, m_1, m_2, m_3, ..., m_t|\mathcal{H})}}.
        '''

        '''
        Obtain historical embeddings.
        All history information is embedded into a historical hidden space, the number of whose dimension is d_history.
        '''
        events_history, events_history_ps = pack(events_history_set, 'b s *')  # [batch_size, seq_len, dim_events]

        mean_time, var_time = mean_and_var_time
        batch_size, seq_len = time_history.shape

        time_history = (time_history - mean_time) / var_time                   # [batch_size, seq_len]
        emb_history = self.embed_time_and_events_history(time_history, events_history, mean_and_var_events)
                                                                               # [batch_size, seq_len, d_history]
        
        # Reshape hidden output for full connection layers.
        hidden_history, (_, _) = self.his_encoder(emb_history)                 # [batch_size, seq_len, d_history]
        hidden_history = self.history_mapper(hidden_history)                   # [batch_size, seq_len, d_pro_integral]

        time_next_zero = (- mean_time / var_time) * torch.ones_like(tau)       # [batch_size, seq_len]
        tau = (tau - mean_time) / var_time                                     # [batch_size, seq_len]
        
        emb_events_lower_bound = self.embed_time_and_events_nonneg(tau, \
                                                       repeat(self.continuous_mark_lowerbound, 'de -> b s de', b = batch_size, s = seq_len), \
                                                       mean_and_var_events, no_norm = True)                                                                               # [batch_size, seq_len, dim_events + 1, d_expression]
                                                                               # [batch_size, seq_len, dim_events + 1, d_expression]
        emb_events_upper_bound = self.embed_time_and_events_nonneg(tau, \
                                                       repeat(self.continuous_mark_upperbound, 'de -> b s de', b = batch_size, s = seq_len), \
                                                       mean_and_var_events, no_norm = True)
                                                                               # [batch_size, seq_len, dim_events + 1, d_expression]
        emb_events_lower_anchor = self.embed_time_and_events_nonneg(time_next_zero, \
                                                       repeat(self.continuous_mark_lowerbound, 'de -> b s de', b = batch_size, s = seq_len), \
                                                       mean_and_var_events, no_norm = True)
                                                                               # [batch_size, seq_len, dim_events + 1, d_expression]
        emb_events_upper_anchor = self.embed_time_and_events_nonneg(time_next_zero, \
                                                       repeat(self.continuous_mark_upperbound, 'de -> b s de', b = batch_size, s = seq_len), \
                                                       mean_and_var_events, no_norm = True)
                                                                               # [batch_size, seq_len, dim_events + 1, d_expression]

        emb_events_lower_bound = self.time_mapper(emb_events_lower_bound)      # [batch_size, seq_len, dim_events + 1, d_pro_integral]
        emb_events_upper_bound = self.time_mapper(emb_events_upper_bound)      # [batch_size, seq_len, dim_events + 1, d_pro_integral]
        emb_events_lower_anchor = self.time_mapper(emb_events_lower_anchor)    # [batch_size, seq_len, dim_events + 1, d_pro_integral]
        emb_events_upper_anchor = self.time_mapper(emb_events_upper_anchor)    # [batch_size, seq_len, dim_events + 1, d_pro_integral]

        output_events_lower_bound = emb_events_lower_bound + rearrange(hidden_history, 'b s di -> b s 1 di')
                                                                               # [batch_size, seq_len, dim_events + 1, d_pro_integral]
        output_events_upper_bound = emb_events_upper_bound + rearrange(hidden_history, 'b s di -> b s 1 di')
                                                                               # [batch_size, seq_len, dim_events + 1, d_pro_integral]
        output_events_lower_anchor = emb_events_lower_anchor + rearrange(hidden_history, 'b s di -> b s 1 di')
                                                                               # [batch_size, seq_len, dim_events + 1, d_pro_integral]
        output_events_upper_anchor = emb_events_upper_anchor + rearrange(hidden_history, 'b s di -> b s 1 di')
                                                                               # [batch_size, seq_len, dim_events + 1, d_pro_integral]
        
        unnormalised_probability_integral_output_events_lower_bound = self.go_through_nonneg_mlps(output_events_lower_bound).squeeze(dim = -1)
                                                                               # [batch_size, seq_len, dim_events + 1]
        unnormalised_probability_integral_output_events_upper_bound = self.go_through_nonneg_mlps(output_events_upper_bound).squeeze(dim = -1)
                                                                               # [batch_size, seq_len, dim_events + 1]
        unnormalised_probability_integral_output_events_lower_anchor = self.go_through_nonneg_mlps(output_events_lower_anchor).squeeze(dim = -1)
                                                                               # [batch_size, seq_len, dim_events + 1]
        unnormalised_probability_integral_output_events_upper_anchor = self.go_through_nonneg_mlps(output_events_upper_anchor).squeeze(dim = -1)
                                                                               # [batch_size, seq_len, dim_events + 1]

        unnormalised_probability_integral = unnormalised_probability_integral_output_events_lower_bound - unnormalised_probability_integral_output_events_upper_bound * self.mask_out_time
                                                                               # [batch_size, seq_len, dim_events + 1]
        normalised_factor = unnormalised_probability_integral_output_events_lower_anchor - unnormalised_probability_integral_output_events_upper_anchor * self.mask_out_time
                                                                               # [batch_size, seq_len, dim_events + 1]
        normalised_subprobability_integrals = (unnormalised_probability_integral / (normalised_factor + self.denominator_shift))
                                                                               # [batch_size, seq_len, dim_events + 1]
        normalised_probability_integral = normalised_subprobability_integrals.prod(dim = -1)
                                                                               # [batch_size, seq_len]

        return normalised_probability_integral


    def probability_integral_from_t_given_marker_space(self, events_history_set, time_history, tau, \
                                                       space_point_at_bottom_left, space_point_at_up_right, \
                                                       mean_and_var_events, mean_and_var_time):
        '''
        prediction_with_in_given_event_space() uses this function to calculate the integral
        \int_{tau}^{+\inf}{p(m, \tau|\mathcal{H})d\tau}
        '''

        '''
        Obtain historical embeddings.
        All history information is embedded into a historical hidden space, the number of whose dimension is d_history.
        '''
        events_history, events_history_ps = pack(events_history_set, 'b s *')  # [batch_size, seq_len, dim_events]

        mean_time, var_time = mean_and_var_time
        batch_size, seq_len = time_history.shape

        time_history = (time_history - mean_time) / var_time                   # [batch_size, seq_len]
        emb_history = self.embed_time_and_events_history(time_history, events_history, mean_and_var_events)
                                                                               # [batch_size, seq_len, d_history]
        
        # Reshape hidden output for full connection layers.
        hidden_history, (_, _) = self.his_encoder(emb_history)                 # [batch_size, seq_len, d_history]
        hidden_history = self.history_mapper(hidden_history)                   # [batch_size, seq_len, d_pro_integral]


        time_next_zero = (- mean_time / var_time) * torch.ones_like(tau)       # [batch_size, seq_len]
        tau = (tau - mean_time) / var_time                                     # [batch_size, seq_len]
        
        emb_events_lower_bound = self.embed_time_and_events_nonneg(tau, \
                                                       space_point_at_bottom_left, \
                                                       mean_and_var_events)    # [batch_size, seq_len, dim_events + 1, d_expression]
        emb_events_upper_bound = self.embed_time_and_events_nonneg(tau, \
                                                       space_point_at_up_right, \
                                                       mean_and_var_events)    # [batch_size, seq_len, dim_events + 1, d_expression]
        emb_events_lower_anchor = self.embed_time_and_events_nonneg(time_next_zero, \
                                                       space_point_at_bottom_left, \
                                                       mean_and_var_events)    # [batch_size, seq_len, dim_events + 1, d_expression]
        emb_events_upper_anchor = self.embed_time_and_events_nonneg(time_next_zero, \
                                                       space_point_at_up_right, \
                                                       mean_and_var_events)    # [batch_size, seq_len, dim_events + 1, d_expression]

        emb_events_lower_bound = self.time_mapper(emb_events_lower_bound)      # [batch_size, seq_len, dim_events + 1, d_pro_integral]
        emb_events_upper_bound = self.time_mapper(emb_events_upper_bound)      # [batch_size, seq_len, dim_events + 1, d_pro_integral]
        emb_events_lower_anchor = self.time_mapper(emb_events_lower_anchor)    # [batch_size, seq_len, dim_events + 1, d_pro_integral]
        emb_events_upper_anchor = self.time_mapper(emb_events_upper_anchor)    # [batch_size, seq_len, dim_events + 1, d_pro_integral]

        output_events_lower_bound = emb_events_lower_bound + rearrange(hidden_history, 'b s di -> b s 1 di')
                                                                               # [batch_size, seq_len, dim_events + 1, d_pro_integral]
        output_events_upper_bound = emb_events_upper_bound + rearrange(hidden_history, 'b s di -> b s 1 di')
                                                                               # [batch_size, seq_len, dim_events + 1, d_pro_integral]
        output_events_lower_anchor = emb_events_lower_anchor + rearrange(hidden_history, 'b s di -> b s 1 di')
                                                                               # [batch_size, seq_len, dim_events + 1, d_pro_integral]
        output_events_upper_anchor = emb_events_upper_anchor + rearrange(hidden_history, 'b s di -> b s 1 di')
                                                                               # [batch_size, seq_len, dim_events + 1, d_pro_integral]
        
        unnormalised_probability_integral_output_events_lower_bound = self.go_through_nonneg_mlps(output_events_lower_bound).squeeze(dim = -1)
                                                                               # [batch_size, seq_len, dim_events + 1]
        unnormalised_probability_integral_output_events_upper_bound = self.go_through_nonneg_mlps(output_events_upper_bound).squeeze(dim = -1)
                                                                               # [batch_size, seq_len, dim_events + 1]
        unnormalised_probability_integral_output_events_lower_anchor = self.go_through_nonneg_mlps(output_events_lower_anchor).squeeze(dim = -1)
                                                                               # [batch_size, seq_len, dim_events + 1]
        unnormalised_probability_integral_output_events_upper_anchor = self.go_through_nonneg_mlps(output_events_upper_anchor).squeeze(dim = -1)
                                                                               # [batch_size, seq_len, dim_events + 1]

        unnormalised_probability_integral = unnormalised_probability_integral_output_events_lower_bound - unnormalised_probability_integral_output_events_upper_bound * self.mask_out_time
                                                                               # [batch_size, seq_len, dim_events + 1]
        normalised_factor = unnormalised_probability_integral_output_events_lower_anchor - unnormalised_probability_integral_output_events_upper_anchor * self.mask_out_time
                                                                               # [batch_size, seq_len, dim_events + 1]
        normalised_subprobability_integrals = (unnormalised_probability_integral / (normalised_factor + self.denominator_shift))
                                                                               # [batch_size, seq_len, dim_events + 1]
        normalised_probability_integral = normalised_subprobability_integrals.prod(dim = -1)
                                                                               # [batch_size, seq_len]

        return normalised_probability_integral
    

    '''
    Why do we need two sampling functions, sample_evaluate() in evaluation_procedure() and sample() elsewhere?
    I believe because of bad design by previous me. 
    Sample_evaluate() explicitly expects the number of sampled points is exactly 36(the maximum that allows training IFIB-N on a graphic card with 24GB memory). 
    We do not need this limit in the evaluation stage. However, for some unknown reasons the previous me chose to implement
    another sample function, the sample(), instead of fixing sample_evaluate(). So we have two sampling functions.
    '''
    def sample(self, events_history_set, sampled_points_set, time_history, time_next, mean_and_var_events, mean_and_var_time):
        batch_size, seq_len = time_history.shape
        events_history, events_history_ps = pack(events_history_set, 'b s *')  # [batch_size, seq_len, dim_events]
        sampled_points, sampled_points_ps = pack(sampled_points_set, 'b s rd *')
                                                                               # [batch_size, seq_len, resolution ** dim_events, dim_events]

        mean_time, var_time = mean_and_var_time
        batch_size, seq_len = time_history.shape

        time_history = (time_history - mean_time) / var_time                   # [batch_size, seq_len]
        emb_history = self.embed_time_and_events_history(time_history, events_history, mean_and_var_events)
                                                                               # [batch_size, seq_len, d_history]
        
        hidden_history, (_, _) = self.his_encoder(emb_history)                 # [batch_size, seq_len, d_history]
        hidden_history = self.history_mapper(hidden_history)                   # [batch_size, seq_len, d_pro_integral]

        time_next_zero = - mean_time / var_time * torch.ones_like(time_next)   # [batch_size, seq_len]
        time_next = (time_next - mean_time) / var_time                         # [batch_size, seq_len]
        
        time_next_sampled = repeat(time_next, 'b s -> b s sample', sample = self.sample_resolution ** self.dim_events)
                                                                               # [batch_size, seq_len, resolution ** dim_events]
        emb_events_next_sampled = self.embed_time_and_events_nonneg_sample(time_next_sampled, sampled_points)
                                                                               # [batch_size, seq_len, resolution ** dim_events, dim_events + 1, d_expression]
        emb_events_upper_bound = self.embed_time_and_events_nonneg(time_next, \
                                                       repeat(self.continuous_mark_upperbound, 'de -> b s de', b = batch_size, s = seq_len), \
                                                       mean_and_var_events, no_norm = True)
                                                                               # [batch_size, seq_len, dim_events + 1, d_expression]
        emb_events_lower_anchor = self.embed_time_and_events_nonneg(time_next_zero, \
                                                       repeat(self.continuous_mark_lowerbound, 'de -> b s de', b = batch_size, s = seq_len), \
                                                       mean_and_var_events, no_norm = True)
                                                                               # [batch_size, seq_len, dim_events + 1, d_expression]
        emb_events_upper_anchor = self.embed_time_and_events_nonneg(time_next_zero, \
                                                       repeat(self.continuous_mark_upperbound, 'de -> b s de', b = batch_size, s = seq_len), \
                                                       mean_and_var_events, no_norm = True)
                                                                               # [batch_size, seq_len, dim_events + 1, d_expression]

        emb_events_next_sampled = self.time_mapper(emb_events_next_sampled)    # [batch_size, seq_len, resolution ** dim_events, dim_events + 1, d_pro_integral]
        emb_events_upper_bound = self.time_mapper(emb_events_upper_bound)      # [batch_size, seq_len, dim_events + 1, d_pro_integral]
        emb_events_lower_anchor = self.time_mapper(emb_events_lower_anchor)    # [batch_size, seq_len, dim_events + 1, d_pro_integral]
        emb_events_upper_anchor = self.time_mapper(emb_events_upper_anchor)    # [batch_size, seq_len, dim_events + 1, d_pro_integral]

        output_events_next_sampled = emb_events_next_sampled + rearrange(hidden_history, 'b s di -> b s 1 1 di')
                                                                               # [batch_size, seq_len, resolution ** dim_events, dim_events + 1, d_pro_integral]
        output_events_upper_bound = emb_events_upper_bound + rearrange(hidden_history, 'b s di -> b s 1 di')
                                                                               # [batch_size, seq_len, dim_events + 1, d_pro_integral]
        output_events_lower_anchor = emb_events_lower_anchor + rearrange(hidden_history, 'b s di -> b s 1 di')
                                                                               # [batch_size, seq_len, dim_events + 1, d_pro_integral]
        output_events_upper_anchor = emb_events_upper_anchor + rearrange(hidden_history, 'b s di -> b s 1 di')
                                                                               # [batch_size, seq_len, dim_events + 1, d_pro_integral]

        unnormalised_probability_integral_output_events_next_sampled = self.go_through_nonneg_mlps(output_events_next_sampled).squeeze(dim = -1)
                                                                               # [batch_size, seq_len, resolution ** dim_events, dim_events + 1]
        unnormalised_probability_integral_output_events_upper_bound = self.go_through_nonneg_mlps(output_events_upper_bound).squeeze(dim = -1)
                                                                               # [batch_size, seq_len, dim_events + 1]
        unnormalised_probability_integral_output_events_lower_anchor = self.go_through_nonneg_mlps(output_events_lower_anchor).squeeze(dim = -1)
                                                                               # [batch_size, seq_len, dim_events + 1]
        unnormalised_probability_integral_output_events_upper_anchor = self.go_through_nonneg_mlps(output_events_upper_anchor).squeeze(dim = -1)
                                                                               # [batch_size, seq_len, dim_events + 1]

        unnormalised_probability_integral = unnormalised_probability_integral_output_events_next_sampled - unnormalised_probability_integral_output_events_upper_bound.unsqueeze(dim = -2) * self.mask_out_time
                                                                               # [batch_size, seq_len, resolution ** dim_events, dim_events + 1]
        normalised_factor = unnormalised_probability_integral_output_events_lower_anchor - unnormalised_probability_integral_output_events_upper_anchor * self.mask_out_time
                                                                               # [batch_size, seq_len, dim_events + 1]
        normalised_subprobability_integrals = unnormalised_probability_integral / (normalised_factor + self.denominator_shift).unsqueeze(dim = -2)
                                                                               # [batch_size, seq_len, resolution ** dim_events, dim_events + 1]
        normalised_probability_integral = normalised_subprobability_integrals.prod(dim = -1)
                                                                               # [batch_size, seq_len, resolution ** dim_events]

        return normalised_probability_integral


    def sample_evaluate(self, events_history_set, sampled_points_set, time_history, time_next, mean_and_var_events, mean_and_var_time):
        batch_size, seq_len = time_history.shape
        events_history, events_history_ps = pack(events_history_set, 'b s *')  # [batch_size, seq_len, dim_events]
        sampled_points, sampled_points_ps = pack(sampled_points_set, 'b s rd *')
                                                                               # [batch_size, seq_len, resolution ** dim_events, dim_events]

        mean_time, var_time = mean_and_var_time
        batch_size, seq_len = time_history.shape

        time_history = (time_history - mean_time) / var_time                   # [batch_size, seq_len]
        emb_history = self.embed_time_and_events_history(time_history, events_history, mean_and_var_events)
                                                                               # [batch_size, seq_len, d_history]
        
        hidden_history, (_, _) = self.his_encoder(emb_history)                 # [batch_size, seq_len, d_history]
        hidden_history = self.history_mapper(hidden_history)                   # [batch_size, seq_len, d_pro_integral]

        time_next_zero = - mean_time / var_time * torch.ones_like(time_next)   # [batch_size, seq_len]
        time_next = (time_next - mean_time) / var_time                         # [batch_size, seq_len]
        
        time_next_sampled = repeat(time_next, 'b s -> b s sample', sample = 6 ** self.dim_events)
                                                                               # [batch_size, seq_len, resolution ** dim_events]
        emb_events_next_sampled = self.embed_time_and_events_nonneg_sample(time_next_sampled, sampled_points)
                                                                               # [batch_size, seq_len, resolution ** dim_events, dim_events + 1, d_expression]
        emb_events_upper_bound = self.embed_time_and_events_nonneg(time_next, \
                                                       repeat(self.continuous_mark_upperbound, 'de -> b s de', b = batch_size, s = seq_len), \
                                                       mean_and_var_events, no_norm = True)
                                                                               # [batch_size, seq_len, dim_events + 1, d_expression]
        emb_events_lower_anchor = self.embed_time_and_events_nonneg(time_next_zero, \
                                                       repeat(self.continuous_mark_lowerbound, 'de -> b s de', b = batch_size, s = seq_len), \
                                                       mean_and_var_events, no_norm = True)
                                                                               # [batch_size, seq_len, dim_events + 1, d_expression]
        emb_events_upper_anchor = self.embed_time_and_events_nonneg(time_next_zero, \
                                                       repeat(self.continuous_mark_upperbound, 'de -> b s de', b = batch_size, s = seq_len), \
                                                       mean_and_var_events, no_norm = True)
                                                                               # [batch_size, seq_len, dim_events + 1, d_expression]

        emb_events_next_sampled = self.time_mapper(emb_events_next_sampled)    # [batch_size, seq_len, resolution ** dim_events, dim_events + 1, d_pro_integral]
        emb_events_upper_bound = self.time_mapper(emb_events_upper_bound)      # [batch_size, seq_len, dim_events + 1, d_pro_integral]
        emb_events_lower_anchor = self.time_mapper(emb_events_lower_anchor)    # [batch_size, seq_len, dim_events + 1, d_pro_integral]
        emb_events_upper_anchor = self.time_mapper(emb_events_upper_anchor)    # [batch_size, seq_len, dim_events + 1, d_pro_integral]

        output_events_next_sampled = emb_events_next_sampled + rearrange(hidden_history, 'b s di -> b s 1 1 di')
                                                                               # [batch_size, seq_len, resolution ** dim_events, dim_events + 1, d_pro_integral]
        output_events_upper_bound = emb_events_upper_bound + rearrange(hidden_history, 'b s di -> b s 1 di')
                                                                               # [batch_size, seq_len, dim_events + 1, d_pro_integral]
        output_events_lower_anchor = emb_events_lower_anchor + rearrange(hidden_history, 'b s di -> b s 1 di')
                                                                               # [batch_size, seq_len, dim_events + 1, d_pro_integral]
        output_events_upper_anchor = emb_events_upper_anchor + rearrange(hidden_history, 'b s di -> b s 1 di')
                                                                               # [batch_size, seq_len, dim_events + 1, d_pro_integral]

        unnormalised_probability_integral_output_events_next_sampled = self.go_through_nonneg_mlps(output_events_next_sampled).squeeze(dim = -1)
                                                                               # [batch_size, seq_len, resolution ** dim_events, dim_events + 1]
        unnormalised_probability_integral_output_events_upper_bound = self.go_through_nonneg_mlps(output_events_upper_bound).squeeze(dim = -1)
                                                                               # [batch_size, seq_len, dim_events + 1]
        unnormalised_probability_integral_output_events_lower_anchor = self.go_through_nonneg_mlps(output_events_lower_anchor).squeeze(dim = -1)
                                                                               # [batch_size, seq_len, dim_events + 1]
        unnormalised_probability_integral_output_events_upper_anchor = self.go_through_nonneg_mlps(output_events_upper_anchor).squeeze(dim = -1)
                                                                               # [batch_size, seq_len, dim_events + 1]

        unnormalised_probability_integral = unnormalised_probability_integral_output_events_next_sampled - unnormalised_probability_integral_output_events_upper_bound.unsqueeze(dim = -2) * self.mask_out_time
                                                                               # [batch_size, seq_len, resolution ** dim_events, dim_events + 1]
        normalised_factor = unnormalised_probability_integral_output_events_lower_anchor - unnormalised_probability_integral_output_events_upper_anchor * self.mask_out_time
                                                                               # [batch_size, seq_len, dim_events + 1]
        normalised_subprobability_integrals = unnormalised_probability_integral / (normalised_factor + self.denominator_shift).unsqueeze(dim = -2)
                                                                               # [batch_size, seq_len, resolution ** dim_events, dim_events + 1]
        normalised_probability_integral = normalised_subprobability_integrals.prod(dim = -1)
                                                                               # [batch_size, seq_len, resolution ** dim_events]

        return normalised_probability_integral


    def probing_probability(self, events_history_set, time_history, time_next, resolution, mean_and_var_events, mean_and_var_time):
        '''
        Used by probability() and get_spearman_and_l1() for probing the learned probability distribution.
        Specifically, we probe p(t) = \int_{D}{p(m_1, m_2, \cdots, m_n, t)dm_1 dm_2 \cdots dm_n}.
        '''
        events_history, events_history_ps = pack(events_history_set, 'b s *')  # [batch_size, seq_len, dim_events]

        mean_time, var_time = mean_and_var_time
        batch_size, seq_len = time_history.shape

        time_history = (time_history - mean_time) / var_time                   # [batch_size, seq_len]
        emb_history = self.embed_time_and_events_history(time_history, events_history, mean_and_var_events)
                                                                               # [batch_size, seq_len, d_history]
        
        hidden_history, (_, _) = self.his_encoder(emb_history)                 # [batch_size, seq_len, d_history]
        hidden_history = self.history_mapper(hidden_history)                   # [batch_size, seq_len, d_pro_integral]

        time_next_zero = (- mean_time / var_time) * torch.ones_like(time_next) # [batch_size, seq_len]
        original_expanded_time_next = time_next.unsqueeze(dim = -1) * torch.linspace(0, 1, resolution, device = self.device)
                                                                               # [batch_size, seq_len, resolution]

        original_expanded_time_next.requires_grad = True
        expanded_time_next = (original_expanded_time_next - mean_time) / var_time
                                                                               # [batch_size, seq_len, resolution]

        expanded_emb_events_lower = self.embed_time_and_events_nonneg_sample(expanded_time_next, \
                                                            repeat(self.continuous_mark_lowerbound, 'de -> b s resolution de', b = batch_size, s = seq_len, resolution = resolution))
                                                                               # [batch_size, seq_len, resolution, dim_events + 1, d_expression]
        expanded_emb_events_upper = self.embed_time_and_events_nonneg_sample(expanded_time_next, \
                                                            repeat(self.continuous_mark_upperbound, 'de -> b s resolution de', b = batch_size, s = seq_len, resolution = resolution))
                                                                               # [batch_size, seq_len, resolution, dim_events + 1, d_expression]
        emb_events_lower_anchor = self.embed_time_and_events_nonneg(time_next_zero, \
                                                       repeat(self.continuous_mark_lowerbound, 'de -> b s de', b = batch_size, s = seq_len),
                                                       mean_and_var_events, no_norm = True)
                                                                               # [batch_size, seq_len, dim_events + 1, d_expression]
        emb_events_upper_anchor = self.embed_time_and_events_nonneg(time_next_zero, \
                                                       repeat(self.continuous_mark_upperbound, 'de -> b s de', b = batch_size, s = seq_len),
                                                       mean_and_var_events, no_norm = True)
                                                                               # [batch_size, seq_len, dim_events + 1, d_expression]

        expanded_emb_events_lower = self.time_mapper(expanded_emb_events_lower)# [batch_size, seq_len, resolution, dim_events + 1, d_pro_integral]
        expanded_emb_events_upper = self.time_mapper(expanded_emb_events_upper)# [batch_size, seq_len, resolution, dim_events + 1, d_pro_integral]
        emb_events_lower_anchor = self.time_mapper(emb_events_lower_anchor)    # [batch_size, seq_len, dim_events + 1, d_pro_integral]
        emb_events_upper_anchor = self.time_mapper(emb_events_upper_anchor)    # [batch_size, seq_len, dim_events + 1, d_pro_integral]

        expanded_output_events_lower = expanded_emb_events_lower + rearrange(hidden_history, 'b s di -> b s 1 1 di')
                                                                               # [batch_size, seq_len, resolution, dim_events + 1, d_pro_integral]
        expanded_output_events_upper = expanded_emb_events_upper + rearrange(hidden_history, 'b s di -> b s 1 1 di')
                                                                               # [batch_size, seq_len, resolution, dim_events + 1, d_pro_integral]
        output_events_lower_anchor = emb_events_lower_anchor + rearrange(hidden_history, 'b s di -> b s 1 di')
                                                                               # [batch_size, seq_len, dim_events + 1, d_pro_integral]
        output_events_upper_anchor = emb_events_upper_anchor + rearrange(hidden_history, 'b s di -> b s 1 di')
                                                                               # [batch_size, seq_len, dim_events + 1, d_pro_integral]

        expanded_unnormalised_probability_integral_lower = self.go_through_nonneg_mlps(expanded_output_events_lower).squeeze(dim = -1)
                                                                               # [batch_size, seq_len, resolution, dim_events + 1]
        expanded_unnormalised_probability_integral_upper = self.go_through_nonneg_mlps(expanded_output_events_upper).squeeze(dim = -1)
                                                                               # [batch_size, seq_len, resolution, dim_events + 1]
        expanded_unnormalised_probability_integral_output_events_lower_anchor = self.go_through_nonneg_mlps(output_events_lower_anchor).squeeze(dim = -1)
                                                                               # [batch_size, seq_len, dim_events + 1]
        expanded_unnormalised_probability_integral_output_events_upper_anchor = self.go_through_nonneg_mlps(output_events_upper_anchor).squeeze(dim = -1)
                                                                               # [batch_size, seq_len, dim_events + 1]
        
        expanded_unnormalised_probability_integral = expanded_unnormalised_probability_integral_lower - expanded_unnormalised_probability_integral_upper * self.mask_out_time
                                                                               # [batch_size, seq_len, resolution, dim_events + 1]
        normalise_factor = expanded_unnormalised_probability_integral_output_events_lower_anchor - expanded_unnormalised_probability_integral_output_events_upper_anchor * self.mask_out_time
                                                                               # [batch_size, seq_len, dim_events + 1]

        expanded_probability_integral = expanded_unnormalised_probability_integral / (normalise_factor + self.denominator_shift).unsqueeze(dim = -2)
                                                                               # [batch_size, seq_len, resolution, dim_events + 1]
        expanded_probability_integral = expanded_probability_integral.prod(dim = -1)
                                                                               # [batch_size, seq_len, resolution]
        
        '''
        Obtains the probability over timeline.
        '''
        expanded_probability = - torch.autograd.grad(
            outputs = expanded_probability_integral,
            inputs = original_expanded_time_next,
            grad_outputs = torch.ones_like(expanded_probability_integral)
        )[0]                                                                   # [batch_size, seq_len, resolution]

        original_expanded_time_next.requires_grad = False

        # Timestamp part
        batch_size, seq_len = time_history.shape
        zero_inception = torch.zeros((batch_size, seq_len, 1), device = self.device)
        timestamp, timstamp_ps = pack(
            [zero_inception, original_expanded_time_next.diff(dim = -1)],
            'b s *')                                                           # [batch_size, seq_len, resolution]


        return expanded_probability, timestamp