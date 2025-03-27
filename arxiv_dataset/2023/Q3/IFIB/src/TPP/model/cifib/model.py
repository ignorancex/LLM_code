import torch
import torch.nn.functional as F
from einops import repeat, pack
from scipy.stats import spearmanr

from src.TPP.model.cifib.submodel import CIFIB
from src.TPP.model.utils import *
from src.TPP.model.cifib.plot import *


class CIFIBModel(BasicModule):
    '''
    The implementation of IFIB-N(Intensity-free Integral-based MTPP-Numerical).
    We have to call it CIFIB(Continuous-marked IFIB) here because our checkpoints only recognize CIFIB.
    The code of IFIB-C(Intensity-free Integral-based MTPP-Categorical) is in src/TPP/model/ifib.

    Similar to IFIB-C, IFIB-N employs RNN modules as the history encoder. We note that changing the history encoder to
    Transformer should be simple. No experiments data are available for IFIB-N with Transformers.
    '''
    def __init__(self, d_history,
                 d_expression,
                 d_pro_integral,
                 dropout,
                 history_module_layers,
                 mlp_layers,
                 nonlinear,
                 probability_threshold,
                 num_events,
                 continuous_mark_upperbound,
                 continuous_mark_lowerbound,
                 device,
                 sample_resolution = 50,
                 history_module = 'LSTM',
                 denominator_shift = 0.0, pretrain = False, alpha = 0.5, beta = 0.1):
        super(CIFIBModel, self).__init__()
        self.device = device
        self.probability_threshold = probability_threshold
        self.dim_events = num_events
        self.continuous_mark_upperbound = continuous_mark_upperbound
        self.continuous_mark_lowerbound = continuous_mark_lowerbound
        self.sample_resolution = sample_resolution
        self.zero_shift_factor = 1e-12
        self.point_sampling_pattern = torch.tensor([0.00, 0.05, 0.1, 0.25, 0.5, 0.75], device = self.device)

        self.model = CIFIB(d_history = d_history, d_expression = d_expression, d_pro_integral = d_pro_integral, dim_events = num_events,
                          dropout = dropout, history_module = history_module, history_module_layers = history_module_layers,
                          mlp_layers = mlp_layers, denominator_shift = denominator_shift, pretrain = pretrain, alpha = alpha, \
                          beta = beta, continuous_mark_upperbound = continuous_mark_upperbound, 
                          continuous_mark_lowerbound = continuous_mark_lowerbound, sample_resolution = sample_resolution, device = device)


    def divide_history_and_next(self, input):
        '''
        Extract the history and prediction sequences from the input sequence.
        '''
        input_history, input_next = input[:, :-1].clone(), input[:, 1:].clone()
        return input_history, input_next


    def divide_history_and_next_set(self, input_set):
        '''
        Extract the history and prediction sequences from a set of input sequences.
        '''
        input_history_set = []
        input_next_set = []
        for each_input in input_set:
            input_history, input_next = self.divide_history_and_next(each_input)
            input_history_set.append(input_history)
            input_next_set.append(input_next)

        return input_history_set, input_next_set


    def forward(self, input_time, input_events, mask, mean_and_var_events, mean_and_var_time, evaluate):
        '''
        The entrance of different procedures.
        '''
        return self.evaluate_procedure(input_time, input_events, mask, mean_and_var_events, mean_and_var_time) if evaluate \
            else self.train_procedure(input_time, input_events, mask, mean_and_var_events, mean_and_var_time)


    def differentiate_through_cifib_to_get_probability_train(self, time_next, events_next_set, probability_integral):
        '''
        Obtain p(m, t|\mathcal{H}) by backpropagation.
        '''
        probability_for_each_event = - torch.autograd.grad(
            outputs = probability_integral,
            inputs = time_next,
            grad_outputs = torch.ones_like(probability_integral),
            create_graph = True
        )[0]                                                                   # [batch_size, seq_len]

        check_tensor(probability_for_each_event)

        for events_next_set_per_dim in events_next_set:
            probability_for_each_event = - torch.autograd.grad(
                outputs = probability_for_each_event,
                inputs = events_next_set_per_dim,
                grad_outputs = torch.ones_like(probability_for_each_event),
                create_graph = True
            )[0]                                                               # [batch_size, seq_len, 1]
        probability_for_each_event = probability_for_each_event.squeeze(dim = -1)
                                                                               # [batch_size, seq_len]
        check_tensor(probability_for_each_event)                               # [batch_size, seq_len]

        return probability_for_each_event


    def differentiate_through_cifib_to_get_probability_test(self, time_next, events_next_set, probability_integral):
        '''
        Obtain p(m, t|\mathcal{H}) by backpropagation.
        This function might help decrease memory usage during evaluation at the cost of the gradient.
        '''
        probability_for_each_event = - torch.autograd.grad(
            outputs = probability_integral,
            inputs = time_next,
            grad_outputs = torch.ones_like(probability_integral),
            create_graph = True
        )[0]                                                                   # [batch_size, seq_len]

        check_tensor(probability_for_each_event)

        for events_next_set_per_dim in events_next_set:
            probability_for_each_event = - torch.autograd.grad(
                outputs = probability_for_each_event,
                inputs = events_next_set_per_dim,
                grad_outputs = torch.ones_like(probability_for_each_event),
                create_graph = True
            )[0]                                                               # [batch_size, seq_len, 1]
        probability_for_each_event = probability_for_each_event.squeeze(dim = -1)
                                                                               # [batch_size, seq_len]
        probability_for_each_event = probability_for_each_event.detach()       # [batch_size, seq_len]
        check_tensor(probability_for_each_event)

        return probability_for_each_event


    def calculate_euclidan_distance(self, point_set_1, point_set_2):
        '''
        This function calculates the euclidian distance between two point sets.
        '''
        return torch.sqrt(torch.sum(torch.pow(point_set_1 - point_set_2, 2), dim = -1))
                                                                               # [batch_size, seq_len]
        

    def train_procedure(self, input_time, input_events, mask, mean_and_var_events, mean_and_var_time):
        '''
        The forwardpropagation function of the IFIB-N used by train_step()
        '''
        self.train()

        time_history, time_next = self.divide_history_and_next(input_time)     # 2 * [batch_size, seq_len]
        events_history_set, events_next_set = self.divide_history_and_next_set(input_events)
                                                                               # 2 * [batch_size, seq_len, dim_events]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]

        time_next.requires_grad = True
        for idx, _ in enumerate(events_next_set):
            events_next_set[idx].requires_grad = True

        '''
        \int_{D}{p(m_1, m_2, ..., m_n, \tau|\mathcal{H})d\tau}
        '''
        probability_integral_from_t_to_infinite = self.model(events_history_set, events_next_set, \
                                                             time_history, time_next, \
                                                             mean_and_var_events = mean_and_var_events, \
                                                             mean_and_var_time = mean_and_var_time)
                                                                               # [batch_size, seq_len]
        

        check_tensor(probability_integral_from_t_to_infinite)                  # [batch_size, seq_len]

        '''
        Obtain probability values.
        '''
        probability_for_each_event = \
            self.differentiate_through_cifib_to_get_probability_train(time_next, events_next_set, probability_integral_from_t_to_infinite)
                                                                               # [batch_size, seq_len]

        time_next.requires_grad = False
        for idx, _ in enumerate(events_next_set):
            events_next_set[idx].requires_grad = False

        assert probability_for_each_event.shape == probability_integral_from_t_to_infinite.shape

        '''
        We don't calculate event loss of CIFIB during training.
        '''
        loss = self.nll_loss(probability = probability_for_each_event, mask_next = mask_next)
        the_number_of_events = mask_next.sum().item()

        return loss, the_number_of_events


    def evaluate_procedure(self, input_time, input_events, mask, mean_and_var_events, mean_and_var_time):
        '''
        The forwardpropagation function of the IFIB-N used by evaluate_step()
        '''
        self.eval()

        time_history, time_next = self.divide_history_and_next(input_time)     # 2 * [batch_size, seq_len]
        events_history_set, events_next_set = self.divide_history_and_next_set(input_events)
                                                                               # 2 * [batch_size, seq_len, dim_events]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]

        mean_time, var_time = mean_and_var_time
        batch_size, seq_len = time_history.shape
        the_number_of_events = mask_next.sum().item()
        
        mae, pred_time = self.mean_absolute_error(events_history_set = events_history_set, time_history = time_history,\
                                                  time_next = time_next, mask_next = mask_next, mean_and_var_events = mean_and_var_events,\
                                                  mean_and_var_time = mean_and_var_time)
                                                                               # 2 * [batch_size, seq_len]
        mae = mae.sum().item() / the_number_of_events

        time_next.requires_grad = True
        for idx, _ in enumerate(events_next_set):
            events_next_set[idx].requires_grad = True

        '''
        Evaluation 1:
        get the value of p(m_1, m_2, ..., m_t, t) at every event_next and time_next point.
        '''
        probability_integral_from_time_next_to_infinite = self.model(events_history_set, \
                                                                     events_next_set, \
                                                                     time_history, time_next, \
                                                                     mean_and_var_events = mean_and_var_events, \
                                                                     mean_and_var_time = mean_and_var_time)
                                                                               # [batch_size, seq_len]

        check_tensor(probability_integral_from_time_next_to_infinite)          # [batch_size, seq_len]


        probability_from_time_next_to_infinite \
            = self.differentiate_through_cifib_to_get_probability_test(time_next, events_next_set, \
                                                                       probability_integral_from_time_next_to_infinite)
                                                                               # [batch_size, seq_len, resolution ** dim_events]

        for idx, _ in enumerate(events_next_set):
            events_next_set[idx].requires_grad = False
        time_next.requires_grad = False

        time_next_loss = self.nll_loss(probability = probability_from_time_next_to_infinite, mask_next = mask_next)

        '''
        Evalution 2:
        Sample in the high-dimensional continuous marker space. Find which area has the highest probability and report the
        corresponding negative log-likelihood loss.
        '''
        sampled_normed_location = \
            self.point_sampling_pattern * \
            (torch.tensor(self.continuous_mark_upperbound, device = self.device) \
           - torch.tensor(self.continuous_mark_lowerbound, device = self.device)).unsqueeze(dim = -1) + \
            torch.tensor(self.continuous_mark_lowerbound, device = self.device).unsqueeze(dim = -1)
                                                                               # [dim_events, resolution]
        sampled_normed_marks = torch.cartesian_prod(*sampled_normed_location)  # [resolution ** dim_events, dim_events]
        sampled_normed_marks = repeat(sampled_normed_marks, 'rd d -> b s rd d', b = batch_size, s = seq_len)
                                                                               # [batch_size, seq_len, resolution ** dim_events, dim_events]

        sampled_normed_marks_set = sampled_normed_marks.chunk(self.dim_events, dim = -1)
                                                                               # [batch_size, seq_len, resolution ** dim_events, 1] * dim_events
        '''
        Sample p(m_1, m_2, m_3, ..., m_t) with given t.
        '''
        pred_time.requires_grad = True
        for idx, _ in enumerate(sampled_normed_marks_set):
            sampled_normed_marks_set[idx].requires_grad = True

        '''
        \int_{t_{pred}}^{+\infty}{\int_{D}{p(m_1, m_2, ..., m_n, \tau|\mathcal{H})d\tau}}
        '''
        probability_integral_from_pred_time_to_infinite = self.model.sample_evaluate(events_history_set, \
                                                                                     sampled_normed_marks_set, \
                                                                                     time_history, pred_time, \
                                                                                     mean_and_var_events = mean_and_var_events, \
                                                                                     mean_and_var_time = mean_and_var_time)
                                                                               # [batch_size, seq_len, resolution ** dim_events]
        check_tensor(probability_integral_from_pred_time_to_infinite)          # [batch_size, seq_len, resolution ** dim_events]

        '''
        Obtain probability values.
        Please take care of memory usage here.
        '''
        probability_from_pred_time_to_infinite \
            = self.differentiate_through_cifib_to_get_probability_test(pred_time, sampled_normed_marks_set, \
                                                                       probability_integral_from_pred_time_to_infinite)
                                                                               # [batch_size, seq_len, resolution ** dim_events]
        
        pred_time.requires_grad = False
        for idx, _ in enumerate(sampled_normed_marks_set):
            sampled_normed_marks_set[idx].requires_grad = False
        
        '''
        We obtain the predicted coordinate.
        '''
        max_probability, location = probability_from_pred_time_to_infinite.max(dim = -1)
                                                                               # [batch_size, seq_len] * 2
        
        selected_location = F.one_hot(location, num_classes = self.point_sampling_pattern.shape[0] ** self.dim_events)
                                                                               # [batch_size, seq_len, dim_events]
        selected_points = []
        for each_dimension in sampled_normed_marks_set:
            selected_expanded_each_dimension = (each_dimension.squeeze(dim = -1) * selected_location).sum(dim = -1)
                                                                               # [batch_size, seq_len]
            selected_points.append(selected_expanded_each_dimension)
        
        selected_points, _ = pack(selected_points, 'b s *')                    # [batch_size, seq_len, dim_events]
        events_next, _ = pack(events_next_set, 'b s *')                        # [batch_size, seq_len, dim_events]

        restored_selected_points = self.model.events_restore(selected_points, mean_and_var_events)
                                                                               # [batch_size, seq_len, dim_events]

        pred_time_loss = self.nll_loss(probability = max_probability, mask_next = mask_next)
        predicted_distance = self.calculate_euclidan_distance(restored_selected_points, events_next) * mask_next
                                                                               # [batch_size, seq_len]

        return time_next_loss, pred_time_loss, predicted_distance, mae, the_number_of_events


    def nll_loss(self, probability, mask_next):
        '''
        This function calculates the NLL loss at every predicted event.
        '''

        log_probability = - torch.log(probability + self.zero_shift_factor)    # [batch_size, seq_len]
        loss = log_probability * mask_next                                     # [batch_size, seq_len]
        loss = torch.sum(loss)

        return loss


    def mean_absolute_error_and_point_prediction_distance(self, events_history_set, events_next_set, time_history, time_next, mask_next,
                                                          mean_and_var_events, mean_and_var_time):
        '''
        This function calculates the MAE and the Euclidian distance between happened and predicted events.
        This function is called by get_mae_and_distance().
        '''

        batch_size, seq_len = time_history.shape

        mae, pred_time = self.mean_absolute_error(events_history_set, time_history, time_next, \
                                                  mask_next, mean_and_var_events, mean_and_var_time)

        sampled_normed_location = \
            torch.linspace(0, 0.99, self.sample_resolution, device = self.device) * \
            (torch.tensor(self.continuous_mark_upperbound, device = self.device) \
           - torch.tensor(self.continuous_mark_lowerbound, device = self.device)).unsqueeze(dim = -1) + \
            torch.tensor(self.continuous_mark_lowerbound, device = self.device).unsqueeze(dim = -1)
                                                                               # [dim_events, resolution]
        sampled_normed_marks = torch.cartesian_prod(*sampled_normed_location)  # [resolution ** dim_events, dim_events]
        sampled_normed_marks = repeat(sampled_normed_marks, 'rd d -> b s rd d', b = batch_size, s = seq_len)
                                                                               # [batch_size, seq_len, resolution ** dim_events, dim_events]

        sampled_normed_marks_set = sampled_normed_marks.chunk(self.dim_events, dim = -1)
                                                                               # [batch_size, seq_len, resolution ** dim_events, 1] * dim_events
        '''
        Sample p(m_1, m_2, m_3, ..., m_t) with given t.
        '''
        pred_time.requires_grad = True
        for idx, _ in enumerate(sampled_normed_marks_set):
            sampled_normed_marks_set[idx].requires_grad = True

        '''
        \int_{t_{pred}}^{+\infty}{\int_{D}{p(m_1, m_2, ..., m_n, \tau|\mathcal{H})d\tau}}
        '''
        probability_integral_from_pred_time_to_infinite = self.model.sample(events_history_set, \
                                                                            sampled_normed_marks_set, \
                                                                            time_history, pred_time, \
                                                                            mean_and_var_events = mean_and_var_events, \
                                                                            mean_and_var_time = mean_and_var_time)
                                                                               # [batch_size, seq_len, resolution ** dim_events]
        check_tensor(probability_integral_from_pred_time_to_infinite)          # [batch_size, seq_len, resolution ** dim_events]

        '''
        Take care of the memory usage here.
        '''
        probability_from_pred_time_to_infinite \
            = self.differentiate_through_cifib_to_get_probability_test(pred_time, sampled_normed_marks_set, \
                                                                       probability_integral_from_pred_time_to_infinite)
                                                                               # [batch_size, seq_len, resolution ** dim_events]
        
        pred_time.requires_grad = False
        for idx, _ in enumerate(sampled_normed_marks_set):
            sampled_normed_marks_set[idx].requires_grad = False

        '''
        We obtain the predicted coordinate.
        '''
        _, location = probability_from_pred_time_to_infinite.max(dim = -1)     # [batch_size, seq_len] * 2
        
        selected_location = F.one_hot(location, num_classes = self.sample_resolution ** self.dim_events)
                                                                               # [batch_size, seq_len, dim_events]
        selected_points = []
        for each_dimension in sampled_normed_marks_set:
            selected_expanded_each_dimension = (each_dimension.squeeze(dim = -1) * selected_location).sum(dim = -1)
                                                                               # [batch_size, seq_len]
            selected_points.append(selected_expanded_each_dimension)
        
        selected_points, selected_points_ps = pack(selected_points, 'b s *')   # [batch_size, seq_len, dim_events]
        events_next, events_next_ps = pack(events_next_set, 'b s *')           # [batch_size, seq_len, dim_events]

        restored_selected_points = self.model.events_restore(selected_points, mean_and_var_events)
                                                                               # [batch_size, seq_len, dim_events]

        predicted_distance = self.calculate_euclidan_distance(restored_selected_points, events_next) * mask_next
                                                                               # [batch_size, seq_len]
        
        return mae, predicted_distance


    def mean_absolute_error(self, events_history_set, time_history, time_next, mask_next, mean_and_var_events, mean_and_var_time):
        '''
        Use bisect method to predict time for the time-event prediction task.
        '''
        def get_sum_of_integral(integral_from_zero_to_inf, taus):
            '''
            Retrieve the sum of all $ \Lambda^*(m, t) $ over all $ m $ at $ \tau $.
            '''
            probability_integral_from_t_to_inf = self.model.probability_integral_from_t_all_markers(
                events_history_set, time_history, taus, mean_and_var_events, mean_and_var_time)
                                                                               # [batch_size, seq_len]
            # P_m(t) = \int_{0}^{t}{\int_{R}{p(t, m_1, m_2, m_3, ..., m_t|\mathcal{H})}}
            probability_integral = integral_from_zero_to_inf - probability_integral_from_t_to_inf
                                                                               # [batch_size, seq_len]
            return probability_integral

        def bisect_target(integral_from_zero_to_inf, taus):
            return get_sum_of_integral(integral_from_zero_to_inf, taus) - self.probability_threshold
            
        def median_prediction(integral_from_zero_to_inf, l, r):
            for _ in range(50):
                c = (l + r)/2
                v = bisect_target(integral_from_zero_to_inf, c)
                l = torch.where(v < 0, c, l)
                r = torch.where(v >= 0, c, r)

            return (l + r)/2
        
        l = 0.0001*torch.ones_like(time_history, dtype = torch.float32)        # [batch_size, seq_len]
        r = 1e6*torch.ones_like(time_history, dtype = torch.float32)           # [batch_size, seq_len]
        integral_from_zero_to_inf_across_all_events = torch.ones_like(time_history)
                                                                               # [batch_size, seq_len]
                                                                               # [batch_size, seq_len]
        tau_pred = median_prediction(integral_from_zero_to_inf_across_all_events, l, r)
                                                                               # [batch_size, seq_len]
        gap = (tau_pred - time_next) * mask_next                               # [batch_size, seq_len]
        gap = torch.abs(gap)                                                   # [batch_size, seq_len]

        return gap, tau_pred


    def mean_absolute_error_e(self, events_history_set, events_next_set, time_history, time_next, \
                              mask_next, mean_and_var_events, mean_and_var_time):
        '''
        Evaluate model performance on the event-time task.
        '''
        self.eval()

        batch_size, seq_len = time_history.shape
        mean_time, var_time = mean_and_var_time
        mean_events, var_events = mean_and_var_events

        '''
        Evaluation part 1: Find the position where the next event most probably happens.
        '''
        # self.point_sampling_pattern * \
        sampled_normed_location = \
            torch.linspace(0, 0.99, self.sample_resolution, device = self.device) * \
            (torch.tensor(self.continuous_mark_upperbound, device = self.device) \
           - torch.tensor(self.continuous_mark_lowerbound, device = self.device)).unsqueeze(dim = -1) + \
            torch.tensor(self.continuous_mark_lowerbound, device = self.device).unsqueeze(dim = -1)
                                                                               # [dim_events, resolution]
        sampled_normed_marks = torch.cartesian_prod(*sampled_normed_location)  # [resolution ** dim_events, dim_events]
        sampled_normed_marks = repeat(sampled_normed_marks, 'rd d -> b s rd d', b = batch_size, s = seq_len)
                                                                               # [batch_size, seq_len, resolution ** dim_events, dim_events]

        sampled_normed_marks_set = sampled_normed_marks.chunk(self.dim_events, dim = -1)
                                                                               # [batch_size, seq_len, resolution ** dim_events, 1] * dim_events
        '''
        Sample p(m_1, m_2, m_3, ..., m_t) with given t.
        '''
        time_next_zero = torch.zeros_like(time_next)                           # [batch_size, seq_len]
        time_next_zero.requires_grad = True
        for idx, _ in enumerate(sampled_normed_marks_set):
            sampled_normed_marks_set[idx].requires_grad = True

        '''
        \int_{t_{pred}}^{+\infty}{\int_{D}{p(m_1, m_2, ..., m_n, \tau|\mathcal{H})d\tau}}
        '''
        probability_integral_from_pred_time_to_infinite = self.model.sample(events_history_set, \
                                                                            sampled_normed_marks_set, \
                                                                            time_history, time_next_zero, \
                                                                            mean_and_var_events = mean_and_var_events, \
                                                                            mean_and_var_time = mean_and_var_time)
                                                                               # [batch_size, seq_len, resolution ** dim_events]
        check_tensor(probability_integral_from_pred_time_to_infinite)          # [batch_size, seq_len, resolution ** dim_events]

        '''
        Take care of the memory usage here.
        '''
        probability_from_zero_to_infinite \
            = self.differentiate_through_cifib_to_get_probability_test(time_next_zero, sampled_normed_marks_set, \
                                                                       probability_integral_from_pred_time_to_infinite)
                                                                               # [batch_size, seq_len, resolution ** dim_events]
        time_next_zero.requires_grad = False
        for idx, _ in enumerate(sampled_normed_marks_set):
            sampled_normed_marks_set[idx].requires_grad = False

        '''
        We obtain the predicted coordinate.
        '''
        _, location = probability_from_zero_to_infinite.max(dim = -1)          # [batch_size, seq_len] * 2
        
        selected_location = F.one_hot(location, num_classes = self.sample_resolution ** self.dim_events)
                                                                               # [batch_size, seq_len, dim_events]
        selected_points = []
        for each_dimension in sampled_normed_marks_set:
            selected_expanded_each_dimension = (each_dimension.squeeze(dim = -1) * selected_location).sum(dim = -1)
                                                                               # [batch_size, seq_len]
            selected_points.append(selected_expanded_each_dimension)
        
        selected_points, _ = pack(selected_points, 'b s *')                    # [batch_size, seq_len, dim_events]
        events_next, _ = pack(events_next_set, 'b s *')                        # [batch_size, seq_len, dim_events]

        restored_selected_points = self.model.events_restore(selected_points, mean_and_var_events)
                                                                               # [batch_size, seq_len, dim_events]

        distance_between_prediction_and_truth = self.calculate_euclidan_distance(restored_selected_points, events_next)[mask_next == 1]
                                                                               # [batch_size, seq_len]
        
        '''
        Evaluation part 2: decide when the next event at restored_selected_points would happen.
        '''
        delta = sampled_normed_location.diff(dim = -1).mean(dim = -1) / 2      # [dim_events]
        space_point_at_bottom_left = restored_selected_points - delta          # [batch_size, seq_len, dim_events]
        space_point_at_up_right = restored_selected_points + delta             # [batch_size, seq_len, dim_events]
        
        '''
        Obtain the timestamp when the probability of one event happening in [prediction - 1/2 * delta, prediction + 1/2 * delta]
        is bigger than self.probability_threshold.
        '''
        pred_time = self.prediction_with_in_given_event_space(events_history_set, time_history, \
                                                              space_point_at_bottom_left, space_point_at_up_right, \
                                                              mean_and_var_events, mean_and_var_time)
                                                                               # [batch_size, seq_len]
        
        mae_e = torch.abs(time_next - pred_time)[mask_next == 1]               # [batch_size, seq_len]

        return selected_points, pred_time, distance_between_prediction_and_truth, mae_e


    def prediction_with_in_given_event_space(self, events_history_set, time_history, space_point_at_bottom_left, \
                                             space_point_at_up_right, mean_and_var_events, mean_and_var_time):
        '''
        Use bisect method to predict time for the event-time prediction task.
        '''
        def evaluate_in_given_event_space(taus):
            # \int_{tau}^{+\inf}{p(m, \tau|\mathcal{H})d\tau}
            probability_integral_from_t_to_infinite \
                = self.model.probability_integral_from_t_given_marker_space(events_history_set, time_history, taus, \
                                                                            space_point_at_bottom_left, space_point_at_up_right, \
                                                                            mean_and_var_events, mean_and_var_time)
                                                                               # [batch_size, seq_len]
            # \int_{0}^{tau}{p(m, \tau|\mathcal{H})d\tau}
            probability_from_zero_to_t = torch.ones_like(probability_integral_from_t_to_infinite) - probability_integral_from_t_to_infinite
                                                                               # [batch_size, seq_len]
            return probability_from_zero_to_t

        def bisect_target(taus):
            p_t_m = evaluate_in_given_event_space(taus)                        # [batch_size, seq_len]
            p_gap = p_t_m - self.probability_threshold                         # [batch_size, seq_len]

            return p_gap
            
        def median_prediction(l, r):
            for _ in range(50):
                c = (l + r)/2
                v = bisect_target(c)
                l = torch.where(v < 0, c, l)
                r = torch.where(v >= 0, c, r)

            return (l + r)/2

        l = 0.0001*torch.ones_like(time_history, dtype = torch.float32, device = self.device)
                                                                               # [batch_size, seq_len, num_events]
        r = 1e6*torch.ones_like(time_history, dtype = torch.float32, device = self.device)
                                                                               # [batch_size, seq_len, num_events]
        tau_pred = median_prediction(l, r)                                     # [batch_size, seq_len, num_events]

        return tau_pred


    '''
    Plot utilities.
    '''
    def plot(self, minibatch, opt):
        plot_type_to_functions = {
            'intensity': self.intensity,
            'integral': self.integral,
            'probability': self.probability,
            'debug': self.debug
        }
    
        return plot_type_to_functions[opt.plot_type](minibatch, opt)


    def extract_plot_data(self, minibatch):
        '''
        This function extracts input_time, input_events, input_intensity, mask, mean, and var from a minibatch.
        '''
        [time_seq, event_seq, score, mask, padded_intensity], mean_and_var_events, mean_and_var_time = minibatch

        return time_seq, event_seq, score, mask, padded_intensity, mean_and_var_events, mean_and_var_time


    def intensity(self, input_data, opt):
        '''
        Function prober, used by tpp_ploter to draw plots.
        '''

        return NotImplementedError('IFIB-N is intensity-free. Therefore, it can not provide the plot for the intensity function.')


    def integral(self, input_data, opt):
        '''
        Function prober, used by tpp_ploter to draw plots.
        '''
        return NotImplementedError('IFIB-N is intensity-free. Therefore, it can not provide the plot for the intensity integral.')


    def probability(self, input_data, opt):
        '''
        Function prober, used by tpp_ploter to draw plots.
        '''
        self.model.eval()

        input_time, input_events_set, score, mask, input_intensity, mean_and_var_events, mean_and_var_time \
            = self.extract_plot_data(input_data)
        
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history_set, events_next_set = self.divide_history_and_next_set(input_events_set)
                                                                               # [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]

        expand_probability, timestamp = \
            self.model.probing_probability(events_history_set, time_history, time_next, opt.resolution, mean_and_var_events, mean_and_var_time)
                                                                               # 2 * [batch_size, seq_len, resolution]

        data = {
            'time_next': time_next,
            'events_next': torch.zeros_like(time_next),
            'mask_next': mask_next,
            'expand_probability': expand_probability,
            'input_intensity': input_intensity
            }
        plots = plot_probability(data, timestamp, opt)
        return plots


    def debug(self, input_data, opt):
        '''
        We use this function to investigate the property of IFIB-N.
        '''

        return NotImplementedError('')

        self.model.eval()
        data = {}

        input_time, input_events_seq, score, mask, padded_intensity, mean_and_var_events, mean_and_var_time \
            = self.extract_plot_data(input_data)

        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history_set, events_next_set = self.divide_history_and_next_set(input_events_seq)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        batch_size, seq_len = time_history.shape

        mae, distance_between_prediction_and_truth_after_time \
            = self.mean_absolute_error_and_point_prediction_distance(events_history_set, events_next_set, \
                                                                     time_history, time_next, mask_history, mask_next, \
                                                                     mean_and_var_events, mean_and_var_time)
                                                                               # [batch_size, seq_len]
        
        selected_points, pred_time, distance_between_prediction_and_truth_before_time, mae_e \
            = self.mean_absolute_error_e(events_history_set, events_next_set, time_history, time_next, \
                                         mask_next, mean_and_var_events, mean_and_var_time)
                                                                               # [batch_size, seq_len] * 2 + 2 * float

        '''
        Append additional info into the data dict.
        '''
        data['dim_events'] = self.dim_events
        data['events_next'] = torch.zeros_like(time_next)
        data['time_next'] = time_next
        data['mask_next'] = mask_next
        data['distance_between_prediction_and_truth_after_time'] = distance_between_prediction_and_truth_after_time
        data['distance_between_prediction_and_truth_before_time'] = distance_between_prediction_and_truth_before_time
        data['mae_before_event'] = mae
        data['maes_after_event'] = mae_e

        '''
        Show how the probability distribution defined on event space goes along the timeline.
        '''
        sampled_normed_location = \
            torch.linspace(0, 1, self.sample_resolution, device = self.device) * \
            (torch.tensor(self.continuous_mark_upperbound, device = self.device) \
           - torch.tensor(self.continuous_mark_lowerbound, device = self.device)).unsqueeze(dim = -1) + \
            torch.tensor(self.continuous_mark_lowerbound, device = self.device).unsqueeze(dim = -1)
                                                                               # [dim_events, resolution]
        sampled_normed_marks = torch.cartesian_prod(*sampled_normed_location)  # [resolution ** dim_events, dim_events]
        sampled_normed_marks = repeat(sampled_normed_marks, 'rd d -> b s rd d', b = batch_size, s = seq_len)
                                                                               # [batch_size, seq_len, resolution ** dim_events, dim_events]

        sampled_normed_marks_set = sampled_normed_marks.chunk(self.dim_events, dim = -1)
                                                                               # [batch_size, seq_len, resolution ** dim_events, 1] * dim_events
        '''
        Sample p(m_1, m_2, m_3, ..., m_t) with given t.
        '''
        original_expanded_time_next = time_next.unsqueeze(dim = -1) * torch.linspace(0, 1, opt.resolution, device = self.device)
                                                                               # [batch_size, seq_len, resolution]
        original_expanded_time_next_list = original_expanded_time_next.chunk(opt.resolution, dim = -1)
                                                                               # [batch_size, seq_len, 1] * resolution
        
        for idx, _ in enumerate(sampled_normed_marks_set):
            sampled_normed_marks_set[idx].requires_grad = True
        
        probability_probe_at_every_timestamp = []
        for each_selected_timestamp in original_expanded_time_next_list:
            each_selected_timestamp = each_selected_timestamp.squeeze(dim = -1)# [batch_size, seq_len]
            each_selected_timestamp.requires_grad = True

            '''
            \int_{t_{pred}}^{+\infty}{\int_{D}{p(m_1, m_2, ..., m_n, \tau|\mathcal{H})d\tau}}
            '''
            probability_integral_from_pred_time_to_infinite = self.model.sample(events_history_set, \
                                                                                sampled_normed_marks_set, \
                                                                                time_history, each_selected_timestamp, \
                                                                                mean_and_var_events = mean_and_var_events, \
                                                                                mean_and_var_time = mean_and_var_time)
                                                                               # [batch_size, seq_len, resolution ** dim_events]
            check_tensor(probability_integral_from_pred_time_to_infinite)      # [batch_size, seq_len, resolution ** dim_events]
    
            '''
            Take care of the memory usage here.
            '''
            probability_from_pred_time_to_each_selected_timestamp \
                = self.differentiate_through_cifib_to_get_probability_test(each_selected_timestamp, sampled_normed_marks_set, \
                                                                           probability_integral_from_pred_time_to_infinite)
                                                                                   # [batch_size, seq_len, resolution ** dim_events]
            each_selected_timestamp.requires_grad = False
            probability_probe_at_every_timestamp.append(probability_from_pred_time_to_each_selected_timestamp.unsqueeze(dim = -2))
                                                                                   # [batch_size, seq_len, 1, resolution ** dim_events]

        for idx, _ in enumerate(sampled_normed_marks_set):
            sampled_normed_marks_set[idx].requires_grad = False

        # Timestamp part
        zero_inception = torch.zeros((batch_size, seq_len, 1), device = self.device)
        timestamp, timstamp_ps = pack(
            [zero_inception, original_expanded_time_next.diff(dim = -1)],
            'b s *')                                                           # [batch_size, seq_len, resolution]
        
        packed_probed_probability, packed_probed_probability_ps = pack(probability_probe_at_every_timestamp, 'b s * rd')
                                                                               # [batch_size, seq_len, resolution, resolution ** dim_events]
        data['probed_probability_distribution'] = packed_probed_probability    # [batch_size, seq_len, resolution, resolution ** dim_events]
        plots = plot_debug(data, timestamp, opt)

        return plots


    '''
    Evaluation over the entire dataset.
    These functions are called by task functions in plotter.py
    '''
    def get_spearman_and_l1(self, input_data, opt):
        self.model.eval()

        input_time, input_events_set, score, mask, input_intensity, mean_and_var_events, mean_and_var_time \
            = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history_set, events_next_set = self.divide_history_and_next_set(input_events_set)
                                                                               # [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]

        expand_probability, timestamp = \
            self.model.probing_probability(events_history_set, time_history, time_next, opt.resolution, mean_and_var_events, mean_and_var_time)
                                                                               # 2 * [batch_size, seq_len, resolution]
        check_tensor(expand_probability)

        true_probability = expand_true_probability(time_next, input_intensity, opt)
                                                                               # [batch_size, seq_len, resolution] or batch_size * None
        
        expand_probability, true_probability, timestamp = move_from_tensor_to_ndarray(expand_probability, true_probability, timestamp)
        zipped_data = zip(expand_probability, true_probability, timestamp, mask_next)

        spearman = 0
        l1 = 0
        for expand_probability_per_seq, true_probability_per_seq, timestamp_per_seq, mask_next_per_seq in zipped_data:
            seq_len = mask_next_per_seq.sum()

            spearman_per_seq = \
                spearmanr(expand_probability_per_seq[:seq_len, :].flatten(), true_probability_per_seq[:seq_len, :].flatten())[0]

            l1_per_seq = L1_distance_between_two_funcs(
                                        x = true_probability_per_seq[:seq_len, :], y = expand_probability_per_seq[:seq_len, :], \
                                        timestamp = timestamp_per_seq, resolution = opt.resolution
                                        )
            spearman += spearman_per_seq
            l1 += l1_per_seq

        batch_size = mask_next.shape[0]
        spearman /= batch_size
        l1 /= batch_size

        return spearman, l1
    

    def get_mae_and_distance(self, input_data, opt):
        time_seq, event_seq, score, mask, padded_intensity, mean_and_var_events, mean_and_var_time = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(time_seq)       # [batch_size, seq_len]
        events_history_set, events_next_set = self.divide_history_and_next_set(event_seq)
                                                                               # [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]


        mae, distance = self.mean_absolute_error_and_point_prediction_distance(events_history_set, events_next_set,
                                                                               time_history, time_next, mask_next,
                                                                               mean_and_var_events, mean_and_var_time)
                                                                               # [batch_size, seq_len]
        mae, distance = move_from_tensor_to_ndarray(mae, distance)

        return mae, distance

    
    def get_mae_e_and_distance(self, input_data, opt):
        time_seq, event_seq, score, mask, padded_intensity, mean_and_var_events, mean_and_var_time = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(time_seq)     # [batch_size, seq_len]
        events_history_set, events_next_set = self.divide_history_and_next_set(event_seq)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        # The sum of probability over the mark and time space should be 1
        selected_points, pred_time, distance_between_prediction_and_truth, mae_e \
            = self.mean_absolute_error_e(events_history_set, events_next_set, time_history, time_next, \
                                         mask_next, mean_and_var_events, mean_and_var_time)
        
        probability_sum = torch.ones_like(mae_e)

        mae_e, distance_between_prediction_and_truth, probability_sum \
            = move_from_tensor_to_ndarray(mae_e, distance_between_prediction_and_truth, probability_sum)

        return mae_e, distance_between_prediction_and_truth, probability_sum


    def train_step(model, minibatch, device):
        model.train()
        [time_seq, event_seq, score, mask], (mean_events, var_events), (mean_time, var_time) = minibatch
        loss, the_number_of_events = model(         
                input_time = time_seq, input_events = event_seq, mask = mask, \
                mean_and_var_events = (mean_events, var_events), mean_and_var_time = (mean_time, var_time), \
                evaluate = False
        )
        
        loss.backward()
        loss = loss.item() / the_number_of_events
        fact = score.sum().item() / the_number_of_events

        return loss, fact
    

    def evaluation_step(model, minibatch, device):    
        model.eval()
        [time_seq, event_seq, score, mask], (mean_events, var_events), (mean_time, var_time) = minibatch
        time_next_loss, pred_time_loss, predicted_distance, mae, the_number_of_events = model(
                input_time = time_seq, input_events = event_seq, mask = mask, \
                mean_and_var_events = (mean_events, var_events), mean_and_var_time = (mean_time, var_time), \
                evaluate = True
        )
    
        time_next_avg_loss = time_next_loss.item() / the_number_of_events
        pred_time_avg_loss = pred_time_loss.item() / the_number_of_events
        predicted_distance_avg = predicted_distance.sum().item() / the_number_of_events
        fact = score.sum().item() / the_number_of_events
        
        return time_next_avg_loss, pred_time_avg_loss, predicted_distance_avg, mae, fact


    def postprocess(input, procedure):
        def train_postprocess(input):
            '''
            Training process
            [absolute loss, relative loss, events loss]
            '''
            return [input[0], input[0] - input[1]]
        
        def test_postprocess(input):
            '''
            Evaluation process
            [absolute loss, relative loss, events loss, mae value]
            '''
            return [input[0], input[0] - input[4], input[1], input[2], input[3]]
        
        return (train_postprocess(input) if procedure == 'Training' else test_postprocess(input))
    

    def log_print_format(input, procedure):
        def train_log_print_format(input):
            format_dict = {}
            format_dict['absolute_loss'] = input[0]
            format_dict['relative_loss'] = input[1]
            format_dict['num_format'] = {'absolute_loss': ':6.5f', 'relative_loss': ':6.5f'}
            return format_dict

        def test_log_print_format(input):
            format_dict = {}
            format_dict['absolute_loss'] = input[0]
            format_dict['relative_loss'] = input[1]
            format_dict['absolute_loss_at_pred_time'] = input[2]
            format_dict['point_distance'] = input[3]
            format_dict['mae'] = input[4]
            format_dict['num_format'] = {'absolute_loss': ':6.5f', 'relative_loss': ':6.5f',
                                         'absolute_loss_at_pred_time': ':6.5f', 'point_distance': ':2.8f', 
                                         'mae': ':2.8f'}
            return format_dict
        
        return (train_log_print_format(input) if procedure == 'Training' else test_log_print_format(input))

    format_dict_length = 5
    

    def choose_metric(evaluation_report_format_dict, test_report_format_dict):
        '''
        [relative loss on evaluation dataset, relative loss on test dataset, event loss on test dataset]
        '''
        return [evaluation_report_format_dict['absolute_loss'], test_report_format_dict['absolute_loss']], \
               ['evaluation_absolute_loss', 'test_absolute_loss']
    
    metric_number = 2 # metric number is the length of the output of choose_metric