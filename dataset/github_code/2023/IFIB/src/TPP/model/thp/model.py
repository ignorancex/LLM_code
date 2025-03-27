import torch
import torch.nn.functional as F
from einops import rearrange
from sklearn.metrics import f1_score, top_k_accuracy_score, accuracy_score
import numpy as np

from src.TPP.model import memory_ceiling
from src.TPP.model.thp.plot import *
from src.TPP.model.thp.submodel import THP
from src.TPP.model.utils import *


class THPWrapper(BasicModule):
    '''
    THP(Transformer Hawkes Process), proposed by Zuo et al.
    We reimplement THP with reference to the paper and the original code at https://github.com/SimiaoZuo/Transformer-Hawkes-Process.
    Credits to Zuo et al.!
    '''
    def __init__(self, num_events, device, d_input = 64, d_rnn = 64, d_hidden = 256, n_layers = 3,
                 n_head = 3, d_qk = 64, d_v = 64, dropout = 0.1, beta = 0, probability_threshold = 0.5, 
                 monte_carlo_resolution = 100):
        '''
        This function creates an THP.
        '''
        super(THPWrapper, self).__init__()
        self.device = device
        self.num_events = num_events if num_events > 0 else 1
        self.probability_threshold = probability_threshold
        self.zero_shift = 1e-12

        self.model = THP(num_events = num_events, d_input = d_input, d_rnn = d_rnn, d_hidden = d_hidden, \
                         n_layers = n_layers, n_head = n_head, d_qk = d_qk, d_v = d_v, dropout = dropout, \
                         beta = beta, monte_carlo_resolution = monte_carlo_resolution, device = device)


    def divide_history_and_next(self, input):
        '''
        Extract the history and prediction sequences from the input sequence.
        '''
        input_history, input_next = input[:, :-1].clone(), input[:, 1:].clone()
        return input_history, input_next


    def forward(self, input_time, input_events, mask, evaluate):
        '''
        The entrance of different procedures.
        '''
        return self.evaluate_procedure(input_time, input_events, mask) if evaluate \
            else self.train_procedure(input_time, input_events, mask)


    '''
    Functions for model propagation and evaluation
    '''
    def train_procedure(self, time, events, mask):
        '''
        The forwardpropagation function of the THP used by train_step()
        '''

        time_history, time_next = self.divide_history_and_next(time)           # [batch_size, seq_len] * 2
        events_history, events_next = self.divide_history_and_next(events)     # [batch_size, seq_len] * 2
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len] * 2

        integral_all_events, intensity_all_events \
            = self.model(time_history, time_next, events_history, mask_history, mask_next)
                                                                               # [batch_size, seq_len, num_events]
        '''
        Calculate the NLL loss.
        L = -log \lambda^*(m, t)} + \sum_{n in M}{\Lambda^*(n, t)}
        '''
        neg_log_likeli_loss, marker_loss = self.negative_log_likelihood_and_event_loss(
             intensity_all_events = intensity_all_events, integral_all_events = integral_all_events,\
             events_next = events_next, mask_next = mask_next
        )
        
        '''
        Event loss. This loss should not be counted into the backward loss
        '''
        the_number_of_events = mask_next.sum().item()

        return neg_log_likeli_loss, marker_loss, the_number_of_events


    def evaluate_procedure(self, time, events, mask):
        '''
        The forwardpropagation function of the THP used by evaluate_step()
        '''

        time_history, time_next = self.divide_history_and_next(time)           # [batch_size, seq_len] * 2
        events_history, events_next = self.divide_history_and_next(events)     # [batch_size, seq_len] * 2
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        '''
        Event loss. This loss should not be counted into the backward loss.
        '''
        the_number_of_events = mask_next.sum().item()
        gap, tau_pred = self.mean_absolute_error(time_history, time_next, events_history, mask_history, mask_next)
        gap_mean = gap.sum().item() / the_number_of_events


        integral_all_events_time_next, intensity_all_events_time_next \
            = self.model(time_history, time_next, events_history, mask_history, mask_next)
                                                                               # 2 * [batch_size, seq_len, num_events]
        integral_all_events_pred, intensity_all_events_pred \
            = self.model(time_history, tau_pred, events_history, mask_history, mask_next)
                                                                               # 2 * [batch_size, seq_len, num_events]

        # temporal point process loss
        log_likeli_loss_time_next, marker_loss_time_next = self.negative_log_likelihood_and_event_loss(
             intensity_all_events = intensity_all_events_time_next, integral_all_events = integral_all_events_time_next,\
             events_next = events_next, mask_next = mask_next
        )
        f1_time_next = self.evaluate_f1(intensity_all_events_time_next, events_next, mask_next)
        log_likeli_loss_pred, marker_loss_pred = self.negative_log_likelihood_and_event_loss(
             intensity_all_events = intensity_all_events_pred, integral_all_events = integral_all_events_pred,\
             events_next = events_next, mask_next = mask_next
        )
        f1_pred = self.evaluate_f1(intensity_all_events_pred, events_next, mask_next)


        return log_likeli_loss_pred, log_likeli_loss_time_next, marker_loss_pred, marker_loss_time_next,\
               gap_mean, f1_time_next, f1_pred, the_number_of_events


    def evaluate_f1(self, intensity_all_events, events_next, mask_next):
        events_prediction_probability = torch.log(intensity_all_events + self.zero_shift)
                                                                               # [batch_size, seq_len, num_events]
        events_prediction_probability = F.softmax(events_prediction_probability, dim = -1)
                                                                               # [batch_size, seq_len, num_events]

        pred_events = torch.argmax(events_prediction_probability, dim = -1)[mask_next == 1]
        true_events = events_next[mask_next == 1]
        pred_events, true_events = move_from_tensor_to_ndarray(pred_events, true_events)

        f1 = f1_score(y_pred = pred_events, y_true = true_events, average = 'macro')
        
        return f1


    '''
    Loss function
    '''
    def negative_log_likelihood_and_event_loss(self, intensity_all_events, integral_all_events, events_next, mask_next):
        if events_next is not None:
            type_mask = F.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        else:
            type_mask = 1

        '''
        MTPP loss function
        '''
        intensity = torch.sum(intensity_all_events * type_mask, dim = -1)      # [batch_size, seq_len]

        log_intensity = torch.log(intensity + self.zero_shift) * mask_next     # [batch_size, seq_len]
        intensity_integral = integral_all_events.sum(dim = -1)                 # [batch_size, seq_len]
        ll = -log_intensity + intensity_integral                               # [batch_size, seq_len]
        mtpp_nll_loss = torch.sum(ll)

        '''
        Event loss function. Only for evaluation, do not use this loss as a part of the training loss.
        '''
        events_prediction_probability = torch.log(intensity_all_events + self.zero_shift)
                                                                               # [batch_size, seq_len, num_events]
        events_prediction_probability = F.softmax(events_prediction_probability, dim = -1)
                                                                               # [batch_size, seq_len, num_events]
        events_prediction_probability = rearrange(events_prediction_probability, 'b s ne -> b ne s')
                                                                               # [batch_size, num_events, seq_len]
        events_loss = F.cross_entropy(input = events_prediction_probability, target = events_next.long(), reduction = 'none')
                                                                               # [batch_size, seq_len]
        events_loss = (events_loss * mask_next).sum()

        return mtpp_nll_loss, events_loss


    def mean_absolute_error_and_f1(self, events_history, time_history, events_next, time_next, mask_history, mask_next, mean, var):
        '''
        Called by get_mae_and_f1(), this function calculates the MAE and macro-F1 of one minibatch.
        '''
        gap, pred_time = self.mean_absolute_error(time_history, time_next, events_history, mask_history, mask_next)

        _, intensity_all_events_pred \
            = self.model(time_history, pred_time, events_history, mask_history, mask_next)
                                                                               # 2 * [batch_size, seq_len, num_events]
        f1_pred = self.evaluate_f1(intensity_all_events_pred, events_next, mask_next)
        
        return gap, f1_pred
    

    def mean_absolute_error(self, time_history, time_next, events_history, mask_history, mask_next):
        '''
        Use bisect method to predict time for time-event prediction task.
        '''
        def bisect_target(taus):
            '''
            MTPP loss function
            '''
            integral_all_events, _ = self.model(time_history, taus, events_history, mask_history, mask_next)
                                                                               # [batch_size, seq_len, num_events]
            gap = integral_all_events.sum(dim = -1) + torch.log(1 - torch.tensor(self.probability_threshold, device = self.device))
                                                                               # [batch_size, seq_len]
            return gap

        def median_prediction(l, r):
            for _ in range(50):
                c = (l + r)/2
                v = bisect_target(c)
                l = torch.where(v < 0, c, l)
                r = torch.where(v >= 0, c, r)

            return (l + r)/2
        
        l = 0.0001*torch.ones_like(time_history, dtype = torch.float32)        # [batch_size, seq_len]
        r = 1e6*torch.ones_like(time_history, dtype = torch.float32)           # [batch_size, seq_len]
        tau_pred = median_prediction(l, r)                                     # [batch_size, seq_len]
        gap = (tau_pred - time_next) * mask_next                               # [batch_size, seq_len]
        gap = torch.abs(gap)                                                   # [batch_size, seq_len]

        return gap, tau_pred


    def mean_absolute_error_e(self, time_history, time_next, events_history, events_next, mask_history, mask_next, mean, var):
        '''
        Evaluate model performance on the event-time task.
        '''
        
        self.eval()

        '''
        set a relatively large number as the infinity and decide resolution based on this large value and
        the memory_ceiling.
        '''
        if mean == 0 and var == 1:
            max_ = time_next.mean() + 10 * time_next.var()
        else:
            max_ = mean + 10 * var

        if mean == 0:
            resolution_between_events = max(min(int(time_next.mean().item() // 0.005), 500), 10)
        else:
            resolution_between_events = max(min(int(mean // 0.005), 500), 10)
        
        max_ = min(1e6, max_)
        time_next_inf = torch.ones_like(time_history, device = self.device) * max_
                                                                               # [batch_size, seq_len]
        resolution_inf = max(int(max_ // 0.005), 100)

        # only works when batch_size = 1
        batch_size, seq_len = events_next.shape
        if batch_size * seq_len * resolution_inf * self.num_events > memory_ceiling:
            resolution_inf = int(memory_ceiling // (seq_len * self.num_events * batch_size))
        
        if batch_size * seq_len * resolution_between_events * self.num_events * self.num_events > memory_ceiling:
            resolution_between_events = int(memory_ceiling // (seq_len * self.num_events * self.num_events * batch_size))

        '''
        Step 1: obtain p^*(m) = \int_{t_l}^{+infty}{p(m, t)\dt}
        '''
        expanded_integral_all_events_to_inf, expanded_intensity_all_events_to_inf, timestamp = \
            self.model.integral_intensity_time_next_2d(events_history, time_history, time_next_inf, mask_history, resolution_inf, mean, var)
                                                                               # 2 * [batch_size, seq_len, resolution, num_events]
        '''
        Step 2: provide event predictions
        '''
        probabilty_expanded_events = \
            torch.exp(-expanded_integral_all_events_to_inf.sum(dim = -1, keepdim = True)) * expanded_intensity_all_events_to_inf
                                                                               # [batch_size, seq_len, resolution, num_events]
        probabilty_expanded_events_for_monte_carlo = probabilty_expanded_events[:, :, :-1, :]
                                                                               # [batch_size, seq_len, resolution - 1, num_events]
        expanded_time_gap_for_monte_carlo = timestamp[:, :, 1:].unsqueeze(dim = -1)
                                                                               # [batch_size, seq_len, resolution - 1, 1]
        probability = probabilty_expanded_events_for_monte_carlo * expanded_time_gap_for_monte_carlo
                                                                               # [batch_size, seq_len, resolution - 1, num_events]
        probability = probability.sum(dim = -2)                                # [batch_size, seq_len, num_events]
        probability_integral_sum = probability.sum(dim = -1)                   # [batch_size, seq_len]
        predicted_events = torch.argmax(probability, dim = -1)                 # [batch_size, seq_len]

        '''
        Step 3: calculate macro-F1 and top-K accuracy
        '''
        f1 = []
        top_k_acc = []
        for (ground_truth_per_seq, probability_integral_per_seq) in zip(events_next, probability):
            f1.append(f1_score(y_true = ground_truth_per_seq.detach().cpu(),
                               y_pred = torch.argmax(probability_integral_per_seq, dim = -1).detach().cpu(), average = 'macro'))
            
            # Only available when batch_size = 1
            top_k_acc_per_seq = []
            if self.num_events > 2:
                for k in range(1, self.num_events):
                    top_k_acc_per_seq.append(
                        top_k_accuracy_score(y_true = ground_truth_per_seq.detach().cpu(),
                                             y_score = probability_integral_per_seq.detach().cpu(),
                                             k = k,
                                             labels = np.arange(self.num_events))
                    )
            else:
                top_k_acc_per_seq.append(
                    accuracy_score(
                        y_true = ground_truth_per_seq.detach().cpu(),
                        y_pred = torch.argmax(probability_integral_per_seq, dim = -1).detach().cpu()
                    )
                )
                top_k_acc.append(1.0)
            top_k_acc.append(top_k_acc_per_seq)
        # F1:        [batch_size]
        # top_k_acc: [batch_size, num_events]

        '''
        Step 4: get the time prediction for all, predicted, and real events.
        '''
        tau_pred_all_event = self.prediction_with_all_event_types(events_history, time_history, probability, resolution_between_events, \
                                                                  mask_history, mean, var, max_)
                                                                               # [batch_size, seq_len, num_events]
        predicted_event_mask = F.one_hot(predicted_events.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        event_next_mask = F.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]

        mae_per_event_pure_predict = torch.abs((tau_pred_all_event * predicted_event_mask).sum(dim = -1) - time_next) * mask_next
                                                                               # [batch_size, seq_len]
        mae_per_event = torch.abs((tau_pred_all_event * event_next_mask).sum(dim = -1) - time_next) * mask_next
                                                                               # [batch_size, seq_len]

        mae_per_event_pure_predict_avg = torch.sum(mae_per_event_pure_predict, dim = -1) / mask_next.sum(dim = -1)
        mae_per_event_avg = torch.sum(mae_per_event, dim = -1) / mask_next.sum(dim = -1)
        
        return f1, top_k_acc, probability_integral_sum, tau_pred_all_event, (mae_per_event_pure_predict_avg, mae_per_event_avg), \
               (mae_per_event_pure_predict, mae_per_event)


    def prediction_with_all_event_types(self, events_history, time_history, p_x, resolution, mask_history, mean, var, max_val):
        '''
        The time prediction of every marker whose probability is not 0. In fact, considering it is nearly impossible that p(m) = 0, we
        always predict time for all available event types.
        '''
        def evaluate_all_event(taus):
            expanded_integral_across_events, expanded_intensity_across_events, timestamp = \
                self.model.integral_intensity_time_next_3d(events_history, time_history, taus, mask_history, resolution, mean, var)
                                                                               # 2 * [batch_size, seq_len, num_events, resolution, num_events] + [batch_size, seq_len, num_events, resolution]
            expanded_integral_sum_across_events = expanded_integral_across_events.sum(dim = -1)
                                                                               # [batch_size, seq_len, num_events, resolution]
            intensity_event_mask = torch.diag(torch.ones(self.num_events, device = self.device))
                                                                               # [batch_size, seq_len, num_events, resolution, num_events]
            intensity_event_mask = rearrange(intensity_event_mask, 'ne ne1 -> 1 1 ne 1 ne1')
                                                                               # [batch_size, seq_len, num_events, resolution, num_events]
            expanded_intensity_per_event = (expanded_intensity_across_events * intensity_event_mask).sum(dim = -1)
                                                                               # [batch_size, seq_len, num_events, resolution]
            expanded_probability_per_event = expanded_intensity_per_event * torch.exp(-expanded_integral_sum_across_events)
                                                                               # [batch_size, seq_len, num_events, resolution]

            expanded_probability_per_event_monte_carlo = expanded_probability_per_event[:, :, :, :-1]
                                                                               # [batch_size, seq_len, num_events, resolution - 1]
            timestamp_monte_carlo = timestamp[:, :, :, 1:]                     # [batch_size, seq_len, num_events, resolution - 1]

            probability = (expanded_probability_per_event_monte_carlo * timestamp_monte_carlo).sum(dim = -1)
                                                                               # [batch_size, seq_len, num_events]
            
            return probability

        def bisect_target(taus):
            p_xt = evaluate_all_event(taus)                                    # [batch_size, seq_len, num_events]
            p_t_x = p_xt / p_x                                                 # [batch_size, seq_len, num_events]
            p_gap = p_t_x - self.probability_threshold                         # [batch_size, seq_len, num_events]

            return p_gap

        def median_prediction(l, r):
            for _ in range(50):
                c = (l + r)/2
                v = bisect_target(c)
                l = torch.where(v < 0, c, l)
                r = torch.where(v >= 0, c, r)

            return (l + r)/2
        
        l = 0.0001*torch.ones((*time_history.shape, self.num_events), dtype = torch.float32, device = self.device)
                                                                               # [batch_size, seq_len, num_events]
        r = max_val*torch.ones((*time_history.shape, self.num_events), dtype = torch.float32, device = self.device)
                                                                               # [batch_size, seq_len, num_events]
        tau_pred = median_prediction(l, r)                                     # [batch_size, seq_len, num_events]

        return tau_pred


    '''
    Plot utilities    '''
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
        input_time, input_events, _, mask, input_intensity = minibatch[0]
        mean, var = minibatch[1]

        return input_time, input_events, input_intensity, mask, mean, var


    def intensity(self, input_data, opt):
        '''
        Intensity function prober, used by tpp_ploter to draw plots.
        '''
        self.model.eval()

        input_time, input_events, input_intensity, mask, mean, var = self.extract_plot_data(input_data)
        
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        expand_integral, expand_intensity, timestamp = \
            self.model.integral_intensity_time_next_2d(events_history, time_history, time_next, mask_history, opt.resolution, mean, var)
                                                                               # 3 * [batch_size, seq_len, resolution, num_events]
        
        check_tensor(expand_integral)
        check_tensor(expand_intensity)
        assert expand_intensity.shape == expand_integral.shape

        data = {
            'time_next': time_next,
            'events_next': events_next,
            'mask_next': mask_next,
            'expand_intensity': expand_intensity,
            'input_intensity': input_intensity
            }
        plots = plot_intensity(data, timestamp, opt)
        
        return plots


    def integral(self, input_data, opt):
        '''
        Intensity integral prober, used by tpp_ploter to draw plots.
        '''
        self.model.eval()

        input_time, input_events, input_intensity, mask, mean, var = self.extract_plot_data(input_data)
        
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        expand_integral, expand_intensity, timestamp = \
            self.model.integral_intensity_time_next_2d(events_history, time_history, time_next, mask_history, opt.resolution, mean, var)
                                                                               # 3 * [batch_size, seq_len, resolution, num_events]
        
        check_tensor(expand_integral)
        check_tensor(expand_intensity)
        assert expand_intensity.shape == expand_integral.shape

        data = {
            'time_next': time_next,
            'events_next': events_next,
            'mask_next': mask_next,
            'expand_integral': expand_integral,
            'input_intensity': input_intensity
            }
        plots = plot_integral(data, timestamp, opt)
        return plots


    def probability(self, input_data, opt):
        '''
        probability distribution prober, used by tpp_ploter to draw plots.
        '''
        self.model.eval()

        input_time, input_events, input_intensity, mask, mean, var = self.extract_plot_data(input_data)
        
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        expand_integral, expand_intensity, timestamp = \
            self.model.integral_intensity_time_next_2d(events_history, time_history, time_next, mask_history, opt.resolution, mean, var)
                                                                               # 3 * [batch_size, seq_len, resolution, num_events]

        check_tensor(expand_integral)
        check_tensor(expand_intensity)
        assert expand_intensity.shape == expand_integral.shape
        expand_probability = expand_intensity * torch.exp(-expand_integral.sum(dim = -1, keepdim = True))
                                                                               # [batch_size, seq_len, resolution, num_events]

        data = {
            'time_next': time_next,
            'events_next': events_next,
            'mask_next': mask_next,
            'expand_probability': expand_probability,
            'input_intensity': input_intensity
            }
        plots = plot_probability(data, timestamp, opt)
        return plots


    def debug(self, input_data, opt):
        '''
        We use this function to investigate the property of THP.
        '''
        self.model.eval()

        input_time, input_events, input_intensity, mask, mean, var = self.extract_plot_data(input_data)

        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        mae, f1_1 = self.mean_absolute_error_and_f1(events_history, time_history, events_next, \
                                                    time_next, mask_history, mask_next, mean, var)
                                                                               # [batch_size, seq_len]
        data, timestamp = self.model.model_probe_function(events_history, time_history, time_next, \
                                                          mask_history, mask_next, opt.resolution, mean, var)
        f1_2, top_k, probability_sum, tau_pred_all_event, maes_avg, maes \
            = self.mean_absolute_error_e(time_history, time_next, events_history, events_next, mask_history, mask_next, mean, var)

        '''
        Append additional info into the data dict.
        '''
        data['events_next'] = events_next
        data['time_next'] = time_next
        data['mask_next'] = mask_next
        data['f1_after_time_pred'] = f1_1
        data['mae_before_event'] = mae
        data['f1_before_time_pred'] = f1_2
        data['top_k'] = top_k
        data['probability_sum'] = probability_sum
        data['tau_pred_all_event'] = tau_pred_all_event
        data['maes_after_event_avg'] = maes_avg
        data['maes_after_event'] = maes

        plots = plot_debug(data, timestamp, opt)

        return plots


    '''
    Evaluation over the entire dataset.
    These functions are called by task functions in plotter.py
    '''
    def get_spearman_and_l1(self, input_data, opt):
        input_time, input_events, input_intensity, mask, mean, var = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        expand_integral, expand_intensity, timestamp = \
            self.model.integral_intensity_time_next_2d(events_history, time_history, time_next, mask_history, opt.resolution, mean, var)
                                                                               # 3 * [batch_size, seq_len, resolution, num_events]

        check_tensor(expand_integral)
        check_tensor(expand_intensity)
        assert expand_intensity.shape == expand_integral.shape
        expand_probability = expand_intensity * torch.exp(-expand_integral.sum(dim = -1, keepdim = True))
                                                                               # [batch_size, seq_len, resolution, num_events]
        expand_probability = expand_probability.sum(dim = -1)                  # [batch_size, seq_len, resolution]
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
    

    def get_mae_and_f1(self, input_data, opt):
        input_time, input_events, input_intensity, mask, mean, var = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        mae, f1_1 = self.mean_absolute_error_and_f1(events_history, time_history, events_next, \
                                                    time_next, mask_history, mask_next, mean, var)
                                                                               # [batch_size, seq_len]
        mae = move_from_tensor_to_ndarray(mae)

        return mae, f1_1

    
    def get_mae_e_and_f1(self, input_data, opt):
        input_time, input_events, input_intensity, mask, mean, var = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        f1_2, top_k, probability_sum, tau_pred_all_event, maes_avg, maes \
            = self.mean_absolute_error_e(time_history, time_next, events_history, \
                                         events_next, mask_history, mask_next, mean, var)
        
        _, maes, probability_sum, = move_from_tensor_to_ndarray(*maes, probability_sum)

        return maes, f1_2, probability_sum


    def train_step(model, minibatch, device):
        model.train()

        '''
        Maybe need another function to extract data from minibatches.
        Currently, we don't acquire any prediction loss to assist the model training.  
        '''
        time, events, fact, mask = minibatch[0]                                 # 3 * [batch_size, seq_len + 1, 1] & [batch_size, seq_len, 1]
        neg_log_likeli_loss, marker_loss, the_number_of_events = model(time, events, mask, evaluate = False)
        loss = neg_log_likeli_loss
        loss.backward()

        tpp_loss, mark_loss = neg_log_likeli_loss.item() / the_number_of_events, marker_loss.item() / the_number_of_events
        fact = fact.sum() / the_number_of_events
    
        return tpp_loss, mark_loss, fact


    def evaluation_step(model, minibatch, device):
        model.eval()

        time, events, fact, mask = minibatch[0]                                 # 3 * [batch_size, seq_len + 1, 1] & [batch_size, seq_len, 1]
        log_likeli_loss_pred, log_likeli_loss_time_next, marker_loss_pred, marker_loss_time_next,\
        gap, f1_time_next, f1_pred, the_number_of_events = model(time, events, mask, evaluate = True)

        log_likeli_loss_pred, log_likeli_loss_time_next \
            = log_likeli_loss_pred.item() / the_number_of_events, log_likeli_loss_time_next.item() / the_number_of_events
        marker_loss_pred, marker_loss_time_next \
            = marker_loss_pred.item() / the_number_of_events, marker_loss_time_next.item() / the_number_of_events
        fact = fact.sum().item() / the_number_of_events

        return log_likeli_loss_pred, log_likeli_loss_time_next, marker_loss_pred, marker_loss_time_next, fact, gap, f1_time_next, f1_pred


    def postprocess(input, procedure):
        def train(input):
            return [input[0], input[0] - input[2], input[1]]
        
        def evaluate(input):
            return [input[0], input[1], input[1] - input[4], input[2], input[3], input[5], input[6], input[7]]

        return train(input) if procedure == 'Training' else evaluate(input)


    format_dict_length = 8


    def log_print_format(input, procedure):
        def train(input):
            format_dict = {}
            format_dict['absolute_loss'] = input[0]
            format_dict['relative_loss'] = input[1]
            format_dict['events_loss'] = input[2]
            format_dict['num_format'] = {'absolute_loss': ':8.5f', 'relative_loss': ':8.5f', 'events_loss': ':8.5f'}
            return format_dict
        
        def evaluate(input):
            format_dict = {}
            format_dict['NLL Loss at predicted timestamps'] = input[0]
            format_dict['NLL Loss at given timestamps'] = input[1]
            format_dict['relative NLL Loss at given timestamps'] = input[2]
            format_dict['marker loss at predicted timestamps'] = input[3]
            format_dict['marker loss at given timestamps'] = input[4]
            format_dict['mae'] = input[5]
            format_dict['f1 at given timestamps'] = input[6]
            format_dict['f1 at predicted timestamps'] = input[7]
            format_dict['num_format'] = {
                'NLL Loss at predicted timestamps': ':8.5f',
                'NLL Loss at given timestamps': ':8.5f',
                'relative NLL Loss at given timestamps': ':8.5f',
                'marker loss at predicted timestamps': ':8.5f',
                'marker loss at given timestamps': ':8.5f',
                'mae': ':2.8f',
                'f1 at given timestamps': ':8.5f',
                'f1 at predicted timestamps': ':8.5f'
            }
            return format_dict
        
        return train(input) if procedure == 'Training' else evaluate(input)


    def choose_metric(evaluation_report_format_dict, test_report_format_dict):
        '''
        [relative loss on evaluation dataset, relative loss on test dataset, event loss on test dataset]
        '''
        return [evaluation_report_format_dict['NLL Loss at given timestamps'], test_report_format_dict['NLL Loss at given timestamps']], \
               ['evaluation NLL loss at given time', 'test NLL loss at given time']
    
    metric_number = 2 # metric number is the length of the output of choose_metric[0]