import torch
import numpy as np
from sklearn.metrics import f1_score, top_k_accuracy_score, accuracy_score
from einops import rearrange, repeat, reduce
from scipy.stats import spearmanr

from src.TPP.model.ifib.submodel import IFIB
from src.TPP.model.utils import *
from src.TPP.model.ifib.plot import *


class IFIBModel(BasicModule):
    '''
    The implementation of IFIB-C(Intensity-free Integral-based MTPP-Categorical).
    We have to call it IFIB here because our checkpoints only recognize IFIB, not IFIB-C.
    The code of IFIB-N(Intensity-free Integral-based MTPP-Numerical) is in src/TPP/model/cifib.

    IFIB-C employs RNN modules as the history encoder. We note that changing the history encoder to
    Transformer should be simple. However, our experiments don't show any performance gain by doing that.
    '''
    def __init__(self, d_history,
                 d_intensity,
                 dropout,
                 history_module_layers,
                 mlp_layers,
                 nonlinear,
                 probability_threshold,
                 num_events,
                 device,
                 history_module = 'LSTM',
                 event_toggle = False, additional_event_loss = False,
                 denominator_shift = 0.0, pretrain = False, alpha = 0.5, beta = 0.1):
        '''
        This function creates a IFIB-C model.
        '''
        super(IFIBModel, self).__init__()
        self.device = device
        self.probability_threshold = probability_threshold
        self.num_events = num_events
        self.event_toggle = event_toggle
        self.additional_event_loss = additional_event_loss
        self.zero_shift_factor = 1e-12

        self.model = IFIB(d_history = d_history, d_intensity = d_intensity, num_events = num_events,
                          dropout = dropout, history_module = history_module, history_module_layers = history_module_layers,
                          mlp_layers = mlp_layers, nonlinear = nonlinear, event_toggle = event_toggle,
                          denominator_shift = denominator_shift, pretrain = pretrain, alpha = alpha, beta = beta, device = device)


    def divide_history_and_next(self, input):
        '''
        Extract the history and prediction sequences from the input sequence.
        '''

        input_history, input_next = input[:, :-1].clone(), input[:, 1:].clone()
        return input_history, input_next


    def forward(self, input_time, input_events, mask, mean, var, evaluate):
        '''
        The entrance of different procedures.
        '''
        return self.evaluate_procedure(input_time, input_events, mask, mean, var) if evaluate \
            else self.train_procedure(input_time, input_events, mask, mean, var)


    def train_procedure(self, input_time, input_events, mask, mean, var):
        '''
        The forwardpropagation function of the IFIB-C used by train_step()
        '''

        self.train()

        time_history, time_next = self.divide_history_and_next(input_time)     # 2 * [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # 2 * [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]

        '''
        preparing for multi-event training when needed
        '''
        if self.event_toggle:
            time_next = repeat(time_next, 'b s -> b s ne', ne = self.num_events)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        time_next.requires_grad = True

        '''
        \int_{t}^{+\inf}{p(m, \tau|\mathcal{H})d\tau}
        '''
        probability_integral_from_t_to_infinite = self.model(events_history, time_history, time_next, mean = mean, var = var)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]

        '''
        the value of probability distribution at t, denoted as p(m, t|\mathcal{H}) in our paper.
        '''
        probability_for_each_event = - torch.autograd.grad(
            outputs = probability_integral_from_t_to_infinite,
            inputs = time_next,
            grad_outputs = torch.ones_like(probability_integral_from_t_to_infinite),
            create_graph = True
        )[0]
        time_next.requires_grad = False
        check_tensor(probability_for_each_event)                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        check_tensor(probability_integral_from_t_to_infinite)                  # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        assert probability_for_each_event.shape == probability_integral_from_t_to_infinite.shape

        '''
        We calculate events_loss when evnet_toggle = True
        '''
        events_loss = torch.tensor(0., dtype = torch.float32)
        if self.event_toggle:
            log_probability_for_each_event = torch.log(probability_for_each_event + self.zero_shift_factor)
                                                                               # [batch_size, seq_len, num_events]
            events_probability = torch.nn.functional.softmax(log_probability_for_each_event, dim = -1)
                                                                               # [batch_size, seq_len, num_events]
            events_loss = torch.nn.functional.cross_entropy(rearrange(events_probability, 'b s ne -> b ne s'), \
                                                                      events_next.long(), reduction = 'none')
                                                                               # [batch_size, seq_len]
            events_loss = events_loss * mask_next                              # [batch_size, seq_len]
            events_loss = events_loss.sum()

        '''
        Calculate the NLL loss of p^*(m, t).
        L = -log p(m, t|\mathcal{H})
        '''
        time_loss = self.nll_loss(probability = probability_for_each_event, mask_next = mask_next, events_next = events_next)
        the_number_of_events = mask_next.sum().item()

        loss = time_loss

        return loss, time_loss, events_loss, the_number_of_events


    def evaluate_procedure(self, input_time, input_events, mask, mean, var):
        '''
        The forwardpropagation function of the IFIB-C used by evaluate_step()
        '''

        self.eval()

        time_history, time_next = self.divide_history_and_next(input_time)     # 2 * [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # 2 * [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]
        the_number_of_events = mask_next.sum().item()
        
        mae, pred_time = self.mean_absolute_error(events_history = events_history, time_history = time_history,\
                                                  time_next = time_next, mask_next = mask_next, mean = mean, var = var)
                                                                               # 2 * [batch_size, seq_len]
        mae = mae.sum().item() / the_number_of_events

        '''
        Prepare for multi-event training when needed
        '''
        if self.event_toggle:
            time_next = repeat(time_next, 'b s -> b s ne', ne = self.num_events)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
            pred_time = repeat(pred_time, 'b s -> b s ne', ne = self.num_events)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        time_zero = torch.zeros_like(time_next, device = self.device)          # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]


        time_next.requires_grad = True                                         # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        pred_time.requires_grad = True                                         # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]

        '''
        \int_{t}^{+\inf}{p(m, \tau|\mathcal{H})d\tau}
        '''
        probability_integral_from_zero_to_infinite = self.model(events_history, time_history, time_zero, mean = mean, var = var)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        probability_integral_from_pred_time_to_infinite = self.model(events_history, time_history, pred_time, mean = mean, var = var)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        probability_integral_from_time_next_to_infinite = self.model(events_history, time_history, time_next, mean = mean, var = var)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        '''
        Obtain p(m, t|\mathcal{H}).
        '''
        probability_for_each_event_at_pred_time = - torch.autograd.grad(
            outputs = probability_integral_from_pred_time_to_infinite,
            inputs = pred_time,
            grad_outputs = torch.ones_like(probability_integral_from_pred_time_to_infinite)
        )[0]                                                                   # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]                

        probability_for_each_event_at_time_next = - torch.autograd.grad(
            outputs = probability_integral_from_time_next_to_infinite,
            inputs = time_next,
            grad_outputs = torch.ones_like(probability_integral_from_time_next_to_infinite)
        )[0]                                                                   # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]                

        pred_time.requires_grad = False
        time_next.requires_grad = False

        f1_pred, f1_pred_at_pred_time = 0, 0
        if self.event_toggle:
            '''
            macro-F1 value, we predict events at time_next.
            '''
            events_pred_index = torch.argmax(probability_integral_from_zero_to_infinite, dim = -1)[mask_next == 1]
            events_true = events_next[mask_next == 1]
            events_pred_index, events_true = move_from_tensor_to_ndarray(events_pred_index, events_true)
            f1_pred = f1_score(y_true = events_true, y_pred = events_pred_index, average = 'macro')

            '''
            macro-F1 value, we predict events at pred_time.
            '''
            events_pred_index_at_pred_time = torch.argmax(probability_for_each_event_at_pred_time, dim = -1)[mask_next == 1]
            events_true = events_next[mask_next == 1]
            events_pred_index_at_pred_time, events_true = move_from_tensor_to_ndarray(events_pred_index_at_pred_time, events_true)
            f1_pred_at_pred_time = f1_score(y_true = events_true, y_pred = events_pred_index_at_pred_time, average = 'macro')

            '''
            Event loss at time_next.
            '''
            log_probability_for_each_event_at_time_next = torch.log(probability_for_each_event_at_time_next + self.zero_shift_factor)
                                                                               # [batch_size, seq_len, num_events]
            events_probability = torch.nn.functional.softmax(log_probability_for_each_event_at_time_next, dim = -1)
                                                                               # [batch_size, seq_len, num_events]
            events_loss = torch.nn.functional.cross_entropy(rearrange(events_probability, 'b s ne -> b ne s'), \
                                                                      events_next.long(), reduction = 'none')
                                                                               # [batch_size, seq_len]
            events_loss = events_loss * mask_next                              # [batch_size, seq_len]
            events_loss = events_loss.sum()
    
        time_loss = self.nll_loss(probability = probability_for_each_event_at_time_next, mask_next = mask_next, events_next = events_next)

        return time_loss, events_loss, mae, f1_pred, f1_pred_at_pred_time, the_number_of_events


    def nll_loss(self, probability, events_next, mask_next):
        '''
        This function calculates the NLL loss at every legit event in events_next.
        '''

        if self.event_toggle:
            probability_mask = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
            log_probability = - torch.log(probability + self.zero_shift_factor) * probability_mask
            log_probability = reduce(log_probability, '... ne -> ...', 'sum')  # [batch_size, seq_len]
        else:
            log_probability = - torch.log(probability + self.zero_shift_factor)# [batch_size, seq_len]

        loss = log_probability * mask_next                                     # [batch_size, seq_len]
        loss = torch.sum(loss)

        return loss


    def mean_absolute_error_and_f1(self, events_history, time_history, events_next, time_next, mask_history, mask_next, mean, var):
        '''
        Called by get_mae_and_f1(), this function calculates the MAE and macro-F1 of one minibatch.
        '''

        mae, pred_time = self.mean_absolute_error(events_history, time_history, time_next, mask_next, mean, var)
        if self.event_toggle:
            time_next_pred = repeat(pred_time, 'b s -> b s ne', ne = self.num_events)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        time_next_pred.requires_grad = True                                    # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]

        probability_integral_from_pred_to_infinite = self.model(events_history, time_history, time_next_pred, mean = mean, var = var)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        probability_for_each_event = - torch.autograd.grad(
            outputs = probability_integral_from_pred_to_infinite,
            inputs = time_next_pred,
            grad_outputs = torch.ones_like(probability_integral_from_pred_to_infinite)
        )[0]                                                                   # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]                

        if self.event_toggle:
            events_pred_index = torch.argmax(probability_for_each_event, dim = -1)[mask_next == 1]
            events_true = events_next[mask_next == 1]
            events_pred_index, events_true = move_from_tensor_to_ndarray(events_pred_index, events_true)
            f1 = f1_score(y_true = events_true, y_pred = events_pred_index, average = 'macro')
        
        return mae, f1


    def mean_absolute_error(self, events_history, time_history, time_next, mask_next, mean, var):
        '''
        Use bisect method to predict time for time-event prediction task.
        '''

        def evaluate(integral_from_zero_to_inf, taus):
            if self.event_toggle:
                taus = repeat(taus, 'b s -> b s ne', ne = self.num_events)     # [batch_size, seq_len, num_events]
            probability_integral_from_t_to_inf = self.model(events_history, time_history, taus, mean, var)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
            '''
            P_m(t) = \int_{0}^{t}{p(t|m, \mathcal{H})}
            '''
            probability_integral = integral_from_zero_to_inf - probability_integral_from_t_to_inf
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
            if self.event_toggle:
                probability_integral = reduce(probability_integral, '... ne -> ...', 'sum')
                                                                               # [batch_size, seq_len]
            return probability_integral

        def bisect_target(integral_from_zero_to_inf, taus):
            return evaluate(integral_from_zero_to_inf, taus) - self.probability_threshold
            
        def median_prediction(integral_from_zero_to_inf, l, r):
            for _ in range(50):
                c = (l + r)/2
                v = bisect_target(integral_from_zero_to_inf, c)
                l = torch.where(v < 0, c, l)
                r = torch.where(v >= 0, c, r)

            return (l + r)/2
        
        l = 0.0001*torch.ones_like(time_history, dtype = torch.float32)        # [batch_size, seq_len]
        r = 1e6*torch.ones_like(time_history, dtype = torch.float32)           # [batch_size, seq_len]
        time_next_zero = torch.zeros_like(time_next)                           # [batch_size, seq_len]
        if self.event_toggle:
            time_next_zero = repeat(time_next_zero, 'b s -> b s ne', ne = self.num_events)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        integral_from_zero_to_inf = self.model(events_history, time_history, time_next_zero, mean = mean, var = var)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        tau_pred = median_prediction(integral_from_zero_to_inf, l, r)          # [batch_size, seq_len]
        gap = (tau_pred - time_next) * mask_next                               # [batch_size, seq_len]
        gap = torch.abs(gap)                                                   # [batch_size, seq_len]

        return gap, tau_pred


    def mean_absolute_error_e(self, events_history, events_next, time_history, time_next, mask_next, mean, var):
        '''
        Evaluate IFIB-C performance on the event-time task.

        No max_ and memory_ceiling required here.
        '''

        time_zero = torch.zeros_like(time_next)                                # [batch_size, seq_len]
        # preparing for multi-event training when needed
        if self.event_toggle:
            time_zero = repeat(time_zero, 'b s -> b s ne', ne = self.num_events)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]

        probability_integral_from_zero_to_infinite = \
            self.model(events_history, time_history, time_zero, mean = mean, var = var)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]

        probability_integral_sum = reduce(probability_integral_from_zero_to_infinite, 'b s ne -> b s', 'sum')
                                                                               # [batch_size, seq_len]
        predict_index = torch.argmax(probability_integral_from_zero_to_infinite, dim = -1)
                                                                               # [batch_size, seq_len]

        f1 = []
        top_k_acc = []
        for (events_next_per_seq, probability_integral_per_seq) in zip(events_next, probability_integral_from_zero_to_infinite):
            f1.append(f1_score(y_true = events_next_per_seq.detach().cpu(),
                          y_pred = torch.argmax(probability_integral_per_seq, dim = -1).detach().cpu(), average = 'macro'))
            top_k_acc_single_event_seq = []
            if self.num_events > 2:
                for k in range(1, self.num_events):
                    top_k_acc_single_event_seq.append(
                        top_k_accuracy_score(y_true = events_next_per_seq.detach().cpu(),
                                             y_score = probability_integral_per_seq.detach().cpu(),
                                             k = k,
                                             labels = np.arange(self.num_events))
                    )
            else:
                top_k_acc_single_event_seq.append(
                    accuracy_score(
                        y_true = events_next_per_seq.detach().cpu(),
                        y_pred = torch.argmax(probability_integral_per_seq, dim = -1).detach().cpu()
                    )
                )
                top_k_acc_single_event_seq.append(1.0)
            top_k_acc.append(top_k_acc_single_event_seq)

        predict_index_one_hot = torch.nn.functional.one_hot(predict_index.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        events_next_one_hot = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]

        # step 2: get the time prediction for that kind of event
        tau_pred_all_event = self.prediction_with_all_event_types(events_history, time_history, \
                                                                  probability_integral_from_zero_to_infinite, mean, var)
                                                                               # [batch_size, seq_len, num_events]
        mae_per_event_pure_predict = torch.abs((tau_pred_all_event * predict_index_one_hot).sum(dim = -1) - time_next) * mask_next
                                                                               # [batch_size, seq_len, num_events]
        mae_per_event = torch.abs((tau_pred_all_event * events_next_one_hot).sum(dim = -1) - time_next) * mask_next
                                                                               # [batch_size, seq_len, num_events]

        mae_per_event_pure_predict_avg = torch.sum(mae_per_event_pure_predict, dim = -1) / mask_next.sum(dim = -1)
        mae_per_event_avg = torch.sum(mae_per_event, dim = -1) / mask_next.sum(dim = -1)

        return f1, top_k_acc, probability_integral_sum, tau_pred_all_event, (mae_per_event_pure_predict_avg, mae_per_event_avg), \
               (mae_per_event_pure_predict, mae_per_event)


    def prediction_with_all_event_types(self, events_history, time_history, p_m, mean, var):
        '''
        The time prediction of every marker whose probability is not 0. In fact, considering it is nearly impossible that p(m) = 0, we
        always predict time for all available event types.
        '''

        def evaluate_all_event(taus):
            # \int_{tau}^{+\inf}{p(m, \tau|\mathcal{H})d\tau}
            probability_integral_from_t_to_infinite = self.model(events_history, time_history, taus, mean = mean, var = var)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
            # \int_{0}^{tau}{p(m, \tau|\mathcal{H})d\tau}
            probability_from_zero_to_t = p_m - probability_integral_from_t_to_infinite
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
            return probability_from_zero_to_t

        def bisect_target(taus):
            p_mt = evaluate_all_event(taus)                                    # [batch_size, seq_len, num_events]
            p_t_m = p_mt / p_m                                                 # [batch_size, seq_len, num_events]
            p_gap = p_t_m - self.probability_threshold                         # [batch_size, seq_len, num_events]

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
        r = 1e6*torch.ones((*time_history.shape, self.num_events), dtype = torch.float32, device = self.device)
                                                                               # [batch_size, seq_len, num_events]
        tau_pred = median_prediction(l, r)                                     # [batch_size, seq_len, num_events]

        return tau_pred


    '''
    Plot utility functions
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
        input_time, input_events, _, mask, input_intensity = minibatch[0]
        mean, var = minibatch[1]

        return input_time, input_events, input_intensity, mask, mean, var


    def intensity(self, input_data, opt):
        '''
        Intensity function prober, used by tpp_ploter to draw plots.

        IFIB-C is intensity-free, so this function just throws a NotImplementedError.
        '''

        return NotImplementedError('IFIB-N is intensity-free. Therefore, it can not provide the intensity function.')


    def integral(self, input_data, opt):
        '''
        Intensity integral prober, used by tpp_ploter to draw plots.

        IFIB-C is intensity-free, so this function just throws a NotImplementedError.
        '''
        return NotImplementedError('IFIB-N is intensity-free. Therefore, it can not provide the integral of the intensity function.')


    def probability(self, input_data, opt):
        '''
        probability distribution prober, used by tpp_ploter to draw plots.
        '''

        self.model.eval()

        input_time, input_events, input_intensity, mask, mean, var = self.extract_plot_data(input_data)
        
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]

        expand_probability, timestamp = \
            self.model.probability(events_history, time_history, time_next, opt.resolution, mean, var)
                                                                               # [batch_size, seq_len, resolution, num_events] if we need events else [batch_size, seq_len, resolution] + [batch_size, seq_len, resolution]

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
        We use this function to investigate the property of IFIB-C.
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
        
        f1_2, top_k, probability_sum, tau_pred_all_event, maes_avg, maes \
            = self.mean_absolute_error_e(events_history, events_next, time_history, time_next, mask_next, mean, var)

        data, timestamp = self.model.model_probe_function(events_history, time_history, time_next, opt.resolution, mean, var, mask_next)

        '''
        Append additional info into the data dict.
        '''
        data['events_next'] = events_next
        data['time_next'] = time_next
        data['mask_next'] = mask_next
        data['f1_after_time_pred'] = f1_1
        data['f1_before_time_pred'] = f1_2
        data['top_k'] = top_k
        data['probability_sum'] = probability_sum
        data['tau_pred_all_event'] = tau_pred_all_event
        data['mae_before_event'] = mae
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
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]

        expand_probability, timestamp = \
            self.model.probability(events_history, time_history, time_next, opt.resolution, mean, var)
                                                                               # [batch_size, seq_len, resolution, num_events] if we need events else [batch_size, seq_len, resolution] + [batch_size, seq_len, resolution]

        if opt.event_toggle:
            expand_probability = expand_probability.sum(dim = -1)              # [batch_size, seq_len, resolution]
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
                                        timestamp = timestamp_per_seq, resolution = opt.resolution)
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
            = self.mean_absolute_error_e(events_history, events_next, time_history, time_next, mask_next, mean, var)
        
        _, maes, probability_sum, = move_from_tensor_to_ndarray(*maes, probability_sum)

        return maes, f1_2, probability_sum


    def train_step(model, minibatch, device):
        model.train()
        [time_seq, event_seq, score, mask], (mean, var) = minibatch
        loss, time_loss, events_loss, the_number_of_events = model(         
                input_time = time_seq, input_events = event_seq, mask = mask, \
                    mean = mean, var = var, evaluate = False
        )
        
        loss.backward()
    
        time_loss = time_loss.item() / the_number_of_events
        events_loss = events_loss.item() / the_number_of_events
        fact = score.sum().item() / the_number_of_events
        
        return time_loss, fact, events_loss
    

    def evaluation_step(model, minibatch, device):
        model.eval()
        [time_seq, event_seq, score, mask], (mean, var) = minibatch
        time_loss, events_loss, mae, f1_pred, f1_pred_at_pred_time, the_number_of_events = model(
                input_time = time_seq, input_events = event_seq, mask = mask, evaluate = True,\
                mean = mean, var = var
        )
    
        time_loss = time_loss.item() / the_number_of_events
        events_loss = events_loss.item() / the_number_of_events
        fact = score.sum().item() / the_number_of_events
        
        return time_loss, fact, events_loss, mae, f1_pred, f1_pred_at_pred_time


    def postprocess(input, procedure):
        def train_postprocess(input):
            '''
            Training process
            '''
            return [input[0], input[0] - input[1], input[2]]
        
        def test_postprocess(input):
            '''
            Evaluation process
            '''
            return [input[0], input[0] - input[1], input[2], input[3], input[4], input[5]]
        
        return (train_postprocess(input) if procedure == 'Training' else test_postprocess(input))
    

    def log_print_format(input, procedure):
        def train_log_print_format(input):
            format_dict = {}
            format_dict['absolute_loss'] = input[0]
            format_dict['relative_loss'] = input[1]
            format_dict['events_loss'] = input[2]
            format_dict['num_format'] = {'absolute_loss': ':6.5f', 'relative_loss': ':6.5f', \
                                         'events_loss': ':6.5f'}
            return format_dict

        def test_log_print_format(input):
            format_dict = {}
            format_dict['absolute_loss'] = input[0]
            format_dict['relative_loss'] = input[1]
            format_dict['events_loss'] = input[2]
            format_dict['mae'] = input[3]
            format_dict['f1_event_first'] = input[4]
            format_dict['f1_time_first'] = input[5]
            format_dict['num_format'] = {'absolute_loss': ':6.5f', 'relative_loss': ':6.5f',
                                         'events_loss': ':6.5f', 'mae': ':2.8f', 
                                         'f1_event_first': ':2.8f', 'f1_time_first': ':2.8f'}
            return format_dict
        
        return (train_log_print_format(input) if procedure == 'Training' else test_log_print_format(input))

    format_dict_length = 6
    

    def choose_metric(evaluation_report_format_dict, test_report_format_dict):
        '''
        [relative loss on evaluation dataset, relative loss on test dataset, event loss on test dataset]
        '''
        return [evaluation_report_format_dict['absolute_loss'], test_report_format_dict['absolute_loss']], \
               ['evaluation_absolute_loss', 'test_f1_absolute_loss']
    
    metric_number = 2 # metric number is the length of the output of choose_metric[0]