from sklearn.metrics import f1_score
import torch
from scipy.stats import spearmanr

from src.TPP.model.utils import *
from src.TPP.model.rmtpp.rmtpp import RMTPPModule
from src.TPP.model.rmtpp.plot import *


class RMTPP(BasicModule):
    '''
    Our reproduction of RMTPP by Du et al.
    '''
    def __init__(self, device, input_size, hidden_size, history_encoder_layers, dropout, num_events, event_toggle, 
                 output_size, limited_history_norm, time_scalar_min = 1e-4, 
                 probability_threshold = 0.5):
        super(RMTPP, self).__init__()
        self.device = device
        self.num_events = num_events
        self.event_toggle = event_toggle
        self.limited_history_norm = limited_history_norm
        self.probability_threshold = probability_threshold
        self.zero_shift = 1e-12

        self.model = RMTPPModule(input_size = input_size, hidden_size = hidden_size, history_encoder_layers = history_encoder_layers, 
                                 dropout = dropout, num_events = num_events, output_size = output_size, event_toggle = event_toggle, 
                                 limited_history_norm = limited_history_norm, time_scalar_min = time_scalar_min, device = device)


    def divide_history_and_next(self, input):
        '''
        Extract the history and prediction sequences from the input sequence.
        '''
        history, next = input[:, :-1].clone(), input[:, 1:].clone()
        return history, next                                                   # [batch_size, seq_len, 1] or [batch_size, seq_len]


    def forward(self, input_time, input_events, mask, mean, var, evaluate):
        '''
        The entrance of different procedures.
        '''
        return self.evaluate_procedure(input_time, input_events, mask, mean, var) if evaluate \
            else self.train_procedure(input_time, input_events, mask, mean, var)


    def train_procedure(self, events, time, mask, mean, var):
        '''
        The forwardpropagation function of the RMTPP used by train_step()
        '''

        self.train()

        events_history, events_next = self.divide_history_and_next(events)     # [batch_size, seq_len]
        time_history, time_next = self.divide_history_and_next(time)           # [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]

        integral, intensity, mark, constant = self.model(events_history, time_history, time_next, mean, var)
                                                                               # [batch_size, seq_len, 1] * 2, [batch_size, seq_length, num_events], and [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len, 1]

        check_tensor(intensity)
        check_tensor(integral)

        loss, time_loss, events_loss, the_number_of_events = \
                   self.loss_function(intensity, integral, mark, events_next, mask_next)

        return loss, time_loss, events_loss, the_number_of_events, constant


    def evaluate_procedure(self, events, time, mask, mean, var):
        '''
        The forwardpropagation function of the RMTPP used by evaluate_step()
        '''

        self.eval()

        events_history, events_next = self.divide_history_and_next(events)     # [batch_size, seq_len]
        time_history, time_next = self.divide_history_and_next(time)           # [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]

        '''
        Calculating MAE here.
        '''
        mae, pred_time = self.mean_absolute_error(events_history, time_history, time_next, mask_next, mean, var)
                                                                               # [batch_size, seq_len] * 2

        integral_time_next, intensity_time_next, mark_time_next, constant_time_next \
                   = self.model(events_history, time_history, time_next, mean, var)
                                                                               # [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len, 1] * 2, [batch_size, seq_length, num_events], and [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len, 1]
        integral_pred_time, intensity_pred_time, mark_pred_time, constant_pred_time \
                   = self.model(events_history, time_history, pred_time, mean, var)
                                                                               # [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len, 1] * 2, [batch_size, seq_length, num_events], and [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len, 1]
        check_tensor(intensity_time_next)
        check_tensor(integral_time_next)
        check_tensor(intensity_pred_time)
        check_tensor(integral_pred_time)
        
        loss_time_next, time_loss_time_next, events_loss_time_next, the_number_of_events = \
                   self.loss_function(intensity_time_next, integral_time_next, mark_time_next, events_next, mask_next)
        loss_pred_time, time_loss_pred_time, events_loss_pred_time, the_number_of_events = \
                   self.loss_function(intensity_pred_time, integral_pred_time, mark_pred_time, events_next, mask_next)

        predicted_events = torch.argmax(mark_pred_time, dim = -1)[mask_next == 1]
        events_true = events_next[mask_next == 1]
        predicted_events, events_true = move_from_tensor_to_ndarray(predicted_events, events_true)
                                                                       # [batch_size, seq_len] * 2
        f1 = f1_score(y_pred = predicted_events, y_true = events_true, average = 'macro')

        return loss_time_next, time_loss_time_next, events_loss_time_next, loss_pred_time, time_loss_pred_time, events_loss_pred_time, \
               mae, f1, the_number_of_events, constant_time_next, constant_pred_time


    def loss_function(self, intensity, integral, mark, events_next, mask_next):
        '''
        This function calculates the NLL loss at every legit event in events_next.
        '''

        loss = 0
        time_loss, events_loss = 0, 0
        if self.event_toggle:
            events_loss = \
                torch.nn.functional.cross_entropy(input = mark.transpose(1, 2), \
                                                  target = events_next.long(), \
                                                  reduction = 'none')          # [batch_size, seq_len]
            events_loss = events_loss * mask_next
            events_loss = events_loss.sum()
        else:
            events_loss = torch.tensor(0., device = self.device)

        time_loss = -torch.log(intensity + self.zero_shift) + integral         # [batch_size, seq_len]
        time_loss = time_loss * mask_next
        time_loss = time_loss.sum()

        loss = time_loss + events_loss

        return loss, time_loss, events_loss, mask_next.sum().item()


    def mean_absolute_error_and_f1(self, events_history, time_history, events_next, time_next, mask_history, mask_next, mean, var):
        '''
        Called by get_mae_and_f1(), this function calculates the MAE and macro-F1 of one minibatch.
        '''

        mae, pred_time = self.mean_absolute_error(events_history, time_history, time_next, mask_next, mean, var)
        integral, intensity, mark, constant = self.model(events_history, time_history, pred_time, mean, var)

        predicted_events = torch.argmax(mark, dim = -1)[mask_next == 1]        # [batch_size, seq_len]
        events_true = events_next[mask_next == 1]                              # [batch_size, seq_len]

        predicted_events, events_true = move_from_tensor_to_ndarray(predicted_events, events_true)
        f1 = f1_score(y_pred = predicted_events, y_true = events_true, average = 'macro')

        return mae, f1


    def mean_absolute_error(self, events_history, time_history, time_next, mask_next, mean, var):
        '''
        Use bisect method to predict time for time-event prediction task.
        '''
        def evaluate(taus):
            integral, _, _, _ = self.model(events_history, time_history, taus, mean, var)
                                                                               # [batch_size, seq_len]

            return integral

        def bisect_target(taus):
            return evaluate(taus) + torch.log(1 - torch.tensor(self.probability_threshold, device = self.device))
            
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


    '''
    Plot utilities    '''
    def plot(self, minibatch, opt):
        '''
        This function extracts input_time, input_events, input_intensity, mask, mean, and var from a minibatch.
        '''
        plot_type_to_functions = {
            'intensity': self.intensity,
            'integral': self.integral,
            'probability': self.probability,
            'debug': self.debug
        }
    
        return plot_type_to_functions[opt.plot_type](minibatch, opt)


    def extract_plot_data(self, minibatch):
        '''
        Intensity function prober, used by tpp_ploter to draw plots.
        '''
        input_time, input_events, _, mask, input_intensity = minibatch[0]
        mean, var = minibatch[1]

        return input_time, input_events, input_intensity, mask, mean, var


    def intensity(self, input_data, opt):
        '''
        Function prober, used by tpp_ploter to draw plots.
        '''
        self.model.eval()

        input_time, input_events, input_intensity, mask, mean, var = self.extract_plot_data(input_data)
        
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        expand_integral, expand_intensity, timestamp = \
            self.model.integral_intensity_time_next_2d(events_history, time_history, time_next, opt.resolution, mean, var)
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
            self.model.integral_intensity_time_next_2d(events_history, time_history, time_next, opt.resolution, mean, var)
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
            self.model.integral_intensity_time_next_2d(events_history, time_history, time_next, opt.resolution, mean, var)
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
        We use this function to investigate the property of RMTPP.
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

        data, timestamp = self.model.model_probe_function(events_history, time_history, \
                                                          time_next, opt.resolution, mean, var, mask_next)

        '''
        Append additional info into the data dict.
        '''
        data['events_next'] = events_next
        data['time_next'] = time_next
        data['mask_next'] = mask_next
        data['f1_after_time_pred'] = f1_1
        data['mae_before_event'] = mae

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

        expand_integral, expand_intensity, timestamp = \
            self.model.integral_intensity_time_next_2d(events_history, time_history, time_next, opt.resolution, mean, var)
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
        '''
        Models learning p^*(t) and p^*(m) can not solve the event-time prediction task.
        '''
        raise NotImplementedError("Event-time evaluation is unavailable because RMTPP is a TPP model.")


    def train_step(model, minibatch, device):
        model.train()
        
        [time, events, score, mask], (mean, var) = minibatch                   # 4 * [batch_size, seq_len + 1]
        loss, time_loss, events_loss, the_number_of_events, constant \
                   = model(events, time, mask, mean, var, evaluate = False)

        loss.backward()

        loss = loss.item() / the_number_of_events
        time_loss = time_loss.item() / the_number_of_events
        events_loss = events_loss.item() / the_number_of_events
        fact = score.sum().item() / the_number_of_events
        constant_norm = torch.linalg.norm(constant).detach().item() / the_number_of_events

        return loss, time_loss, fact, events_loss, constant_norm


    def evaluation_step(model, minibatch, device):
        model.eval()

        [time, events, score, mask], (mean, var) = minibatch                   # 4 * [batch_size, seq_len + 1]
        loss_time_next, time_loss_time_next, events_loss_time_next, \
        loss_pred_time, time_loss_pred_time, events_loss_pred_time, \
        mae, f1, the_number_of_events, constant_time_next, \
        constant_pred_time = model(events, time, mask, mean, var, evaluate = True)
        
        # Loss values and other metrics at time_next
        loss_time_next = loss_time_next.item() / the_number_of_events
        time_loss_time_next = time_loss_time_next.item() / the_number_of_events
        fact = score.sum().item() / the_number_of_events
        events_loss_time_next = events_loss_time_next.item() / the_number_of_events
        constant_time_next = torch.linalg.norm(constant_time_next).item() / the_number_of_events

        # Loss values and other metrics at pred_time
        loss_pred_time = loss_pred_time.item() / the_number_of_events
        time_loss_pred_time = time_loss_pred_time.item() / the_number_of_events
        events_loss_pred_time = events_loss_pred_time.item() / the_number_of_events
        mae = mae.sum().item() / the_number_of_events
        constant_pred_time = torch.linalg.norm(constant_pred_time).item() / the_number_of_events


        return loss_time_next, time_loss_time_next, fact, events_loss_time_next, \
               constant_time_next, loss_pred_time, time_loss_pred_time, \
               events_loss_pred_time, constant_pred_time, mae, f1


    def postprocess(input, procedure):
        if procedure == 'Training':
            return [input[0], input[1], input[1] - input[2], input[3], input[4]]
        else:
            return [input[0], input[1], input[1] - input[2], input[3], \
                    input[4], input[5], input[6], input[7], input[8], \
                    input[9], input[10]]


    format_dict_length = 11


    def log_print_format(input, procedure):
        def format_training(input):
            format_dict = {}
            format_dict['loss'] = input[0]
            format_dict['absolute_time_loss'] = input[1]
            format_dict['relative_time_loss'] = input[2]
            format_dict['events_loss'] = input[3]
            format_dict['constant_norm'] = input[4]
            format_dict['num_format'] = {'loss': ':8.5f', 'absolute_time_loss': ':8.5f', 'relative_time_loss': ':8.5f', \
                                         'events_loss': ':8.5f', 'constant_norm': ':8.5f'}
            return format_dict

        def format_eva_and_test(input):
            format_dict = {}
            '''
            loss_time_next, time_loss_time_next, fact, events_loss_time_next,
            constant_time_next, loss_pred_time, time_loss_pred_time,
            events_loss_pred_time, constant_pred_time, mae, f1
            '''
            format_dict['loss_time_next'] = input[0]
            format_dict['absolute_time_loss_time_next'] = input[1]
            format_dict['relative_time_loss_time_next'] = input[2]
            format_dict['events_loss_time_next'] = input[3]
            format_dict['constant_norm_time_next'] = input[4]
            format_dict['loss_pred_time'] = input[5]
            format_dict['absolute_time_loss_pred_time'] = input[6]
            format_dict['events_loss_pred_time'] = input[7]
            format_dict['constant_norm_pred_time'] = input[8]
            format_dict['mae'] = input[9]
            format_dict['f1'] = input[10]

            format_dict['num_format'] = {
                'loss_time_next': ':8.5f', 'absolute_time_loss_time_next': ':8.5f', 'relative_time_loss_time_next': ':8.5f', 
                'events_loss_time_next': ':8.5f', 'constant_norm_time_next': ':8.5f', 'loss_pred_time': ':8.5f',
                'absolute_time_loss_pred_time': ':8.5f', 'events_loss_pred_time': ':8.5f', 'constant_norm_pred_time': ':8.5f', 
                'mae': ':2.8f', 'f1': ':2.8f'
            }
            return format_dict

        return format_training(input) if procedure == 'Training' else format_eva_and_test(input)
    

    def choose_metric(evaluation_report_format_dict, test_report_format_dict):
        '''
        [relative loss on evaluation dataset, relative loss on test dataset]
        '''
        return [evaluation_report_format_dict['loss_time_next'], test_report_format_dict['loss_time_next']], \
               ['evaluation_absolute_loss', 'test_absolute_loss']
    
    metric_number = 2 # metric number is the length of the output of choose_metric[0]