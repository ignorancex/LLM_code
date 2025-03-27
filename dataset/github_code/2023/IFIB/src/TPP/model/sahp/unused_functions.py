'''
        mae_per_event_pure_predict = self.mean_absolute_error_per_event_worker(events_history, events_next, time_history, \
                                                                  time_next, mask_history, mask_next, probability, \
                                                                  resolution, max_, mean, var)
        mae_per_event = self.mean_absolute_error_per_event_worker(events_history, events_next, time_history, \
                                                                  time_next, mask_history, mask_next, probability, \
                                                                  resolution, max_, mean, var)
'''

'''
    def mean_absolute_error_per_event_worker(self, events_history, events_next, 
        time_history, time_next, mask_history, mask_next, probability_integral, resolution, max_val, mean, var):
        \'''
        The input should be the original minibatch
        MAE evaluation part, dwg and fullynn exclusive
        \'''
        def evaluate_per_event(taus, event_mask):
            expanded_integral_all_events, expanded_intensity_all_events, timestamp = \
                self.model.integral_intensity_time_next_2d(events_history, time_history, taus, mask_history, resolution, mean, var)
                                                                               # 2 * [batch_size, seq_len, resolution, num_events]
            
            expanded_integral_sum_over_events = expanded_integral_all_events.sum(dim = -1, keepdim = True)
                                                                               # [batch_size, seq_len, resolution, 1]
            expanded_probability = expanded_intensity_all_events * torch.exp(-expanded_integral_sum_over_events)
                                                                               # [batch_size, seq_len, resolution, num_events]
            expanded_probability_monte_carlo = expanded_probability[:, :, :-1, :]
                                                                               # [batch_size, seq_len, resolution - 1, num_events]
            timestamp_monte_carlo = timestamp[:, :, 1:].unsqueeze(dim = -1)    # [batch_size, seq_len, resolution - 1, 1]
            expanded_probability_all_events = expanded_probability_monte_carlo * timestamp_monte_carlo
                                                                               # [batch_size, seq_len, resolution - 1, num_events]
            expanded_probability_all_events = expanded_probability_all_events.sum(dim = -2)
                                                                               # [batch_size, seq_len, num_events]
            expanded_probability = (expanded_probability_all_events * event_mask).sum(dim = -1)
                                                                               # [batch_size, seq_len]
            
            return expanded_probability

        def bisect_target(taus):
            events_next_one_hot = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
            p_xt = evaluate_per_event(taus, events_next_one_hot)
                                                                               # [batch_size, seq_len]
            p_x = torch.sum(probability_integral * events_next_one_hot, dim = -1)
                                                                               # [batch_size, seq_len]
            p_t_x = p_xt / p_x                                                 # [batch_size, seq_len]
            p_gap = p_t_x - self.probability_threshold                         # [batch_size, seq_len]

            return p_gap
            
        def median_prediction(l, r):
            for _ in range(50):
                c = (l + r)/2
                v = bisect_target(c)
                l = torch.where(v < 0, c, l)
                r = torch.where(v >= 0, c, r)

            return (l + r)/2
        
        l = 0.0001*torch.ones_like(time_history, dtype = torch.float32)        # [batch_size, seq_len]
        r = max_val*torch.ones_like(time_history, dtype = torch.float32)       # [batch_size, seq_len]
        tau_pred = median_prediction(l, r)
        gap = (tau_pred - time_next) * mask_next
        gap = torch.abs(gap)

        return gap
'''