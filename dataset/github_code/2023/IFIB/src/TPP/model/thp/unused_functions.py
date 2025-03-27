'''
        mae_per_event_pure_predict = self.mean_absolute_error_per_event_worker(events_history, predicted_events, \
                                                                               time_history, time_next, mask_history, \
                                                                               mask_next, probability, resolution, \
                                                                               mean, var, max_)
                                                                               # [batch_size, seq_len]
        mae_per_event = self.mean_absolute_error_per_event_worker(events_history, events_next, \
                                                                  time_history, time_next, mask_history, \
                                                                  mask_next, probability, resolution, \
                                                                  mean, var, max_)
                                                                               # [batch_size, seq_len]
'''

'''

    def mean_absolute_error_per_event_worker(self, events_history, events_next, 
        time_history, time_next, mask_history, mask_next, probability_integral, resolution, mean, var, max_val):
        \'''
        The input should be the original minibatch
        MAE evaluation part, dwg and fullynn exclusive
        \'''
        def evaluate_per_event(taus):
            expanded_integral_all_events, expanded_intensity_all_events, timestamp = \
                self.model.integral_intensity_time_next_2d(events_history, time_history, taus, mask_history, \
                                                           resolution, mean, var)
                                                                               # 2 * [batch_size, seq_len, resolution, num_events] + [batch_size, seq_len, resolution]
            events_next_mask = F.one_hot(events_next.long(), num_classes = self.num_events).unsqueeze(dim = -2)
                                                                               # [batch_size, seq_len, 1, num_events]
            expanded_intensity_selected_event = (expanded_intensity_all_events * events_next_mask).sum(dim = -1)
                                                                               # [batch_size, seq_len, resolution]
            expanded_integral_selected_event = expanded_integral_all_events.sum(dim = -1)
                                                                               # [batch_size, seq_len, resolution]
            expanded_probability_selected_event = expanded_intensity_selected_event * torch.exp(-expanded_integral_selected_event)
                                                                               # [batch_size, seq_len, resolution]

            expanded_probability_selected_event_monte_carlo = expanded_probability_selected_event[:, :, :-1]
                                                                               # [batch_size, seq_len, resolution - 1]
            timestamp_monte_carlo = timestamp[:, :, 1:]                        # [batch_size, seq_len, resolution - 1]
            probability = (expanded_probability_selected_event_monte_carlo * timestamp_monte_carlo).sum(dim = -1)
                                                                               # [batch_size, seq_len]
            return probability

        def bisect_target(taus):
            events_next_one_hot = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
            p_xt = evaluate_per_event(taus)                                    # [batch_size, seq_len]
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