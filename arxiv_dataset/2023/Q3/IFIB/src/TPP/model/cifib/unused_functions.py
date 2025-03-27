'''
    def mean_absolute_error_per_event_worker(self, events_history, events_next, time_history, time_next, p_m, mask_next, mean, var):
        \'''
        The input should be the original minibatch
        MAE evaluation part, dwg and fullynn exclusive
        \'''
        def evaluate_per_event(taus):
            # Train k FullyNN models for k different event types.
            if self.event_toggle:
                taus = repeat(taus, 'b s -> b s ne', ne = self.num_events)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
            taus.requires_grad = True
            # \int_{t}^{+\inf}{p(m, \tau|\mathcal{H})d\tau}
            probability_integral_from_t_to_infinite = self.model(events_history, time_history, taus, mean = mean, var = var)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
    
            if self.event_toggle:
                events_next_index = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
                probability_integral_from_t_to_infinite = probability_integral_from_t_to_infinite * events_next_index
                                                                               # [batch_size, seq_len, num_events]
                probability_integral_from_t_to_infinite = reduce(probability_integral_from_t_to_infinite, 'b s ne -> b s', 'sum')
                                                                               # [batch_size, seq_len]

            probability_integral_from_zero_to_t = p_m - probability_integral_from_t_to_infinite

            return probability_integral_from_zero_to_t

        def bisect_target(taus):
            p_mt = evaluate_per_event(taus)                                    # [batch_size, seq_len]
            p_t_m = p_mt / p_m                                                 # [batch_size, seq_len]
            p_gap = p_t_m - self.probability_threshold                         # [batch_size, seq_len]

            return p_gap
            
        def median_prediction(l, r):
            for _ in range(50):
                c = (l + r)/2
                v = bisect_target(c)
                l = torch.where(v < 0, c, l)
                r = torch.where(v >= 0, c, r)

            return (l + r)/2

        max_ = min(1e6, mean + 10 * var)
        
        l = 0.0001*torch.ones_like(time_history, dtype = torch.float32)        # [batch_size, seq_len]
        r = max_*torch.ones_like(time_history, dtype = torch.float32)          # [batch_size, seq_len]

        tau_pred = median_prediction(l, r)
        gap = (tau_pred - time_next) * mask_next
        gap = torch.abs(gap)

        return gap
'''

'''
        p_m_predicted = reduce(probability_integral_from_zero_to_infinite * predict_index_one_hot, '... ne -> ...', 'sum')
                                                                               # [batch_size, seq_len]
        p_m_real = reduce(probability_integral_from_zero_to_infinite * events_next_one_hot, '... ne -> ...', 'sum')
                                                                               # [batch_size, seq_len]
        mae_per_event_pure_predict = self.mean_absolute_error_per_event_worker(events_history, predict_index, time_history, time_next,
                                                                               p_m_predicted, mask_next, mean, var)
        mae_per_event = self.mean_absolute_error_per_event_worker(events_history, events_next, time_history, time_next, 
                                                                  p_m_real, mask_next, mean, var)
'''