'''
The grave yard of functions.

These functions might be introduced to verify the MAE-E values.
'''

'''
The gap between mae_per_event_with_predict_index and mae_per_event_pure_predict, mae_per_event_with_event_next and 
mae_per_event_next should be minor.

p_m_predicted = reduce(p_m * predict_index_one_hot_mask, '... ne -> ...', 'sum')
                                                                       # [batch_size, seq_len]
p_m_real = reduce(p_m * events_next_one_hot_mask, '... ne -> ...', 'sum')
                                                                       # [batch_size, seq_len]

mae_per_event_pure_predict = self.mean_absolute_error_per_event_worker(events_history, predict_index, time_history, time_next,
                                                                       p_m_predicted, resolution, mask_next, mean, var, max_)
mae_per_event_next = self.mean_absolute_error_per_event_worker(events_history, events_next, time_history, time_next, 
                                                          p_m_real, resolution, mask_next, mean, var, max_)
'''


'''

    def mean_absolute_error_per_event_worker(self, events_history, events_next, time_history, time_next, p_m, resolution, \
                                             mask_next, mean, var, max_val):
        \'''
        The time prediction of given markers

        Args:
        * events_history  type: torch.tensor shape: [batch_size, seq_len]
                          Historical event sequences. Commonly, this sequence is a slice of 
                          the original event sequence from 0 to seq_len - 1(included).
        * events_next     type: torch.tensor shape: [batch_size, seq_len]
                          The mark of the events that we need to predict.
        * time_history    type: torch.tensor shape: [batch_size, seq_len]
                          Historical time sequences. Similar to events_history, we always generate
                          this sequence as a slice of the original time sequence from 0 to seq_len - 1(included).
        * time_next       type: torch.tensor shape: [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len]
                          When the next event actually happens.
        * p_m             type: torch.tensor shape: [batch_size, seq_len]
                          the value of p(m) with given markers.
        * resolution      type: int shape: N/A
                          How many values do we need in each time interval [t_{i}, t_{i + 1}].
        * mask_next       type: torch.tensor shape: [batch_size, seq_len]
                          Needed mask to mask out unneeded loss values.
        * mean            type: float shape: N/A
                          The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide
                          this value if needed.
        * var             type: float shape: N/A
                          The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide
                          this value if needed.
        * max_val         type: float shape: N/A
                          The upper bound used in the bisect method.

        Outputs:
        * gap             type: torch.tensor shape: [batch_size, seq_len]
                          MAE-E = |\hat{t}_m - t|

        \'''
        def evaluate_per_event(taus):
            # Train k FullyNN models for k different event types.
            integral_all_events, intensity_all_events, time_interval \
                = self.model.integral_intensity_time_next_2d(events_history, time_history, taus, resolution, mean, var)  
                                                                               # [batch_size, seq_len, resolution, num_events] * n

            events_next_index = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
            events_next_index = rearrange(events_next_index, 'b s ne -> b s 1 ne')
                                                                               # [batch_size, seq_len, 1, num_events]
            intensity_i = reduce(intensity_all_events * events_next_index, 'b s r ne -> b s r', 'sum')
                                                                               # [batch_size, seq_len, resolution]
            integral_sum = reduce(integral_all_events, 'b s r ne -> b s r', 'sum')
                                                                               # [batch_size, seq_len, resolution]
            p_dist = intensity_i * torch.exp(-integral_sum)                    # [batch_size, seq_len, resolution]

            p_dist_for_monte_carlo = p_dist[:, :, :-1]                         # [batch_size, seq_len, resolution - 1]
            time_interval_for_monte_carlo = time_interval[:, :, 1:]            # [batch_size, seq_len, resolution - 1]

            probability = reduce(p_dist_for_monte_carlo * time_interval_for_monte_carlo, 'b s r -> b s', 'sum')
                                                                               # [batch_size, seq_len]
            return probability

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
        
        l = 0.0001*torch.ones_like(time_history, dtype = torch.float32)        # [batch_size, seq_len]
        r = max_val*torch.ones_like(time_history, dtype = torch.float32)       # [batch_size, seq_len]
        tau_pred = median_prediction(l, r)                                     # [batch_size, seq_len]
        gap = (tau_pred - time_next) * mask_next
        gap = torch.abs(gap)

        return gap

'''