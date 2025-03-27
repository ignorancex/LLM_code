'''

   def mean_absolute_error_per_event(self, input_time, input_events, mask, mean, var, fast):
        if self.original_mark_generation:
            raise Exception('Original RMTPP model is in fact a TPP model with a dedicated event prediction module, so pe-MAE does not function here.')
        
        time_history, time_next = self.divide_history_and_next(input_time, unsqueeze = True)
                                                                               # [batch_size, seq_len, 1]
        events_history, events_next = self.divide_history_and_next(input_events, unsqueeze = False)
                                                                               # [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask, unsqueeze = False)   # [batch_size, seq_len]

        if mean == 0 and var == 1:
            \'''
            This dataset does not apply normalisation, so we need to calculate the mean and variance here.
            \'''
            mean = input_time.mean()
            var = input_time.var()
        
        # Use a relatively large number as the positive infinity.
        max_ = min(1e6, mean + 10 * var)
        resolution = min(int(max_ * 100), 50000)
        time_infinite = torch.ones_like(time_next, device = self.device) * max_# [batch_size, seq_len, 1]

        # First, we find the integral and intensity function that RMTPP estimates.
        # This part is only available when original_mark_generation is false as the original RMTPP model
        # is in fact a TPP model.
        intensity, integral, timestamp = \
                self.submodel.intensity_integral(events_history, time_history, time_infinite, resolution, mean, var, sum = False)
                                                                               # 2 * [batch_size, seq_len * resolution, num_events] + [batch_size, seq_len * resolution]
        intensity = rearrange(intensity, 'b (s r) ne -> b s r ne', r = resolution)
                                                                               # [batch_size, seq_len, resolution, num_events]
        integral = rearrange(integral, 'b (s r) ne -> b s r ne', r = resolution)
                                                                               # [batch_size, seq_len, resolution, num_events]
        timestamp = rearrange(timestamp, 'b (s r) -> b s r 1', r = resolution) # [batch_size, seq_len, resolution, 1]
        probability_dist = intensity * torch.exp(-integral.sum(dim = -1, keepdim = True))
                                                                               # [batch_size, seq_len, resolution, num_events]
        # After investigation, sometimes we could get nan when both intensity and integral are inf
        # Based on TPP's definition, the true value should be 0.
        probability_dist = torch.nan_to_num(probability_dist, nan = 0.0)       # [batch_size, seq_len, resolution, num_events]
        
        cumulated_probability = probability_dist[:, :, :-1, :] * timestamp[:, :, 1:, :] / var
                                                                               # [batch_size, seq_len, resolution, num_events]
        probability = cumulated_probability.sum(dim = -2)                      # [batch_size, seq_len, num_events]
        probability_integral_sum = probability.sum(dim = -1)                   # [batch_size, seq_len]
        predicted_events = torch.argmax(probability, dim = -1)                 # [batch_size, seq_len]

        # F1 value and top_k_acc are only avaliable when batch_size = 1
        f1 = f1_score(y_true = events_next.squeeze().detach().cpu(),
                      y_pred = predicted_events.squeeze().detach().cpu(), average = 'macro')

        # Only available when batch_size = 1
        top_k_acc = []
        if not fast:
            if self.num_events > 2:
                for k in range(1, self.num_events + 1):
                    top_k_acc.append(
                        top_k_accuracy_score(y_true = events_next.squeeze().detach().cpu(),
                                             y_score = probability.reshape(-1, self.num_events).detach().cpu(),
                                             k = k,
                                             labels = np.arange(self.num_events))
                    )
            else:
                top_k_acc.append(
                    accuracy_score(
                        y_true = events_next.squeeze().detach().cpu(),
                        y_pred = predicted_events.squeeze().detach().cpu()
                    )
                )
                top_k_acc.append(1.0)
        
        if mean == 0:
            resolution = max(min(int(input_time.mean().item() * 200), 1000), 1)
        else:
            resolution = max(min(int(mean * 200), 1000), 1)

        tau_pred_all_event = self.prediction_with_all_event_types(events_history, time_history,
                                                                  probability, resolution, mask_next, mean, var, max_)
                                                                               # [batch_size, seq_len, num_events]
        
        mae_per_event_pure_predict = self.mean_absolute_error_per_event_worker(events_history, predicted_events, time_history, time_next,
                                                                               probability, resolution, mask_next, max_, mean, var)
        mae_per_event = self.mean_absolute_error_per_event_worker(events_history, events_next, time_history, time_next, 
                                                                  probability, resolution, mask_next, max_, mean, var)

        mae_per_event_pure_predict_avg = torch.sum(mae_per_event_pure_predict) / mask_next.sum()
        mae_per_event_avg = torch.sum(mae_per_event) / mask_next.sum()
        
        return f1, top_k_acc, probability_integral_sum, tau_pred_all_event, (mae_per_event_pure_predict_avg.item(), mae_per_event_avg.item()), \
               (mae_per_event_pure_predict, mae_per_event)


    def evaluate_per_event(self, events_history, time_history, events_mask, tau, resolution, mean, var):
        intensity, integral, timestamp = \
                        self.model.integral_intensity_time_next_2d(events_history, time_history, tau, resolution, mean, var)
                                                                               # 2 * [batch_size, seq_len * resolution, num_events] + [batch_size, seq_len * resolution]
        probability_dist = intensity * torch.exp(-integral.sum(dim = -1, keepdim = True))
                                                                               # [batch_size, seq_len * resolution, num_events]
        probability_dist = rearrange(probability_dist, 'b (s r) n -> b s r n', r = resolution)
                                                                               # [batch_size, seq_len, resolution, num_events]
        timestamp = rearrange(timestamp, 'b (s r) -> b s r 1', r = resolution)
                                                                               # [batch_size, seq_len, resolution, num_events]
        cumulative_probability = probability_dist[:, :, :-1, :] * timestamp[:, :, 1:, :] / var
                                                                               # [batch_size, seq_len, resolution - 1, num_events]
        cumulative_probability = cumulative_probability.sum(dim = -2)          # [batch_size, seq_len, num_events]
        cumulative_probability = cumulative_probability * events_mask          # [batch_size, seq_len, num_events]
        probability = cumulative_probability.sum(dim = -1)                     # [batch_size, seq_len]

        return probability


    def mean_absolute_error_per_event_worker(self, events_history, events_next, 
        time_history, time_next, probability_integral, resolution, mask, max_val, mean, var):
        \'''
        The input should be the original minibatch
        MAE evaluation part, dwg and fullynn exclusive

        \'''
        def bisect_target(taus):
            events_next_one_hot = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
            p_xt = self.evaluate_per_event(events_history, time_history, events_next_one_hot, taus, resolution, mean, var)
                                                                               # [batch_size, seq_len]
            p_x = torch.sum(probability_integral * events_next_one_hot, dim = -1)
                                                                               # [batch_size, seq_len]
            p_t_x = p_xt / p_x                                                 # [batch_size, seq_len]
            p_gap = p_t_x - self.probability_threshold                         # [batch_size, seq_len]

            return p_gap.unsqueeze(dim = -1)
            
        def median_prediction(l, r):
            for _ in range(50):
                c = (l + r)/2
                v = bisect_target(c)
                l = torch.where(v < 0, c, l)
                r = torch.where(v >= 0, c, r)

            return (l + r)/2
        
        l = 0.0001*torch.ones_like(time_history, dtype = torch.float32)        # [batch_size, seq_len, 1]
        r = 1e6*torch.ones_like(time_history, dtype = torch.float32)           # [batch_size, seq_len, 1]
        tau_pred = median_prediction(l, r)
        gap = (tau_pred - time_next).squeeze(-1) * mask
        gap = torch.abs(gap)

        return gap


    def mean_absolute_error_e(self, events_history, events_next, time_history, time_next, mask_next, mean, var):
        \'''
        MAE-E evaluation module.

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
        * mask_next       type: torch.tensor shape: [batch_size, seq_len]
                          Needed mask to mask out unneeded loss values.
        * mean            type: float shape: N/A
                          The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide
                          this value if needed.
        * var             type: float shape: N/A
                          The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide
                          this value if needed.
        Outputs:
        * mae             type: torch.tensor shape: [batch_size, seq_len]
                          MAE(Mean Absolute Error) between predicted time and ground truth.
        * tau_pred        type: torch.tensor shape: [batch_size, seq_len]
                          Time predicted by the sum of all intensity functions $ \lambda^*(m, t) $ over $ m $.
        \'''
        if self.original_mark_generation:
            raise Exception('The original RMTPP model is in fact a TPP model, not a native MTPP model. So MAE-E calculation is unavailable.')

        \'''
        Set the memory limit
        \'''
        memory_ceiling = 2e7

        \'''
        set a relatively large number as the infinity and decide resolution based on this large value and
        the memory_ceiling.
        \'''
        if mean == 0 and var == 1:
            max_ = time_next.mean() + 10 * time_next.var()
        else:
            max_ = mean + 10 * var
        
        max_ = min(1e6, max_)
        time_next_inf = torch.ones_like(time_history, device = self.device) * max_
                                                                               # [batch_size, seq_len]
        resolution = max(int(max_ // 0.005), 100)

        _, seq_len = events_next.shape
        if seq_len * resolution * self.num_events > memory_ceiling:
            resolution = int(memory_ceiling // (seq_len * self.num_events))

        \'''
        Step 1: obtain p^*(m) = \int_{t_l}^{+infty}{p(m, t)\dt}
        \'''
        expand_integral_to_inf, expand_intensity_to_inf, time_interval \
                = self.model.integral_intensity_time_next_2d(events_history, time_history, time_next_inf, resolution, mean, var)
                                                                               # [batch_size, seq_len, resolution, num_events]

        \'''
        Step 2: provide event predictions
        \'''        
        expand_probability_per_event = expand_intensity_to_inf * torch.exp(-expand_integral_to_inf.sum(dim = -1, keepdim = True))
                                                                               # [batch_size, seq_len, resolution, num_events]
        expand_probability_per_event_for_monte_carlo = expand_probability_per_event[:, :, :-1, :]
                                                                               # [batch_size, seq_len, resolution - 1, num_events]
        time_interval_used_for_monte_carlo = time_interval[:, :, 1:].unsqueeze(dim = -1)
                                                                               # [batch_size, seq_len, resolution - 1, 1]
        probability_integral = expand_probability_per_event_for_monte_carlo * time_interval_used_for_monte_carlo
                                                                               # [batch_size, seq_len, resolution - 1, num_events]
        p_m = reduce(probability_integral, 'b s r ne -> b s ne', 'sum')        # [batch_size, seq_len, num_events]
        probability_integral_sum = reduce(p_m, 'b s ne -> b s', 'sum')         # [batch_size, seq_len]
        predict_index = torch.argmax(p_m, dim = -1)                            # [batch_size, seq_len]

        \'''
        Step 3: calculate macro-F1 and top-K accuracy
        \'''
        f1 = []
        top_k_acc = []
        for (events_next_per_seq, p_m_per_seq) in zip(events_next, p_m):
            f1.append(f1_score(y_true = events_next_per_seq.detach().cpu(),
                               y_pred = torch.argmax(p_m_per_seq, dim = -1).detach().cpu(), average = 'macro'))
            
            top_k_acc_single_event_seq = []
            if self.num_events > 2:
                for k in range(1, self.num_events):
                    top_k_acc_single_event_seq.append(
                        top_k_accuracy_score(y_true = events_next_per_seq.detach().cpu(),
                                             y_score = p_m_per_seq.detach().cpu(),
                                             k = k,
                                             labels = np.arange(self.num_events))
                    )
            else:
                top_k_acc_single_event_seq.append(
                    accuracy_score(
                        y_true = events_next_per_seq.detach().cpu(),
                        y_pred = p_m_per_seq.detach().cpu()
                    )
                )
            top_k_acc.append(top_k_acc_single_event_seq)

        predict_index_one_hot_mask = torch.nn.functional.one_hot(predict_index.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        events_next_one_hot_mask = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        \'''
        Step 4: get the time prediction for all, predicted, and real events.
        \'''
        if mean == 0:
            resolution = max(min(int(time_next.mean().item() // 0.005), 500), 10)
        else:
            resolution = max(min(int(mean // 0.005), 500), 10)

        tau_pred_all_event = self.prediction_with_all_event_types(events_history, time_history, p_m, resolution, mean, var, max_)
                                                                               # [batch_size, seq_len, num_events]
        mae_per_event_with_predict_index = torch.abs(((tau_pred_all_event * predict_index_one_hot_mask).sum(dim = -1)) - time_next) * mask_next
                                                                               # [batch_size, seq_len]
        mae_per_event_with_event_next = torch.abs(((tau_pred_all_event * events_next_one_hot_mask).sum(dim = -1)) - time_next) * mask_next
                                                                               # [batch_size, seq_len]

        mae_per_event_with_predict_index_avg = torch.sum(mae_per_event_with_predict_index, dim = -1) / mask_next.sum(dim = -1)
        mae_per_event_with_event_next_avg = torch.sum(mae_per_event_with_event_next, dim = -1) / mask_next.sum(dim = -1)

        return f1, top_k_acc, probability_integral_sum, tau_pred_all_event, \
               (mae_per_event_with_predict_index_avg, mae_per_event_with_event_next_avg), \
               (mae_per_event_with_predict_index, mae_per_event_with_event_next)


    def prediction_with_all_event_types(self, events_history, time_history, p_m, resolution, mean, var, max_val):
        \'''
        The time prediction of every marker whose probability is not 0.

        Still, this function is currently buggy.

        Args:
        * events_history  type: torch.tensor shape: [batch_size, seq_len]
                          Historical event sequences. Commonly, this sequence is a slice of 
                          the original event sequence from 0 to seq_len - 1(included). 
        * time_history    type: torch.tensor shape: [batch_size, seq_len]
                          Historical time sequences. Similar to events_history, we always generate
                          this sequence as a slice of the original time sequence from 0 to seq_len - 1(included).
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
        * tau_pred        type: torch.tensor shape: [batch_size, seq_len]
                          Time predicted by the sum of all intensity functions $ \lambda^*(m, t) $ over $ m $.
        \'''
        def evaluate_all_event(taus):
            \'''
            placeholder
            \'''
            # Train k FullyNN models for k different event types.
            integral_all_events, intensity_all_events, time_interval \
                    = self.model.integral_intensity_time_next_3d(events_history, time_history, taus, resolution, mean, var)
                                                                               # 2 * [batch_size, seq_len, resolution, num_events, num_events] + [batch_size, seq_len, resolution, num_events]
            event_mask = torch.diag(torch.ones(self.num_events, device = self.device))
                                                                               # [num_events, num_events]
            event_mask = repeat(event_mask, 'ne ne1 -> 1 1 1 ne ne1')          # [batch_size, seq_len, resolution, num_events, num_events]
            intensity_all_events = reduce(intensity_all_events * event_mask, '... ne -> ...', 'sum')
                                                                               # [batch_size, seq_len, resolution, num_events]
            integral_all_events = reduce(integral_all_events, 'b s r ne ne1 -> b s r ne', 'sum')
                                                                               # [batch_size, seq_len, resolution, num_events]
            
            p_dist = intensity_all_events * torch.exp(-integral_all_events)    # [batch_size, seq_len, resolution, num_events]
            
            p_dist_for_monte_carlo = p_dist[:, :, :-1, :]                      # [batch_size, seq_len, resolution - 1, num_events]
            time_interval_for_monte_carlo = time_interval[:, :, 1:, :]         # [batch_size, seq_len, resolution - 1, num_events]
            probability = reduce(p_dist_for_monte_carlo * time_interval_for_monte_carlo, 'b s r ne -> b s ne', 'sum')
                                                                               # [batch_size, seq_len, num_events]
            return probability

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
        r = max_val*torch.ones((*time_history.shape, self.num_events), dtype = torch.float32, device = self.device)
                                                                               # [batch_size, seq_len, num_events]
        tau_pred = median_prediction(l, r)                                     # [batch_size, seq_len, num_events]

        return tau_pred



    def integral_intensity_time_next_3d(self, events_history, time_history, time_next, resolution, mean, var):
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

        history_part = self.intensity(history_output)                          # [batch_size, seq_len, num_events]
        history_part = repeat(history_part, 'b s ne -> b s 1 ne1 ne', ne1 = self.num_events)
                                                                               # [batch_size, seq_len, 1, num_events, num_events]
        if self.limited_history_norm:
            history_part = torch.tanh(history_part)                            # [batch_size, seq_len, 1, num_events, num_events]

        history_part = torch.exp(history_part)                                 # [batch_size, seq_len, 1, num_events, num_events]

        base = self.base_intensity(history_output)                             # [batch_size, seq_len, num_events]
        base = repeat(base, 'b s ne -> b s 1 ne1 ne', ne1 = self.num_events)   # [batch_size, seq_len, 1, num_events, num_events]
        base = torch.exp(base)                                                 # [batch_size, seq_len, 1, num_events, num_events]

        constant = history_part * base                                         # [batch_size, seq_len, 1, num_events, num_events]


        time_scalar = self.time_scalar(history_output)                         # [batch_size, seq_len, num_events]
        time_scalar = repeat(time_scalar, 'b s ne -> b s 1 ne1 ne', ne1 = self.num_events)
                                                                               # [batch_size, seq_len, 1, num_events, num_events]

        # time_scalar can not be zero.
        time_scalar = self.clamp_time_scalar(time_scalar)                      # [batch_size, seq_len, 1, num_events, num_events]


        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
        original_time_expand = time_next.unsqueeze(dim = -1) * time_multiplier # [batch_size, seq_len, num_events, resolution]
        original_time_expand = rearrange(original_time_expand, 'b s ne r -> b s r ne')
                                                                               # [batch_size, seq_len, resolution, num_events]
        time_expand = repeat(original_time_expand, 'b s r ne -> b s r ne ne1', ne1 = self.num_events)
                                                                               # [batch_size, seq_len, resolution, num_events, num_events]
        time_expand_normed = time_expand / var                                 # [batch_size, seq_len, resolution, num_events, num_events]
        
        intensity_events = torch.exp(time_scalar * time_expand_normed) * constant
                                                                               # [batch_size, seq_len, resolution, num_events, num_events]
        integral_events = (intensity_events - constant) / time_scalar          # [batch_size, seq_len, resolution, num_events, num_events]

        # aggregated timestamp
        batch_size, seq_len, _, _ = original_time_expand.shape
        timestamp = torch.cat(
            (torch.zeros((batch_size, seq_len, 1, self.num_events), device = self.device), original_time_expand.diff(dim = -2)),
            dim = -2)                                                          # [batch_size, seq_len, resolution, num_events]

        return integral_events, intensity_events, timestamp

        '''