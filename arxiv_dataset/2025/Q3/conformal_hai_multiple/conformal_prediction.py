from config import conf
import numpy as np
import torch
import torch.nn.functional as F

from scipy import stats
import os

class ConformalPrediction:
    # Implements system with both standard and modified conformal prediction.
    """Implementation and evaluation of the conformal prediction based support system"""
    def __init__(self, X_cal, y_cal, X_est, y_est, model, delta, has_groups=False) -> None:
        self.model = model
        self.X_cal = X_cal
        self.y_cal = y_cal
        self.X_est = X_est
        self.y_est = y_est
        self.calibration_size = len(y_cal)                                  # 1000
        self.delta = delta
        # conformal scores of true labels in calibration set
        model_out = self.model.predict_prob(self.X_cal)                     # 1000 x C just get the model logits (softmax scores?) saved in csv
        self.has_groups = has_groups
        if not has_groups:
            one_hot = np.eye(conf.n_labels)[self.y_cal]                     # 1000xC just make y_cal to one hot encoding
        else:
            one_hot = np.eye(conf.n_labels)[self.y_cal[:,0]]

        true_label_logits = model_out*one_hot                               # 1000xC        # super sparse
        conf_scores = sorted(1 - true_label_logits.sum(axis=1))             # 1000          # higher logits mean lower confidence? uncertainty
        self.conf_scores_t = torch.tensor(conf_scores, device=conf.device)  # 1000          # used in finding a star
    
    def epsilon_fn(self, delta_n_alphas, all_a1_a2):
        """Estimation error"""
        delta_n_alphas_t = torch.tensor(delta_n_alphas)    
        n_alphas = self.calibration_size if not all_a1_a2 else (self.calibration_size*(self.calibration_size+1)/2)
        epsilon = torch.sqrt((torch.log(delta_n_alphas_t))/(2*n_alphas))
        return epsilon

    def find_all_alpha_values(self):
        """Returns all 0<alpha<1 values that can be considered given a fixed calibration set"""
        alphas = 1 - (np.arange(1,self.calibration_size + 1) / (self.calibration_size + 1))     # 1-([2,3,...,1001]/1001) = 1,0.99,...,0
        self.alphas = alphas
        self.n_alphas = self.calibration_size
        return alphas
    
    def find_a_star(self, w_matrix, a1_star_idx=None, all_a1_a2=False):
        """Returns the best alpha value or the best alpha_2 value given alpha_1"""
        a_star_idx = -1
        curr_criterion = 0
        # alphas for standard conformal prediction method
        alphas = self.alphas            # 1000      # 0.99 -> 0
        if a1_star_idx is not None: 
            # alphas and quantiles for shifted quantile method given a_1
            quant_prob_a1 = np.ceil((1 - alphas[a1_star_idx])*(self.calibration_size+1))/self.calibration_size
            qhat_a1 = torch.zeros((1,1), device=conf.device)
            qhat_a1 = torch.quantile(self.conf_scores_t, quant_prob_a1)
            alphas = self.alphas[self.alphas > self.alphas[a1_star_idx]]
            if all_a1_a2 is not None:
                alphas = np.append(alphas, 1)

        # quantile probabilities for each alpha value
        quant_prob = (np.ceil((1 - alphas)*(self.calibration_size+1))/self.calibration_size).flatten()      # 0 -> 1 # 1000
        
        # output scores for each sample in estimation set
        output_scores = 1 - self.model.predict_prob(self.X_est)     # 1000x10  # based on est data # Higher output score means lower model probability # More like uncertainty
        
        # move data to gpu if available
        quant_probs_t = torch.tensor(quant_prob, device=conf.device)                # 1000
        qhats_t = torch.quantile(self.conf_scores_t, quant_probs_t, keepdim=True)   # 1000 x 1 # conf_scores_t from the calibration data    TODO How imbalanced class affects quantile calculation
        qhats_t = qhats_t.unsqueeze(1)                  # 1000 x 1 x 1
        y_est_t = torch.tensor(self.y_est, device=conf.device, dtype=torch.int64)       # 1000
        fill_value_t = torch.tensor(0, dtype=torch.double, device=conf.device)
        output_scores_t = torch.tensor(output_scores, device=conf.device)               # 1000x10 
        if not self.has_groups:
            ws_t = torch.tensor(w_matrix[self.y_est], device=conf.device)   # log of confusion matrix # from expert model # 1000x10
        else:
            ws_t = torch.tensor(w_matrix[self.y_est[:,1],self.y_est[:,0]], device=conf.device)
        
        # estimation error
        delta_n_alphas = (alphas.shape[0]/self.delta) if not all_a1_a2 else (self.calibration_size*(self.calibration_size + 1)/2)/self.delta    # 10000
        epsilon = self.epsilon_fn(delta_n_alphas, all_a1_a2)

        for i,q in enumerate(qhats_t):      #iterate to find the best?     # Iterate on the quantiles      #Note qhats_t[999] = 1.00, qhats_t[950] = 0.2216, qhats_t[900] = 0.0040, qhats_t[800] = 0.0003
            # THIS IS FOR EACH QUANTILE VALUE q. Note is q is element of conf_scores set
            qhats = q.expand(self.calibration_size, conf.n_labels)      # 1000 x 10 of same value   # quantile corresponding to quant_probs_t[i]    

            # sets[sample][label] is 1 for the labels in the prediction set for each sample
            if a1_star_idx is not None:
                # sets for shifted quantile method given a_1
                qhats_a1 = qhat_a1.expand(self.calibration_size, conf.n_labels)
                sets_upper = torch.where(output_scores_t <= qhats_a1, 1, 0)
                sets_lower = torch.where(qhats < output_scores_t, 1, 0)
                sets = sets_upper* sets_lower
            else:
                # sets for standard conformal prediction method
                sets = torch.where(output_scores_t <= qhats, 1, 0)  # 1000 x 10 # very sparse, like 5 for initial i's
            sets_exp_ws = sets * torch.exp(ws_t)                    # 1000 x 10   # where PREDICTION (confusion matrix) MEETS THE EXPERT?    # basically sets x confusion mat values

            # denominators for all P[\hat Y = Y ; C_alpha |  Y \in C_alpha(X), Y=y]
            denominators = torch.sum(sets_exp_ws, axis=1)    # weighted predicted   # 1000 (idea: denominator per sample)
            if not self.has_groups:
                one_hot_ycal = F.one_hot(y_est_t, num_classes=ws_t.shape[1])   # 1000x10
            else:
                one_hot_ycal = F.one_hot(y_est_t[:,0])
            
            # nominators for all P[\hat Y = Y ; C_alpha | Y \in C_alpha(X), Y=y]
            nominators = torch.sum(sets_exp_ws*one_hot_ycal, axis=1)    # weighted true positive predicted TP   # 1000 (per sample)

            # mask for prediction sets that include the true label
            mask = sets * one_hot_ycal      # unweighted true positive    # 1000x10
            true_label_in_sets_idx = torch.sum(mask, axis=1)    # 1000 # equals to one if the true label is in the pred_set

            # apply mask so that Y \in C_alpha(X) is satisfied
            masked_prob = torch.where(true_label_in_sets_idx==1, nominators/denominators, fill_value_t)     # 1000
            
            # empirical estimation of human expected success probability when choosing from the prediction sets.
            expected_correct_prob = masked_prob.sum()/self.calibration_size     # maximizing this one // finding the right quantile
           
            criterion = (expected_correct_prob - epsilon)  # maximizing criterion         # epsilon is fixed/constant

            # Print plots for alpha vs set size or alpha vs success prob # changed  
            if conf.plot_alpha_vs_sets:
                # succprob
                with open(conf.path_plot_alpha_vs_sets, 'a') as file:
                    file.write(f"{quant_probs_t[i].item()} ")
                    file.write(f"{sets.sum().item()}") # expected_correct_prob.item()
                    file.write("\n")
            if conf.plot_alpha_vs_succprob:
                with open(conf.path_plot_alpha_vs_succprob, 'a') as file:
                    file.write(f"{quant_probs_t[i].item()} ")
                    file.write(f"{criterion.item()}") # expected_correct_prob.item()
                    file.write("\n")

            if criterion > curr_criterion:
                a_star_idx = i          # index for the quant value
                curr_criterion = criterion
                
        if all_a1_a2:
            # set all_a1_a2=True when searching for the best a1, a2
            return a_star_idx, curr_criterion, criterion

        return a_star_idx

    def error_given_test_set_given_a(self, X_test, y_test, w_matrix, alpha1_value, alpha2_value=None):
        """Empirical expert misprediction probability given the value of alpha or the values alpha_1, alpha_2 during test"""
        test_size = len(X_test)
        output_scores = 1 - self.model.predict_prob(X_test)
       
       # alphas and quantiles for shifted quantile method
        if alpha2_value is not None: 
            quant_prob_a2 = (np.ceil((1 - alpha2_value)*(self.calibration_size+1))/self.calibration_size)
            qhat_a2 = torch.quantile(self.conf_scores_t, quant_prob_a2)

        quant_prob = (np.ceil((1 - alpha1_value)*(self.calibration_size+1))/self.calibration_size)
         
        # move data to gpu if available
        quanta1_prob_t = torch.tensor(quant_prob, device=conf.device)
        qhata1_t = torch.quantile(self.conf_scores_t, quanta1_prob_t, keepdim=True).unsqueeze(1)
        y_test_t = torch.tensor(y_test, device=conf.device, dtype=torch.int64)
        output_scores_t = torch.tensor(output_scores, device=conf.device)
        if not self.has_groups:
            ws_t = torch.tensor(w_matrix[y_test], device=conf.device)
        else:
            ws_t = torch.tensor(w_matrix[y_test[:,1], y_test[:,0]], device=conf.device)
        
        fill_value_t = torch.exp(ws_t)/(torch.exp(ws_t).sum(axis=1).unsqueeze(1).expand(-1,conf.n_labels))

        qhats_a1 = qhata1_t.expand(test_size, conf.n_labels)
        # sets[sample][label] is 1 for the labels in the prediction set for each sample
        sets = torch.where(output_scores_t <= qhats_a1, 1, 0)
        if alpha2_value is not None:
            # sets for shifted quantile method given alpha_1
            qhats_a2 = torch.ones((test_size,conf.n_labels), device=conf.device)*qhat_a2
            sets_lower = torch.where(qhats_a2 < output_scores_t, 1, 0)
            sets = sets * sets_lower
        
        # denominators for  P[\hat Y = y ; C_alpha | y \in C_alpha(X)]
        sets_exp_ws = sets * torch.exp(ws_t)
        denominators_col = torch.sum(sets_exp_ws, axis=1)
        denominators = denominators_col.unsqueeze(1).expand(-1, conf.n_labels)

        # nominators for P[\hat Y = y ; C_alpha| y \in C_alpha(X)]
        nominators = sets_exp_ws        
    
        # confusion matrix for each prediction set 
        cm = torch.where(denominators>0, nominators/denominators, fill_value_t)

        # human predictions from prediction sets
        y_h = cm.multinomial(num_samples=1, replacement=True, generator=conf.torch_rng).squeeze()

        # set dummy prediction -1 for empty sets, so that it is counted as misprediction
        y_hats = torch.where(denominators_col>0, y_h , -1)

        # misprediction probability
        if not self.has_groups:
            errors = (y_hats!=y_test_t).count_nonzero().double()
        else:
            errors = (y_hats!=y_test_t[:,0]).count_nonzero().double()
            
        return errors/test_size

    def error_given_test_set_per_a(self, X_test, y_test, w_matrix, alphas, a_star_idx=None, star_dummy=None, full_cm=None, mult_h=None, y_humans=None, unc=None):
        """Empirical expert misprediction probability for each value of alpha or alpha_2 given alpha_1 during test"""
        test_size = len(X_test)                                     # 8000
        output_scores = 1 - self.model.predict_prob(X_test)         # 8000x10       # TODO Can we design a score function that is suitable for imbalanced data? or imbalance aware? But what is the problem with this score function?
        
        # alphas and quantiles for shifted quantile method
        if a_star_idx is not None: 
            quant_a1 = (np.ceil((1 - self.alphas[a_star_idx])*(self.calibration_size+1))/self.calibration_size)
            qhat_a1 = torch.quantile(self.conf_scores_t, quant_a1)
            alphas = self.alphas[self.alphas > self.alphas[a_star_idx]]

        quant_prob = (np.ceil((1 - alphas)*(self.calibration_size+1))/self.calibration_size)        # 1000, values from 0 -> 1
        
        error_rate_per_a = torch.zeros((len(quant_prob),), device=conf.device)                      # 1000, per alpha

        # move data to gpu if available
        quant_prob_t = torch.tensor(quant_prob, device=conf.device)                                 # TODO Is quantile calculation affected by imbalanced data? How so?
        qhats_t = torch.quantile(self.conf_scores_t, quant_prob_t, keepdim=True).unsqueeze(1)       # 1000x1x1      # conf_scores_t is based from cal
        y_test_t = torch.tensor(y_test, device=conf.device, dtype=torch.int64)      
        output_scores_t = torch.tensor(output_scores, device=conf.device)   # output_scores is based from test
        if not self.has_groups:                         # changed  
            assert len(conf.expert_matrix) == 1
            if "h_based_ws_t" in conf.expert_matrix:
                
                ws_t = torch.tensor(w_matrix[y_humans], device=conf.device)  
            elif "full_cm_ws_t" in conf.expert_matrix:
                # w_matrix_torch = torch.tensor(np.exp(w_matrix)).reshape(-1, w_matrix.shape[0], w_matrix.shape[1]).to(conf.device)
                # w_matrix_torch = torch.tensor(np.exp(w_matrix.T)).reshape(-1, w_matrix.shape[0], w_matrix.shape[1]).to(conf.device) # Transpose
                # full_cm_repeat = torch.tensor(full_cm, device=conf.device).reshape(full_cm.shape[0],full_cm.shape[1],1).repeat(1,1,full_cm.shape[1]).to(conf.device)
                # mean_cm_w_matrix = (w_matrix_torch * full_cm_repeat).mean(1) / (w_matrix_torch * full_cm_repeat).mean(1).sum(1).reshape(full_cm_repeat.shape[0], -1)
                # ws_t = torch.log(mean_cm_w_matrix)  
                ws_t = torch.tensor(np.log(full_cm+1e-8), device=conf.device)
            elif "orig" in conf.expert_matrix:
                ws_t = torch.tensor(w_matrix[y_test], device=conf.device)       # 8000x10       # many rows are repeated
            if unc is not None:                          # changed  
                unc_factor = 0.001                       # can be hyperparameter searched
                ws_t_normal = torch.normal(mean=torch.exp(ws_t), std=torch.from_numpy(unc*unc_factor).unsqueeze(1).repeat(1,10).to(conf.device))
                ws_t = torch.log(torch.clamp(ws_t_normal, min=0)/torch.clamp(ws_t_normal, min=0).sum(1).unsqueeze(1).repeat(1,conf.n_labels) + 1e-6)
        else:
            ws_t = torch.tensor(w_matrix[y_test[:,1], y_test[:,0]], device=conf.device)

        a_empty_sets = 0
        fill_value_t = torch.exp(ws_t)/(torch.exp(ws_t).sum(axis=1).unsqueeze(1).expand(-1, conf.n_labels)) # 8000x10  # sum per row is all 1 obviously
                        # so fill value is just the expert prediction
        
        
        for i,q in enumerate(qhats_t):

            qhats = q.expand(test_size, conf.n_labels)      # 8000x10
            # sets[sample][label] is 1 for the labels in the prediction set for each sample
            if a_star_idx is not None:
                # sets for shifted quantile method given alpha_1
                qhats_a1 = qhat_a1.expand(test_size,conf.n_labels )
                sets_upper = torch.where(output_scores_t <= qhats_a1 ,1 ,0)
                sets_lower = torch.where(qhats < output_scores_t, 1, 0)
                sets = sets_upper * sets_lower
            else:
                sets = torch.where(output_scores_t <= qhats, 1, 0)      # 8000x10   # prediction sets from conformal prediction (i see!)

            non_empty_sets = sets.sum(axis=1).count_nonzero()           # single value 
            
            if non_empty_sets==0 :
                a_empty_sets+=1

            sets_exp_ws = sets * torch.exp(ws_t)                                        #8000x10    # where conformal prediction meets the expert predictions
            denominators_col = torch.sum(sets_exp_ws, axis=1)                       #8000
            denominators = denominators_col.unsqueeze(1).expand(-1, conf.n_labels)  #8000x10

            # nominators for P[\hat Y = y ; C_alpha | y \in C_alpha(X)]
            nominators = sets_exp_ws          #8000x10
        
            # confusion matrix for each prediction set 
            cm = torch.where(denominators>0, nominators/denominators, fill_value_t)     #8000x10 cm rows sum to 1    # denominators>0 is usually True for all right?

            # human predictions from prediction sets
            y_h = cm.multinomial(num_samples=1, replacement=True, generator=conf.torch_rng).squeeze()       # 8000  # used multinomial to sample

            # set dummy prediction -1 for empty sets, so that it is counted as misprediction
            y_hats = torch.where(denominators_col>0, y_h , -1)      #8000
            # misprediction probability
            if not self.has_groups:
                errors = (y_hats!=y_test_t).count_nonzero().double()  # TODO Are misclassifications due to the samples from minority classes? Why and why not?
            else:
                errors = (y_hats!=y_test_t[:,0]).count_nonzero().double()

            error_rate_per_a[i] = errors/test_size

        ############################################################################################################################################################
        # CALCULATE FOR THE BEST
        for i,q in enumerate(qhats_t[star_dummy]):

            qhats = q.expand(test_size, conf.n_labels)      # 8000x10
            # sets[sample][label] is 1 for the labels in the prediction set for each sample
            if a_star_idx is not None:
                # sets for shifted quantile method given alpha_1
                qhats_a1 = qhat_a1.expand(test_size,conf.n_labels )
                sets_upper = torch.where(output_scores_t <= qhats_a1 ,1 ,0)
                sets_lower = torch.where(qhats < output_scores_t, 1, 0)
                sets = sets_upper * sets_lower
            else:
                sets = torch.where(output_scores_t <= qhats, 1, 0)      # 8000x10   # prediction sets from conformal prediction (i see!)

            # Calculate the metrics for sets such as class-conditional coverage
            # TODO In this metrics, see what changed from the original
            if conf.calc_metric_inference:     # changed  
                y_test_t_onehot = F.one_hot(y_test_t, w_matrix.shape[0])
                print('\nCls Coverage')
                for i in range(w_matrix.shape[0]):
                    # Class-specific coverage: for all class 0, how many are in the set? how many are not in the set
                    label_idx = y_test_t == i           # 4400
                    if label_idx.sum().item()>0:
                        sets_label_i = sets[label_idx]
                        got_correct = sets_label_i[:,i]
                        cls_cov = got_correct.sum() / label_idx.sum()       # 0.9862
                        print(f"{cls_cov} for cls {i}")
                    else:
                        print(f"missing for cls {i}")

            non_empty_sets = sets.sum(axis=1).count_nonzero()           # single value 
            
            if non_empty_sets==0 :
                a_empty_sets+=1


            # MULTIPLE EXPERTS (multiple confusion matrices)
            num_experts = 1              # changed  
            if num_experts == 1:
                # denominators for  P[\hat Y = y ; C_alpha | y \in C_alpha(X)]
                sets_exp_ws = sets * torch.exp(ws_t)                                    #8000x10    # where conformal prediction meets the expert predictions
                denominators_col = torch.sum(sets_exp_ws, axis=1)                           #8000
                denominators = denominators_col.unsqueeze(1).expand(-1, conf.n_labels)      #8000x10

                # nominators for P[\hat Y = y ; C_alpha | y \in C_alpha(X)]
                nominators = sets_exp_ws          #8000x10
            
                # confusion matrix for each prediction set 
                cm = torch.where(denominators>0, nominators/denominators, fill_value_t)     #8000x10 cm rows sum to 1    # denominators>0 is usually True for all right?

                # human predictions from prediction sets
                y_h = cm.multinomial(num_samples=1, replacement=True, generator=conf.torch_rng).squeeze()       # 8000  # used multinomial to sample

                # set dummy prediction -1 for empty sets, so that it is counted as misprediction
                y_hats = torch.where(denominators_col>0, y_h , -1)      #8000
            
            
            print(f"Sets avg {sets.sum().item()/sets.shape[0]}")
            # Misprediction probability
            if not self.has_groups:
                errors = (y_hats!=y_test_t).count_nonzero().double()    # TODO Are misclassifications due to the samples from minority classes? Why and why not?
                # Print total error and class-specific error            # changed  
                print(f"\nFinal_error {errors/y_test_t.shape[0]}")
                print("Error per class")
                for i in range(w_matrix.shape[0]): 
                    print((y_hats!=y_test_t)[y_test_t==i].sum() / (y_test_t==i).sum())      
                print("\n")
            else:
                errors = (y_hats!=y_test_t[:,0]).count_nonzero().double()

            error_rate_per_a[i] = errors/test_size

        return error_rate_per_a                                         # 1000

    def find_m(self, X_est, y_est, w_matrix, alphas, 
                        a_star_idx=None, 
                        star_dummy=None, 
                        full_cm=None, 
                        mult_h=None, 
                        y_humans=None, 
                        unc=None):

        m_opt_optimal = 1
        m_opt = 1
        cnt = 0
        m_best_err = 99999999
        m_err_list = []
        for m_ in range(1,len(y_humans)+1):
            err = self.error_multiple_simulated_humans(X_est, y_est, w_matrix, alphas, 
                                            a_star_idx=None, 
                                            star_dummy=star_dummy, 
                                            full_cm=full_cm, 
                                            mult_h=mult_h, 
                                            y_humans=y_humans[:m_], 
                                            unc=unc, 
                                            y_humans_est=None, 
                                            y_est=None, 
                                            X_est=None,
                                            find_m = True)
            est_team_acc = (stats.mode(y_humans[:m_],0)[0].reshape(-1)==y_est).sum()/len(y_est)
            print(f"est_team_accuracy {est_team_acc}")
            m_err_list.append(err[0].item())
            if err[0].item() < m_best_err and err[0].item() < 1-est_team_acc:
                m_best_err = err[0].item()
                m_opt_optimal = m_
                cnt =+ 1
            if err[0].item() < m_best_err:
                m_opt = m_
        # if cnt < 2:
        #     m_opt_optimal = 1
        if m_opt_optimal == 1:
            m_opt_optimal = 0
        return m_opt_optimal
        


    def error_multiple_simulated_humans(self, X_test, y_test, w_matrix, alphas, 
                                            a_star_idx=None, star_dummy=None, 
                                            full_cm=None, 
                                            mult_h=None, 
                                            y_humans=None, 
                                            unc=None, 
                                            y_humans_est=None, 
                                            y_est=None, 
                                            X_est=None,
                                            find_m=False,
                                            subset_select = 'greedy'):
        """Finding best prediction of multiple experts"""

        # Find the importance of each expert by specifying the weights
        if conf.sim_humans_select == 'weighted':
            mult_expert_weights = self.find_experts_weights(w_matrix, alphas, star_dummy, y_humans_est, y_est, X_est, mult_h)       # TODO Include y_cal here
        
        test_size = len(X_test)                                         # N
        output_scores = 1 - self.model.predict_prob(X_test)             # NxC

        # alphas and quantiles for shifted quantile method
        if a_star_idx is not None: 
            quant_a1 = (np.ceil((1 - self.alphas[a_star_idx])*(self.calibration_size+1))/self.calibration_size)
            qhat_a1 = torch.quantile(self.conf_scores_t, quant_a1)
            alphas = self.alphas[self.alphas > self.alphas[a_star_idx]]

        quant_prob = (np.ceil((1 - alphas)*(self.calibration_size+1))/self.calibration_size)                # 1000, values from 0 -> 1
        
        error_rate_per_a = torch.zeros((len(quant_prob),), device=conf.device)          # 1000              # per alpha

        # move data to gpu if available
        quant_prob_t = torch.tensor(quant_prob, device=conf.device)
        qhats_t = torch.quantile(self.conf_scores_t, quant_prob_t, keepdim=True).unsqueeze(1)  # 1000x1x1   # conf_scores_t is based from cal
        y_test_t = torch.tensor(y_test, device=conf.device, dtype=torch.int64)      
        output_scores_t = torch.tensor(output_scores, device=conf.device)               # output_scores is based from test
        
        # Find the subset of humans
        if subset_select == 'all':
            h_subset = None
        elif subset_select == 'random':
            h_subset = self.find_humans_subset(w_matrix, y_humans, len(y_humans), w_matrix[0].shape[0], qhats_t, star_dummy, a_star_idx, test_size, output_scores_t)
            h_subset = np.array(self.make_onehot(h_subset))
            set_avg = h_subset.sum() / test_size
            h_subset = self.find_humans_subset_random(test_size, len(mult_h), set_avg)
            h_subset = np.array(self.make_onehot(h_subset))
        elif subset_select == 'greedy':
            h_subset = self.find_humans_subset(w_matrix, y_humans, len(y_humans), w_matrix[0].shape[0], qhats_t, star_dummy, a_star_idx, test_size, output_scores_t)
            h_subset = np.array(self.make_onehot(h_subset))
        
        y_hats = []             # initialize empty list for the final prediction of each human expert
        for _, y_humans_item in enumerate(y_humans):    # list of len num_humans, each element size Ntest
            if not self.has_groups:                         # changed  
                assert len(conf.expert_matrix) == 1
                if "h_based_ws_t" in conf.expert_matrix:
                    # ws_t = torch.tensor(mult_h[human_idx][y_humans_item], device=conf.device)  # Case for different w_matrix for all humans
                    ws_t = torch.tensor(w_matrix[y_humans_item], device=conf.device)
                elif "full_cm_ws_t" in conf.expert_matrix:
                    ws_t = torch.tensor(full_cm, device=conf.device)  
                elif "orig" in conf.expert_matrix:
                    ws_t = torch.tensor(w_matrix[y_test], device=conf.device)       # 8000x10       # many rows are repeated
                
                if unc is not None:                          # changed  
                    unc_factor = 0.001                       # can be hyperparameter searched
                    ws_t_normal = torch.normal(mean=torch.exp(ws_t), std=torch.from_numpy(unc*unc_factor).unsqueeze(1).repeat(1,10).to(conf.device))
                    ws_t = torch.log(torch.clamp(ws_t_normal, min=0)/torch.clamp(ws_t_normal, min=0).sum(1).unsqueeze(1).repeat(1,conf.n_labels) + 1e-6)
            else:
                ws_t = torch.tensor(w_matrix[y_test[:,1], y_test[:,0]], device=conf.device)
    
            a_empty_sets = 0
            fill_value_t = torch.exp(ws_t)/(torch.exp(ws_t).sum(axis=1).unsqueeze(1).expand(-1, conf.n_labels))         # 8000x10  # sum per row is all 1 obviously
                                        # so fill value is just the expert prediction
            

            ########################################
            # CALCULATE FOR THE BEST
            q = qhats_t[star_dummy]
            qhats = q.expand(test_size, conf.n_labels)                  # NxC
            # sets[sample][label] is 1 for the labels in the prediction set for each sample
            if a_star_idx is not None:
                # sets for shifted quantile method given alpha_1
                qhats_a1 = qhat_a1.expand(test_size,conf.n_labels )
                sets_upper = torch.where(output_scores_t <= qhats_a1 ,1 ,0)
                sets_lower = torch.where(qhats < output_scores_t, 1, 0)
                sets = sets_upper * sets_lower
            else:
                sets = torch.where(output_scores_t <= qhats, 1, 0)      # NxC           # prediction sets from conformal prediction (i see!)

            # Calculate the metrics for sets such as class-conditional coverage
            if conf.calc_metric_inference and not find_m:     # changed  
                print('\nCls Coverage')
                for i in range(w_matrix.shape[0]):
                    # Class-specific coverage: for all class i, how many are in the set? how many are not in the set?
                    label_idx = y_test_t == i                               # N
                    if label_idx.sum().item()>0:
                        sets_label_i = sets[label_idx]
                        got_correct = sets_label_i[:,i]
                        cls_cov = got_correct.sum() / label_idx.sum()       # 0.9862
                        print(f"{cls_cov} for cls {i}")
                    else:
                        print(f"missing for cls {i}")

            non_empty_sets = sets.sum(axis=1).count_nonzero()               # single value 
            
            if non_empty_sets==0 :
                a_empty_sets+=1

            # denominators for  P[\hat Y = y ; C_alpha | y \in C_alpha(X)]
            sets_exp_ws = sets * torch.exp(ws_t)                                    #8000x10     # where conformal prediction meets the expert predictions
            denominators_col = torch.sum(sets_exp_ws, axis=1)                           #8000
            denominators = denominators_col.unsqueeze(1).expand(-1, conf.n_labels)      #8000x10

            # nominators for P[\hat Y = y ; C_alpha | y \in C_alpha(X)]
            nominators = sets_exp_ws          #8000x10
        
            # confusion matrix for each prediction set
            cm = torch.where(denominators>0, nominators/denominators, fill_value_t)     #8000x10 cm rows sum to 1       # denominators>0 is usually True for all right?

            # human predictions from prediction sets
            y_h = cm.multinomial(num_samples=1, replacement=True, generator=conf.torch_rng).squeeze()   # 8000  # used multinomial to sample

            # set dummy prediction -1 for empty sets, so that it is counted as misprediction
            y_hats.append(torch.where(denominators_col>0, y_h , -1))                    #8000
                
            if not find_m and h_subset is not None: print(f"Sets avg {sets.sum().item()/sets.shape[0]} human sets avg {h_subset.sum()/test_size}")
            elif not find_m: print(f"Sets avg {sets.sum().item()/sets.shape[0]}")

        y_hats_mult = torch.stack(y_hats)                       # 15x8000

        if conf.sim_humans_select == 'mode':
            if h_subset is not None:
                h_subset = torch.from_numpy(h_subset.T).to(conf.device)
                
                y_hats_mult += 1
                y_hats_mult = h_subset*y_hats_mult
                # Create a mask for zero values
                zero_mask = (y_hats_mult == 0).to(conf.device)
                # Generate random integers in the range [100, 10000000] for each zero
                random_numbers = torch.randint(100, 10000001, (zero_mask.sum(),)).to(conf.device)  # Upper bound is exclusive
                # Replace zero values with random numbers
                y_hats_mult[zero_mask] = random_numbers
                y_hats_mult -= 1
                y_hats, _ = torch.mode(y_hats_mult, dim=0)
                
                # Get indices of those having 1 expert only
                # indices_one_human = (h_subset.sum(0)== 1).nonzero(as_tuple=True)[0]
                
            else:
                y_hats, _ = torch.mode(y_hats_mult, dim=0)
                
        elif conf.sim_humans_select == 'weighted':
            mult_expert_weights = np.array(mult_expert_weights, dtype=np.float32)   # TODO Base the weight on human expert accuracy on the cal set or est set?
            # TODO Fix the nu to 100, not 15
            mult_expert_weights = torch.from_numpy(mult_expert_weights).cpu()
            final_pred = []
            for sample_idx in range(y_hats_mult.shape[1]):
                sample_i = y_hats_mult[:, sample_idx]
                pred_candidates = torch.unique(sample_i)
                cand_scores = {}
                for c in pred_candidates:
                    c_idx = (sample_i == c).nonzero(as_tuple=True)[0].cpu()
                    cand_scores[c.item()] = mult_expert_weights[c_idx].sum().item()
                sorted_cand_dict = dict(sorted(cand_scores.items(), key=lambda item: item[1], reverse=True))
                final_pred.append(next(iter(sorted_cand_dict)))     # pick highest accumulated weight
            y_hats = torch.Tensor(final_pred).to(conf.device)

        # Misprediction probability
        if not self.has_groups:
            errors = (y_hats!=y_test_t).count_nonzero().double() 
            
            # Print total error and class-specific error            # changed  
            print(f"\n Final_error {errors/y_test_t.shape[0]}")
            print("Error per class")
            for i in range(w_matrix.shape[0]): 
                print((y_hats!=y_test_t)[y_test_t==i].sum() / (y_test_t==i).sum())      
            print("\n")
        else:
            errors = (y_hats!=y_test_t[:,0]).count_nonzero().double()

        i = 0                                                   # TODO why fixed
        error_rate_per_a[i] = errors/test_size

        return error_rate_per_a                                 # N

    def error_multiple_simulated_humans_conformalized(self, X_test, y_test, w_matrix, alphas, a_star_idx=None, star_dummy=None, full_cm=None, mult_h=None, y_humans=None, unc=None, y_humans_est=None, y_est=None, X_est=None):
        """Finding best prediction using conformalized multiple experts"""
        
        test_size = len(X_test)                                 # 8000
        output_scores = 1 - self.model.predict_prob(X_test)     # 8000x10  
       
       # alphas and quantiles for shifted quantile method
        if a_star_idx is not None: 
            quant_a1 = (np.ceil((1 - self.alphas[a_star_idx])*(self.calibration_size+1))/self.calibration_size)
            qhat_a1 = torch.quantile(self.conf_scores_t, quant_a1)
            alphas = self.alphas[self.alphas > self.alphas[a_star_idx]]

        quant_prob = (np.ceil((1 - alphas)*(self.calibration_size+1))/self.calibration_size)        # 1000, values from 0 -> 1
         
        error_rate_per_a = torch.zeros((len(quant_prob),), device=conf.device) # 1000       # per alpha

        # move data to gpu if available
        quant_prob_t = torch.tensor(quant_prob, device=conf.device) # TODO Is quantile calculation affected by imbalanced data? How so?
        qhats_t = torch.quantile(self.conf_scores_t, quant_prob_t, keepdim=True).unsqueeze(1)  # 1000x1x1      # conf_scores_t is based from cal
        y_test_t = torch.tensor(y_test, device=conf.device, dtype=torch.int64)      
        output_scores_t = torch.tensor(output_scores, device=conf.device)   # output_scores is based from test

        # Write here a conformalized expert prediction with weights
        conformal_humans = []
        conformal_humans_weight = []
        y_humans_np = np.array(y_humans)
        for col_idx in range(y_humans_np.shape[1]):
            unique_values, counts = np.unique(y_humans_np[:, col_idx], return_counts=True)
            sample_mult_pred = []
            sample_mult_weight = []
            for idx, val in enumerate(unique_values):
                sample_mult_pred.append(val)
                sample_mult_weight.append(counts[idx]/np.sum(counts))
            conformal_humans.append(sample_mult_pred)
            conformal_humans_weight.append(sample_mult_weight)

        def pad_val(original_list, pad):
            # Find the maximum length of sublists
            max_length = max(len(sublist) for sublist in original_list)
            # Create a new list with padded sublists
            padded_list = [sublist + [pad] * (max_length - len(sublist)) for sublist in original_list]
            # Convert the list to a NumPy array
            result_array = np.array(padded_list)
            return result_array

        conformal_humans_np = pad_val(conformal_humans, pad=-1)
        conformal_humans_weight_np = pad_val(conformal_humans_weight, pad=0)

        y_hats = []
        if not self.has_groups:                         # changed  
            assert len(conf.expert_matrix) == 1
            if "h_based_ws_t" in conf.expert_matrix:
                ws_t = np.zeros((conformal_humans_np.shape[0],  conf.n_labels))
                for idx in range(conformal_humans_np.shape[1]):                 # TODO What if you include all the classes and use ALL expert data dist as weights
                    ws_t += w_matrix[conformal_humans_np[:,idx]] * conformal_humans_weight_np[:,idx].reshape(-1,1)
                ws_t = torch.tensor(ws_t, device=conf.device)
            elif "full_cm_ws_t" in conf.expert_matrix:
                raise NotImplementedError()
                # ws_t = torch.tensor(full_cm, device=conf.device)  
            elif "orig" in conf.expert_matrix:
                raise NotImplementedError()
                # ws_t = torch.tensor(w_matrix[y_test], device=conf.device)         # 8000x10       # many rows are repeated
            
            if unc is not None:                                                     # changed  
                raise NotImplementedError()
        else:
            ws_t = torch.tensor(w_matrix[y_test[:,1], y_test[:,0]], device=conf.device)

        a_empty_sets = 0
        fill_value_t = torch.exp(ws_t)/(torch.exp(ws_t).sum(axis=1).unsqueeze(1).expand(-1, conf.n_labels)) # 8000x10  # sum per row is all 1 obviously  # so fill value is just the expert prediction

        ############################################################################################################################################################
        # CALCULATE FOR THE BEST
        for i,q in enumerate(qhats_t[star_dummy]):

            qhats = q.expand(test_size, conf.n_labels)                      # 8000x10
            # sets[sample][label] is 1 for the labels in the prediction set for each sample
            if a_star_idx is not None:
                # sets for shifted quantile method given alpha_1
                qhats_a1 = qhat_a1.expand(test_size,conf.n_labels )
                sets_upper = torch.where(output_scores_t <= qhats_a1 ,1 ,0)
                sets_lower = torch.where(qhats < output_scores_t, 1, 0)
                sets = sets_upper * sets_lower
            else:
                sets = torch.where(output_scores_t <= qhats, 1, 0)          # 8000x10   # prediction sets from conformal prediction (i see!)

            # Calculate the metrics for sets such as class-conditional coverage
            if conf.calc_metric_inference:     # changed  
                y_test_t_onehot = F.one_hot(y_test_t, w_matrix.shape[0])
                print('\nCls Coverage')
                for i in range(w_matrix.shape[0]):
                    # Class-specific coverage: for all class 0, how many are in the set? how many are not in the set
                    label_idx = y_test_t == i           # 4400
                    if label_idx.sum().item()>0:
                        sets_label_i = sets[label_idx]
                        got_correct = sets_label_i[:,i]
                        cls_cov = got_correct.sum() / label_idx.sum()       # 0.9862
                        print(f"{cls_cov} for cls {i}")
                    else:
                        print(f"missing for cls {i}")

            non_empty_sets = sets.sum(axis=1).count_nonzero()           # single value 
            
            if non_empty_sets==0 :
                a_empty_sets+=1

            # MULTIPLE EXPERTS (multiple confusion matrices)
            num_experts = 1                                                                 # changed  
            if num_experts == 1:
                # denominators for  P[\hat Y = y ; C_alpha | y \in C_alpha(X)]
                sets_exp_ws = sets * torch.exp(ws_t)                                    #8000x10    # where conformal prediction meets the expert predictions
                denominators_col = torch.sum(sets_exp_ws, axis=1)                           #8000
                denominators = denominators_col.unsqueeze(1).expand(-1, conf.n_labels)      #8000x10

                # nominators for P[\hat Y = y ; C_alpha | y \in C_alpha(X)]
                nominators = sets_exp_ws          #8000x10
            
                # confusion matrix for each prediction set 
                cm = torch.where(denominators>0, nominators/denominators, fill_value_t)                         # 8000x10 cm rows sum to 1    # denominators>0 is usually True for all right?

                # human predictions from prediction sets
                y_h = cm.multinomial(num_samples=1, replacement=True, generator=conf.torch_rng).squeeze()       # 8000  # used multinomial to sample

                # set dummy prediction -1 for empty sets, so that it is counted as misprediction
                y_hats.append(torch.where(denominators_col>0, y_h , -1))      #8000

        y_hats_mult = torch.stack(y_hats)           # 15x8000

        if conf.sim_humans_select == 'mode':
            y_hats, _ = torch.mode(y_hats_mult, dim=0)

        # Misprediction probability
        if not self.has_groups:
            errors = (y_hats!=y_test_t).count_nonzero().double() 
            # Print total error and class-specific error            # changed  
            print(errors)
            print(y_test_t.shape)
            print(f"\nFinal_error {errors/y_test_t.shape[0]}")
            print("Error per class")
            for i in range(w_matrix.shape[0]): 
                print((y_hats!=y_test_t)[y_test_t==i].sum() / (y_test_t==i).sum())      
            print("\n")
        else:
            errors = (y_hats!=y_test_t[:,0]).count_nonzero().double()

        error_rate_per_a[i] = errors/test_size

        return error_rate_per_a                     # 1000

    def error_multiple_simulated_humans_conformalized_all(self, X_test, y_test, w_matrix, alphas, a_star_idx=None, star_dummy=None, full_cm=None, mult_h=None, y_humans=None, unc=None, y_humans_est=None, y_est=None, X_est=None):
        """Finding best prediction using conformalized multiple experts"""
        
        test_size = len(X_test)                                 # 8000
        output_scores = 1 - self.model.predict_prob(X_test)     # 8000x10  
       
       # alphas and quantiles for shifted quantile method
        if a_star_idx is not None: 
            quant_a1 = (np.ceil((1 - self.alphas[a_star_idx])*(self.calibration_size+1))/self.calibration_size)
            qhat_a1 = torch.quantile(self.conf_scores_t, quant_a1)
            alphas = self.alphas[self.alphas > self.alphas[a_star_idx]]

        quant_prob = (np.ceil((1 - alphas)*(self.calibration_size+1))/self.calibration_size)        # 1000, values from 0 -> 1
         
        error_rate_per_a = torch.zeros((len(quant_prob),), device=conf.device) # 1000       # per alpha

        # move data to gpu if available
        quant_prob_t = torch.tensor(quant_prob, device=conf.device) # TODO Is quantile calculation affected by imbalanced data? How so?
        qhats_t = torch.quantile(self.conf_scores_t, quant_prob_t, keepdim=True).unsqueeze(1)  # 1000x1x1      # conf_scores_t is based from cal
        y_test_t = torch.tensor(y_test, device=conf.device, dtype=torch.int64)      
        output_scores_t = torch.tensor(output_scores, device=conf.device)   # output_scores is based from test


        conformal_humans_np = np.arange(conf.n_labels).reshape(1,-1).repeat(X_test.shape[0],0)
        conformal_humans_weight_np = np.copy(full_cm)

        y_hats = []
        if not self.has_groups:                                                 # changed  
            assert len(conf.expert_matrix) == 1
            if "h_based_ws_t" in conf.expert_matrix:
                ws_t = np.zeros((conformal_humans_np.shape[0],  conf.n_labels))
                for idx in range(conformal_humans_np.shape[1]):                 # TODO What if you include all the classes and use ALL expert data dist as weights
                    ws_t += w_matrix[conformal_humans_np[:,idx]] * conformal_humans_weight_np[:,idx].reshape(-1,1)
                ws_t = torch.tensor(ws_t, device=conf.device)
            elif "full_cm_ws_t" in conf.expert_matrix:
                raise NotImplementedError()
                # ws_t = torch.tensor(full_cm, device=conf.device)  
            elif "orig" in conf.expert_matrix:
                raise NotImplementedError()
                # ws_t = torch.tensor(w_matrix[y_test], device=conf.device)     # 8000x10       # many rows are repeated
            
            if unc is not None:                                                 # changed  
                raise NotImplementedError()
        else:
            ws_t = torch.tensor(w_matrix[y_test[:,1], y_test[:,0]], device=conf.device)

        a_empty_sets = 0
        fill_value_t = torch.exp(ws_t)/(torch.exp(ws_t).sum(axis=1).unsqueeze(1).expand(-1, conf.n_labels)) # 8000x10  # sum per row is all 1 obviously  # so fill value is just the expert prediction
        
        ############################################################################################################################################################
        # CALCULATE FOR THE BEST
        for i,q in enumerate(qhats_t[star_dummy]):

            qhats = q.expand(test_size, conf.n_labels)                      # 8000x10
            # sets[sample][label] is 1 for the labels in the prediction set for each sample
            if a_star_idx is not None:
                # sets for shifted quantile method given alpha_1
                qhats_a1 = qhat_a1.expand(test_size,conf.n_labels )
                sets_upper = torch.where(output_scores_t <= qhats_a1 ,1 ,0)
                sets_lower = torch.where(qhats < output_scores_t, 1, 0)
                sets = sets_upper * sets_lower
            else:
                sets = torch.where(output_scores_t <= qhats, 1, 0)          # 8000x10   # prediction sets from conformal prediction (i see!)

            # Calculate the metrics for sets such as class-conditional coverage
            if conf.calc_metric_inference:     # changed  
                y_test_t_onehot = F.one_hot(y_test_t, w_matrix.shape[0])
                print('\nCls Coverage')
                for i in range(w_matrix.shape[0]):
                    # Class-specific coverage: for all class 0, how many are in the set? how many are not in the set
                    label_idx = y_test_t == i           # 4400
                    if label_idx.sum().item()>0:
                        sets_label_i = sets[label_idx]
                        got_correct = sets_label_i[:,i]
                        cls_cov = got_correct.sum() / label_idx.sum()       # 0.9862
                        print(f"{cls_cov} for cls {i}")
                    else:
                        print(f"missing for cls {i}")

            non_empty_sets = sets.sum(axis=1).count_nonzero()           # single value 
            
            if non_empty_sets==0 :
                a_empty_sets+=1

            # MULTIPLE EXPERTS (multiple confusion matrices)
            num_experts = 1                                                                 # changed  
            if num_experts == 1:
                # denominators for  P[\hat Y = y ; C_alpha | y \in C_alpha(X)]
                sets_exp_ws = sets * torch.exp(ws_t)                                    #8000x10    # where conformal prediction meets the expert predictions
                denominators_col = torch.sum(sets_exp_ws, axis=1)                           #8000
                denominators = denominators_col.unsqueeze(1).expand(-1, conf.n_labels)      #8000x10

                # nominators for P[\hat Y = y ; C_alpha | y \in C_alpha(X)]
                nominators = sets_exp_ws          #8000x10
            
                # confusion matrix for each prediction set 
                cm = torch.where(denominators>0, nominators/denominators, fill_value_t)                         # 8000x10 cm rows sum to 1    # denominators>0 is usually True for all right?

                # human predictions from prediction sets
                y_h = cm.multinomial(num_samples=1, replacement=True, generator=conf.torch_rng).squeeze()       # 8000  # used multinomial to sample

                # set dummy prediction -1 for empty sets, so that it is counted as misprediction
                y_hats.append(torch.where(denominators_col>0, y_h , -1))      #8000

        y_hats_mult = torch.stack(y_hats)           # 15x8000

        if conf.sim_humans_select == 'mode':
            y_hats, _ = torch.mode(y_hats_mult, dim=0)

        # Misprediction probability
        if not self.has_groups:
            errors = (y_hats!=y_test_t).count_nonzero().double() 
            # Print total error and class-specific error            # changed  
            print(f"\nFinal_error {errors/y_test_t.shape[0]}")
            print("Error per class")
            for i in range(w_matrix.shape[0]): 
                print((y_hats!=y_test_t)[y_test_t==i].sum() / (y_test_t==i).sum())      
            print("\n")
        else:
            errors = (y_hats!=y_test_t[:,0]).count_nonzero().double()

        error_rate_per_a[i] = errors/test_size

        return error_rate_per_a                     # 1000

    def find_experts_weights(self, w_matrix, alphas, star_dummy, y_humans_est, y_est, X_est, mult_h, a_star_idx=None, unc=None):    # TODO remove a_star_idx
        est_size = len(X_est)                                       # 1000
        output_scores = 1 - self.model.predict_prob(X_est)          # 1000x10  
        
        # alphas and quantiles for shifted quantile method
        if a_star_idx is not None: 
            quant_a1 = (np.ceil((1 - self.alphas[a_star_idx])*(self.calibration_size+1))/self.calibration_size)
            qhat_a1 = torch.quantile(self.conf_scores_t, quant_a1)
            alphas = self.alphas[self.alphas > self.alphas[a_star_idx]]

        quant_prob = (np.ceil((1 - alphas)*(self.calibration_size+1))/self.calibration_size)        # 1000, values from 0 -> 1

        error_rate_per_a = torch.zeros((len(quant_prob),), device=conf.device)                      # 1000       # per alpha

        # move data to gpu if available
        quant_prob_t = torch.tensor(quant_prob, device=conf.device)
        qhats_t = torch.quantile(self.conf_scores_t, quant_prob_t, keepdim=True).unsqueeze(1)       # 1000x1x1      # conf_scores_t is based from cal
        y_est_t = torch.tensor(y_est, device=conf.device, dtype=torch.int64)      
        output_scores_t = torch.tensor(output_scores, device=conf.device)                           # output_scores is based from est
        
        y_hats = []
        for human_idx, y_humans_item in enumerate(y_humans_est):
            if not self.has_groups:                             # changed  
                assert len(conf.expert_matrix) == 1
                if "h_based_ws_t" in conf.expert_matrix:
                    ws_t = torch.tensor(mult_h[human_idx][y_humans_item], device=conf.device)  # Case for different w_matrix for all humans
                    # ws_t = torch.tensor(w_matrix[y_humans_item], device=conf.device)  
                elif "full_cm_ws_t" in conf.expert_matrix:
                    ws_t = torch.tensor(full_cm, device=conf.device)  
                elif "orig" in conf.expert_matrix:
                    ws_t = torch.tensor(w_matrix[y_est], device=conf.device)       # 8000x10       # many rows are repeated
                if unc is not None:                             # changed  
                    unc_factor = 0.001                       # can be hyperparameter searched
                    ws_t_normal = torch.normal(mean=torch.exp(ws_t), std=torch.from_numpy(unc*unc_factor).unsqueeze(1).repeat(1,10).to(conf.device))
                    ws_t = torch.log(torch.clamp(ws_t_normal, min=0)/torch.clamp(ws_t_normal, min=0).sum(1).unsqueeze(1).repeat(1,conf.n_labels) + 1e-6)
            else:
                ws_t = torch.tensor(w_matrix[y_est[:,1], y_est[:,0]], device=conf.device)
    
            a_empty_sets = 0
            fill_value_t = torch.exp(ws_t)/(torch.exp(ws_t).sum(axis=1).unsqueeze(1).expand(-1, conf.n_labels)) # 8000x10  # sum per row is all 1 obviously
                            # so fill value is just the expert prediction
            
            ##################################################################################################################
            # CALCULATE FOR THE BEST
            for i,q in enumerate(qhats_t[star_dummy]):
                qhats = q.expand(est_size, conf.n_labels)                   # 8000x10
                # sets[sample][label] is 1 for the labels in the prediction set for each sample
                if a_star_idx is not None:
                    # sets for shifted quantile method given alpha_1
                    qhats_a1 = qhat_a1.expand(est_size,conf.n_labels )
                    sets_upper = torch.where(output_scores_t <= qhats_a1 ,1 ,0)
                    sets_lower = torch.where(qhats < output_scores_t, 1, 0)
                    sets = sets_upper * sets_lower
                else:
                    sets = torch.where(output_scores_t <= qhats, 1, 0)      # 8000x10   # prediction sets from conformal prediction (i see!)

                non_empty_sets = sets.sum(axis=1).count_nonzero()           # single value 
                
                if non_empty_sets==0 :
                    a_empty_sets+=1

                # denominators for  P[\hat Y = y ; C_alpha | y \in C_alpha(X)]
                sets_exp_ws = sets * torch.exp(ws_t)                                    #8000x10    # where conformal prediction meets the expert predictions
                denominators_col = torch.sum(sets_exp_ws, axis=1)                           #8000
                denominators = denominators_col.unsqueeze(1).expand(-1, conf.n_labels)      #8000x10

                # nominators for P[\hat Y = y ; C_alpha | y \in C_alpha(X)]
                nominators = sets_exp_ws          #8000x10
            
                # confusion matrix for each prediction set 
                cm = torch.where(denominators>0, nominators/denominators, fill_value_t)     #8000x10 cm rows sum to 1    # denominators>0 is usually True for all right?

                # human predictions from prediction sets
                y_h = cm.multinomial(num_samples=1, replacement=True, generator=conf.torch_rng).squeeze()       # 8000  # used multinomial to sample

                # set dummy prediction -1 for empty sets, so that it is counted as misprediction
                y_hats.append(torch.where(denominators_col>0, y_h , -1))      #8000
            
        y_hats_mult = torch.stack(y_hats)
        error = []
        for idx in range(len(y_hats_mult)):
            # Misprediction probability
            if not self.has_groups:
                errors = (y_hats_mult[idx]!=y_est_t).count_nonzero().double().item()/est_size
            else:
                errors = (y_hats!=y_est_t[:,0]).count_nonzero().double()
            error.append(errors)        # Ex.   [0.1, 0.095, 0.089, 0.092, 0.104, 0.091, 0.101, 0.114, 0.107, 0.095, 0.095, 0.098, 0.097, 0.093, 0.088]
        
        # Calculate the weights based on the error
        weights = [(1-err) for err in error]
        final_weights = [0.50 + 0.50 * ((wei - min(weights)) / (max(weights) - min(weights))) for wei in weights]   # Final weights are based on the error on est data

        return final_weights            

    def find_humans_subset(self, w_matrix, y_humans, num_humans, num_classes, qhats_t, star_dummy, a_star_idx, test_size, output_scores_t):
        q = qhats_t[star_dummy]
        qhats = q.expand(test_size, conf.n_labels)                  # NxC
        # sets[sample][label] is 1 for the labels in the prediction set for each sample
        if a_star_idx is not None:
            pass
            # # sets for shifted quantile method given alpha_1
            # qhats_a1 = qhat_a1.expand(test_size,conf.n_labels )
            # sets_upper = torch.where(output_scores_t <= qhats_a1 ,1 ,0)
            # sets_lower = torch.where(qhats < output_scores_t, 1, 0)
            # sets = sets_upper * sets_lower
        else:
            sets = torch.where(output_scores_t <= qhats, 1, 0)      # NxC           # prediction sets from conformal prediction (i see!)
                    
        def f(x):
            return x / (1 - x)
        
        optimal = []

        for idx, p in enumerate(np.array(y_humans).T):                                    # 8000x13                     # iterate for each sample
            m = np.array([[np.exp(w_matrix)[j][p[i]] for j in range(num_classes)] for i in range(num_humans)])   # 13 x 10, for each class, what are the preds of the 13 humans?
            m = m*sets[idx].cpu().numpy()
            m = m / m.sum(1).reshape(m.shape[0],-1)   # normalize
            m = f(m)
            m *= (m > 1)            # zero the m <= 1       # Why the threshold value 1?                # NOTE really accurate predictions have m>1?
            m += (m == 0) * 1       # for those zero, put value 1

            y_opt = np.argmax(np.prod(m, axis=0))       # usually all zero? multiplying human predictions per class

            optimal.append([i for i, x in enumerate(m[:, y_opt]) if x != 1])        # choose which human at the optimal class is not equal to 1?
            if len(optimal[-1]) == 0:
                optimal[-1] = [i for i, x in enumerate(range(m.shape[0]))]
        return optimal
    
    def find_humans_subset_random(self, test_size, cls, recorded_avg):
        import random

        # Parameters
        num_lists = test_size

        # Calculate average sum per inner list
        average_sum_per_list = recorded_avg

        # Create the list of lists
        A = []
        

        # Generate inner lists
        for _ in range(num_lists):
            choices = [val for val in range(0,cls)]
            # Generate a list that sums to average_sum_per_list
            inner_list = []
            cnt = 0
            
            while (random.uniform(0, cls) < average_sum_per_list and cnt < 3) or len(inner_list) == 0:
                # Randomly choose a value (0, 1, or 2)
                value = random.choice(choices)
                
                choices.remove(value)
                inner_list.append(value)
                cnt += 1
            
            # Shuffle the inner list for randomness
            inner_list = sorted(inner_list)
            A.append(inner_list)
        
        return A
    
    def make_onehot(self, original_list):
        # Determine the size of the output based on the maximum index
        max_index = max(max(sublist) for sublist in original_list)

        # Create an output list filled with zeros
        output_list = [[0] * (max_index + 1) for _ in range(len(original_list))]

        # Populate the output list based on the original list
        for i, sublist in enumerate(original_list):
            for index in sublist:
                output_list[i][index] = 1

        return output_list
            
    def test_error_robustness(self, p, X_test, y_test, w_matrix, alpha1_value, alpha2_value=None):
        """Empirical expert misprediction probability during test under IIA violations"""
        test_size = len(X_test)
        output_scores = 1 - self.model.predict_prob(X_test)
       
       # alphas and quantiles for shifted quantile method
        if alpha2_value is not None: 
            quant_prob_a2 = (np.ceil((1 - alpha2_value)*(self.calibration_size+1))/self.calibration_size)
            qhat_a2 = torch.quantile(self.conf_scores_t, quant_prob_a2)

        quant_prob = (np.ceil((1 - alpha1_value)*(self.calibration_size+1))/self.calibration_size)

        # move data to gpu if available
        quanta1_prob_t = torch.tensor(quant_prob, device=conf.device)
        qhata1_t = torch.quantile(self.conf_scores_t, quanta1_prob_t, keepdim=True).unsqueeze(1)
        y_test_t = torch.tensor(y_test, device=conf.device, dtype=torch.int64)
        output_scores_t = torch.tensor(output_scores, device=conf.device)
        ws_t = torch.tensor(w_matrix[y_test], device=conf.device)
        fill_value_t = torch.exp(ws_t)/(torch.exp(ws_t).sum(axis=1).unsqueeze(1).expand(-1, conf.n_labels))
        qhats_a1 = qhata1_t.expand(test_size, conf.n_labels)
        
        # sets[sample][label] is 1 for the labels in the prediction set for each sample
        sets = torch.where(output_scores_t <= qhats_a1, 1, 0)
        if alpha2_value is not None:
            # sets for shifted quantile method given alpha_1
            qhats_a2 = torch.ones((test_size,conf.n_labels), device=conf.device)*qhat_a2
            sets_lower = torch.where(qhats_a2 < output_scores_t, 1, 0)
            sets = sets * sets_lower
        
        # denominators for  P[\hat Y = y ; C_alpha | y \in C_alpha(X)]
        sets_exp_ws = sets * torch.exp(ws_t)

        # nominators for P[\hat Y = y ; C_alpha | y \in C_alpha(X)]
        nominators = sets_exp_ws        

        # sets that include the true label
        one_hot_ycal = F.one_hot(y_test_t)
        mask_sets_with_true_labels = (sets * one_hot_ycal).sum(axis=1, keepdim=True).expand(test_size, conf.n_labels)

        # labels excluded from the sets
        mass_of_labels_not_in_the_set = p * ((1 - sets)*torch.exp(ws_t)).sum(axis=1, keepdim=True).expand(test_size, conf.n_labels)

        # sizes of prediction sets - 1
        sets_sizes_minus1 = sets.sum(axis=1, keepdim=True).expand(test_size, conf.n_labels) - 1
        
        mass_to_add_in_false_labels = torch.where( (one_hot_ycal==0) & (sets==1) & (mask_sets_with_true_labels==1), mass_of_labels_not_in_the_set/sets_sizes_minus1, 0)
        
        # tweaked nominators
        tweaked_nominators = nominators+mass_to_add_in_false_labels 
        denominators_col = tweaked_nominators.sum(axis=1)
        denominators = tweaked_nominators.sum(axis=1, keepdim=True).expand(test_size, conf.n_labels)
        tweaked_cm = torch.where(denominators>0, tweaked_nominators/denominators, fill_value_t)

        # human predictions from prediction sets
        y_h = tweaked_cm.multinomial(num_samples=1, replacement=True, generator=conf.torch_rng).squeeze()

        # set dummy prediction -1 for empty sets, so that it is counted as misprediction
        y_hats = torch.where(denominators_col>0, y_h , -1)
       
        # misprediction probability
        errors = (y_hats!=y_test_t).count_nonzero().double()
            
        return errors/test_size

    def size_given_test_set_per_a(self, X_test, alphas, a_star_idx=None):
        """Empirical average set size for each value of alpha or alpha_2 given alpha_1"""
        test_size = len(X_test)
        output_scores = 1 - self.model.predict_prob(X_test)
        
        if a_star_idx is not None: 
            # alphas and quantiles for shifted quantile method
            quant_a1 = (np.ceil((1 - alphas[a_star_idx])*(self.calibration_size+1))/self.calibration_size)
            qhat_a1 = torch.quantile(self.conf_scores_t, quant_a1)
            alphas =  alphas[alphas > alphas[a_star_idx]]

        quant_prob_t = torch.tensor(np.ceil((1 - alphas)*(self.calibration_size+1))/self.calibration_size, device=conf.device)
        set_size_per_a = torch.zeros((len(alphas),), device=conf.device)

        # move data to gpu if available
        qhats_t = torch.quantile(self.conf_scores_t, quant_prob_t, keepdim=True).unsqueeze(1)
        output_scores_t = torch.tensor(output_scores, device=conf.device)

        for i,q in enumerate(qhats_t):

            qhats = q.expand(test_size, conf.n_labels)
            # sets[sample][label] is 1 for the labels in the prediction set for each sample
            if a_star_idx is not None:
                # sets for shifted quantile method
                qhats_a1 = qhat_a1.expand(test_size, conf.n_labels)
                sets_upper = torch.where(output_scores_t <= qhats_a1, 1, 0)
                sets_lower = torch.where(qhats < output_scores_t, 1, 0)
                sets = sets_upper * sets_lower
            else:
                sets = torch.where(output_scores_t <= qhats, 1, 0)
            size_per_set = sets.sum(axis=1)
            set_size_per_a[i] = size_per_set.sum()/size_per_set.numel()
               
        return set_size_per_a    

    def error_given_test_set_topk(self, X_test, y_test, w_matrix, y_humans, k=5):
        """Emprical misprediction probability of an expert using a top-k predictor"""
        test_size = len(X_test)
        output_scores = self.model.predict_prob(X_test)
        error_rate_per_a = torch.zeros((1,), device=conf.device)

        # move data to gpu if available
        y_test_t = torch.tensor(y_test, device=conf.device, dtype=torch.int64)
        output_scores_t = torch.tensor(output_scores, device=conf.device)
        ws_t = torch.tensor(w_matrix[y_humans], device=conf.device)
        fill_value_t = torch.exp(ws_t)/(torch.exp(ws_t).sum(axis=1).unsqueeze(1).expand(-1, conf.n_labels))
        
        # compute topk labels for each sample
        sorted_output_scores = torch.topk(output_scores_t, k=k, dim=1).indices
        
        # prediction sets with topk labels
        sets = torch.any(F.one_hot(sorted_output_scores, num_classes=conf.n_labels), 1).int()

        # denominators for  P[\hat Y = y ; C_k | y \in C_k(X)]
        sets_exp_ws = sets * torch.exp(ws_t)
        denominators_col = torch.sum(sets_exp_ws, axis=1)
        denominators = denominators_col.unsqueeze(1).expand(-1, conf.n_labels)

        # nominators for P[\hat Y = y ; C_k | y \in C_k(X)]
        nominators = sets_exp_ws        
    
        # confusion matrix for each prediction set 
        cm = torch.where(denominators>0, nominators/denominators, fill_value_t)

        # human predictions from prediction sets
        y_h = cm.multinomial(num_samples=1, replacement=True, generator=conf.torch_rng).squeeze()

        # set dummy prediction -1 for empty sets, so that it is counted as misprediction
        y_hats = torch.where(denominators_col>0, y_h , -1)
        
        # misprediction probability
        errors = (y_hats!=y_test_t).count_nonzero().double()
        # Print total error and class-specific error      
        print(f"\n Final_error {errors/y_test_t.shape[0]}")
        print("Error per class")
        for i in range(w_matrix.shape[0]): 
            print((y_hats!=y_test_t)[y_test_t==i].sum() / (y_test_t==i).sum())    
        print("\n")

        error_rate_per_a = errors/test_size
            
        return error_rate_per_a

    def error_given_test_set_topk_multiple(self, X_test, y_test, w_matrix, y_humans, subset_select, k=5):
        """Emprical misprediction probability of an expert using a top-k predictor"""
        test_size = len(X_test)
        output_scores = self.model.predict_prob(X_test)
        error_rate_per_a = torch.zeros((1,), device=conf.device)

        # move data to gpu if available
        y_test_t = torch.tensor(y_test, device=conf.device, dtype=torch.int64)
        output_scores_t = torch.tensor(output_scores, device=conf.device)
        
        if subset_select == 'all':
            h_subset = None
        elif subset_select == 'random':
            h_subset = self.find_humans_subset_topk(w_matrix, y_humans, len(y_humans), w_matrix[0].shape[0], test_size, output_scores_t, k)
            h_subset = np.array(self.make_onehot(h_subset))
            set_avg = h_subset.sum() / test_size
            h_subset = self.find_humans_subset_random(test_size, len(y_humans), set_avg)
            h_subset = np.array(self.make_onehot(h_subset))
        elif subset_select == 'greedy':
            h_subset = self.find_humans_subset_topk(w_matrix, y_humans, len(y_humans), w_matrix[0].shape[0], test_size, output_scores_t, k)
            h_subset = np.array(self.make_onehot(h_subset))

        y_hats = []
        for human_idx, y_humans_item in enumerate(y_humans):
            ws_t = torch.tensor(w_matrix[y_humans_item], device=conf.device)
            fill_value_t = torch.exp(ws_t)/(torch.exp(ws_t).sum(axis=1).unsqueeze(1).expand(-1, conf.n_labels))
            
            # compute topk labels for each sample
            sorted_output_scores = torch.topk(output_scores_t, k=k, dim=1).indices
            
            # prediction sets with topk labels
            sets = torch.any(F.one_hot(sorted_output_scores, num_classes=conf.n_labels), 1).int()

            # denominators for  P[\hat Y = y ; C_k | y \in C_k(X)]
            sets_exp_ws = sets * torch.exp(ws_t)
            denominators_col = torch.sum(sets_exp_ws, axis=1)
            denominators = denominators_col.unsqueeze(1).expand(-1, conf.n_labels)

            # nominators for P[\hat Y = y ; C_k | y \in C_k(X)]
            nominators = sets_exp_ws        
        
            # confusion matrix for each prediction set 
            cm = torch.where(denominators>0, nominators/denominators, fill_value_t)

            # human predictions from prediction sets
            y_h = cm.multinomial(num_samples=1, replacement=True, generator=conf.torch_rng).squeeze()

            # set dummy prediction -1 for empty sets, so that it is counted as misprediction
            y_hats.append(torch.where(denominators_col>0, y_h , -1))
        
        y_hats_mult = torch.stack(y_hats)                       # 15x8000

        if conf.sim_humans_select == 'mode':
            if h_subset is not None:
                h_subset = torch.from_numpy(h_subset.T).to(conf.device)
                y_hats_mult += 1
                y_hats_mult = h_subset * y_hats_mult
                # Create a mask for zero values
                zero_mask = (y_hats_mult == 0).to(conf.device)

                # Generate random integers in the range [100, 10000000] for each zero
                random_numbers = torch.randint(100, 10000001, (zero_mask.sum(),)).to(conf.device)    # Upper bound is exclusive

                # Replace zero values with random numbers
                y_hats_mult[zero_mask] = random_numbers
                y_hats_mult -= 1
            y_hats, _ = torch.mode(y_hats_mult, dim=0)

        # misprediction probability
        errors = (y_hats!=y_test_t).count_nonzero().double()
        # Print total error and class-specific error      
        print(f"\n Final_error {errors/y_test_t.shape[0]}")
        print("Error per class")
        for i in range(w_matrix.shape[0]): 
            print((y_hats!=y_test_t)[y_test_t==i].sum() / (y_test_t==i).sum())      
        print("\n")

        error_rate_per_a = errors/test_size
            
        return error_rate_per_a
    
    def find_humans_subset_topk(self, w_matrix, y_humans, num_humans, num_classes, test_size, output_scores_t, k):
        # compute topk labels for each sample
        sorted_output_scores = torch.topk(output_scores_t, k=k, dim=1).indices
        
        # prediction sets with topk labels
        sets = torch.any(F.one_hot(sorted_output_scores, num_classes=conf.n_labels), 1).int()
                    
        def f(x):
            return x / (1 - x)
        
        optimal = []

        for idx, p in enumerate(np.array(y_humans).T):                                    # 8000x13                     # iterate for each sample
            m = np.array([[np.exp(w_matrix)[j][p[i]] for j in range(num_classes)] for i in range(num_humans)])   # 13 x 10, for each class, what are the preds of the 13 humans?
            m = m * sets[idx].cpu().numpy()
            m = m / m.sum(1).reshape(m.shape[0],-1)   # normalized
            m = f(m)
            m *= (m > 1)            # zero the m <= 1       # Why the threshold value 1?                # NOTE really accurate predictions have m>1?
            m += (m == 0) * 1       # for those zero, put value 1

            y_opt = np.argmax(np.prod(m, axis=0))       # usually all zero? multiplying human predictions per class

            optimal.append([i for i, x in enumerate(m[:, y_opt]) if x != 1])        # choose which human at the optimal class is not equal to 1?
            if len(optimal[-1]) == 0:
                optimal[-1] = [i for i, x in enumerate(range(m.shape[0]))]
        return optimal
    
    def size_given_test_set_given_a(self, X_test, alpha, alpha2=None):
        """Set size distribution for given alpha or alpha_1, alpha_2 during test"""
        test_size = len(X_test)
        output_scores = 1 - self.model.predict_prob(X_test)
    
        if alpha2 is not None: 
            # alphas and quantiles for shifted quantile method
            quant_a1 = (np.ceil((1 - alpha)*(self.calibration_size+1))/self.calibration_size)
            qhat_a1 = torch.quantile(self.conf_scores_t, quant_a1)
            alpha = alpha2

        quant_prob_t = torch.tensor(np.ceil((1 - alpha)*(self.calibration_size+1))/self.calibration_size, device=conf.device)

        # move data to gpu if available
        qhat_t = torch.quantile(self.conf_scores_t, quant_prob_t)
        output_scores_t = torch.tensor(output_scores,device=conf.device)
        qhats = qhat_t.expand(test_size, conf.n_labels)

        # sets[sample][label] is 1 for the labels in the prediction set for each sample
        if alpha2 is not None:
            # sets for shifted quantile method
            qhats_a1 = qhat_a1.expand(test_size, conf.n_labels)
            sets_upper = torch.where(output_scores_t <= qhats_a1, 1, 0)
            sets_lower = torch.where(qhats < output_scores_t, 1, 0)
            sets = sets_upper * sets_lower
        else:
            sets = torch.where(output_scores_t <= qhats, 1, 0)
        
        size_per_set = sets.sum(axis=1)
        set_sizes_t, counts_t = torch.unique(size_per_set, return_counts=True)
        
        return set_sizes_t, counts_t

    def empirical_coverage(self, X_test, y_test, alpha1, alpha2=None):
        """Empirical coverage on test set given alpha or alpha1, alpha2"""
        test_size = len(X_test)
        output_scores = 1 - self.model.predict_prob(X_test)
       
       # quantiles for shifted quantile method
        if alpha2 is not None: 
            quant_a2 = (np.ceil((1 - alpha2)*(self.calibration_size+1))/self.calibration_size)
            qhat_a2 = torch.quantile(self.conf_scores_t, quant_a2)

        quant_prob_t = torch.tensor(np.ceil((1 - alpha1)*(self.calibration_size+1))/self.calibration_size, device=conf.device)

        # move data to gpu if available
        qhat = torch.quantile(self.conf_scores_t, quant_prob_t) 
        y_test_t = torch.tensor(y_test, device=conf.device, dtype=torch.int64)
        output_scores_t = torch.tensor(output_scores, device=conf.device)
        qhats = torch.ones((test_size,conf.n_labels), device=conf.device)*qhat

        # sets[sample][label] is 1 for the labels in the prediction set for each sample
        sets = torch.where(output_scores_t <= qhats, 1, 0)
        if alpha2 is not None:
            # sets for shifted quantile method given alpha_1
            qhats_a2 = torch.ones((test_size,conf.n_labels), device=conf.device)*qhat_a2
            sets_lower = torch.where(qhats_a2 < output_scores_t, 1, 0)
            sets = sets * sets_lower
        
        one_hot_ycal = F.one_hot(y_test_t)
        # mask for prediction sets that include the true label
        true_label_in_sets = sets * one_hot_ycal
        return true_label_in_sets.sum()/test_size    

    # add this policy; changed
    def policy(self, confusion_matrix, hx, tx, mx, num_humans, num_classes=10):

        def f(x):
            return x / (1 - x)

        policy_name = "pseudo_lb_best_policy_overloaded"

        optimal = []

        for p in hx:                    #10000x13 # iterate for each sample
            m = np.array([[f(confusion_matrix[i][p[i]][j]) for j in range(num_classes)] for i in range(num_humans)])   # 13 x 10, for each sample, what are the preds of the 13 humans?
            m = np.exp(m)
            m *= (m > 1)                # zero the m <= 1 # Why the value 1? # NOTE really accurate predictions have m>1?
            m += (m == 0) * 1           # for those zero, put value 1

            y_opt = np.argmax(np.prod(m, axis=0))                                   # usually all zero? multiplying human predictions per class

            optimal.append([i for i, x in enumerate(m[:, y_opt]) if x != 1])        # choose which human at the optimal class is not equal to 1?
        
        return optimal