
import sys,os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np
import torch

from tqdm import tqdm
import pandas as pd
from baseline_methods.baseline import *
from tabular.tab_models.tab_model import *
from tabular.tabular_data_utils.tabular_dataset import *
from sklearn.linear_model import LinearRegression, LogisticRegression
from econml.metalearners import TLearner
from bartpy.sklearnmodel import SklearnModel
from tvae.tvae import *
from sklearn.model_selection import GridSearchCV

classical_baseline_ls = ["Ganite", "dt", "lr", "rf", "causal_rf", "bart", "tvae"]

def compute_cont_outcome_error(test_dataset, est, X, A, method_name):
    t_grid = test_dataset.t_grid
    n_test = t_grid.shape[1]
    t_grid_hat = torch.zeros(2, n_test)
    t_grid_hat[0, :] = t_grid[0, :]

            

    for n in tqdm(range(n_test)):

        # for step, batch in enumerate(test_loader):
            
        #         # batch = (x.cuda() for x in batch)
        #     # text_id, text_len, text_mask, A, _, Y, (origin_all_other_pats_ls, X_pd_ls) = batch
        #     # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, text_id_ls, text_mask_ls, text_len_ls), Y, A = batch
        #     # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, X), Y, A = batch
        #     idx, origin_X, A, Y, count_Y, D, X, origin_all_other_pats_ls = batch
        #     X =X.to(self.device)
        #     Y = Y.to(self.device)

        A *= 0
        A += t_grid[0,n].item()
        
        if not method_name == "Ganite":
            if not method_name == "bart":
                pred_outcome = est._ortho_learner_model_final._model_final._model.predict(est._ortho_learner_model_final._model_final._combine(X, A))
            else:
                pred_outcome = est.predict(np.concatenate([X, A.reshape(-1,1)], axis=1))
        
        #     all_outcome_ls.append(pred.view(-1))
            
        #     all_gt_treatment_ls.append(A.cpu().view(-1))
        #     all_gt_outcome_ls.append(Y.cpu().view(-1))
        
        # # all_treatment_pred_tensor = torch.cat(all_treatment_ls)
        # all_outcome_pred_tensor = torch.cat(all_outcome_ls)
        # all_gt_outcome_tensor = torch.cat(all_gt_outcome_ls)
        # all_gt_treatment_tensor = torch.cat(all_gt_treatment_ls)
        t_grid_hat[1,n] = np.mean(pred_outcome)
    mse = ((t_grid_hat[1, :].squeeze() - t_grid[1, :].squeeze()) ** 2).mean().data

    print("outcome loss::%f"%(mse))
    
def compute_cont_outcome_error0(method_name, est, X, A, Y):
    if not method_name == "Ganite":
        if not method_name == "bart":
            pred_outcome = est._ortho_learner_model_final._model_final._model.predict(est._ortho_learner_model_final._model_final._combine(X, A))
        else:
            pred_outcome = est.predict(np.concatenate([X, A.reshape(-1,1)], axis=1))

    mse = np.mean((pred_outcome.reshape(-1) - Y.reshape(-1)) ** 2)
    print("outcome loss 0::%f"%(mse))


def compute_outcome_error(est, X, A, Y, method_name):
    # t_grid = test_dataset.t_grid
    # n_test = t_grid.shape[1]
    # t_grid_hat = torch.zeros(2, n_test)
    # t_grid_hat[0, :] = t_grid[0, :]

            

    # for n in tqdm(range(n_test)):

        # for step, batch in enumerate(test_loader):
            
        #         # batch = (x.cuda() for x in batch)
        #     # text_id, text_len, text_mask, A, _, Y, (origin_all_other_pats_ls, X_pd_ls) = batch
        #     # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, text_id_ls, text_mask_ls, text_len_ls), Y, A = batch
        #     # (idx, sample_idx, origin_all_other_pats_ls, X_pd_ls, X), Y, A = batch
        #     idx, origin_X, A, Y, count_Y, D, X, origin_all_other_pats_ls = batch
        #     X =X.to(self.device)
        #     Y = Y.to(self.device)

        # A *= 0
        # A += t_grid[0,n]
    if method_name == "bart":
        pred_outcome = est.predict(np.concatenate([X, A.reshape(-1,1)], axis=1))
        mse = np.mean((pred_outcome.reshape(-1) - Y.reshape(-1)) ** 2)
    elif method_name == "tvae":
        pred_outcome = eval_vae_outcome(est, torch.from_numpy(X), torch.from_numpy(A))
        mse = np.mean((pred_outcome.reshape(-1) - Y.reshape(-1)) ** 2)
    else:
        if not method_name == "Ganite":
            pred_outcome = est._ortho_learner_model_final._model_final._model.predict(est._ortho_learner_model_final._model_final._combine(X, A))
            mse = np.mean((pred_outcome.reshape(-1) - Y.reshape(-1)) ** 2)
        else:
            all_outcome = est.inference_nets(torch.from_numpy(X).cuda())
            mse = ((all_outcome.detach().cpu()[torch.arange(len(all_outcome)), A.long().reshape(-1)].reshape(-1) - Y.reshape(-1)) ** 2).mean().data
        #     all_outcome_ls.append(pred.view(-1))
            
        #     all_gt_treatment_ls.append(A.cpu().view(-1))
        #     all_gt_outcome_ls.append(Y.cpu().view(-1))
        
        # # all_treatment_pred_tensor = torch.cat(all_treatment_ls)
        # all_outcome_pred_tensor = torch.cat(all_outcome_ls)
        # all_gt_outcome_tensor = torch.cat(all_gt_outcome_ls)
        # all_gt_treatment_tensor = torch.cat(all_gt_treatment_ls)
        # t_grid_hat[1,n] = pred_outcome.mean()
    # mse = ((t_grid_hat[1, :].squeeze() - t_grid[1, :].squeeze()) ** 2).mean().data

    print("outcome loss::%f"%(mse))


def random_sampling(X, T, W, Y, ratio=0.2):
    perturbed_ids = torch.randperm(len(X))
    selected_perturb_ids = perturbed_ids[0:int(ratio*len(X))]
    sub_X = X[selected_perturb_ids]
    sub_T = T[selected_perturb_ids]
    sub_W = None
    if W is not None:
        sub_W = W[selected_perturb_ids]
    sub_Y = Y[selected_perturb_ids]
    return sub_X, sub_T, sub_W, sub_Y

def construct_explanations_by_keys(est, test_dataset):
    explanation_key_to_ids_mappings = dict()

    for idx in range(len(test_dataset.origin_features)):
        decision_rule = extract_decision_rules(est, test_dataset.num_cols + test_dataset.cat_cols, test_dataset.origin_features[idx].numpy())
        

        all_decision_key = ""
        # for k in range(self.num_treatments):
        curr_rule = decision_rule[0]
        curr_rule.sort()
        
        # decision_rule_str, selected_col_ids = extract_decision_rules(curr_tree, test_loader.dataset.num_cols + test_loader.dataset.cat_cols, X.numpy(), return_features=True)
        
        decision_key = " ".join(curr_rule)
        
        all_decision_key += decision_key + " "
        if all_decision_key not in explanation_key_to_ids_mappings:
            explanation_key_to_ids_mappings[all_decision_key] = []
        explanation_key_to_ids_mappings[all_decision_key].append(idx)

    return explanation_key_to_ids_mappings


def compute_eval_metrics_baseline(meta_info, test_dataset, num_treatments, do_prediction, train=False):
        mises = []
        ites = []
        dosage_policy_errors = []
        policy_errors = []
        pred_best = []
        pred_vals = []
        true_best = []

        samples_power_of_two = 6
        num_integration_samples = 2 ** samples_power_of_two + 1
        step_size = 1. / num_integration_samples
        treatment_strengths = np.linspace(np.finfo(float).eps, 1, num_integration_samples)

        # for patient in test_patients:
        # for step, batch in enumerate(test_loader):
        with torch.no_grad():
            for idx, origin_X, A, Y, count_Y, D, patient, all_other_pats_ls in tqdm(test_dataset):
                if train and len(pred_best) > 10:
                    return np.sqrt(np.mean(mises)), np.sqrt(np.mean(dosage_policy_errors)), np.sqrt(np.mean(policy_errors)), np.mean(ites)
                for treatment_idx in range(num_treatments):
                    test_data = dict()
                    test_data['x'] = np.repeat(np.expand_dims(patient, axis=0), num_integration_samples, axis=0)
                    X = test_data['x']
                    X_pd_full = X
                    origin_X = X
                    test_data['t'] = np.repeat(treatment_idx, num_integration_samples)
                    A = test_data["t"]
                    test_data['d'] = treatment_strengths
                    D = test_data["d"]
                    X_D = np.concatenate([X, D.reshape(-1,1)], axis=-1)
                    origin_all_other_pats_ls= [all_other_pats_ls.clone() for _ in range(num_integration_samples)]
                    pred_dose_response = do_prediction(X_D, A)
                    # pred_dose_response = get_model_predictions(num_treatments=num_treatments, test_data=test_data, model=model)
                    # pred_dose_response = pred_dose_response * (
                    #         dataset['metadata']['y_max'] - dataset['metadata']['y_min']) + \
                    #                         dataset['metadata']['y_min']

                    true_outcomes = [get_patient_outcome(patient, meta_info, treatment_idx, d) for d in
                                        treatment_strengths]
                    
                    # if len(pred_best) < num_treatments and train == False:
                    #     #print(true_outcomes)
                    #     print([item[0] for item in pred_dose_response])
                    mise = romb(np.square(true_outcomes - pred_dose_response), dx=step_size)
                    inter_r = np.array(true_outcomes) - pred_dose_response.squeeze()
                    ite = np.mean(inter_r ** 2)
                    mises.append(mise)
                    ites.append(ite)

                    best_encountered_x = treatment_strengths[np.argmax(pred_dose_response)]

                    def pred_dose_response_curve(dosage):
                        test_data = dict()
                        test_data['x'] = np.expand_dims(patient, axis=0)
                        test_data['t'] = np.expand_dims(treatment_idx, axis=0)
                        test_data['d'] = np.expand_dims(dosage, axis=0)
                        X = test_data['x']
                        X_pd_full = X
                        origin_X = X
                        # X = X.to(device)
                        A = test_data["t"]
                        D = test_data["d"]
                        X_D = np.concatenate([X, D.reshape(-1,1)], axis=-1)
                        
                        ret_val =do_prediction(X_D, A)
                        # ret_val = get_model_predictions(num_treatments=num_treatments, test_data=test_data, model=model)
                        # ret_val = ret_val * (dataset['metadata']['y_max'] - dataset['metadata']['y_min']) + \
                        #             dataset['metadata']['y_min']
                        return ret_val

                    true_dose_response_curve = get_true_dose_response_curve(meta_info, patient, treatment_idx)

                    min_pred_opt = minimize(lambda x: -1 * pred_dose_response_curve(x),
                                            x0=[best_encountered_x], method="SLSQP", bounds=[(0, 1)])

                    max_pred_opt_y = - min_pred_opt.fun
                    max_pred_dosage = min_pred_opt.x
                    max_pred_y = true_dose_response_curve(max_pred_dosage)

                    min_true_opt = minimize(lambda x: -1 * true_dose_response_curve(x),
                                            x0=[0.5], method="SLSQP", bounds=[(0, 1)])
                    max_true_y = - min_true_opt.fun
                    max_true_dosage = min_true_opt.x

                    dosage_policy_error = (max_true_y - max_pred_y) ** 2
                    dosage_policy_errors.append(dosage_policy_error)

                    pred_best.append(max_pred_opt_y)
                    pred_vals.append(max_pred_y)
                    true_best.append(max_true_y)
                    

                selected_t_pred = np.argmax(pred_vals[-num_treatments:])
                selected_val = pred_best[-num_treatments:][selected_t_pred]
                selected_t_optimal = np.argmax(true_best[-num_treatments:])
                optimal_val = true_best[-num_treatments:][selected_t_optimal]
                policy_error = (optimal_val - selected_val) ** 2
                policy_errors.append(policy_error)

        return np.sqrt(np.mean(mises)), np.sqrt(np.mean(dosage_policy_errors)), np.sqrt(np.mean(policy_errors)), np.mean(ites)



def classical_baseline_main(args, method_name, X, T, W, Y, valid_X, valid_T, valid_W, valid_Y, test_X, test_T,  test_W, test_Y, dataset, test_dataset, count_Y=None, valid_count_Y=None, test_count_Y=None, classification=False):
    # if args.dataset_name == "tcga":
    #     X, T, W, Y = random_sampling(X,T, W, Y)
    if W is not None:
        W = W.view(-1,1)
        valid_W =  valid_W.view(-1,1)
        test_W =  test_W.view(-1,1)       
        X = torch.cat([X, W], dim=-1)
        valid_X=  torch.cat([valid_X, valid_W], dim=-1)
        test_X=  torch.cat([test_X, test_W], dim=-1)
    

    if type(X) is torch.Tensor:
        X =X.numpy()
    if type(valid_X) is torch.Tensor:
        valid_X = valid_X.numpy()
    if type(test_X) is torch.Tensor:
        test_X = test_X.numpy()
    Y = transform_outcome_by_rescale_back(dataset, Y)
    valid_Y = transform_outcome_by_rescale_back(dataset, valid_Y)
    test_Y = transform_outcome_by_rescale_back(dataset, test_Y)
    if count_Y is not None:
        count_Y = transform_outcome_by_rescale_back(dataset, count_Y)
        if type(count_Y) is torch.Tensor:
            count_Y = count_Y.numpy()
    if valid_count_Y is not None:
        valid_count_Y = transform_outcome_by_rescale_back(dataset, valid_count_Y)
        if type(valid_count_Y) is torch.Tensor:
            valid_count_Y = valid_count_Y.numpy()
    if test_count_Y is not None:
        test_count_Y = transform_outcome_by_rescale_back(dataset, test_count_Y)
        if type(test_count_Y) is torch.Tensor:
            test_count_Y = test_count_Y.numpy()
    
    if type(T) is torch.Tensor:
        T = T.numpy()
    if type(valid_T) is torch.Tensor:
        valid_T = valid_T.numpy()
    if type(test_T) is torch.Tensor:
        test_T = test_T.numpy()
    if type(Y) is torch.Tensor:
        Y = Y.numpy()
    if type(valid_Y) is torch.Tensor:
        valid_Y = valid_Y.numpy()
    if type(test_Y) is torch.Tensor:
        test_Y = test_Y.numpy()
    if not args.cont_treatment:
        T = T.astype(int)
        valid_T = valid_T.astype(int)
        test_T = test_T.astype(int)
    cv_count=2 #min(10, args.program_max_len)
    if method_name == "Ganite":
        est = ganite(X, T, Y)
        train_ites = est(X)
        train_ites = train_ites.detach().cpu().numpy()
        valid_ites = est(valid_X)
        valid_ites = valid_ites.detach().cpu().numpy()
        test_ites = est(test_X)
        test_ites = test_ites.detach().cpu().numpy()
    elif method_name == "dt":
        if classification:
            y_model = lambda: GridSearchCV(
                estimator=DecisionTreeClassifier(random_state=args.seed),
                param_grid={
                        'max_depth': [k for k in range(1, args.program_max_len+1, cv_count)],
                        # 'n_estimators': (10, 30, 50, 100, 200),
                        # 'max_features': (1,2,3)
                    }, cv=10, n_jobs=-1, scoring='neg_mean_squared_error'
                )
            # y_model = DecisionTreeClassifier(max_depth=args.program_max_len, random_state=args.seed)
        else:
            y_model = lambda: GridSearchCV(
                estimator=DecisionTreeRegressor(random_state=args.seed),
                param_grid={
                        'max_depth': [k for k in range(1, args.program_max_len+1, cv_count)],
                        # 'n_estimators': (10, 30, 50, 100, 200),
                        # 'max_features': (1,2,3)
                    }, cv=10, n_jobs=-1, scoring='neg_mean_squared_error'
                )
            # y_model = DecisionTreeRegressor(max_depth=args.program_max_len, random_state=args.seed)
            
        if not args.cont_treatment:
            if not args.dataset_name == "tcga":
                model_t = lambda: GridSearchCV(
                    estimator=DecisionTreeClassifier(random_state=args.seed),
                    param_grid={
                            'max_depth': [k for k in range(1, args.program_max_len+1, cv_count)],
                            # 'n_estimators': (10, 30, 50, 100, 200),
                            # 'max_features': (1,2,3)
                        }, cv=10, n_jobs=-1, scoring='neg_mean_squared_error'
                    )
                est = LinearDML(model_y=y_model(), model_t=model_t(), random_state=args.seed)
                
                # model_t = DecisionTreeClassifier(max_depth=args.program_max_len, random_state=args.seed)
                # est = LinearDML(model_y=y_model, model_t=model_t, random_state=args.seed)
            else:
                est = TLearner(models=y_model())#LinearDML(model_y=y_model, model_t=DecisionTreeClassifier(max_depth=args.program_max_len))
        else:
            if not args.dataset_name == "tcga":
                model_t = lambda: GridSearchCV(
                    estimator=DecisionTreeRegressor(random_state=args.seed),
                    param_grid={
                            'max_depth': [k for k in range(1, args.program_max_len+1, cv_count)],
                            # 'n_estimators': (10, 30, 50, 100, 200),
                            # 'max_features': (1,2,3)
                        }, cv=10, n_jobs=-1, scoring='neg_mean_squared_error'
                    )
                est = LinearDML(model_y=y_model(), model_t=model_t(), random_state=args.seed)
                # model_t = DecisionTreeRegressor(max_depth=args.program_max_len, random_state=args.seed)
                # est = LinearDML(model_y=y_model, model_t=model_t, random_state=args.seed)
            else:
                est = TLearner(models=y_model())

        est.fit(Y, T, X=X)
        train_ites = est.effect(X)
        valid_ites = est.effect(valid_X)
        test_ites = est.effect(test_X)
    elif method_name == "lr":
        if classification:
            y_model = LogisticRegression(multi_class="multinomial", random_state=args.seed)
        else:
            y_model = LinearRegression()
        if not args.cont_treatment:
            # if not args.dataset_name == "tcga":
            #     est = LinearDML(model_y=y_model, model_t=LogisticRegression(random_state=args.seed), random_state=args.seed)
            # else:
            est = TLearner(models=y_model)
        else:
            # if not args.dataset_name == "tcga":
            #     est = LinearDML(model_y=y_model, model_t=LinearRegression(), random_state=args.seed)
            # else:
            est = TLearner(models=y_model)
        
        est.fit(Y, T, X=X)
        train_ites = est.effect(X)
        valid_ites = est.effect(valid_X)
        test_ites = est.effect(test_X)
    elif method_name == "rf":
        if classification:
            y_model = lambda: GridSearchCV(
                estimator=RandomForestClassifier(random_state=args.seed),
                param_grid={
                        'max_depth': [k for k in range(1, args.program_max_len+1, cv_count)],
                        'n_estimators': [k for k in range(1, args.topk_act+1, cv_count)] + [args.topk_act],
                        # 'max_features': (1,2,3)
                    }, cv=10, n_jobs=-1, scoring='neg_mean_squared_error'
                )
            discrete_outcome = True
            # y_model = RandomForestClassifier(n_estimators=args.topk_act, max_depth=args.program_max_len, random_state=args.seed)
        else:
            y_model = lambda: GridSearchCV(
                estimator=RandomForestRegressor(random_state=args.seed),
                param_grid={
                        'max_depth': [k for k in range(1, args.program_max_len+1, cv_count)],
                        'n_estimators': [k for k in range(1, args.topk_act+1, cv_count)] + [args.topk_act],
                        # 'max_features': (1,2,3)
                    }, cv=10, n_jobs=-1, scoring='neg_mean_squared_error'
                )
            
            discrete_outcome = False
            
            # y_model = RandomForestRegressor(n_estimators=args.topk_act, max_depth=args.program_max_len, random_state=args.seed, )

        if not args.cont_treatment:
            if not args.dataset_name == "tcga":
                model_t = lambda: GridSearchCV(
                    estimator=RandomForestClassifier(random_state=args.seed),
                    param_grid={
                            'max_depth': [k for k in range(1, args.program_max_len+1, cv_count)],
                            'n_estimators': [k for k in range(1, args.topk_act+1, cv_count)] + [args.topk_act],
                            # 'max_features': (1,2,3)
                        }, cv=10, n_jobs=-1, scoring='neg_mean_squared_error'
                )
                est = LinearDML(model_y=y_model(), model_t=model_t(), random_state=args.seed, discrete_treatment=True)
                
                # model_t = RandomForestClassifier(n_estimators=args.topk_act, max_depth=args.program_max_len, random_state=args.seed)
                # est = LinearDML(model_y=y_model, model_t=model_t, random_state=args.seed, discrete_treatment=True)
                
            else:
                est = TLearner(models=y_model())
        else:
            if not args.dataset_name == "tcga":
                model_t = lambda: GridSearchCV(
                    estimator=RandomForestRegressor(random_state=args.seed),
                    param_grid={
                            'max_depth': [k for k in range(1, args.program_max_len+1, cv_count)],
                            'n_estimators': [k for k in range(1, args.topk_act+1, cv_count)] + [args.topk_act],
                            # 'max_features': (1,2,3)
                        }, cv=10, n_jobs=-1, scoring='neg_mean_squared_error'
                )
                est = LinearDML(model_y=y_model(), model_t=model_t(), random_state=args.seed, discrete_treatment=False)
                # model_t = RandomForestRegressor(n_estimators=args.topk_act, max_depth=args.program_max_len, random_state=args.seed)
                # est = LinearDML(model_y=y_model, model_t=model_t, random_state=args.seed, discrete_treatment=False)
            else:
                est = TLearner(models=y_model())
        
        est.fit(Y, T, X=X)
        train_ites = est.effect(X)
        valid_ites = est.effect(valid_X)
        test_ites = est.effect(test_X)
    elif method_name == "causal_rf":
        
        if classification:
            y_model = lambda: GridSearchCV(
                estimator=RandomForestClassifier(random_state=args.seed),
                param_grid={
                        'max_depth': [k for k in range(1, args.program_max_len+1, cv_count)],
                        'n_estimators': [k for k in range(1, args.topk_act+1, cv_count)] + [args.topk_act],
                        # 'max_features': (1,2,3)
                    }, cv=10, n_jobs=-1, scoring='neg_mean_squared_error'
                )
            # y_model = RandomForestClassifier(n_estimators=args.topk_act, max_depth=args.program_max_len, random_state=args.seed)
        else:
            y_model = lambda: GridSearchCV(
                estimator=RandomForestRegressor(random_state=args.seed),
                param_grid={
                        'max_depth': [k for k in range(1, args.program_max_len+1, cv_count)],
                        'n_estimators': [k for k in range(1, args.topk_act+1, cv_count)] + [args.topk_act],
                        # 'max_features': (1,2,3)
                    }, cv=10, n_jobs=-1, scoring='neg_mean_squared_error'
                )
            
            # y_model = RandomForestRegressor(n_estimators=args.topk_act, max_depth=args.program_max_len, random_state=args.seed, )

        if not args.cont_treatment:
            # if not args.dataset_name == "tcga":
                model_t = lambda: GridSearchCV(
                    estimator=RandomForestClassifier(random_state=args.seed),
                    param_grid={
                            'max_depth': [k for k in range(1, args.program_max_len+1, cv_count)],
                            'n_estimators': [k for k in range(1, args.topk_act+1, cv_count)] + [args.topk_act],
                            # 'max_features': (1,2,3)
                        }, cv=10, n_jobs=-1, scoring='neg_mean_squared_error'
                )
                
            #     est = LinearDML(model_y=y_model(), model_t=model_t(), random_state=args.seed)
            # else:
            #     est = TLearner(models=y_model())
        else:
            # if not args.dataset_name == "tcga":
                model_t = lambda: GridSearchCV(
                    estimator=RandomForestRegressor(random_state=args.seed),
                    param_grid={
                            'max_depth': [k for k in range(1, args.program_max_len+1, cv_count)],
                            'n_estimators': [k for k in range(1, args.topk_act+1, cv_count)] + [args.topk_act],
                            # 'max_features': (1,2,3)
                        }, cv=10, n_jobs=-1, scoring='neg_mean_squared_error'
                )
                # est = LinearDML(model_y=y_model(), model_t=model_t(), random_state=args.seed)
        est = CausalForestDML(model_y=y_model(), model_t=model_t(), random_state=args.seed)
            # else:
            #     est = TLearner(models=y_model())
        # n_estimators = (int(args.topk_act/4) + 1)*4
        # est = CausalForest(criterion='het', n_estimators=n_estimators, max_depth=args.program_max_len,
        #            min_var_fraction_leaf=None, min_var_leaf_on_val=True,
        #            min_impurity_decrease = 0.0, max_samples=0.45, min_balancedness_tol=.45,
        #            warm_start=False, inference=True, fit_intercept=True, subforest_size=4,
        #            honest=True, verbose=0, n_jobs=-1, random_state=args.seed, )
        


        # if classification:
        #     y_model = RandomForestClassifier(n_estimators=args.topk_act, max_depth=args.program_max_len)
        # else:
        #     y_model = RandomForestRegressor(n_estimators=args.topk_act, max_depth=args.program_max_len)
        # # est = causal_forest(X.numpy(), T.view(-1).numpy(), Y.view(-1).numpy())
        # if not args.cont_treatment:
        #     est = CausalForestDML(max_depth=args.program_max_len, n_estimators=4, discrete_treatment=True, model_t=RandomForestClassifier(n_estimators=args.topk_act), model_y=y_model)
        # else:
        #     est = CausalForestDML(max_depth=args.program_max_len, n_estimators=4, discrete_treatment=False, model_t=RandomForestRegressor(n_estimators=args.topk_act), model_y=y_model)
        # est.fit(Y, T, X=X)
        est.fit(X=X, T=T, Y=Y)
        train_ites = est.effect(X)
        valid_ites = est.effect(valid_X)
        test_ites = est.effect(test_X)
        if args.is_log:
            construct_explanations_by_keys(est[0], test_dataset)

                
            # decision_rule = extract_decision_rules(est[0], test_dataset.num_cols + test_dataset.cat_cols, test_dataset.origin_features.numpy())
        # pred_y = causal_forest_predict(model, test_X.numpy(), test_T.numpy())
    elif method_name == "bart":
        est = SklearnModel(n_trees=args.topk_act)        
        est.fit(np.concatenate([X, T.reshape(-1,1)],axis=1), Y.reshape(-1))
        train_ites = est.predict(np.concatenate([X, np.ones_like(T).reshape(-1,1)], axis=1)) - est.predict(np.concatenate([X, np.zeros_like(T).reshape(-1,1)],axis=1))
        valid_ites = est.predict(np.concatenate([valid_X, np.ones_like(valid_T).reshape(-1,1)],axis=1)) - est.predict(np.concatenate([valid_X, np.zeros_like(valid_T).reshape(-1,1)],axis=1))
        test_ites = est.predict(np.concatenate([test_X, np.ones_like(test_T).reshape(-1,1)],axis=1)) - est.predict(np.concatenate([test_X, np.zeros_like(test_T).reshape(-1,1)],axis=1))
    
    
    elif method_name == "tvae":
        est = init_tvae_model(X.shape[1], batch_size=256)
        est = train_vae_model(est, torch.from_numpy(X), torch.from_numpy(T), torch.from_numpy(Y))
        train_ites = eval_vae_ite(est, torch.from_numpy(X), torch.from_numpy(T))
        valid_ites = eval_vae_ite(est, torch.from_numpy(valid_X), torch.from_numpy(valid_T))
        test_ites = eval_vae_ite(est, torch.from_numpy(test_X), torch.from_numpy(test_T))
    
    if count_Y is not None:
        gt_train_ites = (count_Y - Y).reshape(-1)*(T==0).astype(float).reshape(-1) + (Y - count_Y).reshape(-1)*(T==1).astype(float).reshape(-1)
    if valid_count_Y is not None:
        gt_valid_ites = (valid_count_Y - valid_Y).reshape(-1)*(valid_T==0).astype(float).reshape(-1) + (valid_Y - valid_count_Y).reshape(-1)*(valid_T==1).astype(float).reshape(-1)
    if test_count_Y is not None:
        gt_test_ites = (test_count_Y - test_Y).reshape(-1)*(test_T==0).astype(float).reshape(-1) + (test_Y - test_count_Y).reshape(-1)*(test_T==1).astype(float).reshape(-1)

    if not args.cont_treatment and not args.has_dose and args.num_treatments == 2:
        if count_Y is not None:
            best_train_ite, best_train_ate = np.mean(np.abs(train_ites- gt_train_ites)).item(), np.abs(np.mean(train_ites- gt_train_ites)).item()
            best_val_ite, best_val_ate = np.mean(np.abs(valid_ites- gt_valid_ites)).item(), np.abs(np.mean(valid_ites- gt_valid_ites)).item()
            best_test_ite, best_test_ate = np.mean(np.abs(test_ites- gt_test_ites)).item(), np.abs(np.mean(test_ites- gt_test_ites)).item()

            print("best train ite::%f"%(best_train_ite))
            print("best validation ite::%f"%(best_val_ite))
            print("best test ite::%f"%(best_test_ite))
            
            print("best train ate::%f"%(best_train_ate))
            print("best validation ate::%f"%(best_val_ate))
            print("best test ate::%f"%(best_test_ate))
            print()
        else:
            train_ate = np.abs(np.mean(train_ites)).item()
            valid_ate = np.abs(np.mean(valid_ites)).item()
            test_ate = np.abs(np.mean(test_ites)).item()
            
            print("best train ate::%f"%(train_ate))
            print("best validation ate::%f"%(valid_ate))
            print("best test ate::%f"%(test_ate))
        # if not method_name == "causal_rf":
        compute_outcome_error(est, X, T, Y, method_name)
        compute_outcome_error(est, test_X, test_T, test_Y, method_name)
        print()
        # else:
        #     plt.figure(figsize=(20, 10))
        #     plot_tree(est[0], impurity=True, max_depth=3)
        #     plt.savefig(os.path.join(args.log_folder, "casual_rf.png"))

    elif args.has_dose:
        
            def tcga_pred_function(X_D, A):
                res = np.zeros(len(X_D))
                if method_name == "bart":
                    if np.sum(A==1) > 0:
                        res[A==1] = est.predict(np.concatenate([X_D[A==1], np.ones_like(A[A==1]).reshape(-1,1)], axis=1)).reshape(-1)
                    if np.sum(A==0) > 0:
                        res[A==0] = est.predict(np.concatenate([X_D[A==0], np.zeros_like(A[A==0]).reshape(-1,1)], axis=1)).reshape(-1)
                # elif method_name == "causal_rf":
                #     if np.sum(A==1) > 0:
                #         res[A==1] = est._ortho_learner_model_final._model_final._model.predict(est._ortho_learner_model_final._model_final._combine(X_D[A==1], A[A==1])).reshape(-1)
                #     if np.sum(A==0) > 0:
                #         res[A==0] = est._ortho_learner_model_final._model_final._model.predict(est._ortho_learner_model_final._model_final._combine(X_D[A==0], A[A==0])).reshape(-1)
                elif method_name == "tvae":
                    if np.sum(A==1) > 0:
                        res[A==1] = eval_vae_outcome(est, X_D[A==1], A[A==1])
                    if np.sum(A==0) > 0:
                        res[A==0] = eval_vae_outcome(est, X_D[A==0], A[A==0])
                else:   
                    for tr_idx in range(args.num_treatments):
                        if np.sum(A==tr_idx) > 0:
                            res[A==tr_idx] = est.models[tr_idx].predict(X_D[A==tr_idx]).reshape(-1)
                    # if np.sum(A==0) > 0:
                    #     res[A==0] = est.models[0].predict(X_D[A==0]).reshape(-1)
                return res
                
            if not method_name == "bart":
                mise, dpe, pe, ate = compute_eval_metrics_baseline(dataset.metainfo, dataset, args.num_treatments, tcga_pred_function)
                print("Train Mise: %s" % str(mise))
                print("Train DPE: %s" % str(dpe))
                print("Train PE: %s" % str(pe))
                print("Train ATE: %s" % str(ate))
            
            
            mise, dpe, pe, ate = compute_eval_metrics_baseline(test_dataset.metainfo, test_dataset, args.num_treatments, tcga_pred_function)
            print("Mise: %s" % str(mise))
            print("DPE: %s" % str(dpe))
            print("PE: %s" % str(pe))
            print("ATE: %s" % str(ate))

    elif args.cont_treatment:
        compute_cont_outcome_error0(method_name, est, X, T, Y)
        
        compute_cont_outcome_error0(method_name, est, test_X, test_T, test_Y)
        
        compute_cont_outcome_error(dataset, est, X, T, method_name)
    
        compute_cont_outcome_error(test_dataset, est, X, T, method_name)
        

def mean_impute(X, M, mean_X=None):
    X[M==0] = 0
    if mean_X is None:
        mean_X = torch.sum(X*M, dim=0) / (torch.sum(M, dim=0) + 1e-5)
    X = X*M + (1-M)*mean_X.unsqueeze(0)
    return X, mean_X



def calculate_real_bound(feat_range_mappings,feat_name_ls, single_test_X, idx):
    real_bound_val = (feat_range_mappings[feat_name_ls[idx].split("_")[0]][1] - feat_range_mappings[feat_name_ls[idx].split("_")[0]][0])*single_test_X[0,idx] + feat_range_mappings[feat_name_ls[idx].split("_")[0]][0]
    print(single_test_X[0,idx])
    print(real_bound_val)
    print(feat_name_ls[idx])

def transform_tensor_to_df(train_X, train_treatment, train_outcome, feat_name_ls):
    
    all_feat_name_ls = []
    
    all_feat_name_ls += feat_name_ls
    
    all_feat_name_ls += ["treatment", "outcome"]
    
    df = pd.DataFrame(np.concatenate([train_X.numpy(), train_treatment.view(-1,1).numpy(), train_outcome.view(-1,1).numpy()], axis=-1), columns=all_feat_name_ls)
    
    df.reset_index(inplace=True)
    
    df = df.rename(columns={"index":"id"})
    
    return df

# def get_shap_value():
#     mono_model = MonoTransTEE(self.model, treatment)
#     e = shap.DeepExplainer(mono_model, shap.utils.sample(feature_tensor, 100)) 
#     all_shap_values = []
    
#     if subset_ids is None:
#         sample_ids = list(range(feature_tensor.shape[0]))
#     else:
#         sample_ids = subset_ids
#     for sample_id in tqdm(sample_ids):
#     #     shap_values = torch.from_numpy(e.shap_values(feature_tensor[sample_id].view(1,-1)))
#     # for sample_id in tqdm(range(feature_tensor.shape[0])):
#         shap_val = e.shap_values(feature_tensor[sample_id].view(1,-1))
#         if type(shap_val) is list:
        
#             shap_values = torch.from_numpy(shap_val[0])
#         else:
#             shap_values = torch.from_numpy(shap_val)
#         topk_s_val, topk_ids= torch.topk(shap_values.view(-1), k=tree_depth)
        
#         selected_col_ids = {int(topk_ids[idx]):topk_s_val[idx] for idx in range(len(topk_ids))}
        
        
#         all_shap_values.append(selected_col_ids)
#     # all_shap_value_tensor = np.concatenate(all_shap_values, axis=0)
#     decistion_rules_by_treatment[treatment] = all_shap_values
#     explainer_by_treatment[treatment] = e
    