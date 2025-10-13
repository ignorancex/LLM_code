from config import conf
from conformal_prediction import ConformalPrediction
import os
from expert.expert import ExpertReal
import pickle
import utils
from model.model import ModelReal
import sys
import datetime
from tqdm import tqdm
import torch
import numpy as np
from pathlib import Path

import shutil
from scipy import stats



"""Script for real data experiments"""
original_stdout = sys.stdout
original_stderr = sys.stderr

results_root = f"{conf.ROOT_DIR}/results_real"
if not os.path.exists(results_root):
    os.mkdir(results_root)

results_root += f"/calibrationSet{conf.cal_split}_{conf.human_subset_select}_m={conf.y_humans_cnt}_alpha={conf.alpha_index}"

if conf.CM_lapl_smoothing:
    results_root += f"_CMlapl={conf.CM_lapl_param}"

if not os.path.exists(results_root):
    os.mkdir(results_root)

now = lambda:datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Save config file
time, date = datetime.datetime.now().time().strftime("%H:%M:%S"), datetime.datetime.now().date()
if not os.path.isdir(f"{results_root}/_0/"): 
    os.mkdir(f"{results_root}/_0/")
shutil.copy(f"{conf.ROOT_DIR}/config.py", f"{results_root}/_0/config_{date}_{time}.py")

import pdb; pdb.set_trace()
# For a given number of calibration and estimation split 
# Run experiments for all models
for model_name in conf.model_names:
    for run in tqdm(range(conf.n_runs_per_split)):
        res_dir = f"{results_root}/{model_name}_run{run}"
        if not os.path.exists(res_dir):
            os.mkdir(res_dir)
        with open(f"{res_dir}/logs_err.txt", 'w', buffering=1) as f_e:
            sys.stderr = f_e
            try:
                with open(f"{res_dir}/logs.txt", 'w', buffering=1) as f:
                    sys.stdout = f
                    
                    print(f"Creating {conf.data_size} data: "+\
                            f"{conf.cal_split*100}% calibration, "+\
                            f"{(1 - conf.cal_split)*100}% test")
                    X_test, X_cal, X_est, y_test, y_cal, y_est = utils.make_dataset_real(run, model_name)       # Xs are IDs, ys are the labels
                    
                    if conf.y_humans_init in ['single', 'multiple']: 
                        print(f"Initializing human ", end='')
                    conf.accuracy = None
                    
                    human = ExpertReal(X_test, model_name=model_name, y_humans_cnt = conf.y_humans_cnt, lapl_s = conf.CM_lapl_smoothing, lapl_p = conf.CM_lapl_param)  # Note: this ignores test for CM calculation
                    
                    # Simulate human predictions   # changed  
                    if conf.y_humans_init == 'single':
                        if conf.lapl_smoothing:                  # laplace smoothing is a fancy way to make the human performance more inaccurate
                            y_humans_init = human.simulate_humans(X_test, lapl_smoothing=conf.lapl_smoothing, lapl_param=conf.lapl_param)
                        else:
                            y_humans_init = human.simulate_humans(X_test)
                        print(f"human_acc {(y_humans_init==y_test).sum()/len(y_test)}")
                    elif conf.y_humans_init == 'multiple':                              # by sampling from the expert label distribution (calculated from the expert predictions)
                        y_humans_init, y_humans_est = [], []
                        for idx in range(conf.y_humans_cnt):
                            if conf.lapl_smoothing:
                                y_humans_init.append(human.simulate_humans(X_test, lapl_smoothing=conf.lapl_smoothing, lapl_param=conf.lapl_param))
                                y_humans_est.append(human.simulate_humans(X_est, lapl_smoothing=conf.lapl_smoothing, lapl_param=conf.lapl_param))
                            else:
                                y_humans_init.append(human.simulate_humans(X_test))         # generates different human simulations every time
                                y_humans_est.append(human.simulate_humans(X_est))
                        print(f"human_acc {(y_humans_init[0]==y_test).sum()/len(y_test)}")
                        print(f"human_team_accuracy {(stats.mode(y_humans_init,0)[0].reshape(-1)==y_test).sum()/len(y_test)}")
                    else:
                        y_humans_init = None

                    print(f"Initializing the machine learning model ")
                    model = ModelReal(model_name, conf.model_lapl, conf.model_lapl_param)
                    print(f"model_acc {model.test(X_test, y_test)}")                    # use pretrained model to predict and get the accuracy on the test set

                    print(f"{datetime.datetime.now()}: Starting conformal prediction...")
                    conf_pred = ConformalPrediction(X_cal, y_cal, X_est, y_est, model, conf.delta)
                    alphas = conf_pred.find_all_alpha_values()     # [0.999,0.998,...,0] 1 -> 0 # 1000  # 1 - (np.arange(1,self.calibration_size + 1) / (self.calibration_size + 1))
                    with open(f"{res_dir}/alphas1", 'wb') as f1:
                        pickle.dump(alphas, f1, pickle.HIGHEST_PROTOCOL)      
                    
                    print(f"{datetime.datetime.now()}: {alphas.shape[0]} alphas found")
                    
                    if conf.enable_find_alpha:
                        alpha_star_idx = conf_pred.find_a_star(human.w_matrix)              # integer value in [0,1000] # human.w_matrix is 10x10
                    else:
                        alpha_star_idx = conf.alpha_index               # alpha is user-specified
                    with open(f"{res_dir}/alpha1_idx", 'wb') as f1:
                        pickle.dump(alpha_star_idx, f1,  pickle.HIGHEST_PROTOCOL)
                    print(f"alpha_1* {conf_pred.alphas[alpha_star_idx]}")
                    print(f"star index {alpha_star_idx}")

                    # changed  
                    if conf.y_humans_init == 'multiple' and conf.enable_find_m and run==0:
                        expert_cnt = conf_pred.find_m(X_est, y_est, human.w_matrix, alphas, 
                                                        star_dummy=alpha_star_idx, 
                                                        full_cm=human.cm_per_sample[X_est], 
                                                        mult_h=None, 
                                                        y_humans=y_humans_est, 
                                                        unc=None)
                        y_humans_init = y_humans_init[:expert_cnt+1]
                        y_humans_est = y_humans_est[:expert_cnt+1]
                        print(f"Optimal m is {expert_cnt}")
                    print(f"{datetime.datetime.now()}: Calculating error in test set for all alphas")

                    # Performing INFERENCE
                    if conf.y_humans_init == 'multiple':
                        p_error_t = conf_pred.error_multiple_simulated_humans(X_test, y_test, human.w_matrix, alphas, 
                                                                            star_dummy=alpha_star_idx, 
                                                                                full_cm=human.cm_per_sample[X_test], 
                                                                                mult_h=human.mult_w_matrix, 
                                                                                y_humans=y_humans_init,
                                                                                unc=None,
                                                                                y_humans_est=y_humans_est,
                                                                                y_est=y_est,
                                                                                X_est=X_est,
                                                                                subset_select = conf.human_subset_select)    # changed   
                    else:
                        p_error_t = conf_pred.error_given_test_set_per_a(X_test, y_test, human.w_matrix, alphas, 
                                                                            star_dummy=alpha_star_idx, 
                                                                            full_cm=human.cm_per_sample[X_test], 
                                                                            mult_h=human.mult_w_matrix, 
                                                                            y_humans=y_humans_init, 
                                                                            unc=None)                # changed   
                    
                    p_error = p_error_t.detach().cpu().numpy()
                    with open(f"{res_dir}/alpha1_test_error", 'wb') as f1:
                        pickle.dump(p_error, f1, pickle.HIGHEST_PROTOCOL)

                    sys.stdout = original_stdout
                sys.stderr = original_stderr    
            except:
                print(sys.exc_info()[0], file=f_e)
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                raise






















