from config import conf, args
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

import shutil

"""Script for real data experiments when the expert predicts using a top-k predictor"""
# import pdb; pdb.set_trace()
original_stdout = sys.stdout
original_stderr = sys.stderr

# get k from parsed arguments
k = args.topk

results_root = f"{conf.ROOT_DIR}/results_real/top{k}"
if not os.path.exists(results_root):
    os.mkdir(results_root)

results_root+=f"/calibrationSet{conf.cal_split}_{conf.human_subset_select}_m={conf.y_humans_cnt}"
if not os.path.exists(results_root):
    os.mkdir(results_root)

now = lambda:datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# Save config file
time, date = datetime.datetime.now().time().strftime("%H:%M:%S"), datetime.datetime.now().date()
if not os.path.isdir(f"{results_root}/_0/"): os.mkdir(f"{results_root}/_0/")
shutil.copy(f"{conf.ROOT_DIR}/config.py", f"{results_root}/_0/config_{date}_{time}.py")

# For a given number of calibration and estimation split 
# Run experiments for all models
for model_name in conf.model_names:             #conf.model_names: ['densenet-bc-L190-k40', 'preresnet-110', 'resnet-110']  changed   
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
                    X_test, X_cal, X_est, y_test, y_cal, y_est = utils.make_dataset_real(run, model_name)

                    if conf.y_humans_init in ['single', 'multiple']: print(f"Initializing human ", end='')
                    else: print(f"Initializing human ")
                    conf.accuracy = None
                    
                    human = ExpertReal(X_test, model_name=model_name)

                    # Simulate human predictions   # changed   
                    if conf.y_humans_init == 'single':
                        if conf.lapl_smoothing: # laplace smoothing is a fancy way to make the human performance more inaccurate
                            y_humans_init = human.simulate_humans(X_test, lapl_smoothing=conf.lapl_smoothing, lapl_param=conf.lapl_param)
                        else:
                            y_humans_init = human.simulate_humans(X_test)
                        print(f"human_acc {(y_humans_init==y_test).sum()/len(y_test)}")
                    elif conf.y_humans_init == 'multiple':                              # by sampling from the expert label distribution (calculated from the expert predictions)
                        y_humans_init, y_humans_est = [], []
                        for idx in range(conf.y_humans_cnt):
                            if conf.lapl_smoothing: # laplace smoothing is a fancy way to make the human performance more inaccurate
                                y_humans_init.append(human.simulate_humans(X_test, lapl_smoothing=conf.lapl_smoothing, lapl_param=conf.lapl_param))
                                y_humans_est.append(human.simulate_humans(X_est, lapl_smoothing=conf.lapl_smoothing, lapl_param=conf.lapl_param))
                            else:
                                y_humans_init.append(human.simulate_humans(X_test))         # generates different human simulations every time
                                y_humans_est.append(human.simulate_humans(X_est))
                        print(f"human_acc {(y_humans_init[0]==y_test).sum()/len(y_test)}")
                    else:
                        y_humans_init = None

                    print(f"Initializing the machine learning model ")
                    model = ModelReal(model_name, conf.model_lapl, conf.model_lapl_param)
                    print(f"model_acc {model.test(X_test, y_test)}")

                    print(f"{now()}: Evaluating top{k}...")
                    conf_pred = ConformalPrediction(X_cal, y_cal, X_est, y_est, model, conf.delta)
        
                    # Performing INFERENCE
                    if conf.y_humans_init == 'multiple':
                        p_error_t = conf_pred.error_given_test_set_topk_multiple(X_test, y_test, human.w_matrix, y_humans_init, conf.human_subset_select, k=k)
                    else:
                        p_error_t = conf_pred.error_given_test_set_topk(X_test, y_test, human.w_matrix, y_humans_init, k=k)
                    p_error = p_error_t.detach().cpu().numpy()
                    with open(f"{res_dir}/top{k}_test_error", 'wb') as f1:
                        pickle.dump(p_error, f1, pickle.HIGHEST_PROTOCOL)

                    sys.stdout = original_stdout
                sys.stderr = original_stderr    
            except:
                print(sys.exc_info()[0], file=f_e)
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                raise
