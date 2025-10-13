from config import conf, args
from conformal_prediction import ConformalPrediction
import os
from expert.expert import ExpertImageNet16H
import pickle
import utils
from model.model import ModelImageNet16H, ModelTrainer, compute_deferral_metrics
import sys
import datetime
from tqdm import tqdm
import torch.optim as optim
import numpy as np
import shutil
from scipy import stats
import sys
import torch
main_path = conf.base_folder
sys.path.append(main_path + 'human-ai-deferral/')       # we reference from this work # the other datasets are also here
from datasetsdefer.imagenet_16h import *


"""Script for ImageNet data experiments when the expert predicts using a top-k predictor"""
# import pdb; pdb.set_trace()
original_stdout = sys.stdout
original_stderr = sys.stderr

# get k from parsed arguments
k = args.topk

results_root = f"{conf.ROOT_DIR}/results_imagenet/top{k}"
if not os.path.exists(results_root):
    os.mkdir(results_root)

results_root+=f"/calibrationSet{conf.cal_split}_{conf.human_subset_select}_m={conf.y_humans_cnt}"
if not os.path.exists(results_root):
    os.mkdir(results_root)

now = lambda:datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# # FOR TRAINING THE ML MODEL
# optimizer = optim.Adam
# scheduler = None
# lr = 1e-3                   # 1e-2 for HateSpeech
# total_epochs = 100          # 50 for HateSpeech
# dataset = ImageNet16h(True, main_path + "human-ai-deferral/data/imagenet16H/", '080', batch_size=32, get_embeddings=True, transforms=True, mod_for_cp=True, calest_portion=conf.calest_portion)  
# print(f"Initializing model ")
# model_init = ModelImageNet16H(dataset.d, 16)                # 16 classes  
# model = ModelTrainer(1, 300, model_init, device, False)     # taken from l2d code, so set learnable_threshold_rej=False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ModelImageNet16H('080', conf.model_lapl, conf.model_lapl_param)

if False:
    if conf.pretrained_model_path is not None:
        model_init.load_state_dict(torch.load(conf.pretrained_model_path))
        model = ModelTrainer(1, 300, model_init, device, False)
    else: # TODO Fix the model training, remove/comment out code related to surrogate loss or learning to defer and define the loss function
        model.fit_hyperparam(
            dataset.data_train_loader,
            dataset.data_val_loader,
            dataset.data_test_loader,
            epochs=total_epochs,            # 50
            optimizer=optimizer,
            scheduler=scheduler,
            lr=lr,
            verbose=False,
            test_interval=5,
        )

        pretrained_path = main_path + "improve-expert-predictions-conformal-prediction/results_imagenet/calibrationSet0.1/"
        # Assuming 'model' is your model and 'optimizer' is your optimizer
        time, date = datetime.datetime.now().time().strftime("%H:%M:%S"), datetime.datetime.now().date()
        torch.save(model.model.state_dict(), pretrained_path + f"model_{date}_{time}.pth")
        # rs_metrics = compute_deferral_metrics(model.test(dataset.data_test_loader))   # can print metrics
        import pdb; pdb.set_trace()





# Human Expert-AI Interaction Code
time, date = datetime.datetime.now().time().strftime("%H:%M:%S"), datetime.datetime.now().date()
if not os.path.isdir(f"{results_root}/_0/"): os.mkdir(f"{results_root}/_0/")
shutil.copy(f"{conf.ROOT_DIR}/config.py", f"{results_root}/_0/config_{date}_{time}.py")

# For a given number of calibration and estimation split
for run in tqdm(range(conf.n_runs_per_split)):
    res_dir = f"{results_root}/_run{run}"
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

                X_test, X_cal, X_est, y_test, y_cal, y_est = utils.make_dataset_imagenet(run, '080')
                # dataset = ImageNet16h(True, main_path + "human-ai-deferral/data/imagenet16H/", '080', batch_size=32, get_embeddings=True, transforms=True, mod_for_cp=True, testval_shuffle_id=run+15, calest_portion=conf.calest_portion)
                # rs_metrics = compute_deferral_metrics(model.test(dataset.data_test_loader))

                # Define human expert confusion matrix
                print(f"Initializing human ")
                conf.accuracy = None
                human = ExpertImageNet16H(X_test, '080', conf.y_humans_cnt)         # ignore test in expert

                # Simulate human predictions   # changed by HP
                if conf.y_humans_init == 'single':
                    if conf.lapl_smoothing:
                        y_humans_init = human.simulate_humans(X_test, lapl_smoothing=conf.lapl_smoothing, lapl_param=conf.lapl_param)
                    else:
                        y_humans_init = human.simulate_humans(X_test)
                    # Print simulated human accuracy
                    print(f"human_acc {(y_humans_init==y_test).sum()/len(y_test)}")
                elif conf.y_humans_init == 'multiple':
                    y_humans_init, y_humans_est = [], []
                    for idx in range(conf.y_humans_cnt):
                        y_humans_init.append(human.simulate_humans(X_test)) # generates different human simulations every time
                        y_humans_est.append(human.simulate_humans(X_est))
                    print(f"human_acc {(y_humans_init[0]==y_test).sum()/len(y_test)}")
                    print(f"human_team_accuracy {(stats.mode(y_humans_init,0)[0].reshape(-1)==y_test).sum()/len(y_test)}")
                else:
                    y_humans_init = None



                print(f"{now( )}: Evaluating top{k}...")
                conf_pred = ConformalPrediction(X_cal, y_cal, X_est, y_est, model, conf.delta)
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
