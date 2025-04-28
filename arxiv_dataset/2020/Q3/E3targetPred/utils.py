## utils.py
import time
import numpy as np
import pandas as pd
from random import sample
from scipy.io import savemat

from feature_encodings import *
from models import *
from perf_metrics import *

def split_dataset(plist, nlist, target_label, num_train=570, num_valid=30):
    """Return Train, Valid, Test split index vectors"""
    ## TRAIN
    p_train = sample(plist, num_train)
    n_train = sample(nlist, num_train)
    train_list = p_train + n_train

    ## VALID
    p_val_list = set(plist) - set(p_train)
    n_val_list = set(nlist) - set(n_train)
    p_val = sample(p_val_list, 30)
    n_val = sample(n_val_list, 30)

    ## TEST
    val_list = list(p_val) + list(n_val)
    dev_list = train_list + val_list
    test_list = list(set(list(np.where(target_label)[0])) - (set(dev_list)))

    return train_list, val_list, test_list

def load_dataset(E3_features, Sub_features, Pair_features, target_label, sample_list):
    """Load datasets"""
    Xin = np.concatenate((E3_features,Sub_features), axis=1)[sample_list]
    Xout = np.concatenate((E3_features,Sub_features), axis=1)[sample_list] 
    y = target_label[sample_list]

    return Xin, Xout, y

def MCSim(LVs, Gaps, lmd, num_Trials, E3_features, Sub_features, Pair_features, target_label):
    ##### Perform Monte-Carlos Simulations for [num_Trials]# of independent Trials####
    
    ## Define parameters explicitly
    start = time.time()
    LVs = LVs
    Gaps = Gaps
    lmd = lmd
    num_Trials = num_Trials
    
    ## Start loops    
    for gap in Gaps:
        for LV in LVs:
            Stats =[]            
            ## Divide negative and positive samples
            plist = list(np.asarray(np.where(target_label[:,1]==1)).flatten())
            nlist = list(np.asarray(np.where(target_label[:,1]==0)).flatten())
            
            for loop_ind in range(0,num_Trials):
                ## Split Datasets
                train_list, val_list, test_list = split_dataset(plist, nlist, target_label, num_train=570, num_valid=30)
                
                ## Load Datasets
                Xin_train, Xout_train, y_train = load_dataset(E3_features, Sub_features, Pair_features, target_label, train_list)
                Xin_val, Xout_val, y_val = load_dataset(E3_features, Sub_features, Pair_features, target_label, val_list)
                Xin_test, Xout_test, y_test = load_dataset(E3_features, Sub_features, Pair_features, target_label, test_list)
                
                # Define Model
                model = E3_LSE(input_size=Xin_train.shape[1],output_size=Xout_train.shape[1],LV=LV, lmd=lmd)
                es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=300)
                
                checkpoint = ModelCheckpoint('./mc_model-best.h5',
                                             verbose=0, monitor='val_loss',save_best_only=True, mode='auto')
                
                history = model.fit({'enc_input': Xin_train},
                                    {'class_output': y_train, 'decoder_output': Xout_train},
                                    validation_data = ({'enc_input': Xin_val},
                                                       {'class_output': y_val, 'decoder_output': Xout_val}),
                                    epochs=2000, batch_size=1000, callbacks=[checkpoint, es], verbose=0)
                
                del model  # deletes the existing model
                model = load_model('mc_model-best.h5')
                
                ## Calculate Outputs
                y_train_pred, X_train_pred = model.predict(Xin_train,batch_size=1800, verbose=0)
                y_train_score = y_train_pred[:,1]
                y_train_pred = to_categorical(y_train_pred.argmax(axis=1))
                MSE_X_train_pred = (np.square(X_train_pred - Xout_train)).mean(axis=1)
                
                y_val_pred, X_val_pred = model.predict(Xin_val,batch_size=200, verbose=0)
                y_val_score = y_val_pred[:,1]
                y_val_pred = to_categorical(y_val_pred.argmax(axis=1))
                MSE_X_val_pred = (np.square(X_val_pred - Xout_val)).mean(axis=1)
                
                y_test_pred, X_test_pred = model.predict(Xin_test,batch_size=200, verbose=0)
                y_test_score = y_test_pred[:,1]
                y_test_pred = to_categorical(y_test_pred.argmax(axis=1))
                MSE_X_test_pred = (np.square(X_test_pred - Xout_test)).mean(axis=1)
                
                ## Performance Measures
                tr_acc, tr_sen, tr_spe, tr_f1, tr_mcc, tr_bacc, tr_yi, tr_auc, tr_aupr = Calculate_Stats(y_train,y_train_pred, y_train_score);
                v_acc, v_sen, v_spe, v_f1, v_mcc, v_bacc, v_yi, v_auc, v_aupr = Calculate_Stats(y_val,y_val_pred, y_val_score);
                t_acc, t_sen, t_spe, t_f1, t_mcc, t_bacc, t_yi, t_auc, t_aupr = Calculate_Stats(y_test,y_test_pred, y_test_score);

                ## Save Measures for later analysis
                Stats.append([tr_acc, tr_sen, tr_spe, tr_f1, tr_mcc, tr_bacc, tr_yi, tr_auc, tr_aupr, -10*np.log10(MSE_X_train_pred.mean()),
                              v_acc, v_sen, v_spe, v_f1, v_mcc, v_bacc, v_yi, v_auc, v_aupr, -10*np.log10(MSE_X_val_pred.mean()),
                              t_acc, t_sen, t_spe, t_f1, t_mcc, t_bacc, t_yi, t_auc, t_aupr,-10*np.log10(MSE_X_test_pred.mean())])

                
                ## Print performance messages
                print('CKSAAP-Gap:',gap, 'LV=',LV,'Trial:',loop_ind, 'Valid / Test Youden-index:', v_yi,'/',t_yi, 'MCC:', t_mcc, 'AUC:', t_auc, 'AUPR:', t_aupr, 'MSE (dB):', -10*np.log10(MSE_X_test_pred.mean()))
                ## End of single trial
            
            ## save all trials
            Statistics = np.asarray(Stats)
            filename = '../mcsim_result/E3_LSE_STATS_CKSAAP_GAP_' + str(gap) + 'LV' + str(LV) + 'cls' + str(0.99) +'.mat'
            savemat(filename,{'Statistics':Statistics})
                
        ## Show Classification/Reconstruction Statistics for given LV and gap
        Show_Statistics('Training Results (MEAN)',Statistics.mean(axis=0)[0:10])
        Show_Statistics('Validation Results (MEAN)',Statistics.mean(axis=0)[10:20])
        Show_Statistics('Test Results (MEAN)',Statistics.mean(axis=0)[20:30])
            
    end = time.time()
    print(end-start)


def Train_LSE(LV, Gap, lmd, E3_features, Sub_features, Pair_features, target_label):
    ##### Perform Monte-Carlos Simulations for [num_Trials]# of independent Trials####
    
    ## Define parameters explicitly
    LV = LV
    Gap = Gap
    lmd = lmd
   
    Stats =[]            
    ## Divide negative and positive samples
    plist = list(np.asarray(np.where(target_label[:,1]==1)).flatten())
    nlist = list(np.asarray(np.where(target_label[:,1]==0)).flatten())
    
    ## Split Datasets
    train_list, val_list, test_list = split_dataset(plist, nlist, target_label, num_train=570, num_valid=30)
    
    ## Load Datasets
    Xin_train, Xout_train, y_train = load_dataset(E3_features, Sub_features, Pair_features, target_label, train_list)
    Xin_val, Xout_val, y_val = load_dataset(E3_features, Sub_features, Pair_features, target_label, val_list)
    Xin_test, Xout_test, y_test = load_dataset(E3_features, Sub_features, Pair_features, target_label, test_list)
    
    # Define Model
    model = E3_LSE(input_size=Xin_train.shape[1],output_size=Xout_train.shape[1],LV=LV, lmd=lmd)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=300)
    
    checkpoint = ModelCheckpoint('model-best.h5',
                                 verbose=0, monitor='val_loss',save_best_only=True, mode='auto')
    
    history = model.fit({'enc_input': Xin_train},
                        {'class_output': y_train, 'decoder_output': Xout_train},
                        validation_data = ({'enc_input': Xin_val},
                                           {'class_output': y_val, 'decoder_output': Xout_val}),
                        epochs=2000, batch_size=1000, callbacks=[checkpoint, es], verbose=0)
    
    del model  # deletes the existing model
    model = load_model('model-best.h5')
    
    ## Calculate Outputs
    y_train_pred, X_train_pred = model.predict(Xin_train,batch_size=1800, verbose=0)
    y_train_score = y_train_pred[:,1]
    y_train_pred = to_categorical(y_train_pred.argmax(axis=1))
    MSE_X_train_pred = (np.square(X_train_pred - Xout_train)).mean(axis=1)
    
    y_val_pred, X_val_pred = model.predict(Xin_val,batch_size=200, verbose=0)
    y_val_score = y_val_pred[:,1]
    y_val_pred = to_categorical(y_val_pred.argmax(axis=1))
    MSE_X_val_pred = (np.square(X_val_pred - Xout_val)).mean(axis=1)
    
    y_test_pred, X_test_pred = model.predict(Xin_test,batch_size=200, verbose=0)
    y_test_score = y_test_pred[:,1]
    y_test_pred = to_categorical(y_test_pred.argmax(axis=1))
    MSE_X_test_pred = (np.square(X_test_pred - Xout_test)).mean(axis=1)
    
    ## Performance Measures
    tr_acc, tr_sen, tr_spe, tr_f1, tr_mcc, tr_bacc, tr_yi, tr_auc, tr_aupr = Calculate_Stats(y_train,y_train_pred, y_train_score);
    v_acc, v_sen, v_spe, v_f1, v_mcc, v_bacc, v_yi, v_auc, v_aupr = Calculate_Stats(y_val,y_val_pred, y_val_score);
    t_acc, t_sen, t_spe, t_f1, t_mcc, t_bacc, t_yi, t_auc, t_aupr = Calculate_Stats(y_test,y_test_pred, y_test_score);

    ## Save Measures for later analysis
    Stats.append([tr_acc, tr_sen, tr_spe, tr_f1, tr_mcc, tr_bacc, tr_yi, tr_auc, tr_aupr, -10*np.log10(MSE_X_train_pred.mean()),
                  v_acc, v_sen, v_spe, v_f1, v_mcc, v_bacc, v_yi, v_auc, v_aupr, -10*np.log10(MSE_X_val_pred.mean()),
                  t_acc, t_sen, t_spe, t_f1, t_mcc, t_bacc, t_yi, t_auc, t_aupr,-10*np.log10(MSE_X_test_pred.mean())])
    
    ## Print performance messages
    print('CKSAAP-Gap:',gap, 'LV=',LV,'Trial:',loop_ind, 'Valid / Test Youden-index:', v_yi,'/',t_yi, 'MCC:', t_mcc, 'AUC:', t_auc, 'AUPR:', t_aupr, 'MSE (dB):', -10*np.log10(MSE_X_test_pred.mean()))
    ## End of single trial
    
    ## save all trials
    Statistics = np.asarray(Stats)
    # filename = 'E3_LSE_STATS_CKSAAP_GAP_' + str(gap) + 'LV' + str(LV) + 'cls' + str(0.99) +'.mat'
    # savemat(filename,{'Statistics':Statistics})
            
    ## Show Classification/Reconstruction Statistics for given LV and gap
    # Show_Statistics('Training Results (MEAN)',Statistics.mean(axis=0)[0:9])
    # Show_Statistics('Validation Results (MEAN)',Statistics.mean(axis=0)[9:18])
    # Show_Statistics('Test Results (MEAN)',Statistics.mean(axis=0)[18:27])
            
    return Xin_train, y_train, Xin_test, y_test, model
