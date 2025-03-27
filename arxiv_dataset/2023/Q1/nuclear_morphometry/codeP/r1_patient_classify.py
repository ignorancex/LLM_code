
import numpy as np
import math as m
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import h5py
from scipy import interp
from sklearn.metrics import accuracy_score, auc, roc_curve, confusion_matrix
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, GridSearchCV
from os.path import join
import json
#from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from support.optrans.decomposition import PLDA
from sklearn.neighbors import KNeighborsClassifier

from scipy import stats as stts

#%%
if __name__ == '__main__':
    
    #%% INPUTS   
    data_tags=('1a','1b','1c','1d') # ('1a','1b','1c','1d'); ('2d',); ('3a',); ('4a',)
    
    Exp='MAIN' # 'MAIN_CMP_NFE' # 'MAIN'; # 'EXP06_ex1'; # 'EXP06_ex1v2'; 'MAIN_CMP_NFE' _______ - usual - 'MAIN'
    reg_str='BOTH' # 'TOF'; 'BOTH' _______ - usual - 'BOTH'; 'BOTH' (or maybe 'TOF') is for MAIN 'TOF' is for MAIN_CMP_NFE
    
    single_or_composite=0 # 1 = single, 0 composite  _______ - usual - 0 composite
    
    # %
    
    clsfr_vec=('LDA','PLDA','RF','LR','SVM','SVM','kNN') # ('LDA','kNN','PLDA','RF','LR','SVM','SVM')
    kerneltype_vec=('none','none','none','none','linear','rbf','none') # ('none','none','none','none','none','linear','rbf')
    
    Nfold=2; 
    feat_type_tags=('1','2','3')
    
    if single_or_composite==1:
        Ndir=5;
    else:
        Ndir=10;
    
    
    #%% 
    train_test_dir='../DATA/METADATA/'+Exp+'/'
    if single_or_composite==1:
        res_dir = '../RESULTS/'+Exp+'/Dataset_'+data_tags[0][0]+'/test01A_PLDA/'
    else:    
        res_dir = '../RESULTS/'+Exp+'/Dataset_'+data_tags[0][0]+'/testA/'
    
    for cc in range(len(clsfr_vec)):
        clsfr=clsfr_vec[cc]; kerneltype=kerneltype_vec[cc]
        
        for jj in range(len(feat_type_tags)):
            type_tag=feat_type_tags[jj]
            accuracy_train=np.zeros((Nfold,len(data_tags)))
            accuracy_test=np.zeros((Nfold,len(data_tags)))
            u_tr=np.zeros(len(data_tags)); u_te=np.zeros(len(data_tags)); s_tr=np.zeros(len(data_tags)); s_te=np.zeros(len(data_tags))
            
            for ii in range(len(data_tags)):
                tr_n_te_tag=data_tags[ii]
                
                for kk in range(Nfold):
#                    tr_n_te_tag=data_tags[ii]; type_tag=feat_type_tags[jj]

                    if single_or_composite==1:
                        dstart_train='PLDAFeat'+type_tag+'_train_data'+tr_n_te_tag+'_reg'+reg_str+'___fold'+str(kk+1)+'of'+str(Nfold)+'_retdir'+str(Ndir)
                        dstart_test='PLDAFeat'+type_tag+'_test_data'+tr_n_te_tag+'_reg'+reg_str+'___fold'+str(kk+1)+'of'+str(Nfold)+'_retdir'+str(Ndir)
                    else:
                        dstart_train='Feat'+type_tag+'_train_data'+tr_n_te_tag+'_reg'+reg_str+'___fold'+str(kk+1)+'of'+str(Nfold)+'_retdir'+str(Ndir)
                        dstart_test='Feat'+type_tag+'_test_data'+tr_n_te_tag+'_reg'+reg_str+'___fold'+str(kk+1)+'of'+str(Nfold)+'_retdir'+str(Ndir)
                    
                    
                     #%%
                    # Set classifier
                    if clsfr == 'LR':
                        clf = LogisticRegressionCV(Cs=np.logspace(-4,4,9).tolist(), cv=None, dual=False, penalty='l2', solver='lbfgs', class_weight='balanced',
                                                   verbose=0, multi_class='ovr')
                        param_grid = {'fit_intercept': [True, False]}
                    elif clsfr == 'SVM':
                        if kerneltype=='linear':
                            if type_tag=='2':
                                clf = LinearSVC()
                            else:
                                clf = SVC(kernel=kerneltype, class_weight='balanced', probability=True, decision_function_shape='ovr')
                                param_grid = {}
                        else:
                            clf = SVC(kernel=kerneltype, class_weight='balanced', probability=True, decision_function_shape='ovr')
                            param_grid = {'C': np.logspace(-2, 6, 9), 'gamma': np.logspace(-4, 3, 8)}
                    elif clsfr == 'LDA':
                        clf=PLDA(alpha=.001) #LinearDiscriminantAnalysis()
                        param_grid={}
                    elif clsfr == 'kNN':
                        clf=KNeighborsClassifier()
                        param_grid={}
                    elif clsfr == 'PLDA':
                        clf=PLDA()
                        param_grid={}
                    elif clsfr == 'RF':
                        clf=RandomForestClassifier(n_estimators=100)
                        param_grid={}
                    
                    #%%
                    load_file=train_test_dir+dstart_train+'.mat'  
                    temp=loadmat(load_file)
                    if type_tag=='2':
                        Xtr=temp['feat']; ytr_pat=temp['y_pat']; ytr_cell=temp['y_cell'];
                        ytr_pat=np.squeeze(ytr_pat); ytr_cell=np.squeeze(ytr_cell)
                    else:
                        Xtr=temp['feat']; ytr=temp['y_pat'];
                        ytr=np.squeeze(ytr);
                        
                        
                    load_file=train_test_dir+dstart_test+'.mat'  
                    temp=loadmat(load_file)
                    if type_tag=='2':
                        Xte=temp['feat']; yte_pat=temp['y_pat']; yte_cell=temp['y_cell'];
                        yte_pat=np.squeeze(yte_pat); yte_cell=np.squeeze(yte_cell)
                    else:
                        Xte=temp['feat']; yte=temp['y_pat'];
                        yte=np.squeeze(yte);
                        
                    
                    print("Train-test dir: {}".format(train_test_dir))
                    print("Results dir: {}".format(res_dir))
                    print(load_file)
                    print(Xtr.shape); print(Xte.shape); 
    #                print(yte.shape); print(ytr.shape); 
                   
    
                    #%%
                    if type_tag=='2':
                        pat_tg_tr=np.unique(ytr_pat)
                        pat_tg_te=np.unique(yte_pat)
                        
                        clf.fit(Xtr, ytr_cell)
                        
                        y_pred_tr = clf.predict(Xtr)
                        y_pred = clf.predict(Xte)
                        
                        
                        # # #
                        
                        ytr=np.zeros(len(pat_tg_tr)); yptr=np.zeros(len(pat_tg_tr))
                        for mm in range(len(pat_tg_tr)):
                            pat=pat_tg_tr[mm]
                            tmp1=np.unique(ytr_cell[np.where(ytr_pat==pat)])
                            tmp3=stts.mode(y_pred_tr[np.where(ytr_pat==pat)])
                            tmp2=tmp3[0]
                            
                            ytr[mm]=tmp1; yptr[mm]=tmp2
                        acc_tr = accuracy_score(ytr, yptr); acc_tr=acc_tr*100
                        
                        yte=np.zeros(len(pat_tg_te)); ypte=np.zeros(len(pat_tg_te))
                        for mm in range(len(pat_tg_te)):
                            pat=pat_tg_te[mm]
                            tmp1=np.unique(yte_cell[np.where(yte_pat==pat)])
                            tmp3=stts.mode(y_pred[np.where(yte_pat==pat)])
                            tmp2=tmp3[0]
                            
                            yte[mm]=tmp1; ypte[mm]=tmp2
                        acc = accuracy_score(yte, ypte); acc=acc*100
                        
                        # ytr=ytr_cell; yte=yte_cell;
                        # acc_tr = accuracy_score(ytr, y_pred_tr); acc_tr=acc_tr*100
                        # acc = accuracy_score(yte, y_pred); acc=acc*100
                        
                        # # #
                            
                    else:
                        # Run classifier
                        clf.fit(Xtr, ytr)
                        
                        y_pred_tr = clf.predict(Xtr)
                        y_pred = clf.predict(Xte)
                        acc_tr = accuracy_score(ytr, y_pred_tr); acc_tr=acc_tr*100
                        acc = accuracy_score(yte, y_pred); acc=acc*100
                        
                        
                    print(jj); print(ii); print(kk)
                    print("Training accuracy: {}".format(acc_tr))
                    print("Test accuracy: {}".format(acc))
                    
                    accuracy_train[kk,ii]=acc_tr
                    accuracy_test[kk,ii]=acc
                        
                        
                u_tr[ii]=np.mean(accuracy_train[:,ii]); s_tr[ii]=np.std(accuracy_train[:,ii]); u_te[ii]=np.mean(accuracy_test[:,ii]); s_te[ii]=np.std(accuracy_test[:,ii])
                        
            res_file = res_dir+'pat_acc_feat'+type_tag+'_'+clsfr+kerneltype+'.mat'
            Path(res_dir).mkdir(parents=True, exist_ok=True); 
            savemat(res_file,{'acc_tr':accuracy_train,'acc_te':accuracy_test,'u_tr':u_tr,'u_te':u_te,'s_tr':s_tr,'s_te':s_te})

