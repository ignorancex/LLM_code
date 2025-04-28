#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use(['bmh'])
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import  classification_report
from sklearn.model_selection import RepeatedKFold,train_test_split
import os
from sklearn.pipeline import Pipeline


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


from datetime import datetime
start_time = datetime.now()


# In[4]:


path_response='response/response_original/'
path_x = 'OTU/OTUData-1-1/'

path_response_aug='response/response_aug/'
path_x_aug = 'OTU/Aug-1-1/'

path_class = 'OTU/augumented_otu_count/Filter/Class.csv'
data_class = pd.read_csv(path_class,header=0)


# In[5]:


cv=RepeatedKFold(n_splits=10,n_repeats=3, random_state=100)
th=0.05
NumofR=100
num_train=800


# In[6]:


def Compare_random(data,data_train,data_val,cv,NumofR):
    x_column_list = data_train.drop(columns=['y_b']).columns  
    pipeRF = Pipeline([('classifier', [RandomForestClassifier()])])
    param_grid = [
    {'classifier' : [RandomForestClassifier()],
    'classifier__criterion':('gini','entropy'),
    'classifier__class_weight':('balanced','auto')}]
    clf = GridSearchCV(pipeRF, param_grid = param_grid, cv = cv, n_jobs=-1, scoring='f1_weighted')

    clf.fit(data_train[x_column_list],data_train['y_b'])
    best_clf=clf.best_estimator_
    y_valid=best_clf.predict(data_val[x_column_list])

    report_RF = classification_report(data_val['y_b'],y_valid,output_dict=True) 
    F1_0_class_RF=report_RF['0']['f1-score']
    F1_1_class_RF=report_RF['1']['f1-score']

    totalF1_0=[]
    totalF1_1=[]
    for i in range(1,NumofR):
        RD1= np.random.rand(data[x_column_list].shape[0],data[x_column_list].shape[1])
        RD1=pd.DataFrame(RD1)
        RD1.columns =data[x_column_list].columns
        RD11=RD1.div(RD1.sum(axis=1), axis=0)        
        clf = GridSearchCV(pipeRF, param_grid = param_grid, cv = cv, n_jobs=-1, scoring='f1_weighted')
        clf.fit(RD11.iloc[data_train.index,0:len(x_column_list)], data_train['y_b'])
        best_clf=clf.best_estimator_
        y_valid=best_clf.predict(RD11.iloc[data_val.index,0:len(x_column_list)])
        report_Random1 = classification_report(data_val['y_b'],y_valid,output_dict=True)
        totalF1_0.append(report_Random1 ['0']['f1-score'])
        totalF1_1.append(report_Random1 ['1']['f1-score'])    
    r1=pd.DataFrame()
    r1['0']= totalF1_0
    r1['1']= totalF1_1
    EV1=r1[(r1['0'] >= F1_0_class_RF) & (r1['1'] >=F1_1_class_RF)].shape[0]/ (r1.shape[0])
    #senario2
    totalF1_0_2=[]
    totalF1_1_2=[]
    for i in range(1,NumofR):
        RD2=pd.DataFrame(np.random.rand(data[x_column_list].shape[0],data[x_column_list].shape[1]))
        RD2.columns=data[x_column_list].columns
        for col in data[x_column_list].columns:
            dd=np.where(data[x_column_list][col]==0)[0]
            if len(dd)>0:
                RD2[col][dd]=0
        RD22=RD2.div(RD2.sum(axis=1), axis=0)
        clf = GridSearchCV(pipeRF, param_grid = param_grid, cv = cv, n_jobs=-1, scoring='f1_weighted')
        clf.fit(RD22.iloc[data_train.index][x_column_list], data_train['y_b'])
        best_clf=clf.best_estimator_
        y_valid=best_clf.predict(RD22.iloc[data_val.index][x_column_list])
        report_Random2 = classification_report(data_val['y_b'],y_valid,output_dict=True)
        totalF1_0_2.append(report_Random2['0']['f1-score'])
        totalF1_1_2.append(report_Random2['1']['f1-score'])
    r2=pd.DataFrame()
    r2['0']= totalF1_0_2
    r2['1']= totalF1_1_2
    EV2=r2[(r2['0'] >= F1_0_class_RF) & (r2['1'] >=F1_1_class_RF)].shape[0]/ (r2.shape[0])
    #senario3
    totalF1_0_3=[]
    totalF1_1_3=[]
    for i in range(1,NumofR):
        YR=np.random.permutation(data['y_b'].values)
        clf = GridSearchCV(pipeRF, param_grid = param_grid, cv = cv, n_jobs=-1, scoring='f1_weighted')
        clf.fit(data_train[x_column_list],  YR[0:data_train[x_column_list].shape[0]])
        best_clf=clf.best_estimator_
        y_valid=best_clf.predict(data_val[x_column_list])
        report_Random3 = classification_report(data_val['y_b'],y_valid,output_dict=True)
        totalF1_0_3.append(report_Random3['0']['f1-score'])
        totalF1_1_3.append(report_Random3['1']['f1-score'])
    r3=pd.DataFrame()
    r3['0']= totalF1_0_3
    r3['1']= totalF1_1_3
    
    EV3=r3[(r3['0'] >= F1_0_class_RF) & (r3['1'] >=F1_1_class_RF)].shape[0]/ (r3.shape[0])
    #Senario4
    totalF1_0_4=[]
    totalF1_1_4=[]
    for i in range(1,NumofR):
        DR=data[x_column_list].sample(frac=1)
        clf = GridSearchCV(pipeRF, param_grid = param_grid, cv = cv, n_jobs=-1, scoring='f1_weighted')
        clf.fit(DR[0:data_train[x_column_list].shape[0]],  data_train['y_b'])
        best_clf=clf.best_estimator_
        y_valid=best_clf.predict(DR[data_train[x_column_list].shape[0]:])
        report_Random4 = classification_report(data_val['y_b'],y_valid,output_dict=True)
        totalF1_0_4.append(report_Random4['0']['f1-score'])
        totalF1_1_4.append(report_Random4['1']['f1-score'])
    r4=pd.DataFrame()
    r4['0']= totalF1_0_4
    r4['1']= totalF1_1_4
    EV4=r4[(r4['0'] >= F1_0_class_RF) & (r4['1'] >=F1_1_class_RF)].shape[0]/ (r4.shape[0])
    ev_value=[EV1,EV2,EV3,EV4]
    return r1,r2,r3,r4,ev_value,F1_0_class_RF,F1_1_class_RF


# In[27]:


# for file_response in os.listdir(path_response):  
#     if (file_response != '.DS_Store') & (file_response != 'Icon\r') & (file_response[0] != 'O'): 
#         print(file_response)  
#         path_r= path_response+file_response  
#         for re in os.listdir(path_r):  
#             if re[0:8] == 'response':  
#                 response = pd.read_csv(path_r+'/'+re)  
#                 response.rename(columns={'Column1':'Link_ID',response.columns[1]:'y_b'}, inplace=True)   
#                 writer= pd.ExcelWriter(path_r+'/'+'EV_Value.xlsx', engine='xlsxwriter')
#                 writer1= pd.ExcelWriter(path_r+'/'+'RMethod1.xlsx', engine='xlsxwriter')
#                 writer2= pd.ExcelWriter(path_r+'/'+'RMethod2.xlsx', engine='xlsxwriter')
#                 writer3= pd.ExcelWriter(path_r+'/'+'RMethod3.xlsx', engine='xlsxwriter')
#                 writer4= pd.ExcelWriter(path_r+'/'+'RMethod4.xlsx', engine='xlsxwriter')
#                 writer5= pd.ExcelWriter(path_r+'/'+'RFresult.xlsx', engine='xlsxwriter')
#                 for file_folder in os.listdir(path_x):  
#                     if (file_folder[-4:] != '.csv') & (file_folder != '.DS_Store')& (file_folder != 'Icon\r'):          
#                         path = path_x+file_folder    
#                         file_list = []  
#                         tRF=pd.DataFrame()  
#                         tcluster=pd.DataFrame()  
#                         k=0  
#                         for file in os.listdir(path):  
#                             if (file[0] != 't') & (file[-4:] == '.csv') & (file != '.DS_Store')& (file_folder != 'Icon\r'):  
#                                 print(file)  
#                                 file_list.append(file)  
#                                 data_temp = pd.read_csv(path+'/'+file)  
#                                 data_temp.rename(columns={'Unnamed: 0':'Link_ID'}, inplace=True)  
#                                 data=pd.merge(response,data_temp,on='Link_ID') 
#                                 print(data.shape)
#                                 data.drop(columns = 'Link_ID',inplace=True)  
#                                 data_train,data_val = train_test_split(data,train_size=0.8, random_state=42)
#                                 out_random = Compare_random(data,data_train,data_val,cv,NumofR)
#                                 r1=out_random[0]
#                                 r2=out_random[1]
#                                 r3=out_random[2]
#                                 r4=out_random[3]
#                                 EV_Value= out_random[4]
#                         pd.DataFrame(EV_Value).to_excel(writer, sheet_name=file_folder, index=True)
#                         pd.DataFrame(r1).to_excel(writer1, sheet_name=file_folder, index=True)
#                         pd.DataFrame(r2).to_excel(writer2, sheet_name=file_folder, index=True)
#                         pd.DataFrame(r3).to_excel(writer3, sheet_name=file_folder, index=True)
#                         pd.DataFrame(r4).to_excel(writer4, sheet_name=file_folder, index=True)
#                         pd.DataFrame([out_random[5],out_random[6]]).to_excel(writer5, sheet_name=file_folder, index=True)
#                 writer.save()
#                 writer1.save()
#                 writer2.save()
#                 writer3.save()
#                 writer4.save()
#                 writer5.save()


# # Aug Data

# In[8]:


for file_response in os.listdir(path_response_aug):  
    if (file_response != '.DS_Store') & (file_response != 'Icon\r') & (file_response[0] != 'O'): 
        print(file_response)  
        path_r= path_response_aug+file_response    
        for re in os.listdir(path_r):  
            if re[0:8] == 'response':  
                response = pd.read_csv(path_r+'/'+re)  
                response.rename(columns={'Column1':'Link_ID',response.columns[1]:'y_b'}, inplace=True)      
                writer= pd.ExcelWriter(path_r+'/'+'EV_Value.xlsx', engine='xlsxwriter')
                writer1= pd.ExcelWriter(path_r+'/'+'RMethod1.xlsx', engine='xlsxwriter')
                writer2= pd.ExcelWriter(path_r+'/'+'RMethod2.xlsx', engine='xlsxwriter')
                writer3= pd.ExcelWriter(path_r+'/'+'RMethod3.xlsx', engine='xlsxwriter')
                writer4= pd.ExcelWriter(path_r+'/'+'RMethod4.xlsx', engine='xlsxwriter')
                writer5= pd.ExcelWriter(path_r+'/'+'RFresult.xlsx', engine='xlsxwriter')
                for file_folder in os.listdir(path_x_aug):  
                    if (file_folder[-4:] != '.csv') & (file_folder != '.DS_Store')& (file_folder != 'Icon\r'):          
                        path = path_x_aug+file_folder   
                        file_list = []  
                        tRF=pd.DataFrame()  
                        tcluster=pd.DataFrame()  
                        k=0  
                        
                        for file in os.listdir(path):  
                            if (file[0] != 't') & (file[-4:] == '.csv') & (file != '.DS_Store')& (file_folder != 'Icon\r'):  
                                print(file)  
                                file_list.append(file)  
                                data_temp = pd.read_csv(path+'/'+file)  
                                data_temp.rename(columns={'Unnamed: 0':'Link_ID'}, inplace=True) 
                                data_temp['Link_ID']=data_class['Column1']
                                data=pd.merge(response,data_temp,on='Link_ID') 
                                print(data.shape)
                                data.drop(columns = 'Link_ID',inplace=True)  
                                data_train=data.iloc[0:num_train,:]
                                data_val=data.iloc[num_train:,:]
                                out_random = Compare_random(data,data_train,data_val,cv,NumofR)
                                r1=out_random[0]
                                r2=out_random[1]
                                r3=out_random[2]
                                r4=out_random[3]
                                EV_Value= out_random[4]
                                pd.DataFrame(EV_Value).to_excel(writer, sheet_name=file_folder, index=True)
                                pd.DataFrame(r1).to_excel(writer1, sheet_name=file_folder, index=True)
                                pd.DataFrame(r2).to_excel(writer2, sheet_name=file_folder, index=True)
                                pd.DataFrame(r3).to_excel(writer3, sheet_name=file_folder, index=True)
                                pd.DataFrame(r4).to_excel(writer4, sheet_name=file_folder, index=True)
                                pd.DataFrame([out_random[5],out_random[6]]).to_excel(writer5, sheet_name=file_folder, index=True)
                writer.save()
                writer1.save()
                writer2.save()
                writer3.save()
                writer4.save()
                writer5.save()


# In[ ]:


end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

