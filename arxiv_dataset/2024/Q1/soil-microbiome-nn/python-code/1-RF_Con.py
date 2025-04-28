#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
import os
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from pandas import ExcelWriter
from sklearn.ensemble import RandomForestRegressor
import openpyxl
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error 


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


from datetime import datetime
start_time = datetime.now()


# In[4]:


def process_data(data_train,data_val,cv):  
    x_column_list = data_train.drop(columns=['y_b']).columns  
    percent_label=[round(100*len(np.where(data_train['y_b']==0)[0])/len(data_train)),round(100*len(np.where(data_val['y_b']==0)[0])/len(data_val))]  
    #Classification RF  
    pipeRF = Pipeline([('classifier', [RandomForestRegressor()])])  
    param_grid = [  
    {'classifier' : [RandomForestRegressor()], 
    'classifier__n_estimators': [100, 200],
    'classifier__min_samples_split': [8, 10],
    'classifier__min_samples_leaf': [3, 4, 5],
     'classifier__max_depth': [80, 90]}]  
    clf = GridSearchCV(pipeRF, param_grid = param_grid, cv = cv, n_jobs=-1, scoring='neg_mean_squared_error')  
    # Fit on data  
    clf.fit(data_train[x_column_list],data_train['y_b'])  
    best_clf=clf.best_estimator_  
    y_valid=best_clf.predict(data_val[x_column_list])  
    #report_All = classification_report(data_val['y_b'],y_valid,output_dict=True) 
    
    
    report_All = [mean_squared_error(data_val['y_b'],y_valid, squared=False),
                                    mean_absolute_error(data_val['y_b'],y_valid),
                                    r2_score(data_val['y_b'],y_valid),
                                    mean_absolute_percentage_error(data_val['y_b'],y_valid)]
    # Create DataFrame with row and column names
    dAll = pd.DataFrame(report_All, index=['mean_squared_error', 'mean_absolute_error', 'r2', 'mean_absolute_percentage_error'], 
                        columns=['value'])
                
    dAll=pd.DataFrame(report_All) 

    return dAll


# In[5]:


cv=RepeatedKFold(n_splits=10,n_repeats=3, random_state=100)


# In[6]:



path_x = 'OTU/normalized_otu/'
path_response='/response/con_response/'


# In[7]:


for file_response in os.listdir(path_response):  
    if (file_response != '.DS_Store') & (file_response != 'Icon\r'):  
        print(file_response)  
        path_r= path_response+file_response  
        os.chdir(path_r)  
        for re in os.listdir(path_r):  
            if re[0:8] == 'response':  
                response = pd.read_csv(path_response+file_response+'/'+re)  
                response.rename(columns={'Column1':'Link_ID',response.columns[1]:'y_b'}, inplace=True)                 


# In[8]:


for file_response in os.listdir(path_response):  
    if (file_response != '.DS_Store') & (file_response != 'Icon\r'):  
        print(file_response)  
        path_r= path_response+file_response  
        os.chdir(path_r)  
        for re in os.listdir(path_r):  
            if re[0:8] == 'response':  
                response = pd.read_csv(path_response+file_response+'/'+re)  
                response.rename(columns={'Unnamed: 0':'Link_ID',response.columns[1]:'y_b'}, inplace=True)   
                writer= pd.ExcelWriter(path_r+'/'+'classification_RF'+'.xlsx', engine='xlsxwriter')   
                for file_folder in os.listdir(path_x):  
                    if (file_folder[-4:] != '.csv') & (file_folder != '.DS_Store')& (file_folder != 'Icon\r'):          
                        path = path_x+file_folder  
                        os.chdir(path)  
                        file_list = []  
                        tRF=pd.DataFrame()  
                        tcluster=pd.DataFrame()  
                        k=0  
                        for file in os.listdir(path):  
                            if (file[0] != 't') & (file[-4:] == '.csv') & (file != '.DS_Store')& (file!= 'Icon\r'):  
                                print(file)  
                                file_list.append(file)  
                                data_temp = pd.read_csv(file)  
                                data_temp.rename(columns={'Unnamed: 0':'Link_ID'}, inplace=True)  
                                data=pd.merge(response,data_temp,on='Link_ID')  
                                data.drop(columns = 'Link_ID',inplace=True)  
                                data_train,data_val = train_test_split(data,train_size=0.8, random_state=42)  
                                output = process_data(data_train,data_val,cv) 
                                #mean_absolute_percentage_error
                                tRF[file]=output       
                                k=k+1  
                        tRF.index = ['mean_squared_error', 'mean_absolute_error', 'r2', 'mean_absolute_percentage_error']
                        tRF.to_excel(writer, sheet_name=file_folder, index=True)  
                writer.save()                      


# In[9]:


end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# In[ ]:




