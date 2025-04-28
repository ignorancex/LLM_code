#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from pandas import ExcelWriter
from sklearn.ensemble import RandomForestClassifier
import openpyxl


# In[3]:


import warnings
warnings.filterwarnings('ignore')


# In[4]:


def process_data(data_train,data_val,cv):  
    x_column_list = data_train.drop(columns=['y_b']).columns  
    
    #Classification RF  
    pipeRF = Pipeline([('classifier', [RandomForestClassifier()])])  
    param_grid = [  
    {'classifier' : [RandomForestClassifier()], 
    'classifier__n_estimators': [100, 200],
    'classifier__min_samples_split': [8, 10],
    'classifier__min_samples_leaf': [3, 4, 5],
     'classifier__max_depth': [80, 90],
    'classifier__criterion':('gini','entropy'),  
    'classifier__class_weight':('balanced','auto')}]  
    clf = GridSearchCV(pipeRF, param_grid = param_grid, cv = cv, n_jobs=-1, scoring='f1_weighted')  
    # Fit on data  
    clf.fit(data_train[x_column_list],data_train['y_b'])  
    best_clf=clf.best_estimator_  
    y_valid=best_clf.predict(data_val[x_column_list])  
    report_All = classification_report(data_val['y_b'],y_valid,output_dict=True)  
    dAll=pd.DataFrame(report_All).transpose()  
    return dAll


# In[5]:


cv=RepeatedKFold(n_splits=10,n_repeats=3, random_state=100)
num_train=800


# In[6]:

path_response='response/response_aug/'
path_x = 'OTU/normalized_aug/'
path_class = 'OTU/augumented_otu_count/Filter/Class.csv'
data_class = pd.read_csv(path_class,header=0)


# In[7]:


from datetime import datetime
start_time = datetime.now()


# In[8]:


for file_response in os.listdir(path_response):  
    if (file_response != '.DS_Store') & (file_response != 'Icon\r'):  
        print(file_response)  
        path_r= path_response+file_response
        for re in os.listdir(path_r):  
            if re[0:8] == 'response':  
                response = pd.read_csv(path_r+'/'+re)
                response.rename(columns={'Column1':'Link_ID',response.columns[1]:'y_b'}, inplace=True)          
                writer= pd.ExcelWriter(path_r+'/'+'classification_RF'+'.xlsx', engine='xlsxwriter')   
                for file_folder in os.listdir(path_x):  
                    if (file_folder[-4:] != '.csv') & (file_folder != '.DS_Store')& (file_folder != 'Icon\r'):          
                        path = path_x+file_folder
                        file_list = []  
                        tRF=pd.DataFrame()  
                        tcluster=pd.DataFrame()  
                        k=0  
                        for file in os.listdir(path):  
                            if (file[0] != 't') & (file[-4:] == '.csv') & (file != '.DS_Store')& (file_folder != 'Icon\r'):  
                                print(file)  
                                file_list.append(file)  
                                data_temp = pd.read_csv(path+'/'+file)  
                                data_temp.rename(columns={data_temp.columns[0]:'Link_ID'}, inplace=True) 
                                data_temp['Link_ID']=data_class['Column1']
                                data=pd.merge(response,data_temp,on='Link_ID') 
                                print(data.shape)
                                data.drop(columns = 'Link_ID',inplace=True)   
                                data_train=data.iloc[0:num_train,:]
                                data_val=data.iloc[num_train+1:,:]
                                output = process_data(data_train,data_val,cv)  
                                tRF[k]=pd.DataFrame(output['f1-score'].values)         
                                k=k+1  
                        tRF.to_excel(writer, sheet_name=file_folder, index=True)  
                writer.save()                      


# In[9]:


end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

