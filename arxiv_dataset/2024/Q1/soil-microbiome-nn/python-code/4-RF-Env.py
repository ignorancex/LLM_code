#!/usr/bin/env python
# coding: utf-8

# In[5]:


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
from sklearn.ensemble import HistGradientBoostingClassifier
import openpyxl

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer


# In[6]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline


# In[7]:


import warnings
warnings.filterwarnings('ignore')


# In[8]:


from datetime import datetime
start_time = datetime.now()


# In[1]:


list_level = ['Class', 'Family', 'Genus', 'Order', 'Phylum']
path_response='response/response_original/'
path_x ='OTU/normalized_otu/'
path_soil = 'Env/soil_chemistry/'
path_disease = 'Env/disease_suppression/'
path_field='Env/field_information/'
path_alpha='OTU/alpha_diversity/'


# In[15]:


cv=RepeatedKFold(n_splits=10,n_repeats=3, random_state=100)


# In[16]:


def process_data(data_train,data_val,cv):  
    x_column_list = data_train.drop(columns=['y_b']).columns  
    percent_label=[round(100*len(np.where(data_train['y_b']==0)[0])/len(data_train)),round(100*len(np.where(data_val['y_b']==0)[0])/len(data_val))]  
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


# # Alpha diversity

# In[17]:


for file_response in os.listdir(path_response):  
    if (file_response != '.DS_Store') & (file_response != 'Icon\r'):  
        print(file_response)  
        path_r= path_response+file_response
        for re in os.listdir(path_r):  
            if re[0:8] == 'response':  
                response = pd.read_csv(path_r+'/'+re)
                response.rename(columns={'Column1':'Link_ID',response.columns[1]:'y_b'}, inplace=True)     
                writer= pd.ExcelWriter(path_r+'/'+'classification_RF_alpha_diversity'+'.xlsx', engine='xlsxwriter')   
                for file_folder in os.listdir(path_alpha):  
                    if (file_folder[-4:] != '.csv') & (file_folder != '.DS_Store')& (file_folder != 'Icon\r'):          
                        path = path_alpha+file_folder
                        file_list = []  
                        tRF=pd.DataFrame()  
                        tcluster=pd.DataFrame()  
                        k=0  
                        for file in os.listdir(path):  
                            if (file[0] != 'Y') & (file[-4:] == '.csv') & (file != '.DS_Store')& (file_folder != 'Icon\r'):  
                                print(file)  
                                file_list.append(file)  
                                data_temp = pd.read_csv(path+'/'+file)
                                data_temp.drop(columns={'Unnamed: 0'}, inplace=True)  
                                data=pd.merge(response,data_temp,on='Link_ID')   
                                data.drop(columns = 'Link_ID',inplace=True)  
                                data_train,data_val = train_test_split(data,train_size=0.8, random_state=42)  
                                output = process_data(data_train,data_val,cv)  
                                tRF[k]=pd.DataFrame(output['f1-score'].values)         
                                k=k+1  
                        tRF.to_excel(writer, sheet_name=file_folder, index=True)  
                writer.save()                      


# # Soil_chemistry

# In[18]:


for file_response in os.listdir(path_response):  
    if (file_response != '.DS_Store') & (file_response != 'Icon\r'):  
        print(file_response)  
        path_r= path_response+file_response
        for re in os.listdir(path_r):  
            if re[0:8] == 'response':  
                response = pd.read_csv(path_r+'/'+re)
                response.rename(columns={'Column1':'Link_ID',response.columns[1]:'y_b'}, inplace=True)     
                writer= pd.ExcelWriter(path_r+'/'+'classification_RF_soil_chemistry_2'+'.xlsx', engine='xlsxwriter')   
                k=0
                tRF=pd.DataFrame()
                for file in os.listdir(path_soil):  
                    if (file[0] != 's') & (file[-4:] == '.csv') & (file != '.DS_Store')& (file != 'Icon\r'):              
                        print(file)   
                        data_temp = pd.read_csv(path_soil+file)
                        data_temp.drop(columns={'Unnamed: 0'}, inplace=True)  
                        data=pd.merge(response,data_temp,on='Link_ID')   
                        data.drop(columns = 'Link_ID',inplace=True)
                        print(data.shape)
                        data.dropna(inplace=True)
                        print(data.shape)
                        data_train,data_val = train_test_split(data,train_size=0.8, random_state=42)  
                        output = process_data(data_train,data_val,cv)  
                        tRF[k]=pd.DataFrame(output['f1-score'].values)         
                        k=k+1  
                tRF.to_excel(writer, sheet_name=file_folder, index=True)  
        writer.save()                      


# # field_information

# In[19]:


D = pd.read_csv(path_field+'field_information.csv')
D.shape
D.isna().sum()


# In[20]:


DD = D.drop(columns=['Column1','pH_1_1','GPS_Lat','GPS_Lon','Variety'])
for var in DD:
    print(DD[var].unique().shape[0], 'unique values in', var, '\n',DD[var].unique(), '\n')
    print(DD[var].value_counts())


# In[21]:


rf_classifier = RandomForestClassifier(
min_samples_leaf=50,
n_estimators=150,
bootstrap=True,
oob_score=True,
n_jobs=-1,
random_state=42,
max_features='auto') 


# In[22]:


for file_response in os.listdir(path_response):  
    if (file_response != '.DS_Store') & (file_response != 'Icon\r'):  
        print(file_response)  
        path_r= path_response+file_response
        for re in os.listdir(path_r):  
            if re[0:8] == 'response':  
                response = pd.read_csv(path_r+'/'+re)
                response.rename(columns={'Column1':'Link_ID',response.columns[1]:'y_b'}, inplace=True)     
                writer= pd.ExcelWriter(path_r+'/'+'classification_RF_field_information_2'+'.xlsx', engine='xlsxwriter')   
                k=0
                tRF=pd.DataFrame()
                for file in os.listdir(path_field):  
                    if (file[-4:] == '.csv') & (file != '.DS_Store')& (file != 'Icon\r'):              
                        print(file)   
                        D = pd.read_csv(path_field+file)
                        A = D.columns[D.isna().sum()< 40]
                        D2 = D[A]
                        D2.rename(columns={'Column1':'Link_ID'},inplace=True)
                        features_to_encode = D2.columns.drop(['Link_ID','pH_1_1','Variety'])

                        col_trans = make_column_transformer(
                        (OneHotEncoder(),features_to_encode),
                        remainder = "passthrough")


                        data=pd.merge(response,D2,on='Link_ID')
                        #print(data)

                        #data=pd.merge(data,fe_data,on='Link_ID')  
                        data.drop(columns = ['Link_ID','pH_1_1','Variety'],inplace=True)
                        print(data.shape)
                        data.dropna(inplace=True)
                        print(data.shape)
                        AL = []
                        for l in range(0,50):
                            data_train,data_val = train_test_split(data,train_size=0.97, random_state=l)


                            x_column_list = data_train.drop(columns=['y_b']).columns  
                            #Classification RF 

                            pipe = make_pipeline(col_trans, rf_classifier)
                            #pipe.fit(X_train, y_train) 
                            # Fit on data  
                            pipe.fit(data_train[x_column_list],data_train['y_b'])  
                            y_valid = pipe.predict(data_val[x_column_list])

                            report_All = classification_report(data_val['y_b'],y_valid,output_dict=True)  
                            dAll=pd.DataFrame(report_All).transpose() 
                            AL.append(dAll['f1-score'].values)
                        AAA=pd.DataFrame(AL)
                        tRF[k]=AAA.mean()         
                        k=k+1  
                tRF.to_excel(writer, sheet_name=file_folder, index=True)  
        writer.save()  


# # disease_suppression

# In[23]:


D = pd.read_csv(path_disease+'disease_suppression.csv')


# In[24]:


A = D.columns[D.isna().sum()< 10]


# In[25]:


D2 = D[A]


# In[26]:


for file_response in os.listdir(path_response):  
    if (file_response != '.DS_Store') & (file_response != 'Icon\r'):  
        print(file_response)  
        path_r= path_response+file_response
        for re in os.listdir(path_r):  
            if re[0:8] == 'response':  
                response = pd.read_csv(path_r+'/'+re)
                response.rename(columns={'Column1':'Link_ID',response.columns[1]:'y_b'}, inplace=True)     
                writer= pd.ExcelWriter(path_r+'/'+'classification_RF_disease_suppression_2'+'.xlsx', engine='xlsxwriter')   
                k=0
                tRF=pd.DataFrame()
                for file in os.listdir(path_disease):  
                    if (file[0] != 'd') & (file[-4:] == '.csv') & (file != '.DS_Store')& (file != 'Icon\r'):              
                        print(file)   
                        data_temp = pd.read_csv(path_disease+file)
                        data_temp.drop(columns={'Unnamed: 0'}, inplace=True)  
                        data=pd.merge(response,data_temp,on='Link_ID')   
                        data.drop(columns = 'Link_ID',inplace=True)
                        print(data.shape)
                        data.dropna(inplace=True)
                        print(data.shape)
                        data_train,data_val = train_test_split(data,train_size=0.8, random_state=42)  
                        output = process_data(data_train,data_val,cv)  
                        tRF[k]=pd.DataFrame(output['f1-score'].values)         
                        k=k+1  
                tRF.to_excel(writer, sheet_name=file_folder, index=True)  
        writer.save()                      


# # Alpha+soil+disease+field

# In[27]:


fe=pd.read_csv(path_field+'field_information.csv')
fe.rename(columns={"Column1":'Link_ID'},inplace=True)
A = fe.columns[fe.isna().sum()< 30]
fe = fe[A]
level_cols = fe.columns 
level_cols =level_cols.drop(['Link_ID','Variety'])
features_to_encode = level_cols.drop(['pH_1_1'])
col_trans = make_column_transformer((OneHotEncoder(),features_to_encode),remainder = "passthrough")


# In[28]:


rf_classifier = RandomForestClassifier(
min_samples_leaf=50,
n_estimators=150,
bootstrap=True,
oob_score=True,
n_jobs=-1,
random_state=42,
max_features='auto')


# In[31]:


for file_response in os.listdir(path_response):  
    if (file_response != '.DS_Store') & (file_response != 'Icon\r'):  
        print(file_response)  
        path_r= path_response+file_response
        for re in os.listdir(path_r):  
            if re[0:8] == 'response':  
                response = pd.read_csv(path_r+'/'+re)
                response.rename(columns={'Column1':'Link_ID',response.columns[1]:'y_b'}, inplace=True)     
                writer= pd.ExcelWriter(path_r+'/'+'classification_alpha+soil+disease+field'+'.xlsx', engine='xlsxwriter')
                tRF=pd.DataFrame()   
                k=0  

                for level in os.listdir(path_alpha):
                    if (level[-4:] != '.csv') &  (level != '.DS_Store'):
                        for k in range(1,7):
                            f1 = pd.read_csv(path_alpha+level+'/'+str(k)+'.csv')
                            f2 = pd.read_csv(path_soil+str(k)+'.csv')
                            f12 = pd.merge(f1,f2,on='Link_ID')
                            f3 = pd.read_csv(path_disease+str(k)+'.csv')
                            f123=pd.merge(f12,f3,on='Link_ID')
                            f123.drop(columns=['Unnamed: 0_x','Unnamed: 0'],inplace=True)
                            data1 = pd.merge(response,f123,on='Link_ID')
                            fe_data=pd.concat([fe['Link_ID'],fe[level_cols]],axis=1)
                            
                            data = pd.merge(data1,fe_data,on='Link_ID')
                            #print(data)
                            
                            data.drop(columns = 'Link_ID',inplace=True) 
                            data.dropna(inplace=True)
                            print(data.shape)
                            data['Variety2'] = data['Variety2'].replace("RedLittle","Red")
                            
                            
                            AL = []
                            for l in range(0,100):
                                data_train,data_val = train_test_split(data,train_size=0.95, random_state=l)


                                x_column_list = data_train.drop(columns=['y_b']).columns  
                                #Classification RF 

                                pipe = make_pipeline(col_trans, rf_classifier)
                                #pipe.fit(X_train, y_train) 
                                # Fit on data  
                                pipe.fit(data_train[x_column_list],data_train['y_b'])  
                                y_valid = pipe.predict(data_val[x_column_list])

                                report_All = classification_report(data_val['y_b'],y_valid,output_dict=True)  
                                dAll=pd.DataFrame(report_All).transpose() 
                                AL.append(dAll['f1-score'].values)
                            AAA=pd.DataFrame(AL)

                            #output = process_data(data_train,data_val,cv)  
                            tRF[k]=AAA.mean()         
                            k=k+1  
                        tRF.to_excel(writer, sheet_name=level, index=True)  
                writer.save() 
    


# # soil+disease+field

# In[32]:


fe=pd.read_csv(path_field+'field_information.csv')
fe.rename(columns={"Column1":'Link_ID'},inplace=True)
A = fe.columns[fe.isna().sum()< 30]
fe = fe[A]
level_cols = fe.columns 
level_cols =level_cols.drop(['Link_ID','Variety'])
features_to_encode = level_cols.drop(['pH_1_1'])
col_trans = make_column_transformer((OneHotEncoder(),features_to_encode),remainder = "passthrough")


# In[33]:


rf_classifier = RandomForestClassifier(
min_samples_leaf=50,
n_estimators=150,
bootstrap=True,
oob_score=True,
n_jobs=-1,
random_state=42,
max_features='auto')


# In[34]:


for file_response in os.listdir(path_response):  
    if (file_response != '.DS_Store') & (file_response != 'Icon\r'):  
        print(file_response)  
        path_r= path_response+file_response
        for re in os.listdir(path_r):  
            if re[0:8] == 'response':  
                response = pd.read_csv(path_r+'/'+re)
                response.rename(columns={'Column1':'Link_ID',response.columns[1]:'y_b'}, inplace=True)     
                writer= pd.ExcelWriter(path_r+'/'+'classification_soil+disease+field'+'.xlsx', engine='xlsxwriter')
                tRF=pd.DataFrame()   
                kk=0  


                for k in range(1,7):
                    f2 = pd.read_csv(path_soil+str(k)+'.csv')
                    f3 = pd.read_csv(path_disease+str(k)+'.csv')
                    f23=pd.merge(f2,f3,on='Link_ID')
                    f23.drop(columns=['Unnamed: 0_x','Unnamed: 0_y'],inplace=True)
                    data1 = pd.merge(response,f23,on='Link_ID')
                    fe_data=pd.concat([fe['Link_ID'],fe[level_cols]],axis=1)

                    data = pd.merge(data1,fe_data,on='Link_ID')
                    #print(data)

                    data.drop(columns = 'Link_ID',inplace=True) 
                    data.dropna(inplace=True)
                    print(data.shape)
                    data['Variety2'] = data['Variety2'].replace("RedLittle","Red")


                    AL = []
                    for l in range(0,100):
                        data_train,data_val = train_test_split(data,train_size=0.95, random_state=l)


                        x_column_list = data_train.drop(columns=['y_b']).columns  
                        #Classification RF 

                        pipe = make_pipeline(col_trans, rf_classifier)
                        #pipe.fit(X_train, y_train) 
                        # Fit on data  
                        pipe.fit(data_train[x_column_list],data_train['y_b'])  
                        y_valid = pipe.predict(data_val[x_column_list])

                        report_All = classification_report(data_val['y_b'],y_valid,output_dict=True)  
                        dAll=pd.DataFrame(report_All).transpose() 
                        AL.append(dAll['f1-score'].values)
                    AAA=pd.DataFrame(AL)

                    #output = process_data(data_train,data_val,cv)  
                    tRF[kk]=AAA.mean()         
                    k=kk+1  
                tRF.to_excel(writer, index=True)  
        writer.save() 


# # soil+disease

# In[35]:


for file_response in os.listdir(path_response):  
    if (file_response != '.DS_Store') & (file_response != 'Icon\r'):  
        print(file_response)  
        path_r= path_response+file_response
        for re in os.listdir(path_r):  
            if re[0:8] == 'response':  
                response = pd.read_csv(path_r+'/'+re)
                response.rename(columns={'Column1':'Link_ID',response.columns[1]:'y_b'}, inplace=True)     
                writer= pd.ExcelWriter(path_r+'/'+'classification_RF+soil+disease'+'.xlsx', engine='xlsxwriter')   
                k=0
                tRF=pd.DataFrame()
                for k in range(1,7):
                    f2 = pd.read_csv(path_soil+str(k)+'.csv')
                    f3 = pd.read_csv(path_disease+str(k)+'.csv')
                    f23=pd.merge(f2,f3,on='Link_ID')
                    f23.drop(columns=['Unnamed: 0_x','Unnamed: 0_y'],inplace=True)
                    data = pd.merge(response,f23,on='Link_ID') 
                    data.drop(columns = 'Link_ID',inplace=True)
                    print(data.shape)
                    data.dropna(inplace=True)
                    print(data.shape)
                    data_train,data_val = train_test_split(data,train_size=0.8, random_state=42)  
                    output = process_data(data_train,data_val,cv)  
                    tRF[k]=pd.DataFrame(output['f1-score'].values)         
                    k=k+1  
                tRF.to_excel(writer, index=True)  
        writer.save() 


# # alpha+soil+disease features

# In[37]:


for file_response in os.listdir(path_response):  
    if (file_response != '.DS_Store') & (file_response != 'Icon\r'):  
        print(file_response)  
        path_r= path_response+file_response
        for re in os.listdir(path_r):  
            if re[0:8] == 'response':  
                response = pd.read_csv(path_r+'/'+re)
                response.rename(columns={'Column1':'Link_ID',response.columns[1]:'y_b'}, inplace=True)     
                writer= pd.ExcelWriter(path_r+'/'+'classification_alpha+soil+disease'+'.xlsx', engine='xlsxwriter')
                tRF=pd.DataFrame()   
                k=0  




                for level in os.listdir(path_alpha):
                    if (level[-4:] != '.csv') &  (level != '.DS_Store'):
                        for k in range(1,7):
                            f1 = pd.read_csv(path_alpha+level+'/'+str(k)+'.csv')
                            f2 = pd.read_csv(path_soil+str(k)+'.csv')
                            f12 = pd.merge(f1,f2,on='Link_ID')
                            f3 = pd.read_csv(path_disease+str(k)+'.csv')
                            f123=pd.merge(f12,f3,on='Link_ID')
                            f123.drop(columns=['Unnamed: 0_x','Unnamed: 0'],inplace=True)
                            data = pd.merge(response,f123,on='Link_ID')
                            print(data.shape)
                            
                            data.drop(columns = 'Link_ID',inplace=True) 
                            data.dropna(inplace=True)
                            print(data.shape)
                            data_train,data_val = train_test_split(data,train_size=0.8, random_state=42)  
                            output = process_data(data_train,data_val,cv)  
                            tRF[k]=pd.DataFrame(output['f1-score'].values)         
                            k=k+1  
                        tRF.to_excel(writer, sheet_name=level, index=True)  
                writer.save()                      
            
    


# # alpha+soil

# In[38]:


for file_response in os.listdir(path_response):  
    if (file_response != '.DS_Store') & (file_response != 'Icon\r'):  
        print(file_response)  
        path_r= path_response+file_response
        for re in os.listdir(path_r):  
            if re[0:8] == 'response':  
                response = pd.read_csv(path_r+'/'+re)
                response.rename(columns={'Column1':'Link_ID',response.columns[1]:'y_b'}, inplace=True)     
                writer= pd.ExcelWriter(path_r+'/'+'classification_alpha+soil'+'.xlsx', engine='xlsxwriter')
                tRF=pd.DataFrame()   
                k=0  

                for level in os.listdir(path_alpha):
                    if (level[-4:] != '.csv') &  (level != '.DS_Store'):
                        for k in range(1,7):
                            f1 = pd.read_csv(path_alpha+level+'/'+str(k)+'.csv')
                            f2 = pd.read_csv(path_soil+str(k)+'.csv')
                            f12 = pd.merge(f1,f2,on='Link_ID')
                            f12.drop(columns=['Unnamed: 0_x'],inplace=True)
                            data = pd.merge(response,f12,on='Link_ID')
                            print(data.shape)
                            
                            data.drop(columns = 'Link_ID',inplace=True) 
                            data.dropna(inplace=True)
                            print(data.shape)
                            data_train,data_val = train_test_split(data,train_size=0.8, random_state=42)  
                            output = process_data(data_train,data_val,cv)  
                            tRF[k]=pd.DataFrame(output['f1-score'].values)         
                            k=k+1  
                        tRF.to_excel(writer, sheet_name=level, index=True)  
                writer.save()                      
            
    


# In[ ]:


end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

