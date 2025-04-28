#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error   
import openpyxl
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[2]:


import h2o
#h2o.shutdown()  # Shutdown the current cluster
h2o.init(max_mem_size="8G")  # Adjust to your available memory, e.g., 8G for 8 GB


# In[3]:


#import h2o
from h2o.automl import H2OAutoML
import os
import pandas as pd
import xlsxwriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
#h2o.init()
#h2o.init(max_mem_size="4G")


# In[5]:


import warnings
warnings.filterwarnings('ignore')


# In[6]:


from datetime import datetime
start_time = datetime.now()


# In[7]:



path_response='response/response_H2o/'
path_x = 'OTU/normalized_otu/'


# In[8]:


for file_response in os.listdir(path_response):
    if file_response not in ['.DS_Store', 'Icon\r']:
        print(file_response)
        path_r = os.path.join(path_response, file_response)
        os.chdir(path_r)
        
        # Initialize Excel writer for classification results (separate for each response file)
        writer = pd.ExcelWriter(os.path.join(path_r, 'classification_results.xlsx'), engine='xlsxwriter')  
        
        for re in os.listdir(path_r):
            if re.startswith('response'):
                response = pd.read_csv(os.path.join(path_r, re))
                response.rename(columns={response.columns[0]: 'Link_ID', response.columns[1]: 'y_c'}, inplace=True)

                print(response)
                
                for file_folder in os.listdir(path_x):
                    if file_folder[-4:] != '.csv' and file_folder not in ['.DS_Store', 'Icon\r']:
                        path = os.path.join(path_x, file_folder)
                        os.chdir(path)
                        
                        classification_results = pd.DataFrame()  # Create a DataFrame to hold concatenated results
                        
                        for file in os.listdir(path):
                            if file.endswith('.csv') and file[0] != 't' and file not in ['.DS_Store', 'Icon\r']:
                                print(file)

                                # Set sheet name to the folder name (which corresponds to the taxonomic level)
                                sheet_name = file_folder  
                                
                                # Define Excel writer for model results
                                writer_model = pd.ExcelWriter(os.path.join(path_r, f'model_H20_{file_folder}_{file}.xlsx'), engine='xlsxwriter')
                                
                                data_temp = pd.read_csv(file)
                                data_temp.rename(columns={data_temp.columns[0]: 'Link_ID'}, inplace=True)
                                
                                data = pd.merge(response, data_temp, on='Link_ID')
                                data.drop(columns="Link_ID", inplace=True)
                                
                                train_pd, test_pd = train_test_split(data, train_size=0.8, random_state=42)
                                
                                train_pd.to_csv('train.csv', index=False)
                                test_pd.to_csv('test.csv', index=False)
                                
                                train = h2o.import_file('train.csv')
                                test = h2o.import_file('test.csv')
                                
                                x = train.columns
                                y = "y_c"
                                x.remove(y)
                                # Convert response to a factor (for binary classification)
                                train[y] = train[y].asfactor()
                                test[y] = test[y].asfactor()

                                
                                aml = H2OAutoML(max_runtime_secs=70, seed=1)
                                aml.train(x=x, y=y, training_frame=train)
                                
                                lb = aml.leaderboard
                                print(lb)
                                
                                m = aml.get_best_model(criterion="logloss")
                                        
                                if m is not None:
                                    preds = m.predict(test)
                                    y_valid = preds.as_data_frame()
                                    y_pred = y_valid['predict'].to_numpy()
                                    y_true = test_pd['y_c'].values
                                    
                                    accuracy = accuracy_score(y_true, y_pred)
                                    precision = precision_score(y_true, y_pred)
                                    recall = recall_score(y_true, y_pred)
                                    f1 = f1_score(y_true, y_pred)

                                    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")

                                    classification_metrics = pd.DataFrame({
                                        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1'],
                                        file: [accuracy, precision, recall, f1]
                                    })

                                    classification_metrics.set_index('Metric', inplace=True)

                                    # Concatenate results for different files in the same sheet (side by side)
                                    classification_results = pd.concat([classification_results, classification_metrics], axis=1)

                                lb_df = lb.as_data_frame()
                                lb_df.to_excel(writer_model, sheet_name=sheet_name, index=True)  # Save model results
                                writer_model.save()

                        # Save the classification results for the current taxonomic level (concatenated results)
                        classification_results.to_excel(writer, sheet_name=sheet_name, index=True)

        writer.save()  # Save the classification file with separate sheets at the end of processing


# In[ ]:


end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


# In[ ]:




