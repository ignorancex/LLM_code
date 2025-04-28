"""
- Module implements make predictions and save prediction result as .csv extended file 

-Parameters:
------------

-representation_name: String
Protein representation model name

-data_preproceed: Dataframe with Entry and Vector columns

-tested_model: List
Saved model directory

-classifier_name: List 
Classifier model name

"""
import os
import pandas as pd
import numpy as np
import psutil
import joblib
import pickle
from random import random
import sys
import ast
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import Pipeline
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from Function_Prediction import binary_pytorch_network
import copy

def make_prediction(representation_name,data_preproceed,tested_model,classifier_name):

    
    protein_representation = data_preproceed.drop(["Entry"], axis=1)
    proteins=list(data_preproceed['Entry'])
    vectors=list(protein_representation['Vector'])
    protein_and_representation_dictionary=dict(zip(proteins,vectors ))
    row = protein_representation.shape[0]
    row_val = round(math.sqrt(row), 0)
    representation_vector = [ast.literal_eval(label) for label in protein_representation['Vector']]  
    protein_representation_array = np.array(representation_vector, dtype=float)    
    f_max_cv = []  
    path=os.path.join(os.getcwd(),"case_study/case_study_results")
    if 'prediction'  not in os.listdir(path):
        os.makedirs(os.path.join(path,"prediction"),exist_ok=True)         
    
    index=0
    
    for i in range(len(classifier_name)):
        label_lst=[]
        model_label_pred_lst=[]
        if (classifier_name[i]=='RandomForestClassifier'):
           
            #params=ast.literal_eval(parameters[i]['best parameter'][0])
            model = joblib.load(tested_model[i])
            sc = StandardScaler()
            representation_vector_std = sc.fit_transform(representation_vector)           
            model_label_pred_lst=model.predict(representation_vector_std)   
            index=index+1
        elif (classifier_name[i]=='SVC'):
                    
            #params=ast.literal_eval(parameters[i]['best parameter'][0])
            model = joblib.load(tested_model[i])
            
            model_label_pred_lst=model.predict(representation_vector)   
        elif (classifier_name[i]=='KNeighborsClassifier'):
                         
            #params=ast.literal_eval(parameters[i]['best parameter'][0])
            model = joblib.load(tested_model[i])           
            model_label_pred_lst=model.predict(representation_vector)   
      
     
        if (classifier_name[i]== 'Fully_Connected_Neural_Network'):
            
            #import pdb; pdb.set_trace()
            input_size= len(protein_representation_array[0])
            class_num=1
            
            model_class=binary_pytorch_network.Net(input_size,class_num)
            #import pdb; pdb.set_trace()
            model_class.load_state_dict(copy.deepcopy(torch.load(tested_model[i])))
            model_class.eval()
            x = torch.tensor(representation_vector)
            x=x.float()#.double()
            model_label_pred_lst=model_class(x) 
            model_label_pred_lst[model_label_pred_lst >= 0.] = 1    
            model_label_pred_lst[model_label_pred_lst < 0.] = 0
            input_size= len(protein_representation_array)
            protein_name=[]              
            for protein, vector in protein_and_representation_dictionary.items():                 
                protein_name.append(protein)
            model_label_pred_lst= model_label_pred_lst.type(torch.int16)
            model_label_pred_lst=[k for i in model_label_pred_lst.tolist() for k in i]    
 
                       
        else:       
            protein_name=[]              
            for protein, vector in protein_and_representation_dictionary.items():                 
                protein_name.append(protein)
           
                  
        label_predictions=pd.DataFrame(model_label_pred_lst,columns=["Label"])
        #import pdb
        #pdb.set_trace()
        label_predictions.insert(0, "protein_id", protein_name)                  
        label_predictions.to_csv(os.path.join(path,'prediction',representation_name+'_'+"prediction_" +"binary_classifier"+ '_' + classifier_name[i]+".csv"), index=False)        