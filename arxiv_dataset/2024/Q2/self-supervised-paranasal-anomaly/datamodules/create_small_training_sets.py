import pandas as pd 
import nibabel 
import numpy as np
import uuid
import glob
import os
import random
from sklearn.model_selection import StratifiedKFold,train_test_split

"""

this file is used to split the training set into smaller training sets

""" 

directory = "/home/debayan/Desktop/MRI_HCHS/JAMA-labelled-1000-corrected/splits/"

#get list of all folders in directory
folders = glob.glob(directory+"*")

for folder in folders:
    
    #get list of all files in each folder
    sub_folders = glob.glob(folder+"/*")

    for sub_folder in sub_folders:

        train_csv = pd.read_csv(sub_folder+"/train.csv")
        images = train_csv["folder"].tolist()
        labels = train_csv["label"].tolist()

        for percentage_of_training_set in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
            
            

            _, X_train, _, y_train = train_test_split(images, labels, test_size=percentage_of_training_set, random_state=42,stratify=labels)

            train_csv_percentage = pd.DataFrame({"folder":X_train,"label":y_train})
            train_csv_percentage.to_csv(sub_folder+"/train_"+str(percentage_of_training_set)+".csv",index=False)
            
    


"""




X_trainval, X_test, y_trainval, y_test = train_test_split(dataset, labels, test_size=0.30, random_state=42,stratify=labels)


"""













        


        


        





        






            





