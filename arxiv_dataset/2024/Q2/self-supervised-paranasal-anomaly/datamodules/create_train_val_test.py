import pandas as pd 
import nibabel 
import numpy as np
import uuid
import glob
import os
import random
from sklearn.model_selection import StratifiedKFold,train_test_split

def create_dir(path):
    """ Create a directory. """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"Error: creating directory with name {path}")


"""

this file is used to split the dataset into train validation and test split

""" 

split_directory = "/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/dataset/CARS_labelled/splits/"
size_of_crops = [15,20,25,30,35,40,45,50,55]
std_factors = [1]
random.seed(10)
for size_of_crop in size_of_crops:
    
    for std in std_factors:

        root_save_path =  f"/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/dataset/CARS_labelled/crop_size_{size_of_crop}_std_factor_{std}/"
        subd =  [ f.path for f in os.scandir(root_save_path) if f.is_dir() ]
        normal_dirs = []
        abnormal_dirs = []

        for subdir in subd:    
            directories =  [ f.path for f in os.scandir(subdir) if f.is_dir()]
            for dir in directories:
                if "normal" in dir: 
                    normal_dirs.append(dir) 
                elif "polyp" in dir or "cyst" in dir:
                    abnormal_dirs.append(dir)

        random.shuffle(normal_dirs)
        random.shuffle(abnormal_dirs)

        #Test 

        dataset = np.array(normal_dirs + abnormal_dirs)
        labels = np.array([0]*len(normal_dirs) + [1]*len(abnormal_dirs))
        skf = StratifiedKFold(n_splits=10)
        fold = 1


        for trainval_index, test_index in skf.split(dataset, labels):

            #print("TRAIN:", trainval_index, "TEST:", test_index)
            X_trainval, X_test = dataset[trainval_index], dataset[test_index]
            y_trainval, y_test = labels[trainval_index], labels[test_index]
            print(len(labels[labels==1])/len(labels))
            print(len(y_test[y_test==1])/len(y_test))

            
            X_train, X_val, y_train, y_val = train_test_split( X_trainval, y_trainval, test_size=0.10, random_state=42,stratify=y_trainval)
            print("Anomaly percentage in training + val set",len(y_trainval[y_trainval==1])/len(y_trainval))
            print("Anomaly percentage in val set", len(y_val[y_val==1])/len(y_val))

            print("Train set %",len(y_train)/len(dataset))
            print("Val set %",len(y_val)/len(dataset))
            print("Train Val set %",len(y_trainval)/len(dataset))
            print("Test set %",len(y_test)/len(dataset))

            print("Total Training samples",len(y_train))
            print("Total Validation samples",len(y_val))
            print("Total Test samples",len(y_test))
            

            print("####")

            train_dataframe = pd.DataFrame.from_dict({"folder":X_train, "label": y_train})
            val_dataframe   = pd.DataFrame.from_dict({"folder":X_val, "label": y_val})
            test_dataframe  = pd.DataFrame.from_dict({"folder":X_test, "label": y_test})
            trainval_dataframe = pd.DataFrame.from_dict({"folder":X_trainval, "label": y_trainval})
            create_dir(split_directory + f"/crop_size_{size_of_crop}_std_factor_{std}/fold_{fold}/")
            train_dataframe.to_csv(split_directory + f"/crop_size_{size_of_crop}_std_factor_{std}/fold_{fold}/" + "train.csv")
            val_dataframe.to_csv(split_directory + f"/crop_size_{size_of_crop}_std_factor_{std}/fold_{fold}/" + "val.csv")
            test_dataframe.to_csv(split_directory + f"/crop_size_{size_of_crop}_std_factor_{std}/fold_{fold}/" + "test.csv")
            trainval_dataframe.to_csv(split_directory + f"/crop_size_{size_of_crop}_std_factor_{std}/fold_{fold}/" + "trainval.csv")
            fold+=1















        


        


        





        






            





