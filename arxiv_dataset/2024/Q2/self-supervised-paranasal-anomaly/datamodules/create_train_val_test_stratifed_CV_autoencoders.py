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


def count_repetitions(my_list, text):

    count_dict = {}
    for element in my_list:
        if element in count_dict:
            count_dict[element] += 1
        else:
            count_dict[element] = 1

    # Calculate the total number of elements in the list
    total_elements = len(my_list)

    # Iterate over the dictionary to print the percentage and count of each element
    total_anomalies = 0
    for element, count in count_dict.items():

        if element > 0: 
            total_anomalies+=count

        percentage = (count / total_elements) * 100
        print(f"For {text}, {element} appears {count} times, which is {percentage}% of the list.")

    print(f"For {text}, anomaly appears {total_anomalies} times, which is {(total_anomalies / total_elements) * 100}% of the list.")
    print(f"For {text}, normal appears {total_elements - total_anomalies} times, which is {100 - ((total_anomalies / total_elements) * 100)}% of the list.")

    print("----------------------------------------------------------------------------------------------------------")

"""

this file is used to split the dataset into train validation and test split

""" 

split_directory = "/home/debayan/Desktop/MRI_HCHS/JAMA-labelled-1000/splits/"
size_of_crops = [35,40,45,50,55,60,65,70,75,80] #[15,20,25,30,35,40,45,50,55]
std_factors = [1]
random.seed(10)
for size_of_crop in size_of_crops:
    
    for std in std_factors:

        root_save_path =  f"/home/debayan/Desktop/MRI_HCHS/JAMA-labelled-1000/crop_size_{size_of_crop}_std_factor_{std}/"
        subd =  [ f.path for f in os.scandir(root_save_path) if f.is_dir() ]
        normal_dirs = []
        mucosal_thickening_dirs = []
        polyps_dirs = []
        cysts_dirs = []
        fully_occupied_dirs = []

        for subdir in subd:    
            directories =  [ f.path for f in os.scandir(subdir) if f.is_dir()]
            for dir in directories:
                if "normal" in dir: 
                    normal_dirs.append(dir) 
                elif "polyp" in dir:
                    polyps_dirs.append(dir)
                elif "cyst" in dir: 
                    cysts_dirs.append(dir)
                elif "fully_occupied" in dir: 
                    fully_occupied_dirs.append(dir)
                elif "mucosal_thickening" in dir: 
                    mucosal_thickening_dirs.append(dir)
                


        #random.shuffle(normal_dirs)
        #random.shuffle(abnormal_dirs)

        #Test 

        dataset = np.array(normal_dirs + mucosal_thickening_dirs + polyps_dirs + cysts_dirs + fully_occupied_dirs)
        labels = np.array([0]*len(normal_dirs) + [1]*len(mucosal_thickening_dirs) + [2]*len(polyps_dirs) + [3]*len(cysts_dirs) + [4]*len(fully_occupied_dirs))
        #skf = StratifiedKFold(n_splits=2)
        fold = 1

        X_trainval, X_test, y_trainval, y_test = train_test_split(dataset, labels, test_size=0.30, random_state=42,stratify=labels)
        #for trainval_index, test_index in skf.split(dataset, labels):

            #print("TRAIN:", trainval_index, "TEST:", test_index)
        #X_trainval, X_test = dataset[trainval_index], dataset[test_index]
        #y_trainval, y_test = labels[trainval_index], labels[test_index]
        #print(len(labels[labels==1])/len(labels))
        #print(len(y_test[y_test==1])/len(y_test))
        count_repetitions(labels,"Data set Distribution")
        count_repetitions(y_trainval,"Training & Validation Set Distribution")
        count_repetitions(y_test,"Test Set Distribution")
        #StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

        skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

        for i, (train_index, val_index) in enumerate(skf.split(X_trainval, y_trainval)):
            #print(f"Fold {i}:")
        
            X_train, X_val = X_trainval[train_index], X_trainval[val_index]
            y_train, y_val = y_trainval[train_index], y_trainval[val_index]

            #Stores the indices of all normal MS crops  
            X_train_healthy = [x for i,x in enumerate(X_train) if y_train[i] == 0]
            y_train_healthy = [0]*len(X_train_healthy)

            

            count_repetitions(y_train,"Training Set Distribution")
            count_repetitions(y_val,"Validation Set Distribution")
        

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















        


        


        





        






            





