import pandas as pd 
import nibabel 
import numpy as np
import uuid
import glob
import os
import random




def create_dir(path):
    """ Create a directory. """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"Error: creating directory with name {path}")

split_directory = "/home/debayan/Desktop/MRI_HCHS/JAMA-labelled-1000/splits/"
size_of_crops = [65,70,75,80]
number_of_folds = 5
std = 1




    
for fold in range(1,number_of_folds+1):

    df_train = None 
    df_val   = None 
    df_test  = []
    df_train_val = None 
    crop_size_folder = ""

    for size in size_of_crops:

        crop_size_folder = crop_size_folder + f"{size}_"

        split_location =  f"/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/dataset/JAMA-labelled-1000/splits/crop_size_{size}_std_factor_{std}/fold_{fold}/"

        

        df_train = pd.read_csv(split_location + "train.csv")
        df_val   = pd.read_csv(split_location + "val.csv") 
        df_test  = pd.read_csv(split_location + "test.csv") 

        df_train_val_test = pd.concat([df_train, df_val,df_test],ignore_index=True)

        df_train_val_test.to_csv(split_location + f"train_val_test.csv")
