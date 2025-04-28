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
size_of_crops = [15,20,25,30,35,40,45,50,55]
number_of_folds = 5
sizes_to_combine = [35,40,45,50,55,60,65,70,75,80]
std = 1




    
for fold in range(1,number_of_folds+1):

    df_train = None 
    df_val   = None 
    df_test  = []
    df_train_val = None 
    crop_size_folder = ""

    for size in sizes_to_combine:

        crop_size_folder = crop_size_folder + f"{size}_"

        split_location =  f"/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/dataset/JAMA-labelled-1000/splits/crop_size_{size}_std_factor_{std}/fold_{fold}/"

        if df_train is None: 

            df_train = pd.read_csv(split_location + "train.csv")
            df_val   = pd.read_csv(split_location + "val.csv") 
            
            df_train_val   = pd.read_csv(split_location + "trainval.csv")

        else: 

            df_train_temp = pd.read_csv(split_location + "train.csv")
            df_val_temp   = pd.read_csv(split_location + "val.csv") 
            
            df_train_val_temp   = pd.read_csv(split_location + "trainval.csv")


            df_train = pd.concat([df_train, df_train_temp])
            df_val   = pd.concat([df_val, df_val_temp])
            #df_test   = pd.concat([df_test, df_test_temp])
            df_train_val = pd.concat([df_train_val, df_train_val_temp])
        
        df_test.append(pd.read_csv(split_location + "test.csv"))
        
        
    split_location = f"/home/debayan/Desktop/MRI_HCHS/JAMA-labelled-1000/splits_corrected/crop_size_{crop_size_folder}_std_factor_{std}/fold_{fold}/"
    create_dir(split_location)
    df_train['folder'] = df_train['folder'].str.replace('/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/dataset/JAMA-labelled-1000','/home/debayan/Desktop/MRI_HCHS/JAMA-labelled-1000/')
    df_train.to_csv(split_location + "train.csv")

    df_val['folder'] = df_val['folder'].str.replace('/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/dataset/JAMA-labelled-1000','/home/debayan/Desktop/MRI_HCHS/JAMA-labelled-1000/')
    df_val.to_csv(split_location + "val.csv")
    
    df_train_val.to_csv(split_location + "trainval.csv")

    for index,size in enumerate(sizes_to_combine):

        df_test[index]['folder'] = df_test[index]['folder'].str.replace('/media/debayan/c7b64c90-ca4e-4192-8ed9-8fea1d005196/MRI_HCHS/dataset/JAMA-labelled-1000','/home/debayan/Desktop/MRI_HCHS/JAMA-labelled-1000/')
        df_test[index].to_csv(split_location + f"test_{size}.csv")


