import numpy as np
import pandas as pd

import csv


def load_user_item_matrix_Mind_TrainingSet_server(max_user=711222, max_item=79547):
    """
    This function loads the user x items matrix from the MIND dataset.
    Both input parameters represent a threshold for the maximum user id or maximum item id.
    :return: user-item matrix
    """
    print("Loading user-item matrix...", flush=True)
    
    df = np.full((max_user, max_item), np.nan)
    
    # Read the CSV file using the csv module
    with open("/RecSys_News/MIND/MINDlarge/train/behaviour_News_interaction_train.csv", 'r') as f:
        reader = csv.reader(f)
        
        # Loop over each row in the CSV file
        for row in reader:
            user_id, movie_id, rating = map(int, row[:3])
            
            # Check if the user_id and movie_id are within the specified limits
            # if user_id <= max_user and movie_id <= max_item:
                # Update the corresponding entry in the NumPy array
            df[user_id - 1, movie_id - 1] = rating
    
    print("User-item matrix loaded successfully.", flush=True)
    return df

def load_user_item_matrix_Mind_TrainingSet(max_user= 1000, max_item= 9368): 
    """
    this function loads the user x items matrix from the  movie lens data set.
    Both input parameter represent a threshold for the maximum user id or maximum item id
    The highest user id is 6040 and the highest movie id is 3952 for the original data set, however, the masked data
    set contains only 50000 users and 1330 items
    :return: user-item matrix
    """
    print ("user-item matrix", flush = True)
    df = np.zeros(shape=(max_user, max_item))
    with open("MIND/mind_version1/MIND_Demo_Version1/train/behaviour_News_interaction_train_1000Users.csv", 'r') as f: 
# /RecSys_News/MIND/MINDlarge/train/behaviour_News_interaction_train.csv
        for line in f.readlines():
            user_id, movie_id = line.split(",")
            user_id, movie_id = int(user_id), int(movie_id)#, float(rating)#, str (genre)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = 1
    return df

def load_user_item_matrix_Mind_TrainingSet11(max_user= 1000, max_item= 9368): 
    """
    this function loads the user x items matrix from the  movie lens data set.
    Both input parameter represent a threshold for the maximum user id or maximum item id
    The highest user id is 6040 and the highest movie id is 3952 for the original data set, however, the masked data
    set contains only 50000 users and 1330 items
    :return: user-item matrix
    """
    print ("user-item matrix", flush = True)
    df = np.zeros(shape=(max_user, max_item))
    with open("MIND/mind_version1/MIND_Demo_Version1/train/behaviour_News_interaction_train_1000Users.csv", 'r') as f: 
# /RecSys_News/MIND/MINDlarge/train/behaviour_News_interaction_train.csv
        for line in f.readlines():
            user_id, movie_id, rating = line.split(",")
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)#, str (genre)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = rating
    return df


def load_user_item_matrix_Mind_TestSet(max_user=5000, max_item= 15557): #2370 2835 24676 11927 2835
    """
    this function loads the user x items matrix from the  movie lens data set.
    Both input parameter represent a threshold for the maximum user id or maximum item id
    The highest user id is 6040 and the highest movie id is 3952 for the original data set, however, the masked data
    set contains only 50000 users and 1330 items
    :return: user-item matrix
    """
    df = np.zeros(shape=(max_user, max_item))
    with open("MIND/mind_version1/MIND_Demo_Version1/valid/behaviour_News_interaction_valid.csv", 'r') as f: # Flixster/trainingSet_Mind_1.dat New_Flixster/GB_train.csv

        for line in f.readlines():
            user_id, movie_id = line.split(",")
            user_id, movie_id  = int(user_id), int(movie_id) #, float(rating)#, str (genre)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = 1

    return df



def load_user_item_matrix_Mind_Test(max_user=5000, max_item= 15557): # 2370 2008
    """
    this function loads the user x items matrix from the  movie lens data set.
    Both input parameter represent a threshold for the maximum user id or maximum item id
    The highest user id is 6040 and the highest movie id is 3952 for the original data set, however, the masked data
    set contains only 50000 users and 1330 items
    :return: user-item matrix
    """
    df = np.zeros(shape=(max_user, max_item))
    with open("/RecSys_News/goodbook/test_small.csv", 'r') as f:

        for line in f.readlines():
            user_id, movie_id = line.split(",")
            user_id, movie_id  = int(user_id), int(movie_id) #, float(rating)#, str (genre)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = 1

    return df

def load_user_item_Mind_Complet(max_user=2370, max_item=2835):# 2370
    df = np.zeros(shape=(max_user, max_item))
    with open(
            "Flixster/With_Fancy_KNN/TrainingSet_2370_allUsers_KNN_fancy_imputation_Mind_k_30.dat",
            'r') as f: 
        for line in f.readlines():
            user_id, movie_id = line.split(",")
            user_id, movie_id  = int(user_id), int(movie_id) #, float(rating)#, str (genre)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = 1

    return df


def load_user_item_matrix_Mind_trainMasked(max_user=2370, max_item=2835, file_index=-1):
    df = np.zeros(shape=(max_user, max_item))
    masked_files = [
        # ,#0
        
    ]
    with open(masked_files[file_index], 'r') as f:
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split("::")
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = rating

    return df




def load_gender_vector_GB(max_user=2370 ): #2370 2008
    """
        this function loads and returns the gender for all users with an id smaller than max_user
        :param max_user: the highest user id to be retrieved
        :return: the gender vector
        """
    gender_vec = []
    with open("Flixster/subset_Mind_User_O.csv", 'r') as f: 
        for line in f.readlines()[:max_user]: 
            user_id, gender, _ = line.split(",") #, location, _, _, _ , _

            if gender == "M":
                gender_vec.append(0)
            else:
                gender_vec.append(1)

    return np.asarray(gender_vec)

def load_user_item_matrix_Mind_masked(max_user=2370, max_item=2835, file_index=-1):
    files = [
        "Flixster/BlurMe/All_Mind_blurme_obfuscated_0.01_greedy_avg_top-1.dat",#0
        
    ]
    df = np.zeros(shape=(max_user, max_item))

    with open(files[file_index], 'r') as f:
        for line in f.readlines():
            user_id, movie_id = line.split(",")
            user_id, movie_id  = int(user_id), int(movie_id) #, float(rating)#, str (genre)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = 1

    return df
