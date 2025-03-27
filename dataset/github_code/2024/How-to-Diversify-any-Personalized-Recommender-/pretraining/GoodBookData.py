import numpy as np
import pandas as pd

def load_user_item_matrix_GB_All(max_user=943, max_item=761):
    """
    this function loads the user x items matrix from the  movie lens data set.
    Both input parameter represent a threshold for the maximum user id or maximum item id
    The highest user id is 6040 and the highest movie id is 3952 for the original data set, however, the masked data
    set contains only 943 users and 1330 items
    :return: user-item matrix
    """

    # Number of unique user_ids: 943
    # Number of unique book_ids: 761
    # Sparsity of the data: 0.9816
    df = np.zeros(shape=(max_user, max_item))
    with open("RecSys_News/goodbook/ratings_filtered_goodbook.csv", 'r') as f: #subset_GB_O All_2370_allUsers_KNN_fancy_imputation_GB_k_30

        for line in f.readlines():
            user_id, movie_id, rating, genre = line.split(",")
            user_id, movie_id, rating, genre = int(user_id), int(movie_id), float(rating), str (genre)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = rating

    return df

def load_user_item_matrix_GB_TrainingSet(max_user=943, max_item=729): #2370 2835 24676 11927 2835
    """
    this function loads the user x items matrix from the  movie lens data set.
    Both input parameter represent a threshold for the maximum user id or maximum item id
    The highest user id is 6040 and the highest movie id is 3952 for the original data set, however, the masked data
    set contains only 943 users and 1330 items
    :return: user-item matrix
    """
    df = np.zeros(shape=(max_user, max_item))
    with open("RecSys_News/goodbook/trainVal_small.csv", 'r') as f: # Flixster/trainingSet_GB_1.dat New_Flixster/GB_train.csv

        for line in f.readlines():
            user_id, movie_id, rating = line.split(",")
            user_id, movie_id, rating  = int(user_id), int(movie_id), float(rating)#, str (genre)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = rating

    return df


def load_user_item_matrix_GB_TestSet(max_user=943, max_item=729): #2370 2835 24676 11927 2835
    """
    this function loads the user x items matrix from the  movie lens data set.
    Both input parameter represent a threshold for the maximum user id or maximum item id
    The highest user id is 6040 and the highest movie id is 3952 for the original data set, however, the masked data
    set contains only 943 users and 1330 items
    :return: user-item matrix
    """
    df = np.zeros(shape=(max_user, max_item))
    with open("RecSys_News/goodbook/test_small.csv", 'r') as f: # Flixster/trainingSet_GB_1.dat New_Flixster/GB_train.csv

        for line in f.readlines():
            user_id, movie_id, rating = line.split(",")
            user_id, movie_id, rating  = int(user_id), int(movie_id), float(rating)#, str (genre)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = rating

    return df



def load_user_item_matrix_GB_Test(max_user=943, max_item=761): # 2370 2008
    """
    this function loads the user x items matrix from the  movie lens data set.
    Both input parameter represent a threshold for the maximum user id or maximum item id
    The highest user id is 6040 and the highest movie id is 3952 for the original data set, however, the masked data
    set contains only 943 users and 1330 items
    :return: user-item matrix
    """
    df = np.zeros(shape=(max_user, max_item))
    with open("RecSys_News/goodbook/test_small.csv", 'r') as f:

        for line in f.readlines():
            user_id, movie_id, rating, genre = line.split(",")
            user_id, movie_id, rating, genre = int(user_id), int(movie_id), float(rating), str (genre)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = rating

    return df

def load_user_item_GB_Complet(max_user=2370, max_item=2835):# 2370
    df = np.zeros(shape=(max_user, max_item))
    with open(
            "Flixster/With_Fancy_KNN/TrainingSet_2370_allUsers_KNN_fancy_imputation_GB_k_30.dat",
            'r') as f: 
        for line in f.readlines():
            user_id, movie_id, rating, timestamp = line.split("::")
            user_id, movie_id, rating, timestamp = int(user_id), int(movie_id), float(rating), int(timestamp)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id - 1, movie_id - 1] = rating

    return df

def load_user_item_matrix_GB_limited_ratings(limit=20):
    
    user_item = load_user_item_matrix_GB_All()
    user_item_limited = np.zeros(shape=user_item.shape)
    for user_index, user in enumerate(user_item):
        # filter rating indices
        rating_index = np.argwhere(user > 0).reshape(1, -1)[0]
        # shuffle them
        np.random.shuffle(rating_index)
        for i in rating_index[:limit]:
            user_item_limited[user_index, i] = user[i]
    #print(np.sum(user_item_limited, axis=1))
    return user_item_limited

def load_user_item_matrix_GB_trainMasked(max_user=2370, max_item=2835, file_index=-1):
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
    with open("Flixster/subset_GB_User_O.csv", 'r') as f: 
        for line in f.readlines()[:max_user]: 
            user_id, gender, _ = line.split(",") #, location, _, _, _ , _

            if gender == "M":
                gender_vec.append(0)
            else:
                gender_vec.append(1)

    return np.asarray(gender_vec)

def load_user_item_matrix_GB_masked(max_user=2370, max_item=2835, file_index=-1):
    files = [
        # Here add path to your files. Please note that we start from #0 like in the example 
        "Flixster/BlurMe/All_GB_blurme_obfuscated_0.01_greedy_avg_top-1.dat",#0
        
    ]
    df = np.zeros(shape=(max_user, max_item))

    with open(files[file_index], 'r') as f:
        for line in f.readlines():
            user_id, movie_id, rating, _ = line.split("::")
            user_id, movie_id, rating = int(user_id), int(movie_id), float(rating)
            if user_id <= max_user and movie_id <= max_item:
                df[user_id-1, movie_id-1] = rating

    return df
