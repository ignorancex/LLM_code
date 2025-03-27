"""
This function will help in the creation of the personalized list via the generation of confidence score + imputed matrix
you need to go to KNN-knn_impute_few_observed. I uploaded few_observed_entries.py file with the needed added line codes 
that you need to adapt to your few_observed_entries.py file 
"""
import numpy as np
import pandas as pd
from fancyimpute import KNN, MatrixFactorization, NuclearNormMinimization, SoftImpute, BiScaler
import json
import matplotlib.pyplot as plt
import GoodBookData as GBD
import MinData as MD


print ("Hi")
X = MD.load_user_item_matrix_Mind_TrainingSet()

print (X)

# Create a binary mask to indicate missing values
mask = np.isnan(X)

print ("start imputation")

# Initialize KNN imputer
knn_imputer = KNN(k=10000)

# Impute the missing values using the solve method with both data and mask
X_imputed = knn_imputer.solve(X, mask)

# Round imputed values to the nearest integer
X_filled_knn = np.rint(X_imputed)

output_file = "/RecSys_News/MIND/MINDlarge/train/"

print("saving") 
rating = 1
with open(output_file + "TrainingSet_Large_MinD.csv", 'w') as f:
    for index_user, user in enumerate(X_filled_knn):
        for index_movie, rating in enumerate(user):
            if rating > 0:
                f.write(
                    str(index_user + 1) + "," + str(index_movie + 1) + "," + str(int(np.round(rating))) + ",000000000\n")

# Calculate MSE
knn_mse = ((X_filled_knn[mask] - X[mask]) ** 2).mean()
print("knnImpute MSE: %f" % knn_mse)


"""
print ("Hi")
X = MD.load_user_item_matrix_Mind_TrainingSet()
#X, reverse_user_id_mapping, reverse_item_id_mapping =MD.load_user_item_matrix_Mind_TrainingSet ()
# Access the original user and item IDs using the reverse mappings
#original_user_id = reverse_user_id_mapping[0]
#original_item_id = reverse_item_id_mapping[0]
print (X)
# X= X[:2000, :15557]
X [X == 0] = np.nan
print ("start imputation")
knn_imputer = KNN(k= 10000)
# imputing the missing value with knn imputer
X_imputed = knn_imputer.fit_transform(X)

X_filled_knn = np.rint(X_imputed)


output_file = "/RecSys_News/MIND/MINDlarge/train/"
#with open(output_file + "TrainingSet_users_KNN_fancy_imputation_MinD_k1000_AllUsers.dat", 'w') as f:
#    for index_user, user in enumerate(X_filled_knn):
#        for index_movie, rating in enumerate(user):
#            if rating > 0:
                # Use original_user_id and original_item_id to get the original IDs
#                original_user = reverse_user_id_mapping[index_user]
#                original_item = reverse_item_id_mapping[index_movie]
#                f.write(str(original_user) + "::" + str(original_item) + "::" + str(int(np.round(rating))) + "::000000000\n")
print("saving") 
rating = 1
with open(output_file + "TrainingSet_Large_MinD.csv",
                  'w') as f:
        for index_user, user in enumerate(X):
            for index_movie, rating in enumerate(user):
                if rating > 0:
                    f.write(
                        str(index_user + 1) + "," + str(index_movie + 1) + "," + str(
                            int(np.round(rating))) + ",000000000\n")

knn_mse = ((X_filled_knn[missing_mask] - X[missing_mask]) ** 2).mean()
print("knnImpute MSE: %f" % knn_mse)
"""
