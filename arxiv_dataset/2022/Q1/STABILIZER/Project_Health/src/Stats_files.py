import pandas as pd
import numpy as np
import math
import pickle

import utils


# def create_files(path, goal):
#     with open(path + '/Stats/' + goal + '.txt', 'w') as f:
#         seen_self = False
#         for month in [6]:
#             results_df = pd.read_csv(path + '/month_' + str(month) + '_models/' + goal + '/results_tuned.csv')
#             results_df.drop(['Unnamed: 0'], axis = 1, inplace = True)
#             for level in results_df.level.unique():
#                 sub_df = results_df[results_df['level'] == level]
#                 for col in ['self', 'bellwether']:
#                     if seen_self and col == 'self':
#                         continue
#                     values = sub_df[col].values.tolist()
#                     key = 'month_' + str(month) + '_level_' + str(level) + '_' + col
#                     if col == 'self':
#                         key = col
#                     f.write("%s \n" % key)
#                     for i in values:
#                         f.write("%s " % i)
#                     f.write("\n\n")   
#                     if col == 'self':
#                         seen_self = True

# def create_files(path, goal):
#     with open(path + '/with_CFS_DE/Stats/' + goal + '.txt', 'w') as f:
#         seen_self = False
#         for month in [6]:
#             results_df = pd.read_csv(path + '/with_CFS_DE/month_' + str(month) + '_models/' + goal + '/results_tuned_final.csv')
#             results_df.drop(['Unnamed: 0'], axis = 1, inplace = True)
#             for level in results_df.level.unique():
#                 sub_df = results_df[results_df['level'] == level]
#                 for col in ['self', 'bellwether','conv_bell']:
#                     values = sub_df[col].values.tolist()
#                     key = 'month_' + str(month) + '_level_' + str(level) + '_' + col
#                     if col == 'self':
#                         key = col + '_level_' + str(level)
#                     f.write("%s \n" % key)
#                     for i in values:
#                         f.write("%s " % i)
#                     f.write("\n\n")   
#                     if col == 'self':
#                         seen_self = True

def create_files(path, goal):
    for criteria in ['mre','sa']:
        with open(path + '/with_CFS_DE/Stats_new/' + goal + '_' + criteria + '.txt', 'w') as f:
            seen_self = False
            for month in [6]:
                results_df = pd.read_csv(path + '/with_CFS_DE/month_' + str(month) + '_models/' + goal + '/results_tuned_final_sa.csv')
                results_df.drop(['Unnamed: 0'], axis = 1, inplace = True)
                for level in results_df.level.unique():
                    sub_df = results_df[results_df['level'] == level]
                    for col in ['self', 'bellwether','conv_bell','global']:
                        values = sub_df[col + '_' + criteria].values.tolist()
                        key = 'month_' + str(month) + '_level_' + str(level) + '_' + col
                        if col == 'self' + '_' + criteria:
                            key = col + '_level_' + str(level)
                        f.write("%s \n" % key)
                        for i in values:
                            f.write("%s " % i)
                        f.write("\n\n")   
                        if col == 'self':
                            seen_self = True

if __name__ == "__main__":
    path = 'results'
    for i in range(7):
        goal = utils.get_goal(i)
        print("goal:", goal)
        create_files(path, goal)