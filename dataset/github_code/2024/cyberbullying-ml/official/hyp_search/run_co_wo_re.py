# This script combines the specific cyberbullying labels of Dataset 1 such that the dataset only cyberbullying and non-cyberbullying labels. 

import pandas as pd
import numpy as np

import re

from scipy.stats.distributions import uniform, randint
import sys

import sklearn.model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import cross_val_score

import time
import random

import xgboost as xgboost
from xgboost import XGBClassifier
import logging

RUN = 'co_wo_re' 

if __name__ == '__main__':

    try:
            
        random.seed(115)

        logging.basicConfig(filename=f'logs/{RUN}.log', encoding='utf-8', format='{asctime} - {levelname} - {message}', level=logging.INFO, style='{', datefmt='%Y-%m-%d %H:%M')
        
        logging.info('Starting Program')

        pathToData = './data/'
        filename1 = 'bullying_light_clean_2.csv'
        filename2 = 'hatespeech_light_clean_2.csv'

        df= pd.read_csv(pathToData + filename1, encoding='utf-8')
        df2 = pd.read_csv(pathToData + filename2, encoding='utf-8')

        df.dropna(axis=0, inplace=True)
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)

        df2.dropna(axis=0, inplace=True)
        df2.drop_duplicates(inplace=True)
        df2.reset_index(drop=True, inplace=True)

        # split classes
        religion = df[df['cyberbullying_type'] == 'religion']
        age = df[df['cyberbullying_type'] == 'age']
        gender = df[df['cyberbullying_type'] == 'gender']
        ethnicity = df[df['cyberbullying_type'] == 'ethnicity']
        other_cyberbullying = df[df['cyberbullying_type'] == 'other_cyberbullying']
        not_cyberbullying = df[df['cyberbullying_type'] == 'not_cyberbullying']
        
        undersample_value = 5150 # (20731 - 1219 + 6238) / 5  
        ethnicity_us = ethnicity.sample(undersample_value)
        other_cyberbullying_us = other_cyberbullying.sample(undersample_value)
        religion_us = religion.sample(undersample_value)
        age_us = age.sample(undersample_value)
        gender_us = gender.sample(undersample_value)
        
        df = pd.concat([religion_us, age_us, gender_us, ethnicity_us, other_cyberbullying_us, not_cyberbullying], axis=0)
        # convert str labels to integers
        df['cyberbullying_type'] = df['cyberbullying_type'].replace({
                'religion': 0,
                'age': 1,
                'gender': 2,
                'ethnicity': 3,
                'other_cyberbullying': 4,
                'not_cyberbullying': 5
        })
        
        # Relabel Dataset 1 NOTE : you must ajust line 94 as well if you use this line
        # df['cyberbullying_type'] = df['cyberbullying_type'].apply(lambda x: 1 if x == 5 else 0) #
        
        # Relabel Dataset 2 
        df2['class'] = df2['class'].apply(lambda x: x if x == 0 else 1)
        df2 = df2.sample(frac=1, random_state=115) # shuffle the data 
        
        # Redistribute data to other dataset (it is removed from this dataset)
        
        # split by class
        not_cyberbullying = df2[df2['class'] == 1]
        cyberbullying = df2[df2['class'] == 0]
        additional_not_cyberbullying = 19512 # 20731 - 1219
        tmp = not_cyberbullying.iloc[:additional_not_cyberbullying] # 19512
        not_cyberbullying = not_cyberbullying.iloc[additional_not_cyberbullying:] # 1219
        
        # add the additional 'not cyberbullying' data to the training dataset
        tmp.rename(columns={'tweet': 'tweet_text', 'class': 'cyberbullying_type'}, inplace=True)
        tmp['cyberbullying_type'] = 5 
        df = pd.concat([df, tmp])
        df2 = pd.concat([not_cyberbullying, cyberbullying])
        
        # Split training data
        X_train, X_val, Y_train, Y_val = sklearn.model_selection.train_test_split(df.tweet_text, df.cyberbullying_type, random_state=115)
        
        vectorizer = CountVectorizer()
        x_train = vectorizer.fit_transform(X_train)
        X_val = vectorizer.transform(X_val)
        
        # SBERT
        # model = SentenceTransformer('all-MiniLM-L6-v2')
        # X = model.encode(df.tweet_text)
        
        # TF-IDF
        # vectorizer = TfidfVectorizer()
        # X = vectorizer.fit_transform(df.tweet_text)
        
        y_train = Y_train
        Y_val = Y_val
        
        N_FOLDS = 5
        def objective(hyperparameters):
            """Objective function for grid and random search. Returns
            the cross validation score from a set of hyperparameters."""
            
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=115)
            m = XGBClassifier(objective='multi:softmax', num_class=6, device='gpu', n_jobs=-1, verbosity=1, random_state=115, **hyperparameters)
            cv_results = cross_val_score(m, x_train, y_train, cv=skf, error_score='raise') # NOTE : use xgb's internal scoring method
        
            return [cv_results.mean(), cv_results.std(), hyperparameters]
        
        
        def random_search(param_grid, objective, max_evals=10):
            """Random search for hyperparameter optimization"""
            
            # Dataframe for results
            results = pd.DataFrame(columns = ['score (mean)', 'score (std)', 'params'], index = list(range(max_evals)))
            
            # Keep searching until reach max evaluations
            for i in range(max_evals):
                
                # Choose random hyperparameters
                hyperparameters = {}
                for k,v in param_grid.items():
                    v = random.sample(v,1)[0] if isinstance(v, (list, tuple)) else v.rvs(1)[0]
                    if k == 'learning_rate': v = round(v,5)
                    elif not isinstance(v, (int)) and k != 'booster': v = round(v, 3)                
                    hyperparameters[k] = v   
                        
                print(f'Starting iteration: {i}')
                start_time = time.time()
                # Evaluate randomly selected hyperparameters
                logging.info(f'Start Validation: {i}')
                print(hyperparameters)
                eval_results = objective(hyperparameters)
                stop_time = time.time()
                        
                results.loc[i, :] = eval_results
                results.loc[i,:].to_csv(f'results/{RUN}.csv', mode='a', header=False, lineterminator='\n\n', index=False)            
                
                logging.info(f'Elapsed Time: {i} {time.strftime("%H:%M:%S", time.gmtime(stop_time - start_time))}')
            
            # Sort with best score on top
            results.sort_values('score (mean)', ascending=False, inplace=True)
            results.reset_index(inplace = True, drop=True)
            return results 
        
        np.random.seed(115) # this sets the random seed for the distributions below
        
        parameters = {
                'booster'              : ['gbtree','dart'],
                'n_estimators'         : [200, 300, 400, 500, 600, 700, 800],
                'learning_rate'        : uniform(sys.float_info.min, 1),
                'max_depth'            : randint(3, 50),
                'min_child_weight'     : randint(1, 50),
                'subsample'            : uniform(sys.float_info.min, 1),
                'colsample_bytree'     : uniform(0,1),
                'colsample_bylevel'    : uniform(0,1),
                'colsample_bynode'     : uniform(0,1),
                'alpha'                : uniform(0,15),
                'lambda'               : uniform(0,15),
                'gamma'                : uniform(0,15),
        }
        
        result = random_search(parameters, objective, max_evals=1000)
        # Get max and output to a file
        result.iloc[0].to_csv(f'results/{RUN}_best.csv', header=False, index=False) 
    
    except Exception as e:
        logging.error(f'Exception: {e}')            