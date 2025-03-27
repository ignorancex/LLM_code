# This experiment aggregates the 3 datasets and runs CV 

# This file uses the XGBoostClassifier with TfidfVectorizer AND StratifiedShuffleSplit 

from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, cross_validate

from nltk.stem.snowball import SnowballStemmer
# from nltk.stem.porter import PorterStemmer

import numpy as np
import pandas as pd

from pathlib import Path
import random
import itertools
import json
import os
import sys
import logging
import traceback
RANDOM_SEED = 115
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

D1_NAME = '48000_cyberbullying_tweets_basic_clean.csv' 
D2_NAME = 'hatespeech_tweets_basic_clean.csv'
D3_NAME = 'onlineHarassmentDataset_basic_clean.csv'

FNAME = os.path.basename(__file__).split('.')[0]

logging.basicConfig(
    filename=f'log/{FNAME}.log',
    filemode='w+',
    encoding='utf-8',
    format='{asctime} - {levelname} - {message}',
    level=logging.DEBUG,
    style='{',
    datefmt='%Y-%m-%d %H:%M'
)

if __name__ == '__main__':
    try:
        logging.info(f'Starting: {FNAME}\nRandom seed: {RANDOM_SEED}')

        data_path = Path(sys.argv[1])

        d1_path = data_path / D1_NAME
        d2_path = data_path / D2_NAME
        d3_path = data_path / D3_NAME

        d1_df = pd.read_csv(d1_path, usecols=['tweet', 'label'])
        d2_df = pd.read_csv(d2_path, usecols=['tweet', 'class'])
        d3_df = pd.read_csv(d3_path, usecols=['Tweet', 'Code'])

        # remove duplicates
        d1_df = d1_df.drop_duplicates()
        d2_df = d2_df.drop_duplicates()
        d3_df = d3_df.drop_duplicates()

        # remove rows with missing values
        d1_df = d1_df.dropna()
        d2_df = d2_df.dropna()
        d3_df = d3_df.dropna()

        # reset indices
        d1_df.reset_index(drop=True, inplace=True)
        d2_df.reset_index(drop=True, inplace=True)
        d3_df.reset_index(drop=True, inplace=True)

        # rename columns
        d1_df.rename(columns={'tweet': 'tweet', 'label': 'label'}, inplace=True)
        d2_df.rename(columns={'tweet': 'tweet', 'class': 'label'}, inplace=True)
        d3_df.rename(columns={'Tweet': 'tweet', 'Code': 'label'}, inplace=True)

        d1_df['label'] = d1_df['label'].apply(lambda label: 0 if label == 'notcb' else 1)
        d2_df['label'] = d2_df['label'].apply(lambda label: 1 if label == 0 else 0)
        d3_df['label'] = d3_df['label'].apply(lambda label: 1 if label == 'H' else 0)
        # concatenate the datasets
        train_df = pd.concat([d1_df, d2_df, d3_df])

        # split data into train and validation sets
        x_train, x_val, y_train, y_val = train_test_split(
            train_df['tweet'],
            train_df['label'],
            test_size=0.1,
            shuffle=True,
            stratify=train_df['label'],
            random_state=RANDOM_SEED
        )

        snowballer = SnowballStemmer('english')

        def pp_SnowballStemmer(text):
            return ' '.join([snowballer.stem(word) for word in text.split()])

        xgb_param_grid = {
            'vectorizer': [TfidfVectorizer],
            'vectorizer__ngram_range': [(1,1), (1,3)],
            'vectorizer__preprocessor': [pp_SnowballStemmer, None],
            'vectorizer__max_df': [0.5, 0.9],
            'classifier__learning_rate': [0.001, 0.1],
            'classifier__n_estimators': [500],
            'classifier__colsample_bytree': [0.5, 0.9],
            'classifier__max_depth': [6, 12],
            'classifier': [XGBClassifier]
        }

        iters = (v for _, v in xgb_param_grid.items())
        keys = xgb_param_grid.keys()
        param_sets = [params for params in itertools.product(*iters)]

        # calculate the scale_pos_weight
        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        logging.info(f'The positive weight is {pos_weight:.4f}\n')

        all_results = {
            'fit_time_mean': [],
            'fit_time_std_dev': [],
            'f1_macro_mean': [],
            'f1_macro_std_dev': [],
            'f1_weighted_mean': [],
            'f1_weighted_std_dev': [],
            'params': []
        }

        N_SPLITS = 10

        logging.info(f'Starting CV for {len(param_sets)} folds (i.e. {len(param_sets) * N_SPLITS} splits)')
        for idx, params in enumerate(param_sets):
            logging.info(f'Starting run {idx}...\n')

            t = {k: v for k, v in zip(keys, params)}

            vect_params = {}
            classifier_params = {}

            for k in keys:
                if 'vectorizer__' in k:
                    p_name = k.split('__')[1]
                    vect_params[p_name] = t[k]
                if 'classifier__' in k:
                    p_name = k.split('__')[1]
                    classifier_params[p_name] = t[k]
            
            vect = t['vectorizer'](**vect_params)
            classifier:XGBClassifier = t['classifier'](early_stopping_rounds=10, eval_metric='logloss', scale_pos_weight=pos_weight, **classifier_params)

            x_train_copy = vect.fit_transform(x_train.copy())
            x_val_copy = vect.transform(x_val.copy())
            y_train_copy = y_train.copy()
            y_val_copy = y_val.copy()

            skf = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=0.2, random_state=RANDOM_SEED)

            cv_results: dict = cross_validate(
                classifier,
                x_train_copy,
                y_train_copy,
                cv=skf,
                scoring=['f1_macro', 'f1_weighted'],
                verbose=10,
                return_estimator=False,
                params={'eval_set':[(x_val_copy, y_val_copy)]},
                n_jobs=-1
            )

            logging.info(f'The cv results for run {idx} are:\n{pd.DataFrame(cv_results)}\n')

            cv_results['params'] = t
            cv_results['fit_time_mean'] = np.mean(cv_results['fit_time'])
            cv_results['fit_time_std_dev'] = np.std(cv_results.pop('fit_time'))
            cv_results['f1_macro_mean'] = np.mean(cv_results['test_f1_macro'])
            cv_results['f1_macro_std_dev'] = np.std(cv_results.pop('test_f1_macro'))
            cv_results['f1_weighted_mean'] = np.mean(cv_results['test_f1_weighted'])
            cv_results['f1_weighted_std_dev'] = np.std(cv_results.pop('test_f1_weighted'))

            cv_results.pop('score_time')

            for key, value in cv_results.items():
                all_results[key].append(value)

            with open(f'results/{FNAME}__results.json', 'w+') as f:
                json.dump(all_results, f, default=str)

        logging.info(f'Finished running folds. See the results in {FNAME}__results.json')
    except Exception as e:
        logging.error(f'Error: {e}\nStack Trace: {traceback.format_exc()}')
    finally:
        logging.info(f'Terminating Program: {FNAME}')
        sys.exit()
