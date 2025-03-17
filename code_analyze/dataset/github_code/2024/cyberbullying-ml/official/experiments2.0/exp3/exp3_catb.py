# Train on D2
# Test on D1

from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from nltk.stem.snowball import SnowballStemmer

import pandas as pd
import numpy as np

import random
import sys
import os
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import params, results, data

TRAIN_DATA_NAME = 'hatespeech_tweets_basic_clean.csv'
TEST_DATA_NAME = '48000_cyberbullying_tweets_basic_clean.csv'

RANDOM_SEED = 115
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

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
        DATA_PATH = Path(sys.argv[1]).resolve()
        PARAMS_PATH = Path(sys.argv[2]).resolve()
        RESULT_PATH = Path('./results')

        train_df = pd.read_csv(DATA_PATH / TRAIN_DATA_NAME)
        test_df = pd.read_csv(DATA_PATH / TEST_DATA_NAME)

        train_df.dropna(axis=0, inplace=True)
        train_df.drop_duplicates(inplace=True)
        train_df.reset_index(drop=True, inplace=True)

        test_df.dropna(axis=0, inplace=True)
        test_df.drop_duplicates(inplace=True)
        test_df.reset_index(drop=True, inplace=True)

        train_df['class'] = train_df['class'].apply(lambda label: 1 if label == 0 else 0)
        test_df['label'] = test_df['label'].apply(lambda label: 0 if label == 'notcb' else 1)
        
        x_train, x_val, y_train, y_val = train_test_split(
            train_df['tweet'], 
            train_df['class'], 
            test_size=0.1, 
            stratify=train_df['class'],
            random_state=RANDOM_SEED
        ) # for early stopping

        x_test = test_df['tweet']
        y_test = test_df['label']

        catb_param_results:pd.DataFrame = params.get_topn_param_sets(PARAMS_PATH, algo='catb', dataset='d2', n=10, sort_condition='f1_macro_mean')

        catb_param_sets = catb_param_results['params']

        neg_to_pos_ratio = (y_train == 0).sum() / (y_train == 1).sum()

        snowballer = SnowballStemmer('english')

        def pp_SnowballStemmer(text):
            return ' '.join([snowballer.stem(word) for word in text.split()])

        results_file = RESULT_PATH/f'{FNAME}_results.json'
        results.create_results_file(results_file)

        for i, params in enumerate(catb_param_sets):
            logging.info(f'Starting run {i+1}...')

            vectorizer_cls = CountVectorizer if params['vectorizer'] == "<class 'sklearn.feature_extraction.text.CountVectorizer'>" else TfidfVectorizer
            classifier_cls = CatBoostClassifier

            vect_params = {}
            classifier_params = {}

            for k in params.keys():
                if 'vectorizer__' in k:
                    p_name = k.split('__')[1]            
                    if p_name == 'ngram_range':
                        vect_params[p_name] = tuple(params[k])  
                    elif p_name == 'preprocessor':
                        vect_params[p_name] = pp_SnowballStemmer if isinstance(params[k], str) else None
                    else:
                        vect_params[p_name] = params[k]

                if 'classifier__' in k:
                    p_name = k.split('__')[1]
                    classifier_params[p_name] = params[k]

            vectorizer = vectorizer_cls(**vect_params)
            classifier = classifier_cls(**classifier_params, scale_pos_weight=neg_to_pos_ratio, early_stopping_rounds=10, eval_metric='Logloss')

            x_train_transformed = vectorizer.fit_transform(x_train)
            x_val_transformed = vectorizer.transform(x_val)
            x_test_transformed = vectorizer.transform(x_test)

            classifier.fit(x_train_transformed, y_train, eval_set=[(x_val_transformed, y_val)])
            preds = classifier.predict(x_test_transformed)

            val_scores = catb_param_results.iloc[i]

            result = {}
            result['report'] = (report:=classification_report(y_test, preds, output_dict=True))
            classifier_params['n_estimators'] = classifier.tree_count_
            result['classifier_params'] = classifier_params
            result['vectorizer'] = vectorizer.__str__()
            result['vectorizer_params'] = vect_params
            result['num_oov_feats'] = len(data.get_OOV_feats(x_train, x_test, print_oov_feats=False, **vect_params))
            result['val_f1_macro_mean'] = val_f1_macro_mean = val_scores['f1_macro_mean']
            result['val_f1_weighted_mean'] = val_f1_weighted_mean = val_scores['f1_weighted_mean']
            result['drop_in_f1_macro_mean'] = val_f1_macro_mean - report['macro avg']['f1-score']
            result['drop_in_f1_weighed_mean'] = val_f1_weighted_mean - report['weighted avg']['f1-score']

            results.append_results(result, results_file)

    except Exception as e:
        logging.error(f'Exception: {e}')
        exit()