from pathlib import Path
import pandas as pd
import numpy as np
from nltk.stem.snowball import SnowballStemmer 
import sys
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import json

sys.path.insert(0, str(Path.cwd().parent / 'src'))

from utils.params import get_topn_param_sets

PARAMS_PATH = Path(r'C:\Users\rooty\UWEC\Research\CyberBullyingML\cyberbullyingml\cyberbullying-ml\official\hyp_search\results')
DATA_PATH = Path(r'C:\Users\rooty\UWEC\Research\CyberBullyingML\cyberbullyingml\cyberbullying-ml\data\en_only')
DATASET_1_NAME = '48000_cyberbullying_tweets_basic_clean.csv'
DATASET_2_NAME = 'hatespeech_tweets_basic_clean.csv'
DATASET_3_NAME = 'onlineHarassmentDataset_basic_clean.csv'

RANDOM_SEED = 115

from collections import defaultdict
import array
import scipy.sparse as sp

def _make_int_array():
    """Construct an array.array of a type suitable for scipy.sparse indices."""
    return array.array(str("i"))


class MyCountVectorizer(CountVectorizer):
    def __init__(self):
        super().__init__()
        self.oov_token_count = 0
        self.iv_token_count = 0
        
    def _count_vocab(self, raw_documents, fixed_vocab):
        """Create sparse feature matrix, and vocabulary where fixed_vocab=False"""
        if fixed_vocab:
            vocabulary = self.vocabulary_
        else:
            # Add a new value when a new vocabulary item is seen
            vocabulary = defaultdict()
            vocabulary.default_factory = vocabulary.__len__

        analyze = self.build_analyzer()
        j_indices = []
        indptr = []

        values = _make_int_array()
        indptr.append(0)
        for doc in raw_documents:
            feature_counter = {}
            for feature in analyze(doc):
                try:
                    feature_idx = vocabulary[feature]                    
                    if fixed_vocab == True: self.iv_token_count += 1
                    if feature_idx not in feature_counter:
                        feature_counter[feature_idx] = 1
                    else:
                        feature_counter[feature_idx] += 1
                    
                except KeyError:        
                    if fixed_vocab == True: self.oov_token_count += 1
                    # Ignore out-of-vocabulary items for fixed_vocab=True
                    continue

            j_indices.extend(feature_counter.keys())
            values.extend(feature_counter.values())
            indptr.append(len(j_indices))

        if not fixed_vocab:
            # disable defaultdict behaviour
            vocabulary = dict(vocabulary)
            if not vocabulary:
                raise ValueError(
                    "empty vocabulary; perhaps the documents only contain stop words"
                )

        if indptr[-1] > np.iinfo(np.int32).max:  # = 2**31 - 1
            # if _IS_32BIT:
            #     raise ValueError(
            #         (
            #             "sparse CSR array has {} non-zero "
            #             "elements and requires 64 bit indexing, "
            #             "which is unsupported with 32 bit Python."
            #         ).format(indptr[-1])
            #     )
            indices_dtype = np.int64

        else:
            indices_dtype = np.int32
        j_indices = np.asarray(j_indices, dtype=indices_dtype)
        indptr = np.asarray(indptr, dtype=indices_dtype)
        values = np.frombuffer(values, dtype=np.intc)

        X = sp.csr_matrix(
            (values, j_indices, indptr),
            shape=(len(indptr) - 1, len(vocabulary)),
            dtype=self.dtype,
        )
        X.sort_indices()
        return vocabulary, X

d1 = pd.read_csv(DATA_PATH / DATASET_1_NAME)
d2 = pd.read_csv(DATA_PATH / DATASET_2_NAME)
d3 = pd.read_csv(DATA_PATH / DATASET_3_NAME)

d1.dropna(axis=0, inplace=True)
d1.drop_duplicates(inplace=True)
d1.reset_index(drop=True, inplace=True)

d2.dropna(axis=0, inplace=True)
d2.drop_duplicates(inplace=True)
d2.reset_index(drop=True, inplace=True)

d3.dropna(axis=0, inplace=True)
d3.drop_duplicates(inplace=True)
d3.reset_index(drop=True, inplace=True)

d1['label'] = d1['label'].apply(lambda label: 0 if label =='notcb' else 1)
d2['class'] = d2['class'].apply(lambda label: 1 if label == 0 else 0)
d3['Code'] = d3['Code'].apply(lambda label: 1 if label == 'H' else 0)

d1.rename(columns={'tweet': 'text'}, inplace=True)
d2.rename(columns={'tweet': 'text'}, inplace=True)
d2.rename(columns={'class': 'label'}, inplace=True)
d3.rename(columns={'Code': 'label'}, inplace=True)
d3.rename(columns={'Tweet': 'text'}, inplace=True)

results_file = Path.cwd() / 'oov_results'
results = []

for idx, (tr_ds_name, te_ds_name, tr_df, te_df) in enumerate([('d1', 'd2', d1, d2), ('d1', 'd3', d1, d3), ('d2', 'd1', d2, d1), ('d2', 'd3', d2, d3), ('d3', 'd1', d3, d1), ('d3', 'd2', d3, d2)]):
    print(f'Starting run {idx}')
    for model in ['catb', 'xgb']:
        param_results:pd.DataFrame = get_topn_param_sets(PARAMS_PATH, algo=model, dataset=tr_ds_name, n=10, sort_condition='f1_macro_mean')
        param_sets = param_results['params']

        snowballer = SnowballStemmer('english')     
        def pp_SnowballStemmer(text):
            return ' '.join([snowballer.stem(word) for word in text.split()])
        
        for i, params in enumerate(param_sets):
            x_train, x_val, y_train, y_val = train_test_split(
                tr_df['text'],
                tr_df['label'],
                test_size=0.1,
                shuffle=True,
                stratify=tr_df['label'],
                random_state=RANDOM_SEED
            )
            x_test = te_df['text']
            vectorizer_cls = CountVectorizer if 'Count' in params['vectorizer'] else TfidfVectorizer
            vect_params = {}

            for k in params.keys():
                if 'vectorizer__' in k:
                    p_name = k.split('__')[1]            
                    if p_name == 'ngram_range':
                        vect_params[p_name] = tuple(params[k])  
                    elif p_name == 'preprocessor':
                        vect_params[p_name] = pp_SnowballStemmer if isinstance(params[k], str) else None
                    else:
                        vect_params[p_name] = params[k]

            vect1 = vectorizer_cls(**vect_params)
            vect2 = vectorizer_cls(**vect_params)
            vect1.fit(x_train)
            vect2.fit(x_test)
        
            print("Get OOV ratio")
            feats_train = vect1.get_feature_names_out()
            feats_test = vect2.get_feature_names_out()

            print('Start setdiff')
            num_oov = len(np.setdiff1d(feats_test, feats_train, assume_unique=True))

            oov_to_total = num_oov / len(feats_test)
            results.append(oov_to_total)

with open('oov_ratios.json', 'w') as f:
    json.dump(results, f)