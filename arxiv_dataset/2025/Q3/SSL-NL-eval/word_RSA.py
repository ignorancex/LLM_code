"""
Example usage:
python word_RSA.py --embeddings_file="embeddings/amsterdamNLP_Wav2Vec2-NL_word-rsa_embs.pkl" --model_name="w2v2-nl" --subset="MLS"
"""

import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import zscore
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from argparse import ArgumentParser
from pathlib import Path

from model_utils import model_depth_map

def compute_rsa_score(X, Y, dist='cosine', sim='pearson', ci_level=0.95):
    X_pdists = pdist(zscore(X), metric=dist)
    Y_pdists = pdist(zscore(Y), metric=dist)
    
    X_distmat = squareform(X_pdists)
    Y_distmat = squareform(Y_pdists)
    
    mask = ~np.tri(X_distmat.shape[0], X_distmat.shape[1], k=3).astype(bool)
    
    X_triu = np.triu(X_distmat)[mask]
    Y_triu = np.triu(Y_distmat)[mask]
    
    if sim == 'pearson':
        rsa_sim = pearsonr(X_triu, Y_triu)
        rsa_score, pval = rsa_sim
        ci_low, ci_high = rsa_sim.confidence_interval(ci_level)
    return rsa_score, pval, ci_low, ci_high

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--embeddings_file",
        required=True,
        type=str,
        help='filepath to the pkl file with extracted word-rsa embeddings',
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help='name of model that generated the embeddings (for file saving)'
    )
    parser.add_argument(
        "--annotations_file",
        type=str,
        default='SSL-NL/annotations/word-rsa_annotations.csv',
        help='filepath to the word-rsa annotations file'
    )
    parser.add_argument(
        "--fasttext_embs",
        type=str,
        default='embeddings/fasttext_word-rsa_embs.pkl',
        help='filepath to the pkl file with fasttext embeddings'
    )
    parser.add_argument(
        "--subset",
        type=str,
        default='all',
        help='subset to evaluate embeddings on (MLS or IFADV)'
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default='results',
        help='directory to store results to'
    )
    args, unk_args = parser.parse_known_args()

    embeddings = pickle.load(open(args.embeddings_file, 'rb'))
    annotations = pd.read_csv(args.annotations_file)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)

    fasttext_embs = pickle.load(open(args.fasttext_embs, 'rb'))

    N_layers = len(embeddings.keys())

    rsa_scores = []

    print(f'Computing word RSA scores for {args.model_name} on {args.subset} data...')
    
    for layer in tqdm(embeddings.keys(), desc='\tscoring by layer'):
        embs = zscore(embeddings[layer])
        subset_idx = annotations[annotations['subset'] == args.subset].index
        subset_embs = embs[subset_idx]
        rsa_score, pval, ci_low, ci_high = compute_rsa_score(subset_embs, fasttext_embs[args.subset])
        rsa_scores.append((args.subset, args.model_name, layer, model_depth_map[layer], 'pearson_r', rsa_score, pval, ci_low, ci_high))
        
    rsa_score_df = pd.DataFrame(rsa_scores, columns=['eval_set', 'model_name', 'layer', 'model_depth', 'score_name', 'score', 'pval', 'ci_low', 'ci_high'])
    rsa_score_df.to_csv(results_dir / f'word_RSA_layer-scores_{args.subset}_{args.model_name}.csv', index=False)

    print(f'Done! Saved results to {args.results_dir}')