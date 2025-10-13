"""
Example usage:
python phone_ABX.py --embeddings_file="embeddings/amsterdamNLP_Wav2Vec2-NL_phone_embs.pkl" --model_name="w2v2-nl" --subset="MLS"
"""

import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import binomtest, zscore
from scipy.spatial.distance import cdist
from argparse import ArgumentParser
from pathlib import Path

from model_utils import model_depth_map

def get_ABX_scores(A_embs, B_embs, X_embs):
    AX_sim = np.repeat(
    (1 - cdist(A_embs, X_embs, metric='cosine')).flatten(), 
    len(B_embs)
    )
    BX_sim = np.tile(
    (1 - cdist(X_embs, B_embs, metric='cosine')).flatten(), 
    len(A_embs)
    )
    accuracies = AX_sim > BX_sim
                    
    return accuracies

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--embeddings_file",
        required=True,
        type=str,
        help='filepath to the pkl file with extracted phone embeddings',
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
        default='SSL-NL/annotations/phone_annotations.csv',
        help='filepath to the phone annotations file'
    )
    parser.add_argument(
        "--phone_contrasts_file",
        type=str,
        default="phone_contrasts.csv",
        help='filepath to a csv file defining phone contrasts to compute ABX scores over'
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="across-speaker",
        help="whether to evaluate ABX accuracy within or across speakers (one of ['within-speaker', 'across-speaker'])"
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
    phone_contrasts = pd.read_csv(args.phone_contrasts_file)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)

    ABX_contrast_scores = []
    ABX_layer_scores = []

    print(f'Computing {args.condition} ABX scores for {args.model_name} on {args.subset} data...')

    N_layers = len(embeddings.keys())

    for layer in tqdm(embeddings.keys(), desc='\tABX-scoring by layer'):
        embs = zscore(embeddings[layer])
        layer_accuracies = []

        for r, contrast_row in phone_contrasts.iterrows():
            phA = contrast_row['A']
            phB = contrast_row['B']
            contrast_set = annotations[
                (annotations['phone'].isin([phA, phB])) 
                & (annotations['subset'] == args.subset)
            ]
            contrast_accuracies = []

            if args.condition == 'within-speaker':
                for speaker in contrast_set['speaker_id'].unique():

                    speaker_contrast_set = contrast_set[contrast_set['speaker_id'] == speaker]

                    split_idx = len(speaker_contrast_set)//4

                    ## sim(phA, phA) > sim(phA, phB)
                    A1_idx = speaker_contrast_set[speaker_contrast_set['phone'] == phA].index.values[:split_idx]
                    X1_idx = speaker_contrast_set[speaker_contrast_set['phone'] == phA].index.values[split_idx:]
                    B1_idx = speaker_contrast_set[speaker_contrast_set['phone'] == phB].index.values[:split_idx]
                    A1_embs = embs[A1_idx]
                    X1_embs = embs[X1_idx]
                    B1_embs = embs[B1_idx]
                    accuracies1 = get_ABX_scores(A1_embs, B1_embs, X1_embs)

                    ## sim(phB, phB) > sim(phA, phB)
                    A2_idx = speaker_contrast_set[speaker_contrast_set['phone'] == phB].index.values[:split_idx]
                    X2_idx = speaker_contrast_set[speaker_contrast_set['phone'] == phB].index.values[split_idx:]
                    B2_idx = speaker_contrast_set[speaker_contrast_set['phone'] == phA].index.values[:split_idx]
                    A2_embs = embs[A2_idx]
                    X2_embs = embs[X2_idx]
                    B2_embs = embs[B2_idx]
                    accuracies2 = get_ABX_scores(A2_embs, B2_embs, X2_embs)
                    
                    contrast_accuracies.extend(np.concatenate([accuracies1, accuracies2]))

            elif args.condition == 'across-speaker':
                for speaker in contrast_set['speaker_id'].unique():
                    speaker_contrast_set = contrast_set[contrast_set['speaker_id'] == speaker]
                    other_speakers = [sp for sp in contrast_set['speaker_id'].unique() if not sp == speaker]

                    split_idx = len(speaker_contrast_set)//4

                    ## sim(phA, phA) > sim(phA, phB)
                    A1_idx = speaker_contrast_set[speaker_contrast_set['phone'] == phA].index.values[:split_idx]
                    X1_idx = np.empty(A1_idx.shape).astype(int)
                    B1_idx = np.empty(A1_idx.shape).astype(int)
                    for i in range(len(A1_idx)):
                        spX, spB = np.random.choice(other_speakers, size=2, replace=False)
                        X1_idx[i] = contrast_set[(contrast_set['speaker_id'] == spX) & (contrast_set['phone'] == phA)].sample(1).index.item()
                        B1_idx[i] = contrast_set[(contrast_set['speaker_id'] == spB) & (contrast_set['phone'] == phB)].sample(1).index.item()
                    A1_embs = embs[A1_idx]
                    X1_embs = embs[X1_idx]
                    B1_embs = embs[B1_idx]
                    accuracies1 = get_ABX_scores(A1_embs, B1_embs, X1_embs)

                    ## sim(phB, phB) > sim(phA, phB)
                    A2_idx = speaker_contrast_set[speaker_contrast_set['phone'] == phB].index.values[:split_idx]
                    X2_idx = np.empty(A2_idx.shape).astype(int)
                    B2_idx = np.empty(A2_idx.shape).astype(int)
                    for i in range(len(A2_idx)):
                        spX, spB = np.random.choice(other_speakers, size=2, replace=False)
                        X2_idx[i] = contrast_set[(contrast_set['speaker_id'] == spX) & (contrast_set['phone'] == phB)].sample(1).index.item()
                        B2_idx[i] = contrast_set[(contrast_set['speaker_id'] == spB) & (contrast_set['phone'] == phA)].sample(1).index.item()
                    A2_embs = embs[A2_idx]
                    X2_embs = embs[X2_idx]
                    B2_embs = embs[B2_idx]
                    accuracies2 = get_ABX_scores(A2_embs, B2_embs, X2_embs)
                    
                    contrast_accuracies.extend(np.concatenate([accuracies1, accuracies2]))
                    
            layer_accuracies.extend(contrast_accuracies)

        mean_layer_acc = np.mean(layer_accuracies)
        N_successes = np.sum(layer_accuracies)

        binom_result = binomtest(
                    k=N_successes,            # successes
                    n=len(layer_accuracies),  # trials
                    p=0.5                     # chance probability
        )

        ci_low, ci_high = tuple(binom_result.proportion_ci(
            confidence_level=0.95,
            method='wilsoncc'
        ))

        ABX_layer_scores.append((args.subset, args.model_name, layer, model_depth_map[layer], 
                                'mean_acc', mean_layer_acc, binom_result.pvalue, ci_low, ci_high))
        
    layer_score_df = pd.DataFrame(ABX_layer_scores, 
                        columns = ['eval_set', 'model_name', 'layer', 'model_depth', 'score_name', 
                                    'score', 'binom_pval', 'ci_low', 'ci_high'])

    layer_score_df.to_csv(results_dir / f'phone_ABX_layer-scores_{args.condition}_{args.subset}_{args.model_name}.csv', index=False)

    print(f'Done! Saved results to {args.results_dir}')