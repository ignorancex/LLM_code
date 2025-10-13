"""
Example usage:
python phone_PCA.py --embeddings_file="embeddings/amsterdamNLP_Wav2Vec2-NL_phone_embs.pkl" --model_name="w2v2-nl" --subset="MLS"
"""

import pickle
import pandas as pd
from tqdm import tqdm
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
from argparse import ArgumentParser
from pathlib import Path

from model_utils import model_depth_map

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

    N_layers = len(embeddings.keys())

    pca_item_scores = []
    pca_projections = {
        layer: None
        for layer in embeddings.keys()
    }

    print(f'Computing phone PCA clustering scores for {args.model_name} on {args.subset} data...')

    for layer in tqdm(embeddings.keys(), desc='\tclustering & scoring by layer'):
        embs = zscore(embeddings[layer])
        
        if args.subset != 'all':
            train_idx = (annotations['subset'] == args.subset) & (annotations['split'] == 'train')
            test_idx = (annotations['subset'] == args.subset) & (annotations['split'] == 'test')
        else:
            train_idx = annotations['split'] == 'train'
            test_idx = annotations['split'] == 'test'
            
        X_train = embs[train_idx]
        X_test = embs[test_idx]
        y_train = annotations['phone'][train_idx].values
        y_test = annotations['phone'][test_idx].values

        n_comps = len(annotations['phone'].unique())-1
        pca = PCA(n_components=n_comps)
        pca.fit(X_train, y_train)
        
        pca_projections[layer] = pca.transform(embs[test_idx])
        
        silh_sample_scores = silhouette_samples(
            pca_projections[layer],
            y_test,
            metric='cosine'
        )
        
        for i in range(len(silh_sample_scores)):
            pca_item_scores.append((args.subset, args.model_name, layer, model_depth_map[layer], 
                                    y_test[i], 'silhouette_score', silh_sample_scores[i]))

    pca_score_df = pd.DataFrame(pca_item_scores, 
                                columns=['eval_set', 'model_name', 'layer', 'model_depth', 
                                         'target', 'score_name', 'score'])
    
    pca_score_df.to_csv(results_dir / f'phone_PCA_item-scores_{args.subset}_{args.model_name}.csv', index=False)

    print(f'Done! Saved results to {args.results_dir}')