"""
Example usage:
python phone_probing.py --embeddings_file="embeddings/amsterdamNLP_Wav2Vec2-NL_phone_embs.pkl" --model_name="w2v2-nl" --subset="MLS"
"""

import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import binomtest, zscore
from sklearn.linear_model import LogisticRegression
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

    probe_predictions = []
    probe_scores = []

    print(f'Computing phone identity probe scores for {args.model_name} on {args.subset} data...')

    for layer in tqdm(embeddings.keys(), desc='\tprobing by layer'):
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

        logreg_probe = LogisticRegression(max_iter=5000)
        logreg_probe.fit(X_train, y_train)
        predictions = logreg_probe.predict(X_test)
        accuracies = predictions == y_test
        mean_acc = np.mean(accuracies)
        
        binom_result = binomtest(
            k=np.where(accuracies)[0].shape[0], # successes
            n=predictions.shape[0],             # trials
            p=1/logreg_probe.classes_.shape[0]  # chance probability
        )
        ci_low, ci_high = tuple(binom_result.proportion_ci(
            confidence_level=0.95,
            method='wilsoncc'
        ))

        for i in range(len(predictions)):
            probe_predictions.append((args.subset, args.model_name, layer, model_depth_map[layer], y_test[i], predictions[i], accuracies[i]))
        probe_scores.append((args.subset, args.model_name, layer, model_depth_map[layer], 'mean_acc', mean_acc, binom_result.pvalue, ci_low, ci_high))

    item_acc_df = pd.DataFrame(probe_predictions, 
        columns = ['eval_set', 'model_name', 'layer', 'model_depth', 'target', 'prediction', 'accuracy'])
    layer_score_df = pd.DataFrame(probe_scores, 
        columns = ['eval_set', 'model_name', 'layer', 'model_depth', 'score_name', 'score', 
                   'binom_pval', 'ci_low', 'ci_high'])

    item_acc_df.to_csv(results_dir / f'phone_identity-probe_item-scores_{args.subset}_{args.model_name}.csv', index=False)
    layer_score_df.to_csv(results_dir / f'phone_identity-probe_layer-scores_{args.subset}_{args.model_name}.csv', index=False)

    print(f'Done! Saved results to {args.results_dir}')