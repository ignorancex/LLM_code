from argparse import ArgumentParser
from helper_functions import *
import matplotlib.pyplot as plt
from math import isnan
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score
from imblearn.metrics import specificity_score, sensitivity_score
from sklearn.svm import SVC
from classify import *
import joblib

def save_metrics(pred_labels, pred_scores, test_labels, res_dir=None, dataset=None):

    metrics_dict = {}
    metrics_dict['Accuracy'] = [accuracy_score(test_labels, pred_labels)]
    metrics_dict['Recall'] = [recall_score(test_labels, pred_labels)]
    metrics_dict['Precision'] = [precision_score(test_labels, pred_labels)]
    metrics_dict['F1 Score'] = [f1_score(test_labels, pred_labels)]
    metrics_dict['Specificity'] = [specificity_score(test_labels, pred_labels)]
    metrics_dict['Sensitivity'] = [sensitivity_score(test_labels, pred_labels)]
    metrics_dict['AUC'] = [roc_auc_score(test_labels, pred_scores)]

    print('AUC  = {}'.format(metrics_dict['AUC'][0]))

    metrics_df = pd.DataFrame.from_dict(data=metrics_dict)

    if dataset == None:
        if args.cross is False:
            metrics_df.to_csv(os.path.join(res_dir, 'test_results_reduced_features.csv'))
        else:
            metrics_df.to_csv(os.path.join(res_dir, 'test_results_reduced_features_cross.csv'))
    else:
        metrics_df.to_csv(os.path.join(res_dir, 'test_results_reduced_features_{}.csv'.format(dataset)))


    print(classification_report(test_labels, pred_labels, target_names=['TP Lesions', 'FP Lesions']))

    print(metrics_df)

def main(args):

    print('Eval on test-set')
    test_features_dataset = pd.read_pickle(os.path.join(args.out_dir, 'test_dataset.pkl'))

    combined_mode = False

    if args.cross is False:
        out_dir = args.out_dir
        dir_tokens = out_dir.split('/')
        if dir_tokens[-1] == '':
            dataset_idx = -5
        else:
            dataset_idx = -4
        dataset = dir_tokens[dataset_idx]
        if dataset != 'umc' and dataset != 'lits': # Combined mode
            combined_mode = True

    else: # If the out_dir (cmd line) points to UMC (LiTS), then pick the trained model and selected features from LiTS (UMC)
        # Get dataset
        out_dir_elems = args.out_dir.split('/')
        if out_dir_elems[-1] == '':
            curr_dataset = out_dir_elems[-5]
            unc_type = out_dir_elems[-2]
            est_method = out_dir_elems[-4]
        else:
            curr_dataset = out_dir_elems[-4]
            unc_type = out_dir_elems[-1]
            est_method = out_dir_elems[-3]

        print('Current dataset : {}'.format(curr_dataset))
        if curr_dataset == 'umc':
            new_dataset = 'lits'
        elif curr_dataset == 'lits':
            new_dataset = 'umc'
        else:
            raise RuntimeError('{} is an invalid dataset'.format(curr_dataset))

        # Form the directory to find the output directory
        out_dir = '/data/ishaan/{}/{}/fp_results/{}'.format(new_dataset, est_method, unc_type)
        print('Model directory: {}'.format(out_dir))


    model_dir = os.path.join(out_dir, 'seed_{}'.format(args.seed))


    model_path = get_model_path(model_dir,
                                reduce_features=args.reduce_features,
                                oversample=args.oversample)

    clf = joblib.load(model_path)

    res_dir = os.path.join(args.out_dir, 'seed_{}'.format(args.seed))

    if args.reduce_features is True:
        selected_features = joblib.load(os.path.join(out_dir, 'selected_features.pkl'))
    else:
        selected_features = None


    test_features, test_labels, _ = create_features_matrix(test_features_dataset,
                                                           selected_features=selected_features)

    print('Features matrix shape = {}'.format(test_features.shape))

    pred_labels = clf.predict(test_features)
    pred_scores = clf.predict_proba(test_features)[:, 1]

    if combined_mode is True:
        # Separate the predicted and true labels
        dataset_col = test_features_dataset['Dataset'].to_numpy()

        umc_pred_labels = pred_labels[dataset_col == 'umc']
        lits_pred_labels = pred_labels[dataset_col == 'lits']

        umc_pred_scores = pred_scores[dataset_col == 'umc']
        lits_pred_scores = pred_scores[dataset_col == 'lits']

        umc_true_labels = test_labels[dataset_col == 'umc']
        lits_true_labels = test_labels[dataset_col == 'lits']

        save_metrics(pred_labels=umc_pred_labels,
                     pred_scores=umc_pred_scores,
                     test_labels=umc_true_labels,
                     res_dir=res_dir,
                     dataset = 'umc')

        save_metrics(pred_labels=lits_pred_labels,
                     pred_scores=lits_pred_scores,
                     test_labels=lits_true_labels,
                     res_dir=res_dir,
                     dataset = 'lits')

        assert(isinstance(umc_pred_labels, np.ndarray))
        np.save(os.path.join(res_dir, 'umc_pred_labels.npy'), umc_pred_labels)

        assert(isinstance(lits_pred_labels, np.ndarray))
        np.save(os.path.join(res_dir, 'lits_pred_labels.npy'), lits_pred_labels)

        assert(isinstance(umc_pred_scores, np.ndarray))
        np.save(os.path.join(res_dir, 'umc_pred_scores.npy'), umc_pred_scores)

        assert(isinstance(lits_pred_scores, np.ndarray))
        np.save(os.path.join(res_dir, 'lits_pred_scores.npy'), lits_pred_scores)

    else:
        save_metrics(pred_labels=pred_labels,
                     pred_scores=pred_scores,
                     test_labels=test_labels,
                     res_dir=res_dir)

        assert(isinstance(pred_labels, np.ndarray))
        np.save(os.path.join(res_dir, 'pred_labels.npy'), pred_labels)

        assert(isinstance(pred_scores, np.ndarray))
        np.save(os.path.join(res_dir, 'pred_scores.npy'), pred_scores)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--out_dir', type=str, help='Directory where images are stored', default='./results/baseline')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--reduce_features', action='store_true')
    parser.add_argument('--oversample', action='store_true')
    parser.add_argument('--cross', action='store_true')
    args = parser.parse_args()
    main(args)
