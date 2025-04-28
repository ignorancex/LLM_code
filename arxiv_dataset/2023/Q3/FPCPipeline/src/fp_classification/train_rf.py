"""
Script that contains the top-level pipeline to train a classifier to
detect false-positives

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from math import isnan
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from classify import *
import joblib

def calculate_mean_std(metrics):
    if isinstance(metrics, list) is True:
        metrics = np.array(metrics)

    mean = np.mean(metrics)
    std = np.std(metrics)

    return mean, std


def main(args):

    out_dir = args.out_dir

    if os.path.exists(out_dir) is False:
        print('Either incorrect seed supplied or our_dir does not exist')
        return
    else:
        if os.path.exists(os.path.join(out_dir, 'train_dataset.pkl')) is True:
            features_dataset = pd.read_pickle(os.path.join(out_dir, 'train_dataset.pkl'))
        else:
            print('Train features dataset not found')
            return

    # Get list of patient IDs
    patient_ids = features_dataset['Patient ID'].to_numpy()

    # Check balance of dataset
    labels = features_dataset['label'].to_numpy()
    num_tp = np.where(labels==0)[0].shape[0]
    num_fp = np.where(labels==1)[0].shape[0]
    tp_idxs = np.where(labels==0)[0]
    fp_idxs = np.where(labels==1)[0]

    imbalance = min(num_tp/num_fp, num_fp/num_tp)
    print('TPs = {} FPs = {}, Imbalance = {}'.format(num_tp, num_fp, imbalance))


    assert(patient_ids.shape[0] == features_dataset.shape[0])

    save_dir = os.path.join(out_dir, 'seed_{}'.format(args.seed))
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    if args.reduce_features is True:
        selected_features = joblib.load(os.path.join(args.out_dir, 'selected_features.pkl'))
    else:
        selected_features = None

    print('Start cross validation')

    erf_pipeline, param_grid = create_erf_pipeline(seed=args.seed,
                                                   oversample=args.oversample,
                                                   n_jobs=args.n_jobs)


    features, labels, _ = create_features_matrix(features_dataset,
                                                 selected_features=selected_features)

    print('DF shape = {}'.format(features.shape))

    clf = grid_search_for_params(data=features,
                                 labels=labels,
                                 pipeline=erf_pipeline,
                                 param_grid=param_grid,
                                 groups=patient_ids,
                                 n_jobs=args.n_jobs)


    model_path = get_model_path(save_dir,
                                reduce_features=args.reduce_features,
                                oversample=args.oversample)

    joblib.dump(clf, model_path)




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--out_dir', type=str, help='Directory where images are stored', default='./results/baseline')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--reduce_features', action='store_true')
    parser.add_argument('--oversample', action='store_true')
    parser.add_argument('--n_jobs', type=int, help='Number of CPU cores to use', default=-1)
    args = parser.parse_args()
    main(args)


