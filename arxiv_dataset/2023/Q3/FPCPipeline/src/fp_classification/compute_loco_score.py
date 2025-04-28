"""

Compute LOCO for for radiomics features for the false positive classification task

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""

import os
import pandas as pd
from argparse import ArgumentParser
import joblib
from helper_functions import *
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.inspection import permutation_importance
from loco import compute_loco_score
import matplotlib.pyplot as plt
import matplotlib as mpl
# Set style and font
plt.style.use('seaborn-paper')
mpl.rc("font", family="Serif")

if __name__  == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--oversample', action='store_true' )
    parser.add_argument('--save_dir', type=str, default=None)
    args = parser.parse_args()

    # Get the training data

    if args.save_dir is None:
        save_dir = args.out_dir
    else:
        save_dir = args.save_dir

    train_df = pd.read_pickle(os.path.join(args.out_dir, 'train_dataset.pkl'))

    selected_features = joblib.load(os.path.join(args.out_dir, 'selected_features.pkl'))

    features_matrix, labels, feature_names = create_features_matrix(train_df,
                                                                    selected_features=selected_features)

    print('Feature matrix shape : {}'.format(features_matrix.shape))

    patient_ids = train_df['Patient ID'].to_numpy()

    seed_dirs = [f.path for f in os.scandir(args.out_dir) if (f.is_dir() and ('seed' in f.name))]


    loco_mean_over_seeds = np.zeros(shape=(len(seed_dirs), len(selected_features)), dtype=np.float32)
    loco_std_over_seeds = np.zeros(shape=(len(seed_dirs), len(selected_features)), dtype=np.float32)

    for idx, seed_dir in enumerate(seed_dirs):
        print('LOCO for {}'.format(seed_dir.split('/')[-1]))

        # Get the fitted model
        model_path = get_model_path(seed_dir,
                                    reduce_features=True,
                                    oversample=args.oversample)

        clf = joblib.load(model_path)

        # This ensures the lesions from the same patient don't appear in the training and held-out set at the same time
        data_splitter = GroupKFold(n_splits=5)

        results = compute_loco_score(estimator=clf,
                                     X=features_matrix,
                                     y=labels,
                                     cv=data_splitter,
                                     groups=patient_ids)

        loco_mean_over_seeds[idx, :] = results['mean']
        loco_std_over_seeds[idx, :] = results['std']

    med_loco = np.median(loco_mean_over_seeds, axis=0)

    # Order the indices in descending order of (median) permutation importance
    ordered_indices = med_loco.argsort()[::-1]

    ordered_loco_means = loco_mean_over_seeds[:, ordered_indices]
    ordered_feature_names = np.take_along_axis(feature_names, ordered_indices, axis=0)

    np.save(os.path.join(args.out_dir, 'ordered_loco_means.npy'), ordered_loco_means)
    np.save(os.path.join(args.out_dir, 'ordered_feature_names.npy'), ordered_feature_names)

    # Plot a bar-graph
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#
#    bp = ax.boxplot(x=ordered_loco_means,
#                    labels=ordered_feature_names,
#                    vert=False)
#
#    plt.xlabel('LOCO score')
#    plt.xlim((-0.05, 0.08))
#
#    plot_fname = 'loco'
#
#
#
#    if args.oversample is True:
#        plot_fname = plot_fname + '_oversample'
#
#
#    plt.tight_layout()
#    if args.save_dir is None:
#        plt.savefig(os.path.join(save_dir,'{}.pdf'.format(plot_fname)))
#    else:
#        dataset = args.out_dir.split('/')[3]
#        config = '{}_{}'.format(args.out_dir.split('/')[4], args.out_dir.split('/')[-1])
#        save_dir = os.path.join(save_dir, '{}'.format(dataset))
#        if os.path.exists(save_dir) is False:
#            os.makedirs(save_dir)
#        plt.savefig(os.path.join(save_dir,'{}_{}.pdf'.format(plot_fname, config)))
#
#    plt.close()
#
#
#



