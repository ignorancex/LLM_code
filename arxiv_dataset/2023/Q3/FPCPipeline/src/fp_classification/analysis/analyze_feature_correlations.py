"""
We analyze feature correlations (independent of model fit) to handle correlated/collinear features
See: https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import pandas as pd
from argparse import ArgumentParser
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from helper_functions import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial.distance import squareform
# Set style and font
plt.style.use('seaborn-paper')
mpl.rc("font", family="Serif")

from collections import defaultdict
import joblib
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

def compute_correlation_matrix(features_matrix, feature_names):


    features_df = pd.DataFrame(data=features_matrix,
                               columns=feature_names)

    corr_matrix = features_df.corr(method='spearman')

    return corr_matrix.to_numpy()


def compute_collinearity_metric(corr_matrix):
    """
    Compute the collinearity metric as the mean of the sum of the off-diagonal
    elements

    """
    if isinstance(corr_matrix, np.ndarray) is False:
        corr_matrix = np.abs(corr_matrix.to_numpy())
    else:
        corr_matrix = np.abs(corr_matrix)

    num_diagonal_elements = corr_matrix.shape[0]
    num_off_diagonal_elements = corr_matrix.shape[0]*corr_matrix.shape[1] - corr_matrix.shape[0]

    sum_off_diag = np.sum(corr_matrix, axis=None) - np.trace(corr_matrix)

    mean_off_diag = sum_off_diag/num_off_diagonal_elements

    return mean_off_diag


def plot_corr_matrix(corr_matrix=None, feature_names = None, fname=None):


    fig = plt.figure()
    ax = fig.add_subplot(111)

#    if len(feature_names) > 50:
#        fontsize = 20
#    elif len(feature_names) <= 50 and len(feature_names) >=20:
#        fontsize = 35
#    else:
#        fontsize=40

    sns.heatmap(data=corr_matrix,
                xticklabels=feature_names,
                yticklabels=feature_names,
                ax=ax,
                center=0.0)

#    for item in ax.get_xticklabels():
#        item.set_fontsize(fontsize)
#    for item in ax.get_yticklabels():
#        item.set_fontsize(fontsize)
#
    cbar = ax.collections[0].colorbar
    #cbar.ax.tick_params(labelsize=fontsize)

    fig.tight_layout()
    fig.savefig(fname)
    plt.close()


def cluster_features(train_df=None, threshold=1, corr_linkage=None):

    # Get the current set of feature names
    features_matrix, labels, feature_names = create_features_matrix(train_df,
                                                                    selected_features=None)

    print('Num features before clustering : {}'.format(len(feature_names)))

    # Compute MI between each feature label
    features_mi = mutual_info_classif(features_matrix, labels)


    cluster_ids = hierarchy.fcluster(corr_linkage,
                                     t=threshold,
                                     criterion='distance')

    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
            cluster_id_to_feature_ids[cluster_id].append(idx)

    selected_features = []

    for features_in_cluster in cluster_id_to_feature_ids.values():
        mi_cluster = []
        # Choose feature from each cluster s.t it has maximum mutual information w.r.t the label
        for v in features_in_cluster:
            mi_cluster.append(features_mi[v])
        mi_cluster = np.array(mi_cluster)
        max_mi_idx = np.argmax(mi_cluster)
        selected_features.append(features_in_cluster[max_mi_idx])

    selected_features_names = feature_names[selected_features]

    # Compute correlation matrix for the reduced feature set
    reduced_feature_matrix, _, selected_feature_names  = create_features_matrix(train_df,
                                                                                selected_features=selected_features_names)

    print('Num features after clustering: {}'.format(len(selected_features_names)))

    corr_matrix = compute_correlation_matrix(reduced_feature_matrix,
                                             selected_feature_names)



    return selected_feature_names, corr_matrix

def make_dendogram(train_df, selected_features=None, color_threshold=None):

    features_matrix, labels, feature_names = create_features_matrix(train_df,
                                                                    selected_features=selected_features)

    corr_matrix = compute_correlation_matrix(features_matrix,
                                             feature_names)

    # Convert correlation matrix into a distance matrix
    # See: https://stackoverflow.com/questions/38070478/how-to-do-clustering-using-the-matrix-of-correlation-coefficients
    assert(isinstance(corr_matrix, np.ndarray))
    corr_matrix = (corr_matrix + corr_matrix.T)/2
    np.fill_diagonal(corr_matrix, 1)
    distance_matrix = 1 - np.abs(corr_matrix)


    corr_linkage = hierarchy.ward(squareform(distance_matrix))

    # construct the dendrogram to visualize clusters
    if len(feature_names) == 99:
        fig1 = plt.figure(figsize=(10, 7))
    else:
        fig1 = plt.figure()

    ax1 = fig1.add_subplot(111)

    if color_threshold is None:
        dendro = hierarchy.dendrogram(corr_linkage,
                                      labels=list(feature_names),
                                      ax=ax1,
                                      leaf_rotation=90,
                                      distance_sort='ascending')
    else:
        dendro = hierarchy.dendrogram(corr_linkage,
                                      labels=list(feature_names),
                                      ax=ax1,
                                      leaf_rotation=90,
                                      color_threshold=color_threshold,
                                      distance_sort='ascending')

    for item in ax1.get_xticklabels():
        item.set_fontsize(10)
    for item in ax1.get_yticklabels():
        item.set_fontsize(10)

    # Add 'cut-line'
    if len(feature_names) == 99:
        ax1.axhline(y=1, linestyle='--', c='red')

    return corr_linkage, ax1


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--out_dir', type=str, help='Directory where images are stored', default='./results/baseline')
    args = parser.parse_args()

    # Check mode
    dir_tokens = args.out_dir.split('/')
    if dir_tokens[-1] == '':
        dataset = dir_tokens[-5]
        config = dir_tokens[-4]
        unc_type = dir_tokens[-2]
    else:
        dataset = dir_tokens[-4]
        config = dir_tokens[-3]
        unc_type = dir_tokens[-1]


    if dataset == 'umc' or dataset == 'lits': # Single dataset mode
        # Read the training dataset
        train_df = pd.read_pickle(os.path.join(args.out_dir, 'train_dataset.pkl'))
    else:
        # We need to combine the training dataset for UMC and LiTS before feature selection
        # First we check if the directory is already present
        target_dir = '/data/ishaan/{}/{}/fp_results/{}'.format(dataset, config, unc_type)
        if os.path.exists(target_dir) is False:
            os.makedirs(target_dir)
        umc_dir = '/data/ishaan/umc/{}/fp_results/{}'.format(config, unc_type)
        lits_dir = '/data/ishaan/lits/{}/fp_results/{}'.format(config, unc_type)

        umc_train_df = pd.read_pickle(os.path.join(umc_dir, 'train_dataset.pkl'))
        lits_train_df = pd.read_pickle(os.path.join(lits_dir, 'train_dataset.pkl'))

        # Merge both dataframe
        train_df = pd.concat([lits_train_df, umc_train_df], axis=0)
        # Save the merged DF in the target directory
        train_df.to_pickle(os.path.join(target_dir, 'train_dataset.pkl'))

        # Additionally, we merge the test features dataset as well (to simplify scripts later)
        umc_test_df = pd.read_pickle(os.path.join(umc_dir, 'test_dataset.pkl'))
        lits_test_df = pd.read_pickle(os.path.join(lits_dir, 'test_dataset.pkl'))
        umc_test_df['Dataset'] = 'umc'
        lits_test_df['Dataset'] = 'lits'
        test_df = pd.concat([umc_test_df, lits_test_df], axis=0)
        test_df.to_pickle(os.path.join(target_dir, 'test_dataset.pkl'))


    selected_features = None
    features_matrix, _, feature_names = create_features_matrix(train_df,
                                                               selected_features=selected_features)

    corr_matrix_original = compute_correlation_matrix(features_matrix,
                                                      feature_names)

    corr_linkage, ax = make_dendogram(train_df,
                                      color_threshold=1.0)

    # Reduce features in the original set via agglomerative clustering (controlled by a threshold)
    selected_features, corr_matrix_reduced = cluster_features(train_df=train_df,
                                                              corr_linkage=corr_linkage,
                                                              threshold=1.0)

    # Drop features not selected by clustering
    features_to_drop = list(set(feature_names) - set(selected_features))
    train_df = train_df.drop(features_to_drop, axis=1)

    print('Selected features : {}'.format(selected_features))

    joblib.dump(selected_features, os.path.join(args.out_dir, 'selected_features.pkl'))

    # Make the selected feature BOLD in the xticklabels()
    for label in ax.get_xticklabels():
        if label.get_text() in selected_features:
            label.set_fontweight('bold')

    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'dendogram.pdf'))

    # Save the correlation matrices for viz
    plot_corr_matrix(corr_matrix=corr_matrix_original,
                     feature_names = feature_names,
                     fname = os.path.join(args.out_dir, 'corr_matrix_original.pdf'))

    plot_corr_matrix(corr_matrix=corr_matrix_reduced,
                     feature_names = selected_features,
                     fname = os.path.join(args.out_dir, 'corr_matrix_reduced.pdf'))

