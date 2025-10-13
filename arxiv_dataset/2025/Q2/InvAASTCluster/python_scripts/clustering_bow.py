#!/usr/bin/python
#Title			: clustering_bow.py
#Usage			: python clustering_bow.py -h
#Author			: pmorvalho
#Date			: May 07, 2021
#Description	        : Clustering methods for BoW using programs' code, invariants or ASTs
#Notes			:
#Python Version	: 3.8.5
# (C) Copyright 2021 Pedro Orvalho.
#==============================================================================

import argparse
from bag_of_words import *
import glob
import matplotlib.colors as pltc
import matplotlib.pyplot as plt
import matplotlib
import re
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.decomposition import PCA, TruncatedSVD
from scipy.optimize import linear_sum_assignment as linear_assignment
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import time
import warnings
from sklearn import cluster, datasets, mixture
from sklearn.manifold import TSNE
from sklearn.metrics.cluster import *
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

def assignment(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_assignment(w.max() - w)
    return w, row_ind, col_ind
def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # FROM: https://github.com/k-han/DTC/blob/master/utils/util.py

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    w, row_ind, col_ind = assignment(y_true, y_pred)
    return w[row_ind, col_ind].sum()/y_pred.size

def get_clustering_metrics(y_true, y_pred):
    """
    Calculate clustering metrics. Require scikit-learn installed

    # Arguments
        y_true: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        c_method: clustering method name

    # prints
        clustering method and respective metrics

    # Return
        dictionary with values for each metric
    """
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    metrics = dict()
    acc = cluster_acc(y_true, y_pred)
    metrics['acc'] = acc
    rand_ind = rand_score(y_true, y_pred)
    metrics['rand_ind'] = rand_ind
    adj_rand_ind = adjusted_rand_score(y_true, y_pred)
    metrics['adj_rand_ind'] = adj_rand_ind
    nmi = normalized_mutual_info_score(y_true, y_pred)
    metrics['nmi'] = nmi
    ami = adjusted_mutual_info_score(y_true, y_pred)
    metrics['ami'] = ami
    homo_s = homogeneity_score(y_true, y_pred)
    metrics['homo_s'] = homo_s
    comp_s = completeness_score(y_true, y_pred)
    metrics['comp_s'] = comp_s
    v_measure = v_measure_score(y_true, y_pred)
    metrics['v_measure'] = v_measure
    fmi = fowlkes_mallows_score(y_true, y_pred)
    metrics['fmi'] = fmi
    # cm = contingency_matrix(y_true, y_pred)
    # pcm = pair_confusion_matrix(y_true, y_pred)
    print(metrics)
    print()
    print()
    return metrics

def clustering_methods(method, X):
    if method == 'k':
        print("Clustering method chosen: Kmeans")
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        return kmeans.fit_predict(X)
    elif method == 'ap':
        print("Clustering method chosen: Affinity Propagation")
        ap = AffinityPropagation(random_state=0)
        # preference=n_clusters,
        return ap.fit_predict(X)
    elif method == 'ms':
        ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
        return ms.fit_predict(X)
    elif method == 'mbk':
        two_means = cluster.MiniBatchKMeans(n_clusters=n_clusters, random_state=0)
        return two_means.fit_predict(X)
    elif method == 'w':
        ward = cluster.AgglomerativeClustering(
            n_clusters=n_clusters, linkage='ward',
            connectivity=connectivity)
        return ward.fit_predict(X)
    elif method == 'sc':
        spectral = cluster.SpectralClustering(
            n_clusters=n_clusters, eigen_solver='arpack',
            affinity="nearest_neighbors", random_state=0)
        return spectral.fit_predict(X)
    elif method == 'db':
        dbscan = cluster.DBSCAN(eps=0.3)
        return dbscan.fit_predict(X)
    elif method == 'opt':
        optics = cluster.OPTICS()
        return optics.fit_predict(X)
        #min_samples=20, #params['min_samples'], xi=0.05params['xi'], min_cluster_size=params['min_cluster_size'])
    elif method == 'ac':
        average_linkage = cluster.AgglomerativeClustering(
            linkage="average", affinity="cityblock",
            n_clusters=n_clusters, connectivity=connectivity)
        return average_linkage.fit_predict(X)
    elif method == 'b':
        birch = cluster.Birch(n_clusters=n_clusters)
        return birch.fit_predict(X)
    elif method == 'gm':
        gmm = mixture.GaussianMixture(
            n_components=n_clusters, covariance_type='full', random_state=0)
        return gmm.fit_predict(X)
    else:
        exit("Unknown clustering method!")

def print_all(true_labels):
    # From https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py
    # ============
    # Set up cluster parameters
    # ============
    plt.figure(figsize=(9 * 2 + 3, 13))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.95, wspace=.05,
                        hspace=.01)
    plt.rc('font',**{'family':'serif'})
    plot_num = 1

    default_base = {'quantile': .3,
                    'eps': .3,
                    'damping': .9,
                    'preference': -200,
                    'n_neighbors': 10,
                    'n_clusters': n_clusters,
                    'min_samples': 20,
                    'xi': 0.05,
                    'min_cluster_size': 0.1,
                    'x_min': -1.0,
                    'x_max': 1.5,
                    'y_min': -1.0,
                    'y_max': 1.5}

    params = default_base.copy()
    X = Xpca

    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(
        X, n_neighbors=params['n_neighbors'], include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # ============
    # Create cluster objects
    # ============
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'], random_state=0)
    ward = cluster.AgglomerativeClustering(
        n_clusters=params['n_clusters'], linkage='ward',
        connectivity=connectivity)
    spectral = cluster.SpectralClustering(
        n_clusters=params['n_clusters'], eigen_solver='arpack',
        affinity="nearest_neighbors", random_state=0)
    dbscan = cluster.DBSCAN(eps=params['eps'])
    optics = cluster.OPTICS(min_samples=params['min_samples'],
                            xi=params['xi'],
                            min_cluster_size=params['min_cluster_size'])
    affinity_propagation = cluster.AffinityPropagation(
        damping=params['damping'], preference=params['preference'], random_state=0)
    average_linkage = cluster.AgglomerativeClustering(
        linkage="average", affinity="cityblock",
        n_clusters=params['n_clusters'], connectivity=connectivity)
    birch = cluster.Birch(n_clusters=params['n_clusters'])
    gmm = mixture.GaussianMixture(
        n_components=params['n_clusters'], covariance_type='full', random_state=0)

    clustering_algorithms = (
        ('KMeans', kmeans),
        ('MiniBatch\nKMeans', two_means),
        ('Affinity\nPropagation', affinity_propagation),
        ('MeanShift', ms),
        ('Spectral\nClustering', spectral),
        ('Ward', ward),
        ('Agglomerative\nClustering', average_linkage),
        ('DBSCAN', dbscan),
        ('OPTICS', optics),
        ('BIRCH', birch),
        ('Gaussian\nMixture', gmm)
    )
    results = dict()
    for name, algorithm in clustering_algorithms:
        t0 = time.time()
        # catch warnings related to kneighbors_graph
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the " +
                "connectivity matrix is [0-9]{1,2}" +
                " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning)
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding" +
                " may not work as expected.",
                category=UserWarning)
            algorithm.fit(X)

        t1 = time.time()
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(int)
        else:
            y_pred = algorithm.predict(X)

        plt.subplot(1, len(clustering_algorithms)+1, plot_num)
        # if i_dataset == 0:
        plt.title(name, y=0.95, pad=-14, size=15)

        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(int(np.max(y_pred)) + 1))))
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

        plt.xlim(params['x_min'], params['x_max'])
        plt.ylim(params['y_min'], params['y_max'])
        plt.xticks(())
        plt.yticks(())
        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')

        labels = []
        for c in range(n_clusters):
            labels.append(Xpca[y_pred == c])

        c_method = name.replace("\n", " ")
        # avg_oc, min_oc, max_oc = get_clustering_similarity(labels, name.replace("\n", " "))
        print(c_method)
        metrics = get_clustering_metrics(true_labels, y_pred)
        results[c_method] = metrics

        # results[name.replace("\n", " ")] = {"avg_oc":avg_oc, "min_oc":min_oc, "max_oc":max_oc, "time": ('%.2fs' % (t1 - t0)).lstrip('0')}
        plt.text(.99, .04,  "{:.3f}%".format(metrics['acc']),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')

        plot_num += 1

    plt.subplot(1, len(clustering_algorithms)+1, plot_num)
    plt.title("Student\nSubmissions", y=0.95, pad=-14, size=15)

    for c in range(n_clusters):
        plt.scatter(filtered_names[c][:,0] , filtered_names[c][:,1], s=10, color = colors[c])

    plt.xlim(params['x_min'], params['x_max'])
    plt.ylim(params['y_min'], params['y_max'])
    plt.xticks(())
    plt.yticks(())

    # plt.show()

    return results

def clean_exercises_names(ex_names):
    ex_names_cleaned = []
    for ex in ex_names:
        d = ex.split("/")[-1].split(".")[0]
        e = ex.split("/")[-2]
        lab = ex.split("/")[-3]
        ds = ex.split("/")[-4]
        ex_names_cleaned.append("{ds}-{lab}-{e}-{d}".format(ds=ds, lab=lab, e=e, d=d))
        continue
        n_ex = re.sub("injected_programs/", "", ex)
        n_ex = re.sub("ASTs/", "", n_ex)
        n_ex = re.sub("daikon-invariants/", "", n_ex)
        n_ex = re.sub(".*?_submissions/", "", n_ex)
        n_ex = re.sub(".c", "", n_ex)
        n_ex = re.sub(".ast", "", n_ex)
        n_ex = re.sub("invariants.*?gz", "", n_ex)
        n_ex = re.sub("/", "-", n_ex)
        ex_names_cleaned.append(n_ex)

    return ex_names_cleaned

def parser():
    parser = argparse.ArgumentParser(prog='clustering_bow.py', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('clustering', nargs='?', choices=['k','ap','ms','mbk','ac', 'w','sc','db','opt','ac_linkage','b','gm'], default='k',
                        help="Clustering Methods:\nk: KMeans\nap : Affinity Propagation\nms : MeanShift\nmbk: MiniBatch KMeans\nac: Agglomerative Clustering\n w : Ward\nsc : Spectral Clustering\ndb: DBSCAN\nopt: OPTICS\nb: BIRCH\ngm: Gaussian Mixture")
    parser.add_argument('-d', '--dir', nargs='?', default="daikon-invariants/",
                        help="directory with submissions for lab submissions")
    parser.add_argument('--pickle_file', nargs='?', default="results",
                        help="path for pickle file with metrics results")
    parser.add_argument('-s', '--syntactic', action="store_true", default=False,
                        help="creates a bag of words based on program's syntax")
    parser.add_argument('-a', '--ast', action="store_true", default=False,
                        help="creates a bag of words based on program's AST syntax")
    parser.add_argument('-i', '--invariants', action="store_true", default=False,
                        help="creates a bag of words based on program's invariants")
    parser.add_argument('--tsne', action="store_true", default=False,
                        help="Generates tsne plots instead of using TruncatedSVD")
    parser.add_argument('--all', action="store_true", default=False,
                        help="Prints all clustering methods")
    parser.add_argument('-sn','--invariants_scope_name', action="store_true", default=False, help="creates a bag of words based on program's invariants with scopes' names")
    parser.add_argument('-dp', '--invariants_scope_depth', action="store_true", default=False, help="creates a bag of words based on program's invariants with scopes' depth")
    parser.add_argument('-ng', '--ngrams', default=3, type=int,
                        help="Computes ngrams of max length --ng. Ngrams with length from 1 to --ng")
    parser.add_argument('-o', '--output_file', nargs='?', help='the name of the output file.')
    args = parser.parse_args(argv[1:])
    return args

if __name__ == '__main__':
    args = parser()
    bow = BoW(args.syntactic, args.ast, args.invariants, args.invariants_scope_name, args.invariants_scope_depth, args.dir, args.ngrams)
    vectorizer, X = bow.get_BOW()
    if args.tsne:
        clf = TSNE(n_components=2, random_state=0)
    else:
        clf = TruncatedSVD(bow.get_vocabulary_size()-1, random_state=0) # n_components must be < n_features must be < not <=

    Xpca = clf.fit_transform(X)
    # Xpca = X
    cluster_algs_names = {'k': "KMeans",'ap': 'Affinity Propagation','ms': 'MeanShift','mbk':'MiniBatch KMeans','ac':'Agglomerative Clustering', 'w': 'Ward','sc': 'Spectral Clustering','db':'DBSCAN','opt':'OPTICS','b':'BIRCH','gm':'Gaussian Mixture'}
    ex_names = glob.glob(args.dir+'/ex*/', recursive=False)
    if len(ex_names) == 0:
        # print("second")
        ex_names = glob.glob(args.dir+'/*/ex*/', recursive=False)
    if len(ex_names) == 0:
        ex_names = glob.glob(args.dir, recursive=False)
        for en in ex_names:
            en = re.sub("-[0-9]*/*?$", "", en)

        print(set(ex_names))

    ex_names = clean_exercises_names(ex_names)
    print("Dealing with exercises :", ex_names)

    n_clusters = len(ex_names)
    names = []
    true_labels = []
    if n_clusters <= 10:
    # for now we have 10 colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    else:
        # colors = [k for k,v in pltc.cnames.items()][25:]
        colors =  plt.cm.get_cmap("tab20b").colors + plt.cm.get_cmap("tab20c").colors

    markers=[".", "d", "*", "^"]
    for n in bow.get_submissions_names():
        for c in range(n_clusters):
            if ex_names[c] in n:
                names.append(ex_names[c])
                true_labels.append(c)
                break

    names = np.array(names)
    true_labels = np.array(true_labels)

    filtered_names = []
    for c in range(n_clusters):
        filtered_names.append(Xpca[names == ex_names[c]])

    if args.all:
        results = print_all(true_labels)
        pickle_file = args.pickle_file+".pickle"
        with open(pickle_file, "wb") as file:
            pickle.dump(results, file)
        os.system("gzip -f {f}".format(f=pickle_file))
    else:
        fig = plt.figure()
        y_pred = clustering_methods(args.clustering, Xpca)
        filtered_labels = []
        _, clusters_temp, real_clusters = assignment(true_labels, y_pred)
        print("Clusters achieved: ", clusters_temp)
        print("Real Clusters: ", real_clusters)
        for c in range(n_clusters):
            filtered_labels.append(Xpca[y_pred == c])

        plt.subplot(2, 2, 1)
        plt.rc('font',**{'family':'serif'})
        points=0
        for c in range(n_clusters):
            points+=len(filtered_labels[c][:,0])
            plt.scatter(filtered_labels[c][:,0] , filtered_labels[c][:,1] , color = colors[real_clusters[c]], marker=markers[real_clusters[c]%4], facecolors='none')

        plt.gca().set_title(str(cluster_algs_names[args.clustering]))
        print("#Clustering points:", points)

        plt.subplot(2, 2, 2)
        points=0
        for c in range(n_clusters):
            points+=len(filtered_names[c][:,0])
            print("Exercise ", ex_names[c][:-1], " with ", len(filtered_names[c][:,0]), " submissions!")
            plt.scatter(filtered_names[c][:,0] , filtered_names[c][:,1] , color = colors[c], marker=markers[c%4], facecolors='none')

        print("#Submisssions points:", points)
        metrics = get_clustering_metrics(true_labels, y_pred)
        plt.text(.4, .54,  "{:.3f}".format(metrics['acc']),
                 # transform=plt.gca().transAxes,
                 transform=fig.transFigure,
                 size=10,
                 horizontalalignment='left')
        # plt.text(x=0.72, y=0.9, s="Ground Truth", fontsize=12, ha="center", transform=fig.transFigure)
        plt.gca().set_title("Ground Truth")

        # plt.show()

    plt.savefig(args.output_file+".pdf", format="pdf", bbox_inches = 'tight', pad_inches = 0.15)
