#!/usr/bin/python
#Title			: get_bows_clusters.py
#Usage			: python get_bows_clusters.py -h
#Author			: pmorvalho
#Date			: September 20, 2021
#Description	        : Given a Bag of Words model, we use clustering algorithms to find the appropriate number of clusters in the correct submissions directory. For each cluster, we find which program is the cluster representative (closer to the cluster centroid). Then we create a new directory with the representative for each cluster found.
#Notes			:
#Python Version: 3.8.5
# (C) Copyright 2021 Pedro Orvalho.
#==============================================================================

import argparse
from find_closest_program import calculate_program_distances, load_bow_pickle
from create_closest_programs_dir import recreate_submissions_name
import numpy as np
import os
from sklearn import cluster, mixture
from sys import argv

def get_BagOfWords_vectors(bow_name):
    bow = load_bow_pickle(bow_name)
    vectors = bow["vectors"]
    for v in range(len(vectors)):
      vectors[v] = vectors[v].toarray().tolist()[0]
    return vectors, bow["submissions"]

def get_Birch_clusters(vectors):
    # clusters all program prog_vectors using Birth algorithm with unknown number of clusters
    # return a list of labels, the number of clusters found and the centroid for each of those clusters
    birch = cluster.Birch(n_clusters=None)
    birch.fit(vectors)
    labels = birch.predict(vectors)
    return labels, len(set(labels)), birch.subcluster_centers_

def get_KMeans_clusters(vectors, n_clusters, MiniBatch=False):
    # clusters all program prog_vectors using KMeans algorithm
    # return a list of labels, the number of clusters found and the centroid for each of those clusters
    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=0) if not MiniBatch else cluster.MiniBatchKMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(vectors)
    labels = kmeans.predict(vectors)
    return labels, len(set(labels)), kmeans.cluster_centers_

def get_GaussianMixture_clusters(vectors, n_clusters):
    # clusters all program prog_vectors using KMeans algorithm
    # return a list of labels, the number of clusters found and the centroid for each of those clusters
    gm = mixture.GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=0)
    gm.fit(vectors)
    labels = gm.predict(vectors)
    return labels, len(set(labels)), gm.means_

def get_clusters(vectors, alg, n_clusters):
    if alg == "b":
        return get_Birch_clusters(vectors)
    elif alg == "k":
        return get_KMeans_clusters(vectors, n_clusters)
    elif alg == "mbk":
        return get_KMeans_clusters(vectors, n_clusters, MiniBatch=True)
    elif alg == "gm":
        return get_GaussianMixture_clusters(vectors, n_clusters)
    else:
        exit("Clustering algorithm not implemented!!!")

def divide_programs_per_cluster(labels, vectors, prog_names):
    # puts each program (and respectivo submission name) to its own cluster
    programs_per_cluster = []
    for nc in range(len(set(labels))):
        programs_per_cluster.append([])

    for p in range(len(vectors)):
        programs_per_cluster[labels[p]].append((vectors[p], prog_names[p]))

    return programs_per_cluster

def get_real_centroids(programs_per_cluster, theoretical_centroids, verbose=False):
    # the theoretical centroids are means calculated using all vectors in a cluster.
    # now we want to get the closest program to each centroid to know which student program represents each cluster
    clusters_representatives = []
    for nc in range(len(theoretical_centroids)):
    # for each cluster
        centroid = theoretical_centroids[nc]
        prog_vectors_cluster = [v for v, _ in programs_per_cluster[nc]]
        prog_distances = calculate_program_distances(centroid.tolist(), prog_vectors_cluster)
        min_distance = min(prog_distances)
        _, rep_name = programs_per_cluster[nc][prog_distances.index(min_distance)]
        clusters_representatives.append(rep_name)
        if verbose:
            print("Cluster ", nc, " with ", len(programs_per_cluster[nc])," programs. Representative: ", rep_name)

    return clusters_representatives

def parser():
    parser = argparse.ArgumentParser(prog='get_bows_clusters.py', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('clustering_alg', nargs='?', choices=['k','mbk','b','gm'], default='b',
                        help="Clustering Methods:\nk: KMeans\nmbk: MiniBatch KMeans\nb: BIRCH (Default)\ngm: Gaussian Mixture")
    parser.add_argument('-b', '--correct_bow', type=str, help='Path to Bag of Words model (pickle file) with correct submissions.')
    parser.add_argument('-cp', '--cluster_percentage', nargs='?', type=int, default=10, help='(INT) Into how many clusters should the clustering algorithms KMeans, MiniBatchKMeans and GaussianMixture divide the submissions (Percentage). Default=10 (10%)')
    parser.add_argument('-o', '--output_dir', nargs='?', help='Create a directory with each cluster representative. Directory name: birch_clusters_10p/{bow_name}/lab_name (For 10% cluster_percentage).')
    parser.add_argument('-si', '--submission_id', nargs='?', help='path to the student submission. The student we want to ignore.\n The program\'s name follows the structure path_name/exYY-studentID-DATE.c we only want the studentID')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Prints debugging information.')
    args = parser.parse_args(argv[1:])
    return args

cluster_algs = {"k": "kmeans", "mbk": "minibatch_kmeans", "b": "birch", "gm": "gaussian_mixture"}
if __name__ == '__main__':
    args = parser()
    prog_vectors, submission_names = get_BagOfWords_vectors(args.correct_bow)
    if args.submission_id:
        stu_id = args.submission_id.split("/")[-1] # to remove from path name previous directories and number and date of the submissions, we only want the student id
        if "-" in stu_id:
            stu_id = stu_id.split("-")[1]
        stu_id = stu_id.replace(".c", "")   
        for s in range(len(submission_names)-1,-1,-1):
            if stu_id in submission_names[s]:
                submission_names.pop(s)
                prog_vectors.pop(s)
    bow_name = args.correct_bow.split("/")[-1].replace(".pickle.gz", "")
    lab_name = args.correct_bow.replace("bows/correct_programs_clara_supports/", "").replace(bow_name+".pickle.gz", "")
    args.cluster_percentage=args.cluster_percentage/100
    n_clusters=int(args.cluster_percentage*len(prog_vectors))
    if n_clusters == 0: 
        n_clusters = 1 if args.cluster_percentage == 0.1 else 2
    if args.verbose:
        print("Using clustering algorithm: ", cluster_algs[args.clustering_alg])
        print("Dealing with lab: ", lab_name)
        print("Using Bag of Words model: ", bow_name)
        print("Computing clusters...")
        if args.clustering_alg != "b":
            print("Searching for", n_clusters, "clusters!")

    labels, n_clusters, theoretical_centroids = get_clusters(prog_vectors, args.clustering_alg, n_clusters)
    if args.verbose:
        print("Found ", n_clusters, " different clusters")
        print("Computing reference implementations for each cluster...")
    programs_per_cluster = divide_programs_per_cluster(labels, prog_vectors, submission_names)
    # the theoretical centroids are means calculated using all vectors in a cluster.
    # now we want to get the closest program to each centroid to know which student program represents each cluster
    real_centroids = get_real_centroids(programs_per_cluster, theoretical_centroids, verbose=args.verbose)

    if args.output_dir:
        if args.verbose:
            print("Saving clusters...")
        os.system("mkdir -p {d}".format(d=args.output_dir))
        for nc in range(len(real_centroids)):
            sub_name = recreate_submissions_name(real_centroids[nc])
            os.system("cp {p} {d}/c{nc}-{sn}".format(p=sub_name, d=args.output_dir,nc=nc,sn=sub_name.split("/")[-1]))
        if args.verbose:
            print("Clusters saved into: {d}".format(d=args.output_dir))
