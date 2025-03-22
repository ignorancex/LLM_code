import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.icr import ICRDiscovery


def cluster_assign(epoch, images_lists, trainFeatures, feature_index, k_ratio, h_ratio):

    if k_ratio >= h_ratio:
        k_ratio = h_ratio
    # current cluster result is images_lists
    assert images_lists is not None

    icr = ICRDiscovery(trainFeatures.shape[0])

    for cluster, images in enumerate(images_lists):
        if len(images) < 2:
            return icr
        else:

            if len(images) > 5:
                # cluster center
                cluster_center = np.mean(np.array(trainFeatures)[images], axis=0)
                # distance from each point to cluster center
                dist = torch.from_numpy(np.sqrt(
                    np.sum(np.asarray(np.array(cluster_center) - np.array(trainFeatures)[images]) ** 2, axis=1)))
                # find topk index for [images]
                topk_ind = dist.topk(int(int(len(images) * k_ratio)), 0, False)[
                    1].numpy()  # smallest k value index, belong to same cluster

                # current same labels images --> cluster_i_
                cluster_i_ = np.array(feature_index)[np.array(images)[topk_ind]]

            else:
                cluster_i_ = np.array(feature_index)[images]

            if len(cluster_i_) > 1:
                for v in cluster_i_:
                    icr.position[int(v)] = int(v)
                    icr.neighbours[int(v)] = torch.from_numpy(np.delete(cluster_i_, np.where(
                        cluster_i_ == v)[0])).cuda()


    return icr

