import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.icr import ICRDiscovery
import  cv2
import os
ICRset = []

def AllScore(alpha, len_set):
    """
    calculate score when all same
    :param alpha: alpha
    :param len_set: len(ICRset)
    :return:  allscore
    """
    allscore = alpha
    if len_set == 1:
        allscore += 0
    else:
        for i in range(len_set - 1):
            allscore = (allscore + 1) * alpha
    return allscore + 1


def NpWhere(M, anc, nei):
    """

    :param M: An array
    :param anc: instance 1
    :param nei:  instance 2
    :return: is 1 and 2 all in M for same class
    """
    flag = 0
    for v in M:
        if anc in v and nei in v:
            flag = 1
            break
    return flag

def InstanceGroup(M, anc):
    """
    find grop with same class with anc
    :param M:
    :param anc:
    :return:
    """
    index = 0
    for idx, v in enumerate(M):
        if anc in v:
            index = idx
            break
    return M[index]


def PreScore(epoch, images_lists, image_dict, trainFeatures, feature_index, trainset, high_ratio, k_ratio, alpha, beta):
    """
    get reliable group based score threshold
    Args:
        ICRset: save every epoch neighbour information
        ICR: an ICR class save position and neighbour information
        images_lists: include same cluster set, value is feature index
        trainFeatures: current network features
        feature_index: according images_lists to get truly image index
        k_ratio: topk is a ratio
        alpha: coefficient of score
        belta: threshold
    return:
    """
    # r2: out of r2 is negative.   r3: in r3 is positive
    # r2_ratio = 0.8

    # num_highest = 25
    # topscore_num = 5
    beta1 = 0
    beta2 = 3.0

    icr = ICRDiscovery(trainFeatures.shape[0])

    if k_ratio >= high_ratio:
        k_ratio = high_ratio
    # current cluster result is images_lists
    assert images_lists is not None

    # cluster_center = np.zeros((len(images_lists), trainFeatures.shape[1]), dtype='float32')

    epoch_images = []
    save_flag = 0

    for cluster, images in enumerate(images_lists):
        # cluster center
        cluster_center = np.mean(np.array(trainFeatures)[images], axis=0)
        # distance from each point to cluster center
        dist = torch.from_numpy(np.sqrt(
            np.sum(np.asarray(np.array(cluster_center) - np.array(trainFeatures)[images]) ** 2, axis=1)))

        top1 = dist.topk(1, 0, False)[1].numpy()
        anchor = np.array(feature_index)[images[top1[0]]]  # nearset center point

        # find topk index for [images]

        topk_ind = dist.topk(int(len(images) *
                                 k_ratio), 0, False)[1].numpy()  # smallest k value index, belong to same cluster
        topk_r2_ind = dist.topk(int(len(images)), 0, False)[1].numpy()


        if epoch < 20 or len(ICRset) < 15:
            # there's no need to score
            cluster_reliable = np.array(feature_index)[np.array(images)[topk_ind]]

        else:

            # background
            cluster_backfround = np.array(feature_index)[np.array(images)[topk_r2_ind]]
            # topk
            cluster_topk = np.array(feature_index)[np.array(images)[topk_ind]]
            # need score
            cluster_score = np.array(list(set(cluster_backfround).difference(set(cluster_topk))))

            # in topk
            cluster_intopk = np.array(list(set(cluster_topk).difference(set(np.array([anchor])))))

            # reliable
            cluster_reliable = cluster_topk

            anc_score = 1
            # array_score = np.array([], dtype=int)
            # v_add = []
            # v_add_score = []
            # v_remove = []
            # v_remove_score = []

            ### remove in topk
            if len(images) > 0:
                for neighbour in cluster_intopk:
                    for cluster_index in range(len(ICRset) - 1, -1, -1):
                        Dict = ICRset[cluster_index]
                        if Dict[anchor] == Dict[neighbour]:
                            anc_score += (alpha ** (len(ICRset) - cluster_index))
                        else:
                            anc_score += ((-1) * alpha ** (len(ICRset) - cluster_index))
                    # array_score = np.append(array_score, anc_score)
                    if anc_score < beta1:
                        cluster_reliable = np.delete(cluster_reliable, np.where(cluster_reliable == neighbour)[0])
                    anc_score = 1

                # array_score = torch.from_numpy(array_score)
                # topk_score_small = array_score.topk(topscore_num, 0, False)[1].numpy()
                # for ii in topk_score_small:
                #     value = cluster_topk[int(ii)]
                #     # v_remove.append(value)
                #     cluster_reliable = np.delete(cluster_reliable, np.where(cluster_reliable == value)[0])

                # topk_score_small_ = array_score.topk(array_score.size(0), 0, False)[1].numpy()
                # for ii in topk_score_small_:
                #     value = cluster_topk[int(ii)]
                #     v_remove.append(value)
                #     v_remove_score.append(array_score[int(ii)])

            anc_score = 1
            # array_score = np.array([], dtype=int)
            ### add out of topk
            for neighbour in cluster_score:
                for cluster_index in range(len(ICRset) - 1, -1, -1):
                    Dict = ICRset[cluster_index]
                    if Dict[anchor] == Dict[neighbour]:
                        anc_score += (alpha ** (len(ICRset) - cluster_index))
                    else:
                        anc_score += ((-1) * alpha ** (len(ICRset) - cluster_index))

                # array_score = np.append(array_score, anc_score)
                if anc_score >= beta2:
                    cluster_reliable = np.append(cluster_reliable, neighbour)
                anc_score = 1

            # add from out of topk
            # length_ = len(array_score)
            # if length_ > topscore_num:
            #     array_score = torch.from_numpy(array_score)
            #     topk_score = array_score.topk(topscore_num, 0, True)[1].numpy()
            # else:
            #     array_score = torch.from_numpy(array_score)
            #     topk_score = array_score.topk(length_, 0, True)[1].numpy()
            # for ii in topk_score:
            #     cluster_reliable = np.append(cluster_reliable, cluster_score[int(ii)])
                # v_add.append(cluster_score[int(ii)])

            # topk_score_ = array_score.topk(array_score.size(0), 0, True)[1].numpy()
            # for ii in topk_score_:
            #     v_add.append(cluster_score[int(ii)])
            #     v_add_score.append(array_score[int(ii)])

            # if len(images) > num_highest and save_flag == 0 and epoch in [120, 125, 128, 130, 135, 140, 150, 155,
            #                                                               160, 170, 175, 180, 185, 190, 195]:
            #     save_flag = 1
            #     if not os.path.exists('./save/epoch_{}'.format(epoch)):
            #         os.makedirs('./save/epoch_{}'.format(epoch))
            #
            #     for i in cluster_reliable:
            #         cv2.imwrite('./save/epoch_{}/topk_{}_label_{}.jpg'.format(
            #             epoch, i, trainset.targets[i]), trainset.data[i])
            #     cv2.imwrite('./save/epoch_{}/anchor_top1_{}_label_{}.jpg'.format(
            #         epoch, anchor, trainset.targets[anchor]), trainset.data[anchor])
            #
            #     for idx, i in enumerate(v_add):
            #         cv2.imwrite('./save/epoch_{}/topk_{}_label_{}_add_{}_score_{}.jpg'.format(
            #             epoch, i, trainset.targets[i], idx, v_add_score[idx]), trainset.data[i])
            #     for idx, i in enumerate(v_remove):
            #         cv2.imwrite('./save/epoch_{}/topk_{}_label_{}_remove_{}_score_{}.jpg'.format(
            #             epoch, i, trainset.targets[i], idx, v_remove_score[idx]), trainset.data[i])



        if len(cluster_reliable) > 1:
            for v in cluster_reliable:
                icr.position[int(v)] = int(v)
                icr.neighbours[int(v)] = torch.from_numpy(np.delete(cluster_reliable, np.where(
                    cluster_reliable == v)[0])).cuda()

        # epoch_images.append(np.array(feature_index)[np.array(images)])

    ICRset.append(image_dict)
    # define length is 15
    if len(ICRset) > 15:
        del ICRset[0]

    return icr
