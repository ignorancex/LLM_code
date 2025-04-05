#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: jjzhou012
@contact: jjzhou012@163.com
@file: exp_revsel.py
@time: 2020/7/15 21:34
@desc:  python exp_revsel.py --bmname polbooks --cdm FG --sampleRatio 1.6 --randomSample 1 -uew 0
'''

import time
import os
import argparse
import warnings

warnings.filterwarnings('ignore')
import logging

logger = logging.getLogger()
logger.setLevel("INFO")
from tqdm import tqdm
from utils.graph_op import *
from utils.communityDetection_op import *
from source.REVSEL import robustECD_VSEL

#
LARGE_Nets = ['amazon-sub', 'dblp-sub']


def get_para():
    parser = argparse.ArgumentParser(description='manual to this script')
    # file para
    parser.add_argument('--bmname', help="name of benchmark dataset", required=False, type=str, default='karate')
    parser.add_argument('--cdm', help="community detection method", required=True, type=str, default=None)
    parser.add_argument('--sampleRatio', help="ratio of edge sample", required=True, type=float, default=None)
    parser.add_argument('--randomSample', help="weighted random or sorted select", required=True, type=int, default=None)
    parser.add_argument('--threshold', help="prune threshold", required=False, type=int, default=None)
    parser.add_argument('--lpm', help="vertex similarity index", required=False, type=str, default=None)
    parser.add_argument('--lpm_mask', help="select lpm", required=False, type=str, default='11111111')  # default: ''
    parser.add_argument('--iterPerIndex', '-ipi', help="sample times perIndex", required=False, type=int, default=10)
    parser.add_argument('--useEdgeWeight', '-uew', help="assign stray nodes using edgeWeight or similarity", required=False, type=int, default=0)
    parser.add_argument('--printINF', '-pinf', help="print info", required=False, type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    args = get_para()
    print(args)
    bmname = args.bmname
    if bmname not in LARGE_Nets:
        file = 'data/{}/{}.gml'.format(bmname, bmname)
    else:
        file = 'data/large-net/{}.gml'.format(bmname)
    cdm = args.cdm
    sampleRatio = args.sampleRatio
    randomSample = bool(args.randomSample)
    threshold = args.threshold
    lpm = args.lpm
    lpm_mask = args.lpm_mask
    iterPerIndex = args.iterPerIndex
    printINF = bool(args.printINF)

    print('experiment setup.')
    print('bmname', bmname)
    print('cdm', cdm)
    print('sampleRatio', sampleRatio)
    print('randomSample', randomSample)
    print('threshold', threshold)
    print('lpm', lpm)
    print('lpm_mask', lpm_mask)
    print('=====================================\n')

    print('get original result...')
    # labels_true = load_trueLabel(bmname)
    g = nx.read_gml(file, label='id')
    labels_true = list(nx.get_node_attributes(g, 'commID').values())

    G = IG.Read_GML(file)
    # this setting is no randomness
    experiment_time = 1

    orig_qqq_list, orig_nmi_list, orig_ari_list, orig_ff1_list = [], [], [], []
    for i in tqdm(range(experiment_time)):
        if cdm not in ['scd', 'lpnx', 'gemsec', 'n2v_km']:
            community = community_method_dict[cdm].__wrapped__(labels_true, input=G)
            orig_qqq, orig_nmi, orig_ari, orig_ff1 = evaluate_results(g, community, labels_true)
        else:
            community = community_method_dict[cdm].__wrapped__(labels_true, input=g)
            orig_qqq, orig_nmi, orig_ari, orig_ff1 = evaluate_results(g, community, labels_true)
        orig_qqq_list.append(orig_qqq)
        orig_nmi_list.append(orig_nmi)
        orig_ari_list.append(orig_ari)
        orig_ff1_list.append(orig_ff1)

    orig_qqq, orig_nmi, orig_ari, orig_ff1 = np.average(orig_qqq_list), np.average(orig_nmi_list), np.average(orig_ari_list), np.average(orig_ff1_list)
    orig_qqq_std, orig_nmi_std, orig_ari_std, orig_ff1_std = np.std(orig_qqq_list), np.std(orig_nmi_list), np.std(orig_ari_list), np.std(orig_ff1_list)

    print('=====================================')
    print('Metrics of original network:.......  ')
    print('orig qqq: {:.3} ~ {:.3}'.format(orig_qqq, orig_qqq_std))
    print('orig nmi: {:.3} ~ {:.3}'.format(orig_nmi, orig_nmi_std))
    print('orig_ari: {:.3} ~ {:.3}'.format(orig_ari, orig_ari_std))
    print('orig_f1 : {:.3} ~ {:.3}'.format(orig_ff1, orig_ff1_std))
    print('=====================================\n')

    en_qqq_list, en_nmi_list, en_ari_list, en_ff1_list = [], [], [], []
    time_consume = []
    for i in tqdm(range(experiment_time)):
        t1 = time.time()
        if printINF:
            logging.info('Exp id:{}'.format(i + 1))
        # community detection
        en_community = robustECD_VSEL(bmname=bmname,
                                      cdm=cdm,
                                      sampleRatio=sampleRatio,
                                      randomSample=randomSample,
                                      threshold=threshold,
                                      sample_times_perIndex=iterPerIndex,
                                      printINF=printINF,
                                      lpm=lpm,
                                      lpm_mask=lpm_mask).run()
        t2 = time.time()
        time_consume.append(t2 - t1)
        if printINF:
            logging.info('final community num: {}'.format(len(en_community)))
        en_qqq, en_nmi, en_ari, en_ff1 = evaluate_results(g, en_community, labels_true)

        en_qqq_list.append(en_qqq)
        en_nmi_list.append(en_nmi)
        en_ari_list.append(en_ari)
        en_ff1_list.append(en_ff1)
    en_qqq, en_nmi, en_ari, en_ff1 = np.average(en_qqq_list), np.average(en_nmi_list), np.average(en_ari_list), np.average(en_ff1_list)
    en_qqq_std, en_nmi_std, en_ari_std, en_ff1_std = np.std(en_qqq_list), np.std(en_nmi_list), np.std(en_ari_list), np.std(en_ff1_list)

    print('=====================================')
    print('Metrics of original network:.......  ')
    print('orig qqq: {:.3} ~ {:.3}'.format(orig_qqq, orig_qqq_std))
    print('orig nmi: {:.3} ~ {:.3}'.format(orig_nmi, orig_nmi_std))
    print('orig_ari: {:.3} ~ {:.3}'.format(orig_ari, orig_ari_std))
    print('orig_f1 : {:.3} ~ {:.3}'.format(orig_ff1, orig_ff1_std))
    print('=====================================\n')
    print('=====================================')
    print('Metrics of enhenced network:.......  ')
    print('en qqq: {:.3} ~ {:.3}'.format(en_qqq, en_qqq_std))
    print('en nmi: {:.3} ~ {:.3}'.format(en_nmi, en_nmi_std))
    print('en ari: {:.3} ~ {:.3}'.format(en_ari, en_ari_std))
    print('en f1 : {:.3} ~ {:.3}'.format(en_ff1, en_ff1_std))
    print('=====================================\n')
    print('Average run time consumption: {:.3} s'.format(np.average(time_consume)))



if __name__ == '__main__':
    main()
