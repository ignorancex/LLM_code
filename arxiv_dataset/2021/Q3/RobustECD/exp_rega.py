#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: jjzhou012
@contact: jjzhou012@163.com
@file: exp_revsel.py
@time: 2020/7/15 21:34
@desc:  python
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
from source.REGA import robustECD_GA


def get_para():
    parser = argparse.ArgumentParser(description='manual to this script')
    # file para
    parser.add_argument('--bmname', help="name of benchmark dataset", required=False, type=str, default='karate')
    parser.add_argument('--cdm', help="community detection method", required=True, type=str, default=None)
    parser.add_argument('--popsize', help="size of population", required=False, type=int, default=120)
    parser.add_argument('--pc', '-c', help="crossover prob", required=False, type=float, default=0.8)
    parser.add_argument('--pm', '-m', help="mutation prob", required=False, type=float, default=0.2)
    parser.add_argument('--recombine_rate', '-re', help="the rate of recombine", required=False, type=float, default=0.2)
    parser.add_argument('--iter', help="the number of iteration", required=True, type=int, default=10)
    parser.add_argument('--sample_add_ratio', '-aR', help="the sample ratio of add edge", required=True, type=float, default=0.1)
    parser.add_argument('--sample_del_ratio', '-dR', help="the sample ratio of del edge", required=True, type=float, default=0.1)
    parser.add_argument('--varlen', '-r', help="unfixed length of dna", required=False, type=int, default=1)
    args = parser.parse_args()
    return args


def main():
    args = get_para()
    print(args)
    bmname = args.bmname
    file = 'data/{}/{}.gml'.format(bmname, bmname)
    cdm = args.cdm
    popsize = args.popsize
    pc = args.pc
    pm = args.pm
    recombine_rate = args.recombine_rate
    iter = args.iter
    sample_add_ratio = args.sample_add_ratio
    sample_del_ratio = args.sample_del_ratio
    varlen = bool(args.varlen)

    result_dir = 'log/ga/{}/{}/Ra-{}-Rd{}/'.format(bmname, cdm, sample_add_ratio, sample_del_ratio)
    if not os.path.exists(result_dir): os.makedirs(result_dir)

    print('experiment setup.')
    print('bmname', bmname)
    print('cdm', cdm)
    print('popsize', popsize)
    print('pc', pc)
    print('pm', pm)
    print('recombine_rate', recombine_rate)
    print('iter', iter)
    print('sample_add_ratio', sample_add_ratio)
    print('sample_del_ratio', sample_del_ratio)
    print('varlen', varlen)
    print('=====================================\n')

    print('get original result...')
    g = nx.read_gml(file, label='id')
    labels_true = list(nx.get_node_attributes(g, 'commID').values())

    G = IG.Read_GML(file)
    # this setting is no randomness

    experiment_time = 50

    orig_qqq_list, orig_nmi_list, orig_ari_list, orig_ff1_list = [], [], [], []
    for i in tqdm(range(experiment_time)):
        if cdm not in ['scd', 'lpnx', 'gemsec', 'n2v_km']:
            community = community_method_dict[cdm].__wrapped__(labels_true, input=G)
            orig_qqq, orig_nmi, orig_ari, orig_ff1 = evaluate_results(G, community, labels_true)
        else:
            community = community_method_dict[cdm].__wrapped__(labels_true, input=g)
            orig_qqq, orig_nmi, orig_ari, orig_ff1 = evaluate_results(g, community, labels_true)
        orig_qqq_list.append(orig_qqq)
        orig_nmi_list.append(orig_nmi)
        orig_ari_list.append(orig_ari)
        orig_ff1_list.append(orig_ff1)

    orig_qqq, orig_nmi, orig_ari, orig_ff1 = np.average(orig_qqq_list), np.average(orig_nmi_list), np.average(orig_ari_list), np.average(
        orig_ff1_list)
    orig_qqq_std, orig_nmi_std, orig_ari_std, orig_ff1_std = np.std(orig_qqq_list), np.std(orig_nmi_list), np.std(orig_ari_list), np.std(
        orig_ff1_list)

    print('=====================================')
    print('Metrics of original network:.......  ')
    print('orig qqq: {:.3} ~ {:.3}'.format(orig_qqq, orig_qqq_std))
    print('orig nmi: {:.3} ~ {:.3}'.format(orig_nmi, orig_nmi_std))
    print('orig_ari: {:.3} ~ {:.3}'.format(orig_ari, orig_ari_std))
    print('orig_f1 : {:.3} ~ {:.3}'.format(orig_ff1, orig_ff1_std))
    print('=====================================\n')

    en_qqq_list, en_nmi_list, en_ari_list, en_ff1_list = [], [], [], []
    time_consume = []

    log_dir = 'log/ga/CSV_log/{}/{}/Ra-{}-Rd{}/'.format(bmname, cdm, sample_add_ratio, sample_del_ratio)
    if not os.path.exists(log_dir): os.makedirs(log_dir)

    for i in tqdm(range(experiment_time)):
        output_file = log_dir + 'log.csv'
        t1 = time.time()
        # community detection
        en_community = robustECD_GA(bmname=bmname,
                                    cdm=cdm,
                                    popsize=popsize,
                                    Pc=pc, Pm=pm,
                                    recombine_rate=recombine_rate,
                                    sample_del_ratio=sample_del_ratio,
                                    sample_add_ratio=sample_add_ratio,
                                    iterNum=iter,
                                    varlen=varlen,
                                    output_file=output_file).run()
        t2 = time.time()
        time_consume.append(t2 - t1)

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

    save_result = result_dir + '{}_{}_Ra-{}_Rb-{}.txt'.format(bmname, cdm, sample_add_ratio, sample_add_ratio)
    with open(save_result, 'w', encoding='utf-8') as f:
        f.writelines('Metrics of original network:.......  \n')
        f.writelines('orig qqq: {:.3} ~ {:.3}\n'.format(orig_qqq, orig_qqq_std))
        f.writelines('orig nmi: {:.3} ~ {:.3}\n'.format(orig_nmi, orig_nmi_std))
        f.writelines('orig_ari: {:.3} ~ {:.3}\n'.format(orig_ari, orig_ari_std))
        f.writelines('orig_f1 : {:.3} ~ {:.3}\n'.format(orig_ff1, orig_ff1_std))
        f.writelines('Metrics of enhenced network:.......  \n')
        f.writelines('en qqq: {:.3} ~ {:.3}\n'.format(en_qqq, en_qqq_std))
        f.writelines('en nmi: {:.3} ~ {:.3}\n'.format(en_nmi, en_nmi_std))
        f.writelines('en ari: {:.3} ~ {:.3}\n'.format(en_ari, en_ari_std))
        f.writelines('en f1 : {:.3} ~ {:.3}\n'.format(en_ff1, en_ff1_std))
        f.writelines('Average run time consumption: {:.3} s\n'.format(np.average(time_consume)))
    f.close()


if __name__ == '__main__':
    main()
