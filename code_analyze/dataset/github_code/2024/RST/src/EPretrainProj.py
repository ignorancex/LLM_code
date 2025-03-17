import gc
import os
import copy
import random
import math

from termcolor import colored
import numpy as np

from sklearn.utils.extmath import softmax
from scipy.special import softmax

from scipy.stats import entropy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, BatchSampler

from RST.src.ELib import ELib
from RST.src.ETweet import ETweet
from RST.src.ETweet import ELoadType
from RST.src.ELblConf import ELblConf
from RST.src.EBert import EBert, EBertCLSType
from RST.src.EBertUtils import EBertConfig, EInputBundle
from RST.src.EPretrainProjBaselines import EPretrainProjBaselines


class EPretrainCMD:
    none = 0
    bert_mine = 2

    @staticmethod
    def get(value):
        if value == 'bert_mine':
            return EPretrainCMD.bert_mine
        return EPretrainCMD.none


class DistInfo:

    def __init__(self, mean, var):
        self.mean = mean
        self.var = var
        ELib.PASS()

    def __str__(self):
        return 'mean: ' + str(self.mean) + '\tvar: ' + str(self.var)


class EPretrainProj:

    @staticmethod
    def __averaged_tempered_softmax(np_arr, ttt):
        dump = list()
        for cur_pred in np_arr:
            cur_pred_ts = torch.tensor(cur_pred)
            sft = F.softmax(cur_pred_ts / ttt).numpy()
            dump.append(sft)
        result = np.array(dump[0])
        for cur_pred in dump[1:]:
            result = result + cur_pred
        result = result / len(dump)
        result = result.tolist()
        return result

    @staticmethod
    def __sample_from_unlabeled(unlabeled_bundle, count, seed):
        random.seed(seed)
        count = min(count, len(unlabeled_bundle.tws))
        sample_set = set()
        while True:
            tw_ind = random.randint(0, len(unlabeled_bundle.tws) - 1)
            if unlabeled_bundle.tws[tw_ind] not in sample_set:
                sample_set.add(unlabeled_bundle.tws[tw_ind])
                if len(sample_set) >= count:
                    break
        drop_list = [tw for tw in unlabeled_bundle.tws if tw not in sample_set]
        EInputBundle.remove(unlabeled_bundle, drop_list)
        ELib.PASS()

    @staticmethod
    def __stratified_sample_from_bundle(bundle, lc, seed, count):
        count = min(len(bundle.tws), count)
        bundle_copy = copy.deepcopy(bundle)

        tws = ETweet.random_stratified_sample(bundle_copy.tws, lc, 1 - (count / len(bundle.tws)), seed)
        EInputBundle.remove(bundle_copy, tws)
        # ### custom sampling for experiments
        # tws = EPretrainProj.__test_sample(bundle_copy.tws, lc, 1 - (count / len(bundle.tws)), seed)
        # tws = ETweet.filter_by_tweets(bundle_copy.tws, tws)
        # EInputBundle.remove(bundle_copy, tws)
        return bundle_copy

    """================== START: Robust Self-Training Section =================="""

    @staticmethod
    def __select_candidates_using_inter_confidence(pseudo_bundle, unlabeled_bundle, logit_list, count,
                                                   do_softmax=True):
        result = list()
        if len(unlabeled_bundle.tws) == 0:
            return result
        """organize the results of the classifiers"""
        probs = list()
        logit_count = len(logit_list)
        for l_ind in range(logit_count):
            probs.append(list())
            for tw_ind, _ in enumerate(logit_list[l_ind]):
                if do_softmax:
                    probs[-1].append(softmax(logit_list[l_ind][tw_ind]))
                else:
                    probs[-1].append(logit_list[l_ind][tw_ind])
        """calculate the score for the documents"""
        record = list()
        for tw_ind, _ in enumerate(unlabeled_bundle.tws):
            # ent_1 = entropy(probs[0][tw_ind]) / math.log(pseudo_bundle.label_count)
            # ent_2 = entropy(probs[1][tw_ind]) / math.log(pseudo_bundle.label_count)
            # js_distance = distance.jensenshannon(probs[0][tw_ind], probs[1][tw_ind])
            # val = ((1 - ent_1) * (1 - ent_2) + 0.0001) / (js_distance + 0.0001)
            numerator = 1
            for l_ind in range(logit_count):
                numerator *= (1 - entropy(probs[l_ind][tw_ind]) / math.log(pseudo_bundle.label_count))
            denumerator_prob = 0
            denumerator_ent = 0
            for l_ind in range(logit_count):
                denumerator_prob += probs[l_ind][tw_ind]
                denumerator_ent += entropy(probs[l_ind][tw_ind])
            denumerator = entropy(denumerator_prob / logit_count) - denumerator_ent / logit_count
            val = (numerator + 0.0001) / (denumerator + 0.0001)
            record.append([tw_ind, val])
        record.sort(key=lambda item: -1 * item[1])
        """ collect the top pos/neg documents with the highest scores """
        if pseudo_bundle.label_count == 2:
            neg_all = 0
            for tw_ind, _ in enumerate(unlabeled_bundle.tws):
                if unlabeled_bundle.input_y[0][tw_ind] == 0:
                    neg_all += 1
            neg_tobe_added = math.floor(neg_all / len(unlabeled_bundle.tws) * count)
            pos_tobe_added = count - neg_tobe_added
            neg_added = 0
            pos_added = 0
            rec_ind = 0
            while True:
                cur_rec = record[rec_ind]
                if neg_added < neg_tobe_added and unlabeled_bundle.input_y[0][cur_rec[0]] == 0:
                    result.append(unlabeled_bundle.tws[cur_rec[0]])
                    neg_added += 1
                elif pos_added < pos_tobe_added and unlabeled_bundle.input_y[0][cur_rec[0]] == 1:
                    result.append(unlabeled_bundle.tws[cur_rec[0]])
                    pos_added += 1
                rec_ind += 1
                if len(result) >= count or rec_ind >= len(record):
                    break
        else:
            for rec_ind in range(count):
                result.append(unlabeled_bundle.tws[record[rec_ind][0]])
            ELib.PASS()
        return result

    @staticmethod
    def __print_tws_stat(bundle, tws, lc):
        counts = [{'all': 0, 'selected': 0} for ind in range(bundle.label_count)]
        id_set = set()
        for cur_tw in tws:
            id_set.add(cur_tw.Tweetid)
        for tw_ind, _ in enumerate(bundle.tws):
            lbl_new = bundle.input_y[0][tw_ind]
            counts[lbl_new]['all'] += 1
            if bundle.tws[tw_ind].Tweetid in id_set:
                counts[lbl_new]['selected'] += 1
        print(colored('>>> all-labels/added: ', 'red'), end='')
        for ind in range(bundle.label_count):
            print(colored('L{}: {}/{} | '.format(ind, counts[ind]['all'], counts[ind]['selected']), 'red'), end='')
        print()

    class PseudoLabelSampler(Sampler):

        def __init__(self, dataset, batch_size, num_samples):
            """len(dataset) must be equal to num_samples"""
            self.bundle = dataset.input_bundle
            self.batch_size = batch_size
            self.num_samples = num_samples
            if self.num_samples >= batch_size:
                spans = self.__get_itr_spans()
                ordered_batches = list()
                for ind in range(len(spans) - 1):
                    """ for the datapoints in each iteration create the batches """
                    sub_dataset_lbls = self.bundle.input_y_row[0][spans[ind]: spans[ind + 1]]
                    labels = [entry.index(max(entry)) for entry in sub_dataset_lbls]
                    lbl_values = set(labels)
                    class_count = dict()
                    for cur_lbl in lbl_values:
                        class_count[cur_lbl] = labels.count(cur_lbl)
                    cur_sample_weights = [(1 / len(class_count)) / class_count[entry] for entry in labels]
                    sampler = WeightedRandomSampler(cur_sample_weights, len(cur_sample_weights), True)
                    batches = list(BatchSampler(sampler, batch_size=self.batch_size, drop_last=True))
                    """ map the local indices to the actual index in the bundle """
                    batches_adjusted = [[cur_b_ind + spans[ind] for cur_b_ind in cur_b] for cur_b in batches]
                    ordered_batches.extend(batches_adjusted)
                """ reverse the order """
                self.result = copy.deepcopy(ordered_batches)
                self.result.reverse()
            ELib.PASS()

        def __get_itr_spans(self):
            """ find the boundary (index) of each iteration """
            hard_spans = []
            for ind, _ in enumerate(self.bundle.input_meta):
                if ind == 0 or self.bundle.input_meta[ind - 1] != self.bundle.input_meta[ind]:
                    hard_spans.append(ind)
            """ revise the boundaries and make them dividable by the batch_size """
            soft_spans = []
            for ind, itr_bound in enumerate(hard_spans):
                sub_dataset_bound  = (itr_bound // self.batch_size) * self.batch_size
                if ind == 0 or soft_spans[-1] != sub_dataset_bound:
                    soft_spans.append(sub_dataset_bound)
            if len(soft_spans) == 1:
                soft_spans.append(len(self.bundle.input_meta))
            else:
                soft_spans[-1] = len(self.bundle.input_meta)
            return soft_spans

        def __iter__(self):
            for cur_b in self.result:
                yield cur_b
            ELib.PASS()

        def __len__(self):
            return self.num_samples // self.batch_size

    @staticmethod
    def __run_training_with_unlabeled_mine(this_config, lc, query, this_train_bundle, valid_bundle, test_bundle,
                                           this_unlabeled_bundle):
        temperature = 2 # default is 2
        step_sample_ratio = 0.2 # 0.1 (for full labeled data) # 0.2 (otherwise)
        split_sample_ratio = 0.7 # default is 0.7
        distill_weight = 0.3 # default is 0.3
        cls_num = 2 # default is 2

        pseudo_bundle = EInputBundle(this_train_bundle.label_count, this_train_bundle.task_list, list(),
                                     [list() for _ in this_train_bundle.task_list],
                                     [list() for _ in this_train_bundle.task_list], list(), list(),
                                     list(), list())
        unlabeled_bundle = copy.deepcopy(this_unlabeled_bundle)
        config = copy.deepcopy(this_config)
        config.cls_type = EBertCLSType.simple
        cur_step = 1
        step_count = EPretrainProjBaselines.calculate_self_train_steps(len(this_train_bundle.tws),
                                                                       len(unlabeled_bundle.tws), step_sample_ratio)
        while True:
            print('step: ' + str(cur_step) + '/' + str(step_count) +
                  ' train size: ' + str(len(this_train_bundle.tws)) +
                  ' pseudo-train size: ' + str(len(pseudo_bundle.tws)) +
                  ' unlabeled size: ' + str(len(unlabeled_bundle.tws)) +
                  '\t\t' + ELib.get_time())
            un_lbl_list, un_logit_list = list(), list()
            for cls_ind in range(cls_num):
                """constructing the train sets"""
                train_bundle_itr = copy.deepcopy(this_train_bundle)
                remove_tws_itr = ETweet.random_stratified_sample(this_train_bundle.tws, lc, 1 - split_sample_ratio,
                                                               config.seed + cur_step * (cls_ind + 1) * 1234)
                EInputBundle.remove(train_bundle_itr, remove_tws_itr)
                pseudo_bundle_itr = copy.deepcopy(pseudo_bundle)
                remove_ps_tws_itr = ETweet.random_stratified_sample(pseudo_bundle.tws, lc, 1 - split_sample_ratio,
                                                                  config.seed + cur_step * (cls_ind + 1) * 6782)
                EInputBundle.remove(pseudo_bundle_itr, remove_ps_tws_itr)
                """training the classifier"""
                print('>>> training cls-{} using pseudo-train set'.format(cls_ind + 1))
                config.seed = this_config.seed + cur_step * (cls_ind + 1) * 675
                cls_itr = EBert(config)
                config.epoch_count = 2
                config.train_by_log_softmax = True
                config.training_log_softmax_weight = 1
                config.training_softmax_temperature = temperature
                cls_itr.custom_batch_sampler_class = EPretrainProj.PseudoLabelSampler
                cls_itr.train([pseudo_bundle_itr])
                print('>>> training cls-{} using train set'.format(cls_ind + 1))
                config.epoch_count = 3
                if len(pseudo_bundle_itr.tws) > 0:
                    tr_lbl_itr, tr_logit_itr, _, _ = cls_itr.test(train_bundle_itr, False, print_perf=False)
                    train_bundle_itr.input_y_row = [EPretrainProj.__averaged_tempered_softmax([tr_logit_itr], temperature)]
                    config.train_by_log_softmax = True
                    config.training_log_softmax_weight = distill_weight
                    config.training_softmax_temperature = temperature
                else:
                    config.train_by_log_softmax = False
                cls_itr.custom_batch_sampler_class = None
                cls_itr.train([train_bundle_itr])
                print('>>> labeling unlabeled data')
                un_lbl_itr, un_logit_itr, _, _ = cls_itr.test(unlabeled_bundle, False, print_perf=False)
                un_lbl_list.append(un_lbl_itr)
                un_logit_list.append(un_logit_itr)
                del cls_itr
            """update the datasets"""
            for ind, _ in enumerate(unlabeled_bundle.tws):
                unlabeled_bundle.input_meta[ind] = cur_step
            EPretrainProjBaselines.update_label_info(unlabeled_bundle, un_lbl_list, un_logit_list, True, temperature)
            train_size = len(this_train_bundle.tws) + len(pseudo_bundle.tws)
            top_count = min(len(unlabeled_bundle.tws), math.ceil(step_sample_ratio * train_size))
            tops = EPretrainProj.__select_candidates_using_inter_confidence(pseudo_bundle, unlabeled_bundle,
                                                                            un_logit_list, top_count)
            EPretrainProj.__print_tws_stat(unlabeled_bundle, tops, lc)
            EInputBundle.append(pseudo_bundle, unlabeled_bundle, tops)
            EInputBundle.remove(unlabeled_bundle, tops)
            """test"""
            print('>>> training iteration cls using pseudo-train set')
            config.seed = this_config.seed + cur_step * 2357
            cls = EBert(config)
            config.epoch_count = 2
            config.train_by_log_softmax = True
            config.training_log_softmax_weight = 1
            config.training_softmax_temperature = temperature
            cls.custom_batch_sampler_class = EPretrainProj.PseudoLabelSampler
            cls.train([pseudo_bundle])
            print('>>> training iteration cls using train set')
            config.epoch_count = 3
            if len(pseudo_bundle.tws) > 0:
                tr_lbl, tr_logit, _, _ = cls.test(this_train_bundle, False, print_perf=False)
                this_train_bundle.input_y_row = [EPretrainProj.__averaged_tempered_softmax([tr_logit], temperature)]
                config.train_by_log_softmax = True
                config.training_log_softmax_weight = distill_weight
                config.training_softmax_temperature = temperature
            else:
                config.train_by_log_softmax = False
            cls.custom_batch_sampler_class = None
            cls.train([this_train_bundle])
            print('>>> labeling test data')
            lbl, logit, _, perf = cls.test(test_bundle, False)
            print()
            del cls
            gc.collect()
            torch.cuda.empty_cache()
            cur_step += 1
            if len(unlabeled_bundle.tws) == 0:
                break
            ELib.PASS()
        return perf

    """================== END: Robust Self-Training Section =================="""

    @staticmethod
    def __run_training_with_unlabeled(config, lc, query, this_train_bundle, valid_bundle, test_bundle, unlabeled_bundle,
                                      unlabeled_sample):
        unlabeled_count = unlabeled_sample # 10000
        EPretrainProj.__sample_from_unlabeled(unlabeled_bundle, unlabeled_count, config.seed)
        result = None

        """mine"""
        result = EPretrainProj.__run_training_with_unlabeled_mine(config, lc, query, this_train_bundle, valid_bundle,
                                                                  test_bundle, unlabeled_bundle)
        return result

    @staticmethod
    def run(cmd, per_query, train_path, valid_path_nullable, test_path_nullable,
            unlabeled_path_nullable, model_path, model_path_2,
            lm_model_path, t_lbl_path_1, t_lbl_path_2, output_dir,
            device, device_2, seed, train_sample, unlabeled_sample):
        cmd = EPretrainCMD.get(cmd)
        lc = ELblConf.get_regular_lblconfig()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        tws_temp_load = ETweet.load(train_path, ELoadType.none)
        label_count = ETweet.get_label_count(lc, tws_temp_load)
        queries = [None]
        if per_query:
            queries = ETweet.get_queries(tws_temp_load)

        result = list()
        for q_ind, cur_query in enumerate(queries):
            if cur_query is not None:
                print('>>>>>>>> "' + cur_query + '" began')
            # if q_ind < 1:
            #     continue
            if cmd == EPretrainCMD.bert_mine: ## pretraining code
                remove_unlabeled_test_tweets = False    ################ set to False for ADR codalab output
                config = EBertConfig.get_config(cmd, label_count, EBertCLSType.none, model_path, model_path_2,
                                                      lm_model_path, t_lbl_path_1, t_lbl_path_2, output_dir, 5,
                                                      device, device_2, seed, cur_query)
                train_bundle, valid_bundle, test_bundle, unlabeled_bundle = EInputBundle.get_data(config.label_count,
                    lc, train_path, valid_path_nullable, test_path_nullable, unlabeled_path_nullable, cur_query,
                    remove_unlabeled_test_tweets, load_etokens=False, max_set_length=1000) # , max_set_length=100

                train_bundle_semi = EPretrainProj.__stratified_sample_from_bundle(train_bundle, lc, config.seed, train_sample)

                """end of depricated section"""
                cur_perf = EPretrainProj.__run_training_with_unlabeled(config, lc, cur_query, train_bundle_semi,
                                                                       valid_bundle, test_bundle, unlabeled_bundle,
                                                                       unlabeled_sample)
                result.append(cur_perf)
        result = [entry for entry in result if entry is not None]
        if len(result) == 0:
            return None
        else:
            cur_perf = ELib.print_iteration_results(result)
            return cur_perf
        ELib.PASS()




