# -*- coding: utf-8 -*-
# pylint: disable-all

import os
import sys
import shutil
import logging
import argparse
import random
import pickle

import yaml
from scipy.stats import stats

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import setproctitle
from torch.utils.data import Dataset, DataLoader

from aw_nas import utils
from aw_nas.common import get_search_space
from aw_nas.evaluator.arch_network import ArchNetwork


class NasBench201Dataset(Dataset):
    def __init__(self, data, minus=None, div=None):
        self.data = data
        self._len = len(self.data)
        self.minus = minus
        self.div = div

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        data = self.data[idx]
        if self.minus is not None:
            data = (data[0], data[1] - self.minus, data[2] - self.minus)
        if self.div is not None:
            data = (data[0], data[1] / self.div, data[2] / self.div)
        return data

def train_listwise(train_data, model, epoch, args, arch_network_type):
    objs = utils.AverageMeter()
    model.train()
    num_data = len(train_data)
    idx_list = np.arange(num_data)
    num_batches = getattr(
        args, "num_batch_per_epoch",
        int(num_data / (args.batch_size * args.list_length) * args.max_compare_ratio))
    logging.info("Number of batches: {:d}".format(num_batches))
    update_batch_n = getattr(args, "update_batch_n", 1)
    listwise_compare = getattr(args, "listwise_compare", False)
    if listwise_compare:
        assert args.list_length == 2 and update_batch_n == 1
    model.optimizer.zero_grad()
    for step in range(1, num_batches + 1):
        if getattr(args, "bs_replace", False):
            idxes = np.array([np.random.choice(idx_list, size=(args.list_length,), replace=False) for _ in range(args.batch_size)])
        else:
            idxes = np.random.choice(idx_list, size=(args.batch_size, args.list_length), replace=False)
        flat_idxes = idxes.reshape(-1)
        archs, accs, _ = zip(*[train_data[idx] for idx in flat_idxes])
        archs = np.array(archs).reshape((args.batch_size, args.list_length, -1))
        accs = np.array(accs).reshape((args.batch_size, args.list_length))
        # accs[np.arange(0, args.batch_size)[:, None], np.argsort(accs, axis=1)[:, ::-1]]
        if update_batch_n == 1:
            if listwise_compare:
                loss = model.update_compare(archs[:, 0, :], archs[:, 1, :],
                                            accs[:, 1] > accs[:, 0])
            else:
                loss = model.update_argsort(archs, np.argsort(accs, axis=1)[:, ::-1],
                                            first_n=getattr(args, "score_list_length", None))
        else:
            loss = model.update_argsort(archs, np.argsort(accs, axis=1)[:, ::-1],
                                        first_n=getattr(args, "score_list_length", None),
                                        accumulate_only=True)
            if step % update_batch_n == 0:
                model.optimizer.step()
                model.optimizer.zero_grad()
        if arch_network_type != "random_forest":
            objs.update(loss, args.batch_size)
        if step % args.report_freq == 0:
           logging.info("train {:03d} [{:03d}/{:03d}] {:.4f}".format(
                epoch, step, num_batches, objs.avg))
    return objs.avg

def train(train_loader, model, epoch, args, arch_network_type):
    objs = utils.AverageMeter()
    n_diff_pairs_meter = utils.AverageMeter()
    model.train()
    for step, (archs, f_accs, h_accs) in enumerate(train_loader):
        archs = np.array(archs)
        h_accs = np.array(h_accs)
        f_accs = np.array(f_accs)
        n = len(archs)
        if getattr(args, "use_half", False):
            accs = h_accs
        else:
            accs = f_accs
        if args.compare:
            if None in f_accs:
                # some archs only have half-time acc
                n_max_pairs = int(args.max_compare_ratio * n)
                n_max_inter_pairs = int(args.inter_pair_ratio * n_max_pairs)
                half_inds = np.array([ind for ind, acc in enumerate(accs) if acc is None])
                mask = np.zeros(n)
                mask[half_inds] = 1
                final_inds = np.where(1 - mask)[0]

                half_eche = h_accs[half_inds]
                final_eche = h_accs[final_inds]
                half_acc_diff = final_eche[:, None] - half_eche # (num_final, num_half)
                assert (half_acc_diff >= 0).all() # should be >0
                half_ex_thresh_inds = np.where(np.abs(half_acc_diff) > getattr(
                    args, "half_compare_threshold", 2 * args.compare_threshold))
                half_ex_thresh_num = len(half_ex_thresh_inds[0])
                if half_ex_thresh_num > n_max_inter_pairs:
                    # random choose
                    keep_inds = np.random.choice(np.arange(half_ex_thresh_num), n_max_inter_pairs, replace=False)
                    half_ex_thresh_inds = (half_ex_thresh_inds[0][keep_inds],
                                           half_ex_thresh_inds[1][keep_inds])
                inter_archs_1, inter_archs_2, inter_better_lst \
                    = archs[half_inds[half_ex_thresh_inds[1]]], archs[final_inds[half_ex_thresh_inds[0]]], \
                    (half_acc_diff > 0)[half_ex_thresh_inds]
                n_inter_pairs = len(inter_better_lst)

                # only use intra pairs in the final echelon
                n_intra_pairs = n_max_pairs - n_inter_pairs
                accs = np.array(accs)[final_inds]
                archs = archs[final_inds]
                acc_diff = np.array(accs)[:, None] - np.array(accs)
                acc_abs_diff_matrix = np.triu(np.abs(acc_diff), 1)
                ex_thresh_inds = np.where(acc_abs_diff_matrix > args.compare_threshold)
                ex_thresh_num = len(ex_thresh_inds[0])
                if ex_thresh_num > n_intra_pairs:
                    if args.choose_pair_criterion == "diff":
                        keep_inds = np.argpartition(acc_abs_diff_matrix[ex_thresh_inds], -n_intra_pairs)[-n_intra_pairs:]
                    elif args.choose_pair_criterion == "random":
                        keep_inds = np.random.choice(np.arange(ex_thresh_num), n_intra_pairs, replace=False)
                    ex_thresh_inds = (ex_thresh_inds[0][keep_inds], ex_thresh_inds[1][keep_inds])
                archs_1, archs_2, better_lst = archs[ex_thresh_inds[1]], archs[ex_thresh_inds[0]], (acc_diff > 0)[ex_thresh_inds]
                archs_1, archs_2, better_lst = np.concatenate((inter_archs_1, archs_1)),\
                                               np.concatenate((inter_archs_2, archs_2)),\
                                               np.concatenate((inter_better_lst, better_lst))
            else:
                if getattr(args, "compare_split", False):
                    n_pairs = len(archs) // 2
                    accs = np.array(accs)
                    acc_diff_lst = accs[n_pairs:2*n_pairs] - accs[:n_pairs]
                    keep_inds = np.where(np.abs(acc_diff_lst) > args.compare_threshold)[0]
                    better_lst = (np.array(accs[n_pairs:2*n_pairs] - accs[:n_pairs]) > 0)[keep_inds]
                    archs_1 = np.array(archs[:n_pairs])[keep_inds]
                    archs_2 = np.array(archs[n_pairs:2*n_pairs])[keep_inds]
                else:
                    n_max_pairs = int(args.max_compare_ratio * n)
                    acc_diff = np.array(accs)[:, None] - np.array(accs)
                    acc_abs_diff_matrix = np.triu(np.abs(acc_diff), 1)
                    ex_thresh_inds = np.where(acc_abs_diff_matrix > args.compare_threshold)
                    ex_thresh_num = len(ex_thresh_inds[0])
                    if ex_thresh_num > n_max_pairs:
                        if args.choose_pair_criterion == "diff":
                            keep_inds = np.argpartition(acc_abs_diff_matrix[ex_thresh_inds], -n_max_pairs)[-n_max_pairs:]
                        elif args.choose_pair_criterion == "random":
                            keep_inds = np.random.choice(np.arange(ex_thresh_num), n_max_pairs, replace=False)
                        ex_thresh_inds = (ex_thresh_inds[0][keep_inds], ex_thresh_inds[1][keep_inds])
                    archs_1, archs_2, better_lst = archs[ex_thresh_inds[1]], archs[ex_thresh_inds[0]], (acc_diff > 0)[ex_thresh_inds]
            n_diff_pairs = len(better_lst)
            n_diff_pairs_meter.update(float(n_diff_pairs))
            loss = model.update_compare(archs_1, archs_2, better_lst)
            objs.update(loss, n_diff_pairs)
        else:
            loss = model.update_predict(archs, accs)
            if arch_network_type != "random_forest":
                objs.update(loss, n)
        if step % args.report_freq == 0:
            n_pair_per_batch = (args.batch_size * (args.batch_size - 1)) // 2
            logging.info("train {:03d} [{:03d}/{:03d}] {:.4f}; {}".format(
                epoch, step, len(train_loader), objs.avg,
                "different pair ratio: {:.3f} ({:.1f}/{:3d})".format(
                    n_diff_pairs_meter.avg / n_pair_per_batch,
                    n_diff_pairs_meter.avg, n_pair_per_batch) if args.compare else ""))
    return objs.avg

def test_xp(true_scores, predict_scores):
    true_inds = np.argsort(true_scores)[::-1]
    true_scores = np.array(true_scores)
    reorder_true_scores = true_scores[true_inds]
    predict_scores = np.array(predict_scores)
    reorder_predict_scores = predict_scores[true_inds]
    ranks = np.argsort(reorder_predict_scores)[::-1]
    num_archs = len(ranks)
    # calculate precision at each point
    cur_inds = np.zeros(num_archs)
    passed_set = set()
    for i_rank, rank in enumerate(ranks):
        cur_inds[i_rank] = (cur_inds[i_rank - 1] if i_rank > 0 else 0) + \
                           int(i_rank in passed_set) + int(rank <= i_rank)
        passed_set.add(rank)
    patks = cur_inds / (np.arange(num_archs) + 1)
    THRESH = 100
    p_corrs = []
    for prec in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        k = np.where(patks[THRESH:] >= prec)[0][0] + THRESH
        arch_inds = ranks[:k][ranks[:k] < k]
        # stats.kendalltau(arch_inds, np.arange(len(arch_inds)))
        p_corrs.append((k, float(k)/num_archs, len(arch_inds), prec, stats.kendalltau(
            reorder_true_scores[arch_inds],
            reorder_predict_scores[arch_inds]).correlation))
    return p_corrs

def test_xk(true_scores, predict_scores):
    true_inds = np.argsort(true_scores)[::-1]
    true_scores = np.array(true_scores)
    reorder_true_scores = true_scores[true_inds]
    predict_scores = np.array(predict_scores)
    reorder_predict_scores = predict_scores[true_inds]
    ranks = np.argsort(reorder_predict_scores)[::-1]
    num_archs = len(ranks)
    patks = []
    for ratio in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
        k = int(num_archs * ratio)
        p = len(np.where(ranks[:k] < k)[0]) / float(k)
        arch_inds = ranks[:k][ranks[:k] < k]
        patks.append((k, ratio, len(arch_inds), p, stats.kendalltau(
            reorder_true_scores[arch_inds],
            reorder_predict_scores[arch_inds]).correlation))
    return patks

def pairwise_valid(val_loader, model, seed=None, funcs=[]):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    model.eval()
    true_accs = []
    all_archs = []
    for step, (archs, accs, _) in enumerate(val_loader):
        all_archs += list(archs)
        true_accs += list(accs)

    num_valid = len(true_accs)
    pseudo_scores = np.zeros(num_valid)
    # indexes, ranks = model.argsort_list(all_archs, batch_size=512)
    indexes = model.argsort_list(all_archs, batch_size=512)
    pseudo_scores[indexes] = np.arange(num_valid)
    # pseudo_scores[indexes] = ranks

    corr = stats.kendalltau(true_accs, pseudo_scores).correlation
    funcs_res = [func(true_accs, all_scores) for func in funcs]
    return corr, funcs_res

def valid(val_loader, model, args, funcs=[]):
    if not callable(getattr(model, "predict", None)):
        assert callable(getattr(model, "compare", None))
        corrs, funcs_res = zip(*[
            pairwise_valid(val_loader, model, pv_seed, funcs)
            for pv_seed in getattr(
                    args, "pairwise_valid_seeds", [1, 12, 123]
            )])
        funcs_res = np.mean(funcs_res, axis=0)
        logging.info("pairwise: {}".format(corrs))
        # return np.mean(corrs), true_accs, p_scores, funcs_res
        return np.mean(corrs), funcs_res

    model.eval()
    all_scores = []
    true_accs = []
    for step, (archs, accs, _) in enumerate(val_loader):
        scores = list(model.predict(archs).cpu().data.numpy())
        all_scores += scores
        true_accs += list(accs)

    if args.save_predict is not None:
        with open(args.save_predict, "wb") as wf:
            pickle.dump((true_accs, all_scores), wf)

    corr = stats.kendalltau(true_accs, all_scores).correlation
    funcs_res = [func(true_accs, all_scores) for func in funcs]
    return corr, funcs_res

def main(argv):
    parser = argparse.ArgumentParser(prog="train_nasbench201_pkl.py")
    parser.add_argument("cfg_file")
    parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--report_freq", default=200, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--train-dir", default=None, help="Save train log/results into TRAIN_DIR")
    parser.add_argument("--save-every", default=10, type=int)
    parser.add_argument("--test-only", default=False, action="store_true")
    parser.add_argument("--test-funcs", default=None, help="comma-separated list of test funcs")
    parser.add_argument("--load", default=None, help="Load comparator from disk.")
    parser.add_argument("--eval-only-last", default=None, type=int,
                        help=("for pairwise compartor, the evaluation is slow,"
                              " only evaluate in the final epochs"))
    parser.add_argument("--save-predict", default=None, help="Save the predict scores")
    parser.add_argument("--train-pkl", default="nasbench201_05.pkl", help="Training Datasets pickle")
    parser.add_argument("--valid-pkl", default="nasbench201_05_valid.pkl", help="Evaluate Datasets pickle")
    args = parser.parse_args(argv)

    setproctitle.setproctitle("python train_nasbench201_pkl.py config: {}; train_dir: {}; cwd: {}"\
                              .format(args.cfg_file, args.train_dir, os.getcwd()))

    # log
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt="%m/%d %I:%M:%S %p")

    if not args.test_only:
        assert args.train_dir is not None, "Must specificy `--train-dir` when training"
        # if training, setting up log file, backup config file
        if not os.path.exists(args.train_dir):
            os.makedirs(args.train_dir)
        log_file = os.path.join(args.train_dir, "train.log")
        logging.getLogger().addFile(log_file)

        # copy config file
        backup_cfg_file = os.path.join(args.train_dir, "config.yaml")
        shutil.copyfile(args.cfg_file, backup_cfg_file)
    else:
        backup_cfg_file = args.cfg_file

    # cuda
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        cudnn.enabled = True
        logging.info("GPU device = %d" % args.gpu)
    else:
        logging.info("no GPU available, use CPU!!")

    if args.seed is not None:
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    search_space = get_search_space("nasbench-201", load_nasbench=False)
    logging.info("Load pkl cache from nasbench201.pkl and nasbench201_valid.pkl")
    with open(args.train_pkl, "rb") as rf:
        train_data = pickle.load(rf)
    with open(args.valid_pkl, "rb") as rf:
        valid_data = pickle.load(rf)

    with open(backup_cfg_file, "r") as cfg_f:
        cfg = yaml.load(cfg_f)

    logging.info("Config: %s", cfg)

    arch_network_type = cfg.get("arch_network_type", "pointwise_comparator")
    model_cls = ArchNetwork.get_class_(arch_network_type)
    model = model_cls(search_space, **cfg.pop("arch_network_cfg"))
    if args.load is not None:
        logging.info("Load %s from %s", arch_network_type, args.load)
        model.load(args.load)
    model.to(device)

    args.__dict__.update(cfg)
    logging.info("Combined args: %s", args)

    # init nasbench data loaders
    if hasattr(args, "train_ratio") and args.train_ratio is not None:
        _num = len(train_data)
        train_data = train_data[:int(_num * args.train_ratio)]
        logging.info("Train dataset ratio: %.3f", args.train_ratio)
    num_train_archs = len(train_data)
    logging.info("Number of architectures: train: %d; valid: %d", num_train_archs, len(valid_data))
    # decide how many archs would only train to halftime
    if hasattr(args, "ignore_quantile") and args.ignore_quantile is not None:
        half_accs = [item[2] for item in train_data]
        if not args.compare or getattr(args, "ignore_halftime", False):
            # just ignore halftime archs
            full_inds = np.argsort(half_accs)[int(num_train_archs * args.ignore_quantile):]
            train_data = [train_data[ind] for ind in full_inds]
            logging.info("#Train architectures after ignore half-time %.2f bad archs: %d",
                         args.ignore_quantile, len(train_data))
        else:
            half_inds = np.argsort(half_accs)[:int(num_train_archs * args.ignore_quantile)]
            logging.info("#Architectures do not need to be trained to final: %.2f (%.2f %%)",
                         len(half_inds), args.ignore_quantile * 100)
            for ind in half_inds:
                train_data[ind] = (train_data[ind][0], None, train_data[ind][2])
    train_data = NasBench201Dataset(train_data, minus=cfg.get("dataset_minus", None),
                                    div=cfg.get("dataset_div", None))
    valid_data = NasBench201Dataset(valid_data, minus=cfg.get("dataset_minus", None),
                                    div=cfg.get("dataset_div", None))
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers,
        collate_fn=lambda items: list(zip(*items)))
    val_loader = DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers,
        collate_fn=lambda items: list(zip(*items)))

    # init test
    if not arch_network_type in {"pairwise_comparator", "random_forest"} or args.test_only:
        if args.test_funcs is not None:
            test_func_names = args.test_funcs.split(",")
        corr, func_res = valid(val_loader, model, args,
                               funcs=[globals()[func_name] for func_name in test_func_names]
                               if args.test_funcs is not None else [])
        if args.test_funcs is None:
            logging.info("INIT: kendall tau {:.4f}".format(corr))
        else:
            logging.info("INIT: kendall tau {:.4f};\n\t{}".format(
                corr,
                "\n\t".join(["{}: {}".format(name, res)
                             for name, res in zip(test_func_names, func_res)])))

    if args.test_only:
        return

    for i_epoch in range(1, args.epochs + 1):
        model.on_epoch_start(i_epoch)
        if getattr(args, "use_listwise", False):
            avg_loss = train_listwise(train_data, model, i_epoch, args, arch_network_type)
        else:
            avg_loss = train(train_loader, model, i_epoch, args, arch_network_type)
        logging.info("Train: Epoch {:3d}: train loss {:.4f}".format(i_epoch, avg_loss))
        if args.eval_only_last is None or (args.epochs - i_epoch < args.eval_only_last):
            corr, _ = valid(val_loader, model, args)
            logging.info("Valid: Epoch {:3d}: kendall tau {:.4f}".format(i_epoch, corr))
        if i_epoch % args.save_every == 0:
            save_path = os.path.join(args.train_dir, "{}.ckpt".format(i_epoch))
            model.save(save_path)
            logging.info("Epoch {:3d}: Save checkpoint to {}".format(i_epoch, save_path))


if __name__ == "__main__":
    main(sys.argv[1:])
