import torch
from scipy.stats import norm, binom_test
from scipy.special import comb
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint

from torch.distributions.bernoulli import Bernoulli
from tqdm import tqdm
from utils import normalize, sparse_mx_to_torch_sparse_tensor
import scipy.sparse as sp
import numba as nb
from numba import cuda
import time
import torch.nn.functional as F
import threading


from joblib import Parallel, delayed


import time
import random
import torch.multiprocessing as mp
from torch.multiprocessing import Manager




BASE = 100


@cuda.jit
def _tools_accelerated(adj, adj_noise, idx, mask):

    rand_inputs = torch.randint_like(adj[idx], low=0, high=2, device='cuda').squeeze().int()

    adj_noise[idx] = adj[idx] * mask + rand_inputs * (1 - mask)

    # print('#nnz:', (adj_noise[idx] - adj[idx]).sum())
    adj_noise[:, idx] = adj_noise[idx]

    return adj_noise



def accuracy_new(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)



def para(m, adj, idx, adj_noise, base_classifier, fea):

    global counts


    mask = m.sample(adj[idx].shape).squeeze(-1).int()

    rand_inputs = torch.randint_like(adj[idx], low=0, high=2, device='cuda').squeeze().int()

    adj_noise[idx] = adj[idx] * mask + rand_inputs * (1 - mask)

    adj_noise[:, idx] = adj_noise[idx]
    adj_noise_norm = tensor_norm(adj_noise)
    adj_noise_norm = adj_noise_norm.to_sparse()

    predictions = base_classifier(fea, adj_noise_norm).argmax(1)
    prediction = predictions[idx]
    counts[prediction.cpu().numpy()] += 1

    print(counts)

    return 0



# normalize(adj_noise.cpu().numpy() + sp.eye(adj_noise.cpu().numpy().shape[0]))
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv,0)
    mx = r_mat_inv.dot(mx)
    return mx

def tensor_norm(x):

    x = torch.diag(torch.ones(x.shape[0]).float().cuda()) + x

    row_sum = x.sum(dim=1)
    the_one = torch.tensor(1).float().cuda()
    z1 = torch.where(row_sum > float(0), row_sum, the_one)
    z2 = z1 ** (-1)
    row_inv = torch.diag(z2)


    return torch.matmul(row_inv, x)



def fair_judgement(pred, labels, sens, threshold, threshold_flag='parity'):
    idx_s0 = (sens==0).cuda()
    idx_s1 = (sens==1).cuda()

    idx_s0_y1 = idx_s0 * (labels==1)
    idx_s1_y1 = idx_s1 * (labels==1)
    parity = torch.abs(pred[idx_s0].sum() / idx_s0.sum() - pred[idx_s1].sum() / idx_s1.sum())
    equality = torch.abs(pred[idx_s0_y1].sum() / idx_s0_y1.sum() - pred[idx_s1_y1].sum() / idx_s1_y1.sum())

    if threshold_flag == 'parity':

        return parity.item() < threshold, parity.item()
    elif threshold_flag == 'equality':

        return equality.item() < threshold, equality.item()
    else:
        assert 0 == 1



def gaussianCDFInverse(p):
    return torch.sqrt(torch.tensor(2.0)) * torch.erfinv(2.0 * torch.tensor(p).float() - 1.0)






def multi_threading_tst(result_queue, feature, adj_noise_sparse, gaussian_std, base_classifier, idx, labels, sens, threshold, threshold_flag, repeats, alpha_x=0.2):


    fair_results = []
    for _ in range(repeats):

        predictions = base_classifier(feature + gaussian_std * torch.randn(feature.shape[0], feature.shape[1]).cuda(), adj_noise_sparse)
        predictions = predictions.argmax(1)

        fairness_classification = fair_judgement(predictions[idx], labels[idx], sens[idx], threshold, threshold_flag)
        fair_results.append(fairness_classification)

    result_queue.put(sum(fair_results))

    return fair_results




#  NNNew
def sample_noise_gau_multi(feature, adj_noise_sparse, gaussian_std, base_classifier, idx, labels, sens, threshold, threshold_flag, num_x, alpha_x=0.2):



    pool_batch = int(10)

    result_queue = mp.Queue()
    model = base_classifier
    model.share_memory()
    processes = []
    for i in range(pool_batch):
        p = mp.Process(target=multi_threading_tst, args=(result_queue, feature, adj_noise_sparse._indices(), gaussian_std, base_classifier, idx, labels, sens, threshold, threshold_flag, int(num_x/pool_batch),))
        p.start()
        processes.append(p)

    results = [int(result_queue.get()) for _ in range(pool_batch)]

    for p in processes:
        p.join()


    counts_inner_list = torch.FloatTensor(list([num_x - sum(results), sum(results)]))
    print("CHECK:" + str(counts_inner_list))

    NA = counts_inner_list.max().cpu().item()
    pa_bar_from_x = proportion_confint(NA, num_x, alpha=2 * alpha_x, method="beta")[0]
    radius_x = 0.5 * gaussian_std * (gaussianCDFInverse(pa_bar_from_x) - gaussianCDFInverse(1 - pa_bar_from_x))


    # if random.random() < 0.2:
    print("CHECK  pa_bar_from_x  :  " + str(pa_bar_from_x))
    print("CHECK  radius_x  :  " + str(radius_x))

    return sum(results), radius_x, pa_bar_from_x








#  NNNew
def sample_noise_gau(feature, adj_noise_sparse, gaussian_std, base_classifier, idx, labels, sens, threshold, threshold_flag, num_x, idx_vul, alpha_x=0.3):

    counts_inner = 0

    fairness_level_best = 1.0

    corresponding_acc = -1.0


    for _ in tqdm(range(num_x)):
        feature_with_noise = feature.clone().detach()
        feature_with_noise[idx_vul, :] = feature[idx_vul, :] + gaussian_std * torch.randn(feature.shape[0], feature.shape[1]).cuda()[idx_vul, :]
        predictions = base_classifier(feature_with_noise, adj_noise_sparse._indices())

        # nll_loss
        pred_labels = predictions.argmax(1)

        # print(pred_labels.sum())

        # print("************  CHECKING  **************")
        # print(pred_labels[idx])
        fairness_classification, fairness_value = fair_judgement(pred_labels[idx], labels[idx], sens[idx], threshold, threshold_flag)

        # print("************  CHECKING ENDS  **************")
        counts_inner += int(fairness_classification)

        if fairness_value < fairness_level_best:
            fairness_level_best = fairness_value
            corresponding_acc = accuracy_new(predictions[idx], labels[idx])
    
    counts_inner_list = [num_x - counts_inner, counts_inner]
    NA = max(counts_inner_list)

    pa_bar_from_x = proportion_confint(NA, num_x, alpha=2 * alpha_x, method="beta")[0]

    radius_x = 0.5 * gaussian_std * (gaussianCDFInverse(pa_bar_from_x) - gaussianCDFInverse(1 - pa_bar_from_x))

    print("Check 0 NA : " + str(NA))
    print("Check 1 pa_bar_from_x : " + str(pa_bar_from_x))
    print("Check 2 radius_x : " + str(radius_x))
    print("Check 3 fairness_level_best : " + str(fairness_level_best))
    print("Check 4 corresponding_acc : " + str(corresponding_acc))
    print("Check 5 vul num : " + str(idx_vul.shape))

    return counts_inner, radius_x, pa_bar_from_x, fairness_level_best, corresponding_acc




class Smooth_Ber(object):

    """A smoothed classifier g """

    # to abstain, Smooth_Ber returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, dim: int, prob: float, adj: torch.tensor, fea: torch.tensor, cuda: bool, labels):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param prob: the probability binary vector keeps the original value  "added: i.e., beta"
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.dim = dim
        self.prob = prob
        self.adj = adj
        self.fea = fea 
        self.cuda = cuda
        self.labels = labels
        if self.cuda:
            self.m = Bernoulli(1 - torch.tensor([self.prob]).cuda())
        else:
            self.m = Bernoulli(1 - torch.tensor([self.prob]))

        self.eye_mx = torch.eye(adj.shape[0]).to_sparse().cuda().int()

        # binary setting
        label_0 = (1 - self.labels).sum()
        label_1 = self.labels.sum()

        self.indices = None

        print(label_0)
        print(label_1)
        print(torch.nonzero(1 - self.labels).squeeze().cpu().numpy())

        if label_0 > label_1:
            self.indices = np.random.choice(torch.nonzero(1 - self.labels).squeeze().cpu().numpy().tolist(), size=int(label_0 - label_1), replace=False)
        else:
            self.indices = np.random.choice(torch.nonzero(self.labels).squeeze().cpu().numpy().tolist(), size=int(label_1 - label_0), replace=False)

        self.the_one = torch.tensor(1).int().cuda()






    def cal_prob(self, u: int, v: int):
        
        shape = list(self.adj.size())
        p_orig = np.power(int(self.prob * BASE), shape[0]-u) * np.power(int((1-self.prob) * BASE), u)
        p_pertub = np.power(int(self.prob * BASE), shape[0]-v) * np.power(int((1-self.prob) * BASE), v)
        return p_orig, p_pertub


    def sort_ratio(self, K: int):
        
        ratio_list = list()
        for u in range(K+1):
            for v in list(reversed(range(u, K+1))):
                if u + v >= K and np.mod(u + v - K, 2) == 0:
                    ratio_list.append((v-u,u,v))
        sorted_ratio = sorted(ratio_list, key=lambda tup: tup[0], reverse=True)
        return sorted_ratio


    def cal_L(self, K: int, u: int, v: int):
        
        shape = list(self.adj.size())
        i = int((u + v - K) / 2)
        return comb(shape[0]-K, i) * comb(K, u-i)



    def certify_Ber(self, idx: int, n0: int, n: int, alpha: float, num_x, gaussian_std, labels, sens, threshold, threshold_flag, vul_ratio, attacked=0):



        idx_vul = idx if vul_ratio >= 1 else idx[:int(idx.shape[0] * vul_ratio)]


        # only for the attacked data
        if attacked == 1:
            idx_vul = torch.tensor([ 95, 195, 134,  30,   5, 228,  33, 219,  19, 196, 138, 123]).cuda()




        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection, _, _, _ = self._sample_noise_ber(idx, n0, n0, gaussian_std, labels, sens, threshold, threshold_flag, idx_vul)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation, Rx, best_fair_value, cor_acc_value = self._sample_noise_ber(idx, n, num_x, gaussian_std, labels, sens, threshold, threshold_flag, idx_vul)
        
        
        if Rx <= 0.0:
            return Smooth_Ber.ABSTAIN, 0.0, -1, -1, -1
        
        
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()

        pABar = self._lower_confidence_bound(nA, n, alpha)

        if pABar < 0.5 or Rx <= 0.0:
            return Smooth_Ber.ABSTAIN, 0.0, -1, -1, -1
            # return cAHat, 0.0
        else:
            return cAHat, pABar, Rx, best_fair_value, cor_acc_value
            



    def _sample_noise_ber(self, idx: int, num: int, num_x, gaussian_std, labels, sens, threshold, threshold_flag, idx_vul):


        if self.cuda:
            adj = self.adj.to_dense().int().clone().detach().cuda()
            adj_noise = adj.clone().detach().cuda()
        else:
            adj = self.adj.to_dense().int().clone().detach()
            adj_noise = adj.clone().detach()

        the_one = torch.tensor(1).int().cuda()
        with torch.no_grad():
            counts = torch.zeros(2).int().cuda()

            Rx = np.inf
            best_fair = []
            cor_acc = []

            for _ in tqdm(range(num)):

                random_noise = self.m.sample(adj[idx_vul].shape).squeeze(-1).int()

                adj_noise[idx_vul] = torch.logical_xor(adj[idx_vul], random_noise).int()

                adj_noise[:, idx_vul] = torch.transpose(adj_noise[idx_vul], 0, 1)
                adj_noise_sparse = adj_noise.to_sparse()

                counts_inner, radius_x, pa_bar_from_x = None, None, None


                counts_inner, radius_x, _, fairness_level_best, corresponding_acc = sample_noise_gau(self.fea, adj_noise_sparse, gaussian_std, self.base_classifier, idx, labels, sens, threshold,
                                                                threshold_flag, num_x, idx_vul)

                if radius_x < Rx:
                    Rx = radius_x
                    if Rx < 0:
                        print("Inner Certification Fail.")
                        return counts.cpu().numpy(), Rx, 1e3, 0
                

                fairness_certify_x = None

                if counts_inner > int(num_x/2):
                    fairness_certify_x = 1
                    # if the inner certification has already obtained a certified 0, then we do not use its corresponding fairness level, since in this case we have a p>0.5 for a fairness level >= \eta
                    best_fair.append(fairness_level_best)
                    cor_acc.append(corresponding_acc)

                else:
                    fairness_certify_x = 0
                    # if the inner certification has already obtained a certified 0, then we do not use its corresponding fairness level, since in this case we have a p>0.5 for a fairness level >= \eta
                    best_fair.append(1e3)
                    cor_acc.append(corresponding_acc)



                counts[int(fairness_certify_x)] += the_one

                print(counts)
                print(" **** Check 0 min(best_fair) : " + str(min(best_fair)))
                print(" **** Check 1 cor_acc : " + str(cor_acc[np.array(best_fair).argmin()]))
                print(" **** Check 2 Rx : " + str(Rx))

            return counts.cpu().numpy(), Rx, min(best_fair), cor_acc[np.array(best_fair).argmin()]



    def _count_arr(self, arr: np.ndarray, length: int):
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float):
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]