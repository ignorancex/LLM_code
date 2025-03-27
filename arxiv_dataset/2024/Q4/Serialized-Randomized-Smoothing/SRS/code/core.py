import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm, binom_test
import os
import numpy as np
from math import ceil, floor, sqrt
from statsmodels.stats.proportion import proportion_confint

from implicit_model.model import mdeq
from datasets import get_normalize_layer

class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, sigma: float, args=None):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma
        self.args = args
        self.conserve_rate = 1.
        self.alpha = 0
        self.stat = []

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.

        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        if self.args.conf_drop:
            self.alpha = alpha/2
        else:
            self.alpha = alpha
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        nA = nA * (self.conserve_rate)
        n_total = n * (self.conserve_rate)

        pABar = self._lower_confidence_bound(nA, n_total, self.alpha)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]


    def fit_function(self, x: torch.tensor, t: int = 1, num: int = 100):
        """
        This function return an approximated function of randomized smoothing.
        Use this differentiable function to calculate the gradient of the smoothed classifier.

        args:
        k: the hyperparameter controls the increasing speed of the soft function.
        num: the number of samplings to estimate gradient.
        """
        self.base_classifier.eval()
        conf = self._sample_confidence(x, num, batch_size=64)
        soft_conf = F.softmax(t * conf, dim=1)
        fit_value = torch.mean(soft_conf, dim=0).view(1, -1)

        return fit_value


    def _sample_confidence(self, x: torch.tensor, num: int, batch_size):
        """
        Sample the base classifier's confidence under noisy corruptions of the input x.
        This is for calculating gradient, so it is with grad.

        args:
        num: the number of samplings. For the reason of GPU Memory limit, it should not be too large.
        # Can be revised as SGD. It means we can cumulate the gradient in several batches.
        """
        conf = []
        for _ in range(ceil(num / batch_size)):
            this_batch_size = min(batch_size, num)
            num -= this_batch_size

            batch = x.repeat((this_batch_size, 1, 1, 1))
            noise = torch.randn_like(batch, device='cuda') * self.sigma
            # this_conf = self.base_classifier(batch + noise)
            this_conf , _, _, out = self.base_classifier(batch + noise, attack=True, train_step=-1, compute_jac_loss=False,
                                                        start=None, detail=None,
                                                        f_thresh=None, solver=None)

            if type(this_conf) is tuple:
                this_conf = this_conf[0]
            conf.append(this_conf)

        conf = torch.cat(conf, dim=0)

        return conf


    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        self.base_classifier.eval()
        n1 = 0
        n2 = 0
        n0 = num
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            if self.args.srs:
                start_point = 1
            else:
                start_point = None
            if self.args.conf_drop:
                drop_idx = 0
                drop_test_samples = torch.empty((2, self.args.conf_drop), device='cuda', dtype=torch.long)

            for n_batch in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size
                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch, device='cuda') * self.sigma

                if self.args.warmup_thresh and n_batch % self.args.warmup_interval == 0:
                    warmup_f_thresh = self.args.warmup_thresh
                    warmup_solver = self.args.warm_up_solver
                else:
                    warmup_f_thresh = None
                    warmup_solver = None

                if hasattr(self.base_classifier, 'module'):
                    records, _, _, out = self.base_classifier(batch + noise, train_step=-1, compute_jac_loss=False,
                                                              start=start_point, detail=self.args.detail,
                                                              f_thresh=warmup_f_thresh, solver=warmup_solver)
                    if self.args.srs:
                        start_point = out['final_result']

                    if not self.args.detail:
                        predictions = records.argmax(1)
                    else:
                        records = records.argmax(1)
                        predictions = records[-batch.size(0):]
                        self.detail_record(records, predictions, out)
                else:
                    predictions = self.base_classifier(batch + noise).argmax(1)

                """drop unreliable samples"""
                if self.args.conf_drop and drop_idx < self.args.conf_drop and n_batch > 0:
                    records_base, _, _, _ = self.base_classifier(batch + noise, train_step=-1, compute_jac_loss=False,
                                                                 start=None, detail=self.args.detail,
                                                                 f_thresh=self.args.warmup_thresh)
                    predictions_base = records_base.argmax(1)
                    drop_idx += batch_size
                    drop_test_samples[0, drop_idx-batch_size:drop_idx] = predictions.view(1, -1)
                    drop_test_samples[1, drop_idx-batch_size:drop_idx] = predictions_base.view(1, -1)

                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)

        if self.args.conf_drop and drop_idx >= self.args.conf_drop:
            mini_counts = self._count_arr(drop_test_samples[0, :].cpu().numpy(), self.num_classes)
            hat_A = mini_counts.argmax().item()
            nA = mini_counts[hat_A].item()
            correct_A = int(torch.sum((drop_test_samples[0, :] == drop_test_samples[1, :]) * (drop_test_samples[1, :] == hat_A)).item())
            self.conserve_rate = self._lower_confidence_bound(correct_A, nA, alpha=self.alpha)
            # print(self.conserve_rate)

        return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]

    def detail_record(self, records, predictions, out):
        low_step, low_err, final_step, final_err, trace = out['lowest_step'], out['lowest'], out['final_step'], out[
            'final'], out['rel_trace']
        this_batch_size = low_step.size(0)
        if this_batch_size != self.args.N0:
            l = len(self.args.outfile.split('/')[-1])
            outPath = self.args.outfile[:-l]
            info_file = os.path.join(outPath, 'test_final.txt')
            with open(info_file, 'w') as file:
                for i in range(low_step.size(0)):
                    file.write(
                        f"idx: {i}\t lowStep: {low_step[i]}\t lowErr: {low_err[i]}\t finalStep: {final_step[i]}\t finalErr: {final_err[i]}\t Prediction: {predictions[i]}\n")
            trace_file = os.path.join(outPath, 'trace.txt')
            with open(trace_file, 'w') as file:
                for i in range(0, len(trace)):
                    for itm in trace[i]:
                        file.write(f'{itm.item():.2f}\t')
                    file.write('\n')
            label_file = os.path.join(outPath, 'label.txt')
            with open(label_file, 'w') as file:
                for i in range(records.size(0)):
                    file.write(f'{records[i].item()}\t')
                    if (i + 1) % this_batch_size == 0:
                        file.write('\n')
            step_records = out['records']
            step_records_ = torch.zeros(
                (this_batch_size, int(step_records.size(0) / this_batch_size), step_records.size(1)))
            for i in range(step_records.size(0)):
                step_records_[i % this_batch_size, i // this_batch_size, :] = step_records[i, :].view(1, 1, -1)
            num_steps = step_records_.size(1)
            dims = step_records_.size(2)
            center_list = []
            center_list.append(step_records_[:, 10:20, :].mean(1))
            center_list.append(step_records_[:, 20:30, :].mean(1))
            center_list.append(step_records_[:, 80:90, :].mean(1))
            center_list.append(step_records_[:, 90:, :].mean(1))
            num_center = len(center_list)
            distance = torch.zeros((records.size(0), num_center))
            for i in range(this_batch_size):
                for j in range(num_steps):
                    for k in range(num_center):
                        diff = step_records_[i, j, :].view(1, -1) - center_list[k][i, :].view(1, -1)
                        distance[i * num_steps + j, k] = diff.norm(dim=1) / center_list[k][i, :].view(1, -1).norm(dim=1)
            distance_file = os.path.join(outPath, 'distance.txt')
            with open(distance_file, 'w') as file:
                for i in range(distance.size(0)):
                    for j in range(distance.size(1)):
                        file.write(f'{distance[i, j].item()}\t')
                    file.write('\n')


class smooth_fit_function(nn.Module):
    # fit a differentiable function to approximate the smooth classifier
    def __init__(self, smooth, t: int = 1, num: int = 100):
        super(smooth_fit_function, self).__init__()
        self.base_classifier = smooth.base_classifier
        self.smooth = smooth
        self.t = t
        self.num = num

    def forward(self, x):
        out = self.smooth.fit_function(x, self.t, self.num)
        return out

class deq_(nn.Module):
    def __init__(self, base_classifier):
        super(deq_, self).__init__()
        self.base_classifier = base_classifier

    def forward(self, x):
        out, _, _, _ = self.base_classifier(x, attack=True, train_step=-1)
        return out
    