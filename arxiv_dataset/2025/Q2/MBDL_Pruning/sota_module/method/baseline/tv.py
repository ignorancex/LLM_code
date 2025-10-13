import torch
from get_from_config import get_dataset_from_config, get_save_path_from_config
import tqdm
from method.dep_cal import DEQCalibration
from utility import convert_pl_outputs, write_test, check_and_mkdir, get_last_folder, torch_complex_normalize
import os
import datetime
import math
from fwd.pmri import ftran, fmult
from torch.nn.functional import pad
from ray.air import session
from ray import tune


def L(x: torch.Tensor):
    """

    :rtype: y -> shape = (m, n)
    :type x -> shape = (2, m, n) where p = x[0] and q = x[1].
    """
    # y = x.clone()
    #
    # y[0, 1:, :] = x[0, 1:, :] - x[0, :-1, :]
    # y[1, :, 1:] = x[1, :, 1:] - x[1, :, :-1]
    #
    # y = y[0, :, :] + y[1, :, :]

    y = pad(x[0, 1:, :] - x[0, :-1, :], [0, 0, 1, 0]) + pad(x[1, :, 1:] - x[1, :, :-1], [1, 0, 0, 0])

    return y


def L_T(x: torch.Tensor):
    """

    :rtype: y -> shape = (2, m, n) where p = y[0] and q = y[1].
    :type x -> shape = (m, n)
    """
    # m, n = x.shape
    # y = torch.zeros(size=(2, m, n), dtype=x.dtype)
    #
    # if x.is_cuda:
    #     y = y.cuda()
    #
    # y[0, :-1, :] = x[:-1, :] - x[1:, :]
    # y[1, :, :-1] = x[:, :-1] - x[:, 1:]

    y = torch.stack([pad(x[:-1, :] - x[1:, :], [0, 0, 0, 1]), pad(x[:, :-1] - x[:, 1:], [0, 1, 0, 0])], 0)

    return y


def Pc(x: torch.Tensor, C):
    if x.dtype == torch.complex64:
        return x

    x[x < C[0]] = C[0]
    x[x > C[1]] = C[1]

    return x


def Pp(x: torch.Tensor):
    # bottom = torch.sqrt(x[0, :, :] ** 2 + x[1, :, :] ** 2)
    bottom = torch.sum(torch.abs(x) ** 2, 0).sqrt()
    bottom[bottom < 1] = 1

    return x / bottom


class TVOperator:
    def __init__(self, Lambda, N):

        super().__init__()

        self.Lambda = Lambda
        self.N = N

    def eval(self, x):
        y = L_T(x)
        y = torch.sum(torch.sqrt(y[0, :, :] ** 2 + y[1, :, :] ** 2))
        y = self.Lambda * y

        return y

    def prox(self, x):
        return self.fit(x, Lambda=self.Lambda, N=self.N)

    @staticmethod
    def fit(b, Lambda, N, C=(-float('inf'), float('inf')), verbose=False):

        m, n = b.shape

        pq = torch.zeros(size=(2, m, n), dtype=b.dtype)  # p and q are concat together.
        rs = torch.zeros(size=(2, m, n), dtype=b.dtype)  # r and s are concat together.

        if b.is_cuda:
            pq = pq.cuda()
            rs = rs.cuda()

        t = 1

        iter_ = tqdm.tqdm(range(N), desc='TVDenoiser') if verbose else range(N)

        for K in iter_:
            tLast = t
            pqLast = pq

            pq = Pp(rs + (1 / (8 * Lambda)) * L_T(Pc(b - Lambda * L(rs), C)))
            t = (1 + math.sqrt(1 + 4 * (t ** 2))) / 2
            rs = pq + (tLast - 1) / t * (pq - pqLast)

        xStart = Pc(b - Lambda * L(pq), C)

        del pq
        del rs

        return xStart


def run(config):

    with torch.no_grad():

        _, _, tst_dataset = get_dataset_from_config(config)

        tau = config['method']['baseline']['tv']['tau']
        iteration = config['method']['baseline']['tv']['iteration']
        gamma = config['method']['baseline']['tv']['gamma']

        pl_outputs = []
        for item in range(len(tst_dataset)):

            x0, smps, y, fwd_para, x, smps_gt = tst_dataset[item]

            x0, smps, mask, x, y = [torch.unsqueeze(i, 0).cuda() for i in [x0, smps, fwd_para['mask'], x, y]]

            x_hat = x0
            for _ in tqdm.tqdm(range(iteration), desc="[%d/%d] TV MRI Reconstruction" % (item + 1, len(tst_dataset))):

                x_hat = x_hat - gamma * ftran(fmult(x_hat, smps, mask) - y, smps, mask)

                x_hat = torch.squeeze(x_hat, 0)
                x_hat = TVOperator.fit(x_hat, tau, 200, verbose=False)
                x_hat = torch.unsqueeze(x_hat, 0)

            x_hat, x0, x = [i.detach().cpu() for i in [x_hat, x0, x]]

            x_hat = torch_complex_normalize(x_hat)

            tst_psnr, tst_ssim = DEQCalibration.psnr_ssim_helper(x_hat, x, 1)
            tst_psnr_init, tst_ssim_init = DEQCalibration.psnr_ssim_helper(x0, x, 1)

            pl_outputs.append({
                'tst_psnr_x_hat': tst_psnr.item(),
                'tst_ssim_x_hat': tst_ssim.item(),

                'tst_psnr_init': tst_psnr_init.item(),
                'tst_ssim_init': tst_ssim_init.item(),

                'x_hat': x_hat,
                'x_init': x0,
                'x_gt': x
            })

        log_dict, img_dict = convert_pl_outputs(pl_outputs)

        save_path = get_save_path_from_config(config)

        save_path = os.path.join(
            save_path,
            'TEST_' + datetime.datetime.now().strftime("%m%d%H%M") + "_" + get_last_folder(save_path)
        )

        if config['test']['dec'] is not None:
            save_path = save_path + "_" + config['test']['dec']

        check_and_mkdir(save_path)
        write_test(
            save_path=save_path,
            log_dict=log_dict,
            img_dict=img_dict
        )

        if tune.is_session_enabled():
            session.report({
                'tst_psnr': log_dict['tst_psnr_x_hat'].mean().item(),
                'tst_ssim': log_dict['tst_ssim_x_hat'].mean().item(),
            })
