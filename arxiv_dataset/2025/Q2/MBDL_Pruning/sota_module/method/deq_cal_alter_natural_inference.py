from torch import nn
import pytorch_lightning as pl
from method.warmup import load_warmup
from baseline.e2e_varnet.fastmri import SSIMLoss
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
import torch
from get_from_config import get_trainer_from_config, get_dataset_from_config, get_save_path_from_config, \
    get_module_from_config
from torch.utils.data import DataLoader, Subset
import os
from einops import rearrange
from fwd.pmri import ftran, fmult, divided_by_rss
from fwd.pmri_modl import gradient_smps
from utility import convert_pl_outputs, write_test, check_and_mkdir, get_last_folder
from ray.air import session
from ray import tune
import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import io
import random
from dataset.natural_image import G, Gt, grad_theta, load_kernel_via_idx
from collections import defaultdict


class ImageUpdateBLS(nn.Module):
    def __init__(self, net_x, config):
        super().__init__()

        self.config = config
        self.cnn = net_x()

        self.gamma = config['method']['deq_cal']['x_gamma']
        self.alpha = config['method']['deq_cal']['x_alpha']
        self.lambd = config['method']['deq_cal']['x_lambda']

        self.sigma = config['method']['deq_cal']['warmup']['x_sigma'] / 255

    def denoise_complex(self, x, sigma=None, is_return_N=False):

        x_hat = torch.view_as_real(x)
        x_hat = rearrange(x_hat, 'b w h c -> b c w h')

        if sigma is None:
            sigma = self.sigma

        Dg, N = self.cnn.net.calculate_grad(x_hat, sigma)
        x_hat = x_hat - Dg

        x_hat = rearrange(x_hat, 'b c w h -> b w h c')
        x_hat = x_hat[..., 0] + x_hat[..., 1] * 1j

        if not is_return_N:
            return x_hat

        else:
            N = rearrange(N, 'b c w h -> b w h c')
            N = N[..., 0] + N[..., 1] * 1j

            return x_hat, N

    def forward(self, x, theta, mask, y):
        gamma = self.gamma
        beta = 0.95
        x_old = x

        # Denoising of x_old and calculation of F_old

        dc = ftran(fmult(x, theta, mask) - y, theta, mask)
        x = x - self.gamma * self.lambd * dc  # x^+ = x - gamma * A^H (Ax - y)

        Dx, N = self.denoise_complex(x, is_return_N=True)
        x_hat = self.alpha * Dx + (1 - self.alpha) * x

        g = 0.5 * (torch.norm(torch.view_as_real(x - N), p=2) ** 2)
        regul = (1 / self.lambd) * (g - (1 / 2) * torch.norm(x - x_hat, p=2) ** 2)
        f = 0.5 * (torch.norm(torch.view_as_real(fmult(x_old, theta, mask) - y), p=2) ** 2)
        F = f + regul

        F_old = F

        backtracking_check = True

        while not backtracking_check:

            x = x_old
            dc = ftran(fmult(x, theta, mask) - y, theta, mask)
            x = x - gamma * self.lambd * dc  # x^+ = x - gamma * A^H (Ax - y)

            Dx, N = self.denoise_complex(x, is_return_N=True)
            x_hat = self.alpha * Dx + (1 - self.alpha) * x

            g = 0.5 * (torch.norm(torch.view_as_real(x - N), p=2) ** 2)
            regul = (1 / self.lambd) * (g - (1 / 2) * torch.norm(x - x_hat, p=2) ** 2)
            f = 0.5 * (torch.norm(torch.view_as_real(fmult(x, theta, mask) - y), p=2) ** 2)
            F = f + regul

            F_new = F
            x = x_hat

            diff_x = (torch.norm(torch.view_as_real(x - x_old), p=2) ** 2)
            diff_F = F_old - F_new

            if diff_F < (0.1 / gamma) * diff_x:
                backtracking_check = False
                gamma = beta * gamma

                print("BLS diff_F:", diff_F.item(), 'diff_x:', diff_x.item(), 'gamma:', gamma)

            else:
                backtracking_check = True

        print('F:', F.item(), 'f:', f.item(), 'regul:', regul.item())

        return x_hat


class ImageUpdate(nn.Module):
    def __init__(self, net_x, config):
        super().__init__()

        self.config = config
        self.cnn = net_x()

        self.gamma = config['method']['deq_cal']['x_gamma']
        self.alpha = config['method']['deq_cal']['x_alpha']

        self.sigma = config['method']['deq_cal']['warmup']['x_sigma'] / 255

    def denoise_complex(self, x, sigma=None):

        x_hat = x

        if sigma is not None:
            noise_level_map = torch.FloatTensor(x_hat.size(0), 1, x_hat.size(2), x_hat.size(3)).fill_(sigma).to(x_hat.device)
            x_hat = torch.cat((x_hat, noise_level_map), 1)
            x_hat = self.cnn(x_hat)

        else:
            x_hat = self.cnn(x_hat)

        return x_hat

        # print(x.shape, x.dtype)
        #
        # x_hat = torch.view_as_real(x)
        # x_hat = rearrange(x_hat, 'b w h c -> b c w h')
        #
        # if sigma is not None:
        #     noise_level_map = torch.FloatTensor(x_hat.size(0), 1, x_hat.size(2), x_hat.size(3)).fill_(sigma).to(x_hat.device)
        #     x_hat = torch.cat((x_hat, noise_level_map), 1)
        #     x_hat = self.cnn(x_hat)
        #
        # else:
        #     x_hat = self.cnn(x_hat)
        #
        # x_hat = rearrange(x_hat, 'b c w h -> b w h c')
        # x_hat = x_hat[..., 0] + x_hat[..., 1] * 1j
        #
        # return x_hat

    def forward(self, x, theta, y):

        dc = Gt(G(x, theta, sf=self.config['dataset']['natural']['down_sampling_factor']) - y, theta, sf=self.config['dataset']['natural']['down_sampling_factor'])

        x = x - self.gamma * dc  # x^+ = x - gamma * A^H (Ax - y)

        if self.config['method']['deq_cal']['warmup']['x_ckpt'] == 'g_denoise':
            prior = self.denoise_complex(x, self.sigma)
        else:
            prior = self.denoise_complex(x)

        x_hat = self.alpha * prior + (1 - self.alpha) * x

        return x_hat

        # print("here")
        # exit(0)
        #
        # dc = ftran(fmult(x, theta, mask) - y, theta, mask)
        #
        # x = x - self.gamma * dc  # x^+ = x - gamma * A^H (Ax - y)
        #
        # if self.config['method']['deq_cal']['warmup']['x_ckpt'] == 'g_denoise':
        #     prior = self.denoise_complex(x, self.sigma)
        # else:
        #     prior = self.denoise_complex(x)
        #
        # x_hat = self.alpha * prior + (1 - self.alpha) * x
        #
        # return x_hat


class ParameterUpdate(nn.Module):
    def __init__(self, net_theta, config):
        super().__init__()

        self.config = config
        self.cnn = net_theta()

        self.gamma = config['method']['deq_cal']['theta_gamma']
        self.alpha = config['method']['deq_cal']['theta_alpha']

        self.is_update_theta_iteratively = config['method']['deq_cal']['is_update_theta_iteratively']
        self.is_update_theta_iteratively_bc = config['method']['deq_cal']['is_update_theta_iteratively_bc']

        self.sigma = config['method']['deq_cal']['warmup']['theta_sigma'] / 255

    def calibrate_complex(self, x, sigma=None):

        # batch_size = x.shape[0]
        #
        # x = torch.view_as_real(x)
        # x = rearrange(x, 'b l h w c -> (b l) c h w')

        if sigma is not None:
            noise_level_map = torch.FloatTensor(x.size(0), 1, x.size(2), x.size(3)).fill_(sigma).to(
                x.device)
            x_hat = torch.cat((x, noise_level_map), 1)
            x_hat = self.cnn(x_hat)

        else:
            x_hat = self.cnn(x)

        # x_hat = rearrange(x_hat, '(b l) c h w -> b l h w c', b=batch_size)
        # x_hat = x_hat[..., 0] + x_hat[..., 1] * 1j

        # x_hat = divided_by_rss(x_hat)

        return x_hat

    def forward(self, theta, x, y, theta_label):
        if self.is_update_theta_iteratively:
            # dc = gradient_smps(theta, x, y, mask)
            dc = grad_theta(x, y, theta, sf=self.config['dataset']['natural']['down_sampling_factor'])

            if self.is_update_theta_iteratively_bc:
                theta = theta - self.gamma * dc

                if self.config['method']['deq_cal']['warmup']['theta_ckpt'] == 'g_denoise':
                    prior = self.calibrate_complex(theta, self.sigma)
                else:
                    prior = self.calibrate_complex(theta)

                theta = self.alpha * prior + (1 - self.alpha) * theta

            else:
                prior = theta - theta_label
                theta = theta - self.gamma * (dc + self.alpha * prior)

            # theta = divided_by_rss(theta)

        return theta


class GenericAccelerator:
    def __init__(self, x_init):
        self.t = 1.0
        self.x_prev = x_init

    def __call__(self, f, s, **kwargs):
        xnext = f(s, **kwargs)

        res = (xnext - self.x_prev).norm().item() / xnext.norm().item()

        self.x_prev = xnext

        return xnext, res


class NesterovAccelerator:
    def __init__(self, x_init):
        self.t = 1.0
        self.x_prev = x_init

    def __call__(self, f, s, **kwargs):

        xnext = f(s, **kwargs)
        tnext = 0.5 * (1 + math.sqrt(1 + 4 * self.t * self.t))
        s = xnext + ((self.t - 1) / tnext) * (xnext - self.x_prev)

        res = (xnext - self.x_prev).norm().item() / xnext.norm().item()

        self.t = tnext
        self.x_prev = xnext

        return s, res


class AndersonAccelerator:
    def __init__(self, x_init, m=5, lam=1e-4, beta=1.0):
        self.m = m
        self.lam = lam
        self.beta = beta
        self.k = 0

        self.x0 = x_init
        self.bsz = x_init.shape[0]
        total_elements = 1
        for i in x_init.shape[1:]:
            total_elements = total_elements * i

        self.X = torch.zeros(self.bsz, self.m, total_elements, dtype=x_init.dtype, device=x_init.device)
        self.F = torch.zeros(self.bsz, self.m, total_elements, dtype=x_init.dtype, device=x_init.device)

        self.X[:, 0] = self.x0.view(self.bsz, -1)

        self.H = torch.zeros(self.bsz, m + 1, m + 1, dtype=self.x0.dtype, device=self.x0.device)
        self.H[:, 0, 1:] = self.H[:, 1:, 0] = 1
        self.y = torch.zeros(self.bsz, m + 1, 1, dtype=self.x0.dtype, device=self.x0.device)
        self.y[:, 0] = 1

    def __call__(self, f, s, **kwargs):

        if self.k == 0:
            self.F[:, 0] = f(s, **kwargs).view(self.bsz, -1)

            self.k += 1

            return self.F[:, 0].view_as(self.x0), 1e7

        elif self.k == 1:
            self.X[:, 1], self.F[:, 1] = self.F[:, 0], f(self.F[:, 0].view_as(self.x0), **kwargs).view(self.bsz, -1)

            self.k += 1

            return self.F[:, 1].view_as(self.x0), 1e7

        else:
            n = min(self.k, self.m)
            G = self.F[:, :n] - self.X[:, :n]
            self.H[:, 1:n + 1, 1:n + 1] = torch.bmm(G, G.transpose(1, 2)) + self.lam * torch.eye(n, dtype=self.x0.dtype, device=self.x0.device)[None]

            # alpha = torch.solve(self.y[:, :n + 1], self.H[:, :n + 1, :n + 1])[0][:, 1:n + 1, 0]  # (bsz x n)
            alpha = torch.linalg.solve(self.H[:, :n + 1, :n + 1], self.y[:, :n + 1])[:, 1:n + 1, 0]

            self.X[:, self.k % self.m] = self.beta * (alpha[:, None] @ self.F[:, :n])[:, 0] + (1 - self.beta) * (alpha[:, None] @ self.X[:, :n])[:, 0]
            self.F[:, self.k % self.m] = f(self.X[:, self.k % self.m].view_as(self.x0), **kwargs).view(self.bsz, -1)

            res = (self.F[:, self.k % self.m] - self.X[:, self.k % self.m]).norm().item() / (1e-5 + self.F[:, self.k % self.m].norm().item())

            self.k += 1

            return self.X[:, (self.k - 1) % self.m].view_as(self.x0), res


class DEQCalibrationMRI(pl.LightningModule):
    def __init__(self, net_x, net_theta, config):
        super().__init__()

        self.config = config
        self.iterations = self.config['method']['deq_cal']['iterations']

        assert self.iterations > 1 or self.iterations == -1

        self.x_operator = ImageUpdate(net_x, config)
        # self.x_operator = ImageUpdateBLS(net_x, config)
        self.theta_operator = ParameterUpdate(net_theta, config)

        x_pattern = self.config['method']['deq_cal']['warmup']['x_ckpt']
        if x_pattern is not None:
            load_warmup(
                target_module=self.x_operator.cnn,
                dataset=self.config['setting']['dataset'],
                gt_type='x',
                pattern=x_pattern,
                sigma=self.config['method']['deq_cal']['warmup']['x_sigma'],
                prefix='',
                network=self.config['method']['deq_cal']['x_module'],
                is_load_state_dict=False
            )

        theta_pattern = self.config['method']['deq_cal']['warmup']['theta_ckpt']
        if theta_pattern is not None:
            load_warmup(
                target_module=self.theta_operator.cnn,
                dataset=self.config['setting']['dataset'],
                gt_type='theta',
                pattern=theta_pattern,
                sigma=self.config['method']['deq_cal']['warmup']['theta_sigma'],
                prefix='net.',
                network=self.config['method']['deq_cal']['theta_module']
            )

        loss_fn_dict = {
            'mse': lambda: nn.MSELoss(reduction='none'),
            'ssim': lambda: SSIMLoss()
        }
        self.loss_fn = loss_fn_dict[self.config['method']['deq_cal']['loss']]()

        self.is_joint_cal = self.config['method']['deq_cal']['is_joint_cal']
        self.is_use_theta_gt = self.config['method']['deq_cal']['is_use_gt_theta']

        self.accelerator_dict = {
            'generic': lambda x_init: GenericAccelerator(x_init),
            'nesterov': lambda x_init: NesterovAccelerator(x_init),
            'anderson': lambda x_init: AndersonAccelerator(x_init),
        }
        self.accelerator = self.config['method']['deq_cal']['accelerator']

    def forward(self, x_init, theta_init, y):

        x_hat, theta_hat = x_init, theta_init
        # theta_hat = divided_by_rss(theta_hat)

        theta_label = None

        if self.iterations == -1:

            max_iter = self.config['method']['deq_cal']['max_iter']
            tol = self.config['method']['deq_cal']['tol']

            with torch.no_grad():

                x_pre, theta_pre = x_hat, theta_hat

                x_pre_list, theta_pre_list = [], []
                res_x_list, res_theta_list, res_list = [], [], []

                x_accelerator = self.accelerator_dict[self.accelerator](x_pre)
                theta_accelerator = self.accelerator_dict[self.accelerator](theta_pre)

                update_idx = [0, 1]

                for forward_iter in range(max_iter):

                    x_pre_list.append(x_pre.detach().cpu())
                    theta_pre_list.append(theta_pre.detach().cpu())

                    random.shuffle(update_idx)

                    for idx in update_idx:

                        if idx == 0:
                            x_hat, _ = x_accelerator(
                                self.x_operator.forward, x_pre,
                                theta=theta_hat, y=y
                            )

                        elif idx == 1:

                            if self.is_joint_cal:
                                theta_hat, _ = theta_accelerator(
                                    self.theta_operator.forward, theta_pre,
                                    x=x_pre, y=y, theta_label=theta_label
                                )
                            else:
                                theta_hat = theta_pre

                    forward_res_theta = torch.norm(theta_pre - theta_hat) ** 2
                    forward_res_x = torch.norm(x_pre - x_hat) ** 2

                    forward_res = forward_res_x + forward_res_theta

                    res_x_list.append(forward_res_x.item())
                    res_theta_list.append(forward_res_theta.item())
                    res_list.append(forward_res.item())

                    if torch.isnan(forward_res):
                        raise ValueError('meet nan in the iteration')

                    if forward_res < tol:
                        break

                    x_pre = x_hat
                    theta_pre = theta_hat

                    # if self.config['setting']['mode'] in ['dug']:
                    #     print("x_res: ", forward_res_x.item(), "theta_res: ", forward_res_theta.item(),
                    #           "forward_res: ", forward_res.item(), "forward_iter", forward_iter,
                    #           "x_accelerator.t: ", x_accelerator.t)

                print("[FINAL] x_res: ", forward_res_x.item(), "theta_res: ", forward_res_theta.item(),
                      "forward_res: ", forward_res.item(), "forward_iter", forward_iter)

                self.log(name='forward_res', value=forward_res, prog_bar=True)
                self.log(name='forward_iter', value=forward_iter, prog_bar=True)

            x_hat, theta_hat = x_pre, theta_pre

            x_pre_list, theta_pre_list = [torch.stack(i, 0) for i in [x_pre_list, theta_pre_list]]

        return x_hat, theta_hat, res_x_list, res_theta_list, res_list, x_pre_list, theta_pre_list

    def step_helper(self, batch):
        # x_input, theta_input, y, mask, x_gt, theta_gt = batch

        x_gt, theta_gt, y = [torch.squeeze(i, dim=0) for i in batch]

        theta_input = load_kernel_via_idx(
            self.config['dataset']['natural']['input_idx']
        ).unsqueeze(0).unsqueeze(0).cuda()
        # theta_input = theta_gt + torch.randn_like(theta_gt) * 0.05

        x_input = Gt(y, theta_input, sf=self.config['dataset']['natural']['down_sampling_factor'])

        if self.is_use_theta_gt:
            theta_input = theta_gt

        x_hat, theta_hat, res_x_list, res_theta_list, res_list, x_pre_list, theta_pre_list = self(x_input, theta_input, y)

        loss = self.loss_helper(x_hat, x_gt)

        psnr, ssim = self.psnr_ssim_helper(x_hat, x_gt, 1)

        theta_hat[torch.abs(theta_gt) == 0] = 0
        theta_mse = self.theta_mse_helper(theta_hat, theta_gt)

        return loss, psnr, ssim, theta_mse, x_input, x_hat, x_gt, theta_input, theta_hat, theta_gt, res_x_list, res_theta_list, res_list, x_pre_list, theta_pre_list

    def test_step(self, batch, batch_idx):
        loss, psnr, ssim, theta_mse, x_input, x_hat, x_gt, theta_input, theta_hat, theta_gt, res_x_list, res_theta_list, res_list, x_pre_list, theta_pre_list = self.step_helper(batch)

        x_pre_list = torch.squeeze(x_pre_list)
        theta_pre_list = torch.squeeze(theta_pre_list)

        self.log(name='tst_psnr', value=psnr, prog_bar=True)
        self.log(name='tst_ssim', value=ssim, prog_bar=True)
        self.log(name='tst_theta_mse', value=theta_mse, prog_bar=True)

        fig = plt.figure(figsize=(19.2, 4.8))
        plt.subplot(1, 3, 1)
        plt.plot(res_list, label='total residuals')
        plt.yscale('log')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(res_x_list, label='x residuals')
        plt.yscale('log')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(res_theta_list, label='theta residuals')
        plt.yscale('log')
        plt.legend()

        plt.suptitle('batch_idx [%d]' % batch_idx)

        io_buf = io.BytesIO()
        fig.savefig(io_buf, format='raw')
        io_buf.seek(0)
        img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                             newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
        img_arr = torch.from_numpy(img_arr)
        io_buf.close()
        plt.close()

        res = {
            'tst_psnr_x_hat': psnr.item(),
            'tst_ssim_x_hat': ssim.item(),
            'tst_mse_theta_hat': theta_mse.item(),
        }

        if self.config['setting']['mode'] == 'tst':
            res.update({
                # 'x_input_real': x_input.real,
                # 'x_hat_real': x_hat.real,
                # 'x_gt_real': x_gt.real,
                #
                # 'x_input_imag': x_input.imag,
                # 'x_hat_imag': x_hat.imag,
                # 'x_gt_imag': x_gt.imag,

                'x_gt': x_gt,
                'theta_gt': theta_gt,

                # 'theta_input_real': theta_input.real,
                # 'theta_hat_real': theta_hat.real,
                # 'theta_gt_real': theta_gt.real,
                #
                # 'theta_input_imag': theta_input.imag,
                # 'theta_hat_imag': theta_hat.imag,
                # 'theta_gt_imag': theta_gt.imag,

                'plot': img_arr,

                'x_hat': x_hat,
                'theta_hat': theta_hat,

                'theta_input': theta_input,
                'x_input': x_input,
            })

            if self.config['test']['save_pre_list']:
                res.update({

                    'x_pre_list': x_pre_list,
                    'theta_pre_list': theta_pre_list

                })

        return res

    def test_epoch_end(self, outputs) -> None:

        if tune.is_session_enabled():
            ret = {}
            for k in self.trainer.logged_metrics:
                ret.update({
                    k: self.trainer.logged_metrics[k].item()
                })
            session.report(ret)

        save_path = get_save_path_from_config(self.config)

        save_path = os.path.join(
            save_path,
            'TEST_' + datetime.datetime.now().strftime("%m%d%H%M") + "_" + get_last_folder(save_path)
        )

        if self.config['test']['dec'] is not None:
            save_path = save_path + "_" + self.config['test']['dec']

        check_and_mkdir(save_path)

        for i in range(len(outputs)):

            log_dict, img_dict = convert_pl_outputs([outputs[i]])

            save_path_item = os.path.join(save_path, 'item_%d' % i)
            check_and_mkdir(save_path_item)

            write_test(
                save_path=save_path_item,
                log_dict=log_dict,
                img_dict=img_dict
            )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['train']['lr'])

        return optimizer

    def loss_helper(self, predict, label):
        if self.config['method']['deq_cal']['loss'] == 'mse':
            if predict.dtype == torch.complex64:
                predict = torch.view_as_real(predict)
                label = torch.view_as_real(label)

            return self.loss_fn(predict.view(predict.size()[0], -1), label.view(label.size()[0], -1)).mean(dim=1)

        elif self.config['method']['deq_cal']['loss'] == 'ssim':
            if predict.dtype == torch.complex64:
                predict = torch.abs(predict)
                label = torch.abs(label)

            num_batch = predict.shape[0]

            max_value = torch.cat([torch.Tensor([1.0])] * num_batch, 0).to(torch.float32).to(predict.device)
            loss = self.loss_fn(predict.unsqueeze(1), label.unsqueeze(1), max_value, False)

            loss = torch.reshape(loss, [num_batch, -1]).mean(dim=1)

        return loss

    @staticmethod
    def psnr_ssim_helper(x_hat, x_gt, data_range):
        if x_hat.dtype == torch.complex64:
            x_hat = torch.abs(x_hat)
            x_gt = torch.abs(x_gt)

        x_hat[x_gt == 0] = 0

        if x_hat.dim() == 3:
            x_hat = x_hat.unsqueeze(1)
            x_gt = x_gt.unsqueeze(1)

        elif x_hat.dim() == 2:
            x_hat = x_hat.unsqueeze(0).unsqueeze(0)
            x_gt = x_gt.unsqueeze(0).unsqueeze(0)

        return peak_signal_noise_ratio(x_hat, x_gt, data_range=data_range), \
            structural_similarity_index_measure(x_hat, x_gt, data_range=data_range)

    @staticmethod
    def theta_mse_helper(theta_hat, theta_gt):
        # if theta_gt.dtype == torch.complex64:
        #     theta_hat = torch.view_as_real(theta_hat)
        #     theta_gt = torch.view_as_real(theta_gt)

        return torch.sqrt(torch.mean(torch.abs(theta_hat - theta_gt) ** 2)) / torch.sqrt(torch.mean(torch.abs(theta_gt) ** 2))

    @staticmethod
    def to_two_dim_magnitude_image(x):
        if x.dtype == torch.complex64:
            x = torch.abs(x)

        if x.dim() == 2:
            return x

        elif x.dim() == 3:
            return x[0]

        elif x.dim() == 4:
            return x[0][0]


def run(config):

    trainer = get_trainer_from_config(config)

    tra_dataset, val_dataset, tst_dataset = get_dataset_from_config(config)

    model = DEQCalibrationMRI(
        net_x=get_module_from_config(config, type_='x', use_sigma_map=(config['method']['deq_cal']['warmup']['x_ckpt'] == 'g_denoise')),
        net_theta=get_module_from_config(config, type_='theta', use_sigma_map=(config['method']['deq_cal']['warmup']['theta_ckpt'] == 'g_denoise')),
        config=config
    )

    # if config['setting']['mode'] == 'dug':
    #     tst_dataset = Subset(tst_dataset, [5])

    # tst_dataset = Subset(tst_dataset, [5])

    ckpt_path = None

    tst_dataloader = DataLoader(
        tst_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=True
    )

    trainer.test(
        model=model,
        dataloaders=tst_dataloader,
        ckpt_path=ckpt_path
    )
