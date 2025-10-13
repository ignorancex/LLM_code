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
import random


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

        x_hat = torch.view_as_real(x)
        x_hat = rearrange(x_hat, 'b w h c -> b c w h')

        if sigma is not None:
            x_hat = self.cnn(x_hat, sigma)
        else:
            x_hat = self.cnn(x_hat, self.sigma)

        x_hat = rearrange(x_hat, 'b c w h -> b w h c')
        x_hat = x_hat[..., 0] + x_hat[..., 1] * 1j

        return x_hat

    def forward(self, x, theta, mask, y):
        dc = ftran(fmult(x, theta, mask) - y, theta, mask)

        x = x - self.gamma * dc  # x^+ = x - gamma * A^H (Ax - y)
        prior = self.denoise_complex(x)
        x_hat = self.alpha * prior + (1 - self.alpha) * x

        return x_hat


class ParameterUpdate(nn.Module):
    def __init__(self, net_theta, config):
        super().__init__()

        self.config = config
        self.cnn = net_theta()

        self.gamma = config['method']['deq_cal']['theta_gamma']
        self.alpha = config['method']['deq_cal']['theta_alpha']

        self.is_update_theta_iteratively = config['method']['deq_cal']['is_update_theta_iteratively']
        self.is_update_theta_iteratively_bc = config['method']['deq_cal']['is_update_theta_iteratively_bc']

    def calibrate_complex(self, x):
        batch_size = x.shape[0]

        x = torch.view_as_real(x)
        x = rearrange(x, 'b l h w c -> (b l) c h w')

        x_hat = self.cnn(x)

        x_hat = rearrange(x_hat, '(b l) c h w -> b l h w c', b=batch_size)
        x_hat = x_hat[..., 0] + x_hat[..., 1] * 1j

        x_hat = divided_by_rss(x_hat)

        return x_hat

    def forward(self, theta, x, mask, y, theta_label):
        if self.is_update_theta_iteratively:
            dc = gradient_smps(theta, x, y, mask)

            if self.is_update_theta_iteratively_bc:
                theta = theta - self.gamma * dc
                prior = self.calibrate_complex(theta)
                theta = self.alpha * prior + (1 - self.alpha) * theta

            else:
                prior = theta - theta_label
                theta = theta - self.gamma * (dc + self.alpha * prior)

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
                prefix='net.',
            )

        theta_pattern = self.config['method']['deq_cal']['warmup']['theta_ckpt']
        if theta_pattern is not None:
            load_warmup(
                target_module=self.theta_operator.cnn,
                dataset=self.config['setting']['dataset'],
                gt_type='theta',
                pattern=theta_pattern,
                sigma=self.config['method']['deq_cal']['warmup']['theta_sigma'],
                prefix='net.'
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

    def forward(self, x_init, theta_init, mask, y):

        x_hat, theta_hat = x_init, theta_init
        theta_label = None

        if self.iterations == -1:

            max_iter = self.config['method']['deq_cal']['max_iter']
            tol = self.config['method']['deq_cal']['tol']

            with torch.no_grad():

                if self.is_joint_cal:
                    theta_hat = self.theta_operator.calibrate_complex(theta_init)
                    theta_label = theta_hat

                x_pre, theta_pre = x_hat, theta_hat

                x_accelerator = self.accelerator_dict[self.accelerator](x_pre)
                theta_accelerator = self.accelerator_dict[self.accelerator](theta_pre)

                for forward_iter in range(max_iter):

                    if self.is_joint_cal:
                        theta_hat, _ = theta_accelerator(
                            self.theta_operator.forward, theta_pre,
                            x=x_pre, mask=mask, y=y, theta_label=theta_label
                        )

                    x_hat, forward_res = x_accelerator(
                        self.x_operator.forward, x_pre,
                        theta=theta_hat, mask=mask, y=y
                    )

                    if forward_res < tol:
                        break

                    x_pre = x_hat
                    theta_pre = theta_hat

                if self.config['setting']['mode'] in ['tst', 'dug']:
                    print("forward_res: ", forward_res, "forward_iter", forward_iter)

                self.log(name='forward_res', value=forward_res, prog_bar=True)
                self.log(name='forward_iter', value=forward_iter, prog_bar=True)

            if self.is_joint_cal:
                theta_label = self.theta_operator.calibrate_complex(theta_init)
                theta_hat = self.theta_operator(theta_pre, x_pre, mask, y, theta_label)

            x_hat = self.x_operator(x_pre, theta_hat, mask, y)

        else:

            if self.is_joint_cal:
                theta_hat = self.theta_operator.calibrate_complex(theta_init)
                theta_label = theta_hat

            for _ in range(self.iterations):
                x_hat = self.x_operator(x_hat, theta_hat, mask, y)

                if self.is_joint_cal:
                    theta_hat = self.theta_operator(theta_hat, x_hat, mask, y, theta_label)

        return x_hat, theta_hat

    def step_helper(self, batch):
        x_input, theta_input, y, mask, x_gt, theta_gt = batch

        if self.is_use_theta_gt:
            theta_input = theta_gt

        x_hat, theta_hat = self(x_input, theta_input, mask, y)

        loss = self.loss_helper(x_hat, x_gt)

        psnr, ssim = self.psnr_ssim_helper(x_hat, x_gt, 1)
        theta_mse = self.theta_mse_helper(theta_hat, theta_gt)

        return loss, psnr, ssim, theta_mse, x_input, x_hat, x_gt, theta_input, theta_hat, theta_gt

    def training_step(self, batch, batch_idx):
        loss, psnr, ssim, theta_mse, x_input, x_hat, x_gt, theta_input, theta_gt, theta_gt = self.step_helper(batch)

        self.log(name='tra_psnr', value=psnr, prog_bar=True)
        self.log(name='tra_ssim', value=ssim, prog_bar=True)
        self.log(name='tra_theta_mse', value=theta_mse, prog_bar=True)

        jacobian_loss_weight = self.config['method']['deq_cal']['jacobian_spectral_norm_reg']['jacobian_loss_weight']
        if jacobian_loss_weight > 0:
            sigma = random.uniform(0, 25) / 255

            y = x_gt + torch.randn(size=x_gt.shape, dtype=x_gt.dtype, device=x_gt.device) * sigma

            jacobian_norm_x = self.jacobian_spectral_norm(
                y=y, x_hat=x_hat, module=self.x_operator.denoise_complex, sigma=sigma, interpolation=False
            )

            jacobian_loss_x = torch.maximum(
                jacobian_norm_x,
                torch.ones_like(jacobian_norm_x) -
                self.config['method']['deq_cal']['jacobian_spectral_norm_reg']['eps_jacobian_loss'],
            )
            jacobian_loss_x = torch.clip(jacobian_loss_x, 0, 1e3)

            self.log(name='jacobian_loss_max_x', value=jacobian_loss_x.max(), prog_bar=True)
            self.log(name='jacobian_norm_max_x', value=jacobian_norm_x.max(), prog_bar=True)

            loss = loss + jacobian_loss_weight * jacobian_loss_x

        loss = loss.mean()

        self.log(name='loss', value=loss, prog_bar=False)

        if batch_idx == 0:
            self.logger.log_image(key='tra_x_hat', images=[self.to_two_dim_magnitude_image(x_hat)])
            self.logger.log_image(key='tra_x_gt', images=[self.to_two_dim_magnitude_image(x_gt)])
            self.logger.log_image(key='tra_x_init', images=[self.to_two_dim_magnitude_image(x_input)])

        return loss

    def validation_step(self, batch, batch_idx):
        loss, psnr, ssim, theta_mse, x_input, x_hat, x_gt, theta_input, theta_gt, theta_gt = self.step_helper(batch)

        self.log(name='val_psnr', value=psnr, prog_bar=True)
        self.log(name='val_ssim', value=ssim, prog_bar=True)
        self.log(name='val_theta_mse', value=theta_mse, prog_bar=True)

        loss = loss.mean()

        self.log(name='val_loss', value=loss, prog_bar=False)

        if batch_idx == 0:
            self.logger.log_image(key='val_x_hat', images=[self.to_two_dim_magnitude_image(x_hat)])
            self.logger.log_image(key='val_x_gt', images=[self.to_two_dim_magnitude_image(x_gt)])
            self.logger.log_image(key='val_x_init', images=[self.to_two_dim_magnitude_image(x_input)])

    def test_step(self, batch, batch_idx):
        loss, psnr, ssim, theta_mse, x_input, x_hat, x_gt, theta_input, theta_hat, theta_gt = self.step_helper(batch)

        self.log(name='tst_psnr', value=psnr, prog_bar=True)
        self.log(name='tst_ssim', value=ssim, prog_bar=True)
        self.log(name='tst_theta_mse', value=theta_mse, prog_bar=True)

        return {
            'tst_psnr_x_hat': psnr.item(),
            'tst_ssim_x_hat': ssim.item(),
            'tst_mse_theta_hat': theta_mse.item(),

            'x_input': x_input,
            'x_hat': x_hat,
            'x_gt': x_gt,

            'theta_input': theta_input,
            'theta_hat': theta_hat,
            'theta_gt': theta_gt,
        }

    def test_epoch_end(self, outputs) -> None:

        if tune.is_session_enabled():
            ret = {}
            for k in self.trainer.logged_metrics:
                ret.update({
                    k: self.trainer.logged_metrics[k].item()
                })
            session.report(ret)

        log_dict, img_dict = convert_pl_outputs(outputs)

        save_path = get_save_path_from_config(self.config)

        save_path = os.path.join(
            save_path,
            'TEST_' + datetime.datetime.now().strftime("%m%d%H%M") + "_" + get_last_folder(save_path)
        )

        if self.config['test']['dec'] is not None:
            save_path = save_path + "_" + self.config['test']['dec']

        check_and_mkdir(save_path)
        write_test(
            save_path=save_path,
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
        if theta_gt.dtype == torch.complex64:
            theta_hat = torch.view_as_real(theta_hat)
            theta_gt = torch.view_as_real(theta_gt)

        return torch.mean((theta_hat - theta_gt) ** 2)

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

    @torch.enable_grad()
    def power_iteration(self, operator, vector_size, steps=100, momentum=0.0, eps=1e-3,
                        init_vec=None, verbose=False):
        """
        Power iteration algorithm for spectral norm calculation
        """
        with torch.no_grad():
            if init_vec is None:
                vec = torch.rand(vector_size, dtype=torch.complex64).to(self.device)
            else:
                vec = init_vec.to(self.device)

            vec /= torch.norm(vec.view(vector_size[0], -1), dim=1, p=2).view(vector_size[0], 1, 1)

            for i in range(steps):

                new_vec = operator(vec)
                new_vec = new_vec / torch.norm(new_vec.view(vector_size[0], -1), dim=1, p=2).view(vector_size[0], 1, 1)
                if momentum > 0 and i > 1:
                    new_vec -= momentum * old_vec
                old_vec = vec
                vec = new_vec
                diff_vec = torch.norm(new_vec - old_vec, p=2)
                if diff_vec < eps:
                    if verbose:
                        print("Power iteration converged at iteration: ", i)
                    break

        new_vec = operator(vec)
        div = torch.norm(vec.view(vector_size[0], -1), dim=1, p=2).view(vector_size[0])
        lambda_estimate = torch.abs(
            torch.sum(vec.view(vector_size[0], -1) * new_vec.view(vector_size[0], -1), dim=1)) / div

        return lambda_estimate

    @torch.enable_grad()
    def jacobian_spectral_norm(self, y, x_hat, module, sigma, interpolation=False):
        # WG: here y indeed means noisy image, and x_hat means y's reconstruction counterpart (says, the output of DEQ).

        if interpolation:
            eta = torch.rand(y.size(0), 1, 1, 1, requires_grad=True).to(self.device)
            x = eta * y.detach() + (1 - eta) * x_hat.detach()
            x = x.to(self.device)
        else:
            x = y

        x.requires_grad_()
        x_hat = module(x, sigma)
        Dg = x - x_hat

        if self.config['module']['gs_denoiser']['grad_matching']:
            # we calculate the lipschitz constant of the gradient operator Dg=Id-D
            operator = lambda vec: \
                torch.autograd.grad(Dg, x, grad_outputs=vec, create_graph=True, retain_graph=True, only_inputs=True)[0]
        else:
            # we calculate the lipschitz constant of the denoiser operator D
            f = x_hat
            operator = lambda vec: \
                torch.autograd.grad(f, x, grad_outputs=vec, create_graph=True, retain_graph=True, only_inputs=True)[0]

        lambda_estimate = self.power_iteration(
            operator, x.size(),
            steps=self.config['method']['deq_cal']['jacobian_spectral_norm_reg']['power_method_nb_step'],
            momentum=self.config['method']['deq_cal']['jacobian_spectral_norm_reg']['power_method_error_momentum'],
            eps=self.config['method']['deq_cal']['jacobian_spectral_norm_reg']['power_method_error_threshold']
        )

        return lambda_estimate


def run(config):
    if config['setting']['mode'] == 'tra':
        config['test']['dec'] = 'BEST_val_loss'

    trainer = get_trainer_from_config(config)

    tra_dataset, val_dataset, tst_dataset = get_dataset_from_config(config)

    if config['setting']['mode'] == 'tst':
        model = DEQCalibrationMRI.load_from_checkpoint(
            checkpoint_path=os.path.join('/opt/experiment', get_save_path_from_config(config),
                                         config['test']['checkpoint_path']),
            net_x=get_module_from_config(config, type_='x'),
            net_theta=get_module_from_config(config, type_='theta'),
            config=config
        )
    else:
        if config['setting']['ckpt_path_module'] is not None:

            model = DEQCalibrationMRI.load_from_checkpoint(
                checkpoint_path=os.path.join('/opt/experiment', config['setting']['ckpt_path_module']),
                net_x=get_module_from_config(config, type_='x'),
                net_theta=get_module_from_config(config, type_='theta'),
                config=config
            )
        else:
            model = DEQCalibrationMRI(
                net_x=get_module_from_config(config, type_='x'),
                net_theta=get_module_from_config(config, type_='theta'),
                config=config
            )

    # if config['setting']['mode'] == 'dug':
    #     tst_dataset = Subset(tst_dataset, [10, 20, 30, 40, 50])

    ckpt_path = None

    if config['setting']['mode'] == 'tra':
        tra_dataloader = DataLoader(
            tra_dataset,
            batch_size=config['train']['batch_size'],
            shuffle=True,
            num_workers=config['train']['num_workers'],
            drop_last=True
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config['train']['batch_size'],
            shuffle=False,
            num_workers=config['train']['num_workers'],
            drop_last=True
        )

        trainer.fit(
            model=model,
            train_dataloaders=tra_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=os.path.join('/opt/experiment', config['setting']['ckpt_path_trainer']) if
            config['setting']['ckpt_path_trainer'] is not None else None
        )

        ckpt_path = 'best'

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

