import os.path

import pytorch_lightning as pl
from torch import nn
from einops import rearrange
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
from sota_module.fwd.ct import CTForwardModel
from .warmup import load_warmup
import torch
from sota_module.utility import convert_pl_outputs, write_test, check_and_mkdir, get_last_folder
from sota_module.get_from_config import get_module_from_config, get_dataset_from_config, get_trainer_from_config, \
    get_save_path_from_config
from torch.utils.data import DataLoader, Subset
import datetime
from sota_module.fwd.pmri import divided_by_rss
import random
from torch.optim import lr_scheduler
from ray.air import session
from ray import tune


def default_solver(f, z0, max_iter, tol, is_accelerate=True):
    last_x, last_theta, _ = z0
    s_last_x = last_x.clone()

    next_x, next_theta = last_x.clone(), last_theta.clone()

    t = torch.tensor(1., dtype=torch.float32).to(last_x.device)

    res = []
    for k in range(max_iter):
        next_x, next_theta, _ = f((s_last_x, last_theta, k))

        res.append((next_x - last_x).norm().item() / (1e-5 + next_x.norm()))
        if res[-1] < tol:
            break

        if is_accelerate:
            tnext = 0.5 * (1 + torch.sqrt(1 + 4 * t * t))
        else:
            tnext = 1

        s_last_x = next_x + ((t - 1) / tnext) * (next_x - last_x)

        # update
        t = tnext

        last_theta = next_theta
        last_x = next_x

    return (next_x, next_theta, k), res


def anderson_solver(f, z0, m=5, lam=1e-4, max_iter=50, tol=1e-4, beta=1.0):
    """ Anderson's acceleration for fixed point iteration. """

    x0, theta0, iteration0 = z0

    two_dim = (x0.dim() == 2)
    if two_dim:
        x0 = x0.unsqueeze(0)

    bsz, H, W = x0.shape
    X = torch.zeros(bsz, m, H * W, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, H * W, dtype=x0.dtype, device=x0.device)

    if two_dim:
        new_x, new_theta, new_iteration = f((x0.squeeze(0), theta0, iteration0))
        X[:, 0], F[:, 0] = x0.view(bsz, -1), new_x.unsqueeze(0).view(bsz, -1)

        new_x, new_theta, new_iteration = f((F[:, 0].view_as(x0).squeeze(0), new_theta, new_iteration))
        X[:, 1], F[:, 1] = F[:, 0], new_x.unsqueeze(0).view(bsz, -1)

    else:
        new_x, new_theta, new_iteration = f((x0, theta0, iteration0))
        X[:, 0], F[:, 0] = x0.view(bsz, -1), new_x.view(bsz, -1)

        new_x, new_theta, new_iteration = f((F[:, 0].view_as(x0), new_theta, new_iteration))
        X[:, 1], F[:, 1] = F[:, 0], new_x.view(bsz, -1)

    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    res = []

    iter_ = range(2, max_iter)

    for k in iter_:
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H[:, 1:n + 1, 1:n + 1] = torch.bmm(G, G.transpose(1, 2)) + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[
            None]

        # The following function is used for earlier pytorch version.
        # alpha = torch.solve(y[:, :n + 1], H[:, :n + 1, :n + 1])[0][:, 1:n + 1, 0]  # (bsz x n)

        alpha = torch.linalg.solve(H[:, :n + 1, :n + 1], y[:, :n + 1])[:, 1:n + 1, 0]  # (bsz x n)

        X[:, k % m] = beta * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]

        if two_dim:
            new_x, new_theta, new_iteration = f((X[:, k % m].view_as(x0).squeeze(0), new_theta, new_iteration))
            F[:, k % m] = new_x.unsqueeze(0).view(bsz, -1)

        else:
            new_x, new_theta, new_iteration = f((X[:, k % m].view_as(x0), new_theta, new_iteration))
            F[:, k % m] = new_x.view(bsz, -1)

        res.append((F[:, k % m] - X[:, k % m]).norm().item() / (1e-5 + F[:, k % m].norm().item()))

        if res[-1] < tol:
            break

    if two_dim:
        return (X[:, k % m].view_as(x0).squeeze(0), new_theta, new_iteration), res
    else:
        return (X[:, k % m].view_as(x0), new_theta, new_iteration), res


solver_dict = {
    'anderson': anderson_solver,
    'generic': lambda f, z0, max_iter, tol: default_solver(f, z0, max_iter, tol, is_accelerate=False),
    'nesterov': lambda f, z0, max_iter, tol: default_solver(f, z0, max_iter, tol, is_accelerate=True)
}


class CNNBlock(nn.Module):

    def __init__(self, net, config):
        super().__init__()

        self.net = net()
        self.config = config

    def forward(self, x, sigma):

        if x.dtype == torch.complex64:
            x_hat = torch.view_as_real(x)
            x_hat = rearrange(x_hat, 'b w h c -> b c w h')

            x_hat = self.net(x_hat, sigma)

            x_hat = rearrange(x_hat, 'b c w h -> b w h c')
            x_hat = x_hat[..., 0] + x_hat[..., 1] * 1j

        elif x.dtype == torch.float32 and x.dim() == 2:

            x_hat = self.net(x.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

        elif x.dtype == torch.float32 and x.dim() == 3:

            x_hat = self.net(x.unsqueeze(1)).squeeze(1)

        else:
            raise NotImplementedError()

        return x_hat


class DeepUnfoldingBlock(nn.Module):
    def __init__(self, net, config):
        super().__init__()

        self.net = CNNBlock(net, config)
        self.config = config

        self.gamma = config['method']['deq_cal']['gamma']
        self.alpha = config['method']['deq_cal']['alpha']

        self.sigma = config['method']['deq_cal']['warmup']['x_sigma'] / 255

        if self.config['setting']['fwd'] in ['pmri']:
            from fwd.pmri import ftran, fmult
            self.ftran = ftran
            self.fmult = fmult

        elif self.config['setting']['fwd'] in ['ct']:
            self.ct_fwd = CTForwardModel()
            self.ftran = self.ct_fwd.ftran
            self.fmult = self.ct_fwd.fmult

        else:
            raise NotImplementedError()

    @torch.enable_grad()
    def forward(self, x, y, **kwargs):

        if self.config['setting']['fwd'] in ['pmri']:

            dc = self.ftran(
                self.fmult(x, smps=kwargs['theta'], mask=kwargs['fwd_para']['mask']) - y,
                smps=kwargs['theta'], mask=kwargs['fwd_para']['mask'])

        elif self.config['setting']['fwd'] in ['ct']:
            dc = self.ftran(
                self.fmult(x, theta=kwargs['theta'], img_size=kwargs['fwd_para']['img_size']) - y,
                theta=kwargs['theta'], img_size=kwargs['fwd_para']['img_size'])

        else:
            raise NotImplementedError()

        if self.config['method']['deq_cal']['type'] == 'pnp':
            x = x - self.gamma * dc  # x^+ = x - gamma * A^H (Ax - y)
            prior = self.net(x, self.sigma)
            x_hat = self.alpha * prior + (1 - self.alpha) * x

        elif self.config['method']['deq_cal']['type'] == 'red':
            x_hat = x - self.gamma * (dc + self.alpha * (x - self.net(x, self.sigma)))

        elif self.config['method']['deq_cal']['type'] == 'mred':

            with torch.no_grad():
                G = dc + self.alpha * (x - self.net(x, self.sigma))

            x = x.requires_grad_()
            r = x - self.net(x, self.sigma)

            delta_R = torch.autograd.grad(r, x, grad_outputs=G)[0].detach()
            ATAG = self.fmult(G, smps=kwargs['theta'], mask=kwargs['fwd_para']['mask'])  # A G
            ATAG = self.ftran(ATAG, smps=kwargs['theta'], mask=kwargs['fwd_para']['mask'])  # A^T (AG)
            x_hat = x - self.gamma * (ATAG + self.alpha * delta_R)

        else:
            raise ValueError()

        return x_hat


class DEQCalibration(pl.LightningModule):
    def __init__(self, net_x, net_cal, config):
        super().__init__()

        self.config = config

        self.iterations = self.config['method']['deq_cal']['iterations']

        assert self.iterations > 1 or self.iterations == -1

        self.net = DeepUnfoldingBlock(net_x, config)
        self.cal = CalibrationModule(net_cal, config)

        if 'warmup' not in self.config['setting']['method']:

            x_pattern = self.config['method']['deq_cal']['warmup']['x_ckpt']
            if x_pattern is not None:
                load_warmup(
                    target_module=self.net.net,
                    dataset=self.config['setting']['dataset'],
                    gt_type='x',
                    pattern=x_pattern,
                    sigma=self.config['method']['deq_cal']['warmup']['x_sigma'],
                    prefix='net.net.',
                )

            self.is_joint_cal = config['method']['deq_cal']['is_joint_cal']

            if self.is_joint_cal and self.config['setting']['fwd'] == 'pmri':

                theta_pattern = self.config['method']['deq_cal']['warmup']['cal_ckpt']
                if theta_pattern is not None:
                    load_warmup(
                        target_module=self.cal,
                        dataset=self.config['setting']['dataset'],
                        gt_type='theta',
                        pattern=theta_pattern,
                        sigma=self.config['method']['deq_cal']['warmup']['cal_sigma'],
                        prefix='cal.'
                    )

        from baseline.e2e_varnet.fastmri import SSIMLoss
        loss_fn_dict = {
            'mse': lambda: nn.MSELoss(reduction='none'),
            'ssim': lambda: SSIMLoss()
        }
        self.loss_fn = loss_fn_dict[self.config['method']['deq_cal']['loss']]()

    def forward(self, z0, y, **kwargs):

        if self.iterations == -1:

            with torch.no_grad():
                z_fixed, forward_res = solver_dict[self.config['method']['deq_cal']['accelerator']](
                    lambda z: self.one_shot_iteration(z, y, **kwargs), z0,
                    max_iter=self.config['method']['deq_cal']['max_iter'],
                    tol=self.config['method']['deq_cal']['tol'],
                )

                if self.config['setting']['mode'] in ['tst', 'dug']:
                    print("forward_res: ", forward_res[-1], "forward_iter", len(forward_res))

                self.log(name='forward_res', value=forward_res[-1], prog_bar=True)
                self.log(name='forward_iter', value=len(forward_res), prog_bar=True)

            x_fixed, theta_fixed, iteration_fixed = z_fixed
            iteration_fixed = 0

            z_hat = self.one_shot_iteration((x_fixed, theta_fixed, iteration_fixed), y, **kwargs)

        else:

            z_hat = z0
            for _ in range(self.iterations):
                z_hat = self.one_shot_iteration(z_hat, y, **kwargs)

        x_hat, theta_hat, _ = z_hat

        return x_hat, theta_hat

    def one_shot_iteration(self, z, y, **kwargs):

        x_current, theta_current, iteration_current = z

        if self.config['setting']['fwd'] == 'pmri':

            if self.is_joint_cal:

                if self.config['method']['deq_cal']['is_use_input_cal']:
                    if iteration_current == 0:
                        theta_hat = self.cal(kwargs['theta_input'], self.config['method']['deq_cal']['warmup']['cal_sigma'] / 255)
                        theta_hat = divided_by_rss(theta_hat)
                    else:
                        theta_hat = theta_current
                else:
                    theta_hat = self.cal(theta_current, self.config['method']['deq_cal']['warmup']['cal_sigma'] / 255)
                    theta_hat = divided_by_rss(theta_hat)

            else:

                theta_hat = theta_current

        elif self.config['setting']['fwd'] == 'ct':

            if self.is_joint_cal:

                cal_gamma = self.config['method']['deq_cal']['cal_gamma']
                cal_alpha = self.config['method']['deq_cal']['cal_alpha']

                with torch.no_grad():

                    g_theta_dc = self.net.ct_fwd.grad_theta(x_current, y, theta=theta_current,
                                                            img_size=kwargs['fwd_para']['img_size'])
                    g_theta_prior = theta_current - kwargs['theta_input']

                    theta_hat = theta_current - cal_gamma * (g_theta_dc + cal_alpha * g_theta_prior)

            else:

                theta_hat = theta_current

        else:

            raise NotImplementedError()

        kwargs.update({
            'theta': theta_hat
        })

        x_hat = self.net(x_current, y, **kwargs)

        return x_hat, theta_hat, iteration_current + 1

    def step_helper(self, batch):

        x_input, theta_input, y, mask, x_gt, theta_gt = batch
        fwd_para = {'mask': mask}

        if self.config['setting']['fwd'] == 'ct':
            # CT only support batch size = 1
            x_input, theta_input, y, x_gt, theta_gt = [torch.squeeze(i, 0) for i in [
                x_input, theta_input, y, x_gt, theta_gt]]

            for k in fwd_para:
                new_value = torch.squeeze(fwd_para[k], 0)
                fwd_para.update({
                    k: new_value
                })

        if self.config['setting']['method'] == 'cal_warmup':
            if self.config['setting']['fwd'] == 'ct':
                raise NotImplementedError()

            elif self.config['setting']['fwd'] == 'pmri':

                num_coil = theta_input.shape[1]
                x_input, x_hat, x_gt = self.warmup_forward(
                    x=theta_input[:, random.sample(range(num_coil), 4)], sigma=self.config['method']['deq_cal']['warmup']['cal_sigma'], module=self.cal
                )

                theta_hat = theta_input

            else:
                raise NotImplementedError()

        elif self.config['setting']['method'] == 'x_warmup':
            x_input, x_hat, x_gt = self.warmup_forward(
                x=x_gt, sigma=self.config['method']['deq_cal']['warmup']['x_sigma'], module=self.net.net
            )

            theta_hat = theta_input

        else:
            iteration_input = 0

            if self.config['method']['deq_cal']['is_use_gt_theta']:
                theta_input = theta_gt

            x_hat, theta_hat = self(z0=(x_input, theta_input, iteration_input), y=y, fwd_para=fwd_para, theta_input=theta_input)

        return x_input, x_hat, x_gt, theta_input, theta_hat, theta_gt

    def training_step(self, batch, batch_idx):
        x_input, x_hat, x_gt, theta_input, theta_hat, theta_gt = self.step_helper(batch)

        loss = self.loss_helper(x_hat, x_gt)

        jacobian_loss_weight = self.config['method']['deq_cal']['jacobian_spectral_norm_reg']['jacobian_loss_weight']
        if jacobian_loss_weight > 0:
            sigma = random.uniform(0, 25) / 255

            y = x_gt + torch.randn(size=x_gt.shape, dtype=x_gt.dtype, device=x_gt.device) * sigma

            jacobian_norm_x = self.jacobian_spectral_norm(
                y=y, x_hat=x_hat, module=self.net.net, sigma=sigma, interpolation=False
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

        tra_psnr, tra_ssim = self.psnr_ssim_helper(x_hat, x_gt, 1)
        tra_theta_mse = self.theta_mse_helper(theta_hat, theta_gt)

        self.log(name='tra_psnr', value=tra_psnr, prog_bar=True)
        self.log(name='tra_ssim', value=tra_ssim, prog_bar=True)
        self.log(name='tra_theta_mse', value=tra_theta_mse, prog_bar=True)
        self.log(name='loss', value=loss, prog_bar=False)

        if batch_idx == 0:
            self.logger.log_image(key='tra_x_hat', images=[self.to_two_dim_magnitude_image(x_hat)])
            self.logger.log_image(key='tra_x', images=[self.to_two_dim_magnitude_image(x_gt)])
            self.logger.log_image(key='tra_x0', images=[self.to_two_dim_magnitude_image(x_input)])

        return loss

    def validation_step(self, batch, batch_idx):
        x_input, x_hat, x_gt, theta_input, theta_hat, theta_gt = self.step_helper(batch)

        val_loss = self.loss_helper(x_hat, x_gt).mean()
        val_psnr, val_ssim = self.psnr_ssim_helper(x_hat, x_gt, 1)
        val_theta_mse = self.theta_mse_helper(theta_hat, theta_gt)

        self.log(name='val_loss', value=val_loss, prog_bar=True)
        self.log(name='val_psnr', value=val_psnr, prog_bar=True)
        self.log(name='val_ssim', value=val_ssim, prog_bar=True)
        self.log(name='val_theta_mse', value=val_theta_mse, prog_bar=True)

        if batch_idx == 0:
            self.logger.log_image(key='val_x_hat', images=[self.to_two_dim_magnitude_image(x_hat)])
            self.logger.log_image(key='val_x', images=[self.to_two_dim_magnitude_image(x_gt)])
            self.logger.log_image(key='val_x0', images=[self.to_two_dim_magnitude_image(x_input)])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['train']['lr'])

        if self.config['train']['is_use_schduler']:
            scheduler = lr_scheduler.MultiStepLR(optimizer,
                                                 [500, 1000, 1500, 2000],
                                                 0.5)
            return [optimizer], [scheduler]

        else:
            return optimizer

    def test_step(self, batch, batch_idx):
        x_input, x_hat, x_gt, theta_input, theta_hat, theta_gt = self.step_helper(batch)

        tst_psnr, tst_ssim = self.psnr_ssim_helper(x_hat, x_gt, 1)
        tst_psnr_init, tst_ssim_init = self.psnr_ssim_helper(x_input, x_gt, 1)

        tst_theta_mse = self.theta_mse_helper(theta_hat, theta_gt)
        tst_theta_mse_init = self.theta_mse_helper(theta_input, theta_gt)

        self.log(name='tst_psnr', value=tst_psnr, prog_bar=True)
        self.log(name='tst_ssim', value=tst_ssim, prog_bar=True)
        self.log(name='tst_theta_mse', value=tst_theta_mse, prog_bar=True)

        ret = {
            'tst_psnr_x_hat': tst_psnr.item(),
            'tst_ssim_x_hat': tst_ssim.item(),
            'tst_mse_theta_hat': tst_theta_mse.item(),

            'tst_psnr_init': tst_psnr_init.item(),
            'tst_ssim_init': tst_ssim_init.item(),
            'tst_mse_theta_init': tst_theta_mse_init.item(),

            'x_hat': x_hat,
            'x_init': x_input,
        }

        if self.config['test']['is_compute_Lip']:
            # Estimate spectrum norm
            for sigma in [0, 5, 10, 15, 20, 25]:
                sigma = sigma / 255

                y = x_gt + torch.randn(size=x_gt.shape, dtype=x_gt.dtype, device=x_gt.device) * sigma

                jacobian_norm_x = self.jacobian_spectral_norm(
                    y=y, x_hat=x_hat, module=self.net.net, sigma=sigma, interpolation=False
                )

                self.log(name='tst_jacobian_norm_x_sigma_%d' % (sigma * 255), value=jacobian_norm_x.mean(), prog_bar=True)

                ret.update({
                    'tst_jacobian_norm_x_sigma_%d' % (sigma * 255): jacobian_norm_x.mean().item()
                })

        return ret

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

    def warmup_forward(self, x, sigma, module):
        if self.config['method']['deq_cal']['warmup']['is_variable_sigma']:
            sigma = random.uniform(0, self.config['method']['deq_cal']['warmup']['variable_max_sigma']) / 255
        else:
            sigma = sigma / 255

        x0 = x + torch.randn(size=x.shape, dtype=x.dtype, device=x.device) * sigma
        x_hat = module(x0, sigma)

        return x0, x_hat, x

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
    def theta_mse_helper(theta_hat, theta_gt, mask=None):
        # if theta_hat.dim() <= 2:
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


class CalibrationModule(nn.Module):

    def __init__(self, net, config):

        super().__init__()

        self.config = config
        self.net = net()

    def forward(self, x0, sigma):

        if self.config['setting']['fwd'] == 'pmri':

            batch_size = x0.shape[0]

            x0 = torch.view_as_real(x0)
            x0 = rearrange(x0, 'b l h w c -> (b l) c h w')

            x_hat = self.net(x0, sigma)

            x_hat = rearrange(x_hat, '(b l) c h w -> b l h w c', b=batch_size)

            x_hat = x_hat[..., 0] + x_hat[..., 1] * 1j

        else:

            raise NotImplementedError()

        return x_hat


def run(config):
    if config['setting']['mode'] == 'tra':
        config['test']['dec'] = 'BEST_val_loss'

    trainer = get_trainer_from_config(config)

    tra_dataset, val_dataset, tst_dataset = get_dataset_from_config(config)

    if config['setting']['mode'] == 'tst':
        model = DEQCalibration.load_from_checkpoint(
            checkpoint_path=os.path.join('/opt/experiment', get_save_path_from_config(config), config['test']['checkpoint_path']),
            net_x=get_module_from_config(config, type_='x'),
            net_cal=get_module_from_config(config, type_='theta'),
            config=config
        )
    else:
        if config['setting']['ckpt_path_module'] is not None:

            model = DEQCalibration.load_from_checkpoint(
                checkpoint_path=os.path.join('/opt/experiment', config['setting']['ckpt_path_module']),
                net_x=get_module_from_config(config, type_='x'),
                net_cal=get_module_from_config(config, type_='theta'),
                config=config
            )
        else:
            model = DEQCalibration(
                net_x=get_module_from_config(config, type_='x'),
                net_cal=get_module_from_config(config, type_='theta'),
                config=config
            )

    if config['setting']['mode'] == 'dug':
        tst_dataset = Subset(tst_dataset, [10, 20, 30, 40, 50])

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
