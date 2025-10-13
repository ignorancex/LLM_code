import numpy as np

from get_from_config import get_module_from_config, get_dataset_from_config
from torch.utils.data import Subset
import torch
from method.warmup import load_warmup
from einops import rearrange
from method.deq_cal_alter_pmri_inference import DEQCalibrationMRI
import tabulate
from tabulate import SEPARATING_LINE
import tqdm


def run(config):

    SIGMA_LIST = [1, 5, 10, 15, 20, 25]

    config['setting']['mode'] = 'tst'

    _, _, tst_dataset = get_dataset_from_config(config)

    for index in [0]:

        table = []

        for method in ['gs_denoiser', 'unetres']:

            config['method']['deq_cal']['x_module'] = method.replace('-v', '')

            tmp_ = [method]
            for sigma in tqdm.tqdm(SIGMA_LIST, desc="idx=[%d] method=[%s]" % (index, method)):

                with torch.no_grad():

                    net = get_module_from_config(config, 'x' if index == 0 else 'theta', use_sigma_map=True)()

                    load_warmup(
                        net.net if method in ['gs_denoiser'] else net,
                        dataset=config['setting']['dataset'],
                        gt_type='x' if index == 0 else 'theta',
                        pattern='g_denoise',
                        sigma=sigma,
                        prefix='',
                        is_print=False,
                        network=method.replace('-v', ''),
                        is_load_state_dict=True if method in ['gs_denoiser'] else False
                    )

                    net.cuda()
                    if method in ['gs_denoiser']:
                        net.net.cuda()

                    psnr, ssim = [], []

                    for i in range(len(tst_dataset)):

                        x_gt = tst_dataset[i][index]
                        x = x_gt + torch.randn(size=x_gt.shape, dtype=x_gt.dtype, device=x_gt.device) * sigma / 255

                        # exit(0)
                        #
                        # x = torch.view_as_real(x_noisy)
                        # x = rearrange(x, 'b w h c -> b c w h')

                        if method in ['gs_denoiser']:
                            x_pre = net(x.cuda(), sigma / 255).cpu()
                        else:
                            noise_level_map = torch.FloatTensor(x.size(0), 1, x.size(2), x.size(3)).fill_(
                                sigma / 255).to(x.device)
                            x = torch.cat((x, noise_level_map), 1)

                            x_pre = net(x.cuda()).cpu()

                        # x_pre = rearrange(x_pre, 'b c w h -> b w h c')
                        # x_pre = x_pre[..., 0] + x_pre[..., 1] * 1j

                        psnr_tmp, ssim_tmp = DEQCalibrationMRI.psnr_ssim_helper(x_pre, x_gt, 1)

                        psnr.append(psnr_tmp)
                        ssim.append(ssim_tmp)

                    psnr = np.array(psnr).mean()
                    ssim = np.array(ssim).mean()

                    tmp_.append('%.2f / %.3f' % (psnr, ssim))

            table.append(tmp_)

        if index == -1:
            print("theta map denoising")
        else:
            print("x denoising")

        print(tabulate.tabulate(table, headers=['sigma'] + SIGMA_LIST, tablefmt="rst"))