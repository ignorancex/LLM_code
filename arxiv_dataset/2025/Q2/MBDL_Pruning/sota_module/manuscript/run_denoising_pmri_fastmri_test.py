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
    tst_dataset = Subset(tst_dataset, indices=[0, 9, 19, 29, 39, 49, 59, 69, 79, 89])

    # for index in [-2, -1]:
    for index in [-2, -1]:

        table = []

        for method in ['unetres', 'unet']:

            config['method']['deq_cal']['x_module'] = method.replace('-v', '')
            config['method']['deq_cal']['theta_module'] = method.replace('-v', '')

            tmp_ = [method]
            for sigma in SIGMA_LIST:

                with torch.no_grad():

                    net = get_module_from_config(config, 'x' if index == -2 else 'theta', use_sigma_map=True)()

                    load_warmup(
                        net,
                        dataset=config['setting']['dataset'],
                        gt_type='x' if index == -2 else 'theta',
                        pattern='g_denoise',
                        sigma=sigma,
                        prefix='net.',
                        is_print=False,
                        network=method.replace('-v', ''),
                        is_load_state_dict=True
                    )

                    net.cuda()
                    # if method in ['gs_denoiser']:
                    #     net.net.cuda()

                    psnr, ssim = [], []

                    # for i in range(len(tst_dataset)):
                    for i in tqdm.tqdm(range(len(tst_dataset)), desc="idx=[%d] sigma=[%d] method=[%s]" % (index, sigma, method)):

                        x_gt = tst_dataset[i][index]

                        x = x_gt + torch.randn(size=x_gt.shape, dtype=x_gt.dtype, device=x_gt.device) * sigma / 255

                        x = torch.view_as_real(x)
                        if index == -2:
                            x = rearrange(x, 'w h c -> 1 c w h')
                        else:
                            x = rearrange(x, 'b w h c -> b c w h')

                        noise_level_map = torch.FloatTensor(x.size(0), 1, x.size(2), x.size(3)).fill_(
                            sigma / 255).to(x.device)
                        x = torch.cat((x, noise_level_map), 1)

                        pad = 0
                        if x.shape[-1] == 396:
                            pad = 4

                        if pad > 0:
                            x = torch.nn.functional.pad(x, [pad, 0])

                        x_pre = net(x.cuda()).cpu()

                        if pad > 0:
                            x_pre = x_pre[..., pad:]

                        x_pre = rearrange(x_pre, 'b c w h -> b w h c')
                        x_pre = x_pre[..., 0] + x_pre[..., 1] * 1j

                        x_pre = torch.squeeze(x_pre, 0)

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
