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

    SIGMA_LIST = [1, 3, 5, 7, 10, 15]

    config['setting']['mode'] = 'tst'

    _, _, tst_dataset = get_dataset_from_config(config)
    tst_dataset = Subset(tst_dataset, [5, 10, 15, 20, 25, 30, 35, 40, 45, 50])

    for index in [-2, -1]:

        x_gt = torch.stack([tst_dataset[i][index] for i in range(len(tst_dataset))], 0)

        if index == -1:
            x_gt = rearrange(x_gt, 'b c w h -> (b c) w h')

        table = []

        for method in ['unetres-v', 'dncnn-v', 'unet', 'unet-v']:

            config['method']['deq_cal']['x_module'] = method.replace('-v', '')
            config['method']['deq_cal']['theta_module'] = method.replace('-v', '')

            tmp_ = [method]
            for sigma in tqdm.tqdm(SIGMA_LIST, desc="idx=[%d] method=[%s]" % (index, method)):

                with torch.no_grad():

                    x_noisy = x_gt + torch.randn(size=x_gt.shape, dtype=x_gt.dtype, device=x_gt.device) * sigma / 255

                    if 'v' not in method:
                        net = get_module_from_config(config, 'x' if index == -2 else 'theta', use_sigma_map=False)()
                        net.cuda()

                        load_warmup(
                            net,
                            dataset=config['setting']['dataset'],
                            gt_type='x' if index == -2 else 'theta',
                            pattern='denoise',
                            sigma=sigma,
                            prefix='net.',
                            is_print=False,
                            network=method.replace('-v', '')
                        )

                        x = torch.view_as_real(x_noisy)
                        x = rearrange(x, 'b w h c -> b c w h')

                        x_pre = net(x.cuda()).cpu()

                        x_pre = rearrange(x_pre, 'b c w h -> b w h c')
                        x_pre = x_pre[..., 0] + x_pre[..., 1] * 1j

                        psnr, ssim = DEQCalibrationMRI.psnr_ssim_helper(x_pre, x_gt, 1)

                    else:

                        net = get_module_from_config(config, 'x' if index == -2 else 'theta', use_sigma_map=True)()
                        net.cuda()
                        load_warmup(
                            net,
                            dataset=config['setting']['dataset'],
                            gt_type='x' if index == -2 else 'theta',
                            pattern='g_denoise',
                            sigma=sigma,
                            prefix='net.',
                            is_print=False,
                            network=method.replace('-v', '')
                        )

                        x = torch.view_as_real(x_noisy)
                        x = rearrange(x, 'b w h c -> b c w h')

                        noise_level_map = torch.FloatTensor(x.size(0), 1, x.size(2), x.size(3)).fill_(sigma / 255).to(x.device)
                        x = torch.cat((x, noise_level_map), 1)

                        x_pre = net(x.cuda()).cpu()

                        x_pre = rearrange(x_pre, 'b c w h -> b w h c')
                        x_pre = x_pre[..., 0] + x_pre[..., 1] * 1j

                        psnr, ssim = DEQCalibrationMRI.psnr_ssim_helper(x_pre, x_gt, 1)

                    tmp_.append('%.2f / %.3f' % (psnr, ssim))

            table.append(tmp_)

        if index == -1:
            print("theta map denoising")
        else:
            print("x denoising")

        print(tabulate.tabulate(table, headers=['sigma'] + SIGMA_LIST, tablefmt="rst"))