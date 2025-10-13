import torch
import os
import pytorch_lightning as pl
from einops import rearrange
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
import random

# warmup_list = {
#     'aapm_ct': {
#
#         'x': {
#             'denoise_2': '20221117_x_warmup_aapm_ct_ct_sigma_2/epoch=599.ckpt',
#             'denoise_5': '20221117_x_warmup_aapm_ct_ct_sigma_5/epoch=599.ckpt',
#             'denoise_7': '20221117_x_warmup_aapm_ct_ct_sigma_7/epoch=499.ckpt',
#             'denoise_10': '20221117_x_warmup_aapm_ct_ct_sigma_10/epoch=399.ckpt',
#             'denoise_15': '20221117_x_warmup_aapm_ct_ct_sigma_15/epoch=299.ckpt'
#         }
#     },
#
#
#     'pmri_modl': {
#         'x': {
#             'denoise_2': '20221122_x_warmup_MoDL_pmri_sigma2/epoch=499.ckpt',
#             'denoise_5': '20221122_x_warmup_MoDL_pmri_sigma5/epoch=499.ckpt',
#             'denoise_7': '20221122_x_warmup_MoDL_pmri_sigma7/epoch=499.ckpt',
#             'denoise_10': '20221122_x_warmup_MoDL_pmri_sigma10/epoch=499.ckpt',
#
#             'gs_DRUNET': '20221209_GS_Denoiser_Demo_LR1eN4_grad_matching_True_DynamicSigma_RERUN_Batch8_Refined0.001/epoch=008_val_loss=0.0001852002169471234.ckpt'
#         },
#
#         'theta': {
#             'denoise_2': '20221122_cal_warmup_MoDL_pmri_sigma2/epoch=299.ckpt',
#             'denoise_5': '20221122_cal_warmup_MoDL_pmri_sigma5/epoch=299.ckpt',
#             'denoise_7': '20221122_cal_warmup_MoDL_pmri_sigma7/epoch=299.ckpt',
#             'denoise_10': '20221122_cal_warmup_MoDL_pmri_sigma10/epoch=299.ckpt',
#
#             'gs_DRUNET': '20221210_cal_warmup_gs_denoiser_gradient_matching_true_nc_16_compute_smps_esp_gradient_clip_val_0.01_stochastic_coil/epoch=909_val_loss=6.118460078141652e-06.ckpt'
#         }
#     }
# }


# warmup_list = {
#
#     'pmri_modl': {
#         'x': {
#             'denoise_1': '20230119_warmup_collections_image_batch_size[8]/ray_tune_run/ray_tune_run_76084_00000_0_x_sigma=1_2023-01-18_17-56-40/epoch=292_val_loss=1.1589214409468696e-05.ckpt',
#             'denoise_2': '20230119_warmup_collections_image_batch_size[8]/ray_tune_run/ray_tune_run_76084_00001_1_x_sigma=2_2023-01-18_17-56-44/epoch=297_val_loss=2.413324182271026e-05.ckpt',
#             'denoise_3': '20230119_warmup_collections_image_batch_size[8]/ray_tune_run/ray_tune_run_76084_00002_2_x_sigma=3_2023-01-18_17-56-44/epoch=295_val_loss=4.022785651613958e-05.ckpt',
#             'denoise_4': '20230119_warmup_collections_image_batch_size[8]/ray_tune_run/ray_tune_run_76084_00003_3_x_sigma=4_2023-01-18_20-10-25/epoch=297_val_loss=6.520637543871999e-05.ckpt',
#             'denoise_5': '20230119_warmup_collections_image_batch_size[8]/ray_tune_run/ray_tune_run_76084_00004_4_x_sigma=5_2023-01-18_20-28-07/epoch=299_val_loss=8.733104914426804e-05.ckpt',
#             'denoise_6': '20230119_warmup_collections_image_batch_size[8]/ray_tune_run/ray_tune_run_76084_00005_5_x_sigma=6_2023-01-18_20-34-25/epoch=297_val_loss=0.00010807000944623724.ckpt',
#             'denoise_7': '20230119_warmup_collections_image_batch_size[8]/ray_tune_run/ray_tune_run_76084_00006_6_x_sigma=7_2023-01-18_22-43-23/epoch=299_val_loss=0.00013200419198255986.ckpt',
#             'denoise_8': '20230119_warmup_collections_image_batch_size[8]/ray_tune_run/ray_tune_run_76084_00007_7_x_sigma=8_2023-01-18_22-52-59/epoch=299_val_loss=0.00015116769645828754.ckpt',
#             'denoise_9': '20230119_warmup_collections_image_batch_size[8]/ray_tune_run/ray_tune_run_76084_00008_8_x_sigma=9_2023-01-18_23-01-42/epoch=296_val_loss=0.00017141997523140162.ckpt',
#             'denoise_10': '20230119_warmup_collections_image_batch_size[8]/ray_tune_run/ray_tune_run_76084_00009_9_x_sigma=10_2023-01-19_00-55-44/epoch=297_val_loss=0.00019197714573238045.ckpt',
#             'denoise_11': '20230119_warmup_collections_image_batch_size[8]/ray_tune_run/ray_tune_run_76084_00010_10_x_sigma=11_2023-01-19_01-22-56/epoch=295_val_loss=0.00021258278866298497.ckpt',
#             'denoise_12': '20230119_warmup_collections_image_batch_size[8]/ray_tune_run/ray_tune_run_76084_00011_11_x_sigma=12_2023-01-19_01-50-45/epoch=297_val_loss=0.00022897250892128795.ckpt',
#             'denoise_13': '20230119_warmup_collections_image_batch_size[8]/ray_tune_run/ray_tune_run_76084_00012_12_x_sigma=13_2023-01-19_02-59-48/epoch=296_val_loss=0.0002453632478136569.ckpt',
#             'denoise_14': '20230119_warmup_collections_image_batch_size[8]/ray_tune_run/ray_tune_run_76084_00013_13_x_sigma=14_2023-01-19_04-08-37/epoch=297_val_loss=0.0002642155741341412.ckpt',
#             'denoise_15': '20230119_warmup_collections_image_batch_size[8]/ray_tune_run/ray_tune_run_76084_00014_14_x_sigma=15_2023-01-19_04-36-19/epoch=297_val_loss=0.0002800650545395911.ckpt',
#         },
#
#         'theta': {
#             'denoise_1': '20230119_warmup_collections_theta_batch_size[1]/ray_tune_run/ray_tune_run_9a796_00000_0_theta_sigma=1_2023-01-18_17-57-42/epoch=274_val_loss=4.187303304092893e-08.ckpt',
#             'denoise_2': '20230119_warmup_collections_theta_batch_size[1]/ray_tune_run/ray_tune_run_9a796_00001_1_theta_sigma=2_2023-01-18_17-57-47/epoch=296_val_loss=7.663975765126452e-08.ckpt',
#             'denoise_3': '20230119_warmup_collections_theta_batch_size[1]/ray_tune_run/ray_tune_run_9a796_00002_2_theta_sigma=3_2023-01-18_17-57-52/epoch=248_val_loss=9.908215048426428e-08.ckpt',
#             'denoise_4': '20230119_warmup_collections_theta_batch_size[1]/ray_tune_run/ray_tune_run_9a796_00003_3_theta_sigma=4_2023-01-19_03-26-40/epoch=297_val_loss=1.2948777339261142e-07.ckpt',
#             'denoise_5': '20230119_warmup_collections_theta_batch_size[1]/ray_tune_run/ray_tune_run_9a796_00004_4_theta_sigma=5_2023-01-19_03-30-14/epoch=296_val_loss=1.6875333130883519e-07.ckpt',
#             'denoise_6': '20230119_warmup_collections_theta_batch_size[1]/ray_tune_run/ray_tune_run_9a796_00005_5_theta_sigma=6_2023-01-19_03-35-25/epoch=274_val_loss=1.8314354122139775e-07.ckpt',
#             'denoise_7': '20230119_warmup_collections_theta_batch_size[1]/ray_tune_run/ray_tune_run_9a796_00006_6_theta_sigma=7_2023-01-19_12-41-53/epoch=295_val_loss=2.257677351735765e-07.ckpt',
#             'denoise_8': '20230119_warmup_collections_theta_batch_size[1]/ray_tune_run/ray_tune_run_9a796_00007_7_theta_sigma=8_2023-01-19_12-49-09/epoch=282_val_loss=2.4022466504902695e-07.ckpt',
#             'denoise_9': '20230119_warmup_collections_theta_batch_size[1]/ray_tune_run/ray_tune_run_9a796_00008_8_theta_sigma=9_2023-01-19_13-02-21/epoch=282_val_loss=2.6655661145014165e-07.ckpt',
#             'denoise_10': '20230119_warmup_collections_theta_batch_size[1]/ray_tune_run/ray_tune_run_9a796_00009_9_theta_sigma=10_2023-01-19_21-52-07/epoch=282_val_loss=2.9868547812839097e-07.ckpt',
#             'denoise_11': '20230119_warmup_collections_theta_batch_size[1]/ray_tune_run/ray_tune_run_9a796_00010_10_theta_sigma=11_2023-01-19_22-04-28/epoch=295_val_loss=3.154602268296003e-07.ckpt',
#             'denoise_12': '20230119_warmup_collections_theta_batch_size[1]/ray_tune_run/ray_tune_run_9a796_00011_11_theta_sigma=12_2023-01-19_22-25-08/epoch=282_val_loss=3.5660912089952035e-07.ckpt',
#             'denoise_13': '20230119_warmup_collections_theta_batch_size[1]/ray_tune_run/ray_tune_run_9a796_00012_12_theta_sigma=13_2023-01-20_07-12-52/epoch=276_val_loss=4.111422242658591e-07.ckpt',
#             'denoise_14': '20230119_warmup_collections_theta_batch_size[1]/ray_tune_run/ray_tune_run_9a796_00013_13_theta_sigma=14_2023-01-20_07-30-47/epoch=296_val_loss=4.337205439242098e-07.ckpt',
#             'denoise_15': '20230119_warmup_collections_theta_batch_size[1]/ray_tune_run/ray_tune_run_9a796_00014_14_theta_sigma=15_2023-01-20_07-57-05/epoch=299_val_loss=4.868361429544166e-07.ckpt',
#         }
#     }
# }


warmup_list = {

    # 'pmri_modl': {
    #     'x': {
    #         'unet': {
    #             'denoise_1': '20230128_warmup_x_pmri_RERUN/ray_tune_run/ray_tune_run_63937_00006_6_x_sigma=1,is_spe_norm=False_2023-01-29_01-11-24/epoch=497_val_loss=8.357998012797907e-06.ckpt',
    #             'denoise_3': '20230128_warmup_x_pmri_RERUN/ray_tune_run/ray_tune_run_63937_00007_7_x_sigma=3,is_spe_norm=False_2023-01-29_01-15-10/epoch=498_val_loss=3.9397167711285874e-05.ckpt',
    #             'denoise_5': '20230128_warmup_x_pmri_RERUN/ray_tune_run/ray_tune_run_63937_00008_8_x_sigma=5,is_spe_norm=False_2023-01-29_01-16-08/epoch=498_val_loss=8.331829303642735e-05.ckpt',
    #             'denoise_7': '20230128_warmup_x_pmri_RERUN/ray_tune_run/ray_tune_run_63937_00009_9_x_sigma=7,is_spe_norm=False_2023-01-29_02-49-32/epoch=495_val_loss=0.00012371211778372526.ckpt',
    #             'denoise_10': '20230128_warmup_x_pmri_RERUN/ray_tune_run/ray_tune_run_63937_00010_10_x_sigma=10,is_spe_norm=False_2023-01-29_02-56-04/epoch=498_val_loss=0.00017984042642638087.ckpt',
    #             'denoise_15': '20230128_warmup_x_pmri_RERUN/ray_tune_run/ray_tune_run_63937_00011_11_x_sigma=15,is_spe_norm=False_2023-01-29_03-12-08/epoch=499_val_loss=0.00027169391978532076.ckpt',
    #
    #             'g_denoise': '20230207_warmup_x_pmri_g_denoise_unet_sigma=25_RERUN_epoch[3000]/epoch=829_val_loss=6.498958100564778e-05.ckpt'
    #         },
    #
    #         'dncnn': {
    #             'g_denoise': '20230207_warmup_x_pmri_g_denoise_dncnn_sigma=25_RERUN_epoch[3000]/epoch=829_val_loss=6.668602145509794e-05.ckpt'
    #         },
    #
    #         'unetres': {
    #             'g_denoise': '20230218_warmup_x_pmri_g_denoise_unetres_sigma=25_RERUN_epoch[3000]/epoch=829_val_loss=7.343234756262973e-05.ckpt',
    #         }
    #     },
    #
    #     'theta':
    #         {
    #             'unet': {
    #                 'denoise_1': '20230128_warmup_theta_pmri_RERUN/ray_tune_run/ray_tune_run_38d11_00006_6_theta_sigma=1,is_spe_norm=False_2023-01-29_04-25-56/epoch=499_val_loss=5.493902790476568e-07.ckpt',
    #                 'denoise_3': '20230128_warmup_theta_pmri_RERUN/ray_tune_run/ray_tune_run_38d11_00007_7_theta_sigma=3,is_spe_norm=False_2023-01-29_04-27-06/epoch=499_val_loss=1.1865950000355951e-06.ckpt',
    #                 'denoise_5': '20230128_warmup_theta_pmri_RERUN/ray_tune_run/ray_tune_run_38d11_00008_8_theta_sigma=5,is_spe_norm=False_2023-01-29_10-39-49/epoch=491_val_loss=1.6651260921207722e-06.ckpt',
    #                 'denoise_7': '20230128_warmup_theta_pmri_RERUN/ray_tune_run/ray_tune_run_38d11_00009_9_theta_sigma=7,is_spe_norm=False_2023-01-29_10-43-01/epoch=491_val_loss=2.3465765934815863e-06.ckpt',
    #                 'denoise_10': '20230128_warmup_theta_pmri_RERUN/ray_tune_run/ray_tune_run_38d11_00010_10_theta_sigma=10,is_spe_norm=False_2023-01-29_10-47-13/epoch=496_val_loss=3.7015113321103854e-06.ckpt',
    #                 'denoise_15': '20230128_warmup_theta_pmri_RERUN/ray_tune_run/ray_tune_run_38d11_00011_11_theta_sigma=15,is_spe_norm=False_2023-01-29_10-50-26/epoch=485_val_loss=6.475591817434179e-06.ckpt',
    #
    #                 'g_denoise': '20230207_warmup_theta_pmri_g_denoise_unet_sigma=25_RERUN_epoch[3000]/epoch=2605_val_loss=3.2253328754450195e-06.ckpt'
    #             },
    #
    #             'dncnn': {
    #                 'g_denoise': '20230207_warmup_theta_pmri_g_denoise_dncnn_sigma=25_RERUN_epoch[3000]/epoch=2772_val_loss=4.545771389530273e-06.ckpt'
    #             },
    #
    #             'unetres':{
    #                 'g_denoise': '20230218_warmup_theta_pmri_g_denoise_unetres_sigma=25_RERUN_epoch[3000]_nc_16/epoch=2547_val_loss=3.6682322388514876e-06.ckpt',
    #             }
    #     }
    # },

    'pmri_fastmri': {
        'x': {
                'unet': {
                    'g_denoise': '20230403_warmup_x_pmri_pmri_fastmri_unet_tiny_g_denoise/epoch=165_val_loss=5.126764790475136e-06.ckpt'
                },

                'unetres': {
                    'g_denoise': '20230402_warmup_x_pmri_pmri_fastmri_x_unetres_g_denoise_RERUN_V2/epoch=165_val_loss=3.789535639953101e-06.ckpt',
                }
        },

        'theta':
            {
                'unet': {
                    'g_denoise': '20230403_warmup_theta_pmri_pmri_fastmri_unet_tiny_g_denoise/epoch=077_val_loss=1.0197123629041016e-05.ckpt'
                },

                'unetres': {
                    'g_denoise': '20230402_warmup_theta_pmri_pmri_fastmri_unetres_g_denoise_RERUN_V2/epoch=138_val_loss=6.743206313331029e-07.ckpt',
                }
            }
    },

    'natural': {
        'x': {

            'unetres': {
                'g_denoise': 'model_zoo/drunet_gray.pth'
            },

            'gs_denoiser': {
                'g_denoise': 'model_zoo/GSDRUNet_grayscale.ckpt'
            }
        },

        'theta': {

            'dncnn': {
                'denoise_0.01': '20230317_warmup_theta_deconv_denoise/ray_tune_run/ray_tune_run_438c7_00000_0_theta_sigma=0.0100_2023-03-17_06-16-33/epoch=2978_val_loss=9.874592116210223e-11.ckpt',
                'denoise_0.05': '20230317_warmup_theta_deconv_denoise/ray_tune_run/ray_tune_run_438c7_00001_1_theta_sigma=0.0500_2023-03-17_06-16-37/epoch=2740_val_loss=4.848927370737499e-10.ckpt',
                'denoise_0.10': '20230317_warmup_theta_deconv_denoise/ray_tune_run/ray_tune_run_438c7_00002_2_theta_sigma=0.1000_2023-03-17_06-16-37/epoch=2670_val_loss=9.695483305094399e-10.ckpt',
                'denoise_0.50': '20230317_warmup_theta_deconv_denoise/ray_tune_run/ray_tune_run_438c7_00003_3_theta_sigma=0.5000_2023-03-17_06-16-37/epoch=2886_val_loss=1.0822108542640763e-08.ckpt',
                'denoise_1': '20230317_warmup_theta_deconv_denoise/ray_tune_run/ray_tune_run_438c7_00004_4_theta_sigma=1_2023-03-18_21-15-22/epoch=2001_val_loss=3.5147113663924756e-08.ckpt',
                'denoise_1.50': '20230317_warmup_theta_deconv_denoise/ray_tune_run/ray_tune_run_438c7_00005_5_theta_sigma=1.5000_2023-03-18_21-45-41/epoch=2962_val_loss=6.762126503190302e-08.ckpt',
                'denoise_2': '20230317_warmup_theta_deconv_denoise/ray_tune_run/ray_tune_run_438c7_00006_6_theta_sigma=2_2023-03-18_22-11-27/epoch=2962_val_loss=1.1687393453030381e-07.ckpt',
                'denoise_2.50': '20230317_warmup_theta_deconv_denoise/ray_tune_run/ray_tune_run_438c7_00007_7_theta_sigma=2.5000_2023-03-19_01-11-15/epoch=2619_val_loss=1.7132539653630374e-07.ckpt',
                'denoise_3': '20230317_warmup_theta_deconv_denoise/ray_tune_run/ray_tune_run_438c7_00008_8_theta_sigma=3_2023-03-20_14-48-47/epoch=2619_val_loss=2.370624656578002e-07.ckpt',
            },

        }
    }
}


def addwgn(x: torch.Tensor, input_snr):
    noiseNorm = torch.norm(x.flatten()) * 10 ** (-input_snr / 20)

    noise = torch.randn(x.size()).to(x.device)

    noise = noise / torch.norm(noise.flatten()) * noiseNorm

    y = x + noise
    return y


def load_warmup(target_module, dataset, gt_type, pattern, sigma, prefix, is_print=True, network='unet', is_load_state_dict=True):

    if pattern == 'denoise':
        if isinstance(sigma, int):
            x_ckpt = warmup_list[dataset][gt_type][network]["%s_%d" % (pattern, sigma)]
        else:
            x_ckpt = warmup_list[dataset][gt_type][network]["%s_%.2f" % (pattern, sigma)]
    else:
        x_ckpt = warmup_list[dataset][gt_type][network][pattern]

    if is_print:
        print("Loading ckpt from", x_ckpt)

    if is_load_state_dict:
        x_ckpt = torch.load(os.path.join('/opt/experiment', x_ckpt))['state_dict']
    else:
        x_ckpt = torch.load(os.path.join('/opt/experiment', x_ckpt))

    x_self = target_module.state_dict()

    # for name, param in x_ckpt.items():
    #     print(name)
    #
    # print("=====")
    #
    # for name, param in x_self.items():
    #     print(name)
    #
    # exit(0)

    for name, param in x_self.items():

        name_ckpt = prefix + name

        if name_ckpt not in x_ckpt:
            raise ValueError('cannot find %s in the checkpoint' % name_ckpt)

        param_ckpt = x_ckpt[name_ckpt]
        if isinstance(param_ckpt, torch.nn.parameter.Parameter):
            # backwards compatibility for serialized parameters
            param_ckpt = param_ckpt.data

        x_self[name].copy_(param_ckpt)


class DenoiserBase(pl.LightningModule):
    def __init__(self, net, sigma, lr, is_g_denoise):
        super().__init__()

        self.sigma = sigma / 255
        self.is_g_denoise = is_g_denoise

        self.lr = lr

        self.net = net()
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x, sigma=None):
        is_complex = False

        if x.dtype == torch.complex64:
            is_complex = True

        if is_complex:
            x = torch.view_as_real(x)
            x = rearrange(x, 'b w h c -> b c w h')

        pad = 0
        if x.shape[-1] == 396:
            pad = 4

        if pad > 0:
            x = torch.nn.functional.pad(x, [pad, 0])

        if sigma is not None:
            noise_level_map = torch.FloatTensor(x.size(0), 1, x.size(2), x.size(3)).fill_(sigma).to(self.device)
            x = torch.cat((x, noise_level_map), 1)

            x_hat = self.net(x)
        else:
            x_hat = self.net(x)

        if pad > 0:
            x_hat = x_hat[..., pad:]

        if is_complex:
            x_hat = rearrange(x_hat, 'b c w h -> b w h c')
            x_hat = x_hat[..., 0] + x_hat[..., 1] * 1j

        return x_hat

    def get_groundtruth_from_batch(self, batch):
        pass

    def step_helper(self, batch):
        x = self.get_groundtruth_from_batch(batch)

        is_complex = False
        if x.dtype == torch.complex64:
            is_complex = True

        if self.is_g_denoise:
            sigma = random.uniform(0, self.sigma)
        else:
            sigma = self.sigma

        x0 = x + torch.randn(size=x.shape, dtype=x.dtype, device=x.device) * sigma

        if self.is_g_denoise:
            x_hat = self(x0, sigma)
        else:
            x_hat = self(x0)

        if is_complex:
            x = torch.view_as_real(x)
            x_hat = torch.view_as_real(x_hat)

        # loss = self.loss_fn(x, x_hat)
        # loss = torch.nn.functional.mse_loss(x.clone(), x_hat)
        loss = torch.mean((x - x_hat) ** 2)

        if is_complex:
            x = torch.view_as_complex(x)
            x_hat = torch.view_as_complex(x_hat)

        return x, x0, x_hat, loss

    def training_step(self, batch, batch_idx):
        x, x0, x_hat, loss = self.step_helper(batch)

        tra_psnr, tra_ssim = self.psnr_ssim_helper(x_hat, x, 1)

        self.log(name='tra_psnr', value=tra_psnr, prog_bar=True)
        self.log(name='tra_ssim', value=tra_ssim, prog_bar=True)

        if batch_idx == 0:
            self.logger.log_image(key='tra_x_hat', images=[self.to_two_dim_magnitude_image(x_hat)])
            self.logger.log_image(key='tra_x_gt', images=[self.to_two_dim_magnitude_image(x)])
            self.logger.log_image(key='tra_x_noisy', images=[self.to_two_dim_magnitude_image(x0)])

        return loss

    def validation_step(self, batch, batch_idx):
        x, x0, x_hat, loss = self.step_helper(batch)

        val_psnr, val_ssim = self.psnr_ssim_helper(x_hat, x, 1)

        self.log(name='val_psnr', value=val_psnr, prog_bar=True)
        self.log(name='val_ssim', value=val_ssim, prog_bar=True)
        self.log(name='val_loss', value=loss, prog_bar=True)

        if batch_idx == 0:
            self.logger.log_image(key='val_x_hat', images=[self.to_two_dim_magnitude_image(x_hat)])
            self.logger.log_image(key='val_x_gt', images=[self.to_two_dim_magnitude_image(x)])
            self.logger.log_image(key='val_x_noisy', images=[self.to_two_dim_magnitude_image(x0)])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        return optimizer

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
    def to_two_dim_magnitude_image(x):
        if x.dtype == torch.complex64:
            x = torch.abs(x)

        if x.dim() == 2:
            return x

        elif x.dim() == 3:
            return x[0]

        elif x.dim() == 4:
            return x[0][0]