# file to define constants
import os
PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))

DOWNLOAD_LINKS = {
    "sample_data":"https://api.github.com/repos/GillesVanDeVyver/us_cardiac_sample_data/contents/sample_data",
    "sample_data_segmentations":"https://api.github.com/repos/GillesVanDeVyver/us_cardiac_sample_data/contents/sample_data_seg",
    "CAMUS_diffusion_model": ("gillesvdv/CAMUS_diffusion_model","CAMUS_diffusion_model.pt"),
}

MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

DEFAULT_CONFIG = {
        'attention_resolutions': '16,8',
        'class_cond': False,
        'diffusion_steps': 4000,
        'learn_sigma': True,
        'noise_schedule': 'linear',
        'num_channels': 64,
        'num_head_channels': 1,
        'num_heads': 4,
        'num_res_blocks': 3,
        'resblock_updown': False,
        'use_fp16': True,
        'use_scale_shift_norm': True,
        'classifier_scale': 4.0,
        'lr_kernel_n_std': 2,
        'num_samples': 8,
        'show_progress': True,
        'timestep_respacing': '250',
        'use_kl': False,
        'predict_xstart': False,
        'rescale_timesteps': False,
        'rescale_learned_sigmas': False,
        'num_heads_upsample': -1,
        'channel_mult': '',
        'dropout': 0.0,
        'use_checkpoint': False,
        'use_new_attention_order': False,
        'clip_denoised': True,
        'use_ddim': False,
        'image_size': 256,
        'model_path': 'TO_BE_FILLED',
        'name': 'camus_all',
        'inpa_inj_sched_prev': True,
        'n_jobs': 25,
        'print_estimated_vars': True,
        'inpa_inj_sched_prev_cumnoise': False,
        'schedule_jump_params': {
            't_T': 250,
            'n_sample': 1,
            'jump_length': 10,
            'jump_n_sample': 10
        },
        'data': {
            'eval': {
                'inference': {
                    'mask_loader': True,
                    'gt_path': 'TO_BE_FILLED',
                    'mask_path': 'TO_BE_FILLED',
                    'image_size': 256,
                    'class_cond': False,
                    'deterministic': True,
                    'random_crop': False,
                    'random_flip': False,
                    'return_dict': True,
                    'drop_last': False,
                    'batch_size': 32,
                    'return_dataloader': True,
                    'ds_conf': {
                        'name': 'debug_conf'
                    },
                    'max_len': 0,
                    'paths': {
                        'srs': 'TO_BE_FILLED',
                        'lrs': 'TO_BE_FILLED',
                        'gts': 'TO_BE_FILLED',
                        'gt_keep_masks': 'TO_BE_FILLED'
                    }
                }
            }
        }
    }