"""Experiment Settings"""
n_measure_64 = [100, 200, 300, 400, 600, 1000, 2000, 3000, 4000, 6000]
n_measure_128 = [400, 800, 1200, 1600, 2400, 4000, 8000, 12000, 16000, 24000]

forward_models = {
    'began_cs': {
        'GaussianCompressiveSensing': [{
            'n_measure': x
        } for x in n_measure_128]
    },
    'began_cs_n_cuts': {
        'GaussianCompressiveSensing': [{
            'n_measure': x
        } for x in n_measure_128]
    },
    'began_cs_other_init': {
        'GaussianCompressiveSensing': [{
            'n_measure': 8000
        }],
    },
    'began_inv': {
        'InpaintingScatter': [{
            'fraction_kept': 0.1
        }],
        'SuperResolution': [{
            'scale_factor': 0.25,
            'mode': 'bilinear',
            'align_corners': True
        }],
    },
    'began_noop': {
        'NoOp': [{}],
    },
    'began_opt_error_fake_imgs': {
        'NoOp': [{}],
    },
    'began_untrained_cs': {
        'GaussianCompressiveSensing': [{
            'n_measure': x
        } for x in n_measure_128]
    },
    'began_restarts_cs': {
        'GaussianCompressiveSensing': [{
            'n_measure': x
        } for x in n_measure_128]
    },
    'biggan_inv': {
        'InpaintingScatter': [{
            'fraction_kept': 0.1
        }],
        'SuperResolution': [{
            'scale_factor': 0.25,
            'mode': 'bilinear',
            'align_corners': True
        }],
    },
    'biggan_noop': {
        'NoOp': [{}],
    },
    'dcgan_cs': {
        'GaussianCompressiveSensing': [{
            'n_measure': x
        } for x in n_measure_64]
    },
    'dcgan_cs_n_cuts': {
        'GaussianCompressiveSensing': [{
            'n_measure': x
        } for x in n_measure_64]
    },
    'dcgan_inv': {
        'InpaintingScatter': [{
            'fraction_kept': 0.1
        }],
        'SuperResolution': [{
            'scale_factor': 0.25,
            'mode': 'bilinear',
            'align_corners': True
        }],
    },
    'dcgan_noop': {
        'NoOp': [{}],
    },
    'dcgan_untrained_cs': {
        'GaussianCompressiveSensing': [{
            'n_measure': x
        } for x in n_measure_64]
    },
    'vanilla_vae_cs': {
        'GaussianCompressiveSensing': [{
            'n_measure': x
        } for x in n_measure_128]
    },
    'vanilla_vae_cs_n_cuts': {
        'GaussianCompressiveSensing': [{
            'n_measure': x
        } for x in n_measure_128]
    },
    'vanilla_vae_inv': {
        'InpaintingScatter': [{
            'fraction_kept': 0.1
        }],
        'SuperResolution': [{
            'scale_factor': 0.25,
            'mode': 'bilinear',
            'align_corners': True
        }],
    },
    'vanilla_vae_noop': {
        'NoOp': [{}],
    },
    'vanilla_vae_untrained_cs': {
        'GaussianCompressiveSensing': [{
            'n_measure': x
        } for x in n_measure_128]
    },
    'mgan_began_cs': {
        'GaussianCompressiveSensing': [{
            'n_measure': x
        } for x in n_measure_128]
    },
    'mgan_vanilla_vae_cs': {
        'GaussianCompressiveSensing': [{
            'n_measure': x
        } for x in n_measure_128]
    },
    'mgan_dcgan_cs': {
        'GaussianCompressiveSensing': [{
            'n_measure': x
        } for x in n_measure_64]
    },
    'iagan_dcgan_cs': {
        'GaussianCompressiveSensing': [{
            'n_measure': x
        } for x in n_measure_64]
    },
    'iagan_began_cs': {
        'GaussianCompressiveSensing': [{
            'n_measure': x
        } for x in n_measure_128]
    },
    'iagan_vanilla_vae_cs': {
        'GaussianCompressiveSensing': [{
            'n_measure': x
        } for x in n_measure_128]
    },
    'deep_decoder_64_cs': {
        'GaussianCompressiveSensing': [{
            'n_measure': x
        } for x in n_measure_64]
    },
    'deep_decoder_128_cs': {
        'GaussianCompressiveSensing': [{
            'n_measure': x
        } for x in n_measure_128]
    },
}

baseline_settings = {
    'lasso-dct-64': {
        'img_size': 64,
        'lasso_coeff': [0.01] * len(n_measure_64),
        'n_measure': n_measure_64,
    },
    'lasso-dct-128': {
        'img_size': 128,
        'lasso_coeff': [0.01] * len(n_measure_128),
        'n_measure': n_measure_128,
    },
}

recovery_settings = {
    'began_cs': {
        'optimizer': 'lbfgs',
        'n_steps': 25,
        'z_lr': 1,
        'z_init_mode': ['clamped_normal'],
        'restarts': 3,
        'n_cuts_list': [0, 2],
        'limit': [1],
    },
    'began_cs_n_cuts': {
        'optimizer': 'lbfgs',
        'n_steps': 25,
        'z_lr': 1,
        'z_init_mode': ['clamped_normal'],
        'restarts': 3,
        'n_cuts_list': [0, 1, 2, 3, 4, 5, 7, 9, 11, 13],
        'limit': [1],
    },
    'began_cs_other_init': {
        'z_init_mode':
        ['lasso_inverse', 'clamped_normal', 'normal', 'normal', 'zero'],
        'limit': [1, 1, 5, 10, None],
        'optimizer':
        'lbfgs',
        'n_steps':
        25,
        'z_lr':
        1,
        'restarts':
        3,
        'n_cuts_list': [2],
    },
    'began_inv': {
        'optimizer': 'lbfgs',
        'n_steps': 25,
        'z_lr': 1,
        'z_init_mode': ['clamped_normal'],
        'restarts': 3,
        'n_cuts_list': [0, 2],
        'limit': [1],
    },
    'began_noop': {
        'optimizer': 'lbfgs',
        'n_steps': 100,
        'z_lr': 1,
        'z_init_mode': ['clamped_normal'],
        'restarts': 3,
        'n_cuts_list': [0, 2],
        'limit': [1],
    },
    'began_opt_error_fake_imgs': {
        'z_init_mode': ['clamped_normal'],
        'limit': [1],
        'optimizer': 'lbfgs',
        'n_steps': 100,
        'z_lr': 1,
        'restarts': 3,
        'n_cuts_list': [0, 2],
    },
    'began_untrained_cs': {
        'optimizer': 'lbfgs',
        'n_steps': 25,
        'z_lr': 1,
        'z_init_mode': ['clamped_normal'],
        'restarts': 3,
        'n_cuts_list': [2],
        'limit': [1],
    },
    'began_restarts_cs': {
        'optimizer': 'lbfgs',
        'n_steps': 100,
        'z_lr': 1,
        'z_init_mode': ['clamped_normal'],
        'restarts': 20,
        'n_cuts_list': [0],
        'limit': [1],
    },
    'biggan_inv': {
        'optimizer': 'lbfgs',
        'n_steps': 25,
        'z_lr': 1.5,
        'z_init_mode': ['clamped_normal'],
        'limit': [1],
        'restarts': 3,
        'n_cuts_list': [0, 7],
    },
    'biggan_noop': {
        'optimizer': 'lbfgs',
        'n_steps': 200,
        'z_lr': 1.5,
        'z_init_mode': ['clamped_normal'],
        'limit': [1],
        'restarts': 3,
        'n_cuts_list': [0, 7],
    },
    'dcgan_cs': {
        'optimizer': 'lbfgs',
        'n_steps': 25,
        'z_lr': 0.1,
        'z_init_mode': ['clamped_normal'],
        'restarts': 3,
        'n_cuts_list': [0, 1],
        'limit': [1],
    },
    'dcgan_cs_n_cuts': {
        'optimizer': 'lbfgs',
        'n_steps': 25,
        'z_lr': 0.1,
        'z_init_mode': ['clamped_normal'],
        'restarts': 3,
        'n_cuts_list': [0, 1, 2, 3, 4],
        'limit': [1],
    },
    'dcgan_inv': {
        'optimizer': 'lbfgs',
        'n_steps': 25,
        'z_lr': 0.1,
        'z_init_mode': ['clamped_normal'],
        'restarts': 3,
        'n_cuts_list': [0, 1],
        'limit': [1],
    },
    'dcgan_noop': {
        'optimizer': 'lbfgs',
        'n_steps': 100,
        'z_lr': 1,
        'z_init_mode': ['clamped_normal'],
        'restarts': 3,
        'n_cuts_list': [0, 1],
        'limit': [1],
    },
    'dcgan_untrained_cs': {
        'optimizer': 'lbfgs',
        'n_steps': 25,
        'z_lr': 0.1,
        'z_init_mode': ['clamped_normal'],
        'restarts': 3,
        'n_cuts_list': [1],
        'limit': [1],
    },
    'vanilla_vae_cs': {
        'optimizer': 'lbfgs',
        'n_steps': 25,
        'z_lr': 1,
        'z_init_mode': ['clamped_normal'],
        'restarts': 3,
        'n_cuts_list': [0, 1],
        'limit': [1],
    },
    'vanilla_vae_cs_n_cuts': {
        'optimizer': 'lbfgs',
        'n_steps': 25,
        'z_lr': 1,
        'z_init_mode': ['clamped_normal'],
        'restarts': 3,
        'n_cuts_list': [0, 1, 2, 3, 4, 5],
        'limit': [1],
    },
    'vanilla_vae_inv': {
        'optimizer': 'lbfgs',
        'n_steps': 25,
        'z_lr': 1,
        'z_init_mode': ['clamped_normal'],
        'restarts': 3,
        'n_cuts_list': [0, 1],
        'limit': [1],
    },
    'vanilla_vae_noop': {
        'optimizer': 'lbfgs',
        'n_steps': 40,
        'z_lr': 1,
        'z_init_mode': ['clamped_normal'],
        'restarts': 3,
        'n_cuts_list': [0, 1],
        'limit': [1],
    },
    'vanilla_vae_untrained_cs': {
        'optimizer': 'lbfgs',
        'n_steps': 25,
        'z_lr': 1,
        'z_init_mode': ['clamped_normal'],
        'restarts': 3,
        'n_cuts_list': [1],
        'limit': [1],
    },
    'mgan_dcgan_cs': {
        'optimizer': 'adam',
        'n_steps': 5000,
        'z_lr': 3e-2,
        'z_init_mode': ['clamped_normal'],
        'limit': [1],
        'z_number': 10,
        'restarts': 3,
        'n_cuts_list': [1],
    },
    'mgan_began_cs': {
        'optimizer': 'adam',
        'n_steps': 3000,
        'z_lr': 1e-3,
        'z_init_mode': ['clamped_normal'],
        'limit': [1],
        'z_number': 20,
        'restarts': 3,
        'n_cuts_list': [2],
    },
    'mgan_vanilla_vae_cs': {
        'optimizer': 'adam',
        'n_steps': 3000,
        'z_lr': 5e-3,
        'z_init_mode': ['clamped_normal'],
        'limit': [1],
        'z_number': 20,
        'restarts': 3,
        'n_cuts_list': [1],
    },
    'iagan_dcgan_cs': {
        'optimizer': 'adam',
        'z_steps1': 1600,
        'z_steps2': 600,
        'z_lr1': 0.1,
        'z_lr2': 1e-4,
        'model_lr': 1e-4,
        'restarts': 3,
        'z_init_mode': ['clamped_normal'],
        'limit': [1],
    },
    'iagan_began_cs': {
        'optimizer': 'adam',
        'z_steps1': 1600,
        'z_steps2': 600,
        'z_lr1': 0.1,
        'z_lr2': 1e-4,
        'model_lr': 1e-4,
        'restarts': 3,
        'z_init_mode': ['clamped_normal'],
        'limit': [1],
    },
    'iagan_vanilla_vae_cs': {
        'optimizer': 'adam',
        'z_steps1': 1000,
        'z_steps2': 600,
        'z_lr1': 0.1,
        'z_lr2': 1e-4,
        'model_lr': 1e-4,
        'restarts': 3,
        'z_init_mode': ['clamped_normal'],
        'limit': [1],
    },
    'deep_decoder_64_cs': {
        'optimizer': 'adam',
        'steps': 5000,
        'lr': 1e-2,
        'restarts': 3,
        'depth': 5,
        'num_filters': 44,
        'img_size': 64,
    },
    'deep_decoder_128_cs': {
        'optimizer': 'adam',
        'steps': 5000,
        'lr': 1e-2,
        'restarts': 3,
        'depth': 6,
        'num_filters': 89,
        'img_size': 128,
    },
}
