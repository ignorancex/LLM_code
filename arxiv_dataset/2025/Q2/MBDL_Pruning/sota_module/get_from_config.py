import os
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from sota_module.utility import copy_code_to_path, merge_child_dict
from sota_module.method.module.unet import UNet
from ray import tune
from sota_module.method.module.gs_denoiser.gs_denoiser import GradMatchWG as GradDenoise
from sota_module.fwd.pmri_modl import ParallelMRIMoDL
from sota_module.fwd.pmri_fastmri_brain_wrapper import ParallelMRIFastMRI

from sota_module.method.module.gs_denoiser.models import DNCNN
from sota_module.method.module.gs_denoiser.models.network_unet import UNetRes
from sota_module.dataset.kernel_generator import GenerateGaussianKernel
from sota_module.dataset.natural_image import NaturalImageDataset


def get_save_path_from_config(config):
    save_path = tune.get_trial_dir()

    if save_path is None:
        save_path = os.path.join(config['setting']['root_path'], config['setting']['save_path'])

    return save_path


def get_save_name_from_config(config):
    save_path = get_save_path_from_config(config)
    save_name = save_path.replace(config['setting']['root_path'], '').replace('ray_tune_run/ray_tune_', '').replace("/", "")

    return save_name


def get_trainer_from_config(config):

    save_path = get_save_path_from_config(config)

    callback = []

    if config['setting']['mode'] == 'tra':

        if not config['setting']['mode'] == 'dug':

            os.environ['WANDB_API_KEY'] = '59f82e496fd93bb506e92465eb6ba1d008c0e8ae'
            os.environ['WANDB_CONFIG_DIR'] = '/tmp/wandb_config/'
            os.environ['WANDB_CACHE_DIR'] = '/tmp/wandb_cache/'
            os.environ['WANDB_START_METHOD'] = 'fork'

            import wandb
            wandb.finish()

            logger = WandbLogger(
                project='deq_cal',
                name=get_save_name_from_config(config),
                entity='wjgancn',
                dir='/tmp/' + get_save_name_from_config(config),
                save_dir='/tmp/' + get_save_name_from_config(config),
            )

        else:
            logger = None

        callback.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=save_path,
                monitor='val_loss',
                filename='{epoch:03d}_{val_loss}',
                save_last=True,
                every_n_epochs=config['train']['every_n_epochs'],
                save_top_k=3,
                mode='min'
            )
        )

        trainer = pl.Trainer(
            accelerator='gpu',
            max_epochs=config['train']['max_epochs'],
            default_root_dir=save_path,
            logger=logger,
            callbacks=callback,
            log_every_n_steps=10,
            strategy="ddp",
            inference_mode=False,
            gradient_clip_val=config['train']['gradient_clip_val'],
            detect_anomaly=True
        )

        # copy_code_to_path(src_path='/opt/project/', file_path=save_path)

        if trainer.global_rank == 0 and not config['setting']['mode'] == 'dug':
            logger.experiment.config.update(
                merge_child_dict(config, {}), allow_val_change=True
            )

    elif config['setting']['mode'] == 'tst' or config['setting']['mode'] == 'dug':

        trainer = pl.Trainer(
            accelerator='gpu',
            default_root_dir=save_path,
            log_every_n_steps=10,
            strategy="ddp",
            callbacks=callback,
            inference_mode=False
        )

    else:

        raise ValueError()

    return trainer


def get_module_from_config(config, type_='x', use_sigma_map=False):
    assert type_ in ['x', 'theta']

    if type_ == 'x':
        DRUNET_nc = config['module']['gs_denoiser']['DRUNET_nc_x']
        f_root=config['module']['unet']['f_root_x']
    else:
        DRUNET_nc = config['module']['gs_denoiser']['DRUNET_nc_cal']
        f_root = config['module']['unet']['f_root_theta']

    if 'pmri' in config['setting']['dataset']:
        nc = 2
    else:
        nc = 1

    module_dict = {
        'unet': lambda: UNet(
            dimension=2,
            i_nc=nc + 1 if use_sigma_map else nc,
            o_nc=nc,
            f_root=f_root,
            conv_times=config['module']['unet']['conv_times'],
            up_down_times=config['module']['unet']['up_down_times'],
            is_spe_norm=config['module']['unet']['is_spe_norm'],
        ),

        'dncnn': lambda: DNCNN.dncnn(
            nc + 1 if use_sigma_map else nc, nc, config['module']['dncnn']['num_layers'], 'C', 'r', True
        ),

        'gs_denoiser': lambda: GradDenoise(
            grad_matching=config['module']['gs_denoiser']['grad_matching'],
            channels=nc,
            model_name=config['module']['gs_denoiser']['model_name'],
            DRUNET_nc=DRUNET_nc,
            act_mode=config['module']['gs_denoiser']['act_mode']
        ),

        'unetres': lambda: UNetRes(
            in_nc=nc + 1 if use_sigma_map else nc,
            out_nc=nc,
            nc=[DRUNET_nc, DRUNET_nc * 2, DRUNET_nc * 4, DRUNET_nc * 8],
            nb=4,
            act_mode='R',
            downsample_mode='strideconv',
            upsample_mode='convtranspose')
    }

    return module_dict[config['method']['deq_cal'][type_ + '_module']]


def get_dataset_from_config(config):

    dataset_dict = {
        'pmri_modl': lambda mode, is_pre_load: ParallelMRIMoDL(
            mode=mode,
            root_path=config['dataset']['pmri_modl']['root_path'],
            acceleration_rate=config['dataset']['pmri_modl']['acceleration_rate'],
            noise_snr=config['dataset']['pmri_modl']['noise_snr'],
            num_of_coil=config['dataset']['pmri_modl']['num_of_coil'],
            is_pre_load=is_pre_load,
            mask_pattern='uniformly_cartesian',
            birdcage_maps_dim=config['dataset']['pmri_modl']['birdcage_maps_dim'],
            smps_hat_method=config['dataset']['pmri_modl']['smps_hat_method'],
            acs_percentage=config['dataset']['pmri_modl']['acs_percentage'],
            randomly_return=config['dataset']['pmri_modl']['randomly_return'],
            low_k_size=config['dataset']['pmri_modl']['low_k_size']
        ),

        'deconv_kernel': lambda mode, is_pre_load: GenerateGaussianKernel(
            mode=mode,
            root_path=config['dataset']['deconv_kernel']['root_path'],
            kernel_size=config['dataset']['deconv_kernel']['kernel_size'],
            sigma_val=config['dataset']['deconv_kernel']['sigma_val'],
        ),

        'natural': lambda mode, is_pre_load: NaturalImageDataset(
            subset=config['dataset']['natural']['subset'] + '_' + mode,
            root_path=config['dataset']['natural']['root_path'],
            noise_snr=config['dataset']['natural']['noise_snr'],
            kernel_idx=config['dataset']['natural']['kernel_idx'],
            down_sampling_factor=config['dataset']['natural']['down_sampling_factor'],
            cache_id=config['dataset']['natural']['cache_id'],
            is_preload=is_pre_load,
        ),

        'pmri_fastmri': lambda mode, is_pre_load: ParallelMRIFastMRI(
            mode=mode,
            root_path=config['dataset']['pmri_fastmri']['root_path'],
            is_pre_load=is_pre_load,
            smps_method=config['dataset']['pmri_fastmri']['smps_method'],
            acceleration_rate=config['dataset']['pmri_fastmri']['acceleration_rate'],
            acs_percentage=config['dataset']['pmri_fastmri']['acs_percentage'],
            noise_snr=config['dataset']['pmri_fastmri']['noise_snr'],
            smps_hat_method=config['dataset']['pmri_fastmri']['smps_hat_method'],
            low_k_size=config['dataset']['pmri_fastmri']['low_k_size'],
            num_of_coils=config['dataset']['pmri_fastmri']['num_of_coils'],
        )
    }

    if config['setting']['mode'] in ['tra']:
        tra_dataset = dataset_dict[config['setting']['dataset']](mode='tra', is_pre_load=config['dataset']['is_pre_load'])
        val_dataset = dataset_dict[config['setting']['dataset']](mode='val', is_pre_load=config['dataset']['is_pre_load'])

    else:
        tra_dataset = val_dataset = None

    tst_dataset = dataset_dict[config['setting']['dataset']](mode='tst', is_pre_load=False)

    return tra_dataset, val_dataset, tst_dataset
