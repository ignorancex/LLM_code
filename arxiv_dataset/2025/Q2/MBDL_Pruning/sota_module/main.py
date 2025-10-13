import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import ray

os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba'
os.environ['CUPY_CACHE_DIR'] = '/tmp/cupy'
os.environ['TUNE_DISABLE_STRICT_METRIC_CHECKING'] = '1'

import pytorch_lightning as pl
pl.seed_everything(1016)

from method.dep_cal import run as deq_cal_run
from method.deq_cal_alter_pmri import run as deq_cal_alter_pmri_run
from method.bccal_iteration_pmri import run as bccal_iteration_run
from method.deq_cal_alter_pmri_inference import run as deq_cal_alter_pmri_inference_run
from method.deq_cal_alter_natural_inference import run as deq_cal_alter_natural_inference_run

from method.baseline.tv import run as tv_run
from method.baseline.e2evarnet import run as e2evarnet_run
from method.baseline.standard_unet import run as unet_run
from method.baseline.istanetplus import run as istanetplus_run
from method.baseline.MoDL_pmri import run as modl_pmri_run
from debug.gradient_smps import run as debug_gradient_smps_run

from method.warmup_x_pmri import run as warmup_x_pmri_run
from method.warmup_theta_pmri import run as warmup_theta_pmri_run
from method.warmup_theta_deconv import run as warmup_theta_deconv_run
from manuscript.run_denosing_test import run as manuscript_denoising_run
from manuscript.run_denoising_natrual_test import run as manuscript_denoising_run_natural
from manuscript.update_kernel_natural import run as update_kernel_natural_run
from manuscript.run_denoising_pmri_fastmri_test import run as run_denoising_pmri_fastmri
from manuscript.find_maximum_value_dataset import run as find_maximum_value_dataset_run

import yaml
import sys
from utility import ray_tune_config_to_param_space, ray_tune_override_config_from_param_space, copy_code_to_path
from get_from_config import get_save_path_from_config
import copy
from ray import tune, air
from ray.tune import CLIReporter
import torch


model_dict = {
    'deq_cal': deq_cal_run,
    'deq_cal_alter_pmri': deq_cal_alter_pmri_run,
    'deq_cal_alter_pmri_inference': deq_cal_alter_pmri_inference_run,
    'deq_cal_alter_natural_inference': deq_cal_alter_natural_inference_run,

    'bccal_iteration': bccal_iteration_run,
    'warmup_x_pmri': warmup_x_pmri_run,
    'warmup_theta_pmri': warmup_theta_pmri_run,

    'tv': tv_run,
    'e2evarnet': e2evarnet_run,
    'unet': unet_run,
    'modl_pmri': modl_pmri_run,
    'istanetplus': istanetplus_run,

    'debug_gradient_smps': debug_gradient_smps_run,
    'manuscript_denoising': manuscript_denoising_run,
    'manuscript_denoising_run_natural': manuscript_denoising_run_natural,
    'update_kernel_natural': update_kernel_natural_run,
    'run_denoising_pmri_fastmri': run_denoising_pmri_fastmri,
    'find_maximum_value_dataset': find_maximum_value_dataset_run,

    'warmup_theta_deconv': warmup_theta_deconv_run,
}


def ray_tune_run(param_space_local, config_local):

    config_new = ray_tune_override_config_from_param_space(
        copy.deepcopy(config_local), copy.deepcopy(param_space_local))
    return model_dict[config_new['setting']['method']](config_new)


if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        config = yaml.safe_load(f)

    param_space = ray_tune_config_to_param_space(config)

    if param_space == dict():

        copy_code_to_path(src_path=config['setting']['project_path'], file_path=get_save_path_from_config(config))

        model_dict[config['setting']['method']](config)

    else:

        ray.init(num_cpus=torch.cuda.device_count() * 2)

        tune_param_ray_with_parameters = tune.with_parameters(ray_tune_run, config_local=config)

        reporter = CLIReporter(
            parameter_columns=list(param_space.keys()),
            metric_columns=["tst_psnr", "tst_ssim"])

        tuner = tune.Tuner(
            tune.with_resources(
                tune_param_ray_with_parameters,
                resources={"cpu": 2, "gpu": 1}
            ),
            param_space=param_space,
            run_config=air.RunConfig(
                name="ray_tune_run",
                local_dir=get_save_path_from_config(config),
                verbose=1,
                progress_reporter=reporter
            ),
            tune_config=tune.TuneConfig(
                metric="tst_psnr",
                mode="max", )
        )

        copy_code_to_path(src_path=config['setting']['project_path'], file_path=get_save_path_from_config(config))

        results = tuner.fit()
        print("Best hyperparameters found were: ", results.get_best_result().config)
        print(results.get_best_result())
