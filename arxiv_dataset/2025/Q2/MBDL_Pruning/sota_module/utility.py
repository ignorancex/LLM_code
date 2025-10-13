import os
import shutil
import torch
import numpy as np
from tifffile import imwrite
from collections import defaultdict
import pathlib
from ray import tune


def torch_complex_normalize(x):
    x_angle = torch.angle(x)
    x_abs = torch.abs(x)

    x_abs -= torch.min(x_abs)
    x_abs /= torch.max(x_abs)

    x = x_abs * np.exp(1j * x_angle)

    return x


def strip_empties_from_dict(data):
    new_data = {}
    for k, v in data.items():
        if isinstance(v, dict):
            v = strip_empties_from_dict(v)

        if v not in (None, str(), list(), dict(),):
            new_data[k] = v
    return new_data


def ray_tune_config_to_param_space(config, param_space=None):
    if param_space is None:
        param_space = {}

    for k in config:
        if isinstance(config[k], dict):
            param_space.update({k: {}})
            ray_tune_config_to_param_space(config[k], param_space[k])
        else:
            if isinstance(config[k], list) and len(config[k]) > 1:
                param_space.update({
                    k: tune.grid_search(config[k])
                })

    return strip_empties_from_dict(param_space)


def ray_tune_override_config_from_param_space(config, param_space):
    for k in param_space:
        if isinstance(param_space[k], dict):
            ray_tune_override_config_from_param_space(config[k], param_space[k])

        else:
            config[k] = param_space[k]

    return config


def get_last_folder(path):
    return pathlib.PurePath(path).name


def convert_pl_outputs(outputs):
    outputs_dict = defaultdict(list)

    for i in range(len(outputs)):
        for k in outputs[i]:
            outputs_dict[k].append(outputs[i][k])

    log_dict, img_dict = {}, {}
    for k in outputs_dict:
        try:
            tmp = torch.Tensor(outputs_dict[k]).detach().cpu()

            log_dict.update({
                k: tmp
            })

        except Exception:
            if outputs_dict[k][0].dim() == 2:
                tmp = torch.stack(outputs_dict[k], 0).detach().cpu()
            else:
                tmp = torch.cat(outputs_dict[k], 0).detach().cpu()

            if tmp.dtype == torch.complex64:
                tmp = torch.abs(tmp)

            img_dict.update({
                k: tmp
            })

    return log_dict, img_dict


def check_and_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def copy_code_to_path(src_path=None, file_path=None):
    if (file_path is not None) and (src_path is not None):
        check_and_mkdir(file_path)

        max_code_save = 100  # only 100 copies can be saved
        for i in range(max_code_save):
            code_path = os.path.join(file_path, 'code%d/' % i)
            if not os.path.exists(code_path):
                shutil.copytree(src=src_path, dst=code_path)
                break


def merge_child_dict(d, ret, prefix=''):

    for k in d:
        if k in ['setting', 'test']:
            continue

        if isinstance(d[k], dict):
            merge_child_dict(d[k], ret=ret, prefix= prefix + k + '/')
        else:
            ret.update({
                prefix + k: d[k]
            })

    return ret


def write_test(save_path, log_dict=None, img_dict=None):

    if log_dict:

        cvs_data = torch.stack([log_dict[k] for k in log_dict], 0).numpy()
        cvs_data = np.transpose(cvs_data, [1, 0])

        cvs_data_mean = cvs_data.mean(0)
        cvs_data_mean.shape = [1, -1]

        cvs_data_std = cvs_data.std(0)
        cvs_data_std.shape = [1, -1]

        cvs_data_min = cvs_data.min(0)
        cvs_data_min.shape = [1, -1]

        cvs_data_max = cvs_data.max(0)
        cvs_data_max.shape = [1, -1]

        num_index = cvs_data.shape[0]
        cvs_index = np.arange(num_index) + 1
        cvs_index.shape = [-1, 1]

        cvs_data_with_index = np.concatenate([cvs_index, cvs_data], 1)

        cvs_header = ''
        for k in log_dict:
            cvs_header = cvs_header + k + ','

        np.savetxt(os.path.join(save_path, 'metrics.csv'), cvs_data_with_index, delimiter=',', fmt='%.5f', header='index,' + cvs_header)
        np.savetxt(os.path.join(save_path, 'metrics_mean.csv'), cvs_data_mean,  delimiter=',', fmt='%.5f', header=cvs_header)
        np.savetxt(os.path.join(save_path, 'metrics_std.csv'), cvs_data_std, delimiter=',', fmt='%.5f',  header=cvs_header)
        np.savetxt(os.path.join(save_path, 'metrics_min.csv'), cvs_data_min, delimiter=',', fmt='%.5f', header=cvs_header)
        np.savetxt(os.path.join(save_path, 'metrics_max.csv'), cvs_data_max, delimiter=',', fmt='%.5f', header=cvs_header)

        print("==========================")
        print("HEADER:", cvs_header)
        print("MEAN:", cvs_data_mean)
        print("STD:", cvs_data_std)
        print("MAX:", cvs_data_max)
        print("MIN:", cvs_data_min)
        print("==========================")

    if img_dict:

        for k in img_dict:

            imwrite(file=os.path.join(save_path, k + '.tiff'), data=np.array(img_dict[k]), imagej=True)
