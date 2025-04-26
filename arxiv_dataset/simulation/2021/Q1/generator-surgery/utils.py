import io
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import truncnorm
from torchvision import transforms


def print_torchvec(x):
    return ','.join([f'{i:0.3f}' for i in x.tolist()])


def dict_to_str(d):
    s = []
    for k, v in d.items():
        s.append(f"{k}={v}")
    return ".".join(s)


def str_to_dict(s):
    blocks = s.split('.')
    clean_blocks = []
    for b in blocks:
        if '=' in b:
            clean_blocks.append(b)
        else:
            clean_blocks[-1] += ('.' + b)

    d = {}
    for b in clean_blocks:
        k, v = b.split('=')
        d[k] = v

    return d


def weight_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)


def _gen_img(img):
    plt.figure(figsize=(16, 9))
    plt.imshow(img)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf


def normalize(img):
    return (img - img.min()) / (img.max() - img.min())


def load_trained_net(net, ckpt_filename):
    ckpt = torch.load(ckpt_filename, map_location='cpu')['model_state_dict']
    fixed_ckpt = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            fixed_ckpt[k[7:]] = v
        else:
            fixed_ckpt[k] = v
    net.load_state_dict(fixed_ckpt)
    return net


def load_target_image(img, target_size):
    if img.endswith('.pt'):
        x = torch.load(img)
    else:
        image = Image.open(img)
        height, width = image.size

        if height > width:
            crop = transforms.CenterCrop((width, width))
        else:
            crop = transforms.CenterCrop((height, height))

        t = transforms.Compose([
            crop,
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor()
        ])
        x = t(image)
    return x


def psnr(img1, img2):
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1)

    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2)

    mse = F.mse_loss(img1, img2)
    return psnr_from_mse(mse)


def psnr_from_mse(mse):
    if mse == 0:
        return -1
    pixel_max = torch.tensor(1.0)
    p = 20 * torch.log10(pixel_max) - 10 * torch.log10(mse)
    if isinstance(p, torch.Tensor):
        p = p.item()
    return p


def get_z_vector(shape, mode, limit=1, **kwargs):
    if mode == 'normal':
        z = torch.randn(*shape, **kwargs) * limit
    elif mode == 'clamped_normal':
        # Clamp between -truncation, truncation
        z = torch.clamp(torch.randn(*shape, **kwargs), -limit, limit)
    elif mode == 'truncated_normal':
        # Resample if any point lands outside -limit, limit
        values = truncnorm.rvs(-2, 2, size=shape).astype(np.float32)
        z = limit * torch.from_numpy(values).to(kwargs['device'])
        # raise NotImplementedError()
    elif mode == 'rectified_normal':
        # Max(N(0,1), 0)
        raise NotImplementedError()
    elif mode == 'uniform':
        z = 2 * limit * torch.rand(shape, **kwargs) - limit
    elif mode == 'zero':
        z = torch.zeros(*shape, **kwargs)
    else:
        raise NotImplementedError()
    return z


def get_images_folder(split, image_name, img_size, base_dir):
    return Path(base_dir) / 'images' / split / image_name / str(img_size)


def parse_images_folder(p):
    p = Path(p)
    _, _, split, image_name, img_size = p.parts
    return split, image_name, img_size


# Use get_results_folder for all models, use dummy n_cuts if necessary
def get_results_folder(image_name, model, n_cuts, split, forward_model,
                       recovery_params, base_dir):
    return (Path(base_dir) / 'results' / model / f'n_cuts={n_cuts}' / split /
            image_name / str(forward_model) / recovery_params)


def parse_results_folder(root='./final_runs/results'):
    rows_list = []
    p = Path(root)

    def get_img_size(model_name):
        d = {
            'dcgan': '64',
            'began': '128',
            'vanilla_vae': '128',
            'beta_vae': '128',
            'biggan': '512',
            'iagan_dcgan': '64',
            'iagan_began': '128',
            'iagan_vanilla_vae': '128',
            'mgan_dcgan': '64',
            'mgan_began': '128',
            'mgan_vanilla_vae': '128',
            'deep_decoder_64': '64',
            'deep_decoder_128': '128',
        }
        for k, v in d.items():
            if model_name.startswith(k):
                return v
        raise KeyError()

    for model_path in p.iterdir():
        model = model_path.name

        image_size = get_img_size(model)
        for n_cuts_path in model_path.iterdir():
            n_cuts = int(n_cuts_path.name.split('=')[1])
            for split_path in n_cuts_path.iterdir():
                split = split_path.name
                for image_name_path in split_path.iterdir():
                    image_name = image_name_path.name
                    for fm_path in image_name_path.iterdir():
                        fm_name, fm_params = forward_model_from_str(
                            fm_path.name)
                        for params_path in fm_path.iterdir():
                            if not os.path.exists(
                                    params_path / 'metadata.pkl'):
                                print('metadata.pkl not found for',
                                      params_path)
                                continue
                            with open(params_path / 'metadata.pkl', 'rb') as f:
                                metadata = pickle.load(f)
                            if os.path.exists(params_path /
                                              'psnr_clamped.pkl'):
                                with open(params_path / 'psnr_clamped.pkl',
                                          'rb') as f:
                                    ps = pickle.load(f)
                            else:
                                recovered = torch.load(str(params_path /
                                                           'recovered.pt'),
                                                       map_location='cpu')
                                orig_path = p.parent.joinpath(
                                    'images', split, image_name, image_size,
                                    'original.pt')
                                orig = torch.load(str(orig_path),
                                                  map_location='cpu')

                                if recovered.dim() == 4:
                                    recovered = recovered.squeeze()
                                    torch.save(recovered,
                                               params_path / 'recovered.pt')

                                ps = psnr(orig, recovered.clamp(0, 1))

                                with open(params_path / 'psnr_clamped.pkl',
                                          'wb') as f:
                                    pickle.dump(float(ps), f)

                            current_row_dict = {}

                            current_row_dict['model'] = model
                            current_row_dict['n_cuts'] = n_cuts
                            current_row_dict['split'] = split
                            current_row_dict['image_name'] = image_name
                            current_row_dict['fm'] = fm_name
                            current_row_dict['fraction_kept'] = float(
                                fm_params.get('fraction_kept', -1.))
                            current_row_dict['scale_factor'] = float(
                                fm_params.get('scale_factor', -1.))
                            current_row_dict['n_measure'] = int(
                                fm_params.get('n_measure', -1))
                            current_row_dict['lasso_coeff'] = float(
                                fm_params.get('lasso_coeff', -1.))
                            current_row_dict.update(metadata)
                            current_row_dict['psnr'] = float(ps)

                            rows_list.append(current_row_dict)

    df = pd.DataFrame(rows_list)
    os.makedirs(p.parent / 'processed_results', exist_ok=True)
    with open(p.parent / 'processed_results/df_results.pkl', 'wb') as f:
        pickle.dump(df, f)


def forward_model_from_str(s):
    lst = s.split('.', 1)
    if len(lst) > 1:
        name = lst[0]
        params = str_to_dict(lst[1])
        return name, params
    else:
        name = lst[0]
        return name, {}


def get_baseline_results_folder(image_name, model, split, n_measure,
                                lasso_coeff, base_dir):
    return (Path(base_dir) / 'baseline_results' / model / split / image_name /
            f'n_measure={n_measure}.lasso_coeff={lasso_coeff}')


def parse_baseline_results_folder(root='./runs/baseline_results'):
    rows_list = []
    p = Path(root)
    for model_path in p.iterdir():
        model = model_path.name
        if model.endswith('64'):
            image_size = '64'
        elif model.endswith('128'):
            image_size = '128'
        else:
            raise KeyError()
        for split_path in model_path.iterdir():
            split = split_path.name
            for image_name_path in split_path.iterdir():
                image_name = image_name_path.name
                for params_path in image_name_path.iterdir():
                    params = str_to_dict(params_path.name)
                    if not os.path.exists(params_path / 'metadata.pkl'):
                        print('metadata.pkl not found for', params_path)
                        continue
                    # with open(params_path / 'metadata.pkl', 'rb') as f:
                    #     metadata = pickle.load(f)

                    if os.path.exists(params_path / 'psnr_clamped.pkl'):
                        with open(params_path / 'psnr_clamped.pkl', 'rb') as f:
                            ps = pickle.load(f)
                    else:
                        recovered = np.load(params_path / 'recovered.npy')
                        orig_path = p.parent.joinpath('images', split,
                                                      image_name, image_size,
                                                      'original.npy')
                        orig = np.load(orig_path)

                        ps = psnr(torch.from_numpy(orig),
                                  torch.from_numpy(recovered).clamp(0, 1))

                        with open(params_path / 'psnr_clamped.pkl', 'wb') as f:
                            pickle.dump(float(ps), f)

                    current_row_dict = {}

                    current_row_dict['model'] = model
                    current_row_dict['split'] = split
                    current_row_dict['image_name'] = image_name
                    current_row_dict['n_measure'] = int(params['n_measure'])
                    current_row_dict['lasso_coeff'] = float(
                        params['lasso_coeff'])
                    # current_row_dict.update(metadata)
                    current_row_dict['psnr'] = float(ps)

                    rows_list.append(current_row_dict)

    df = pd.DataFrame(rows_list)
    os.makedirs(p.parent / 'processed_results', exist_ok=True)
    with open(p.parent / 'processed_results/df_baseline_results.pkl',
              'wb') as f:
        pickle.dump(df, f)


if __name__ == '__main__':
    parse_results_folder('./final_runs/results')
    parse_baseline_results_folder('./final_runs/baseline_results')
