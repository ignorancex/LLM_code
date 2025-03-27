
import os

import math
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
import einops
import pytorch_lightning as pl
from PIL import Image
from omegaconf import OmegaConf

from ldm.xformers_state import disable_xformers
from model.cldm import ControlLDM
from utils.common import instantiate_from_config, load_state_dict



import torch.nn.functional as F
from pathlib import Path
from torchvision import transforms
import json
from collections import defaultdict



from cal_metrics.iqa import single_iqa, get_fid_from_path
single_iqa = single_iqa()





def parse_args() -> Namespace:
    parser = ArgumentParser()
    
    ckpt = 'magc_ckpts/ckpts_stage2/v40_step=120999-lpips=0.3132.ckpt'
    ckpt_name = ckpt.split('/')[-1]
    config = 'configs/model/cldm_stage2.yaml'
    input_path = '../dataset/Synthetic-v18-45k/test_4500'
    step = 50
    batchsize = 30
    output_path = f'metrics_4500_magc/{ckpt_name}'

    os.makedirs(output_path,exist_ok=True)


    if not os.path.isdir(output_path):
        os.makedirs(output_path)


    # TODO: add help info for these options
    parser.add_argument("--ckpt", type=str,default=ckpt, help="full checkpoint path")
    parser.add_argument("--config", type=str,default=config, help="model config path")

    
    parser.add_argument("--input_path", type=str, default=input_path)
    parser.add_argument("--steps", default = step, type=int)
    parser.add_argument("--batchsize", default = batchsize, type=int)
    parser.add_argument("--sr_scale", type=float, default=1) # 迭代次数
    parser.add_argument("--repeat_times", type=int, default=1) # 重复次数
    parser.add_argument("--disable_preprocess_model", action="store_true")
    
    # patch-based sampling
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--tile_stride", type=int, default=256)
    
    # latent image guidance
    parser.add_argument("--use_guidance", action="store_true")
    parser.add_argument("--g_scale", type=float, default=0.0)
    parser.add_argument("--g_t_start", type=int, default=1001)
    parser.add_argument("--g_t_stop", type=int, default=-1)
    parser.add_argument("--g_space", type=str, default="latent")
    parser.add_argument("--g_repeat", type=int, default=5)
    
    parser.add_argument("--color_fix_type", type=str, default="wavelet", choices=["wavelet", "adain", "none"])
    parser.add_argument("--output_path", type=str, default=output_path)
    parser.add_argument("--show_lq", action="store_true")
    parser.add_argument("--skip_if_exist", action="store_true")
    
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    
    return parser.parse_args()

def check_device(device):
    if device == "cuda":
        # check if CUDA is available
        if not torch.cuda.is_available():
            print("CUDA not available because the current PyTorch install was not "
                "built with CUDA enabled.")
            device = "cpu"
    else:
        # xformers only support CUDA. Disable xformers when using cpu or mps.
        disable_xformers()
        if device == "mps":
            # check if MPS is available
            if not torch.backends.mps.is_available():
                if not torch.backends.mps.is_built():
                    print("MPS not available because the current PyTorch install was not "
                        "built with MPS enabled.")
                    device = "cpu"
                else:
                    print("MPS not available because the current MacOS version is not 12.3+ "
                        "and/or you do not have an MPS-enabled device on this machine.")
                    device = "cpu"
    print(f'using device {device}')
    return device


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse)

def cal_metrics_and_save(samples, model, filenames, save_path, refs):
    # save_path: inference_output/vertion0

    # samples['img_gt'] [0,1], samples['samples'] [0,1], samples['bpp']
    img_gt = samples['img_gt']
    img_rec = samples['samples']

    img_num = len(filenames)
    metrics_batch_total = defaultdict(float)

    # 计算并保存单张图片的指标
    for i in range(img_num):
        metrics_single = {}
        img_gt_single = img_gt[i,:,:,:].unsqueeze(0)
        img_rec_single = img_rec[i,:,:,:].clamp_(0, 1).unsqueeze(0)
        metrics = single_iqa.cal_metrics(img_gt_single, img_rec_single)
        metrics_single['bpp'] = samples['bpp_list'][i]

        for k in metrics:
            metrics_single[k] = metrics[k]




        for k, v in metrics_single.items():
            metrics_batch_total[k] += v

        _filename = filenames[i].split('.')[0]

        save_path_img = os.path.join(save_path, f'{_filename}.png')
        save_path_metrics = os.path.join(save_path, f'{_filename}.json')


        img_gt_single = transforms.ToPILImage()(img_gt_single.squeeze(0).cpu()) # 用这一步之前一定一定要clamp
        img_rec_single = transforms.ToPILImage()(img_rec_single.squeeze(0).cpu())
        img_ref_single = refs[i]
        result_img = Image.new('RGB', (img_gt_single.width + img_rec_single.width + img_ref_single.width + 20 , img_gt_single.height))
        result_img.paste(img_gt_single, (0, 0))
        result_img.paste(img_rec_single, (img_gt_single.width + 10, 0))
        result_img.paste(img_ref_single, (img_gt_single.width + img_rec_single.width + 20, 0))
        result_img.save(save_path_img)

        with Path(save_path_metrics).open("wb") as f:
            output = {
                "results": metrics_single,
            }
            f.write(json.dumps(output, indent=2).encode())


    return metrics_batch_total




@torch.no_grad()
def inference_batch(
    model: ControlLDM,
    batch: dict, 
    recon_path: str,
    steps: int,
    ):

    img_tensor = torch.tensor(np.stack(batch['imgs']) / 255.0, dtype=torch.float32, device=model.device).clamp_(0, 1)
    ref_tensor = torch.tensor(np.stack(batch['refs']) / 255.0, dtype=torch.float32, device=model.device).clamp_(0, 1)
    img_tensor = einops.rearrange(img_tensor, "n h w c -> n c h w").contiguous()
    ref_tensor = einops.rearrange(ref_tensor, "n h w c -> n c h w").contiguous()

    txt = [''] * 10
    batch_input = {}
    batch_input['img_gt'] = img_tensor * 2 - 1.0
    batch_input['ref_gt'] = ref_tensor
    batch_input['txt'] = txt
    

    # 编码过程，将img转换成latent, 再将latent转化成hyper，计算码流
    # 解码过程，从x_T到x_0
    with torch.cuda.amp.autocast():
        samples = model.log_images(batch = batch_input, sample_steps = steps) # samples['img_gt'] [0,1], samples['samples'] [0,1], samples['bpp']
        metrics_batch_total = cal_metrics_and_save(samples, model, batch['filenames'], recon_path, batch['refs'])

    return metrics_batch_total


def get_batch_list(input_dir, batchsize):
    assert os.path.isdir(input_dir)

    # path_ref = os.path.join(input_dir, 'ref_256_repaint_8x_resample')
    path_ref = os.path.join(input_dir, 'ref_256')
    path_img = os.path.join(input_dir, 'hr_256')
    
    list_ref = [os.path.join(path_ref, item) for item in os.listdir(path_ref) if item.endswith('.png')]
    list_img = [os.path.join(path_img, item) for item in os.listdir(path_img) if item.endswith('.png')]
    assert len(list_ref) == len(list_img)
    img_num = len(list_ref)
    list_ref.sort()
    list_img.sort()

    batch_list = []
    batch_dict = {}
    batch_dict['imgs'] = []
    batch_dict['refs'] = []
    batch_dict['filenames'] = []
    for idx in range(img_num):
        file_name =  os.path.relpath(list_img[idx], path_img) # ###.png
        img = Image.open(list_img[idx]).convert("RGB")
        ref = Image.open(list_ref[idx]).convert("RGB")
        batch_dict['imgs'].append(img)
        batch_dict['refs'].append(ref)
        batch_dict['filenames'].append(file_name)
        if (idx+1) % batchsize == 0 or idx==(img_num-1):
            batch_list.append(batch_dict.copy())
            batch_dict['imgs'] = []
            batch_dict['refs'] = []
            batch_dict['filenames'] = []
    return batch_list





def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed)
    args.device = check_device(args.device)
    
    # 加载模型权重
    print('model init...')
    model: ControlLDM = instantiate_from_config(OmegaConf.load(args.config))
    print('model init over.')

    print('loading state dict...')
    load_state_dict(model, torch.load(args.ckpt, map_location="cpu"), strict=True)
    print('loading over.')

    model.hyper_encoder.update(force=True)
    # reload preprocess model if specified
    model.freeze()
    model.to(args.device)
    
    # 获取数据集
    batch_list = get_batch_list(args.input_path, args.batchsize)

    results = defaultdict(float)
    for i, batch in enumerate(batch_list): 

        print('processing batch {}:'.format(i+1))
        metrics_batch_total = inference_batch(model, batch,args.output_path,
            steps=args.steps,
        )

        for k, v in metrics_batch_total.items():
            results[k] += v


    img_paths = os.path.join(args.input_path, 'hr_256')
    img_num = len(os.listdir(img_paths))
    for k, v in results.items():
        results[k] = v / img_num


    # 计算fid
    results['fid'],results['kid'] = get_fid_from_path(args.output_path)
    
    for i in results: # 保留4位小数
        results[i] = "{:.4f}".format(results[i])

    output = {
        "results": results,
    }

    with (Path(f"{args.output_path}/average").with_suffix('.json')).open(
        "wb"
    ) as f:
        f.write(json.dumps(output, indent=2).encode())


if __name__ == "__main__":
    main()
