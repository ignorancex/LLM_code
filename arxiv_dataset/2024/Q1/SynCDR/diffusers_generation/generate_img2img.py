import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..')) # to get myutils

from diffusers import StableDiffusionImg2ImgPipeline
import argparse
from myutils import ioutils
from data.image_folder_w_path import ImageFolderWPath
from pathlib import Path
import copy
import torch
from tqdm import tqdm
from torchvision import transforms
from SynCDR.data_list import Imagelist
import numpy as np
from data.subset_w_idx import SubsetWIdx

TGT2PROMPT = {
    'office_home' : {
        'Product' : 'A photo of an object without a background.',
        'Clipart' : 'A clipart image of an object.',
        'Real' : 'A realistic photo of an object.',
        'Art' : 'A painting of an object.',
    },
    'cub' : {
        'Real' : 'A realistic photo of a bird.',
        'Painting' : 'A painting of a bird.',
    },
    'domainnet' : {
        'clipart' : 'A clipart image of an object.',
        'painting' : 'A painting of an object.',
        'real' : 'A colorful realistic photo of an object.',
        'sketch' : 'A pencil/charcoal sketch of an object.',
    }
}

def gen_images(pipeline, dataset, args):
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, pin_memory=True)
    
    if args.show_progress:
        pbar = tqdm(total=len(loader))
        
    for i, batch in enumerate(loader):
        imgs = batch[0].to(pipeline.device)
        if args.split:
            # dataset is of type Imagelist
            src_paths = [loader.dataset.imgs[idx] for idx in batch[2]]
            src_paths = [str(Path(p).relative_to(Path(p).parents[1])) for p in src_paths]
        else:
            src_paths = batch[2]
        
        out_paths = [str(Path(args.root_dir) / p) for p in src_paths]
        if all([os.path.exists(p) for p in out_paths]):
            continue
        
        generator = torch.Generator(pipeline.device).manual_seed(args.seed)
        out = pipeline(
            image=imgs, prompt=[args.tgt_prompt]*len(imgs),
            strength=args.strength, guidance_scale=args.guidance_scale, 
            num_inference_steps=args.num_inference_steps, generator=generator)
        
        for path, img in zip(src_paths, out.images):
            out_path = Path(args.root_dir) / path
            os.makedirs(out_path.parent, exist_ok=True)
            img.save(out_path)
        
        if args.show_progress:
            pbar.update(1)
    
    if args.show_progress:
        pbar.close()
    
def main(args):
    args = copy.deepcopy(args) # To avoid modifying the original args
    wandb = ioutils.WandbWrapper(debug=True, silent=False, write_to_disk=True)
    
    scenario_ref = f'{args.source[0]}2{args.target[0]}'
    args.root_dir = str(Path(args.root_dir) / args.dataset / scenario_ref)
    if not os.path.exists(args.root_dir):
        os.makedirs(args.root_dir, exist_ok=True)
    wandb.init(dir=args.root_dir, config={**vars(args), 'git_hash': ioutils.get_sha()})
    
    if not args.tgt_prompt:
        args.tgt_prompt = TGT2PROMPT[args.dataset][args.target]
    
    transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])
    src_dataset = Imagelist(Path(args.filelist_root) / f'{args.dataset}/{args.source}_{args.split}.txt', transform=transform)
    if args.num_jobs > 1:
        src_dataset.mode_self = False # So the original dataset does not return indices
        curr_idxs = np.array_split(np.arange(len(src_dataset)), args.num_jobs)[args.job_idx]
        src_dataset_new = SubsetWIdx(src_dataset, curr_idxs)
        src_dataset_new.imgs = [src_dataset.imgs[idx] for idx in curr_idxs]
        src_dataset_new.labels = np.array(src_dataset.labels)[curr_idxs]
        src_dataset = src_dataset_new
    
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', local_files_only=True).to(args.device)
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(disable=True)
    gen_images(pipeline, src_dataset, args)
    
def parse_args(ret='parsed'):
    parser = argparse.ArgumentParser('Generate Stable Diffusion Img2Img')
    parser.add_argument('--dataset', default='domainnet', type=str, help='Dataset name')
    parser.add_argument('--root_dir', default='synthetic_data/img2img', type=str, help='Root directory of generated dataset')
    parser.add_argument('--tgt_prompt', default='', type=str, help='Prompt for target dataset')
    parser.add_argument('--device', default='cuda:0', type=str, help='Device to run on')
    parser.add_argument('--source', default='painting', type=str, help='Source dataset')
    parser.add_argument('--target', default='clipart', type=str, help='Target dataset')
    parser.add_argument('--split', default='train', type=str, help='Split to use')
    parser.add_argument('--filelist_root', default='../SynCDR/data', type=str, help='Filelist root path')
    
    parser.add_argument('--num_workers', default=3, type=int, help='Number of workers for dataloader')
    parser.add_argument('--show_progress', default=False, action='store_true', help='Show progress bar')
    
    parser.add_argument('--num_jobs', default=1, type=int, help='Number of jobs to run')
    parser.add_argument('--job_idx', default=0, type=int, help='Job index')
    
    # generation parameters
    parser.add_argument('--batch_size', default=12, type=int, help='Batch size')
    parser.add_argument('--strength', default=0.3, type=float, help='Diffusion strength')
    parser.add_argument('--guidance_scale', default=10, type=float, help='Guidance scale')
    parser.add_argument('--num_inference_steps', default=50, type=int, help='Number of diffusion steps')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    
    if ret == 'parsed':
        return parser.parse_args()
    elif ret == 'parser':
        return parser
    elif ret == 'default':
        return parser.parse_args([])
    else:
        raise ValueError('Invalid value for ret')

if __name__ == '__main__':
    args = parse_args()
    main(args)