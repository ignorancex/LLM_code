import argparse
import itertools
import os
import random

import torch
from torchvision.transforms import ToPILImage
from tqdm import tqdm

import utils
# from model_bridge import load_adaattn_model, make_adaattn_input
from model_bridge_stya2k import load_stya2k_model, make_stya2k_input
from argparse import Namespace
from PIL import Image
from torchvision.utils import save_image

# import pdb

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--style_reps_dir', type=str, required=True, help='path to source folder')
    parser.add_argument('--output_dir', type=str, required=True, help='path to target folder')
    parser.add_argument('--content_paths', nargs='+', type=str, default=[], help='paths to content file or dirs')
    parser.add_argument('--stya2k_path', type=str, default='./StyA2K', help='path to StyA2K model')
    parser.add_argument('--mask_paths', type=int, default=['/path/to/Input/masks/'], help='Segmentation Mask Path')
    parser.add_argument('--n_ens', type=int, default=None, help='number of ensemble')
    return parser.parse_args()

def build_sty_path_dict(result_dir):
    # print("result_dir in build_sty_path_dict", result_dir)
    img_paths = os.listdir(result_dir)
    # print("image_paths=", image_path)
    img_paths = [name for name in img_paths if f'-style.' in name]
    # print("len(img_paths=)", len(img_paths))   #List type
    sty_path_dict = {}
    for img_path in img_paths:
        style = img_path.split('_')[0]
        if style not in sty_path_dict:
            sty_path_dict[style] = []
        sty_path_dict[style].append(os.path.join(result_dir, img_path))
    # print("len(sty_path_dict)=", len(sty_path_dict))
    # print("sty_path_dict=", sty_path_dict)  

    return sty_path_dict

def main(args):
    # args.adaattn_path = os.path.abspath(args.adaattn_path)
    args.stya2k_path = os.path.abspath(args.stya2k_path)
    contents = {}
    # for content_path in args.content_paths:
    for content_path in args.content_paths:
        assert os.path.exists(content_path), f'Content file {content_path} does not exist.' 
        if os.path.isfile(content_path):
            content_name = os.path.splitext(os.path.basename(content_path))[0]
            contents[content_name] = content_path
        elif os.path.isdir(content_path):
            # collect all files in the content folder, recursively
            for root, dirs, files in os.walk(content_path):
                for name in files:
                    if name.endswith('.png') or name.endswith('.jpg'):
                        content_name = os.path.splitext(name)[0]
                        contents[content_name] = os.path.join(root, name)
    masks = {}
    # for mask_path in args.content_paths:
    for mask_path in args.mask_paths:
        assert os.path.exists(mask_path), f'Content file {mask_path} does not exist.' 
        if os.path.isfile(mask_path):
            mask_name = os.path.splitext(os.path.basename(mask_path))[0]
            masks[mask_name] = mask_path
        elif os.path.isdir(mask_path):
            # collect all files in the content folder, recursively
            for root, dirs, files in os.walk(mask_path):
                for name in files:
                    if name.endswith('.png') or name.endswith('.jpg'):
                        mask_name = os.path.splitext(name)[0]
                        masks[mask_name] = os.path.join(root, name)
    print("contents=", contents)  
    print("masks=", masks)
    # adaattn = load_adaattn_model(args.adaattn_path)
    stya2k = load_stya2k_model(args.stya2k_path)

    # @torch.no_grad()
    # def multi_adaattn_ens(content_path, style_paths):
    #     print("content_path=", content_path, "style_paths=", style_paths)
    #     content = utils.load_image_512(content_path)
    #     style_imgs = []
    #     for sp in style_paths:
    #         s = utils.load_image_512(sp)
    #         sstd = s.std()
    #         if sstd < 0.03:
    #             print(f'Warning: {sp} has std {sstd}, this style image will be ignored.')
    #         else:
    #             style_imgs.append(s)
    #     print("len of style images=", len(style_imgs))
    #     print("type(style_imgs[0])",type(style_imgs[0]))
    #     # for i in range(0, len(style_imgs)):
    #         # print("style_image", i , style_imgs[i].shape)  #[1, 3, 512, 512]
    #         # print("style_image", i , "=", style_imgs[i])
    #     adaattn.set_input(make_adaattn_input(content, style_imgs))
    #     adaattn.forward()
    #     target = adaattn.cs
    #     return target
    @torch.no_grad()
    def multi_stya2k_ens(content_path, style_paths):
    # def multi_stya2k_ens(content_path, mask_path, style_paths):
        # print("In multi_stya2k_ens")
        # print("content_path=", content_path, "style_paths=", style_paths)
        # print("content_path=", content_path)
        # print("mask_path=", mask_path)
        content = utils.load_image_512(content_path)
        # maskf = utils.load_image_512(mask_path)
        style_imgs = []
        for sp in style_paths:
            s = utils.load_image_512(sp)
            sstd = s.std()
            if sstd < 0.03:
                print(f'Warning: {sp} has std {sstd}, this style image will be ignored.')
            else:
                style_imgs.append(s)
        # print("len of style images=", len(style_imgs))
        # print("type(style_imgs[0])",type(style_imgs[0]))
        # for i in range(0, len(style_imgs)):
            # print("style_image", i , style_imgs[i].shape)  #[1, 3, 512, 512]
            # print("style_image", i , "=", style_imgs[i])
        # adaattn.set_input(make_adaattn_input(content, style_imgs))
        # mask=maskf.int()
        # print("mask.shape==", mask.shape)
        # print("mask=", mask)
        # maskedContent= content.masked_fill(mask, 1)
        # stya2k.set_input(make_stya2k_input(content, style_imgs))
        stya2k.set_input(make_stya2k_input(content, style_imgs))
        # adaattn.forward()
        stya2k.forward()
        # target = adaattn.cs
        target = stya2k.cs
        return target

    sd = build_sty_path_dict(args.style_reps_dir)
    # print("type(sd=)", type(sd))
    # print(sd)
    
    os.makedirs(args.output_dir, exist_ok=True)
    maskedImages={}
    maskedContent_Dir="/home/phdcs2/Hard_Disk/Projects/T2I/mmist/input/contents_DSM/srg/crf/maskedContent1/"
    for c, m in zip(contents, masks):
        content_path = contents[c] 
        mask_path=masks[m]
        mask_filename = os.path.basename(mask_path)
        print("content_path=", content_path)
        print("mask_path=", mask_path)
        # print("len(contents)=", len(contents))
        # print("len(masks)=", len(masks))
        # print("c=", c)
        # print("m=", m)
        content = utils.load_image_512(content_path)
        maskf = utils.load_image_512(mask_path)
        # mask=maskf.int()
        # print("mask.shape==", mask.shape)
        # print("mask=", mask)
        # maskedContent= content.masked_fill(mask, 0)
        maskedContent=content*maskf
        save_image(maskedContent, maskedContent_Dir + mask_filename , normalize=True, format="PNG")
        # maskedContent.save(maskedContent_Dir + mask_path,format='PNG')
        maskedImages[mask_filename] =maskedContent_Dir + mask_filename
    print("maskedIimages=", maskedImages)
    # print("type(contents)=", type(contents))
    # for s, c in tqdm(itertools.product(sd, contents), total=len(sd)*len(contents)):
    for s, c in tqdm(itertools.product(sd, maskedImages), total=len(sd)*len(maskedImages)):
    # for s, c, m in tqdm(itertools.product(sd, contents, masks), total=len(sd)*len(contents)):
        # content_path = contents[c] 
        content_path = maskedImages[c] 

       
        # print("len(sd)=", len(sd))
        # print("s=", s)

        # print("type(s)=", type(s), "type(c)=", type(c), "type(content_path=)", type(content_path))
        # print("s=", s, "c=", c, "content_path=", content_path)
        if args.n_ens is None:
            print("args.n_ens=",args.n_ens)
            # result = multi_adaattn_ens(content_path, sd[s])
            # print("content_path=", content_path)
            # print("sd[s]=", sd[s]
            result = multi_stya2k_ens(content_path, sd[s])
            # result = multi_stya2k_ens(content_path, sd[s])
            # result = multi_stya2k_ens(content_path, mask_path, sd[s])
        else:
            # print("args.n_ens=",args.n_ens)
            # result = multi_adaattn_ens(content_path, random.sample(sd[s], args.n_ens))
            result = multi_stya2k_ens(content_path, random.sample(sd[s], args.n_ens))          
           
          
        # result = multi_stya2k_ens(content_path, mask_path, random.sample(sd[s], args.n_ens))
        s=s.replace(" ","")
        result_path = os.path.join(args.output_dir, f'{s}_{c}')
        # print("result_path=", result_path)
        result_pil = ToPILImage()(result.squeeze().clamp(0,1))
        result_pil.save(result_path)        
if __name__ == '__main__':
    main(get_args())

    