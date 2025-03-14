import os
import argparse
from typing import List
from functools import partial
from contextlib import suppress

import cv2
import wandb
import matplotlib.pyplot as plt
from einops import reduce, rearrange
from timm.models import create_model
import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid

import models
from datasets import build_dataset
from train import get_args_parser, adjust_config, set_seed, set_run_name
from models.attention_rollout import attention_rollout
from utils import inverse_normalize



class VisHook:
    def __init__(self,
                 model: nn.Module,
                 model_layers: List[str] = None,
                 device: str ='cpu'):
        """
        :param model: (nn.Module) Neural Network 1
        :param model_layers: (List) List of layers to extract features from
        :param device: Device to run the model
        """

        self.model = model

        self.device = device

        self.model_info = {}

        self.model_info['Layers'] = []

        self.model_features = {}

        self.model_layers = model_layers

        self._insert_hooks()
        self.model = self.model.to(self.device)

        self.model.eval()

        print(self.model_info)


    def _log_layer(self,
                   model: str,
                   name: str,
                   layer: nn.Module,
                   inp: torch.Tensor,
                   out: torch.Tensor):

        if model == "model":
            self.model_features[name] = out
        else:
            raise RuntimeError("Unknown model name for _log_layer.")


    def _insert_hooks(self):
        # Model 1
        for name, layer in self.model.named_modules():
            if self.model_layers is not None:
                if name in self.model_layers:
                    self.model_info['Layers'] += [name]
                    layer.register_forward_hook(partial(self._log_layer, "model", name))
            else:
                self.model_info['Layers'] += [name]
                layer.register_forward_hook(partial(self._log_layer, "model", name))


    def extract_features(self, images) -> None:
        """
        Computes the attention rollout for the image(s)
        :param x: (input tensor)
        """
        self.model_features = {}

        images = images.to(self.device)
        _ = self.model(images)

        return self.model_features


    def save_vis(self, args, loader, split):
        images, _ = next(iter(loader))

        features = self.extract_features(images)
        if 'reg4' in args.model:
            regs = 4
            regs_end = False
        elif args.num_clr:
            regs = args.num_clr
            regs_end = True
        else:
            regs = 0
            regs_end = False
        masks = calc_masks(features, args.vis_mask, regs=regs, regs_end=regs_end)

        masked_imgs = []
        for i in range(images.shape[0]):
            img_unnorm = inverse_normalize(images[i].detach().clone())
            img_masked = apply_mask(img_unnorm, masks[i], args.vis_mask_pow)
            masked_imgs.append(img_masked)

        number_imgs = len(masked_imgs)
        ncols = args.vis_cols if args.vis_cols else int(number_imgs ** 0.5)
        nrows = number_imgs // ncols
        number_imgs = ncols * nrows

        fig = plt.figure(figsize=(ncols, nrows))
        grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols),
                        axes_pad=(0.01, 0.01), direction='row', aspect=True)

        for i, (ax, np_arr) in enumerate(zip(grid, masked_imgs)):
            ax.axis('off')
            ax.imshow(np_arr)

        plt.tight_layout()
        save_images(fig, args.vis_mask, args.vis_mask_pow, split, args.output_dir, args.debugging)
            
        return 0


def save_images(fig, vis_mask, power, split, output_dir, debugging=True):
    pow = '_power' if power else ''
    fn = f'{vis_mask}{pow}_{split}.png'
    fp = os.path.join(output_dir, fn)
    fig.savefig(fp, dpi=300, bbox_inches='tight', pad_inches=0.01)
    print('Saved ', fp)

    if not debugging:
        wandb.log({f'{split}': wandb.Image(fig)})

    return 0


def calc_masks(features, vis_mask='rollout', def_rollout_end=4, regs=0, regs_end=False):
    if vis_mask is None:
        bs = list(features.values())[0].shape[0]
        return [None for _ in range(bs)]
    elif 'rollout' in vis_mask or 'attention' in vis_mask:
        features = {k: v for k, v in features.items() if 'attn' in k}
        print(features.keys())
        features = list(features.values())
    elif vis_mask == 'gls':
        features = {k: v for k, v in features.items() if 'norm' in k}
        print(features.keys())
        features = list(features.values())[-1]
    else:
        raise NotImplementedError

    if 'rollout' in vis_mask:
        splits = vis_mask.split('_')
        if len(splits) == 1:
            rollout_start = 0
            rollout_end = int(def_rollout_end)
        elif len(splits) == 2:
            rollout_start = 0
            rollout_end = int(splits[-1])
        elif len(splits) == 3:
            rollout_start = int(splits[1])
            rollout_end = int(splits[-1])
        else:
            raise NotImplementedError

        # input: list of length L with each element shape: B, NH, S, S
        # use only first 4 (should be similar enough to full rollout)
        # because scores after 4 should have different shape
        # output: B, S, S
        # select 1st token attention for the rest []:, 0, 1:] -> B, S-1

        attention = features[rollout_start:rollout_end]
        attention = attention_rollout(attention)
        if regs > 0 and regs_end:
            masks = attention[:, 0, 1:-regs]
        elif regs > 0:
            masks = attention[:, 0, 1+regs:]
        else:
            masks = attention[:, 0, 1:]

    elif 'attention' in vis_mask:
        splits = vis_mask.split('_')
        if len(splits) == 1:
            layer = 0
        elif len(splits) == 2:
            layer = int(splits[-1])
        else:
            raise NotImplementedError

        attention = features[layer]
        attention = reduce(attention, 'b h s1 s2 -> b s1 s2', 'mean')
        if regs > 0 and regs_end:
            masks = attention[:, 0, 1:-regs]
        elif regs > 0:
            masks = attention[:, 0, 1+regs:]
        else:
            masks = attention[:, 0, 1:]

    elif vis_mask == 'gls':
        fh = int(features.shape[1] ** 0.5)
        if fh ** 2 == features.shape[1]:
            g = reduce(features, 'b s d -> b d', 'mean')
            l = features
        else:
            g = features[:, :1]
            l = features[:, 1:]

        masks = F.cosine_similarity(g, l, dim=-1)

    else:
        raise NotImplementedError

    return masks


def apply_mask(img, mask, power=False, color=True):
    '''
    img are pytorch tensors of size C x S1 x S2, range 0-255 (unnormalized)
    mask are pytorch tensors of size (S1 / P * S2 / P) of floating values (range prob -1 ~ 1)
    heatmap combination requires using opencv (and therefore numpy arrays)
    '''

    if mask is None:
        img = rearrange(img.cpu().numpy(), 'c h w -> h w c')
        img = cv2.cvtColor(img * 255, cv2.COLOR_RGB2BGR).astype('uint8')
        result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = Image.fromarray(result)
        return result


    if power:
        mask = (mask ** 4)

    # convert to numpy array
    mask = rearrange(mask, '(h w) -> h w', h=int(mask.shape[0] ** 0.5))
    mask = mask.cpu().numpy()

    if color:
        mask = cv2.normalize(
            mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        mask = mask.astype('uint8')

    mask = cv2.resize(mask, (img.shape[-1], img.shape[-1]))

    img = rearrange(img.cpu().numpy(), 'c h w -> h w c')
    img = cv2.cvtColor(img * 255, cv2.COLOR_RGB2BGR).astype('uint8')

    if color:
        mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    else:
        mask = rearrange(mask, 'h w -> h w 1')        

    if color:
        result = cv2.addWeighted(mask, 0.5, img, 0.5, 0)
    else:
        result = (mask * img)
        result = cv2.normalize(
            result, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result = result.astype('uint8')

    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    result = Image.fromarray(result)

    return result


def setup_environment(args):
    set_seed(args.seed)

    dataset_train, args.num_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    loader_train = torch.utils.data.DataLoader(
        dataset_train, shuffle=args.shuffle_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    loader_test = torch.utils.data.DataLoader(
        dataset_val, shuffle=args.shuffle_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    print(args.dataset_name, args.num_classes, len(dataset_train), len(dataset_val))

    args.setting = 'ft' if args.finetune else 'fz'
    set_run_name(args, setting=True)

    if not args.debugging:
        wandb.init(project=args.wandb_project, entity=args.wandb_group, settings=wandb.Settings(start_method="fork"), config=args)
        wandb.run.name = args.run_name

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=True,
        pretrained_cfg=None,
        pretrained_cfg_overlay=None,
        num_classes=1000,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        img_size=args.input_size,
        args = args
    )
    if args.dataset_name.lower() != "imagenet":
        model.reset_classifier(args.num_classes)
    if args.num_clr:
        model.add_clr(args.num_clr)

    model.to(args.device)

    model.eval()

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)
        print('Loaded checkpoint: ', args.finetune)

    # only works for deit/vit base
    layers = []
    for name, _ in model.named_modules():
        # print(name)
        if ('vit_b16' in args.model) and ('attn.drop' in name or 'encoder_norm' in name):
            layers.append(name)
        elif ('deit' in args.model or 'vit' in args.model) and ('attn_drop' in name or name == 'model.norm'):
            layers.append(name)

    hook = VisHook(model, layers, args.device)

    return loader_train, loader_test, hook


def main():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    parser.add_argument('--shuffle_train', action='store_false', help='def uses random sampler')
    parser.add_argument('--shuffle_test', action='store_false', help='def uses random sampler')
    parser.add_argument('--vis_mask', type=str, default=None, help='')
    parser.add_argument('--vis_mask_pow', action='store_true',
                        help='power (of 4) masks when applying heatmap')
    parser.add_argument('--vis_cols', type=int, default=6,
                        help='how many columns when visualizing images')
    parser.set_defaults(output_dir='results_inference', batch_size=6)
    args = parser.parse_args()
    adjust_config(args)

    loader_train, loader_test, hook = setup_environment(args)

    amp_autocast = torch.cuda.amp.autocast if args.use_amp else suppress

    set_seed(args.seed)
    with torch.no_grad():
        with amp_autocast():
            hook.save_vis(args, loader_train, split='train')
            hook.save_vis(args, loader_test, split='test')

    if not args.debugging:
        wandb.finish()

    return 0


if __name__ == "__main__":
    main()

