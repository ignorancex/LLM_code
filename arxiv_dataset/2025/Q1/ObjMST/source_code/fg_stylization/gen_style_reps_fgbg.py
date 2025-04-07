import argparse
import os
import random
import time
from pprint import pprint

import torch
import torch.nn
import torch.optim as optim
from torchvision import utils as vutils

import loss_fn_fgbg
import model_bridge_stya2k
import utils_fgbg as utils

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--content_base', type=str, default='/path/to/Input/Images/', help='path to content base folder')
parser.add_argument('--masks_base', type=str, default='/path/to/Input/masks/', help='path to masks folder')
parser.add_argument('--exp_name', type=str, default="default_new", help='Experiment name')
parser.add_argument('--sty_text', nargs='*', type=str, default=[], help='style description text')
parser.add_argument('--sty_img', nargs='*', type=str,  default=[], help='paths to style image')
parser.add_argument('--alpha_img', nargs='*', type=float, default=[], help='input style image patch loss parameter')
parser.add_argument('--bidirectional_isi', action='store_true', help='use bidirectional input style image loss')
parser.add_argument('--not_pairwise_isi', action='store_true', help='do not use pairwise input style image loss')
parser.add_argument('--resample_isi_patch', action='store_true', help='resample input style image patch loss')


parser.add_argument('--alpha_text', nargs='*', type=float, default=[],
                    help='PatchCLIP loss parameter for the style image')
parser.add_argument('--crop_size', type=int, default=256,
                    help='cropped image size')
parser.add_argument('--crop_size_isi', type=int, default=None,
                    help='cropped image size')
parser.add_argument('--num_crops', type=int, default=64,
                    help='number of patches')
parser.add_argument('--img_width', type=int, default=512,
                    help='size of images')
parser.add_argument('--img_height', type=int, default=512,
                    help='size of images')
parser.add_argument('--max_step', type=int, default=40, help='max steps')
parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
parser.add_argument('--sty_thresh', type=float, default=0.0, help='style patch loss threshold')
parser.add_argument('--ws_mode', type=str, default='global_directional')
parser.add_argument('--num_ens', type=int, default=8)
parser.add_argument('--output_dir', type=str, default='./outputs_fgbg/')
parser.add_argument('--augs', type=str, nargs='*', default=['perspective', ])
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--random_ws', action='store_true', help='randomly init the w+ space')
parser.add_argument('--naive_inversion', action='store_true', help='naive inversion for style image')

parser.add_argument('--stylegan3_path', type=str, default='/path/to/source_code/fg_stylization/stylegan3')
parser.add_argument('--stylegan3_pkl', type=str, default='/path/to/source_code/fg_stylization/stylegan3/models/wikiart-1024-stylegan3-t-17.2Mimg.pkl')
parser.add_argument('--save_ws', action='store_true', help='save the inverted ws to file')
parser.add_argument('--save_rep_every_log', action='store_true', help='save the style representation every 20 iters')

args = parser.parse_args()

assert (args.img_width%8)==0, "width must be multiple of 8"
assert (args.img_height%8)==0, "height must be multiple of 8"
style_mode = 'stylegan3'

if args.crop_size_isi is None:
    args.crop_size_isi = args.crop_size


if len(args.sty_text) > 1 and len(args.alpha_text) == 1:
    args.alpha_text = [args.alpha_text[0] / len(args.sty_text) ] * len(args.sty_text)
else: 
    assert not args.sty_text or len(args.alpha_text) == len(args.sty_text), "number of style text and style patch loss must match"

if len(args.sty_img) > 1 and len(args.alpha_img) == 1:
    args.alpha_img = [args.alpha_img[0] / len(args.sty_img) ] * len(args.sty_img)
else:
    assert not args.sty_img or len(args.alpha_img) == len(args.sty_img), "number of style image and style patch loss must match"



args.stylegan3_path = os.path.abspath(args.stylegan3_path)
args.stylegan3_pkl = os.path.abspath(args.stylegan3_pkl)

print('\n\n')
print('==================================================')
print('Using the following args:')
pprint(vars(args))
print('==================================================')


print(f'used augmentations: {args.augs}')


num_crops = args.num_crops
source_text = 'a Photo'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

exp = args.exp_name + f'_{style_mode}'
exp_dict = {
    '_sty_thresh': args.sty_thresh,
    '_csize': args.crop_size,
    '_lr': args.lr,
    '_wm': utils.shorten(args.ws_mode),
}
for key, value in exp_dict.items():
    exp += f'{key}{value}'


clip_loss_model = loss_fn_fgbg.CLIPLossModel('ViT-B/32', device, jit=False)


stylegan, ws = model_bridge_stya2k.load_stylegan3_model_with_ws(
    stylegan3_path=args.stylegan3_path,
    network_pkl=args.stylegan3_pkl,
    seed=None
)

style_image_list = []

content_imgs = []
masks_imgs=[]
# print("content_base", args.content_base)
for cpath in os.listdir(args.content_base):
    # print("cpath=", cpath)
    content_image = utils.load_image_512(os.path.join(args.content_base, cpath))
    content_imgs.append(content_image.to(device))
# print("content_imgs", content_imgs)

print("masks_base", args.masks_base)
for mpath in os.listdir(args.masks_base):
    # print("mpath=", mpath)
    mask_image = utils.load_image_512(os.path.join(args.masks_base, mpath))
    masks_imgs.append(mask_image.to(device))
# print(len(masks_imgs))
# print(masks_imgs[0].shape)


if args.sty_img:
    input_sty_img =  []
    isi_names = []
    for isipath in args.sty_img:
        input_sty_img.append(utils.load_image_512(isipath).to(device))
        isi_names.append(os.path.splitext(os.path.basename(isipath))[0].replace('_', '-'))
else:
    isi_names = []
# print("isi_names", isi_names)

mse_loss = torch.nn.MSELoss()


def gen_one(num):
    output_dir = os.path.join(args.output_dir, args.exp_name, 'style_reps')
    os.makedirs(output_dir, exist_ok=True)
    # print("output_dir=", output_dir)

    # randomly choose one content image from content_imgs
    content_image = random.choice(content_imgs)
    index_random_contentImage = next(i for i, tensor in enumerate(content_imgs) if torch.equal(tensor, content_image))
    # print("index of content image is", "=", index_random_contentImage)
    masked_image=masks_imgs[index_random_contentImage]
    # print("masked_image.shape=",masked_image.shape)
    # print("len(content_imgs)", len(content_imgs))
    # print("content_image shape=", content_image.shape)

    num_ws_samples = 1 if args.random_ws else 128
    # print("num_ws_samples", num_ws_samples)
    # print("len(input_sty_img)=", len(input_sty_img))
    # print("input_sty_img[0].shape=", input_sty_img[0].shape)
    if args.sty_img:
        input_sty_img_patch = [utils.make_patchs(isi, num_crops, args.crop_size_isi, augs=args.augs) for isi in input_sty_img]
    else:
        input_sty_img_patch = None
    # print("len(input_sty_img_patch)=", len(input_sty_img_patch))
    # print("input_sty_img_patch[0].shape=", input_sty_img_patch[0].shape)
    # print("args.ws_mode", args.ws_mode)
    ws = utils.get_best_ws_multi(
        stylegan=stylegan, 
        clip_loss_model=clip_loss_model, 
        style_text=args.sty_text,
        isi_patches = input_sty_img_patch,
        bidirectional_isi=args.bidirectional_isi,
        pairwise_isi=not args.not_pairwise_isi,
        lambda_style_patch=args.alpha_text * 4,
        lambda_isi_patch=args.alpha_img * 4,
        mode=args.ws_mode, 
        content_image=content_image, 
        masked_image=masked_image,
        source_text=source_text,
        num=num_ws_samples,
    )
    # print("ws.shape=", ws.shape)
    optimizer = optim.Adam([ws], lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1/3)
    steps = args.max_step

    total_loss_epoch = []


    epoch_start_time = time.time()
    print('Initialization time: {:.2f}s'.format(epoch_start_time - start_time))
    mask=masked_image.bool()
    for epoch in range(0, steps+1):
        style_image = stylegan.forward_512(ws)
        # print("----------------------------------style image from latent code w_S---------------------")
        # print("style_image.shape", style_image.shape)
        if not 0.0 <= style_image.min() and style_image.max() <= 1.0:
            print(f'min: {style_image.min()}, max: {style_image.max()}')
            raise ValueError('style_image out of range')

        style_patch = utils.make_patchs(style_image, num_crops, args.crop_size, augs=args.augs)
        mask_patch= utils.make_patchs(masked_image, num_crops, args.crop_size, augs=args.augs)
        bool_mask_patch=mask_patch.bool()
        content_image = random.choice(content_imgs)
        index_random_contentImage = next(i for i, tensor in enumerate(content_imgs) if torch.equal(tensor, content_image))
        # print("index of content image is", "=", index_random_contentImage)
        # print("style_patch.shape=", style_patch.shape)
        # print("content_image.shape=", content_image.shape)

        # print("\n")
        total_loss = 0.0

        loss_style_patch = torch.tensor(0.0).to(device)
        if args.sty_text and any([l > 0 for l in args.alpha_text]):
            # print("-------------------------In Text--------------------------------")
            for sty_weight, st in zip(args.alpha_text, args.sty_text):                
                # print("sty_weight=", sty_weight, "st=", st, "source_text=", source_text)
                # print("style_patch.shape (target1)=", style_patch.shape, "content_image.shape (source1)=", content_image.shape)
                # print("st (target2)=", st, "source_text= (source2)", source_text)
                loss_style_patch = loss_style_patch + sty_weight * clip_loss_model.directional_loss(
                    style_patch, content_image.masked_fill(mask, 1), st, source_text, thresh=args.sty_thresh
                )
    
            total_loss = total_loss + loss_style_patch 
            # print("loss_style_patch", loss_style_patch)
            # print("total Loss=", total_loss)
        loss_isi_patch = torch.tensor(0.0).to(device)
        if args.sty_img and any([l > 0 for l in args.alpha_img]):
            # print("\n-------------------------In Image--------------------------------")
            if args.resample_isi_patch:
                input_sty_img_patch = [utils.make_patchs(isi, num_crops, args.crop_size_isi, augs=args.augs) for isi in input_sty_img]
                # print("input_sty_img_patch.shape=", input_sty_img_patch.shape)
                 
            if args.naive_inversion:
                print("In Naive inversion:")
                for isi_weight, isi in zip(args.alpha_img, input_sty_img):
                    # print("style image.shape=", style_image.shape, "isi.shape=" ,isi.shape)
                    loss_isi_patch = loss_isi_patch + isi_weight * mse_loss(style_image, isi)
            else:
                for isi_weight, isi_patch in zip(args.alpha_img, input_sty_img):
                # for isi_weight, isi_patch, mask_pch in zip(args.alpha_img, input_sty_img, mask_ctnt_img_patch):
                    # reshaped_mask_pch = mask_pch.unsqueeze(1).repeat(1, 3, 1, 1)
                    # boolreshaped_mask_pch=reshaped_mask_pch.bool()
                    # print("style patch.shape=", style_patch.shape, "isi_patch.shape=" ,isi_patch.shape)
                    # print("bool_mask_patch.shape=", bool_mask_patch.shape, "mask.shape=" ,mask.shape)
                    # print("style_patch.shape (target1)=", style_patch.shape, "content_image.shape (source1)=", content_image.shape)
                    # print("isi_patch.shape (target2)=", isi_patch.shape,"content_image.shape (source2)=", content_image.shape)
                    loss_isi_patch = loss_isi_patch + isi_weight * clip_loss_model.directional_loss(
                        style_patch, content_image.masked_fill(mask, 1), isi_patch, content_image.masked_fill(mask, 1),
                        thresh=args.sty_thresh, bidirectional_isi=args.bidirectional_isi, 
                        pairwise_isi=not args.not_pairwise_isi
                    )

            total_loss = total_loss + loss_isi_patch
            # print("loss_isi_patch", loss_isi_patch)
            # print("total Loss=", total_loss)
            

        total_loss_epoch.append(total_loss.item())
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 20 == 0:
            print("\nAfter %d iterions:" % epoch)
            print('ws norm: ', ws.norm().item())
            print('Total loss: ', total_loss.item())
            if loss_style_patch.item() > 0.0:
                print('Style patch loss: ', loss_style_patch.item() / sum(args.alpha_text))
            if loss_isi_patch.item() > 0.0:
                print('ISI patch loss: ', loss_isi_patch.item() / sum(args.alpha_img))

            print(f'20 epochs takes: {time.time() - epoch_start_time}s')
            epoch_start_time = time.time()


            if args.save_rep_every_log:
                sty_path = os.path.join(output_dir, '+'.join(args.sty_text + isi_names) + '_' + exp + f'_{num}-style_iter{epoch}.png')

                print(os.path.abspath(sty_path))
                vutils.save_image(
                    style_image,
                    sty_path,
                    nrow=1,
                    normalize=True
                )

        if args.save_ws and epoch % 100 == 0:
            ws_path = sty_path.replace('.png', '.pt')
            torch.save(ws.detach().cpu(), ws_path)
            print(f'saved ws to {ws_path}')                 


    sty_path = os.path.join(output_dir, '+'.join(args.sty_text + isi_names) + '_' + exp + f'_{num}-style.png')

    print(os.path.abspath(sty_path))
    vutils.save_image(
        style_image,
        sty_path,
        nrow=1,
        normalize=True
    )
    
    if args.save_ws:
        ws_path = sty_path.replace('.png', '.pt')
        torch.save(ws.detach().cpu(), ws_path)
        print(f'saved ws to {ws_path}')
            

    return style_image.detach()


for num in range(args.num_ens):
    print(f'{num + 1}th/{args.num_ens}  generation')
    style_image_list.append(gen_one(num))

duration = time.time() - start_time
print(f'Total time: {duration}s')
