import os
import argparse
import json
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
from PIL import Image
from models import CompletionNetwork
from utils import poisson_blend, gen_input_mask
#from models_with_dual_dilatedconv import DualCompletionNetwork  # if you used DualCompletionNetwork
from modified_models import CompletionNetwork_ModifiedDilation

import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('config')
parser.add_argument('input_path')
parser.add_argument('output_path')
parser.add_argument('--max_holes', type=int, default=1)
parser.add_argument('--img_size', type=int, default=256)  # 160)
parser.add_argument('--hole_min_w', type=int, default=50)   #24
parser.add_argument('--hole_max_w', type=int, default=50)   #48
parser.add_argument('--hole_min_h', type=int, default=50)   #24
parser.add_argument('--hole_max_h', type=int, default=50)   #48


def predict(mpv, model, input_img, img_size, output_img):
    # convert img to tensor
    img = Image.open(input_img)
    img = transforms.Resize(img_size)(img)
    img = transforms.RandomCrop((img_size, img_size))(img)
    x = transforms.ToTensor()(img)
    x = torch.unsqueeze(x, dim=0)
    #ORG x = torch.cat((x, x, x), dim=1)  # convert to 3-channel format #SJ_FIX

    # create mask
    mask = gen_input_mask(
        shape=(1, 1, x.shape[2], x.shape[3]),
        hole_size=(
            (args.hole_min_w, args.hole_max_w),
            (args.hole_min_h, args.hole_max_h),
        ),
        max_holes=args.max_holes,
    )

    # inpaint
    model.eval()
    with torch.no_grad():
        x_mask = x - x * mask + mpv * mask
        input = torch.cat((x_mask, mask), dim=1)
        output = model(input)
        inpainted = poisson_blend(x_mask, output, mask)
        imgs = torch.cat((x, x_mask, inpainted), dim=0)
        save_image(imgs, output_img, nrow=3)
    print('output img was saved as %s.' % output_img)


def main(args):
    args.model = os.path.expanduser(args.model)
    args.config = os.path.expanduser(args.config)
    args.input_img = os.path.expanduser(args.input_path)
    args.output_img = os.path.expanduser(args.output_path)

    # =============================================
    # Load model
    # =============================================
    with open(args.config, 'r') as f:
        config = json.load(f)
    mpv = torch.tensor(config['mpv']).view(1, 3, 1, 1)
    #model = CompletionNetwork()  # Specify the model class name
    #model = DualCompletionNetwork() # TODO : Specify the completion network
    model = CompletionNetwork_ModifiedDilation()

    # -------------------------------
    # if model is trained one gpu
    # -------------------------------
    # ORG model.load_state_dict(torch.load(args.model, map_location='cpu')) # This is original way to load model, but my training is done in parallel mode

    # --------------------------------------------------------------------------------
    # To load model that was pre-trained with parallel training (using nn.DataParallel)
    # --------------------------------------------------------------------------------
    # original saved file with DataParallel
    state_dict = torch.load(args.model)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        print('[debug] k : ', k)
        #name = k  # k[7:] # remove `module.`
        name = k[7:]
        new_state_dict[name] = v
        # load params
    model.load_state_dict(new_state_dict)

    # -------------------------
    # Predict all test images
    # -------------------------
    test_image_path = args.input_path
    predicted_image_path = args.output_path
    img_size = args.img_size

    for f in glob.glob(test_image_path + "*.png"):
        fname = os.path.basename(f) # file name only
        input_img = test_image_path + fname
        output_img = predicted_image_path + fname
        predict(mpv, model, input_img, img_size, output_img)

def resize(output_path):
    for file in glob.glob(args.output_path + "*.png"):
        only_filename = os.path.basename(file)
        image = Image.open(file)
        image = image.resize((768, 256), Image.ANTIALIAS)
        image.save(fp=args.output_path + only_filename)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
    resize(args.output_path)
