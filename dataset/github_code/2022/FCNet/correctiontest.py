import os
import cv2 as cv
import argparse
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from models import FCNet
import torch
import torchvision.transforms.functional as F
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics.simple_metrics import peak_signal_noise_ratio as psnr


def load_imgs(paths):
    if not isinstance(paths, list):
        paths = [paths]

    imgs = ()
    for path in paths:
        img = Image.open(path)
        imgs += (F.to_tensor(img).unsqueeze(0),)

    return torch.cat(imgs, dim=0)

def suffix_belong(path, suffixes):
    if not isinstance(suffixes, list):
        suffixes = [suffixes]

    for suffix in suffixes:
        if path.endswith(suffix):
            return True

    return False

def evaluate(model, output_dir, input_label_dict, under_exp_suffixes, over_exp_suffixes):
    under_eval_1, under_eval_2, under_n = 0., 0., 0
    over_eval_1, over_eval_2, over_n = 0., 0., 0
    all_eval_1, all_eval_2, all_n = 0., 0., 0
    for i, (label_path, input_paths) in tqdm(enumerate(input_label_dict.items())):
        for input_path in input_paths:
            input_img = load_imgs(input_path).cuda()
            _, _, _, output = model(input_img)

            output_path = os.path.join(output_dir, os.path.basename(input_path))
            save_image(output, output_path)

            image = cv.imread(output_path)
            label = cv.imread(label_path)
            os.remove(output_path)

            if suffix_belong(output_path, under_exp_suffixes):
                under_eval_1 += psnr(image, label)
                under_eval_2 += ssim(image, label, multichannel=True)
                under_n += 1
            elif suffix_belong(output_path, over_exp_suffixes):
                over_eval_1 += psnr(image, label)
                over_eval_2 += ssim(image, label, multichannel=True)
                over_n += 1

            all_eval_1 += psnr(image, label)
            all_eval_2 += ssim(image, label, multichannel=True)
            all_n += 1

        under_eval_psnr = under_eval_1 / under_n
        under_eval_ssim = under_eval_2 / under_n

        over_eval_psnr = over_eval_1 / over_n
        over_eval_ssim = over_eval_2 / over_n

        all_eval_psnr = all_eval_1 / all_n
        all_eval_ssim = all_eval_2 / all_n

        f = os.path.join(output_dir, 'scores.txt')
        with open(f, "w") as file:
            file.write("under_exp_ssim=" + str(under_eval_ssim) + "\n")
            file.write("under_exp_psnr=" + str(under_eval_psnr) + "\n")
            file.write("over_eval_ssim=" + str(over_eval_ssim) + "\n")
            file.write("over_eval_psnr=" + str(over_eval_psnr) + "\n")
            file.write("all_eval_ssim=" + str(all_eval_ssim) + "\n")
            file.write("all_eval_psnr=" + str(all_eval_psnr) + "\n")

def init_parser():
    parser = argparse.ArgumentParser(description='Fusion evaluation')
    parser.add_argument('--test-dir', default='testing/INPUT_IMAGES', type=str, help='input path')
    parser.add_argument('--test-ext', default='.JPG', type=str, help='file extension of input images')
    parser.add_argument('--label-dir', default='testing/expert_c_testing_set', type=str, help='label path')
    parser.add_argument('--label-ext', default='.jpg', type=str, help='file extension of label images')
    parser.add_argument('--results-dir', default='results', type=str, help='results path')
    parser.add_argument('--checkpoint', default='snapshots/Epoch149.pth', type=str, help='checkpoint path')
    return parser


if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()

    output_dir = os.path.join(args.results_dir, 'correction')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    under_exp_suffixes = [f'{kw}{args.test_ext}' for kw in ['_0', '_N1', '_N1.5']]
    over_exp_suffixes = [f'{kw}{args.test_ext}' for kw in ['_P1', '_P1.5']]

    model = FCNet.FCNet(num_high=3).cuda()
    model.load_state_dict(torch.load(args.checkpoint))

    label_paths = Path(args.label_dir).glob('*' + args.label_ext)
    label_names = [os.path.split(path)[-1].replace(args.label_ext, '') for path in label_paths]

    input_label_dict = {}
    for name in label_names:
        label_path = os.path.join(args.label_dir, name + args.label_ext)
        input_paths = [str(path) for path in Path(args.test_dir).glob(name + '*')]
        input_label_dict[label_path] = input_paths

    evaluate(model, output_dir, input_label_dict, under_exp_suffixes, over_exp_suffixes)