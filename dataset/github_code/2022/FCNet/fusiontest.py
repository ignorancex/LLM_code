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


def load_imgs(paths, suffixes):
    if not isinstance(paths, list):
        paths = [paths]

    if not isinstance(suffixes, list):
        suffixes = [suffixes]

    imgs = ()
    suffixes_ = suffixes[:]
    for path in paths:
        for i, suffix in enumerate(suffixes_):
            if path.endswith(suffix):
                suffixes_.pop(i)
                img = Image.open(path)
                imgs += (F.to_tensor(img).unsqueeze(0),)
                break

    assert len(suffixes_) == 0, "all the suffixes must be found in the paths"

    return torch.cat(imgs, dim=0)

def evaluate(model, output_dir, input_label_dict, test_suffix):
    eval_1, eval_2, n = 0., 0., 0
    for i, (label_path, input_paths) in tqdm(enumerate(input_label_dict.items())):
        input_imgs = load_imgs(input_paths, test_suffix).cuda()
        _, _, _, output = model(input_imgs)

        output_path = os.path.join(output_dir, os.path.basename(label_path))
        save_image(output, output_path)

        image = cv.imread(output_path)
        label = cv.imread(label_path)
        os.remove(output_path)

        eval_1 += psnr(image, label)
        eval_2 += ssim(image, label, multichannel=True)
        n += 1

        eval_psnr = eval_1 / n
        eval_ssim = eval_2 / n

        f = os.path.join(output_dir, 'scores.txt')
        with open(f, "w") as file:
            file.write("ssim=" + str(eval_ssim) + "\n")
            file.write("psnr=" + str(eval_psnr) + "\n")

def init_parser():
    parser = argparse.ArgumentParser(description='Fusion evaluation')
    parser.add_argument('--exposure', choices=['under', 'over', 'all'])
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

    output_dir = os.path.join(args.results_dir, f'{args.exposure}-fusion')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.exposure == 'under':
        key_words = ['_0', '_N1', '_N1.5']
        test_suffix = [f'{kw}{args.test_ext}' for kw in key_words]
    elif args.exposure == 'over':
        key_words = ['_0', '_P1', '_P1.5']
        test_suffix = [f'{kw}{args.test_ext}' for kw in key_words]
    else:
        key_words = ['_0', '_N1', '_N1.5', '_P1', '_P1.5']
        test_suffix = [f'{kw}{args.test_ext}' for kw in key_words]

    model = FCNet.FCNet(num_high=3).cuda()
    model.load_state_dict(torch.load(args.checkpoint))

    label_paths = Path(args.label_dir).glob('*' + args.label_ext)
    label_names = [os.path.split(path)[-1].replace(args.label_ext, '') for path in label_paths]

    input_label_dict = {}
    for name in label_names:
        label_path = os.path.join(args.label_dir, name + args.label_ext)
        input_paths = [str(path) for path in Path(args.test_dir).glob(name + '*')]
        input_label_dict[label_path] = input_paths

    evaluate(model, output_dir, input_label_dict, test_suffix)