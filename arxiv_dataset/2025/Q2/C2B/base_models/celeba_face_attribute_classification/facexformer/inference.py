import argparse
import os

import numpy as np
import torch
import torchvision
from facenet_pytorch import MTCNN
from PIL import Image
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm

from network import FaceXFormer


IMAGE_EXTENSIONS = ['jpg', 'png', 'webp', 'jpeg', 'gif', 'bmp']
IMAGE_EXTENSIONS.extend([ext.upper() for ext in IMAGE_EXTENSIONS])


def adjust_bbox(x_min, y_min, x_max, y_max, image_width, image_height, margin_percentage=50):
    width = x_max - x_min
    height = y_max - y_min
    
    increase_width = width * (margin_percentage / 100.0) / 2
    increase_height = height * (margin_percentage / 100.0) / 2
    
    x_min_adjusted = max(0, x_min - increase_width)
    y_min_adjusted = max(0, y_min - increase_height)
    x_max_adjusted = min(image_width, x_max + increase_width)
    y_max_adjusted = min(image_height, y_max + increase_height)
    
    return x_min_adjusted, y_min_adjusted, x_max_adjusted, y_max_adjusted


def test(args):
    with torch.no_grad():
        device = "cuda:" + str(args.gpu_num)
        model = FaceXFormer().to(device)
        weights_path = args.model_path
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict_backbone'])

        model.eval()
        transforms_image = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(size=(224, 224), interpolation=InterpolationMode.BICUBIC),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        mtcnn = MTCNN(keep_all=True)

        image_files = [f for f in sorted(os.listdir(args.image_path)) if f.split('.')[-1] in IMAGE_EXTENSIONS]
        BS = min(args.batch_size, len(image_files))
        iterations = len(image_files) // BS + ((len(image_files) % BS) != 0)

        for i in tqdm(range(iterations)):
            i_files = image_files[i * BS:(i + 1) * BS]
            images = [Image.open(os.path.join(args.image_path, f)) for f in i_files]
            widths = [image.size[0] for image in images]
            heights = [image.size[1] for image in images]

            good_images_idx = []
            good_images = []

            B = len(i_files)

            for j in range(B):
                image = images[j]
                try:
                    boxes, _ = mtcnn.detect(image)
                except RuntimeError:
                    image = image.convert('RGB')
                    try:
                        boxes, _ = mtcnn.detect(image)
                    except RuntimeError:
                        continue
                try:
                    x_min, y_min, x_max, y_max = boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3]
                    x_min, y_min, x_max, y_max = adjust_bbox(x_min, y_min, x_max, y_max, widths[j], heights[j])
                    image = image.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
                    good_images.append(transforms_image(image))
                    good_images_idx.append(j)
                except TypeError:
                    pass

            if args.task == "parsing":
                task = torch.tensor([0])
            elif args.task == "landmarks":
                task = torch.tensor([1])
            elif args.task == "headpose":
                task = torch.tensor([2])
            elif args.task == "attributes":
                task = torch.tensor([3])
            elif args.task == "age_gender_race":
                task = torch.tensor([4])
            elif args.task == "visibility":
                task = torch.tensor([5])
            data = {'images': images, 'label': {"segmentation": torch.zeros([224, 224]), "lnm_seg": torch.zeros([5, 2]),
                                                "landmark": torch.zeros([68, 2]), "headpose": torch.zeros([3]),
                                                "attribute": torch.zeros([40]), "a_g_e": torch.zeros([3]),
                                                'visibility': torch.zeros([29])}, 'task': task}
            images, labels, tasks = data["images"], data["label"], data["task"]
            tasks = tasks.to(device=device)
            if good_images:
                images = torch.stack(good_images, dim=0).to(device=device)
                for k in labels.keys():
                    labels[k] = labels[k].unsqueeze(0).to(device=device)
                attribute_output = model(images, labels, tasks)
            if tasks[0] == 3:
                os.makedirs(args.results_path, exist_ok=True)
                output_logit_path = os.path.join(args.results_path, f'logits-{i:05d}.npy')
                dummy_output = np.full((B, 40), np.nan, dtype=np.float32)
                if good_images:
                    dummy_output[good_images_idx] = attribute_output.cpu().numpy()
                np.save(output_logit_path, dummy_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Provide absolute path to your weights file")
    parser.add_argument("--image_path", type=str, help="Provide absolute path to the image folder you want to perform inference on")
    parser.add_argument("--results_path", type=str, help="Provide path to the folder where results need to be saved")
    parser.add_argument("--task", type=str, help="parsing" or "landmarks" or "headpose" or "attributes" or "age_gender_race" or "visibility")
    parser.add_argument("--gpu_num", type=str, help="Provide the gpu number")
    parser.add_argument("--batch_size", type=int, help="Provide the batch size")
    args = parser.parse_args()
    test(args)
