from tools import *
import argparse
from pathlib import Path
import os
from skimage import morphology, measure
from tqdm import tqdm
import cv2
import torch
import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull


def preprocessing(args):
    Path(args.save_root).mkdir(exist_ok=True, parents=True)
    for model_name in args.model_name:
        print(f" this is {model_name}")
        predictor = load_model(model_name)

        directories_and_parts = {
            # "TG3K": ["test"],
            # "TN3K": ["test"],
            "DDTI": ["test"]
        }

        for directory, parts in directories_and_parts.items():
            for part in parts:
                image_path = f"./datasets/raw/{directory}/{part}/images"
                mask_path = f"./datasets/raw/{directory}/{part}/masks"

                crop_image_path = f"{args.save_root}/crop_image"
                Path(crop_image_path).mkdir(parents=True, exist_ok=True)

                crop_predict_path = f"{args.save_root}/crop_predict"
                Path(crop_predict_path).mkdir(parents=True, exist_ok=True)

                crop_mask_path = f"{args.save_root}/crop_mask"
                Path(crop_mask_path).mkdir(parents=True, exist_ok=True)

                sample_list = sorted(os.listdir(image_path))
                sample_list = [i for i in sample_list if i != ".ipynb_checkpoints"]

                for index, sample_name in tqdm(enumerate(sample_list), total=len(sample_list), desc=f"{directory} {part}"):
                    image = Image.open(f"{image_path}/{sample_name}").convert('RGB')
                    image = np.array(image)
                    image_h, image_w = image.shape[:2]
                    predictor.set_image(image)
                    retval, _, stats, centroids, mask = calc_information(os.path.join(mask_path, sample_name))
                    for i in range(retval):
                        x1, y1, x2, y2 = stats[i][0], stats[i][1], stats[i][0] + stats[i][2], stats[i][1] + stats[i][3]
                        if x2 - x1 >= 16 and y2 - y1 >= 16:
                            if args.prompt == "bbox":
                                input_bbox = np.array([[x1, y1, x2, y2]])
                                pred_raw, _, _ = predictor.predict(box=input_bbox, multimask_output=False, return_logits=True)
                                pred = torch.from_numpy(pred_raw).sigmoid().permute(1, 2, 0).numpy().astype(np.float32)
                                pred_mask = (pred > 0.5).astype(np.float32)

                                crop_image=Image.fromarray(image[y1:y2, x1:x2])
                                crop_predict = pred_mask[y1:y2, x1:x2].astype(np.uint8) * 255
                                crop_mask = mask[y1:y2, x1:x2]*255
                                crop_image.save(f"{crop_image_path}/{i}_{model_name}_{directory}_{part}_{sample_name}")
                                cv2.imwrite(f"{crop_mask_path}/{i}_{model_name}_{directory}_{part}_{sample_name}", crop_mask)
                                cv2.imwrite(f"{crop_predict_path}/{i}_{model_name}_{directory}_{part}_{sample_name}", crop_predict)
                            else:
                                input_point = np.array([centroids[i]])
                                input_label = np.array([1])
                                pred_raw, _, _ = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=False, return_logits=True)
                                pred = torch.from_numpy(pred_raw).sigmoid().permute(1, 2, 0).numpy().astype(np.float32)

                                temp_full = morphology.convex_hull_image(binary(pred[:, :, 0]))
                                temp_full = np.array(temp_full, dtype=np.uint8) * 255

                                _, _, pred_stats, _, _ = calc_predict_bbox(temp_full)
                                if len(pred_stats) != 0:
                                    x_1, y_1, x_2, y_2 = pred_stats[0][0], pred_stats[0][1], pred_stats[0][0] + pred_stats[0][2], pred_stats[0][1] + pred_stats[0][3]
                                    new_width = (x_2 - x_1) * 1.5
                                    new_height = (y_2 - y_1) * 1.5

                                    new_x_1 = int(x_1 - (new_width - (x_2 - x_1)) / 2)
                                    new_x_2 = int(x_2 + (new_width - (x_2 - x_1)) / 2)
                                    new_y_1 = int(y_1 - (new_height - (y_2 - y_1)) / 2)
                                    new_y_2 = int(y_2 + (new_height - (y_2 - y_1)) / 2)

                                    pad_top = max(0, -new_y_1)
                                    pad_bottom = max(0, new_y_2 - image_h)
                                    pad_left = max(0, -new_x_1)
                                    pad_right = max(0, new_x_2 - image_w)

                                    padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)
                                    padded_pred = np.pad(pred, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)
                                    padded_mask = np.pad(mask, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)

                                    new_x_1 += pad_left
                                    new_x_2 += pad_left
                                    new_y_1 += pad_top
                                    new_y_2 += pad_top

                                    padded_pred = (padded_pred > 0.5).astype(np.float32)

                                    crop_image = Image.fromarray(padded_image[new_y_1:new_y_2, new_x_1:new_x_2])
                                    crop_predict = padded_pred[new_y_1:new_y_2, new_x_1:new_x_2].astype(np.uint8) * 255
                                    crop_mask = padded_mask[new_y_1:new_y_2, new_x_1:new_x_2] * 255

                                    crop_image.save(f"{crop_image_path}/{i}_{model_name}_{directory}_{part}_{sample_name}")
                                    cv2.imwrite(f"{crop_mask_path}/{i}_{model_name}_{directory}_{part}_{sample_name}", crop_mask)
                                    cv2.imwrite(f"{crop_predict_path}/{i}_{model_name}_{directory}_{part}_{sample_name}", crop_predict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, nargs='+', default=["SAM"])  # "SAM", "SAM_Med2D", "MedSAM_bbox"
    parser.add_argument("--save_root", type=str, default="./datasets/preprocess/test_sam_DDTI")
    parser.add_argument("--prompt", type=str, default="bbox")
    args = parser.parse_args()
    preprocessing(args)


