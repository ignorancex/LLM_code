from configparser import Interpolation
import os
import sys
sys.path.append("./")

import time
import numpy as np
import cv2
import numpy as np

import torch.utils.data
import albumentations as A
import torchvision.transforms as transforms
import torch.nn.functional as F

from datasets.aug_default import aug_params 
from datasets.aug_default import aug_this_try, aug_None, aug_flip, aug_color_jitter, aug_blur, aug_distort
from utils.util_funcs import draw_dilation, gen_targets

import random
colors = [[0  ,   0,   0], [255,   0,   0], [0  , 255,   0], 
              [0  ,   0, 255], [255, 255,   0], [255, 165,   0]]

def visualize_hv_map(hv_map, base_path, img_idx):
    h_map = hv_map[:, :, 0]
    v_map = hv_map[:, :, 1]
    slash_45_map = hv_map[:, :, 2]
    slash_135_map = hv_map[:, :, 3]

    h_map = (h_map + 1) / 2 * 255
    h_map = np.array(h_map, dtype=np.uint8)
    h_map_vis = cv2.applyColorMap(h_map, cv2.COLORMAP_JET) 
    
    v_map = (v_map + 1) / 2 * 255
    v_map = np.array(v_map, dtype=np.uint8)
    v_map_vis = cv2.applyColorMap(v_map, cv2.COLORMAP_JET)

    slash_45_map = (slash_45_map + 1) / 2 * 255
    slash_45_map = np.array(slash_45_map, dtype=np.uint8)
    slash_45_map_vis = cv2.applyColorMap(slash_45_map, cv2.COLORMAP_JET)

    slash_135_map = (slash_135_map + 1) / 2 * 255
    slash_135_map = np.array(slash_135_map, dtype=np.uint8)
    slash_135_map_vis = cv2.applyColorMap(slash_135_map, cv2.COLORMAP_JET)

    h_path = f"{base_path}/{img_idx:04d}_h.png"
    v_path = f"{base_path}/{img_idx:04d}_v.png"
    slash_45_path = f"{base_path}/{img_idx:04d}_45.png"
    slash_135_path = f"{base_path}/{img_idx:04d}_135.png"

    cv2.imwrite(h_path, h_map_vis)
    cv2.imwrite(v_path, v_map_vis)
    cv2.imwrite(slash_45_path, slash_45_map_vis)
    cv2.imwrite(slash_135_path, slash_135_map_vis)


####
class CoNICDataset(torch.utils.data.Dataset):
    """Data Loader. Loads images from a file list and 
    performs augmentation with the albumentation library.
    After augmentation, horizontal and vertical maps are 
    generated.

    Args:
        file_list: list of filenames to load
        input_shape: shape of the input [h,w] - defined in config.py
        mask_shape: shape of the output [h,w] - defined in config.py
        mode: 'train' or 'valid'
        
    """

    # TODO: doc string

    def __init__(
        self,
        img_path,
        ann_path,
        indices=None,
        resample_indices=None,
        input_shape=None,
        mask_shape=None,
        run_mode="train",
        setup_augmentor=True,
        preprocess=None,
        visualize=False
    ):
        assert input_shape is not None and mask_shape is not None
        self.run_mode = run_mode
        self.aug_params = aug_params
        self.aug_this_try = aug_this_try
        self.visualize = visualize

        if preprocess == "aug_none":
            self.aug_this_try = aug_None
        elif preprocess == "aug_flip":
            self.aug_this_try = aug_flip
        elif preprocess == "aug_color_jitter":
            self.aug_this_try = aug_color_jitter
        elif preprocess == "aug_blur":
            self.aug_this_try = aug_blur
        elif preprocess == "aug_distort":
            self.aug_this_try = aug_distort   

        print(self.aug_this_try)            
                    
        self.imgs = np.load(img_path, mmap_mode='r')[indices]
        self.anns = np.load(ann_path, mmap_mode='r')[indices]

        self.resample_indices = resample_indices
        self.mask_shape = mask_shape
        self.input_shape = input_shape
        self.id = 0
        self.target_gen_func = gen_targets
        self.to_tensor = transforms.Compose([transforms.ToTensor()])

        print("len imgs: ", len(self.imgs))

        if setup_augmentor:
            self.setup_augmentor(0, 0)
        return

    def setup_augmentor(self, worker_id, seed):
        self.augmentor = self.__get_augmentation(self.run_mode, self.aug_params, self.aug_this_try)
        self.augs = self.augmentor
        self.id = self.id + worker_id
        return

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # RGB images

        img = np.array(self.imgs[idx]).astype("uint8")
        # instance ID map and type map
        ann = np.array(self.anns[idx]).astype("int32")

        if self.augs is not None:
            transformed = self.augs(image=img, mask=ann)
            img = transformed["image"]
            ann = transformed["mask"]
        
        feed_dict = {"img": img}
        feed_dict["tp_map"] = ann[..., 1]

        target_dict = self.target_gen_func(ann[..., 0], self.mask_shape)
        feed_dict.update(target_dict)

        img = self.to_tensor(img)
        np_map = (self.to_tensor(feed_dict["np_map"])).squeeze(0).type(torch.int64)
        hv_map = self.to_tensor(feed_dict["hv_map"])
        tp_map = (self.to_tensor(feed_dict["tp_map"])).squeeze(0).type(torch.int64)

        np_map = F.one_hot(np_map, num_classes=2)
        np_map = np_map.permute(2, 0, 1).contiguous()

        tp_map = F.one_hot(tp_map, num_classes=7)
        tp_map = tp_map.permute(2, 0, 1).contiguous()

        return img, np_map, hv_map, tp_map

    def __get_augmentation(self, mode, aug_params, aug_this_try):
        aug_dict = {# image and mask augmentation
                    # unharmful
                    "horizontal_flip": A.HorizontalFlip(p=0.5), 
                    "vertical_flip": A.VerticalFlip(p=0.5),
                    "transpose": A.Transpose(),
                    "random_rotate90": A.RandomRotate90(),

                    # might be harmful
                    "grid_distortion": A.GridDistortion(num_steps=aug_params['grid_distortion_num_steps'],
                                                        distort_limit=aug_params['grid_distortion_distort_limit'], p=0.8),
                    "grid_dropout": A.GridDropout(ratio=aug_params['grid_dropout_ratio'], p=0.8),
                    "random_resizecrop": A.RandomResizedCrop(height=self.input_shape[0], width=self.input_shape[1], 
                                                            scale=(aug_params['random_resizecrop_min'], 1), 
                                                            ratio=(aug_params['random_resizecrop_ration_min'], aug_params['random_resizecrop_ration_max']),
                                                            interpolation=cv2.INTER_CUBIC, p=0.8),
                    "random_grid_shuffle": A.RandomGridShuffle(grid=(aug_params["random_grid_shuffle_grid"], aug_params["random_grid_shuffle_grid"])),
                    "elastic_transform": A.ElasticTransform(sigma=aug_params['elastic_transform_sigma'],
                                                            alpha_affine=aug_params['elastic_transform_alpha_affine'], p=0.8),
                    "piecewise_affine": A.PiecewiseAffine(scale=(aug_params['piecewise_affine_scale_min'], aug_params['piecewise_affine_scale_max']),
                                                            nb_rows=aug_params['piecewise_affine_nb_rows'], nb_cols=aug_params['piecewise_affine_nb_cols'], p=0.8),
                    
                    ## Image augmentation
                    "clahe": A.CLAHE(clip_limit=aug_params['clahe_clip_limit'], p=0.8),
                    "emboss": A.Emboss(alpha=(aug_params["emboss_alpha_min"], aug_params["emboss_alpha_max"]), 
                                        strength=(aug_params["emboss_strength_min"], aug_params["emboss_strength_max"])),
                    "random_tone_curve": A.RandomToneCurve(scale=aug_params["random_tone_curve_scale"]),
                    
                    ## Blur
                    "down_scale": A.Downscale(scale_min=aug_params["down_scale_min"], scale_max=aug_params["down_scale_max"], interpolation=cv2.INTER_CUBIC, p=0.8),
                    "image_compression": A.ImageCompression(quality_lower=aug_params["img_compression_quality_lower"],
                                                            quality_upper=aug_params["img_compression_quality_upper"]),
                    "Gaussian_blur": A.GaussianBlur(blur_limit=aug_params["Gauss_blur_limit"]),
                    "glass_blur": A.GlassBlur(sigma=aug_params["glass_blur_sigma"]),
                    "median_blur": A.MedianBlur(blur_limit=aug_params["median_blur_limit"]),
                    "motion_blur": A.MotionBlur(blur_limit=(3, aug_params["motion_blur_limit"])),
                    "superpixels": A.Superpixels(n_segments=aug_params["superpixel_n_segments"]),
                    
                    # color jitter   
                    "color_jitter": A.ColorJitter(brightness=aug_params["color_jitter_brightness"], contrast=aug_params["color_jitter_contrast"],
                                                 saturation=aug_params["color_jitter_saturation"], hue=aug_params["color_jitter_hue"], 
                                                 p=0.8),                 
                    "hsv_jitter": A.HueSaturationValue(hue_shift_limit=aug_params["hsv_hue"], sat_shift_limit=aug_params["hsv_sat"],
                                                        val_shift_limit=aug_params["hsv_val"]),
                    "posterize": A.Posterize(num_bits=aug_params["posterrize_num_bits"]),
                    "random_brightness_contrast": A.RandomBrightnessContrast(brightness_limit=aug_params["random_brightness_contrast_brightness"],
                                                                            contrast_limit=aug_params["random_brightness_contrast_contrast"]),
                    "random_gamma": A.RandomGamma(gamma_limit=(aug_params["random_gamma_limit_min"], aug_params["random_gamma_limit_max"])),

                    
                    # Noise
                    "iso_noise": A.ISONoise(color_shift=(aug_params["iso_color_shift_min"], aug_params["iso_color_shift_max"]),
                                            intensity=(aug_params["iso_intensity_min"], aug_params["iso_intensity_max"])),
                    "Gauss_noise": A.GaussNoise(var_limit=(aug_params["Gauss_noise_var_min"], aug_params["Gauss_noise_var_max"])),
                    "multiplicative_noise": A.MultiplicativeNoise(multiplier=(aug_params["multiplicative_noise_multiplier_min"], 
                                                                            aug_params["multiplicative_noise_multiplier_max"])),
                    "random_fog": A.RandomFog(fog_coef_lower=aug_params["random_fog_coef_lower"],
                                                fog_coef_upper=aug_params["random_fog_coef_upper"],
                                                alpha_coef=aug_params["random_fog_alpha_coef"]),
                    "random_rain": A.RandomRain(slant_lower=aug_params["random_rain_slant_lower"], 
                                                slant_upper=aug_params["random_rain_slant_upper"],
                                                drop_length=aug_params["random_rain_drop_length"]),

                    # random shadow generates random shadow triangles
                    "random_shadow": A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_upper=aug_params["random_shadow_num_shadows_upper"]),
                    "random_snow": A.RandomSnow(snow_point_upper=aug_params["random_snow_point_upper"]),
                    "random_sun_flare": A.RandomSunFlare(flare_roi=(0, 0, 1, 1)),
                    
                    #must have
                    "center_crop": A.CenterCrop(self.input_shape[0], self.input_shape[1])
                    }

        if mode == "train":
            augmentation_list = [A.OneOf([aug_dict[_] for _ in aug_this_try["Oneof"]])]
            
            if len(aug_this_try["Someof"]) > 2:
                augmentation_list += [A.SomeOf([aug_dict[_] for _ in aug_this_try["Someof"]], n=2)]

            augmentation_list += [aug_dict[_] for compose in aug_this_try["Compose"] for _ in compose]
            augs = A.Compose(augmentation_list)
        else:
            augs = A.Compose([
                A.CenterCrop(self.input_shape[0], self.input_shape[1])
            ])

        print(augs)
        return augs


class CoNICDatasetPanMon(torch.utils.data.Dataset):
    """Data Loader. Loads images from a file list and 
    performs augmentation with the albumentation library.
    After augmentation, horizontal and vertical maps are 
    generated.

    Args:
        file_list: list of filenames to load
        input_shape: shape of the input [h,w] - defined in config.py
        mask_shape: shape of the output [h,w] - defined in config.py
        mode: 'train' or 'valid'
        
    """

    # TODO: doc string

    def __init__(
        self,
        img_path,
        ann_path,
        indices=None,
        resample_indices=None,
        input_shape=None,
        mask_shape=None,
        run_mode="train",
        setup_augmentor=True,
        augmentation_add=None,
        visualize=False
    ):
        assert input_shape is not None and mask_shape is not None
        self.run_mode = run_mode
        self.aug_params = aug_params
        self.aug_this_try = aug_this_try
        self.visualize = visualize

        if "monusac" in img_path:
            self.num_classes = 5
        elif "pannuke" in img_path:
            self.num_classes = 6

        if augmentation_add is not None:
            for key in augmentation_add.keys(): # Compose, Someof, Oneof
                for aug_key in augmentation_add[key].keys(): # color_jitter, iso_noise
                    if aug_key not in self.aug_this_try[key]:
                        self.aug_this_try[key].append(aug_key) 
                    for param_key in augmentation_add[key][aug_key].keys(): # color_jitter_brightness
                        self.aug_params[param_key] = augmentation_add[key][aug_key][param_key]
                    
        self.imgs = np.load(img_path, mmap_mode='r')
        self.anns = np.load(ann_path, mmap_mode='r')

        if indices is not None:
            self.imgs = self.imgs[indices]
            self.anns = self.anns[indices]

        self.resample_indices = resample_indices
        self.mask_shape = mask_shape
        self.input_shape = input_shape
        self.id = 0
        self.target_gen_func = gen_targets
        self.to_tensor = transforms.Compose([transforms.ToTensor()])

        print("len imgs: ", len(self.imgs))

        if setup_augmentor:
            self.setup_augmentor(0, 0)
        return

    def setup_augmentor(self, worker_id, seed):
        self.augmentor = self.__get_augmentation(self.run_mode, self.aug_params, self.aug_this_try)
        self.augs = self.augmentor
        self.id = self.id + worker_id
        return

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # RGB images
        img = np.array(self.imgs[idx]).astype("uint8")
        # instance ID map and type map
        ann = np.array(self.anns[idx]).astype("int32")

        if self.augs is not None:
            transformed = self.augs(image=img, mask=ann)
            img = transformed["image"]
            ann = transformed["mask"]
        
        feed_dict = {"img": img}
        feed_dict["tp_map"] = ann[..., 1]

        target_dict = self.target_gen_func(ann[..., 0], self.mask_shape)
        feed_dict.update(target_dict)

        img = self.to_tensor(img)
        np_map = (self.to_tensor(feed_dict["np_map"])).squeeze(0).type(torch.int64)
        hv_map = self.to_tensor(feed_dict["hv_map"])
        tp_map = (self.to_tensor(feed_dict["tp_map"])).squeeze(0).type(torch.int64)

        np_map = F.one_hot(np_map, num_classes=2)
        np_map = np_map.permute(2, 0, 1).contiguous()

        tp_map = F.one_hot(tp_map, num_classes=self.num_classes)
        tp_map = tp_map.permute(2, 0, 1).contiguous()

        # print("np map: ", np_map.shape, "hv map: ", hv_map.shape, "tp map: ", tp_map.shape)
        return img, np_map, hv_map, tp_map

    def __get_augmentation(self, mode, aug_params, aug_this_try):
        aug_dict = {# image and mask augmentation
                    # unharmful
                    "horizontal_flip": A.HorizontalFlip(p=0.5), 
                    "vertical_flip": A.VerticalFlip(p=0.5),
                    "transpose": A.Transpose(),
                    "random_rotate90": A.RandomRotate90(),

                    # might be harmful
                    "grid_distortion": A.GridDistortion(num_steps=aug_params['grid_distortion_num_steps'],
                                                        distort_limit=aug_params['grid_distortion_distort_limit'], p=0.8),
                    "grid_dropout": A.GridDropout(ratio=aug_params['grid_dropout_ratio'], p=0.8),
                    "random_resizecrop": A.RandomResizedCrop(height=self.input_shape[0], width=self.input_shape[1], 
                                                            scale=(aug_params['random_resizecrop_min'], 1), 
                                                            ratio=(aug_params['random_resizecrop_ration_min'], aug_params['random_resizecrop_ration_max']),
                                                            interpolation=cv2.INTER_CUBIC, p=0.8),
                    "random_grid_shuffle": A.RandomGridShuffle(grid=(aug_params["random_grid_shuffle_grid"], aug_params["random_grid_shuffle_grid"])),
                    "elastic_transform": A.ElasticTransform(sigma=aug_params['elastic_transform_sigma'],
                                                            alpha_affine=aug_params['elastic_transform_alpha_affine'], p=0.8),
                    "piecewise_affine": A.PiecewiseAffine(scale=(aug_params['piecewise_affine_scale_min'], aug_params['piecewise_affine_scale_max']),
                                                            nb_rows=aug_params['piecewise_affine_nb_rows'], nb_cols=aug_params['piecewise_affine_nb_cols'], p=0.8),
                    
                    ## Image augmentation
                    "clahe": A.CLAHE(clip_limit=aug_params['clahe_clip_limit'], p=0.8),
                    "emboss": A.Emboss(alpha=(aug_params["emboss_alpha_min"], aug_params["emboss_alpha_max"]), 
                                        strength=(aug_params["emboss_strength_min"], aug_params["emboss_strength_max"])),
                    "random_tone_curve": A.RandomToneCurve(scale=aug_params["random_tone_curve_scale"]),
                    
                    ## Blur
                    "down_scale": A.Downscale(scale_min=aug_params["down_scale_min"], scale_max=aug_params["down_scale_max"], interpolation=cv2.INTER_CUBIC, p=0.8),
                    "image_compression": A.ImageCompression(quality_lower=aug_params["img_compression_quality_lower"],
                                                            quality_upper=aug_params["img_compression_quality_upper"]),
                    "Gaussian_blur": A.GaussianBlur(blur_limit=aug_params["Gauss_blur_limit"]),
                    "glass_blur": A.GlassBlur(sigma=aug_params["glass_blur_sigma"]),
                    "median_blur": A.MedianBlur(blur_limit=aug_params["median_blur_limit"]),
                    "motion_blur": A.MotionBlur(blur_limit=(3, aug_params["motion_blur_limit"])),
                    "superpixels": A.Superpixels(n_segments=aug_params["superpixel_n_segments"]),
                    
                    # color jitter   
                    "color_jitter": A.ColorJitter(brightness=aug_params["color_jitter_brightness"], contrast=aug_params["color_jitter_contrast"],
                                                 saturation=aug_params["color_jitter_saturation"], hue=aug_params["color_jitter_hue"], 
                                                 p=0.8),                 
                    "hsv_jitter": A.HueSaturationValue(hue_shift_limit=aug_params["hsv_hue"], sat_shift_limit=aug_params["hsv_sat"],
                                                        val_shift_limit=aug_params["hsv_val"]),
                    "posterize": A.Posterize(num_bits=aug_params["posterrize_num_bits"]),
                    "random_brightness_contrast": A.RandomBrightnessContrast(brightness_limit=aug_params["random_brightness_contrast_brightness"],
                                                                            contrast_limit=aug_params["random_brightness_contrast_contrast"]),
                    "random_gamma": A.RandomGamma(gamma_limit=(aug_params["random_gamma_limit_min"], aug_params["random_gamma_limit_max"])),

                    
                    # Noise
                    "iso_noise": A.ISONoise(color_shift=(aug_params["iso_color_shift_min"], aug_params["iso_color_shift_max"]),
                                            intensity=(aug_params["iso_intensity_min"], aug_params["iso_intensity_max"])),
                    "Gauss_noise": A.GaussNoise(var_limit=(aug_params["Gauss_noise_var_min"], aug_params["Gauss_noise_var_max"])),
                    "multiplicative_noise": A.MultiplicativeNoise(multiplier=(aug_params["multiplicative_noise_multiplier_min"], 
                                                                            aug_params["multiplicative_noise_multiplier_max"])),
                    "random_fog": A.RandomFog(fog_coef_lower=aug_params["random_fog_coef_lower"],
                                                fog_coef_upper=aug_params["random_fog_coef_upper"],
                                                alpha_coef=aug_params["random_fog_alpha_coef"]),
                    "random_rain": A.RandomRain(slant_lower=aug_params["random_rain_slant_lower"], 
                                                slant_upper=aug_params["random_rain_slant_upper"],
                                                drop_length=aug_params["random_rain_drop_length"]),

                    # random shadow generates random shadow triangles
                    "random_shadow": A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_upper=aug_params["random_shadow_num_shadows_upper"]),
                    "random_snow": A.RandomSnow(snow_point_upper=aug_params["random_snow_point_upper"]),
                    "random_sun_flare": A.RandomSunFlare(flare_roi=(0, 0, 1, 1)),
                    
                    #must have
                    "center_crop": A.CenterCrop(self.input_shape[0], self.input_shape[1]),
                    # "resize": A.Resize(self.input_shape[0], self.input_shape[1])
                    }

        if mode == "train":
            augmentation_list = [A.OneOf([aug_dict[_] for _ in aug_this_try["Oneof"]])]
            
            if len(aug_this_try["Someof"]) > 2:
                augmentation_list += [A.SomeOf([aug_dict[_] for _ in aug_this_try["Someof"]], n=2)]

            augmentation_list += [aug_dict[_] for compose in aug_this_try["Compose"] for _ in compose]
            augs = A.Compose(augmentation_list)
        else:
            augs = A.Compose([
                A.CenterCrop(self.input_shape[0], self.input_shape[1])
            ])

        return augs


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    dataset = CoNICDatasetPanNuke("/mnt/group-ai-medical-SHARD/private/wenhuazhang/data/nuclei-level-multi-class/hovernet/monusac/")

    for img, np_map, hv_map, tp_map in dataset:
        print("img shape: ", img.shape, " img max: ", torch.max(img), " img min: ", torch.min(img))
        print("np map shape:  ", np_map.shape, " np map max: ", torch.max(np_map), " np map min: ", torch.min(np_map))
        print("hv map shape: ", hv_map.shape, " hv_map max: ", torch.max(hv_map), " hv_map min: ", torch.min(hv_map))
        print("tp map shape: ", tp_map.shape, " tp map max: ", torch.max(tp_map), " tp_map min: ", torch.min(tp_map))