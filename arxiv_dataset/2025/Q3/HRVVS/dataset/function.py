import random
import numpy as np
from PIL import Image, ImageEnhance
import re
from torch.utils.data import Sampler


def sort_key(s):
    numbers = re.findall(r'\d+', s)
    if numbers:
        return int(''.join(numbers))
    else:
        return s



class MVA_aug:
    def cv_random_flip(self, img, label):
        flip_flag = random.randint(0, 1)
        if flip_flag == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        return img, label


    def randomCrop(self, image, label):
        border = 30
        image_width = image.size[0]
        image_height = image.size[1]
        crop_win_width = np.random.randint(image_width - border, image_width)
        crop_win_height = np.random.randint(image_height - border, image_height)
        random_region = (
            (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
            (image_height + crop_win_height) >> 1)
        return image.crop(random_region), label.crop(random_region)


    def randomRotation(self, image, label):
        mode = Image.BICUBIC
        if random.random() > 0.8:
            random_angle = np.random.randint(-15, 15)
            image = image.rotate(random_angle, mode)
            label = label.rotate(random_angle, mode)
        return image, label


    def colorEnhance(self, image):
        bright_intensity = random.randint(5, 15) / 10.0
        image = ImageEnhance.Brightness(image).enhance(bright_intensity)
        contrast_intensity = random.randint(5, 15) / 10.0
        image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
        color_intensity = random.randint(0, 20) / 10.0
        image = ImageEnhance.Color(image).enhance(color_intensity)
        sharp_intensity = random.randint(0, 30) / 10.0
        image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
        return image


    def __call__(self, image, label):
        image, label = self.cv_random_flip(image, label)
        image, label = self.randomCrop(image, label)
        image, label = self.randomRotation(image, label)
        image = self.colorEnhance(image)
        return image, label



class VideoShuffleSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        video_lengths = [len(img_files) for img_files in data_source.videos]
        self.video_indices = []
        for i, length in enumerate(video_lengths):
            self.video_indices.extend([i] * length)

    
    def __iter__(self):
        video_starts = [0]
        for img_files in self.data_source.videos:
            video_starts.append(video_starts[-1] + len(img_files))
        video_order = list(range(len(self.data_source.videos)))
        random.shuffle(video_order)
        indices = []
        for video_idx in video_order:
            start = video_starts[video_idx]
            end = video_starts[video_idx + 1]
            indices.extend(range(start, end))
        return iter(indices)

    
    def __len__(self):
        return len(self.data_source)