from noise_layers.identity import Identity
from noise_layers.diff_jpeg import DiffJPEG
from noise_layers.gaussian import Gaussian
from noise_layers.brightness import Brightness
from noise_layers.gaussian_blur import GaussianBlur
import random
import torch


def add_noise(encoded_images, device, target=True):
    noised_images = None
    noise_list = [0, 1, 2, 3, 4]
    for i in range(encoded_images.shape[0]):
        choice = random.choice(noise_list)
        if choice == 0:
            noise_layers = Identity()
        elif choice == 1:
            if target:
                noise_layers = DiffJPEG(random.randint(10, 99), device)
            else:
                noise_layers = DiffJPEG(random.randint(50, 99), device)
        elif choice == 2:
            noise_layers = Gaussian(random.uniform(0, 0.1))
        elif choice == 3:
            if target:
                noise_layers = GaussianBlur(std=random.uniform(0, 2.0))
            else:
                noise_layers = GaussianBlur(std=random.uniform(0, 1.0))
        elif choice == 4:
            noise_layers = Brightness(random.uniform(1.0, 3))
        img_noised = noise_layers(encoded_images[i:i + 1, :, :, :])
        if i == 0:
            noised_images = img_noised
        else:
            noised_images = torch.cat((noised_images, img_noised), 0)
    return noised_images
