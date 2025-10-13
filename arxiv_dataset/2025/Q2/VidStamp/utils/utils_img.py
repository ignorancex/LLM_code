import numpy as np
from augly.image import functional as aug_functional
import torch
from torchvision import transforms
from torchvision.transforms import functional
import cv2
import random
from PIL import Image, ImageFilter
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

default_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

normalize_vqgan = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize (x - 0.5) / 0.5
unnormalize_vqgan = transforms.Normalize(mean=[-1, -1, -1], std=[1/0.5, 1/0.5, 1/0.5]) # Unnormalize (x * 0.5) + 0.5
normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize (x - mean) / std
unnormalize_img = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]) # Unnormalize (x * std) + mean

def center_crop(x, scale):
    """ Perform center crop such that the target area of the crop is at a given scale
    Args:
        x: PIL image
        scale: target area scale 
    """
    scale = np.sqrt(scale)
    new_edges_size = [int(s*scale) for s in x.shape[-2:]][::-1]
    return functional.center_crop(x, new_edges_size)

def resize(x, scale):
    """ Perform center crop such that the target area of the crop is at a given scale
    Args:
        x: PIL image
        scale: target area scale 
    """
    scale = np.sqrt(scale)
    new_edges_size = [int(s*scale) for s in x.shape[-2:]][::-1]
    return functional.resize(x, new_edges_size)

def rotate(x, angle):
    """ Rotate image by angle
    Args:
        x: image (PIl or tensor)
        angle: angle in degrees
    """
    return functional.rotate(x, angle)

def adjust_brightness(x, brightness_factor):
    """ Adjust brightness of an image
    Args:
        x: PIL image
        brightness_factor: brightness factor
    """
    return normalize_img(functional.adjust_brightness(unnormalize_img(x), brightness_factor))

def adjust_contrast(x, contrast_factor):
    """ Adjust contrast of an image
    Args:
        x: PIL image
        contrast_factor: contrast factor
    """
    return normalize_img(functional.adjust_contrast(unnormalize_img(x), contrast_factor))

def adjust_saturation(x, saturation_factor):
    """ Adjust saturation of an image
    Args:
        x: PIL image
        saturation_factor: saturation factor
    """
    return normalize_img(functional.adjust_saturation(unnormalize_img(x), saturation_factor))

def adjust_hue(x, hue_factor):
    """ Adjust hue of an image
    Args:
        x: PIL image
        hue_factor: hue factor
    """
    return normalize_img(functional.adjust_hue(unnormalize_img(x), hue_factor))

def adjust_gamma(x, gamma, gain=1):
    """ Adjust gamma of an image
    Args:
        x: PIL image
        gamma: gamma factor
        gain: gain factor
    """
    return normalize_img(functional.adjust_gamma(unnormalize_img(x), gamma, gain))

def adjust_sharpness(x, sharpness_factor):
    """ Adjust sharpness of an image
    Args:
        x: PIL image
        sharpness_factor: sharpness factor
    """
    return normalize_img(functional.adjust_sharpness(unnormalize_img(x), sharpness_factor))

def overlay_text(x, text='Lorem Ipsum'):
    """ Overlay text on image
    Args:
        x: PIL image
        text: text to overlay
        font_path: path to font
        font_size: font size
        color: text color
        position: text position
    """
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    img_aug = torch.zeros_like(x, device=x.device)
    for ii,img in enumerate(x):
        pil_img = to_pil(unnormalize_img(img))
        img_aug[ii] = to_tensor(aug_functional.overlay_text(pil_img, text=text))
    return normalize_img(img_aug)

def jpeg_compress(x, quality_factor):
    """ Apply jpeg compression to image
    Args:
        x: PIL image
        quality_factor: quality factor
    """
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    img_aug = torch.zeros_like(x, device=x.device)
    for ii,img in enumerate(x):
        pil_img = to_pil(unnormalize_img(img))
        img_aug[ii] = to_tensor(aug_functional.encoding_quality(pil_img, quality=quality_factor))
    return normalize_img(img_aug)


def gaussian_blur(x, kernel_size=4):
    """ Apply Gaussian blur to an image
    Args:
        x: PIL image
        kernel_size: Size of the Gaussian kernel
    """
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    img_aug = torch.zeros_like(x, device=x.device)
    for ii, img in enumerate(x):
        pil_img = to_pil(unnormalize_img(img))
        img_aug[ii] = to_tensor(pil_img.filter(ImageFilter.GaussianBlur(kernel_size)))
    return normalize_img(img_aug)

def add_noise(x, mean=0, std=0.1):
    """ Add Gaussian noise to an image
    Args:
        x: Tensor image
        mean: Mean of Gaussian noise
        std: Standard deviation of Gaussian noise
    """
    noise = torch.randn_like(x) * std + mean
    return torch.clamp(x + noise, -1, 1)  # Ensure pixel values remain in range


def unnormalize_image(img):
    """ Ensure image tensor is properly normalized for visualization. """
    return img.clamp(0, 1)

def encode_mpeg4(x):
    """ Encode frames using MPEG-4 and return the processed frames.
    
    Args:
        x: Tensor image (BxCxHxW)
    
    Returns:
        Tensor image with MPEG-4-encoded and decoded frames.
    """
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    
    height, width = x.shape[-2:]
    filename = "temp_mpeg4.mp4"

    # MPEG-4 Encoding (Use 'MP4V' as the codec)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(filename, fourcc, 30, (width, height))

    if not out.isOpened():
        print("ERROR: MPEG-4 VideoWriter failed to open!")
        return x  # Return original tensor

    # Write frames to an MPEG-4 encoded video
    for ii, img in enumerate(x):
        pil_img = to_pil(unnormalize_image(img))
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        out.write(frame)
        #print(f"Writing frame {ii} to {filename}")

    out.release()
    time.sleep(1)  # Ensure file is fully written before reading

    # Read back the video and extract frames
    cap = cv2.VideoCapture(filename)

    if not cap.isOpened():
        print("ERROR: Failed to open MPEG-4 video file!")
        return x  # Return original tensor

    frames = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("No more frames to read!")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = to_tensor(Image.fromarray(frame))
        frames.append(frame)
        #print(f"Read frame {frame_idx} from {filename}")
        frame_idx += 1

    cap.release()

    if len(frames) == 0:
        print("ERROR: No frames were extracted from the encoded video!")
        return x  # Return original tensor

    return torch.stack(frames).to(x.device)
