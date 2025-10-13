from PIL import Image, ImageOps
import cv2
import numpy as np
import os

def load_image_chw(image_path):
    """
    Load an image in the (c, h, w) format.

    :param image_path: Path to the input image.
    :return: Image array in (c, h, w) format.
    """
    # Open the image using Pillow
    img = Image.open(image_path)
    # Convert the image to RGB format (3 channels)
    img = img.convert('RGB')
    # Convert the PIL image to a numpy array
    img_array = np.array(img)
    # Convert (h, w, c) to (c, h, w) format using numpy's transpose function
    img_chw = img_array.transpose((2, 0, 1))
    
    return img_chw

def resize_image(image, size=(512, 512), rgba=False, channel_first=False):
    """
    Resize the input image to the desired size without cropping, and save to the output path.

    :param input_path: Path to the input image.
    :param output_path: Path to save the resized image.
    :param size: Desired size (width, height).
    :param pad_color: Color to use for padding.
    """
    # Convert (c, h, w) to (h, w, c) format
    if channel_first:
        image_data_hwc = image.transpose((1, 2, 0))
        image = Image.fromarray(image_data_hwc.astype(np.uint8))
    
    # Calculate the padding
    delta_w = max(0, size[0] - image.width)
    delta_h = max(0, size[1] - image.height)
    if delta_w == 0 and delta_h == 0:
        return image
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    
    # Add padding
    padded_img = ImageOps.expand(image, padding)
    
    # Resize image
    resized_img = padded_img.resize(size)
    
    return resized_img

def resize_to_16_9(image_array):
    """
    Resize the input NumPy array image to a 16:9 ratio by cropping vertically.

    :param image_array: Image represented as a numpy array of shape (H, W, C).
    :return: Resized image as a numpy array.
    """
    original_height, original_width, _ = image_array.shape

    # Desired aspect ratio
    target_aspect_ratio = 16 / 9

    # Calculate the required height for the current width to meet the 16:9 ratio
    target_height = int(original_width / target_aspect_ratio)

    # Calculate cropping values for height
    excess_height = original_height - target_height
    top_crop = excess_height // 2
    bottom_crop = original_height - (excess_height - top_crop)

    # Crop the image
    cropped_image_array = image_array[top_crop:bottom_crop, :]

    return cropped_image_array

# sample reference frames from the whole video
def get_ref_index(f, neighbor_ids, length, num_ref, ref_length):
    ref_index = []
    if num_ref == -1:
        for i in range(0, length, ref_length):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, f - ref_length * (num_ref // 2))
        end_idx = min(length-1, f + ref_length * (num_ref // 2))
        for i in range(start_idx, end_idx + 1, ref_length):
            if i not in neighbor_ids:
                if len(ref_index) > num_ref:
                    break
                ref_index.append(i)
    return ref_index

# read frame-wise masks
def read_mask(mpath, size, video_length=None):
    masks = []
    mnames = os.listdir(mpath)
    mnames.sort()
    if video_length is not None:
        mnames = mnames[:video_length]
    for mp in mnames:
        m = Image.open(os.path.join(mpath, mp))
        m = resize_image(m,size)
        m = np.array(m.convert('L'))
        m = np.array(m > 0).astype(np.uint8)
        m = cv2.dilate(m,
                       cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                       iterations=4)
        masks.append(Image.fromarray(m * 255))
    return masks

def resize_frames(frames, size=None):
    if size is not None:
        frames = [resize_image(f,size) for f in frames]
    else:
        size = frames[0].size
        print(size)
    return frames, size

def read_frames_from_video(vname):
    frames = []
    vidcap = cv2.VideoCapture(vname)
    success, image = vidcap.read()
    count = 0
    while success:
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        frames.append(image)
        success, image = vidcap.read()
        count += 1
    return frames
