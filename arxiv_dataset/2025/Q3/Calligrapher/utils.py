"""
    Helper functions mainly for data preparation.
"""
import cv2
import numpy as np
from PIL import Image


# Extracting the image and mask from gr.ImageEditor.
def process_gradio_source(editor_data):
    original_image = editor_data["background"].convert("RGB")
    layers = editor_data.get("layers", [])

    full_mask = Image.new("L", original_image.size, 0)
    cropped_region = original_image.copy()

    if not layers:
        return full_mask, cropped_region, original_image

    try:
        layer_image = layers[0]
        layer_pos = (0, 0)

        if layer_image.mode != "RGBA":
            layer_image = layer_image.convert("RGBA")

        alpha_channel = layer_image.split()[-1]

        full_mask = Image.new("L", original_image.size, 0)
        full_mask.paste(alpha_channel, layer_pos)
        full_mask = full_mask.point(lambda p: 255 if p > 128 else 0)

        original_np = np.array(original_image)
        mask_np = np.array(full_mask)

        mask_bool = mask_np > 0
        cropped_array = np.zeros_like(original_np)
        cropped_array[mask_bool] = original_np[mask_bool]

        cropped_region = Image.fromarray(cropped_array)

    except Exception as e:
        print(f"Raise error: {str(e)}")

    return original_image, full_mask, cropped_region


# Get bounding box from the mask.
def get_bbox_from_mask(mask_image):
    mask_array = np.array(mask_image)
    if mask_array.ndim == 3:
        mask_array = cv2.cvtColor(mask_array, cv2.COLOR_RGB2GRAY)
    _, binary_mask = cv2.threshold(mask_array, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    return (x, y), (x + w, y + h)


# Crop image from the bounding box coord.
def crop_image_from_bb(original_image, top_left, bottom_right):
    x1, y1 = map(int, top_left)
    x2, y2 = map(int, bottom_right)
    if not (0 <= x1 < x2 <= original_image.width and 0 <= y1 < y2 <= original_image.height):
        raise ValueError("Invalid bounding box coordinates")
    crop_box = (x1, y1, x2, y2)
    cropped_image = original_image.crop(crop_box)
    return cropped_image


# Resize image to the target size with zero paddings.
def resize_img_and_pad(input_image, target_size):
    cropped_width, cropped_height = input_image.size
    target_width, target_height = target_size
    scale = min(target_width / cropped_width, target_height / cropped_height)
    new_width = int(cropped_width * scale)
    new_height = int(cropped_height * scale)

    resized_image = input_image.resize((new_width, new_height), Image.BILINEAR)

    padded_image = Image.new("RGB", target_size, (0, 0, 0))

    left_padding = (target_width - new_width) // 2
    top_padding = (target_height - new_height) // 2
    padded_image.paste(resized_image, (left_padding, top_padding))

    return padded_image


def pil_to_np(pil_img):
    if pil_img.mode == 'RGBA':
        rgb = pil_img.convert('RGB')
        alpha = pil_img.split()[3]
        img = np.array(rgb)
        alpha = np.array(alpha)
    else:
        img = np.array(pil_img.convert('RGB'))
    return img


# Helper function considering the VAE compression factor.
def nearest_multiple_of_16(ref_height):
    lower = (ref_height // 16) * 16
    upper = ((ref_height + 15) // 16) * 16
    if ref_height - lower <= upper - ref_height:
        return lower
    else:
        return upper


# Resize to generate the reference context.
def generate_context_reference_image(reference_image_pil, img_width=512):
    reference_image_rgb = pil_to_np(reference_image_pil)
    ref_height, ref_width = reference_image_rgb.shape[0], reference_image_rgb.shape[1]

    ref_height = int((img_width / ref_width) * ref_height)
    ref_height = nearest_multiple_of_16(ref_height)

    reference_context = cv2.resize(reference_image_rgb, (img_width, ref_height))
    reference_new_pil = Image.fromarray(reference_context)
    return reference_new_pil
