import torch
from PIL import Image, ImageOps
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from improved_consistency_model_conditional import ConsistencySamplingAndEditing
from lolv1.script import UNet
import os
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim
import numpy as np
import random

# =======================
# Configuration Parameters
# =======================

# Path to the trained LOLv2 model
model_path = "checkpoints/lolv1_128x128"  # Update this path if different

# Dataset paths
visible_folder = "../datasets/lolv1/eval15/low/"        # Visible (Low Exposure) images
infrared_folder = "../datasets/lolv1/eval15/high/"   # Infrared (Normal Exposure) images

# Output directories
output_folder = "lolv1_128x128"                                      # Directory to save generated images
metrics_file = "metrics_lolv1.txt"                                          # File to save PSNR and SSIM metrics

# Transformation after cropping and resizing
transform = T.Compose([
    T.ToTensor(),
    T.Lambda(lambda x: (x * 2) - 1),  # Normalize to [-1, 1]
])

# Sigma schedule for noise addition (adjust based on training)
sigmas = [80.0, 40.0, 20.0, 10.0, 5.0, 2.5, 1.25, 0.625, 0.3125, 0.15625, 0.078125]

# Number of images to process (set to None to process all images)
num_images_to_process = None  # Set to an integer value to limit the number of images

# Seed for reproducibility
random_seed = 28

# =======================
# Initialize Model and Device
# =======================

# Load the trained UNet model
model = UNet.from_pretrained(model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device).eval()

# Initialize the consistency sampling instance
consistency_sampling = ConsistencySamplingAndEditing()

# =======================
# Utility Functions
# =======================

def denormalize(tensor):
    """
    Denormalize a tensor from [-1, 1] to [0, 1].
    """
    return (tensor + 1) / 2

def calculate_metrics(reference, generated):
    """
    Calculate PSNR and SSIM between two images.
    
    Args:
        reference (numpy.ndarray): Reference image array.
        generated (numpy.ndarray): Generated image array.
        
    Returns:
        tuple: PSNR and SSIM values.
    """
    # Ensure the images are in the range [0, 1]
    reference = reference.astype(np.float32) / 255.0
    generated = generated.astype(np.float32) / 255.0

    # Calculate PSNR
    psnr_value = calculate_psnr(reference, generated, data_range=1.0)

    # Calculate SSIM
    ssim_value = calculate_ssim(reference, generated, data_range=1.0, multichannel=True, win_size=3)

    return psnr_value, ssim_value

def get_infrared_image_name(visible_image_name):
    """
    Given a visible image name like 'lowXXXX.png', return the corresponding
    infrared image name like 'normalXXXX.png'.
    
    Args:
        visible_image_name (str): Filename of the visible image.
        
    Returns:
        str: Corresponding infrared image filename.
        
    Raises:
        ValueError: If the visible image does not start with 'low'.
    """
    if visible_image_name.lower().startswith('low'):
        return visible_image_name
    else:
        raise ValueError(f"Unexpected visible image prefix in {visible_image_name}")

# =======================
# Main Processing Function
# =======================

def process_image(visible_image_name):
    """
    Process a single image pair: apply random crop, resize, generate denoised infrared,
    and calculate PSNR and SSIM.
    
    Args:
        visible_image_name (str): Filename of the visible image.
    """
    global total_psnr, total_ssim, num_processed_images

    # try:
        # Map to corresponding infrared image
        #infrared_image_name = get_infrared_image_name(visible_image_name)
    #except ValueError as ve:
        #print(f"Skipping {visible_image_name}: {ve}")
        #return

    # Construct full image paths
    visible_image_path = os.path.join(visible_folder, visible_image_name)
    infrared_image_path = os.path.join(infrared_folder, visible_image_name)

    # Check existence of infrared image
    if not os.path.exists(infrared_image_path):
        print(f"Infrared image {infrared_image_path} not found, skipping.")
        return

    # Load images
    try:
        visible_image = Image.open(visible_image_path).convert("RGB")
        infrared_image = Image.open(infrared_image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading images {visible_image_name} and/or {infrared_image_name}: {e}, skipping.")
        return


    # Crop both images
    visible_cropped = visible_image
    infrared_cropped = infrared_image

    # Resize to 128x128
    # visible_resized = visible_cropped.resize((128, 128), Image.BICUBIC)
    # infrared_resized = infrared_cropped.resize((128, 128), Image.BICUBIC)

    # Apply transformations
    visible_tensor = transform(visible_cropped).unsqueeze(0).to(device)
    infrared_tensor = transform(infrared_cropped).unsqueeze(0).to(device)

    # Add noise to the infrared image
    max_sigma = sigmas[0]  # Highest sigma value
    noise = torch.randn_like(infrared_tensor) * max_sigma
    noisy_infrared_tensor = noise

    # Generate the infrared image starting from the noisy infrared image
    try:
        with torch.no_grad():
            generated_infrared_tensor = consistency_sampling(
                model=model,
                y=noisy_infrared_tensor,
                v=visible_tensor,
                sigmas=sigmas,
                start_from_y=True,
                add_initial_noise=False,
                clip_denoised=True,
                verbose=False,  # Set verbose=False to reduce output
            )
    except Exception as e:
        print(f"Error during model inference for {visible_image_name}: {e}, skipping.")
        return

    # Denormalize tensors
    generated_infrared_denorm = denormalize(generated_infrared_tensor.squeeze(0).cpu())

    # Convert tensors to PIL images
    generated_infrared_pil = TF.to_pil_image(generated_infrared_denorm)

    # Reference infrared image (already resized to 128x128)
    reference_infrared_pil = infrared_cropped

    # Convert images to numpy arrays for metric calculation
    reference_image_np = np.array(reference_infrared_pil)
    generated_image_np = np.array(generated_infrared_pil)

    # Calculate PSNR and SSIM
    psnr_value, ssim_value = calculate_metrics(reference_image_np, generated_image_np)

    # Accumulate PSNR and SSIM
    total_psnr += psnr_value
    total_ssim += ssim_value
    num_processed_images += 1

    # Print PSNR and SSIM for the current image
    print(f"Image : PSNR = {psnr_value:.2f}, SSIM = {ssim_value:.4f}")

    # Save the generated infrared image for visual inspection
    output_filename_infrared = f"generated_infrared_{visible_image_name}"
    output_path_infrared = os.path.join(output_folder, output_filename_infrared)
    generated_infrared_pil.save(output_path_infrared)
    print(f"Saved generated infrared image to {output_path_infrared}\n")

# =======================
# Processing All Images
# =======================

def main():
    global total_psnr, total_ssim, num_processed_images
    total_psnr = 0.0
    total_ssim = 0.0
    num_processed_images = 0

    # Get a list of all images in the visible folder
    visible_images = os.listdir(visible_folder)
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    visible_images = [img for img in visible_images if img.lower().endswith(image_extensions)]

    # Optionally limit the number of images to process
    if num_images_to_process is not None:
        selected_visible_images = random.sample(visible_images, min(num_images_to_process, len(visible_images)))
    else:
        selected_visible_images = visible_images

    print(f"Processing {len(selected_visible_images)} images...\n")

    for idx, visible_image_name in enumerate(selected_visible_images, start=1):
        print(f"Processing image {idx}/{len(selected_visible_images)}: {visible_image_name}")
        process_image(visible_image_name)

    # Calculate and print average PSNR and SSIM
    if num_processed_images > 0:
        average_psnr = total_psnr / num_processed_images
        average_ssim = total_ssim / num_processed_images
        print(f"\nProcessed {num_processed_images} images.")
        print(f"Average PSNR: {average_psnr:.2f}")
        print(f"Average SSIM: {average_ssim:.4f}")

        # Save metrics to a text file
        with open(metrics_file, "a") as f:
            f.write(f"Processed {num_processed_images} images.\n")
            f.write(f"Average PSNR: {average_psnr:.2f}\n")
            f.write(f"Average SSIM: {average_ssim:.4f}\n\n")
        print(f"Saved metrics to {metrics_file}")
    else:
        print("No images were processed.")

if __name__ == "__main__":
    main()
