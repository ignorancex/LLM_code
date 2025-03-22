import torch
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from improved_consistency_model_conditional import ConsistencySamplingAndEditing
from bci.script import UNet
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np


##In this context visible = HE, infrared = IHC (metrics was borrowed from llvip, hence the nomenclature)
# ===============================
# Configuration and Setup
# ===============================

# -------------------------------
# 1. Model Loading
# -------------------------------

# Path to the pre-trained model checkpoint
model_path = "checkpoints/bci"

# Load the pre-trained UNet model
model = UNet.from_pretrained(model_path)

# Select device: GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device).eval()

# -------------------------------
# 2. Image Transformations
# -------------------------------

# Transformation pipeline: Convert PIL Image to Tensor and normalize to [-1, 1]
transform = T.Compose([
    T.ToTensor(),                           # Converts PIL Image to Tensor and scales to [0, 1]
    T.Lambda(lambda x: (x * 2) - 1),        # Normalize to [-1, 1]
])

# Inverse transformation: Denormalize to [0, 1] and clamp
inverse_transform = T.Compose([
    T.Lambda(lambda x: (x + 1) / 2),        # Denormalize to [0, 1]
    T.Lambda(lambda x: x.clamp(0, 1))       # Clamp values to [0, 1]
])

# -------------------------------
# 3. Define Image Folders
# -------------------------------

visible_folder = "../datasets/bci/HE/test"   # Folder containing visible (HE) images
infrared_folder = "../datasets/bci/IHC/test" # Folder containing infrared (IHC) images

# -------------------------------
# 4. Sampling Instance and Sigma Schedule
# -------------------------------

# Initialize the consistency sampling and editing instance
consistency_sampling = ConsistencySamplingAndEditing()

# Define the sigma schedule for noise levels
sigmas = [80.0, 40.0, 20.0, 10.0, 5.0, 2.5, 1.25, 0.625, 0.3125, 0.15625, 0.078125]

# -------------------------------
# 5. Metrics Accumulators and Results Folder
# -------------------------------

# Initialize accumulators for PSNR and SSIM metrics
total_psnr = 0.0
total_ssim = 0.0
num_images = 0

# Create results folder to save concatenated images
results_folder = "results_bci"
os.makedirs(results_folder, exist_ok=True)

# -------------------------------
# 6. Define Resize Transforms
# -------------------------------

# Resize down to 512x512 with bicubic interpolation and anti-aliasing
resize_down = T.Resize((512, 512), interpolation=T.InterpolationMode.BICUBIC, antialias=True)

# Resize up to 1024x1024 with bicubic interpolation
resize_up = T.Resize((1024, 1024), interpolation=T.InterpolationMode.BICUBIC)

# ===============================
# Main Processing Loop
# ===============================

# Iterate through all visible images
for idx, visible_image_name in enumerate(os.listdir(visible_folder), start=1):
    # Construct full paths to visible and infrared images
    visible_image_path = os.path.join(visible_folder, visible_image_name)
    infrared_image_path = os.path.join(infrared_folder, visible_image_name)

    # Check if the corresponding infrared image exists
    if not os.path.exists(infrared_image_path):
        print(f"[{idx}] Infrared image {infrared_image_path} not found, skipping.")
        continue

    try:
        # Load images and convert to RGB
        visible_image = Image.open(visible_image_path).convert("RGB")
        infrared_image = Image.open(infrared_image_path).convert("RGB")
    except Exception as e:
        print(f"[{idx}] Error loading images: {e}, skipping {visible_image_name}")
        continue

    # Apply transformations: Convert images to tensors and normalize
    visible_tensor = transform(visible_image).unsqueeze(0).to(device)    # Shape: [1, 3, 1024, 1024]
    infrared_tensor = transform(infrared_image).unsqueeze(0).to(device)  # Shape: [1, 3, 1024, 1024]

    # Resize infrared image to 512x512
    infrared_resized = TF.resize(infrared_tensor, [512, 512], interpolation=TF.InterpolationMode.BICUBIC, antialias=True)  # Shape: [1, 3, 512, 512]

    # Resize visible image to 512x512 for model input
    visible_resized = TF.resize(visible_tensor, [512, 512], interpolation=TF.InterpolationMode.BICUBIC, antialias=True)  # Shape: [1, 3, 512, 512]

    # Add Gaussian noise to the resized infrared image
    noise = torch.randn_like(infrared_resized) * sigmas[0]  # Sigma = 80.0
    noisy_infrared_tensor = infrared_resized + noise

    try:
        # Generate denoised infrared image using the model
        with torch.no_grad():
            generated_infrared_tensor = consistency_sampling(
                model=model,
                y=noisy_infrared_tensor,
                v=visible_resized,
                sigmas=sigmas,
                start_from_y=True,
                add_initial_noise=False,
                clip_denoised=True,
                verbose=False,
            )
    except Exception as e:
        print(f"[{idx}] Error during model inference: {e}, skipping {visible_image_name}")
        continue

    # Denormalize the generated infrared tensor to [0, 1]
    generated_infrared_denorm = inverse_transform(generated_infrared_tensor.squeeze(0).cpu()).clamp(0, 1).numpy().transpose(1, 2, 0)  # [512, 512, 3]

    # Upsize the generated infrared image back to 1024x1024
    generated_infrared_resized_pil = resize_up(T.ToPILImage()(generated_infrared_denorm)).convert("RGB")  # PIL Image
    generated_infrared_resized = T.ToTensor()(generated_infrared_resized_pil).numpy().transpose(1, 2, 0)      # [1024, 1024, 3]

    # Convert generated infrared image to uint8 for saving
    generated_image_save = (generated_infrared_resized * 255).astype(np.uint8)

    # Original visible and infrared images are already at 1024x1024
    visible_image_save = visible_image
    infrared_image_save = infrared_image

    # -------------------------------
    # Concatenate Images for Visualization
    # -------------------------------

    # Calculate concatenated image dimensions
    concatenated_width = visible_image_save.width + infrared_image_save.width + generated_image_save.shape[1]  # 1024 + 1024 + 1024 = 3072
    concatenated_height = max(visible_image_save.height, infrared_image_save.height, generated_image_save.shape[0])   # 1024

    # Create a new blank image for concatenation
    concatenated_image = Image.new("RGB", (concatenated_width, concatenated_height))

    # Paste the original visible image
    concatenated_image.paste(visible_image_save, (0, 0))

    # Paste the original infrared image next to the visible image
    concatenated_image.paste(infrared_image_save, (visible_image_save.width, 0))

    # Paste the generated infrared image next to the original infrared image
    concatenated_image.paste(Image.fromarray(generated_image_save), (visible_image_save.width + infrared_image_save.width, 0))

    # Save the concatenated image
    concatenated_image_path = os.path.join(results_folder, f"concatenated_{visible_image_name}")
    concatenated_image.save(concatenated_image_path)

    # -------------------------------
    # Calculate PSNR and SSIM Metrics
    # -------------------------------

    # Convert original infrared image to tensor and denormalize to [0, 1] for metric calculation
    infrared_original_tensor = transform(infrared_image).unsqueeze(0).to(device)  # Shape: [1, 3, 1024, 1024]
    infrared_original_denorm = inverse_transform(infrared_original_tensor.squeeze(0).cpu()).clamp(0, 1).numpy().transpose(1, 2, 0)  # [1024, 1024, 3]

    # Ensure generated_infrared_resized is in [0,1] range
    generated_infrared_resized = generated_infrared_resized.clip(0, 1)

    # Calculate PSNR between original and generated infrared images
    psnr_value = psnr(infrared_original_denorm, generated_infrared_resized, data_range=1.0)

    # Calculate SSIM between original and generated infrared images
    ssim_value = ssim(infrared_original_denorm, generated_infrared_resized, data_range=1.0, multichannel=True, win_size=3, gaussian_weights=True, sigma=1.5)

    # Accumulate metrics
    total_psnr += psnr_value
    total_ssim += ssim_value
    num_images += 1

    # Print metrics for the current image
    print(f"[{idx}] Image: {visible_image_name} | PSNR: {psnr_value:.2f} | SSIM: {ssim_value:.4f}")

# ===============================
# Final Metrics Calculation
# ===============================

# Calculate and print average metrics
if num_images > 0:
    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images

    print(f"\nProcessed {num_images} images.")
    print(f"Average PSNR: {avg_psnr:.2f}")
    print(f"Average SSIM: {avg_ssim:.4f}")

    # Save metrics to a text file
    with open("metrics.txt", "a") as f:
        f.write(f"Processed {num_images} images.\n")
        f.write(f"Average PSNR: {avg_psnr:.2f}\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
else:
    print("No images were processed.")
