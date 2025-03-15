import os
import torch
from PIL import Image
import torchvision.transforms as T
from improved_consistency_model_conditional import ConsistencySamplingAndEditing
from llvip.script import UNet
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# Load the pretrained UNet model
model_path = "checkpoints/llvip512x512_128x128/"
model = UNet.from_pretrained(model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device).eval()

# Define image transformation: Resize to 256x256 and normalize to [-1, 1]
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Lambda(lambda x: (x * 2) - 1),  # Normalize pixel values
])

# Input data directories for visible and infrared test sets
visible_folder = "../datasets/LLVIP/visible/test/"
infrared_folder = "../datasets/LLVIP/infrared/test/"

# Create the consistency sampling instance for generating images
consistency_sampling = ConsistencySamplingAndEditing()

# Define the noise sigma schedule
sigmas = [80.0, 40.0, 20.0, 10.0, 5.0, 2.5, 1.25, 0.625, 0.3125, 0.15625, 0.078125]

# Initialize accumulators for PSNR and SSIM metrics
total_psnr = 0.0
total_ssim = 0.0
num_images = 0

# Directory to save generated results
results_folder = "results_llvip_128x128"
os.makedirs(results_folder, exist_ok=True)

# Loop through all visible images in the test set
for idx, visible_image_name in enumerate(os.listdir(visible_folder), start=1):
    visible_image_path = os.path.join(visible_folder, visible_image_name)
    infrared_image_path = os.path.join(infrared_folder, visible_image_name)

    if not os.path.exists(infrared_image_path):
        print(f"Infrared image {infrared_image_path} not found, skipping.")
        continue

    try:
        # Load visible and infrared images
        visible_image = Image.open(visible_image_path).convert("RGB")
        infrared_image = Image.open(infrared_image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading images: {e}, skipping {visible_image_name}")
        continue

    # Apply transformations to resize and normalize images
    visible_tensor = transform(visible_image).unsqueeze(0).to(device)
    infrared_tensor = transform(infrared_image).unsqueeze(0).to(device)

    # Add noise to the infrared image
    noise = torch.randn_like(infrared_tensor) * sigmas[0]
    noisy_infrared_tensor = infrared_tensor + noise

    try:
        # Generate the infrared image using consistency sampling
        with torch.no_grad():
            generated_infrared_tensor = consistency_sampling(
                model=model,
                y=noisy_infrared_tensor,
                v=visible_tensor,
                sigmas=sigmas,
                start_from_y=True,
                add_initial_noise=False,
                clip_denoised=True,
                verbose=False,
            )
    except Exception as e:
        print(f"Error during model inference: {e}, skipping {visible_image_name}")
        continue

    # Denormalize tensors to convert to valid image range [0, 1]
    visible_denorm = ((visible_tensor.squeeze(0).cpu() + 1) / 2).clamp(0, 1).numpy().transpose(1, 2, 0)
    infrared_denorm = ((infrared_tensor.squeeze(0).cpu() + 1) / 2).clamp(0, 1).numpy().transpose(1, 2, 0)
    generated_infrared_denorm = ((generated_infrared_tensor.squeeze(0).cpu() + 1) / 2).clamp(0, 1).numpy().transpose(1, 2, 0)

    # Convert to uint8 format for saving
    visible_image_save = (visible_denorm * 255).astype(np.uint8)
    infrared_image_save = (infrared_denorm * 255).astype(np.uint8)
    generated_image_save = (generated_infrared_denorm * 255).astype(np.uint8)

    # Save generated and ground truth images for reference
    base_name, ext = os.path.splitext(visible_image_name)
    Image.fromarray(generated_image_save).save(os.path.join(results_folder, f"generated_{base_name}{ext}"))
    Image.fromarray(infrared_image_save).save(os.path.join(results_folder, f"groundtruth_{base_name}{ext}"))
    Image.fromarray(visible_image_save).save(os.path.join(results_folder, f"visible_{base_name}{ext}"))

    # Calculate PSNR and SSIM metrics
    psnr_value = psnr(infrared_denorm, generated_infrared_denorm, data_range=1.0)
    ssim_value = ssim(infrared_denorm, generated_infrared_denorm, data_range=1.0, multichannel=True, win_size=3)

    # Accumulate metrics
    total_psnr += psnr_value
    total_ssim += ssim_value
    num_images += 1

    print(f"Image {idx}: PSNR = {psnr_value:.2f}, SSIM = {ssim_value:.4f}")

# Compute and save average metrics
if num_images > 0:
    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images

    print(f"Average PSNR: {avg_psnr:.2f}")
    print(f"Average SSIM: {avg_ssim:.4f}")

    with open("metrics.txt", "w") as f:
        f.write(f"Average PSNR: {avg_psnr:.2f}\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
else:
    print("No images processed.")
