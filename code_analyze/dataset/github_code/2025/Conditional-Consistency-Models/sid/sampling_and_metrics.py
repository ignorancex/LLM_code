import torch
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from improved_consistency_model_conditional import ConsistencySamplingAndEditing
from sid.script import UNet  # Replace 'script_name' with the actual script where UNet is defined
import os
import random
import rawpy
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# Function to load raw images
def load_raw_image(path: str) -> torch.Tensor:
    with rawpy.imread(path) as raw:
        # Postprocess to get RGB image
        rgb_image = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        # Convert to float and normalize
        rgb_image = np.float32(rgb_image) / (2**16 - 1)
        return torch.from_numpy(rgb_image).permute(2, 0, 1)  # Shape: (C, H, W)

# Function to denormalize tensors
def denormalize(tensor):
    return (tensor + 1) / 2  # Convert from [-1,1] to [0,1]

# Define the transform
def transform(image):
    return (image * 2) - 1  # Normalize to [-1, 1]

# Paths
data_dir = "../datasets/sid"
txt_file = os.path.join(data_dir, "Sony_test_list.txt")  # Use 'Sony_test_list.txt' if available
output_folder = "results_sid"

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load the list of image pairs
image_pairs = []
with open(txt_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        short_path, long_path, iso, f_number = line.strip().split()
        # Extract exposure times from file names
        short_exposure = float(os.path.basename(short_path).split('_')[-1].replace('s.ARW', ''))
        long_exposure = float(os.path.basename(long_path).split('_')[-1].replace('s.ARW', ''))
        ratio = long_exposure / short_exposure
        image_pairs.append((short_path, long_path, ratio))

# Number of images to process
# num_images = 10  # Remove or comment out to process all images

# Uncomment the following line to process all images
num_images = len(image_pairs)

# Optionally, limit the number of images (for testing purposes)
# num_images = min(num_images, len(image_pairs))

# Randomly select images
random.seed(40)  # Optional: Set seed for reproducibility
selected_images = random.sample(image_pairs, num_images)

# Load the model
model_path = "checkpoints/sid"  # Replace with your actual model checkpoint path
model = UNet.from_pretrained(model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device).eval()

# Create the sampling instance
consistency_sampling = ConsistencySamplingAndEditing()

# Define the sigma schedule
sigmas = [80.0, 40.0, 20.0, 10.0, 5.0, 2.5, 1.25, 0.625, 0.3125, 0.15625, 0.078125]

# Initialize variables to accumulate PSNR and SSIM
total_psnr = 0.0
total_ssim = 0.0
processed_count = 0  # To handle any skipped images

for idx, (short_path, long_path, ratio) in enumerate(selected_images, start=1):
    short_image_path = os.path.join(data_dir, short_path)
    long_image_path = os.path.join(data_dir, long_path)
    
    # Load images
    try:
        short_image = load_raw_image(short_image_path)
        long_image = load_raw_image(long_image_path)
    except Exception as e:
        print(f"Error loading images: {e}, skipping {short_path}")
        continue

    # Multiply short_image by ratio and clip
    short_image = torch.clamp(short_image * ratio, 0.0, 1.0)
    
    # Apply center crop and resize
    # crop_size = (512, 512)
    resize_size = (512, 512)
    
    # # Synchronized center crop
    # center_crop = T.CenterCrop(crop_size)
    # short_image = center_crop(short_image)
    # long_image = center_crop(long_image)
    
    # Resize to desired size
    short_image = TF.resize(short_image, resize_size)
    long_image = TF.resize(long_image, resize_size)
    
    # Normalize to [-1,1]
    short_image = transform(short_image)
    long_image = transform(long_image)
    
    # Move tensors to device
    short_tensor = short_image.unsqueeze(0).to(device)
    long_tensor = long_image.unsqueeze(0).to(device)
    
    # Add noise to the long image
    max_sigma = sigmas[0]  # Highest sigma value
    noise = torch.randn_like(long_tensor) * max_sigma
    noisy_long_tensor = noise  # Start from pure noise
    
    # Generate the long image starting from the noisy long image
    try:
        with torch.no_grad():
            generated_long_tensor = consistency_sampling(
                model=model,
                y=noisy_long_tensor,
                v=short_tensor,
                sigmas=sigmas,
                start_from_y=True,
                add_initial_noise=False,
                clip_denoised=True,
                verbose=False,
            )
    except Exception as e:
        print(f"Error during model inference: {e}, skipping {short_path}")
        continue
    
    # Denormalize tensors
    short_denorm = denormalize(short_tensor.squeeze(0).cpu())
    long_denorm = denormalize(long_tensor.squeeze(0).cpu())
    generated_long_denorm = denormalize(generated_long_tensor.squeeze(0).cpu())
    
    # Convert tensors to PIL images
    short_image_pil = TF.to_pil_image(short_denorm)
    long_image_pil = TF.to_pil_image(long_denorm)
    generated_long_image_pil = TF.to_pil_image(generated_long_denorm)
    
    # Combine images side by side (optional, can be skipped if not needed)
    combined_width = short_image_pil.width * 3
    combined_height = short_image_pil.height
    combined_image = Image.new('RGB', (combined_width, combined_height))
    combined_image.paste(short_image_pil, (0, 0))
    combined_image.paste(long_image_pil, (short_image_pil.width, 0))
    combined_image.paste(generated_long_image_pil, (short_image_pil.width * 2, 0))
    
    # Save the combined image
    output_filename = f'comparison_sid_{idx}.png'
    output_path = os.path.join(output_folder, output_filename)
    combined_image.save(output_path)
    
    # Convert denormalized tensors to NumPy arrays for metric computation
    # Ensure the arrays are in the range [0, 1]
    long_np = long_denorm.permute(1, 2, 0).numpy()  # Shape: (H, W, C)
    generated_long_np = generated_long_denorm.permute(1, 2, 0).numpy()
    
    # Compute PSNR
    current_psnr = psnr(long_np, generated_long_np, data_range=1.0)
    
    # Compute SSIM
    # Convert RGB to grayscale for SSIM or compute multi-channel SSIM
    # Here, we'll compute multi-channel SSIM
    current_ssim = ssim(long_np, generated_long_np, data_range=1.0, multichannel=True, win_size=3)
    
    # Accumulate the metrics
    total_psnr += current_psnr
    total_ssim += current_ssim
    processed_count += 1
    
    # Optional: Print metrics for each image
    print(f"Image {idx}: PSNR = {current_psnr:.2f} dB, SSIM = {current_ssim:.4f}")

# Calculate average PSNR and SSIM
if processed_count > 0:
    average_psnr = total_psnr / processed_count
    average_ssim = total_ssim / processed_count
    print(f"\nProcessed {processed_count} images.")
    print(f"Average PSNR: {average_psnr:.2f} dB")
    print(f"Average SSIM: {average_ssim:.4f}")
else:
    print("No images were processed successfully.")

# Optional: Save the average metrics to a text file
metrics_output_path = os.path.join(output_folder, "metrics.txt")
with open(metrics_output_path, 'w') as f:
    f.write(f"Processed {processed_count} images.\n")
    f.write(f"Average PSNR: {average_psnr:.2f} dB\n")
    f.write(f"Average SSIM: {average_ssim:.4f}\n")
print(f"Metrics saved to {metrics_output_path}")
