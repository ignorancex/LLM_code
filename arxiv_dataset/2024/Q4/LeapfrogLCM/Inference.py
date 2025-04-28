from diffusers import UNet2DConditionModel, DiffusionPipeline, LCMScheduler
import torch
import os

# Initialize the UNet model
unet = UNet2DConditionModel.from_pretrained("path/to/fine-tuned/weight", torch_dtype=torch.float16, variant="fp16")

# Initialize the diffusion pipeline
pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", unet=unet, torch_dtype=torch.float16, variant="fp16")

# Set the loaded scheduler in the pipeline
pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
pipeline.to("cuda")

# Define the prompt
prompt = " your prompt goes here "

# Set the number of inference steps
inference_steps = 4

# Output directory
output_dir = "path/to/generated/image"

# Generate and save a single image for the prompt with 4 inference steps directly to the output directory
image = pipeline(prompt, num_inference_steps=inference_steps, guidance_scale=2).images[0]
image = image.resize((512, 512))
image_path = os.path.join(output_dir, "image.jpg")
image.save(image_path)
