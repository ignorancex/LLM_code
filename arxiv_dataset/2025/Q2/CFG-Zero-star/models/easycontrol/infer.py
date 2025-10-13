import spaces
import os
import json
import time
import torch
from PIL import Image
from tqdm import tqdm
import gradio as gr

from safetensors.torch import save_file
from src.pipeline import FluxPipeline
from src.transformer_flux import FluxTransformer2DModel
from src.lora_helper import set_single_lora, set_multi_lora, unset_lora

from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="Xiaojiu-Z/EasyControl", filename="models/Ghibli.safetensors", local_dir="./checkpoints/models/")

# Initialize the image processor
base_path = "black-forest-labs/FLUX.1-dev"    
lora_base_path = "checkpoints/models/models"

pipe = FluxPipeline.from_pretrained(base_path, torch_dtype=torch.bfloat16)
transformer = FluxTransformer2DModel.from_pretrained(base_path, subfolder="transformer", torch_dtype=torch.bfloat16)
pipe.transformer = transformer
pipe.to("cuda")

def clear_cache(transformer):
    for name, attn_processor in transformer.attn_processors.items():
        attn_processor.bank_kv.clear()

# Define the Gradio interface
@spaces.GPU()
def dual_condition_generate_image(prompt, spatial_img, height, width, seed, control_type, zero_steps):
    # Set the control type
    if control_type == "Ghibli":
        lora_path = os.path.join(lora_base_path, "Ghibli.safetensors")
    set_single_lora(pipe.transformer, lora_path, lora_weights=[1], cond_size=512)

    # Process the image
    spatial_imgs = [spatial_img] if spatial_img else []

    # Image with use_zero_init=True
    image_true = pipe(
        prompt,
        height=int(height),
        width=int(width),
        guidance_scale=3.5,
        num_inference_steps=25,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(seed), 
        subject_images=[],
        spatial_images=spatial_imgs,
        cond_size=512,
        use_zero_init=True,
        zero_steps=int(zero_steps)
    ).images[0]
    clear_cache(pipe.transformer)

    # Image with use_zero_init=False
    image_false = pipe(
        prompt,
        height=int(height),
        width=int(width),
        guidance_scale=3.5,
        num_inference_steps=25,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(seed), 
        subject_images=[],
        spatial_images=spatial_imgs,
        cond_size=512,
        use_zero_init=False
    ).images[0]
    clear_cache(pipe.transformer)

    return image_true, image_false

# Define the Gradio interface components
control_types = ["Ghibli"]

# Create the Gradio Blocks interface
with gr.Blocks() as demo:
    gr.Markdown("# Ghibli Studio Control Image Generation with EasyControl")
    gr.Markdown("The model is trained on **only 100 real Asian faces** paired with **GPT-4o-generated Ghibli-style counterparts**, and it preserves facial features while applying the iconic anime aesthetic.")
    gr.Markdown("Generate images using EasyControl with Ghibli control LoRAs.（Due to hardware constraints, only low-resolution images can be generated. For high-resolution (1024+), please set up your own environment.）")

    gr.Markdown("**[Attention!!]**：The recommended prompts for using Ghibli Control LoRA should include the trigger words: Ghibli Studio style, Charming hand-drawn anime-style illustration")
    gr.Markdown("😊😊If you like this demo, please give us a star (github: [EasyControl](https://github.com/Xiaojiu-z/EasyControl))")

    with gr.Tab("Ghibli Condition Generation"):
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", value="Ghibli Studio style, Charming hand-drawn anime-style illustration")
                spatial_img = gr.Image(label="Ghibli Image", type="pil")
                height = gr.Slider(minimum=256, maximum=1024, step=64, label="Height", value=768)
                width = gr.Slider(minimum=256, maximum=1024, step=64, label="Width", value=768)
                seed = gr.Number(label="Seed", value=42)
                zero_steps = gr.Number(label="Zero Init Steps", value=1)
                control_type = gr.Dropdown(choices=control_types, label="Control Type")
                single_generate_btn = gr.Button("Generate Image")
            with gr.Column():
                image_with_zero_init = gr.Image(label="Image CFG-Zero*")
                image_without_zero_init = gr.Image(label="Image CFG")

    # Link the buttons to the functions
    single_generate_btn.click(
        dual_condition_generate_image,
        inputs=[prompt, spatial_img, height, width, seed, control_type, zero_steps],
        outputs=[image_with_zero_init, image_without_zero_init]
    )

# Launch the Gradio app
demo.queue().launch()

