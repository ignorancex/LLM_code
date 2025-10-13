"""
    Gradio demo for text customization with Calligrapher (the reference is uploaded by the user),
    which supports multilingual text image customization.
    Acknowledgement: Supported by TextFLUX: https://github.com/yyyyyxie/textflux.
"""

import os
import json
import gradio as gr
import numpy as np
from datetime import datetime
import torch
from PIL import Image

from pipeline_calligrapher import CalligrapherPipeline
from models.calligrapher import Calligrapher
from models.transformer_flux_inpainting import FluxTransformer2DModel
from utils import process_gradio_source, get_bbox_from_mask, crop_image_from_bb, resize_img_and_pad
from utils_multilingual import run_multilingual_inference

# Global settings.
with open(os.path.join(os.path.dirname(__file__), 'path_dict.json'), 'r') as f:
    path_dict = json.load(f)
SAVE_DIR = path_dict['gradio_save_dir']
os.environ["GRADIO_TEMP_DIR"] = path_dict['gradio_temp_dir']
os.environ['TMPDIR'] = path_dict['gradio_temp_dir']


# Function of loading pre-trained models.
def load_models():
    base_model_path = path_dict['base_model_path']
    image_encoder_path = path_dict['image_encoder_path']
    calligrapher_path = path_dict['calligrapher_path']
    textflux_path = path_dict['textflux_path']
    transformer = FluxTransformer2DModel.from_pretrained(base_model_path, subfolder="transformer",
                                                         torch_dtype=torch.bfloat16)
    # Load textflux lora weights.
    state_dict, network_alphas = CalligrapherPipeline.lora_state_dict(
        pretrained_model_name_or_path_or_dict=textflux_path,
        return_alphas=True
    )
    is_correct_format = all("lora" in key or "dora_scale" in key for key in state_dict.keys())
    if not is_correct_format:
        raise ValueError("Invalid LoRA checkpoint!")
    CalligrapherPipeline.load_lora_into_transformer(
        state_dict=state_dict,
        network_alphas=network_alphas,
        transformer=transformer,
    )
    pipe = CalligrapherPipeline.from_pretrained(base_model_path, transformer=transformer,
                                                torch_dtype=torch.bfloat16).to("cuda")
    model = Calligrapher(pipe, image_encoder_path, calligrapher_path, device="cuda", num_tokens=128)
    return model


# Init models.
model = load_models()
print('Model loaded!')


def process_and_generate(editor_component, reference_image, prompt, height, width,
                         scale, steps=50, seed=42, num_images=1):
    print('Begin processing!')
    # Job directory.
    job_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    job_dir = os.path.join(SAVE_DIR, job_name)
    os.makedirs(job_dir, exist_ok=True)

    # Get source, mask, and cropped images from gr.ImageEditor.
    source_image, mask_image, cropped_image = process_gradio_source(editor_component)
    source_image.save(os.path.join(job_dir, 'source_image.png'))
    mask_image.save(os.path.join(job_dir, 'mask_image.png'))
    cropped_image.save(os.path.join(job_dir, 'cropped_image.png'))

    # Resize source and mask.
    source_image = source_image.resize((width, height))
    mask_image = mask_image.resize((width, height), Image.NEAREST)
    mask_np = np.array(mask_image)
    mask_np[mask_np > 0] = 255
    mask_image = Image.fromarray(mask_np.astype(np.uint8))

    if reference_image is None:
        # If self-inpaint (no input ref): (1) get bounding box from the mask and (2) perform cropping to get the ref image.
        tl, br = get_bbox_from_mask(mask_image)
        # Convert irregularly shaped masks into rectangles.
        reference_image = crop_image_from_bb(source_image, tl, br)
    # Raw reference image before resizing.
    reference_image.save(os.path.join(job_dir, 'reference_image_raw.png'))
    reference_image_to_encoder = resize_img_and_pad(reference_image, target_size=(512, 512))
    reference_image_to_encoder.save(os.path.join(job_dir, 'reference_to_encoder.png'))

    all_generated_images = run_multilingual_inference(model, source_image, mask_image, reference_image_to_encoder,
                                                      prompt, num_steps=steps, seed=seed, num_images=num_images)
    vis_all_generated_images = []
    for i in range(len(all_generated_images)):
        res = all_generated_images[i]
        res_vis = res.crop((source_image.width, 0, res.width, res.height))
        mask_vis = mask_image
        res_vis.save(os.path.join(job_dir, f'result_{i}.png'))
        vis_all_generated_images.append((res_vis, f"Generated #{i + 1} (Seed: {seed + i})"))
    return mask_vis, reference_image_to_encoder, vis_all_generated_images


# Main gradio codes.
with gr.Blocks(theme="default", css=".image-editor img {max-width: 70%; height: 70%;}") as demo:
    gr.Markdown(
        """
        # 🖌️ Calligrapher: Freestyle Text Image Customization (Multilingual)
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### 🎨 Image Editing Panel")
            editor_component = gr.ImageEditor(
                label="Upload or Draw",
                type="pil",
                brush=gr.Brush(colors=["#FFFFFF"], default_size=30, color_mode="fixed"),
                layers=True,
                interactive=True,
            )

            gr.Markdown("### 📤 Output Result")
            gallery = gr.Gallery(label="🖼️ Result Gallery")
            gr.Markdown(
                """<br>
                
                 ### ✨User Tips:
                 1. **Quality of multilingual generation.** This implementation strategy combines Calligrapher with the fine-tuned base model (textflux) without additional fine-tuning, please temper expectations regarding output quality.
                 
                 2. **Speed vs Quality Trade-off.** Use fewer steps (e.g., 10-step which takes ~4s/image on a single A6000 GPU) for faster generation, but quality may be lower.
                
                 3. **Inpaint Position Freedom.**  Inpainting positions are flexible - they don't necessarily need to match the original text locations in the input image.
                 
                 4. **Iterative Editing.** Drag outputs from the gallery to the Image Editing Panel (clean the Editing Panel first) for quick refinements.
                   
                 5. **Mask Optimization.** Adjust mask size/aspect ratio to match your desired content. The model tends to fill the masks, and harmonizes the generation with background in terms of color and lighting.
                
                 6. **Reference Image Tip.**  White-background references improve style consistency - the encoder also considers background context of the given reference image.
                
                 7. **Resolution Balance.** Very high-resolution generation sometimes triggers spelling errors. 512/768px are recommended considering the model is trained under the resolution of 512.
                """
            )
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️Settings")
            reference_image = gr.Image(
                label="🧩 Reference Image  (skip this if self-reference)",
                sources=["upload"],
                type="pil",
            )
            prompt = gr.Textbox(
                label="📝 Prompt",
                placeholder="你好",
                value="你好"
            )

            with gr.Accordion("🔧 Additional Settings", open=True):
                with gr.Row():
                    height = gr.Number(label="Height", value=512, precision=0)
                    width = gr.Number(label="Width", value=512, precision=0)
                scale = gr.Slider(0.0, 2.0, 1.0, step=0.1, value=1.0, label="🎚️ Strength")
                steps = gr.Slider(1, 100, 30, step=1, label="🔁 Steps")
                with gr.Row():
                    seed = gr.Number(label="🎲 Seed", value=56, precision=0)
                num_images = gr.Slider(1, 16, 2, step=1, label="🖼️ Sample Amount")

            run_btn = gr.Button("🚀 Run", variant="primary")

            mask_output = gr.Image(label="🟩 Mask Demo")
            reference_demo = gr.Image(label="🧩 Reference Demo")

    # Run button event.
    run_btn.click(
        fn=process_and_generate,
        inputs=[
            editor_component,
            reference_image,
            prompt,
            height,
            width,
            scale,
            steps,
            seed,
            num_images
        ],
        outputs=[
            mask_output,
            reference_demo,
            gallery
        ]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=1234, share=False)
