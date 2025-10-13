"""
    Gradio demo for text customization with Calligrapher (the reference AND MASK are uploaded by the user).
    This demo is useful for reproduction.

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
from utils import get_bbox_from_mask, crop_image_from_bb, \
                  resize_img_and_pad, generate_context_reference_image

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
    transformer = FluxTransformer2DModel.from_pretrained(base_model_path, subfolder="transformer",
                                                         torch_dtype=torch.bfloat16)
    pipe = CalligrapherPipeline.from_pretrained(base_model_path, transformer=transformer,
                                                torch_dtype=torch.bfloat16).to("cuda")
    model = Calligrapher(pipe, image_encoder_path, calligrapher_path, device="cuda", num_tokens=128)
    return model


# Init models.
model = load_models()
print('Model loaded!')


def process_and_generate(source_image, mask_image, reference_image, prompt, height, width,
                         scale, steps=50, seed=42, use_context=True, num_images=1):
    print('Begin processing!')
    # Job directory.
    job_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    job_dir = os.path.join(SAVE_DIR, job_name)
    os.makedirs(job_dir, exist_ok=True)

    # Get source, mask, and cropped images from gr.ImageEditor.
    source_image.save(os.path.join(job_dir, 'source_image.png'))
    mask_image.save(os.path.join(job_dir, 'mask_image.png'))

    # Resize source and mask.
    source_image = source_image.resize((width, height))
    mask_image = mask_image.resize((width, height), Image.NEAREST)
    mask_np = np.array(mask_image)
    mask_np[mask_np > 0] = 255
    mask_image = Image.fromarray(mask_np.astype(np.uint8))

    if reference_image is None:
        # If self-inpaint (no input ref): (1) get bounding box from the mask and (2) perform cropping to get ref image.
        tl, br = get_bbox_from_mask(mask_image)
        # Convert irregularly shaped masks into rectangles.
        reference_image = crop_image_from_bb(source_image, tl, br)
    # Raw reference image before resizing.
    reference_image.save(os.path.join(job_dir, 'reference_image_raw.png'))
    reference_image_to_encoder = resize_img_and_pad(reference_image, target_size=(512, 512))  # 512 considering SigLip
    reference_image_to_encoder.save(os.path.join(job_dir, 'reference_to_encoder.png'))
    reference_context = generate_context_reference_image(reference_image, width)

    if use_context:
        # Concat the context on the top of the input masked image in the pixel space.
        source_with_context = Image.new(source_image.mode, (width, reference_context.size[1] + height))
        source_with_context.paste(reference_context, (0, 0))
        source_with_context.paste(source_image, (0, reference_context.size[1]))
        source_with_context.save(os.path.join(job_dir, 'source_with_context.png'))
        # Concat the zero mask on the top of the mask image.
        mask_with_context = Image.new(mask_image.mode,
                                      (mask_image.size[0], reference_context.size[1] + mask_image.size[0]), color=0)
        mask_with_context.paste(mask_image, (0, reference_context.size[1]))

        source_image = source_with_context
        mask_image = mask_with_context

    all_generated_images = []
    for i in range(num_images):
        res = model.generate(
            image=source_image,
            mask_image=mask_image,
            ref_image=reference_image_to_encoder,
            prompt=prompt,
            scale=scale,
            num_inference_steps=steps,
            width=source_image.size[0],
            height=source_image.size[1],
            seed=seed + i,
        )[0]
        if use_context:
            res_vis = res.crop((0, reference_context.size[1], res.width, res.height))  # remove context
            mask_vis = mask_image.crop(
                (0, reference_context.size[1], mask_image.width, mask_image.height))  # remove context mask
        else:
            res_vis = res
            mask_vis = mask_image
        res_vis.save(os.path.join(job_dir, f'result_{i}.png'))
        all_generated_images.append((res_vis, f"Generating {i + 1} (Seed: {seed + i})"))

    return mask_vis, reference_image_to_encoder, all_generated_images


# Construct example data.
sample_data = [
    {
        "source": Image.open("samples/test11_source.png"),
        "mask": Image.open("samples/test11_mask.png"),
        "reference": Image.open("samples/test11_ref.png"),
        "prompt": "The text is 'Rose'."
    },
    {
        "source": Image.open("samples/test17_source.png"),
        "mask": Image.open("samples/test17_mask.png"),
        "reference": Image.open("samples/rainbow.jpg"),
        "prompt": "The text is 'Rainbow'."
    },
    {
        "source": Image.open("samples/test17_source.png"),
        "mask": Image.open("samples/test17_mask.png"),
        "reference": Image.open("samples/fire.jpg"),
        "prompt": "The text is 'Fire!'."
    }
]

examples = [
    [
        sample["source"],
        sample["mask"],
        sample["reference"],
        sample["prompt"],
        512,  # height
        512,  # width
        1.0,  # scale
        10,  # steps
        2025,  # seed
        True,  # use_context
        2  # num_images
    ]
    for sample in sample_data
]

# Main gradio codes.
with gr.Blocks(theme="default", css=".image-editor img {max-width: 70%; height: 70%;}") as demo:
    gr.Markdown(
        """
        # 🖌️ Calligrapher: Freestyle Text Image Customization
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### 🎨 Image Editing Panel")
            source_image = gr.Image(
                label="Source Image",
                sources=["upload"],
                type="pil",
                value=Image.open(f"samples/test50_source.png"),
            )

            gr.Markdown("### 📤 Output Result")
            gallery = gr.Gallery(label="🖼️ Result Gallery")
            gr.Markdown(
                """<br>

                 ### ✨User Tips:

                 1. **Speed vs Quality Trade-off.** Use fewer steps (e.g., 10-step which takes ~4s/image on a single A6000 GPU) for faster generation, but quality may be lower.

                 2. **Inpaint Position Freedom.**  Inpainting positions are flexible - they don't necessarily need to match the original text locations in the input image.

                 3. **Iterative Editing.** Drag outputs from the gallery to the Image Editing Panel (clean the Editing Panel first) for quick refinements.

                 4. **Mask Optimization.** Adjust mask size/aspect ratio to match your desired content. The model tends to fill the masks, and harmonizes the generation with background in terms of color and lighting.

                 5. **Reference Image Tip.**  White-background references improve style consistency - the encoder also considers background context of the given reference image.

                 6. **Resolution Balance.** Very high-resolution generation sometimes triggers spelling errors. 512/768px are recommended considering the model is trained under the resolution of 512.
                """
            )
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️Settings")
            mask_image = gr.Image(
                label="🧩 Mask Image",
                sources=["upload"],
                type="pil",
                value=Image.open(f"samples/test50_mask.png"),
            )
            reference_image = gr.Image(
                label="🧩 Reference Image",
                sources=["upload"],
                type="pil",
                value=Image.open(f"samples/test50_ref.png")
            )
            prompt = gr.Textbox(
                label="📝 Prompt",
                placeholder="The text is 'Image'...",
                value="The text is 'Balloon'."
            )

            with gr.Accordion("🔧 Additional Settings", open=True):
                with gr.Row():
                    height = gr.Number(label="Height", value=512, precision=0)
                    width = gr.Number(label="Width", value=512, precision=0)
                scale = gr.Slider(0.0, 2.0, 1.0, step=0.1, value=1.0, label="🎚️ Strength")
                steps = gr.Slider(1, 100, 50, step=1, label="🔁 Steps")
                with gr.Row():
                    seed = gr.Number(label="🎲 Seed", value=56, precision=0)
                    use_context = gr.Checkbox(value=True, label="🔍 Use Context", interactive=True)
                num_images = gr.Slider(1, 16, 2, step=1, label="🖼️ Sample Amount")

            run_btn = gr.Button("🚀 Run", variant="primary")

            mask_output = gr.Image(label="🟩 Mask Demo")
            reference_demo = gr.Image(label="🧩 Reference Demo")

    # Run button event.
    run_btn.click(
        fn=process_and_generate,
        inputs=[
            source_image,
            mask_image,
            reference_image,
            prompt,
            height,
            width,
            scale,
            steps,
            seed,
            use_context,
            num_images
        ],
        outputs=[
            mask_output,
            reference_demo,
            gallery
        ]
    )

    gr.Examples(
        examples=examples,
        inputs=[
            source_image,
            mask_image,
            reference_image,
            prompt,
            height,
            width,
            scale,
            steps,
            seed,
            use_context,
            num_images
        ],
        outputs=[
            mask_output,
            reference_demo,
            gallery
        ],
        fn=process_and_generate,
        label="✨ Example Inputs: Click any example below to load it.",
        examples_per_page=3
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=1234, share=False)
