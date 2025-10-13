'''
python scripts/gradio_demo.py 
'''

import sys
import os
# workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../icedit"))

# if workspace_dir not in sys.path:
#     sys.path.insert(0, workspace_dir)
    
from diffusers import FluxFillPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
import gradio as gr
import numpy as np
import torch
import spaces
import argparse
import random 
from diffusers import FluxFillPipeline
from PIL import Image

from transformers import T5EncoderModel

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

current_lora_scale = 1.0

parser = argparse.ArgumentParser() 
parser.add_argument("--server_name", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=7860, help="Port for the Gradio app")
parser.add_argument("--share", action="store_true")
parser.add_argument("--output-dir", type=str, default="gradio_results", help="Directory to save the output image")
parser.add_argument("--flux-path", type=str, default='black-forest-labs/flux.1-fill-dev', help="Path to the model")
parser.add_argument("--lora-path", type=str, default='RiverZ/normal-lora', help="Path to the LoRA weights")
parser.add_argument("--transformer", type=str, default=None, help="The gguf model of FluxTransformer2DModel")
parser.add_argument("--text_encoder_2", type=str, default=None, help="The gguf model of T5EncoderModel")
parser.add_argument("--enable-model-cpu-offload", action="store_true", help="Enable CPU offloading for the model")
args = parser.parse_args()

if args.transformer:
    args.transformer = os.path.abspath(args.transformer)
    transformer = FluxTransformer2DModel.from_single_file(
        args.transformer,
        config="scripts/config.json",
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16,
    )
else:
    transformer = FluxTransformer2DModel.from_pretrained(
        args.flux_path, 
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )

if args.text_encoder_2:
    args.text_encoder_2 = os.path.abspath(args.text_encoder_2)
    text_encoder_2 = T5EncoderModel.from_pretrained(
        args.flux_path,
        subfolder="text_encoder_2",
        gguf_file=f"{args.text_encoder_2}",
        torch_dtype=torch.bfloat16,
    )
else:
    text_encoder_2 = T5EncoderModel.from_pretrained(
        args.flux_path,
        subfolder="text_encoder_2",
        torch_dtype=torch.bfloat16,
    )

pipe = FluxFillPipeline.from_pretrained(
    args.flux_path, 
    transformer=transformer, 
    text_encoder_2=text_encoder_2, 
    torch_dtype=torch.bfloat16
)
pipe.load_lora_weights(args.lora_path, adapter_name="icedit")
pipe.set_adapters("icedit", 1.0)

if args.enable_model_cpu_offload:
    pipe.enable_model_cpu_offload()
else:
    pipe = pipe.to("cuda")


@spaces.GPU
def infer(edit_images,
          prompt,
          seed=666,
          randomize_seed=False,
          width=1024,
          height=1024,
          guidance_scale=50,
          num_inference_steps=28,
          lora_scale=1.0,
          progress=gr.Progress(track_tqdm=True)):

    global current_lora_scale
    
    if lora_scale != current_lora_scale:
        print(f"\033[93m[INFO] LoRA scale changed from {current_lora_scale} to {lora_scale}, reloading LoRA weights\033[0m")
        pipe.set_adapters("icedit", lora_scale)
        current_lora_scale = lora_scale

    image = edit_images

    if image.size[0] != 512:
        print("\033[93m[WARNING] We can only deal with the case where the image's width is 512.\033[0m")
        new_width = 512
        scale = new_width / image.size[0]
        new_height = int(image.size[1] * scale)
        new_height = (new_height // 8) * 8
        image = image.resize((new_width, new_height))
        print(f"\033[93m[WARNING] Resizing the image to {new_width} x {new_height}\033[0m")

    image = image.convert("RGB")
    width, height = image.size
    image = image.resize((512, int(512 * height / width)))
    combined_image = Image.new("RGB", (width * 2, height))
    combined_image.paste(image, (0, 0))
    mask_array = np.zeros((height, width * 2), dtype=np.uint8)
    mask_array[:, width:] = 255
    mask = Image.fromarray(mask_array)
    instruction = f'A diptych with two side-by-side images of the same scene. On the right, the scene is exactly the same as on the left but {prompt}'

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    image = pipe(
        prompt=instruction,
        image=combined_image,
        mask_image=mask,
        height=height,
        width=width * 2,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator().manual_seed(seed),
    ).images[0]

    w, h = image.size
    image = image.crop((w // 2, 0, w, h))

    os.makedirs(args.output_dir, exist_ok=True)

    index = len(os.listdir(args.output_dir))
    image.save(f"{args.output_dir}/result_{index}.png")

    return image, seed

original_examples = [
    "a tiny astronaut hatching from an egg on the moon",
    "a cat holding a sign that says hello world",
    "an anime illustration of a wiener schnitzel",
]

new_examples = [
    ['assets/girl.png', 'Make her hair dark green and her clothes checked.', 304897401],
    ['assets/boy.png', 'Change the sunglasses to a Christmas hat.', 748891420],
    ['assets/kaori.jpg', 'Make it a sketch.', 484817364]
]

css = """
#col-container {
    margin: 0 auto;
    max-width: 1000px;
}
"""

with gr.Blocks(css=css) as demo:

    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"""# IC-Edit
A demo for [IC-Edit](https://arxiv.org/pdf/2504.20690).
More **open-source**, with **lower costs**, **faster speed** (it takes about 9 seconds to process one image), and **powerful performance**.
For more details, check out our [Github Repository](https://github.com/River-Zhang/ICEdit) and [website](https://river-zhang.github.io/ICEdit-gh-pages/). If our project resonates with you or proves useful, we'd be truly grateful if you could spare a moment to give it a star.
""")
        with gr.Row():
            with gr.Column():
                edit_image = gr.Image(
                    label='Upload image for editing',
                    type='pil',
                    sources=["upload", "webcam"],
                    image_mode='RGB',
                    height=600
                )
                prompt = gr.Text(
                    label="Prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your prompt",
                    container=False,
                )
                run_button = gr.Button("Run")

            result = gr.Image(label="Result", show_label=False)

        with gr.Accordion("Advanced Settings", open=True):

            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )

            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

            with gr.Row():

                width = gr.Slider(
                    label="Width",
                    minimum=512,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,
                    visible=False
                )

                height = gr.Slider(
                    label="Height",
                    minimum=512,
                    maximum=MAX_IMAGE_SIZE,
                    step=32,
                    value=1024,
                    visible=False
                )

            with gr.Row():

                guidance_scale = gr.Slider(
                    label="Guidance Scale",
                    minimum=1,
                    maximum=100,
                    step=0.5,
                    value=50,
                )

                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=28,
                )
                
            lora_scale = gr.Slider(
                label="LoRA Scale",
                minimum=0,
                maximum=1.0,
                step=0.01,
                value=1.0,
            )

        def process_example(edit_image, prompt, seed, randomize_seed):
            result, seed_out = infer(edit_image, prompt, seed, False, 1024, 1024, 50, 28, 1.0)
            return result, seed_out, False

        gr.Examples(
            examples=new_examples,
            inputs=[edit_image, prompt, seed, randomize_seed],
            outputs=[result, seed, randomize_seed],
            fn=process_example,
            cache_examples=False
        )

    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=[edit_image, prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps, lora_scale],
        outputs=[result, seed]
    )
    
if __name__ == "__main__": 
    demo.launch(
        server_name=args.server_name, 
        server_port=args.port,
        share=args.share, 
        inbrowser=True,
    )
