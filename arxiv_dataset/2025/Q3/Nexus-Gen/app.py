import gradio as gr
import torch
from PIL import Image
import os
import random
from transformers import AutoConfig
from qwen_vl_utils import process_vision_info, smart_resize
from modeling.decoder.generation_decoder import NexusGenGenerationDecoder
from modeling.decoder.editing_decoder import NexusGenEditingDecoder
from modeling.ar.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from modeling.ar.processing_qwen2_5_vl import Qwen2_5_VLProcessor
import numpy as np

def bound_image(image, max_pixels=262640):
    resized_height, resized_width = smart_resize(
        image.height,
        image.width,
        max_pixels=max_pixels,
    )
    return image.resize((resized_width, resized_height))

# Initialize model and processor
model_path = 'models/Nexus-GenV2'
model_config = AutoConfig.from_pretrained(model_path)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    config=model_config,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map=device,
)
processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
model.eval()

# Initialize Flux Decoder
flux_path = "models"
generation_decoder_path = "models/Nexus-GenV2/generation_decoder.bin"
editing_decoder_path = "models/Nexus-GenV2/edit_decoder.bin"
generation_decoder = NexusGenGenerationDecoder(generation_decoder_path, flux_path, device=device, enable_cpu_offload=True)
editing_decoder = NexusGenEditingDecoder(editing_decoder_path, flux_path, model_path, device=device, enable_cpu_offload=True)


# Define system prompt
SYSTEM_PROMPT = "You are a helpful assistant."

def image_understanding(image, question):
    """Multimodal Q&A function - supports both visual and text Q&A"""
    if image is not None:
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": question if question else "Please give a brief description of the image."},
                ],
            }
        ]
    else:
        # Text-only Q&A mode
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                ],
            }
        ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    if image is not None:
        image_inputs, _ = process_vision_info(messages)
        image_inputs = [bound_image(image) for image in image_inputs]
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
    else:
        inputs = processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )

    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

def image_generation(prompt):
    """Image generation function"""
    generation_instruction = 'Generate an image according to the following description: {}'
    prompt = generation_instruction.format(prompt)

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    generation_image_grid_thw = torch.tensor([[1, 18, 18]]).to(model.device) # refer to 81 image tokens

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1024, return_dict_in_generate=True, generation_image_grid_thw=generation_image_grid_thw,)

        if not hasattr(outputs, 'output_image_embeddings'):
            raise ValueError("Failed to generate image embeddings")
        else:
            output_image_embeddings = outputs.output_image_embeddings
    seed = random.randint(0, 10000)
    image = generation_decoder.decode_image_embeds(output_image_embeddings, cfg_scale=3.0, seed=seed)
    return image


def get_image_embedding(vision_encoder, processor, image, target_size=(504, 504)):
    image = image.resize(target_size, Image.BILINEAR)
    inputs = processor.image_processor(images=[image], videos=None, return_tensors='pt', do_resize=False)
    pixel_values = inputs["pixel_values"].to(vision_encoder.device)
    image_grid_thw = inputs["image_grid_thw"].to(vision_encoder.device)
    pixel_values = pixel_values.type(vision_encoder.dtype)
    with torch.no_grad():
        image_embeds = vision_encoder(pixel_values, grid_thw=image_grid_thw)
    return image_embeds


def image_editing(image, instruction):
    """Image editing function"""
    if '<image>' not in instruction:
        instruction = '<image> ' + instruction
    instruction = instruction.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
    messages = [{"role": "user", "content": [{"type": "text", "text": instruction}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Convert numpy array to PIL Image if needed
    input_image = Image.fromarray(image) if not isinstance(image, Image.Image) else image
    bounded_image = bound_image(input_image)

    inputs = processor(
        text=[text],
        images=[bounded_image],
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    generation_image_grid_thw = torch.tensor([[1, 18, 18]]).to(model.device) # refer to 81 image tokens

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1024, return_dict_in_generate=True, generation_image_grid_thw=generation_image_grid_thw)
        if not hasattr(outputs, 'output_image_embeddings'):
            raise ValueError("Failed to generate image embeddings")
        else:
            output_image_embeddings = outputs.output_image_embeddings
    ref_embeddings = get_image_embedding(model.visual, processor, input_image, target_size=(504, 504))
    edited_image = editing_decoder.decode_image_embeds(output_image_embeddings, ref_embed=ref_embeddings, cfg_scale=1.0, )
    return edited_image

# Create Gradio interface
with gr.Blocks(title="Nexus-Gen Demo") as demo:
    gr.Markdown("# Nexus-Gen Demo")
    with gr.Tab("Multimodal Q&A"):
        with gr.Row():
            with gr.Column():
                understanding_input = gr.Image(label="Upload Image (Optional)")
                understanding_question = gr.Textbox(
                    label="Input Question",
                    lines=2,
                    placeholder="You can:\n1. Upload an image and ask questions about it\n2. Ask text-only questions\n3. Upload an image without a question for automatic description"
                )
                understanding_button = gr.Button("Generate Response")
            with gr.Column():
                understanding_output = gr.Markdown(label="Response")
        understanding_button.click(
            fn=image_understanding,
            inputs=[understanding_input, understanding_question],
            outputs=[understanding_output]
        )
        gr.Examples(
            examples=[
                # Visual Q&A examples
                ["assets/examples/cat.png", "What color is the cat?"],
                # Text Q&A examples
                [None, "What are the main differences between electric and traditional fuel vehicles?"],
                # Image description example
                ["assets/examples/cat.png", ""],
            ],
            inputs=[understanding_input, understanding_question],
            outputs=understanding_output,
            fn=image_understanding,
            cache_examples=False,
        )

    with gr.Tab("Image Generation"):
        with gr.Row():
            with gr.Column():
                generation_input = gr.Textbox(label="Input Prompt", lines=3, placeholder="Describe the image you want to generate")
                generation_button = gr.Button("Generate Image")
            with gr.Column():
                generation_output = gr.Image(label="Generated Result")

        def generate_with_option(prompt):
            image = image_generation(prompt)
            return image

        generation_button.click(
            fn=generate_with_option,
            inputs=[generation_input],
            outputs=[generation_output]
        )

        gr.Examples(
            examples=[
                "A cut dog sitting on a bench in a park, wearing a red collar.",
                "A woman in a blue dress standing on a beach at sunset.",
                "一只可爱的猫。"
            ],
            inputs=[generation_input],
            outputs=[generation_output],
            fn=generate_with_option,
            cache_examples=False,
        )

    with gr.Tab("Image Editing"):
        with gr.Row():
            with gr.Column():
                editing_image = gr.Image(label="Upload Image to Edit")
                editing_instruction = gr.Textbox(label="Edit Instruction", lines=2)
                editing_button = gr.Button("Start Editing")
            with gr.Column():
                editing_output = gr.Image(label="Edited Result")
        editing_button.click(
            fn=image_editing,
            inputs=[editing_image, editing_instruction],
            outputs=[editing_output]
        )
        gr.Examples(
            examples=[
                ["assets/examples/cat.png", "Add a pair of sunglasses for the cat."],
                ["assets/examples/cat.png", "给猫加一副太阳镜。"],
            ],
            inputs=[editing_image, editing_instruction],
            outputs=editing_output,
            fn=image_editing,
            cache_examples=False,
        )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861
    )
