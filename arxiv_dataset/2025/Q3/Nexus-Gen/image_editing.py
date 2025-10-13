import argparse
import torch
from PIL import Image
from transformers import AutoConfig
from qwen_vl_utils import smart_resize
from modeling.decoder.editing_decoder import NexusGenEditingDecoder
from modeling.decoder.generation_decoder import NexusGenGenerationDecoder
from modeling.ar.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from modeling.ar.processing_qwen2_5_vl import Qwen2_5_VLProcessor


def get_image_embedding(vision_encoder, processor, image, target_size=(504, 504)):
    image = image.resize(target_size, Image.BILINEAR)
    inputs = processor.image_processor(images=[image], videos=None, return_tensors='pt', do_resize=False)
    pixel_values = inputs["pixel_values"].to(vision_encoder.device)
    image_grid_thw = inputs["image_grid_thw"].to(vision_encoder.device)
    pixel_values = pixel_values.type(vision_encoder.dtype)
    with torch.no_grad():
        image_embeds = vision_encoder(pixel_values, grid_thw=image_grid_thw)
    return image_embeds


def bound_image(image, max_pixels=262640):
    resized_height, resized_width = smart_resize(
        image.height,
        image.width,
        max_pixels=max_pixels,
    )
    return image.resize((resized_width, resized_height))


def parse_args():
    parser = argparse.ArgumentParser(description="Nexus-Gen Image Editing")

    parser.add_argument("--input_image", type=str, default='assets/examples/cat.png', help="Path to input image")
    parser.add_argument("--instruction", type=str, default='Add a pair of sunglasses.', help="Editing instruction")
    parser.add_argument("--use_generation_decoder", default=False, action='store_true', help="Where to use generation decoder for conceptual editing. Default: False to use editing decoder.")
    parser.add_argument("--height", type=int, default=512, help="Output image height")
    parser.add_argument("--width", type=int, default=512, help="Output image width")
    parser.add_argument("--max_pixels", type=int, default=262640, help="Maximum pixels for autoregressive model input")
    parser.add_argument("--ckpt_path", type=str, default='models/Nexus-GenV2', help="Path to model checkpoint")
    parser.add_argument("--flux_path", type=str, default='models', help="Path to flux models")
    parser.add_argument("--editing_decoder_path", type=str, default="models/Nexus-GenV2/edit_decoder.bin", help="Path to flux decoder")
    parser.add_argument("--generation_decoder_path", type=str, default="models/Nexus-GenV2/generation_decoder.bin", help="Path to flux decoder")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of diffusion steps")
    parser.add_argument("--negative_prompt", type=str, default='', help="Negative prompt for guidance")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="Classifier-free guidance scale")
    parser.add_argument("--embedded_guidance", type=float, default=3.5, help="Embedding guidance strength")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default='cuda:0', help="Device to use (e.g., 'cuda:0', 'cpu')")
    parser.add_argument("--enable_cpu_offload", default=True, action='store_true', help="Enable CPU offloading for memory optimization")
    parser.add_argument("--fp8_quantization", default=False, action='store_true', help="Enable FP8 quantization of the decoder for memory optimization")
    parser.add_argument("--result_path", type=str, default='edit_result.png', help="Path to save edited image")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    model_config = AutoConfig.from_pretrained(args.ckpt_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.ckpt_path,
        config=model_config,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map=args.device
    )
    processor = Qwen2_5_VLProcessor.from_pretrained(args.ckpt_path)
    model.eval()

    if '<image>' not in args.instruction:
        instruction = '<image> ' + args.instruction
    instruction = instruction.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
    messages = [{"role": "user", "content": [{"type": "text", "text": instruction}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_image = Image.open(args.input_image).convert('RGB')
    bounded_image = bound_image(input_image, max_pixels=args.max_pixels)
    inputs = processor(
        text=[text],
        images=[bounded_image],
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    generation_image_grid_thw = torch.tensor([[1, 18, 18]]).to(args.device) # refer to 81 image tokens

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            return_dict_in_generate=True,
            generation_image_grid_thw=generation_image_grid_thw
        )
    generated_ids = outputs['sequences']
    output_image_embeddings = outputs['output_image_embeddings']
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)]
    output_text = processor.batch_decode_all2all(generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    print(output_text)

    if args.use_generation_decoder:
        if args.enable_cpu_offload:
            model.cpu()
        flux_decoder = NexusGenGenerationDecoder(
            args.generation_decoder_path, 
            args.flux_path, 
            device=args.device, 
            enable_cpu_offload=args.enable_cpu_offload,
            fp8_quantization=args.fp8_quantization
        )

        image = flux_decoder.decode_image_embeds(
            output_image_embeddings,
            height=args.height,
            width=args.width,
            negative_prompt=args.negative_prompt,
            cfg_scale=args.cfg_scale,
            num_inference_steps=args.num_inference_steps,
            embedded_guidance=args.embedded_guidance,
            seed=args.seed
        )
    else:
        # (504, 504) refer to 324 image embeddings
        ref_embeddings = get_image_embedding(
            model.visual, 
            processor, 
            input_image, 
            target_size=(504, 504)
        )
        if args.enable_cpu_offload:
            model.cpu()

        flux_decoder = NexusGenEditingDecoder(
            args.editing_decoder_path, 
            args.flux_path, 
            args.ckpt_path, 
            device=args.device,
            enable_cpu_offload=args.enable_cpu_offload,
            fp8_quantization=args.fp8_quantization
        )
        image = flux_decoder.decode_image_embeds(
            output_image_embeddings, 
            ref_embed=ref_embeddings, 
            height=args.height,
            width=args.width, 
            negative_prompt=args.negative_prompt, 
            cfg_scale=args.cfg_scale, 
            num_inference_steps=args.num_inference_steps, 
            embedded_guidance=args.embedded_guidance, 
            seed=args.seed
        )

    image.save(args.result_path)
    print(f"Edited image saved to {args.result_path}")
