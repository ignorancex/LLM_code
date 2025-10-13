import argparse
import torch
from transformers import AutoConfig
from modeling.decoder.generation_decoder import NexusGenGenerationDecoder
from modeling.ar.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from modeling.ar.processing_qwen2_5_vl import Qwen2_5_VLProcessor

EN_TEMPLATE = "Generate an image according to the following description: {}"
ZH_TEMPLATE = "根据以下描述生成一张图像：{}"

def parse_args():
    parser = argparse.ArgumentParser(description="Nexus-Gen Image Generation")

    parser.add_argument("--prompt", type=str,
                        default="A middle-aged man with a graying beard and short hair stands on a quiet urban street, wearing a black jacket. He is looking off to the side with a thoughtful expression, his arms crossed. The background features blurred buildings with warm lights and a few indistinct figures walking in the distance. The scene has a calm, contemplative atmosphere.",
                        help="Text prompt for image generation")
    parser.add_argument("--language", type=str, choices=['en', 'zh'], default='en', help="Language for prompt template")
    parser.add_argument("--width", type=int, default=512, help="Output image width")
    parser.add_argument("--height", type=int, default=512, help="Output image height")
    parser.add_argument("--ckpt_path", type=str, default='models/Nexus-GenV2', help="Path to model checkpoint")
    parser.add_argument("--generation_decoder_path", type=str, default="models/Nexus-GenV2/generation_decoder.bin", help="Path to flux decoder")
    parser.add_argument("--flux_path", type=str, default='models', help="Path to flux models")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of diffusion steps")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt for classifier-free guidance")
    parser.add_argument("--cfg_scale", type=float, default=3.0, help="Classifier-free guidance scale")
    parser.add_argument("--embedded_guidance", type=float, default=3.5, help="Embedding guidance strength")
    parser.add_argument("--seed", type=int, default=52, help="Random seed")    
    parser.add_argument("--device", type=str, default='cuda:0', help="Device to use (e.g., 'cuda:0', 'cpu')")
    parser.add_argument("--enable_cpu_offload", default=True, action='store_true', help="Enable CPU offloading for memory optimization")
    parser.add_argument("--fp8_quantization", default=False, action='store_true', help="Enable FP8 quantization of the decoder for memory optimization")
    parser.add_argument("--result_path", type=str, default='gen_result.png', help="Path to save generated image")

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

    template = EN_TEMPLATE if args.language == 'en' else ZH_TEMPLATE
    formatted_prompt = template.format(args.prompt)
    print(formatted_prompt)

    messages = [{"role": "user", "content": [{"type": "text", "text": formatted_prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt")
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
    
    image.save(args.result_path)
    print(f"Image saved to {args.result_path}")
