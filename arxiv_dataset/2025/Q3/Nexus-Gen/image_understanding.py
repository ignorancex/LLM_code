import argparse
from qwen_vl_utils import smart_resize
from transformers import AutoConfig
from qwen_vl_utils import process_vision_info
from modeling.ar.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from modeling.ar.processing_qwen2_5_vl import Qwen2_5_VLProcessor


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
    parser.add_argument("--instruction", type=str, default='Please give a brief description of the image.', help="Instruction for image understanding")
    parser.add_argument("--ckpt_path", type=str, default='models/Nexus-GenV2', help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default='cuda:0', help="Device to use (e.g., 'cuda:0', 'cpu')")
    parser.add_argument("--max_pixels", type=int, default=262640, help="Maximum pixels for autoregressive model input")

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

    messages = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": args.input_image,
            },
            {
                "type": "text",
                "text": args.instruction
            },
        ],
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    image_inputs = [bound_image(image) for image in image_inputs]
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(output_text)
