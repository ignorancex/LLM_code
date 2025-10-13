import dotenv

dotenv.load_dotenv(override=True)

import base64
from io import BytesIO

import datasets
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)


def pil_to_base64_data_uri(img: Image.Image, format="JPEG"):
    buffer = BytesIO()
    img.save(buffer, format=format)
    img_bytes = buffer.getvalue()
    base64_str = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/base64,{base64_str}"


data_source = "med-vlrm/PMC-VQA-EasyR1"

dataset = datasets.load_dataset(data_source)
dataset = dataset["train"]

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
sample = dataset[0]
image = sample["image"]

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": pil_to_base64_data_uri(image),
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
breakpoint()
